"""通用 CSV 交割单解析器。

它要对付的真实情况（每一条都是实盘文件里真的会遇到的）：

1. **编码是 GBK**。券商系统多为 Windows 遗留，导出默认 GBK。
   直接按 UTF-8 读会 UnicodeDecodeError；用 errors="ignore" 会把证券名称吃成空串。
2. **表头不在第一行**。前面常有"资金账号:xxx""查询区间:xxx"之类的抬头，
   有时还有空行。硬编码 `header=0` 一定崩。
3. **列名各家不同**。同花顺叫"成交日期/业务名称"，东财叫"发生日期/摘要"，
   华泰叫"成交日期/操作"。只能模糊匹配，不能枚举。
4. **数量带符号**。卖出常记为负数，取绝对值之后还要保留方向信息。
5. **费用拆成多列**。有的券商一列"手续费"包干，有的拆成净佣金/规费/过户费/印花税。
   加错就会重复计费。

设计上的取舍：**宁可告警，不可静默丢弃，也不可猜测。**
认不出的行业务名统一进 warnings，让人自己看；
猜错一行脏数据造成的账目错误，比少算一行更难排查。
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..model import Event, EventType, classify_business
from .base import ParseResult, ParseWarning, Parser

#: 编码尝试顺序。utf-8-sig 必须在 utf-8 之前 ——
#: 否则 BOM 会被当成一个不可见字符粘在第一列列名上，导致"成交日期"匹配不上。
#: gb18030 是 GBK 的超集，放最后兜底。
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

#: 列名候选。顺序即优先级：越具体越靠前。
#: 注意 "成交日期" 必须排在 "日期" 之前，否则"发生日期"会被 "日期" 抢先命中。
DATE_KEYS = ("成交日期", "发生日期", "交易日期", "清算日期", "委托日期", "成交时间", "日期")
CODE_KEYS = ("证券代码", "股票代码", "证券编码", "合约代码", "代码")
NAME_KEYS = ("证券名称", "股票名称", "证券简称", "名称")
BIZ_KEYS = ("业务名称", "业务类型", "交易类别", "业务", "摘要", "操作", "委托类别")
FLAG_KEYS = ("买卖标志", "买卖方向", "买卖", "操作方向", "委托方向")
QTY_KEYS = ("成交数量", "发生数量", "成交股数", "成交份额", "成交笔数", "数量")
PRICE_KEYS = ("成交价格", "成交均价", "成交价", "委托价格", "价格")
AMOUNT_KEYS = ("成交金额", "发生金额", "成交额", "金额")

#: 包干型费用列。存在时优先用它，不再累加分项。
FEE_SINGLE_KEYS = ("手续费", "费用合计", "总费用", "交易费用")
#: 分项费用列。仅在找不到包干列时才累加。
FEE_PART_KEYS = ("净佣金", "佣金", "印花税", "过户费", "规费", "经手费", "证管费")

#: 汇总行，静默跳过。它们通常长这样："合计,,,12345.67,"
TOTAL_ROW_MARKS = ("合计", "小计", "总计", "汇总")

_SIX_DIGIT = re.compile(r"\d{6}")


def _norm(s: str) -> str:
    """归一化列名：全角转半角、去空白。

    这一步不做，全角空格（券商导出里非常常见）会让 "成交 数量" 匹配不上 "成交数量"。
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", "", s)


def read_text(path: Path) -> tuple[str, str]:
    """按可能的编码逐个尝试解码，返回 (文本, 实际编码)。"""
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # 全部失败时用替换模式，至少让用户看到能看的部分并告警
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def detect_delimiter(sample: str) -> str:
    """探测分隔符。

    Sniffer 在"整份文件只有几十行且列数不一致"时会抛异常，
    所以兜底逻辑要按字符出现频次选，不能直接假定逗号 ——
    有些券商导出用 Tab。
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in ",\t;|"}
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else ","


def to_decimal(raw: str) -> Decimal:
    """把 "1,234.56" / "¥1,234.56" / "(500.00)" / "--" 解析成 Decimal。"""
    s = _norm(raw).replace(",", "").replace("¥", "").replace("￥", "").replace('"', "")
    if s in ("", "-", "--", "—", "N/A", "nan"):
        return Decimal("0")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return Decimal("0")


def to_date(raw: str):
    """解析日期。支持 2026-10-30 / 2026/10/30 / 20261030 / 2026.10.30，
    以及带时间的 "2026-10-30 09:31:25"。"""
    s = _norm(raw)
    if not s:
        return None
    s = s.split("T")[0].split(" ")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})[-/.]?(\d{2})[-/.]?(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except ValueError:
            return None
    return None


def normalize_symbol(raw: str) -> str:
    """提取 6 位证券代码。带 sh/sz 前缀或 .SH 后缀都能处理。"""
    m = _SIX_DIGIT.search(_norm(raw))
    return m.group(0) if m else ""


def _find_col(headers: list[str], keys: tuple[str, ...]) -> int | None:
    """按"完全相等优先，其次包含"的规则找列索引。

    为什么不能只做包含匹配：一份文件里同时有"成交日期"和"发生日期"时，
    只做包含匹配会命中先出现的那个，而正确做法是按 keys 的优先级来。
    """
    norms = [_norm(h) for h in headers]
    for k in keys:
        for i, h in enumerate(norms):
            if h == k:
                return i
    for k in keys:
        for i, h in enumerate(norms):
            if h and k in h:
                return i
    return None


def _find_all_cols(headers: list[str], keys: tuple[str, ...]) -> list[int]:
    norms = [_norm(h) for h in headers]
    out: list[int] = []
    for i, h in enumerate(norms):
        if h and any(k in h for k in keys):
            out.append(i)
    return out


def _cell(row: list[str], idx: int | None) -> str:
    if idx is None or idx >= len(row):
        return ""
    return (row[idx] or "").strip()


class GenericCsvParser(Parser):
    name = "csv-generic"
    description = "通用 CSV 交割单（编码/列名/分隔符自动适配）"

    def sniff(self, path: Path, head: str) -> float:
        """命中 3 个以上已知列名即认为是交割单。"""
        if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
            return 0.0
        all_keys = (
            DATE_KEYS + CODE_KEYS + NAME_KEYS + BIZ_KEYS
            + QTY_KEYS + PRICE_KEYS + AMOUNT_KEYS
        )
        hits = sum(1 for k in all_keys if k in _norm(head))
        if hits >= 4:
            return 0.9
        if hits >= 2:
            return 0.5
        return 0.0

    def parse(self, path: Path) -> ParseResult:
        text, encoding = read_text(path)
        delim = detect_delimiter(text[:8000])
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        result = ParseResult()
        result.meta.update({
            "parser": self.name,
            "encoding": encoding,
            "delimiter": "\t" if delim == "\t" else delim,
            "total_rows": len(rows),
        })

        header_idx = self._locate_header(rows)
        if header_idx is None:
            result.warnings.append(
                ParseWarning(0, "找不到表头行（前 15 行里没有识别出足够的已知列名）")
            )
            return result

        headers = rows[header_idx]
        cols = {
            "date": _find_col(headers, DATE_KEYS),
            "code": _find_col(headers, CODE_KEYS),
            "name": _find_col(headers, NAME_KEYS),
            "biz": _find_col(headers, BIZ_KEYS),
            "flag": _find_col(headers, FLAG_KEYS),
            "qty": _find_col(headers, QTY_KEYS),
            "price": _find_col(headers, PRICE_KEYS),
            "amount": _find_col(headers, AMOUNT_KEYS),
        }
        fee_single = _find_col(headers, FEE_SINGLE_KEYS)
        fee_parts = [] if fee_single is not None else _find_all_cols(headers, FEE_PART_KEYS)
        cols["fee"] = fee_single
        cols["fee_parts"] = fee_parts  # type: ignore[assignment]

        result.meta["header_row"] = header_idx + 1
        result.meta["columns"] = {
            k: (headers[v] if isinstance(v, int) and v < len(headers) else None)
            for k, v in cols.items() if k != "fee_parts"
        }
        result.meta["fee_split_columns"] = [
            headers[i] for i in fee_parts if i < len(headers)
        ]

        missing = [k for k in ("date", "code") if cols[k] is None]
        if missing:
            result.warnings.append(
                ParseWarning(
                    header_idx + 1,
                    f"缺少必需列: {', '.join(missing)}。"
                    f"实际表头: {headers[:12]}",
                )
            )
            return result

        for offset, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
            if not any((c or "").strip() for c in row):
                continue
            joined = "".join(row)
            if any(m in joined for m in TOTAL_ROW_MARKS):
                continue
            ev, warn = self._parse_row(row, cols, offset)
            if ev is not None:
                result.events.append(ev)
            if warn is not None:
                result.warnings.append(warn)

        result.events.sort(key=lambda e: (e.dt, e.symbol))
        return result

    @staticmethod
    def _locate_header(rows: list[list[str]]) -> int | None:
        """在前 15 行里找"像表头"的那行。

        判据不是"第一个非空行"，而是"命中已知列名数量最多的行"。
        这样即使用户的文件前面有 5 行账户信息、后面又有一行分隔线，也能找对。
        """
        all_keys = (
            DATE_KEYS + CODE_KEYS + NAME_KEYS + BIZ_KEYS
            + QTY_KEYS + PRICE_KEYS + AMOUNT_KEYS + FEE_SINGLE_KEYS
        )
        best_idx, best_hits = None, 0
        for i, row in enumerate(rows[:15]):
            hits = sum(1 for c in row if any(k in _norm(c) for k in all_keys))
            if hits > best_hits:
                best_idx, best_hits = i, hits
        return best_idx if best_hits >= 3 else None

    def _parse_row(
        self, row: list[str], cols: dict, row_no: int
    ) -> tuple[Event | None, ParseWarning | None]:
        raw_date = _cell(row, cols["date"])
        raw_code = _cell(row, cols["code"])
        raw_biz = _cell(row, cols["biz"])
        raw_flag = _cell(row, cols["flag"])

        dt = to_date(raw_date)
        if dt is None:
            return None, ParseWarning(row_no, f"日期无法解析: {raw_date!r}", ",".join(row))

        symbol = normalize_symbol(raw_code)
        if not symbol:
            return None, ParseWarning(row_no, f"证券代码无法识别: {raw_code!r}", ",".join(row))

        qty = to_decimal(_cell(row, cols["qty"]))
        price = to_decimal(_cell(row, cols["price"]))
        amount = to_decimal(_cell(row, cols["amount"]))

        # 费用：包干列优先；没有则累加分项。
        # 同时存在两种时只取包干列，否则会重复计费（这是最容易出错的一步）。
        if cols["fee"] is not None:
            fee = to_decimal(_cell(row, cols["fee"]))
        else:
            fee = sum(
                (to_decimal(_cell(row, i)) for i in cols["fee_parts"]),
                Decimal("0"),
            )

        # ---- 方向判定，按可靠性排序 ----
        etype = classify_business(raw_biz)
        if etype is EventType.OTHER and raw_flag:
            etype = classify_business(raw_flag)
        if etype is EventType.OTHER and qty != 0:
            # 兜底：数量符号。券商普遍用负数表示卖出。
            etype = EventType.SELL if qty < 0 else EventType.BUY

        qty_abs = abs(qty)

        # 价格缺失但金额和数量都有 → 反推价格（部分券商只给金额）
        if price == 0 and qty_abs > 0 and amount != 0:
            price = (abs(amount) / qty_abs).quantize(Decimal("0.0001"))

        if etype is EventType.OTHER:
            return None, ParseWarning(
                row_no, f"未识别的业务类型: {raw_biz or raw_flag or '(空)'!r}", ",".join(row)
            )

        if etype in (EventType.BUY, EventType.SELL):
            if qty_abs == 0:
                return None, ParseWarning(row_no, "买卖记录数量为 0", ",".join(row))
            # 买卖类事件的 fee 语义是"该笔交易的费用"（买计入成本，卖从收入扣）
            return (
                Event(dt=dt, symbol=symbol, etype=etype,
                      qty=qty_abs, price=price, fee=fee),
                None,
            )

        if etype is EventType.DIVIDEND:
            cash = abs(amount) if amount != 0 else Decimal("0")
            if cash == 0 and fee == 0:
                return None, ParseWarning(row_no, "分红记录金额为 0", ",".join(row))
            return (
                Event(dt=dt, symbol=symbol, etype=etype,
                      qty=Decimal("0"), price=Decimal("0"), fee=fee, cash=cash),
                None,
            )

        if etype is EventType.SPLIT:
            if qty_abs == 0:
                return None, ParseWarning(row_no, "送转记录数量为 0", ",".join(row))
            return (
                Event(dt=dt, symbol=symbol, etype=etype, qty=qty_abs),
                None,
            )

        if etype in (EventType.RIGHTS, EventType.IPO_ALLOT):
            paid = abs(amount) if amount != 0 else qty_abs * price
            return (
                Event(dt=dt, symbol=symbol, etype=etype,
                      qty=qty_abs, price=price, cash=paid),
                None,
            )

        return None, ParseWarning(row_no, f"未处理的事件类型: {etype}", ",".join(row))
