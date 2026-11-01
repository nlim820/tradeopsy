"""逐日价格序列 —— 滚动口径 PGR/PLR 的唯一外部依赖。

为什么需要单独一个模块：
    期末快照口径（`--prices`，代码 + 最新价）只看得见"现在还没卖的"头寸。
    Odean (1998) 原文是**逐卖出日滚动**口径：每个卖出日都要问一次
    「当时还有哪些浮盈 / 浮亏的头寸本可以卖、却没卖」。
    回答这个问题，必须知道**每个历史日期**的价格 —— 一个当前价不够。

数据格式（长格式，三列）：
    日期,证券代码,收盘价
    2025-11-05,000001,10.20

    为什么只收长格式、不收"日期为行、代码为列"的宽表：
    宽表在标的增删时要改列、停牌要留空、导出时行数随标的数据量爆炸；
    长格式是行情接口（akshare / tushare）的天然形态，一行一个观测，
    缺失就是缺行 —— 语义干净。看到宽表会明确拒绝并给出转法，
    而不是猜一个可能错位的列映射。

设计上一个刻意的选择：**不补全缺失日期**。
    把停牌日填成上一日收盘价，看起来很体贴，实际是伪造数据 ——
    伪造出来的价格会静默进入 PGR 分母，而分母错了整份报告就废了。
    所以 `on()` 在无覆盖时返回 None，由上层决定是跳过、还是带上
    「这价格是多少个交易日之前的」这层信息谨慎使用。
"""

from __future__ import annotations

import csv
from bisect import bisect_right, insort
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

__all__ = ["PriceHistory", "read_price_history", "looks_like_wide"]


#: 表头别名。真实导出的列名五花八门，这里覆盖常见几种；
#: 匹配时先精确、再去空格大小写，避免把「收盘价(元)」这类漏掉。
DATE_KEYS = ("日期", "交易日期", "交易日", "date", "Date", "dt", "时间")
SYMBOL_KEYS = ("证券代码", "代码", "股票代码", "基金代码", "symbol", "Symbol", "code", "Code")
PRICE_KEYS = ("收盘价", "收盘", "close", "Close", "价格", "最新价", "收盘价(元)", "price", "Price")


def _norm(s: str) -> str:
    return s.strip().strip('"').strip("'").replace(" ", "").replace("\ufeff", "")


def _match(header: list[str], keys: tuple[str, ...]) -> int:
    """在表头里找第一个命中 keys 的列号；命中不了返回 -1。

    两轮：先精确匹配，再前缀匹配（让「收盘价(元)」「收盘价_前复权」也能命中）。
    前缀匹配放在第二轮，是为了不让「开盘价」被「价格」误抓。
    """
    norm = [_norm(h) for h in header]
    for i, h in enumerate(norm):
        if h in keys:
            return i
    for i, h in enumerate(norm):
        if any(h.startswith(k) for k in keys if len(k) >= 2 and not k.isascii()):
            return i
    return -1


def _to_decimal(raw: str) -> Decimal | None:
    tok = _norm(raw).replace(",", "")
    if not tok or tok in ("-", "--", "None", "nan", "NaN"):
        return None
    try:
        return Decimal(tok)
    except (InvalidOperation, ValueError):
        return None


def _to_date(raw: str) -> date | None:
    tok = _norm(raw)
    if not tok:
        return None
    # 兼容 20251105 / 2025-11-05 / 2025/11/05 / 2025-11-05 15:00:00
    tok = tok.replace("/", "-").split(" ")[0]
    if len(tok) == 8 and tok.isdigit():
        tok = f"{tok[:4]}-{tok[4:6]}-{tok[6:]}"
    try:
        return date.fromisoformat(tok[:10])
    except ValueError:
        return None


def looks_like_wide(header: list[str]) -> bool:
    """判断表头是不是「日期, 600519, 000001, ...」这种宽表。

    判据：除首列外，出现过半数的列名是纯数字代码（4~6 位）。
    阈值取半数而不是"任意一列"，是因为长格式里也可能有一列叫「1」的
    奇怪表头，单列命中不足以定性。
    """
    if len(header) < 3:
        return False
    tail = [_norm(h) for h in header[1:]]
    hit = sum(1 for h in tail if h.isdigit() and 4 <= len(h) <= 6)
    return hit * 2 >= len(tail) and hit > 0


class PriceHistory:
    """多标的的日线收盘价，按标的分别维护升序日期数组。

    查询复杂度 O(log n)：bisect 找 <= dt 的最近一个观测。
    不用 dict 逐日补全的理由见模块 docstring。
    """

    def __init__(
        self,
        series: dict[str, dict[date, Decimal]] | None = None,
        *,
        source: str = "explicit",
    ) -> None:
        self._series: dict[str, dict[date, Decimal]] = {
            k: dict(v) for k, v in (series or {}).items() if v
        }
        self._dates: dict[str, list[date]] = {
            k: sorted(v) for k, v in self._series.items()
        }
        self.source = source
        #: 同一 (代码, 日期) 重复出现的次数（后值覆盖前值，这里只做披露）
        self.dups = 0
        #: 无法解析被丢弃的行数
        self.bad_rows = 0
        #: 致命问题描述（空串表示没问题）。用字段而不是抛异常，
        #: 是为了让 CLI 能用一句人话告诉用户"哪一列没认出来"，而不是甩堆栈。
        self.problem = ""

    # ---------- 基本属性 ----------

    def __bool__(self) -> bool:
        return bool(self._series)

    def __len__(self) -> int:
        return len(self._series)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._series

    @property
    def symbols(self) -> set[str]:
        return set(self._series)

    @property
    def n_points(self) -> int:
        return sum(len(v) for v in self._series.values())

    @property
    def span(self) -> tuple[date, date] | None:
        if not self._series:
            return None
        return (
            min(d for v in self._series.values() for d in v),
            max(d for v in self._series.values() for d in v),
        )

    def describe(self) -> str:
        sp = self.span
        label = {"csv": "CSV 长格式", "explicit": "内存构造"}.get(self.source, self.source)
        if sp is None:
            return f"空价格序列（{label}）"
        return (
            f"{sp[0]} → {sp[1]} · {len(self)} 只标的 · {self.n_points} 个观测 · 来源：{label}"
        )

    # ---------- 查询 ----------

    def on(self, symbol: str, dt: date) -> tuple[Decimal | None, date | None]:
        """取 `dt` 当日（或此前最近一个交易日）的收盘价。

        返回 (价格, 该价格所属日期)：
            有当日价格        -> (px, dt)
            只有更早的价格    -> (px, 更早的日期)   ← 调用方据此判断"陈旧度"
            完全没有覆盖      -> (None, None)

        **不做**向前填充，也不设隐含上限：陈旧度交回调用方，
        因为"能容忍多旧"取决于业务（滚动口径里 3 天可以，3 个月不行）。
        """
        ds = self._dates.get(symbol)
        if not ds:
            return None, None
        i = bisect_right(ds, dt)
        if i == 0:
            return None, None
        d = ds[i - 1]
        return self._series[symbol][d], d

    def last_price(self, symbol: str) -> Decimal | None:
        ds = self._dates.get(symbol)
        if not ds:
            return None
        return self._series[symbol][ds[-1]]

    def snapshot(self) -> dict[str, Decimal]:
        """每个标的的最后一个收盘价 —— 用来把序列降级成期末快照口径。

        这样用户只提供一个 `--price-history` 就能同时得到两套口径，
        不必再手工维护一个「最新价」文件。两份文件一旦不同步，
        两套口径就会建立在不同的价格基础上，比较也就失去意义。
        """
        out: dict[str, Decimal] = {}
        for sym in self._series:
            px = self.last_price(sym)
            if px is not None:
                out[sym] = px
        return out

    # ---------- 构造 ----------

    def add(self, symbol: str, dt: date, px: Decimal) -> None:
        """单点加入（测试与增量场景用）。

        日期数组保持有序：绝大多数行情文件本身按日期升序，
        走 append 分支是 O(1)；只有乱序行才付出一次 insort 的代价。
        早期版本每行都 `sort()`，几十万行时会退化成 O(n² log n)。
        """
        sym = str(symbol).strip()
        bucket = self._series.setdefault(sym, {})
        existed = dt in bucket
        if existed and bucket[dt] != px:
            self.dups += 1
        bucket[dt] = px
        if existed:
            return
        ds = self._dates.setdefault(sym, [])
        if not ds or dt > ds[-1]:
            ds.append(dt)
        else:
            insort(ds, dt)

    @classmethod
    def from_rows(cls, rows: list[tuple[date, str, Decimal]], *, source: str = "explicit") -> "PriceHistory":
        ph = cls(source=source)
        for d, sym, px in rows:
            ph.add(sym, d, px)
        return ph


def read_price_history(path: str) -> PriceHistory:
    """读长格式价格序列 CSV：`日期,证券代码,收盘价`（表头可省）。

    编码先按 utf-8-sig，失败回退 gb18030 —— 券商导出十有八九是 GBK 系。
    列序不固定：有表头时按列名定位，无表头时按 1/2/3 位置。
    """
    p = Path(path)
    ph = PriceHistory(source="csv")
    if not p.exists():
        ph.problem = f"价格序列文件不存在：{path}"
        return ph

    text: str | None = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            text = p.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        ph.problem = f"无法解码价格文件（试过 utf-8 与 gb18030）：{path}"
        return ph

    # 逐行解析而不用 csv.reader 整表读入：行情文件动辄几十万行，
    # 一次性 list() 会吃掉不必要的内存。
    reader = csv.reader(text.splitlines())
    try:
        first = next(reader)
    except StopIteration:
        ph.problem = "价格文件是空的"
        return ph

    header = list(first)
    has_header = _to_date(header[0]) is None

    if has_header:
        if looks_like_wide(header):
            ph.problem = (
                "看起来是宽表（日期为行、代码为列）。请转成三列长格式："
                "`日期,代码,收盘价`。宽表不支持的 reason：停牌要留空、"
                "标的增删要改列，列映射一旦错位会静默算错，宁可不猜。"
            )
            return ph
        i_d = _match(header, DATE_KEYS)
        i_s = _match(header, SYMBOL_KEYS)
        i_p = _match(header, PRICE_KEYS)
        missing = [
            name
            for name, idx in (("日期", i_d), ("证券代码", i_s), ("收盘价", i_p))
            if idx < 0
        ]
        if missing:
            ph.problem = (
                f"表头里找不到这些列：{'、'.join(missing)}。"
                f"实际表头是 {header}。请改用三列长格式：`日期,代码,收盘价`。"
            )
            return ph
        body = reader
    else:
        i_d, i_s, i_p = 0, 1, 2
        body = iter([header] + list(reader))

    width = max(i_d, i_s, i_p) + 1
    for row in body:
        if not row or all(not c.strip() for c in row):
            continue
        if len(row) < width:
            ph.bad_rows += 1
            continue
        d = _to_date(row[i_d])
        px = _to_decimal(row[i_p])
        sym = _norm(row[i_s])
        if d is None or px is None or not sym:
            ph.bad_rows += 1
            continue
        if px <= 0:
            # 0 或负数价格不是真实行情，多半是占位符；放进分母会污染统计
            ph.bad_rows += 1
            continue
        ph.add(sym, d, px)

    if not ph:
        ph.problem = ph.problem or "价格文件读完了，但没有一行有效数据"
    return ph
