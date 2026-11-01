"""命令行入口：`tradeopsy analyze [路径]`

默认读 `private/` —— 那是唯一被 .gitignore 硬隔离的目录，
把交割单丢进去就不用担心误提交。

为什么不支持管道输入：交割单是多列结构化数据，
stdin 传进来的几乎一定是用户手工复制粘贴的残缺版本，
解析失败时无法回溯原文件，排障成本远高于收益。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from .analytics.benchmark import BenchmarkReport, Comparison
from .analytics.disposition import (
    DispositionReport,
    compute_disposition,
    open_positions_from_ledger,
)
from .analytics.rolling import RollingReport, rolling_disposition
from .analytics.significance import SignificanceResult, bootstrap_disposition
from .ledger import PositionLedger, summarize, timeline
from .load import collect_inputs as _collect_inputs
from .load import dedupe as _dedupe
from .load import load_dataset, read_benchmark, read_calendar, read_prices
from .model import EventType
from .parsers import parse_file
from .prices import PriceHistory, read_price_history
from .tradingcal import TradingCalendar

TYPE_CN = {
    EventType.BUY: "买入",
    EventType.SELL: "卖出",
    EventType.DIVIDEND: "分红",
    EventType.SPLIT: "送转",
    EventType.RIGHTS: "配股",
    EventType.IPO_ALLOT: "打新",
    EventType.OTHER: "其他",
}

DATA_SUFFIXES = (".csv", ".txt", ".tsv")


def _w(s: object) -> int:
    """计算字符串的终端显示宽度。中文占两列。

    不做这个，表格会歪 —— 因为 "证券名称" 和 "abc" 的 len() 都是 4，
    但前者在终端里占 8 列。这种歪斜会让人觉得工具很糙。
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s: object, width: int, align: str = "left") -> str:
    gap = max(0, width - _w(s))
    return (" " * gap + str(s)) if align == "right" else (str(s) + " " * gap)


def _by_symbol(chunks: list) -> list[tuple[str, int, Decimal, int]]:
    """按标的聚合已实现盈亏，用于回答"我到底在哪只票上亏了钱"。"""
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": Decimal("0"), "win": 0})
    for c in chunks:
        d = agg[c.symbol]
        d["n"] += 1
        d["pnl"] += c.pnl
        if c.is_win:
            d["win"] += 1
    rows = [(s, d["n"], d["pnl"], d["win"]) for s, d in agg.items()]
    rows.sort(key=lambda r: r[2])
    return rows


def _render_disposition(rep: DispositionReport) -> list[str]:
    """处置效应板块。纯文本、离线可用。"""
    lines: list[str] = []
    lines.append("\n【处置效应 · 行为归因】")
    if rep.has_full:
        lines.append(f"  PGR 盈利实现比例  {rep.pgr:.2%}")
        lines.append(f"  PLR 亏损实现比例  {rep.plr:.2%}")
        lines.append(f"  处置效应 DE       {rep.disposition_effect:+.2%}")
        lines.append(f"  → {rep.verdict}")
    else:
        lines.append("  PGR/PLR 不可用（未提供未平仓市值）：")
        lines.append(f"  已实现盈利 {rep.realized_gain:,.2f} · 已实现亏损 {rep.realized_loss:,.2f}")
        lines.append(f"  → {rep.verdict}")
        lines.append("  提示：分析时加 --prices 价文件 可算完整 PGR/PLR。")
    lines.append(
        f"  胜率 {rep.win_rate:.2%} · 赢家均持 {rep.avg_hold_win} "
        f"{'交易日' if rep.hold_days_trading else '自然日'}"
        f" · 输家均持 {rep.avg_hold_loss} "
        f"{'交易日' if rep.hold_days_trading else '自然日'}"
    )
    if rep.hold_days_trading and rep.avg_hold_win_calendar:
        # 把两种口径同时摆出来：口径差多少是可见的，读者不必信任一个孤零零的数
        lines.append(
            f"  （同一口径对照：按自然日 赢家 {rep.avg_hold_win_calendar} 天"
            f" / 输家 {rep.avg_hold_loss_calendar} 天）"
        )
    for n in rep.notes:
        lines.append(f"   · {n}")
    return lines


def _render_significance(sig: SignificanceResult) -> str:
    """显著性检验板块（计数口径）。"""
    lines = ["", "【显著性检验 · 计数口径 Odean】"]
    lines.append(
        f"  机会：盈利 卖出 {sig.n_gain_sold}/持有 {sig.n_gain_held}"
        f" · 亏损 割肉 {sig.n_loss_sold}/持有 {sig.n_loss_held}"
    )
    if sig.pgr_count is not None and sig.plr_count is not None:
        lines.append(
            f"  计数 PGR {sig.pgr_count:.2%} · PLR {sig.plr_count:.2%}"
            f" · DE {sig.de_count:+.2%}"
        )
    if sig.ci_low is not None and sig.ci_high is not None:
        z = f" · z p={sig.p_value_z:.4f}" if sig.p_value_z is not None else ""
        lines.append(
            f"  95% CI [{sig.ci_low:+.2%}, {sig.ci_high:+.2%}]"
            f" · bootstrap p={sig.p_value_boot:.4f}{z}"
        )
    lines.append(f"  → {sig.verdict}")
    for n in sig.notes:
        lines.append(f"   · {n}")
    return "\n".join(lines)


def _render_rolling(roll: RollingReport, rep: DispositionReport | None = None) -> str:
    """逐日滚动口径板块：Odean (1998) 原文口径，与期末快照口径并列。

    两个口径摆在一起不是凑数 —— 它们的差额就是"口径敏感度"：
    差额大说明期末那一天的账面恰好不代表历史，只看期末会看错人。
    """
    lines = ["", "【处置效应 · 逐日滚动口径（Odean 1998 原文）】"]

    if roll.mode != "rolling" or not roll.has_result:
        lines.append("  不可计算 —— 这不是 bug，是数据不足。")
        for n in roll.notes:
            lines.append(f"   · {n}")
        return "\n".join(lines)

    def p(x: float | None) -> str:
        return "—" if x is None else f"{x:.2%}"

    lines.append(
        f"  有效观测 {roll.n_obs} 个卖出日"
        f"（每个卖出日都按当日收盘价重估一次账面持仓）"
    )
    lines.append(
        f"  盈利机会 {roll.n_gain_opp} 次 · 兑现 {roll.n_gain_sold} 次"
        f"  → 计数 PGR {p(roll.pgr)}"
    )
    lines.append(
        f"  亏损机会 {roll.n_loss_opp} 次 · 兑现 {roll.n_loss_sold} 次"
        f"  → 计数 PLR {p(roll.plr)}"
    )
    if roll.de is not None:
        lines.append(f"  处置效应 DE       {roll.de:+.2%}")
    if roll.de_amount is not None:
        lines.append(f"  金额口径 DE       {roll.de_amount:+.2%}"
                     f"（PGR {p(roll.pgr_amount)} / PLR {p(roll.plr_amount)}）")
    if roll.daily_de_mean is not None:
        lines.append(
            f"  日度 DE 均值      {roll.daily_de_mean:+.3f}"
            f" ± {roll.daily_de_std:.3f}（{roll.n_daily_de} 日，观测间不独立，仅供参考）"
        )
    lines.append(f"  → {roll.verdict}")

    # 口径对照：把两个数放一起，让人自己判断该信哪个
    if rep is not None and rep.has_full:
        snap = rep.disposition_effect or 0.0
        if roll.de is not None:
            diff = roll.de - snap
            lines.append("")
            lines.append(
                f"  口径对照：期末快照 DE {snap:+.2%}"
                f"  vs  逐日滚动 DE {roll.de:+.2%}  （差 {diff * 100:+.1f} 个百分点）"
            )
            lines.append("  " + _compare_verdict(snap, roll.de))
    for n in roll.notes:
        lines.append(f"   · {n}")
    return "\n".join(lines)


def _render_benchmark(b: BenchmarkReport) -> str:
    """同行基准板块。

    刻意不叫"分位数"/"排名" —— 文献只给群体点估计，没有个体分布，
    算不出百分位。这里做的是区间包含检验：文献值是否落在你的置信区间内。
    """
    lines = ["", "【同行基准 · 点估计区间对照】"]

    if not b.comparable:
        lines.append("  不做对照 —— 这不是缺功能，是口径不可比。")
        for seg in _wrap(b.blocked_reason or "", 66):
            lines.append(f"   {seg}")
        return "\n".join(lines)

    # 同一条基准可能同时提供 DE 与比值两种口径，按基准合并成一块
    order: list[str] = []
    grouped: dict[str, list[Comparison]] = {}
    for c in b.comparisons:
        if c.bench.id not in grouped:
            grouped[c.bench.id] = []
            order.append(c.bench.id)
        grouped[c.bench.id].append(c)

    for bid in order:
        items = grouped[bid]
        bench = items[0].bench
        lines.append("")
        lines.append(f"  {bench.market}")
        lines.append(f"    出处：{bench.source}")
        lines.append(f"    样本：{bench.sample_size}（{bench.sample_period}）")

        for c in items:
            if c.metric == "de":
                lines.append(
                    f"    文献 PGR {bench.pgr:.1%} / PLR {bench.plr:.1%}"
                    f"  →  DE {bench.de * 100:+.1f} 个百分点"
                )
                mine = "—" if c.mine is None else f"{c.mine * 100:+.1f} 个百分点"
                ci = ""
                if b.ci_low is not None and b.ci_high is not None:
                    ci = (f"（95% 区间 {b.ci_low * 100:+.1f} ~ "
                          f"{b.ci_high * 100:+.1f} 个百分点）")
                lines.append(f"    你的 DE {mine}{ci}")
            else:
                note = f"（{bench.ratio_note}）" if bench.ratio_note else ""
                lines.append(f"    文献 PGR/PLR ≈ {bench.ratio:.2f}{note}")
                mine = "—" if c.mine is None else f"{c.mine:.2f}"
                lines.append(f"    你的 PGR/PLR {mine}（比值口径，未做显著性）")

            for i, seg in enumerate(_wrap(c.text, 62)):
                lines.append(f"    {'→ ' if i == 0 else '  '}{seg}")

        if bench.caveat:
            for seg in _wrap("口径提醒：" + bench.caveat, 64):
                lines.append(f"      {seg}")

    if b.notes:
        lines.append("")
    for n in b.notes:
        for i, seg in enumerate(_wrap(n, 66)):
            lines.append(f"  {'⚠ ' if i == 0 else '  '}{seg}")

    return "\n".join(lines)


#: 折行切分：ASCII 词（带常见标点）｜空白串｜其余任意单字符。
#: 第三支是兜底 —— 少了它，破折号、方括号、波浪线会被 findall 静默丢弃，
#: 把「[0,1]」变成「0,1」这种内容损坏，而且看不出来。
_WRAP_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%/\-_:]*|\s+|.", re.S)
#: 优先在这些字符之后断行，避免把「极敏感」拆成「极敏 / 感」
_BREAK_CHARS = set("，。、；：！？）】」』—…%,;:)]·")


def _disp_w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _wrap(text: str, width: int) -> list[str]:
    """按显示宽度折行（中文按 2 列算），优先在标点或空格处断开。

    不能逐字符切：那样会把 0.148 切成 "0." + "148"，把 Odean 切成两截。
    也不能只在超宽处切：中文没有词边界，会切出「极敏 / 感」这种断词。
    做法是记住最近一个可断点，超宽时回退到那里。
    """
    out: list[str] = []
    cur = ""
    cur_w = 0
    break_at = -1          # cur 中最近的可断位置
    break_w = 0            # 断点之前的显示宽度

    for tok in _WRAP_TOKEN.findall(text):
        tok_w = 2 if unicodedata.east_asian_width(tok[0]) in ("W", "F") else len(tok)

        if cur and cur_w + tok_w > width:
            if 0 < break_at < len(cur):
                out.append(cur[:break_at].rstrip())
                cur = cur[break_at:]
                cur_w -= break_w
            else:
                out.append(cur.rstrip())
                cur, cur_w = "", 0
            break_at, break_w = -1, 0

            # 避头尾：中文标点不该出现在行首，挪回上一行末尾
            while cur and cur[0] in _BREAK_CHARS and out:
                out[-1] += cur[0]
                cur = cur[1:]
                cur_w -= 1

        if not cur and not tok.strip():
            continue        # 行首不留空白

        # 避头尾（补漏）：断点恰好落在行尾时，紧随的标点会掉到下一行开头，
        # 这里直接并回上一行末尾。
        if not cur and out and tok in _BREAK_CHARS:
            out[-1] += tok
            continue

        cur += tok
        cur_w += tok_w
        if tok in _BREAK_CHARS or tok.isspace():
            break_at, break_w = len(cur), cur_w

    if cur.strip():
        out.append(cur.rstrip())
    return out or [""]


def _compare_verdict(snap: float, rolling_de: float) -> str:
    """期末快照口径与逐日滚动口径的差异判读。

    分三档而不是简单报个数：
      方向相反 —— 最严重，说明"看期末那一天"会得出完全错误的结论；
      幅度差异 —— 方向一致但量级不同，通常因为历史中的浮亏没被计入；
      一致     —— 结论稳健，不依赖用哪天的价格。
    """
    diff = rolling_de - snap
    if snap * rolling_de < 0:
        return ("→ 两种口径**方向相反**：期末那一天的账面持仓不能代表历史，"
                "只看期末会得出完全错误的结论。以逐日滚动口径为准。")
    if abs(diff) < 0.05:
        return "→ 两种口径基本一致，结论稳健，不依赖用哪一天的价格。"
    if diff > 0:
        return ("→ 逐日滚动更高：历史上有大量「浮盈没卖」的机会被期末口径漏掉了，"
                "期末口径低估了你的处置效应。")
    return ("→ 逐日滚动更低：历史上「浮亏没割」的机会被期末口径漏掉了。"
            "注意：这说明你现在手里的浮亏不代表历史全貌。")


def render(paths: list[Path], result, ledger: PositionLedger, dup: int, limit: int,
           report: DispositionReport | None = None,
           file_metas: list[tuple[str, dict]] | None = None,
           calendar: TradingCalendar | None = None) -> str:
    lines: list[str] = []
    bar = "=" * 70

    lines.append(bar)
    lines.append("  tradeopsy · 交割单体检")
    lines.append(bar)

    lines.append("\n【文件】")
    # 逐文件展示元信息。多份文件来自不同券商/不同年份时，编码、表头位置、
    # 费用列形态都可能不同 —— 只显示第一份会让人误以为"所有文件的解析方式一样"，
    # 而这恰恰是排障时最需要看清的东西。
    if file_metas:
        for name, m in file_metas:
            head = f"  {name}"
            if m.get("encoding"):
                head += (f" · {m['encoding']} · 表头第 {m.get('header_row')} 行"
                         f" · {m.get('total_rows')} 行")
            lines.append(head)
            cols = m.get("columns") or {}
            mapped = " / ".join(f"{k}={v}" for k, v in cols.items() if v)
            if mapped:
                lines.append(f"     列映射 {mapped}")
            if m.get("fee_split_columns"):
                lines.append(f"     费用分项列 {m['fee_split_columns']}（已累加）")
    else:
        for p in paths:
            lines.append(f"  {p.name}")
    if dup:
        lines.append(f"  跨文件去重 {dup} 条重复记录")

    span = timeline(result.events)
    counts: dict[EventType, int] = defaultdict(int)
    for e in result.events:
        counts[e.etype] += 1

    lines.append("\n【数据覆盖】")
    if span:
        days = (span[1] - span[0]).days
        lines.append(f"  {span[0]} → {span[1]}（{days} 天）")
    lines.append(f"  标的 {len(result.symbols())} 只 · 事件 {len(result.events)} 条")
    detail = " · ".join(f"{TYPE_CN.get(k, k)} {v}" for k, v in sorted(
        counts.items(), key=lambda kv: -kv[1]))
    lines.append(f"  {detail}")

    s = summarize(ledger, calendar)
    lines.append("\n【交易汇总】")
    if calendar is not None:
        lines.append(f"  持有期口径 {calendar.describe()}")
    pct_keys = {"胜率"}
    for k, v in s.items():
        if k == "盈亏比" and v is None:
            display = "—（无亏损笔）"
        elif k in pct_keys and isinstance(v, float):
            display = f"{v:.2%}"
        elif isinstance(v, float):
            display = f"{v:,.2f}"
        else:
            display = str(v)
        lines.append(f"  {_pad(k, 22, 'left')}{display}")

    rows = _by_symbol(ledger.chunks)
    if rows:
        lines.append("\n【按标的】（按已实现盈亏升序，亏得最多的在最上面）")
        lines.append(
            f"  {_pad('代码', 10)}{_pad('回合', 6, 'right')}"
            f"{_pad('胜', 6, 'right')}{_pad('已实现盈亏', 14, 'right')}"
        )
        for sym, n, pnl, win in rows:
            lines.append(
                f"  {_pad(sym, 10)}{_pad(n, 6, 'right')}"
                f"{_pad(win, 6, 'right')}{_pad(f'{pnl:,.2f}', 14, 'right')}"
            )

    chunks = ledger.chunks
    shown = chunks if limit <= 0 else chunks[:limit]
    if shown:
        unit = "交易日" if calendar is not None else "自然日"
        lines.append(f"\n【逐笔已实现】{'全部' if limit <= 0 else f'前 {len(shown)} 笔'}")
        lines.append(
            f"  {_pad('标的', 9)}{_pad('建仓', 12)}{_pad('平仓', 12)}"
            f"{_pad('数量', 8, 'right')}{_pad('成本', 12, 'right')}"
            f"{_pad('净收入', 12, 'right')}{_pad('盈亏', 11, 'right')}"
            f"{_pad(unit, 7, 'right')}"
        )
        for c in shown:
            held = calendar.count(c.open_dt, c.close_dt) if calendar is not None else c.hold_days
            lines.append(
                f"  {_pad(c.symbol, 9)}{_pad(c.open_dt, 12)}{_pad(c.close_dt, 12)}"
                f"{_pad(f'{float(c.qty):.0f}', 8, 'right')}"
                f"{_pad(f'{float(c.cost):,.2f}', 12, 'right')}"
                f"{_pad(f'{float(c.proceeds):,.2f}', 12, 'right')}"
                f"{_pad(f'{float(c.pnl):,.2f}', 11, 'right')}"
                f"{_pad(held, 7, 'right')}"
            )

    open_pos = ledger.open_positions()
    if open_pos:
        lines.append(f"\n【未平仓】{len(open_pos)} 个批次")
        for p in open_pos[:10]:
            lines.append(
                f"  {_pad(p.symbol, 9)}{_pad(p.open_dt, 12)}"
                f"{_pad(f'{float(p.qty):.0f}', 9, 'right')}股"
                f"  成本 {float(p.unit_cost):.4f}"
            )
        if len(open_pos) > 10:
            lines.append(f"  ... 另 {len(open_pos) - 10} 个批次")

    all_warn = list(result.warnings) + list(ledger.warnings)
    if all_warn:
        lines.append(f"\n【告警】{len(all_warn)} 条（未中断解析，需人工复核）")
        for wn in all_warn[:15]:
            lines.append(f"  {wn}")
        if len(all_warn) > 15:
            lines.append(f"  ... 另 {len(all_warn) - 15} 条")

    if report is not None:
        lines.extend(_render_disposition(report))

    lines.append("")
    return "\n".join(lines)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """analyze 与 report 共用的输入参数 —— 两个命令必须能接受完全相同的输入，
    否则用户会在"命令行能跑、网页跑不出来"的差异上反复踩坑。"""
    p.add_argument("input", nargs="?", default="private",
                   help="交割单文件或目录，默认 private/")
    p.add_argument("--prices", default=None,
                   help="未平仓标的的最新价文件(CSV: 代码,价格)，用于算完整 PGR/PLR")
    p.add_argument("--price-history", default=None,
                   help="逐日价格序列(CSV: 日期,代码,收盘价)，用于逐卖出日滚动口径的 PGR/PLR"
                        "（Odean 原文口径）；未给 --prices 时自动用其末价补期末口径")
    p.add_argument("--calendar", default=None,
                   help="交易日历文件（每行 YYYY-MM-DD，建议用行情K线日期导出）；"
                        "缺省按工作日近似")
    p.add_argument("--natural-days", action="store_true",
                   help="持有期用自然日（对照用，非A股严谨口径）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tradeopsy", description="A股交割单体检：把流水变成看得懂的行为归因")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze", help="终端里看体检报告（纯文本）")
    _add_common_args(a)
    a.add_argument("--limit", type=int, default=0,
                   help="逐笔明细最多显示几笔，0=全部")
    a.add_argument("--out", default=None, help="把文本报告同时写入文件")

    r = sub.add_parser("report", help="生成单文件 HTML 体检报告（可截图、可分享）")
    _add_common_args(r)
    r.add_argument("-o", "--out", default="tradeopsy-report.html",
                   help="输出 HTML 路径，默认 tradeopsy-report.html")
    r.add_argument("--benchmark", default=None,
                   help="基准收盘价文件(CSV: YYYY-MM-DD,收盘价)，用于对比；不提供则不画")
    r.add_argument("--title", default="交易体检报告", help="报告标题")

    sub.add_parser("formats", help="列出支持的格式")

    args = ap.parse_args(argv)

    if args.cmd == "formats":
        from .parsers import PARSERS
        for p in PARSERS:
            print(f"{p.name:<16}{p.description}")
        return 0
    if args.cmd is None:
        ap.print_help()
        return 0

    target = Path(args.input)
    if not _collect_inputs(target):
        print(f"没找到可读的交割单。请把 CSV 放进 "
              f"{target if target.is_dir() else 'private/'} 目录。", file=sys.stderr)
        print("该目录已在 .gitignore 中隔离，文件不会进 git。", file=sys.stderr)
        return 2

    ds = load_dataset(target, calendar_path=args.calendar,
                      natural_days=args.natural_days)
    if not ds.events:
        from .parsers.base import ParseResult
        merged = ParseResult(events=[], warnings=ds.warnings)
        print(render(ds.paths, merged, PositionLedger(), ds.dup, 0, None,
                     ds.file_metas, ds.calendar))
        print("解析成功 0 条事件。请检查上面的【文件】信息与【告警】。", file=sys.stderr)
        return 1

    prices = read_prices(args.prices) if getattr(args, "prices", None) else None

    price_history: PriceHistory | None = None
    if getattr(args, "price_history", None):
        price_history = read_price_history(args.price_history)
        if price_history.problem:
            # 价格序列读不出来就不能假装算了滚动口径 —— 静默降级会让
            # 用户以为"这个功能对我没用"，而实际是他文件格式不对。
            print(f"价格序列读取失败：{price_history.problem}", file=sys.stderr)
            return 2
        print(f"价格序列：{price_history.describe()}", file=sys.stderr)
        if price_history.dups:
            print(f"  同一(代码,日期)重复 {price_history.dups} 次，已按后出现的值覆盖",
                  file=sys.stderr)
        if price_history.bad_rows:
            print(f"  无法解析的行 {price_history.bad_rows} 条，已跳过", file=sys.stderr)
        if not prices:
            # 只给序列也能跑两套口径，避免用户维护两份不同步的价格文件
            prices = price_history.snapshot()
            print(f"未给 --prices，用价格序列末价充当期末快照（{len(prices)} 只）",
                  file=sys.stderr)

    if args.cmd == "report":
        from .report import build_report

        if prices:
            print(f"未平仓价格：{len(prices)} 只", file=sys.stderr)
        bench = read_benchmark(args.benchmark) if args.benchmark else {}
        if args.benchmark and not bench:
            print(f"基准文件没读到有效数据：{args.benchmark}", file=sys.stderr)
        html = build_report(ds, prices=prices or None, benchmark=bench or None,
                            title=args.title, price_history=price_history)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        size_kb = len(html.encode("utf-8")) / 1024
        print(f"报告已生成：{out.resolve()}（{size_kb:.0f} KB，单文件、离线可看）")
        print("提醒：报告含你的持仓标的与金额，分享前请确认是否要脱敏。")
        return 0

    report = ds.disposition(prices or None)
    sig = ds.significance(prices or None) if prices else None
    roll = ds.rolling(price_history) if price_history else None

    from .parsers.base import ParseResult
    merged = ParseResult(events=ds.events, warnings=ds.warnings)
    out_text = render(ds.paths, merged, ds.ledger, ds.dup, args.limit,
                      report, ds.file_metas, ds.calendar)
    if roll is not None:
        out_text += "\n" + _render_rolling(roll, report)
        # 基准对照只能在滚动口径下做（见 analytics/benchmark.py 的口径约束）
        out_text += "\n" + _render_benchmark(ds.benchmark(roll))
    if sig is not None:
        out_text += "\n" + _render_significance(sig)
    print(out_text)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
