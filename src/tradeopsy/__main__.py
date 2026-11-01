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
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from .analytics.disposition import (
    DispositionReport,
    compute_disposition,
    open_positions_from_ledger,
)
from .analytics.significance import SignificanceResult, bootstrap_disposition
from .ledger import PositionLedger, summarize, timeline
from .load import collect_inputs as _collect_inputs
from .load import dedupe as _dedupe
from .load import load_dataset, read_benchmark, read_calendar, read_prices
from .model import EventType
from .parsers import parse_file
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

    if args.cmd == "report":
        from .report import build_report

        if prices:
            print(f"未平仓价格：{len(prices)} 只", file=sys.stderr)
        bench = read_benchmark(args.benchmark) if args.benchmark else {}
        if args.benchmark and not bench:
            print(f"基准文件没读到有效数据：{args.benchmark}", file=sys.stderr)
        html = build_report(ds, prices=prices or None, benchmark=bench or None,
                            title=args.title)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        size_kb = len(html.encode("utf-8")) / 1024
        print(f"报告已生成：{out.resolve()}（{size_kb:.0f} KB，单文件、离线可看）")
        print("提醒：报告含你的持仓标的与金额，分享前请确认是否要脱敏。")
        return 0

    report = ds.disposition(prices or None)
    sig = ds.significance(prices or None) if prices else None

    from .parsers.base import ParseResult
    merged = ParseResult(events=ds.events, warnings=ds.warnings)
    out_text = render(ds.paths, merged, ds.ledger, ds.dup, args.limit,
                      report, ds.file_metas, ds.calendar)
    if sig is not None:
        out_text += "\n" + _render_significance(sig)
    print(out_text)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
