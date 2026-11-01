"""命令行入口：`tradeopsy analyze [路径]`

默认读 `private/` —— 那是唯一被 .gitignore 硬隔离的目录，
把交割单丢进去就不用担心误提交。

为什么不支持管道输入：交割单是多列结构化数据，
stdin 传进来的几乎一定是用户手工复制粘贴的残缺版本，
解析失败时无法回溯原文件，排障成本远高于收益。
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from .ledger import PositionLedger, summarize, timeline
from .model import EventType
from .parsers import parse_file

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


def _collect_inputs(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and not p.name.startswith(".")
        )
    return [target] if target.exists() else []


def _dedupe(events: list) -> tuple[list, int]:
    """跨文件去重。

    真实场景：用户按月导出，相邻两个文件的区间可能重叠几天
    （比如 3 月文件带上了 4 月 1 日的成交）。不去重就是把同一笔算两遍，
    而盈亏会直接翻倍 —— 这种错误看起来很"合理"，最难被发现。
    """
    seen: set[tuple] = set()
    out = []
    dup = 0
    for e in sorted(events, key=lambda x: (x.dt, x.symbol, x.etype.value)):
        key = (e.dt, e.symbol, e.etype, e.qty, e.price, e.fee, e.cash)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(e)
    return out, dup


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


def render(paths: list[Path], result, ledger: PositionLedger, dup: int, limit: int) -> str:
    lines: list[str] = []
    bar = "=" * 70

    lines.append(bar)
    lines.append("  tradeopsy · 交割单体检")
    lines.append(bar)

    lines.append("\n【文件】")
    for p in paths:
        lines.append(f"  {p.name}")
    m = result.meta
    if "encoding" in m:
        lines.append(
            f"  编码 {m.get('encoding')} · 分隔符 {m.get('delimiter')!r} · "
            f"表头第 {m.get('header_row')} 行 · 共 {m.get('total_rows')} 行"
        )
        cols = m.get("columns") or {}
        mapped = " / ".join(f"{k}={v}" for k, v in cols.items() if v)
        lines.append(f"  列映射 {mapped}")
        if m.get("fee_split_columns"):
            lines.append(f"  费用分项列 {m['fee_split_columns']}（已累加）")
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

    s = summarize(ledger)
    lines.append("\n【交易汇总】")
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
        lines.append(f"\n【逐笔已实现】{'全部' if limit <= 0 else f'前 {len(shown)} 笔'}")
        lines.append(
            f"  {_pad('标的', 9)}{_pad('建仓', 12)}{_pad('平仓', 12)}"
            f"{_pad('数量', 8, 'right')}{_pad('成本', 12, 'right')}"
            f"{_pad('净收入', 12, 'right')}{_pad('盈亏', 11, 'right')}{_pad('天数', 6, 'right')}"
        )
        for c in shown:
            lines.append(
                f"  {_pad(c.symbol, 9)}{_pad(c.open_dt, 12)}{_pad(c.close_dt, 12)}"
                f"{_pad(f'{float(c.qty):.0f}', 8, 'right')}"
                f"{_pad(f'{float(c.cost):,.2f}', 12, 'right')}"
                f"{_pad(f'{float(c.proceeds):,.2f}', 12, 'right')}"
                f"{_pad(f'{float(c.pnl):,.2f}', 11, 'right')}"
                f"{_pad(c.hold_days, 6, 'right')}"
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

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tradeopsy", description="A股交割单体检：把流水变成看得懂的行为归因")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze", help="分析交割单（文件或目录）")
    a.add_argument("input", nargs="?", default="private",
                   help="交割单文件或目录，默认 private/")
    a.add_argument("--limit", type=int, default=0,
                   help="逐笔明细最多显示几笔，0=全部")
    a.add_argument("--out", default=None, help="把报告同时写入文本文件")

    sub.add_parser("formats", help="列出支持的格式")

    args = ap.parse_args(argv)

    if args.cmd == "formats":
        from .parsers import PARSERS
        for p in PARSERS:
            print(f"{p.name:<16}{p.description}")
        return 0

    target = Path(getattr(args, "input", "private"))
    paths = _collect_inputs(target)
    if not paths:
        print(f"没找到可读的交割单。请把 CSV 放进 {target if target.is_dir() else 'private/'} 目录。",
              file=sys.stderr)
        print("该目录已在 .gitignore 中隔离，文件不会进 git。", file=sys.stderr)
        return 2

    all_events = []
    warnings_all = []
    meta: dict = {}
    for p in paths:
        r = parse_file(p)
        all_events.extend(r.events)
        warnings_all.extend(r.warnings)
        if not meta:
            meta = r.meta

    from .parsers.base import ParseResult
    merged = ParseResult(events=all_events, warnings=warnings_all, meta=meta)
    events, dup = _dedupe(merged.events)
    merged.events = events

    if not events:
        print(render(paths, merged, PositionLedger(), dup, 0))
        print("解析成功 0 条事件。请检查上面的【文件】信息与【告警】。", file=sys.stderr)
        return 1

    ledger = PositionLedger()
    ledger.apply_all(events)

    report = render(paths, merged, ledger, dup, args.limit)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
