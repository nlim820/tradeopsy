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
from decimal import Decimal
from pathlib import Path

from .analytics.disposition import (
    DispositionReport,
    compute_disposition,
    open_positions_from_ledger,
)
from .analytics.significance import SignificanceResult, bootstrap_disposition
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


def _load_prices(path: str) -> dict[str, Decimal]:
    """读未平仓标的最新价文件。

    支持两种格式：带表头（证券代码/代码 + 最新价/价格）或不带表头（第1列代码、第2列价）。
    行内若混入千分位逗号先去掉。
    """
    out: dict[str, Decimal] = {}
    p = Path(path)
    if not p.exists():
        print(f"价文件不存在：{path}", file=sys.stderr)
        return out
    with p.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return out
    header = [c.strip() for c in rows[0]]
    if header and header[0] in ("证券代码", "代码", "symbol", "Symbol"):
        data = rows[1:]
    else:
        data = rows
    for r in data:
        if len(r) < 2:
            continue
        sym = r[0].strip()
        try:
            val = Decimal(r[1].strip().replace(",", ""))
        except Exception:
            continue
        if sym:
            out[sym] = val
    return out


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
        f"  胜率 {rep.win_rate:.2%} · 赢家均持 {rep.avg_hold_win} 天"
        f" · 输家均持 {rep.avg_hold_loss} 天"
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
           report: DispositionReport | None = None) -> str:
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

    if report is not None:
        lines.extend(_render_disposition(report))

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
    a.add_argument("--prices", default=None,
                   help="未平仓标的的最新价文件(CSV: 代码,价格)，用于算完整 PGR/PLR")
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

    report = None
    sig = None
    if ledger.chunks:
        prices = _load_prices(args.prices) if args.prices else None
        opens = open_positions_from_ledger(ledger, prices)
        report = compute_disposition(ledger.chunks, opens if prices is not None else None)
        if prices is not None:
            sig = bootstrap_disposition(ledger.chunks, opens)

    out_text = render(paths, merged, ledger, dup, args.limit, report)
    if sig is not None:
        out_text += "\n" + _render_significance(sig)
    print(out_text)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
