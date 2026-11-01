"""处置效应演示卡（Wrapped 风格）。

**这是演示，不是你的真实数据。**

数据来源优先级：
    1. 仓库里的 paper/trades.csv —— 纸面账户回放的真实成交（需要你先跑过一次
       `tradeopsy-paper` 或等 GitHub Actions 自动跑出来）。这正好把你已经做好的
       纸面账户引擎，接成了处置效应引擎的"演示输入"。
    2. 内置合成场景 —— 离线也能跑，立刻看到一张完整的体检卡。

用法：
    python examples/disposition_demo.py            # 自动选数据源
    python examples/disposition_demo.py --live     # 强制用 paper/trades.csv（没有就报错）

未平仓市值默认用最新行情（联网取）；取不到时按成本计（paper gain = 0），
卡片会明确标注，不会假装算出了 paper 端。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.analytics.disposition import (  # noqa: E402
    OpenPositionMV,
    compute_disposition,
    open_positions_from_ledger,
)
from tradeopsy.analytics.significance import bootstrap_disposition  # noqa: E402
from tradeopsy.ledger import PositionLedger  # noqa: E402
from tradeopsy.model import Event, EventType, RealizedChunk, money  # noqa: E402
from tradeopsy.tradingcal import TradingCalendar  # noqa: E402

D = Decimal
REPO = Path(__file__).resolve().parents[1]
TRADES_CSV = REPO / "paper" / "trades.csv"


def _events_from_trades_csv(path: Path) -> list[Event]:
    """把 paper 回放产出的 trades.csv（已是标准中文列）转成 Event。"""
    events: list[Event] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            etype = EventType.BUY if row["业务名称"] == "证券买入" else EventType.SELL
            events.append(Event(
                dt=date.fromisoformat(row["成交日期"]),
                symbol=row["证券代码"],
                etype=etype,
                qty=D(row["成交数量"]),
                price=D(row["成交价格"]),
                fee=D(row["费用"]),
            ))
    return events


def _synthetic_events() -> list[Event]:
    """内置合成场景：刻意构造一个"明显处置效应"的账户，便于教学演示。

    处置效应 = 卖盈持亏（PGR 高、PLR 低），所以场景要这样搭：
    - A：赚 1000，15 天就落袋        -> 已实现盈利 rg=1000（快卖赢家）
    - B：浮盈 200，一直拿着没卖       -> 账面盈利 pg=200（拿住小赢家）
    - C：亏 200，扛了 89 天才割        -> 已实现亏损 rl=200（慢割输家）
    - D：浮亏 1000，继续死扛          -> 账面亏损 pl=1000（拿住大输家）

    => PGR = 1000/(1000+200)  = 0.8333
       PLR =  200/(200+1000)  = 0.1667
       DE  = PGR - PLR        = +0.6667  -> 明显处置效应
    同时赢家均持 15 天、输家均持 89 天，构成"赢家拿得比输家短"的典型指纹。
    """
    return [
        Event(date(2026, 1, 5), "A", EventType.BUY, D("100"), D("10"), D("0")),
        Event(date(2026, 1, 20), "A", EventType.SELL, D("100"), D("20"), D("0")),
        Event(date(2026, 1, 10), "B", EventType.BUY, D("100"), D("10"), D("0")),   # 未平仓
        Event(date(2026, 2, 1), "C", EventType.BUY, D("100"), D("10"), D("0")),
        Event(date(2026, 5, 1), "C", EventType.SELL, D("100"), D("8"), D("0")),
        Event(date(2026, 3, 2), "D", EventType.BUY, D("200"), D("10"), D("0")),    # 未平仓
    ]


def _synthetic_prices() -> dict[str, Decimal]:
    # B 现价 12（浮盈 200），D 现价 5（浮亏 1000）
    return {"B": D("12"), "D": D("5")}


def _last_prices(symbols: set[str]) -> dict[str, Decimal]:
    """尝试用联网行情取最新价（仅供未平仓市值）。失败返回空 dict。"""
    if not symbols:
        return {}
    try:
        from tradeopsy.paper import market
        out: dict[str, Decimal] = {}
        for s in symbols:
            bars = market.fetch_bars(s, date.today().replace(day=1) - __import__("datetime").timedelta(days=40), date.today())
            if bars:
                out[s] = money(bars[-1].close)
        return out
    except Exception:  # noqa: BLE001
        return {}


def _card(rep, source: str) -> str:
    bar = "=" * 60
    lines = [bar, "  tradeopsy · 交易行为体检卡（处置效应）", bar, f"  数据来源：{source}"]
    if rep.has_full:
        lines.append(f"\n  PGR 盈利实现比例（金额口径）   {rep.pgr:.2%}")
        lines.append(f"  PLR 亏损实现比例（金额口径）   {rep.plr:.2%}")
        de = rep.disposition_effect or 0.0
        lines.append(f"  处置效应 DE（金额口径）        {de:+.2%}")
        lines.append(f"\n  → {rep.verdict}")
    else:
        lines.append("\n  PGR/PLR 不可用（缺未平仓市值）：")
        lines.append(f"  已实现盈利 {rep.realized_gain:,.2f} · 已实现亏损 {rep.realized_loss:,.2f}")
        lines.append(f"\n  → {rep.verdict}")

    unit = "交易日" if rep.hold_days_trading else "自然日"
    lines.append(f"\n  胜率 {rep.win_rate:.2%}"
                 f" · 赢家平均持有 {rep.avg_hold_win} {unit}"
                 f" · 输家平均持有 {rep.avg_hold_loss} {unit}")
    if rep.hold_days_trading and rep.avg_hold_win_calendar:
        lines.append(f"  （按自然日对照：赢家 {rep.avg_hold_win_calendar} 天"
                     f" · 输家 {rep.avg_hold_loss_calendar} 天）")
    if rep.avg_hold_win and rep.avg_hold_loss and rep.mode == "full":
        if rep.avg_hold_win < rep.avg_hold_loss:
            lines.append("  （赢家拿得比输家短 —— 处置效应的典型指纹）")
        elif rep.avg_hold_win > rep.avg_hold_loss:
            lines.append("  （赢家拿得比输家长 —— 反向，少见）")

    lines.append("\n  说明：")
    for n in rep.notes:
        lines.append(f"   · {n}")
    lines.append("")
    return "\n".join(lines)


def _large_dataset() -> tuple[list[RealizedChunk], list[OpenPositionMV]]:
    """教学用大样本：卖盈持亏 -> 计数 PGR=0.80 > PLR=0.45，DE=+0.35，足以检出显著性。

    每笔盈亏金额相同（±100），所以金额口径与计数口径一致，便于对照。
    """
    chunks: list[RealizedChunk] = []
    for i in range(32):
        chunks.append(RealizedChunk(f"G{i}", date(2026, 1, 5), date(2026, 2, 5),
                                    D("100"), D("1000"), money(D("1100"))))
    for i in range(18):
        chunks.append(RealizedChunk(f"L{i}", date(2026, 1, 5), date(2026, 2, 5),
                                    D("100"), D("1000"), money(D("900"))))
    opens = [OpenPositionMV(f"PG{i}", D("1000"), D("1200")) for i in range(8)]
    opens += [OpenPositionMV(f"PL{i}", D("1000"), D("800")) for i in range(22)]
    return chunks, opens


def _sig_card(sig, label: str = "计数口径 · Odean 原文") -> str:
    lines = ["", "-" * 60, f"  显著性检验（{label}）"]
    lines.append(
        f"  机会分布：盈利 卖出 {sig.n_gain_sold} / 持有 {sig.n_gain_held}"
        f" · 亏损 割肉 {sig.n_loss_sold} / 持有 {sig.n_loss_held}"
    )
    if sig.pgr_count is not None and sig.plr_count is not None:
        lines.append(
            f"  计数 PGR {sig.pgr_count:.2%} · PLR {sig.plr_count:.2%}"
            f" · DE {sig.de_count:+.2%}"
        )
    if sig.ci_low is not None and sig.ci_high is not None:
        z = f" · 双比例 z p={sig.p_value_z:.4f}" if sig.p_value_z is not None else ""
        lines.append(
            f"  95% CI [{sig.ci_low:+.2%}, {sig.ci_high:+.2%}]"
            f" · bootstrap p={sig.p_value_boot:.4f}{z}"
        )
    lines.append(f"  → {sig.verdict}")
    for n in sig.notes:
        lines.append(f"   · {n}")
    lines.append("   注：金额口径与计数口径可能不一致，学术判定以计数口径为准。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="强制使用 paper/trades.csv")
    ap.add_argument("--large", action="store_true", help="用大样本教学数据集演示显著性")
    args = ap.parse_args()

    if args.large:
        chunks, opens = _large_dataset()
        source = "内置合成场景·大样本（教学用，非真实数据）"
    elif args.live or TRADES_CSV.exists():
        if not TRADES_CSV.exists():
            print(f"没有 {TRADES_CSV}，无法用 --live。", file=sys.stderr)
            return 1
        events = _events_from_trades_csv(TRADES_CSV)
        source = f"paper/trades.csv（{len(events)} 笔成交，回放演示数据）"
        ledger = PositionLedger()
        ledger.apply_all(events)
        prices = _last_prices({op.symbol for op in ledger.open_positions()})
        opens = open_positions_from_ledger(ledger, prices or None)
        chunks = ledger.chunks
    else:
        events = _synthetic_events()
        source = "内置合成场景（离线演示，非真实数据）"
        ledger = PositionLedger()
        ledger.apply_all(events)
        opens = open_positions_from_ledger(ledger, _synthetic_prices())
        chunks = ledger.chunks

    # 交易日口径：离线演示没有行情 K 线，用工作日近似（比自然日接近 A 股实际）。
    # 真实使用时 `analyze --calendar 行情日期.csv` 可拿到完全精确的日历。
    cal = None
    if chunks:
        cal = TradingCalendar.weekday_approx(
            min(c.open_dt for c in chunks), max(c.close_dt for c in chunks)
        )

    rep = compute_disposition(chunks, opens, trading_calendar=cal)
    sig = bootstrap_disposition(chunks, opens)
    print(_card(rep, source))
    print(_sig_card(sig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
