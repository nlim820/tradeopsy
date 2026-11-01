"""回放演示：用历史数据把纸面账户完整跑一遍。

**这不是账户记录，是功能演示。**
真实的 paper/state.json 是从 2026-10-30 开始逐日累积的，
commit 历史可以逐条核对。这里用临时目录跑，不碰真实账本。

用法：
    python examples/paper_replay.py                # 从 3 个月前开始
    python examples/paper_replay.py --since 2026-06-01

它证明两件事：
    1. 补跑能力 —— 中间缺多少天都能一次补齐，且结果与逐日运行一致
    2. 策略在这段行情里到底做了什么（包括空仓，空仓也是一种决策）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.paper import runner, strategy  # noqa: E402

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return SPARK[0] * len(values)
    return "".join(
        SPARK[min(int((v - lo) / (hi - lo) * (len(SPARK) - 1)), len(SPARK) - 1)]
        for v in values
    )


def _benchmark(since: date, until: date):
    """取沪深300ETF 作为基准，返回 (收益率, 期初价, 期末价)。

    基准用同一只 510300 的前复权序列 —— 和策略用的是同一份数据，
    这样对比才公平。如果基准用不复权价，会把分红算成下跌，凭空多送几个点超额。
    """
    from tradeopsy.paper import market

    try:
        bars = market.fetch_bars("510300", since, until)
    except Exception:  # noqa: BLE001
        return None
    bars = [b for b in bars if b.dt >= since]
    if len(bars) < 2:
        return None
    first, last = bars[0].open, bars[-1].close
    if first <= 0:
        return None
    return (float(last / first - 1), float(first), float(last))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="回放起点 YYYY-MM-DD")
    ap.add_argument("--today", default=None, help="终点 YYYY-MM-DD，默认今天")
    args = ap.parse_args()

    today = (
        datetime.strptime(args.today, "%Y-%m-%d").date()
        if args.today
        else date.today()
    )
    since = (
        datetime.strptime(args.since, "%Y-%m-%d").date()
        if args.since
        else today - timedelta(days=90)
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "paper").mkdir()
        (root / "paper" / "state.json").write_text(
            json.dumps({
                "version": 1,
                "as_of": since.isoformat(),
                "cash": "100000.00",
                "initial_cash": "100000.00",
                "positions": {},
                "pending_from": "",
                "pending_targets": {},
            }),
            encoding="utf-8",
        )

        summary = runner.run(root, today=today, verbose=False)

        nav_rows = list(
            csv.DictReader((root / "paper" / "nav.csv").open(encoding="utf-8-sig"))
        )
        trades: list[dict] = []
        tp = root / "paper" / "trades.csv"
        if tp.exists():
            trades = list(
                csv.DictReader(tp.open(encoding="utf-8-sig"))
            )

    if not nav_rows:
        print("没有可回放的数据。")
        return 1

    print("=" * 68)
    print(f"  tradeopsy 纸面账户 · 回放 {since} → {nav_rows[-1]['日期']}")
    print("  （功能演示，非真实账户记录）")
    print("=" * 68)

    vals = [float(r["总资产"]) for r in nav_rows]
    print(f"\n交易日数   {len(nav_rows)}")
    print(f"期初       100,000.00")
    print(f"期末       {vals[-1]:,.2f}")
    print(f"累计收益   {(vals[-1] / 100000 - 1):+.2%}")
    print(f"峰值       {max(vals):,.2f}")
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    print(f"最大回撤   {mdd:.2%}")

    # 基准对比。这一段不能省：只看绝对收益会得出"策略不行"的错误结论，
    # 而同期宽基可能跌得更多。超额收益才是这个账户真正在证明的东西。
    bench = _benchmark(since, date.fromisoformat(nav_rows[-1]["日期"]))
    if bench is not None:
        b_ret, b_first, b_last = bench
        excess = (vals[-1] / 100000 - 1) - b_ret
        print(f"基准      沪深300ETF {b_first:.3f} → {b_last:.3f}，{b_ret:+.2%}")
        print(f"超额收益   {excess:+.2%}")

    print(f"\n净值曲线（{len(vals)} 个交易日）")
    print("  " + sparkline(vals))

    if trades:
        print(f"\n成交明细（{len(trades)} 笔）")
        print(f"  {'日期':<12}{'方向':<10}{'标的':<10}{'数量':>8}{'价格':>9}{'金额':>12}")
        for t in trades:
            print(
                f"  {t['成交日期']:<12}{t['业务名称']:<10}"
                f"{t['证券名称']:<10}{float(t['成交数量']):>8.0f}"
                f"{float(t['成交价格']):>9.3f}{float(t['成交金额']):>12,.2f}"
            )
        total_fee = sum(float(t["费用"]) for t in trades)
        print(f"  手续费合计 {total_fee:,.2f} 元"
              f"（占期初 {total_fee / 100000:.2%}）")
    else:
        print("\n期间无成交（全程未触发买入信号）")

    last_date = nav_rows[-1]["日期"]
    print(f"\n{last_date} 目标：{summary.get('targets') or '空仓'}")
    print(f"数据源：{', '.join(summary.get('sources', []))}")
    for w in summary.get("warnings", []):
        print(f"[warn] {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
