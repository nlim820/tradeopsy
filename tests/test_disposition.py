"""处置效应度量测试（golden fixtures）。

预期值全部按 Odean (1998) 口径手算：
    PGR = RG / (RG + PG)
    PLR = RL / (RL + PL)
    DE  = PLR - PGR

不依赖任何真实交割单，也不联网。对不上就是 bug。
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.analytics.disposition import (  # noqa: E402
    DispositionReport,
    OpenPositionMV,
    compute_disposition,
    open_positions_from_ledger,
)
from tradeopsy.ledger import PositionLedger  # noqa: E402
from tradeopsy.model import Event, EventType, RealizedChunk  # noqa: E402

D = Decimal


def chunk(symbol: str, open_d: date, close_d: date, cost: str, pnl: str) -> RealizedChunk:
    """构造一笔已实现片段，cost 给定，proceeds = cost + pnl。"""
    c = D(cost)
    return RealizedChunk(
        symbol=symbol,
        open_dt=open_d,
        close_dt=close_d,
        qty=D("100"),
        cost=c,
        proceeds=money(c + D(pnl)),
    )


def mv(symbol: str, cost: str, market: str) -> OpenPositionMV:
    return OpenPositionMV(symbol=symbol, cost=D(cost), market_value=D(market))


def money(x: Decimal) -> Decimal:
    from tradeopsy.model import money as _m
    return _m(x)


class TestDispositionNoEffect(unittest.TestCase):
    """构造一个 PGR == PLR 的中性样本，DE 应为 0。"""

    def setUp(self) -> None:
        self.chunks = [
            chunk("A", date(2026, 1, 5), date(2026, 3, 5), "1000.00", "1000.00"),   # win +1000
            chunk("A", date(2026, 1, 5), date(2026, 3, 5), "2000.00", "2000.00"),   # win +2000
            chunk("B", date(2026, 2, 1), date(2026, 4, 1), "1000.00", "-500.00"),   # loss -500
            chunk("B", date(2026, 2, 1), date(2026, 4, 1), "2000.00", "-1000.00"),  # loss -1000
        ]
        self.opens = [
            mv("C", "1000.00", "4000.00"),   # paper gain +3000
            mv("D", "2000.00", "500.00"),     # paper loss -1500
        ]

    def test_pgr_plr_equal(self):
        r = compute_disposition(self.chunks, self.opens)
        self.assertAlmostEqual(r.pgr, 0.5, places=6)   # 3000 / (3000 + 3000)
        self.assertAlmostEqual(r.plr, 0.5, places=6)   # 1500 / (1500 + 1500)
        self.assertAlmostEqual(r.disposition_effect, 0.0, places=6)

    def test_realized_aggregates(self):
        r = compute_disposition(self.chunks, self.opens)
        self.assertEqual(r.realized_gain, 3000.0)
        self.assertEqual(r.realized_loss, 1500.0)
        self.assertEqual(r.paper_gain, 3000.0)
        self.assertEqual(r.paper_loss, 1500.0)
        self.assertEqual(r.n_realized, 4)
        self.assertEqual(r.n_open, 2)
        self.assertAlmostEqual(r.win_rate, 0.5, places=4)

    def test_mode_full_and_verdict_balanced(self):
        r = compute_disposition(self.chunks, self.opens)
        self.assertEqual(r.mode, "full")
        self.assertIn("处置效应不明显", r.verdict)


class TestDispositionStrongEffect(unittest.TestCase):
    """明显处置效应：PGR 低、PLR 高，DE > 0.10。"""

    def setUp(self) -> None:
        self.chunks = [
            chunk("A", date(2026, 1, 5), date(2026, 3, 5), "1000.00", "1000.00"),   # rg +1000
            chunk("C", date(2026, 2, 1), date(2026, 4, 1), "1000.00", "-200.00"),   # rl -200
        ]
        self.opens = [
            mv("B", "1000.00", "4000.00"),   # pg +3000 -> PGR = 1000/4000 = 0.25
            mv("D", "1000.00", "800.00"),     # pl +200  -> PLR = 200/400 = 0.5
        ]

    def test_pgr_plr_and_effect(self):
        r = compute_disposition(self.chunks, self.opens)
        self.assertAlmostEqual(r.pgr, 0.25, places=6)
        self.assertAlmostEqual(r.plr, 0.5, places=6)
        self.assertAlmostEqual(r.disposition_effect, 0.25, places=6)

    def test_verdict_strong(self):
        r = compute_disposition(self.chunks, self.opens)
        self.assertIn("明显处置效应", r.verdict)


class TestRealizedOnly(unittest.TestCase):
    """未给未平仓市值时退化为 realized_only，PGR/PLR 应为 None。"""

    def test_degenerate(self):
        chunks = [
            chunk("A", date(2026, 1, 5), date(2026, 3, 5), "1000.00", "1000.00"),
            chunk("C", date(2026, 2, 1), date(2026, 4, 1), "1000.00", "-200.00"),
        ]
        r = compute_disposition(chunks)  # 不传 open_positions
        self.assertIsNone(r.pgr)
        self.assertIsNone(r.plr)
        self.assertIsNone(r.disposition_effect)
        self.assertEqual(r.mode, "realized_only")
        self.assertIn("未提供未平仓市值", r.notes[0])

    def test_win_rate_still_computed(self):
        chunks = [
            chunk("A", date(2026, 1, 5), date(2026, 3, 5), "1000.00", "1000.00"),
            chunk("C", date(2026, 2, 1), date(2026, 4, 1), "1000.00", "-200.00"),
        ]
        r = compute_disposition(chunks)
        self.assertAlmostEqual(r.win_rate, 0.5, places=4)


class TestTradingDayHold(unittest.TestCase):
    """持有天数：自然日 vs 交易日（跨周末时应更短）。"""

    def test_calendar_days(self):
        chunks = [
            chunk("A", date(2026, 1, 5), date(2026, 1, 11), "1000.00", "100.00"),
            chunk("A", date(2026, 1, 5), date(2026, 1, 11), "1000.00", "100.00"),
        ]
        r = compute_disposition(chunks)  # 无交易日历
        self.assertEqual(r.avg_hold_win, 6.0)  # 1/5 -> 1/11 自然日 6 天
        self.assertFalse(r.hold_days_trading)

    def test_trading_days_excludes_weekend(self):
        # 1/5 周一 ~ 1/11 周日：只有 1/5~1/9 是交易日；开仓后到平仓之间的交易日为 1/6,7,8,9 共 4 天
        # 注意：交易日历不能把周六(1/10)、周日(1/11)算进去，否则会退化回自然日
        cal = {date(2026, 1, d) for d in (5, 6, 7, 8, 9)}
        chunks = [
            chunk("A", date(2026, 1, 5), date(2026, 1, 11), "1000.00", "100.00"),
            chunk("A", date(2026, 1, 5), date(2026, 1, 11), "1000.00", "100.00"),
        ]
        r = compute_disposition(chunks, trading_calendar=cal)
        self.assertEqual(r.avg_hold_win, 4.0)
        self.assertTrue(r.hold_days_trading)
        self.assertIn("交易日", r.notes[1])


class TestOpenPositionsFromLedger(unittest.TestCase):
    """账本未平仓头寸 + 注入行情价 -> OpenPositionMV。"""

    def test_builds_with_prices(self):
        ledger = PositionLedger()
        ledger.apply_all([
            Event(dt=date(2026, 1, 2), symbol="X", etype=EventType.BUY, qty=D("100"), price=D("10"), fee=D("5")),
            # 部分卖出，留 40 股未平
            Event(dt=date(2026, 3, 1), symbol="X", etype=EventType.SELL, qty=D("60"), price=D("12"), fee=D("3")),
        ])
        opens = open_positions_from_ledger(ledger, prices={"X": D("15")})
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0].symbol, "X")
        # 成本 = 100*10+5 = 1005；剩余 40 股 -> 成本 40/100*1005 = 402.00
        self.assertEqual(opens[0].cost, D("402.00"))
        # 市值 = 40 * 15 = 600
        self.assertEqual(opens[0].market_value, D("600.00"))
        self.assertEqual(opens[0].paper_gain, D("198.00"))

    def test_missing_price_falls_back_to_cost(self):
        ledger = PositionLedger()
        ledger.apply_all([
            Event(dt=date(2026, 1, 2), symbol="Y", etype=EventType.BUY, qty=D("100"), price=D("10"), fee=D("0")),
        ])
        opens = open_positions_from_ledger(ledger, prices=None)
        self.assertEqual(opens[0].paper_gain, D("0.00"))


if __name__ == "__main__":
    unittest.main()
