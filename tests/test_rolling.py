"""逐日滚动口径 PGR/PLR（analytics/rolling.py）测试。

这个模块的三个致命错误，测试必须一个一个钉死：

1. **T+1 例外**：当日买入的批次当天不可卖，把它算进"本可以卖却没卖"
   的分母是错的 —— 这一条没有测试盯着，迟早有人在重构时"顺手"删掉。
2. **等价性**：逐日重放的账本必须和一次性 apply_all 的账本完全一致。
   如果哪天有人改了 ledger 的排序假设，这个测试会第一个叫。
3. **假精确防线**：账面端两侧全 0 的日子（当日清仓）必须跳过，
   否则会出现 PGR=1.0 这种数学成立但毫无意义的数。
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.analytics.rolling import (  # noqa: E402
    DailyObs,
    rolling_disposition,
    replay_by_day,
)
from tradeopsy.ledger import PositionLedger  # noqa: E402
from tradeopsy.model import Event, EventType  # noqa: E402
from tradeopsy.prices import PriceHistory  # noqa: E402
from tradeopsy.tradingcal import TradingCalendar  # noqa: E402

D = Decimal
B, S = EventType.BUY, EventType.SELL

CAL = TradingCalendar.weekday_approx(date(2026, 1, 1), date(2026, 3, 1))


def ev(d: str, sym: str, t: EventType, qty: str, px: str) -> Event:
    return Event(dt=date.fromisoformat(d), symbol=sym, etype=t, qty=D(qty), price=D(px))


def flat_prices(ph: PriceHistory, symbols: list[str], days: list[str], px: str) -> None:
    for d in days:
        for s in symbols:
            ph.add(s, date.fromisoformat(d), D(px))


class TestReplayEquivalence(unittest.TestCase):
    def test_chunks_identical_to_apply_all(self):
        """逐日重放与一次性喂入，产出的已实现片段必须逐字段相等。"""
        events = [
            ev("2026-01-05", "A", B, "1000", "10"),
            ev("2026-01-12", "A", S, "1000", "12"),
            ev("2026-01-05", "B", B, "1000", "20"),
            ev("2026-01-20", "B", S, "500", "18"),
            ev("2026-02-02", "A", B, "500", "11"),
        ]
        one_shot = PositionLedger()
        one_shot.apply_all(events)
        replayed = [c for _, _, chunks in replay_by_day(events) for c in chunks]

        self.assertEqual(len(one_shot.chunks), len(replayed))
        for a, b in zip(one_shot.chunks, replayed):
            self.assertEqual((a.symbol, a.open_dt, a.close_dt, a.qty, a.cost, a.proceeds),
                             (b.symbol, b.open_dt, b.close_dt, b.qty, b.cost, b.proceeds))

    def test_final_positions_identical(self):
        events = [ev("2026-01-05", "A", B, "1000", "10"),
                  ev("2026-01-20", "A", S, "400", "12")]
        one_shot = PositionLedger()
        one_shot.apply_all(events)
        last_ledger = replay_by_day(events)[-1][1]
        self.assertEqual(
            [(p.symbol, p.qty, p.cost) for p in one_shot.open_positions()],
            [(p.symbol, p.qty, p.cost) for p in last_ledger.open_positions()],
        )


class TestGoldenSample(unittest.TestCase):
    """手算样本：卖出 A（+2000），同时持有 B（浮亏）与 C（浮盈）。"""

    def setUp(self):
        self.events = [
            ev("2026-01-05", "A", B, "1000", "10"),
            ev("2026-01-12", "A", S, "1000", "12"),
            ev("2026-01-05", "B", B, "1000", "20"),
            ev("2026-01-05", "C", B, "1000", "30"),
        ]
        self.ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]:
            dd = date.fromisoformat(d)
            self.ph.add("A", dd, D("11"))
            self.ph.add("B", dd, D("18"))   # 浮亏 2000
            self.ph.add("C", dd, D("33"))   # 浮盈 3000
        self.rep = rolling_disposition(self.events, self.ph, trading_calendar=CAL)

    def test_one_observation(self):
        self.assertEqual(self.rep.mode, "rolling")
        self.assertEqual(self.rep.n_obs, 1)

    def test_hand_computed(self):
        o = self.rep.obs[0]
        self.assertEqual((o.n_gain_sold, o.n_loss_sold), (1, 0))
        self.assertEqual((o.n_gain_open, o.n_loss_open), (1, 1))
        # PGR = 1/(1+1) = 0.5；PLR = 0/(0+1) = 0
        self.assertAlmostEqual(self.rep.pgr, 0.5)
        self.assertAlmostEqual(self.rep.plr, 0.0)
        self.assertAlmostEqual(self.rep.de, 0.5)

    def test_amount_pool(self):
        # 金额口径：已实现 +2000、账面 +3000 / -2000
        # PGR_amt = 2000/(2000+3000) = 0.4；PLR_amt = 0
        self.assertAlmostEqual(self.rep.pgr_amount, 0.4)
        self.assertAlmostEqual(self.rep.plr_amount, 0.0)
        self.assertAlmostEqual(self.rep.de_amount, 0.4)


class TestT1(unittest.TestCase):
    def test_same_day_buy_excluded(self):
        """当日买入的批次当天不可卖，不得进账面端分母。"""
        events = [
            ev("2026-01-05", "A", B, "1000", "10"),
            ev("2026-01-12", "A", S, "1000", "12"),
            ev("2026-01-05", "B", B, "1000", "20"),      # 支撑持仓：保证当日账面端非空
            ev("2026-01-12", "D", B, "1000", "10"),      # 当天买入且当天大涨
        ]
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]:
            dd = date.fromisoformat(d)
            ph.add("A", dd, D("11"))
            ph.add("B", dd, D("18"))
        ph.add("D", date.fromisoformat("2026-01-12"), D("12"))
        rep = rolling_disposition(events, ph, trading_calendar=CAL)
        o = rep.obs[0]
        # D 当日 +20% 也不得出现；若被错误计入，n_gain_open 会是 2
        self.assertEqual(o.n_gain_open, 0, "当日买入的 D 不得计入账面端")
        self.assertEqual(o.n_loss_open, 1)
        self.assertEqual(o.unpriced, 0)

    def test_yesterday_buy_included(self):
        """昨天买的今天就持有，是合法的账面端成员（部分卖出的剩余批次照常估值）。"""
        events = [
            ev("2026-01-05", "A", B, "1000", "10"),
            ev("2026-01-12", "A", S, "500", "12"),
            ev("2026-01-05", "B", B, "1000", "20"),
        ]
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]:
            dd = date.fromisoformat(d)
            ph.add("A", dd, D("11"))
            ph.add("B", dd, D("18"))
        rep = rolling_disposition(events, ph, trading_calendar=CAL)
        o = rep.obs[0]
        self.assertEqual(o.n_gain_open, 1)   # A 剩余 500 股：成本 5000、现价 5500 → 浮盈
        self.assertEqual(o.n_loss_open, 1)   # B 浮亏


class TestSkipAndDegenerate(unittest.TestCase):
    def test_full_liquidation_day_skipped(self):
        """当日清仓 → 账面端两侧全 0 → 整日跳过，绝不输出 PGR=1。"""
        events = [
            ev("2026-01-05", "A", B, "1000", "10"),
            ev("2026-01-12", "A", S, "1000", "12"),   # 唯一持仓也被卖光
        ]
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]:
            ph.add("A", date.fromisoformat(d), D("11"))
        rep = rolling_disposition(events, ph)
        self.assertEqual(rep.mode, "insufficient")
        self.assertEqual(rep.skipped_days, 1)
        self.assertIsNone(rep.pgr)

    def test_no_prices(self):
        events = [ev("2026-01-05", "A", B, "1000", "10")]
        rep = rolling_disposition(events, PriceHistory())
        self.assertEqual(rep.mode, "no_prices")
        self.assertTrue(rep.notes)

    def test_price_gap_symbols_listed(self):
        events = [ev("2026-01-05", "A", B, "1000", "10")]
        ph = PriceHistory()
        ph.add("OTHER", date.fromisoformat("2026-01-05"), D("1"))
        rep = rolling_disposition(events, ph)
        self.assertIn("A", rep.missing_symbols)


class TestStalePrice(unittest.TestCase):
    def _events(self):
        return [ev("2026-01-05", "A", B, "1000", "10"),
                ev("2026-01-05", "B", B, "1000", "20"),
                ev("2026-01-12", "A", S, "1000", "12")]

    def test_stale_counted_in_trading_days(self):
        """B 只有 01-05 的价：到 01-12 恰好 5 个交易日，30 天限内可用。"""
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                  "2026-01-09", "2026-01-12"]:
            ph.add("A", date.fromisoformat(d), D("11"))
        ph.add("B", date.fromisoformat("2026-01-05"), D("18"))
        rep = rolling_disposition(self._events(), ph, trading_calendar=CAL)
        self.assertEqual(rep.stale_max, 5)
        self.assertGreater(rep.n_obs, 0)

    def test_over_limit_dropped(self):
        """上限设 3：B 的价格（5 个交易日前）被剔除，该日无人可估 → 整日不可用。"""
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                  "2026-01-09", "2026-01-12"]:
            ph.add("A", date.fromisoformat(d), D("11"))
        ph.add("B", date.fromisoformat("2026-01-05"), D("18"))
        rep = rolling_disposition(self._events(), ph, trading_calendar=CAL,
                                  max_stale_days=3)
        self.assertEqual(rep.mode, "insufficient")
        self.assertEqual(rep.unpriced, 1, "被剔除的批次要计数，不能无声消失")

    def test_boundary_exactly_at_limit_is_kept(self):
        """恰好等于上限应该保留：">"而不是">="，边界要钉死。"""
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                  "2026-01-09", "2026-01-12"]:
            ph.add("A", date.fromisoformat(d), D("11"))
        ph.add("B", date.fromisoformat("2026-01-05"), D("18"))
        rep = rolling_disposition(self._events(), ph, trading_calendar=CAL,
                                  max_stale_days=5)
        self.assertEqual(rep.n_obs, 1)


class TestParity(unittest.TestCase):
    def test_zero_gain_neutral(self):
        """现价恰好等于成本：不盈不亏，不进任何一端（用 B 的浮亏撑起分母）。"""
        events = [ev("2026-01-05", "A", B, "1000", "11"),
                  ev("2026-01-12", "A", S, "500", "12"),
                  ev("2026-01-05", "B", B, "1000", "20")]
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]:
            dd = date.fromisoformat(d)
            ph.add("A", dd, D("11"))
            ph.add("B", dd, D("18"))
        rep = rolling_disposition(events, ph, trading_calendar=CAL)
        o = rep.obs[0]
        self.assertEqual(o.n_gain_open, 0, "A 剩余批次平价，不得计入浮盈")
        self.assertEqual(o.n_loss_open, 1)   # 只有 B 的浮亏


class TestOnlySellDaysCount(unittest.TestCase):
    def test_non_sell_days_produce_no_observation(self):
        events = [
            ev("2026-01-05", "A", B, "1000", "10"),
            ev("2026-01-12", "A", S, "1000", "12"),
            ev("2026-01-05", "B", B, "1000", "20"),
        ]
        ph = PriceHistory()
        for d in ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                  "2026-01-09", "2026-01-12", "2026-01-13"]:
            for s, px in (("A", "11"), ("B", "18")):
                ph.add(s, date.fromisoformat(d), D(px))
        rep = rolling_disposition(events, ph, trading_calendar=CAL)
        self.assertEqual(rep.n_obs, 1,
                         "只有卖出日构成观测；非卖出日的 PGR 恒为 0，掺进来只会稀释")


class TestDailyObs(unittest.TestCase):
    def test_de_property(self):
        o = DailyObs(dt=date(2026, 1, 5), n_gain_sold=2, n_gain_open=2,
                     n_loss_sold=1, n_loss_open=3)
        self.assertAlmostEqual(o.pgr, 0.5)
        self.assertAlmostEqual(o.plr, 0.25)
        self.assertAlmostEqual(o.de, 0.25)

    def test_missing_side_is_none(self):
        o = DailyObs(dt=date(2026, 1, 5), n_gain_sold=1, n_loss_sold=0,
                     n_gain_open=1, n_loss_open=0)
        self.assertIsNotNone(o.pgr)
        self.assertIsNone(o.plr)
        self.assertIsNone(o.de)


if __name__ == "__main__":
    unittest.main()
