"""交易日历测试。

重点钉死两件事，因为它们在真实数据里最容易静默出错：

1. **区间语义是半开 (start, end]**：建仓日当天不算持有。
   写错成闭区间，所有当日买当日卖（T+0 品种）会变成持有 1 天，
   而不是 0 天 —— 这会让"可转债 T+0 交易者持有期 1 天"这种假结论出现。
2. **周末不误报覆盖缺失**：周五买、周日平仓是合法跨期，
   如果把它算作"日历不完整"，用户会被无意义的提示淹没。

参照日历固定用 2026 年 1 月：1/5 周一，1/11 周日。
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.tradingcal import TradingCalendar, weekdays_between  # noqa: E402


def d(m: int, day: int) -> date:
    return date(2026, m, day)


class TestWeekdaysBetween(unittest.TestCase):
    def test_one_week(self):
        # 1/5 周一 ~ 1/11 周日 -> 5 个工作日
        self.assertEqual(len(weekdays_between(d(1, 5), d(1, 11))), 5)

    def test_single_weekend_day(self):
        self.assertEqual(weekdays_between(d(1, 10), d(1, 10)), [])  # 周六

    def test_reversed_range_is_empty(self):
        self.assertEqual(weekdays_between(d(1, 11), d(1, 5)), [])


class TestFromDates(unittest.TestCase):
    def test_dedup_and_sort(self):
        cal = TradingCalendar.from_dates([d(1, 7), d(1, 5), d(1, 7), d(1, 6)])
        self.assertEqual(cal.days, (d(1, 5), d(1, 6), d(1, 7)))
        self.assertEqual(cal.source, "explicit")

    def test_empty(self):
        cal = TradingCalendar.from_dates([])
        self.assertIsNone(cal.span)
        self.assertEqual(len(cal), 0)


class TestCount(unittest.TestCase):
    def setUp(self):
        # 1/5~1/9 五个交易日
        self.cal = TradingCalendar.from_dates([d(1, x) for x in (5, 6, 7, 8, 9)])

    def test_same_day_is_zero(self):
        """当日买当日卖 = 持有 0 个交易日（半开区间语义）。"""
        self.assertEqual(self.cal.count(d(1, 5), d(1, 5)), 0)

    def test_one_day_hold(self):
        self.assertEqual(self.cal.count(d(1, 5), d(1, 6)), 1)

    def test_cross_weekend(self):
        # 1/5 建仓 -> 1/9 平仓：中间是 1/6,7,8,9 共 4 个交易日
        self.assertEqual(self.cal.count(d(1, 5), d(1, 9)), 4)

    def test_close_before_open(self):
        self.assertEqual(self.cal.count(d(1, 9), d(1, 5)), 0)

    def test_is_trading_day(self):
        self.assertTrue(self.cal.is_trading_day(d(1, 5)))
        self.assertFalse(self.cal.is_trading_day(d(1, 10)))  # 周六


class TestCovers(unittest.TestCase):
    def setUp(self):
        self.cal = TradingCalendar.from_dates([d(1, x) for x in (5, 6, 7, 8, 9)])

    def test_inside(self):
        self.assertTrue(self.cal.covers(d(1, 5), d(1, 9)))

    def test_weekend_tail_is_tolerated(self):
        """1/9(五) 之后到 1/11(日) 没有工作日 -> 不算覆盖缺失。"""
        self.assertTrue(self.cal.covers(d(1, 5), d(1, 11)))

    def test_weekday_tail_is_not_covered(self):
        """1/12(一) 是工作日但日历里没有 -> 真的缺数据。"""
        self.assertFalse(self.cal.covers(d(1, 5), d(1, 12)))

    def test_before_start_not_covered(self):
        self.assertFalse(self.cal.covers(d(1, 2), d(1, 9)))

    def test_empty_calendar_never_covers(self):
        self.assertFalse(TradingCalendar.from_dates([]).covers(d(1, 5), d(1, 6)))


class TestWeekdayApprox(unittest.TestCase):
    def test_excludes_weekend(self):
        cal = TradingCalendar.weekday_approx(d(1, 5), d(1, 11))
        self.assertEqual(len(cal), 5)
        self.assertEqual(cal.source, "weekday_approx")
        self.assertNotIn(d(1, 10), cal.days)

    def test_holiday_removed(self):
        # 假设 1/7 休市
        cal = TradingCalendar.weekday_approx(d(1, 5), d(1, 9), holidays=[d(1, 7)])
        self.assertNotIn(d(1, 7), cal.days)
        self.assertEqual(len(cal), 4)
        self.assertEqual(cal.count(d(1, 5), d(1, 9)), 3)

    def test_describe_mentions_source(self):
        cal = TradingCalendar.weekday_approx(d(1, 5), d(1, 9))
        self.assertIn("工作日近似", cal.describe())
        self.assertIn("5", cal.describe())


class TestAverageHoldIntegration(unittest.TestCase):
    """台账汇总层：同样的数据，交易日口径必须比自然日口径小。"""

    def _ledger(self):
        from tradeopsy.ledger import PositionLedger
        from tradeopsy.model import Event, EventType

        lg = PositionLedger()
        lg.apply_all([
            # 1/5(一) 买，1/9(五) 卖 -> 自然日 4 天，交易日 4 天（1/6,7,8,9）
            Event(d(1, 5), "A", EventType.BUY, "100", "10", "0"),
            Event(d(1, 9), "A", EventType.SELL, "100", "12", "0"),
            # 1/5(一) 买，1/11(日) 挂到下一周才卖不了，用 1/12(一) -> 自然日 7 天
            Event(d(1, 5), "B", EventType.BUY, "100", "10", "0"),
            Event(d(1, 12), "B", EventType.SELL, "100", "8", "0"),
        ])
        return lg

    def test_natural_vs_trading_labels(self):
        from tradeopsy.ledger import summarize

        lg = self._ledger()
        nat = summarize(lg)
        self.assertIn("平均持有自然日", nat)

        cal = TradingCalendar.weekday_approx(d(1, 1), d(1, 31))
        tra = summarize(lg, cal)
        self.assertIn("平均持有交易日", tra)
        # 自然日：(4 + 7) / 2 = 5.5；交易日：(4 + 5) / 2 = 4.5
        self.assertAlmostEqual(nat["平均持有自然日"], 5.5)
        self.assertAlmostEqual(tra["平均持有交易日"], 4.5)
        self.assertLess(tra["平均持有交易日"], nat["平均持有自然日"])

    def test_disposition_accepts_calendar_object(self):
        from tradeopsy.analytics.disposition import compute_disposition

        lg = self._ledger()
        cal = TradingCalendar.weekday_approx(d(1, 1), d(1, 31))
        rep = compute_disposition(lg.chunks, trading_calendar=cal)
        self.assertTrue(rep.hold_days_trading)
        # 赢家 1 笔（A 赚），交易日 4 天；自然日对照 4 天
        self.assertAlmostEqual(rep.avg_hold_win, 4.0)
        self.assertAlmostEqual(rep.avg_hold_win_calendar, 4.0)
        # 输家 1 笔（B 亏），交易日 5 天；自然日 7 天 -> 口径差 2 天
        self.assertAlmostEqual(rep.avg_hold_loss, 5.0)
        self.assertAlmostEqual(rep.avg_hold_loss_calendar, 7.0)

    def test_short_calendar_triggers_warning(self):
        from tradeopsy.analytics.disposition import compute_disposition

        lg = self._ledger()
        # 日历只到 1/6，后面全缺
        cal = TradingCalendar.from_dates([d(1, 5), d(1, 6)])
        rep = compute_disposition(lg.chunks, trading_calendar=cal)
        self.assertTrue(any("超出交易日历覆盖范围" in n for n in rep.notes))


if __name__ == "__main__":
    unittest.main()
