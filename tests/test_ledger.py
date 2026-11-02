"""黄金样本测试（golden fixtures）。

每个场景的交易都是手写的，**预期值全部由人工按 A 股实际规则手算**，
引擎跑出来必须对上。对不上就是 bug，不需要任何真实交割单来判断对错。

这是本项目「先做技术验证」的具体形态：
    正确性不靠主观感觉，靠可判定的数字。

A 股费用口径（2026 年现行，作为手算依据）：
    佣金      双边收取，通常万 2.5，单笔最低 5 元 —— 测试里直接给定金额
    印花税    仅卖出单边，成交额 0.05%（2023-08-28 起由 0.1% 减半）
    过户费    双边，成交额 0.001%（2022-04-29 起沪深统一）
    可转债    免印花税，仅收佣金
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.ledger import PositionLedger, summarize  # noqa: E402
from tradeopsy.model import Event, EventType, money  # noqa: E402

D = Decimal


def buy(d: date, sym: str, qty: str, px: str, fee: str = "0") -> Event:
    return Event(dt=d, symbol=sym, etype=EventType.BUY, qty=D(qty), price=D(px), fee=D(fee))


def sell(d: date, sym: str, qty: str, px: str, fee: str = "0") -> Event:
    return Event(dt=d, symbol=sym, etype=EventType.SELL, qty=D(qty), price=D(px), fee=D(fee))


def split(d: date, sym: str, add_qty: str) -> Event:
    return Event(dt=d, symbol=sym, etype=EventType.SPLIT, qty=D(add_qty))


def dividend(d: date, sym: str, cash: str, tax: str = "0") -> Event:
    return Event(dt=d, symbol=sym, etype=EventType.DIVIDEND, cash=D(cash), fee=D(tax))


def rights(d: date, sym: str, qty: str, px: str) -> Event:
    return Event(dt=d, symbol=sym, etype=EventType.RIGHTS, qty=D(qty), price=D(px))


class TestBasicRoundTrip(unittest.TestCase):
    """F1 最朴素的买入卖出，验证费用归属方向正确。"""

    def test_simple_roundtrip(self):
        led = PositionLedger()
        led.apply_all(
            [
                # 买 1000 股 @10.00：佣金 5.00 + 过户费 0.10 = 5.10，计入成本
                buy(date(2026, 1, 5), "600519", "1000", "10.00", "5.10"),
                # 卖 1000 股 @12.00：佣金 5.00 + 印花税 6.00 + 过户费 0.12 = 11.12
                sell(date(2026, 1, 15), "600519", "1000", "12.00", "11.12"),
            ]
        )
        self.assertEqual(len(led.chunks), 1)
        c = led.chunks[0]
        self.assertEqual(c.cost, money("10005.10"))
        self.assertEqual(c.proceeds, money("11988.88"))
        self.assertEqual(c.pnl, money("1983.78"))
        self.assertEqual(c.hold_days, 10)
        self.assertEqual(summarize(led)["胜率"], 1.0)


class TestSplitKeepsCost(unittest.TestCase):
    """F2 10 转 5：股数变 1.5 倍，总成本必须分毫不动。"""

    def test_split_then_sell(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 1, 5), "600519", "1000", "10.00", "5.10"),
                split(date(2026, 1, 6), "600519", "500"),  # 10 转 5
                sell(date(2026, 1, 15), "600519", "1500", "8.00", "11.12"),
            ]
        )
        c = led.chunks[0]
        self.assertEqual(c.qty, D("1500"))
        self.assertEqual(c.cost, money("10005.10"))  # 与 F1 完全一致：送转不改变总成本
        self.assertEqual(c.proceeds, money("11988.88"))
        self.assertEqual(c.pnl, money("1983.78"))


class TestDividendSeparate(unittest.TestCase):
    """F3 分红单独记账，不冲减持仓成本。"""

    def test_dividend_not_in_cost(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 1, 5), "600519", "1000", "10.00", "5.10"),
                dividend(date(2026, 1, 6), "600519", "300.00"),  # 每 10 股派 3 元
                sell(date(2026, 1, 15), "600519", "1000", "10.00", "10.10"),
            ]
        )
        c = led.chunks[0]
        self.assertEqual(c.cost, money("10005.10"))
        self.assertEqual(c.pnl, money("-15.20"))  # 价差是亏的
        self.assertEqual(led.total_dividend(), money("300.00"))
        self.assertEqual(led.total_pnl(), money("284.80"))  # 加上分红才是真收益


class TestFifoAcrossLots(unittest.TestCase):
    """F4 两次建仓后卖 600 股，必须先进先出，且成本/费用之和守恒。"""

    def test_fifo_split_sell(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 2, 2), "000001", "500", "10.00", "5.05"),
                buy(date(2026, 2, 9), "000001", "500", "12.00", "5.06"),
                # 卖 600 @15：佣金 5 + 印花 4.50 + 过户 0.09 = 9.59
                sell(date(2026, 2, 20), "000001", "600", "15.00", "9.59"),
            ]
        )
        self.assertEqual(len(led.chunks), 2)
        first, second = led.chunks
        self.assertEqual(first.qty, D("500"))  # 先消耗 2 月 2 日那批
        self.assertEqual(second.qty, D("100"))  # 再从 2 月 9 日那批取 100

        # 成本守恒：两个批次转移的成本之和 == 两笔买入总成本中已卖出的部分
        self.assertEqual(first.cost, money("5005.05"))
        self.assertEqual(second.cost, money("1201.01"))
        self.assertEqual(first.cost + second.cost, money("6206.06"))

        # 费用守恒：分摊后必须精确等于实收 9.59
        self.assertEqual(first.sell_fee + second.sell_fee, money("9.59"))

        # 剩余持仓：400 股，成本 6005.06 - 1201.01 = 4804.05
        pos = led.open_positions()
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].qty, D("400"))
        self.assertEqual(pos[0].cost, money("4804.05"))


class TestRightsIssue(unittest.TestCase):
    """F5 配股：掏钱换新股，总成本要增加。"""

    def test_rights_increases_cost(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 3, 2), "600000", "1000", "10.00", "5.10"),
                rights(date(2026, 3, 10), "600000", "200", "6.00"),  # 10 配 2，配股价 6
            ]
        )
        lots = led._lots["600000"]
        total_qty = sum(l.qty for l in lots)
        total_cost = sum(l.cost for l in lots)
        self.assertEqual(total_qty, D("1200"))
        # 10005.10 + 200×6.00 = 11205.10
        self.assertEqual(total_cost, money("11205.10"))

        led.apply(sell(date(2026, 3, 20), "600000", "1200", "9.34", "10.71"))
        # 卖出 1200 股后成本全额转出，账上不留残值
        self.assertEqual(led.open_positions(), [])
        self.assertEqual(sum(c.cost for c in led.chunks), money("11205.10"))
        # 11197.29 - 11205.10 = -7.81
        self.assertEqual(sum(c.pnl for c in led.chunks), money("-7.81"))


class TestSameDayRoundTrip(unittest.TestCase):
    """F6 可转债 T+0：同一天买、同一天卖，持有 0 天，且免印花税。"""

    def test_same_day(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 4, 1), "113050", "100", "100.00", "5.00"),
                sell(date(2026, 4, 1), "113050", "100", "101.00", "5.00"),
            ]
        )
        c = led.chunks[0]
        self.assertEqual(c.hold_days, 0)
        self.assertEqual(c.pnl, money("90.00"))  # 10000 差价 − 10 元双向佣金


class TestPartialClose(unittest.TestCase):
    """F7 部分平仓：卖 400 留 600。"""

    def test_partial(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 5, 6), "300750", "1000", "10.00", "5.10"),
                # 卖 400 @11：佣金 5 + 印花 2.20 + 过户 0.04 = 7.24
                sell(date(2026, 5, 8), "300750", "400", "11.00", "7.24"),
            ]
        )
        c = led.chunks[0]
        self.assertEqual(c.cost, money("4002.04"))  # 按 400/1000 比例
        self.assertEqual(c.proceeds, money("4392.76"))
        self.assertEqual(c.pnl, money("390.72"))
        pos = led.open_positions()[0]
        self.assertEqual(pos.qty, D("600"))
        # 10005.10 − 4002.04 = 6003.06，且 4002.04 + 6003.06 == 10005.10（成本守恒）
        self.assertEqual(pos.cost, money("6003.06"))
        self.assertEqual(c.cost + pos.cost, money("10005.10"))


class TestCrossYear(unittest.TestCase):
    """F8 跨年持仓：持有天数按自然日算。"""

    def test_cross_year(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2025, 12, 20), "601318", "1000", "10.00", "5.10"),
                sell(date(2026, 1, 20), "601318", "1000", "10.50", "10.36"),
            ]
        )
        c = led.chunks[0]
        self.assertEqual(c.hold_days, 31)
        self.assertEqual(c.proceeds, money("10489.64"))
        self.assertEqual(c.pnl, money("484.54"))


class TestSplitAllocation(unittest.TestCase):
    """F11 送转股数分配：不整除时总股数必须精确、且不许出现小数股。

    回归背景：旧实现按 ratio = new_total/total_qty 逐批相乘再舍入，
    非整除时每批漂移 ±0.00005 股且产生小数批次。
    """

    def test_split_allocation_is_exact_and_integer(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 7, 1), "600000", "7", "10.00"),
                buy(date(2026, 7, 2), "600000", "100", "10.00"),
                # 总持仓 107，10 送 3 应得 32.1 -> 实际到账 32 股
                split(date(2026, 7, 3), "600000", "32"),
            ]
        )
        lots = led._lots["600000"]
        total = sum(l.qty for l in lots)
        self.assertEqual(total, D("139"))          # 107 + 32，精确
        for l in lots:
            self.assertEqual(l.qty % 1, D("0"))    # 每批都是整数股
        # 成本一分不动
        self.assertEqual(sum(l.cost for l in lots), money("1070.00"))

    def test_split_zero_qty_is_rejected(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 7, 1), "600000", "100", "10.00"),
                split(date(2026, 7, 2), "600000", "0"),
            ]
        )
        lots = led._lots["600000"]
        self.assertEqual(sum(l.qty for l in lots), D("100"))
        self.assertTrue(led.warnings)


class TestSellZeroQty(unittest.TestCase):
    """F12 卖出数量为 0：整笔跳过，费用不得记到不存在的卖出上。"""

    def test_zero_qty_sell_skipped(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 7, 1), "600000", "100", "10.00", "5.00"),
                sell(date(2026, 7, 2), "600000", "0", "10.00", "3.00"),
            ]
        )
        self.assertEqual(led.chunks, [])
        self.assertEqual(led.sell_fees, D("0"))
        self.assertTrue(led.warnings)


class TestRobustness(unittest.TestCase):
    """F9/F10 脏数据容错：宁可警告也不要炸掉整批。"""

    def test_sell_without_position_warns(self):
        led = PositionLedger()
        led.apply(sell(date(2026, 6, 1), "000002", "100", "10.00", "5.00"))
        self.assertEqual(led.chunks, [])
        self.assertTrue(led.warnings)

    def test_oversell_is_capped(self):
        led = PositionLedger()
        led.apply_all(
            [
                buy(date(2026, 6, 1), "000002", "100", "10.00", "5.00"),
                sell(date(2026, 6, 2), "000002", "500", "11.00", "5.00"),
            ]
        )
        self.assertEqual(led.chunks[0].qty, D("100"))  # 按持仓上限成交
        self.assertEqual(led.open_positions(), [])

    def test_unknown_business_is_kept_as_warning(self):
        led = PositionLedger()
        led.apply(
            Event(
                dt=date(2026, 6, 3),
                symbol="000002",
                etype=EventType.OTHER,
                note="担保品划转",
            )
        )
        self.assertTrue(any("担保品划转" in w for w in led.warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
