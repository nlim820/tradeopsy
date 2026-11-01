"""paper 模块的单元测试。

原则和 P0 的黄金样本一致：**不依赖网络、不依赖系统时间、手算预期值**。
所有涉及"今天"的地方都通过参数注入，否则测试会在某一天自动失效。

这里重点盯三类 bug，它们都有共同特征 —— 不报错、不告警、净值曲线看起来正常：
  1. 空仓时不清仓（账户退化成买入持有）
  2. T+1 被绕过（当日买入当日卖出）
  3. 碎单反复调仓（被最低佣金吃穿）
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from tradeopsy.paper import broker, runner, strategy
from tradeopsy.paper.broker import (
    Account,
    Position,
    commission,
    execute_targets,
    roll_day,
)
from tradeopsy.paper.market import Bar


def D(x) -> Decimal:
    return Decimal(str(x))


def mkbar(d: str, o, c, h=None, l=None, adj: bool = True) -> Bar:
    """造一根日线。high/low 默认向外张开 0.01，避免意外造出 high==low 的一字板。"""
    o, c = D(o), D(c)
    hi = D(h) if h is not None else max(o, c) + D("0.01")
    lo = D(l) if l is not None else min(o, c) - D("0.01")
    return Bar(dt=date.fromisoformat(d), open=o, close=c, high=hi, low=lo, adjusted=adj)


NAMES = {"510300": "沪深300ETF", "510500": "中证500ETF"}


class TestCommission(unittest.TestCase):
    def test_minimum_five_yuan(self):
        # 5000 元成交额，按万2.5 算只有 1.25 元，但券商最低收 5 元。
        # 这 4 倍差距就是"小额频繁交易必亏"的数学根源。
        self.assertEqual(commission(D("5000")), D("5.00"))

    def test_normal_rate(self):
        self.assertEqual(commission(D("100000")), D("25.00"))

    def test_zero(self):
        self.assertEqual(commission(D("0")), D("0.00"))


class TestExecuteTargets(unittest.TestCase):
    def _acc(self, cash="100000", qty="0", price="4.0") -> Account:
        acc = Account(cash=D(cash), initial_cash=D("100000"))
        if qty != "0":
            acc.positions["510300"] = Position(
                symbol="510300",
                name="沪深300ETF",
                qty=D(qty),
                available=D(qty),
                last_price=D(price),
            )
        return acc

    def test_flat_targets_liquidates(self):
        """回归测试：目标为空 = 空仓，必须清掉所有持仓。

        这个 bug 曾经真实存在过 —— execute_targets 只遍历 targets，
        于是空仓信号下持仓永不卖出。三天后回看明细才发现，
        账户已经默默变成"买入并持有"。
        """
        acc = self._acc(qty="1000")
        bars = {"510300": mkbar("2026-10-30", "4.00", "4.10")}
        fills = execute_targets(acc, {}, bars, NAMES, date(2026, 10, 30))

        self.assertNotIn("510300", acc.positions)
        # 4000 元卖出，扣 5 元最低佣金
        self.assertEqual(acc.cash, D("103995.00"))
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].side, "sell")
        self.assertEqual(fills[0].qty, D("1000"))

    def test_buy_respects_lot_size_and_t1(self):
        acc = self._acc()
        bars = {"510300": mkbar("2026-10-30", "4.00", "4.05")}
        execute_targets(acc, {"510300": D("0.5")}, bars, NAMES, date(2026, 10, 30))

        pos = acc.positions["510300"]
        # 50000 / 4.00 = 12500 股，正好 125 手
        self.assertEqual(pos.qty, D("12500"))
        # T+1：当日买入不可卖
        self.assertEqual(pos.available, D("0"))
        self.assertEqual(acc.cash, D("49987.50"))  # 100000 - 50000 - 12.5

    def test_limit_up_board_skips_buy(self):
        """一字板：开盘=最高=最低，撮合必须跳过。"""
        acc = self._acc()
        bars = {"510300": mkbar("2026-10-30", "4.40", "4.40", h="4.40", l="4.40")}
        fills = execute_targets(acc, {"510300": D("0.5")}, bars, NAMES, date(2026, 10, 30))

        self.assertNotIn("510300", acc.positions)
        self.assertEqual(acc.cash, D("100000"))
        self.assertTrue(any(not f.ok and "一字板" in f.reason for f in fills))

    def test_small_gap_does_not_trade(self):
        """差额小于 3% 总资产不调仓，避免碎单被最低佣金吃掉。

        构造时悠着点：总资产 = 现金 + 持仓市值，两者都要算进去，
        否则一不留神就造出一个差几十个百分点的场景，测了个寂寞。
        """
        # 总资产 100021，目标 50010，持仓 48521，差额 1489 < 3000
        acc = self._acc(cash="51500", qty="12100", price="4.01")
        bars = {"510300": mkbar("2026-10-30", "4.01", "4.05")}
        fills = execute_targets(acc, {"510300": D("0.5")}, bars, NAMES, date(2026, 10, 30))
        self.assertEqual([f for f in fills if f.ok], [])

    def test_suspended_symbol_kept(self):
        """取不到当日行情（停牌）时，持仓不能因为"没有价格"就被清掉。"""
        acc = self._acc(qty="1000")
        fills = execute_targets(acc, {}, {}, NAMES, date(2026, 10, 30))
        self.assertIn("510300", acc.positions)
        self.assertEqual(fills, [])

    def test_sell_then_buy_releases_cash(self):
        """换仓日必须能靠卖出回款完成买入，而不是等第二天。"""
        acc = self._acc(qty="12000", price="4.00")
        acc.positions["510300"].available = D("12000")
        acc.cash = D("100.00")  # 现金几乎为零，只能靠卖出换标的
        bars = {
            "510300": mkbar("2026-10-30", "4.00", "4.00"),
            "510500": mkbar("2026-10-30", "6.00", "6.00"),
        }
        fills = execute_targets(
            acc, {"510500": D("0.5")}, bars, NAMES, date(2026, 10, 30)
        )
        sides = [f.side for f in fills if f.ok]
        self.assertEqual(sides, ["sell", "buy"])
        self.assertIn("510500", acc.positions)


class TestRollDay(unittest.TestCase):
    def test_unlocks_previous_day_holdings(self):
        acc = Account(cash=D("0"), initial_cash=D("100000"))
        acc.positions["510300"] = Position(
            symbol="510300", name="x", qty=D("1000"), available=D("0")
        )
        roll_day(acc)
        self.assertEqual(acc.positions["510300"].available, D("1000"))


def _trend(symbol: str, days: int, start: float, step: float) -> list[Bar]:
    """造一段单调走势的日线。step 为正=上涨，为负=下跌。"""
    out: list[Bar] = []
    d = date(2026, 1, 1)
    px = start
    for _ in range(days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        nxt = px + step
        out.append(
            Bar(dt=d, open=D(round(px, 3)), close=D(round(nxt, 3)),
                high=D(round(max(px, nxt) + 0.01, 3)),
                low=D(round(min(px, nxt) - 0.01, 3)), volume=D("1000"))
        )
        px = nxt
        d += timedelta(days=1)
    return out


class TestStrategy(unittest.TestCase):
    def test_insufficient_history_returns_empty(self):
        hist = {s: _trend(s, 10, 4.0, 0.01) for s, _ in strategy.UNIVERSE}
        d = strategy.decide(hist)
        self.assertTrue(d.is_flat)
        self.assertTrue(all("历史不足" in c.reason for c in d.candidates))

    def test_picks_strongest_uptrend(self):
        cfg = strategy.StrategyConfig()
        hist = {}
        for sym, _ in strategy.UNIVERSE:
            hist[sym] = _trend(sym, 120, 4.0, -0.002)  # 默认全部下跌
        hist["510300"] = _trend("510300", 120, 4.0, 0.01)  # 只有它上涨
        d = strategy.decide(hist, cfg)

        self.assertIn("510300", d.targets)
        self.assertEqual(d.targets["510300"], D("0.5"))  # 只命中一只，只给一半仓位
        passed = [c for c in d.candidates if c.passed]
        self.assertEqual([c.symbol for c in passed], ["510300"])

    def test_downtrend_all_filtered(self):
        hist = {s: _trend(s, 120, 6.0, -0.01) for s, _ in strategy.UNIVERSE}
        d = strategy.decide(hist)
        self.assertTrue(d.is_flat)


class TestRunnerIdempotency(unittest.TestCase):
    """日更任务最重要的性质：跑两次不能算两遍。"""

    def _fake_bars(self) -> dict[str, list[Bar]]:
        hist = {}
        for i, (sym, _) in enumerate(strategy.UNIVERSE):
            # 一半涨一半跌，保证有信号也有落选，路径覆盖更全
            step = 0.01 if i % 2 == 0 else -0.004
            hist[sym] = _trend(sym, 140, 4.0, step)
        return hist

    def test_second_run_is_noop(self):
        fake = self._fake_bars()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runner.market, "fetch_many", return_value=fake):
                first = runner.run(root, today=date(2026, 8, 1), verbose=False)
                second = runner.run(root, today=date(2026, 8, 1), verbose=False)

            self.assertEqual(first["new_days"], 1)
            self.assertEqual(second["new_days"], 0)
            self.assertEqual(first["total"], second["total"])

            state = json.loads((root / "paper" / "state.json").read_text("utf-8"))
            self.assertEqual(state["as_of"], first["as_of"])

    def test_catch_up_replays_every_trading_day(self):
        """伪造一个落后的 as_of，系统必须把中间所有交易日补齐。"""
        fake = self._fake_bars()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "paper").mkdir()
            (root / "paper" / "state.json").write_text(
                json.dumps({
                    "version": 1, "as_of": "2026-04-30", "cash": "100000.00",
                    "initial_cash": "100000.00", "positions": {},
                    "pending_from": "", "pending_targets": {},
                }),
                encoding="utf-8",
            )
            with patch.object(runner.market, "fetch_many", return_value=fake):
                summary = runner.run(root, today=date(2026, 8, 1), verbose=False)

            self.assertGreater(summary["new_days"], 40)
            nav_lines = (root / "paper" / "nav.csv").read_text("utf-8-sig").strip().splitlines()
            # 表头 + 每个交易日一行
            self.assertEqual(len(nav_lines) - 1, summary["new_days"])


if __name__ == "__main__":
    unittest.main()
