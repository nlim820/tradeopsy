"""显著性检验测试（计数口径 bootstrap）。

不联网、固定 seed，结果可复现。
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.analytics.disposition import OpenPositionMV  # noqa: E402
from tradeopsy.analytics.significance import (  # noqa: E402
    MIN_OBS,
    bootstrap_disposition,
)
from tradeopsy.model import RealizedChunk, money  # noqa: E402

D = Decimal


def chunk(symbol: str, pnl: str) -> RealizedChunk:
    c = D("1000")
    return RealizedChunk(
        symbol=symbol,
        open_dt=date(2026, 1, 5),
        close_dt=date(2026, 2, 5),
        qty=D("100"),
        cost=c,
        proceeds=money(c + D(pnl)),
    )


def gain(symbol: str = "G") -> RealizedChunk:
    return chunk(symbol, "100.00")


def loss(symbol: str = "L") -> RealizedChunk:
    return chunk(symbol, "-100.00")


def pg(symbol: str = "PG") -> OpenPositionMV:
    return OpenPositionMV(symbol, D("1000"), D("1200"))


def pl(symbol: str = "PL") -> OpenPositionMV:
    return OpenPositionMV(symbol, D("1000"), D("800"))


class TestCounts(unittest.TestCase):
    def test_point_estimate(self):
        chunks = [gain("A"), gain("B"), gain("C"), loss("X")]
        opens = [pg("P1"), pl("P2")]
        r = bootstrap_disposition(chunks, opens, n_boot=200)
        self.assertEqual(r.n_gain_sold, 3)
        self.assertEqual(r.n_gain_held, 1)
        self.assertEqual(r.n_loss_sold, 1)
        self.assertEqual(r.n_loss_held, 1)
        self.assertAlmostEqual(r.pgr_count, 0.75)   # 3 / 4
        self.assertAlmostEqual(r.plr_count, 0.5)    # 1 / 2
        self.assertAlmostEqual(r.de_count, 0.25)    # 0.75 - 0.50（Odean 口径 PGR-PLR）


class TestInsufficient(unittest.TestCase):
    def test_small_sample_is_insufficient(self):
        # 每侧只有 2 个机会，低于 MIN_OBS
        chunks = [gain("A"), gain("B"), loss("X"), loss("Y")]
        opens = []
        r = bootstrap_disposition(chunks, opens, n_boot=200)
        self.assertFalse(r.sufficient)
        self.assertIsNone(r.p_value_boot)
        self.assertIsNone(r.ci_low)
        self.assertIn("证据不足", r.verdict)

    def test_missing_loss_side(self):
        chunks = [gain("A"), gain("B"), gain("C"), gain("D"), gain("E")]
        opens = [pg("P1"), pg("P2")]  # 没有亏损机会
        r = bootstrap_disposition(chunks, opens, n_boot=200)
        self.assertIsNone(r.de_count)
        self.assertIn("证据不足", r.verdict)


class TestSignificant(unittest.TestCase):
    """构造清晰的处置效应：卖盈持亏 -> PGR=0.80 > PLR=0.45，DE=+0.35，样本足够大。"""

    def setUp(self) -> None:
        self.chunks = [gain(f"G{i}") for i in range(32)] + [loss(f"L{i}") for i in range(18)]
        self.opens = [pg(f"PG{i}") for i in range(8)] + [pl(f"PL{i}") for i in range(22)]

    def test_counts(self):
        r = bootstrap_disposition(self.chunks, self.opens, n_boot=1000, seed=7)
        self.assertEqual(r.n_gain_sold, 32)
        self.assertEqual(r.n_gain_held, 8)
        self.assertEqual(r.n_loss_sold, 18)
        self.assertEqual(r.n_loss_held, 22)
        self.assertAlmostEqual(r.pgr_count, 32 / 40)
        self.assertAlmostEqual(r.plr_count, 18 / 40)
        self.assertAlmostEqual(r.de_count, 0.35)

    def test_significant(self):
        r = bootstrap_disposition(self.chunks, self.opens, n_boot=2000, seed=7)
        self.assertGreater(r.ci_low, 0.0)           # 95% CI 不含 0
        self.assertLess(r.p_value_boot, 0.05)
        self.assertIn("显著", r.verdict)
        self.assertIn("处置效应", r.verdict)

    def test_z_test_agrees(self):
        r = bootstrap_disposition(self.chunks, self.opens, n_boot=1000, seed=7)
        self.assertIsNotNone(r.p_value_z)
        self.assertLess(r.p_value_z, 0.05)


class TestReproducible(unittest.TestCase):
    def test_same_seed_same_ci(self):
        chunks = [gain(f"G{i}") for i in range(32)] + [loss(f"L{i}") for i in range(18)]
        opens = [pg(f"PG{i}") for i in range(8)] + [pl(f"PL{i}") for i in range(22)]
        a = bootstrap_disposition(chunks, opens, n_boot=500, seed=123)
        b = bootstrap_disposition(chunks, opens, n_boot=500, seed=123)
        self.assertEqual(a.ci_low, b.ci_low)
        self.assertEqual(a.ci_high, b.ci_high)
        self.assertEqual(a.p_value_boot, b.p_value_boot)


class TestConstants(unittest.TestCase):
    def test_min_obs_exposed(self):
        self.assertGreaterEqual(MIN_OBS, 3)


if __name__ == "__main__":
    unittest.main()
