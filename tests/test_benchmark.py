"""同行基准对照的测试。

重点不是「数字算得对」（那是 bootstrap 的事，已在 test_significance 覆盖），
而是三件容易被做错的事：
  1. 口径拦截 —— 期末快照口径绝不允许与文献对照；
  2. 区间包含的三种判读方向不能搞反；
  3. 比值口径在 PLR 极小时必须承认不稳定，而不是输出一个吓人的大数。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.analytics.benchmark import (
    BenchmarkReport,
    compare_to_benchmarks,
    load_benchmarks,
)


class TestCaliberGate(unittest.TestCase):
    """口径硬拦截：这是本模块存在的第一理由。"""

    def test_snapshot_refuses(self):
        r = compare_to_benchmarks(
            caliber="snapshot", my_de=0.16, ci_low=0.08, ci_high=0.24
        )
        self.assertFalse(r.comparable)
        self.assertIn("期末快照", r.blocked_reason)
        self.assertEqual(r.comparisons, [])

    def test_none_refuses(self):
        r = compare_to_benchmarks(caliber="none")
        self.assertFalse(r.comparable)
        self.assertIn("--price-history", r.blocked_reason)

    def test_insufficient_refuses(self):
        r = compare_to_benchmarks(caliber="insufficient")
        self.assertFalse(r.comparable)
        self.assertIn("凑齐", r.blocked_reason)

    def test_rolling_allows(self):
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.16, ci_low=0.08, ci_high=0.24
        )
        self.assertTrue(r.comparable)
        self.assertTrue(r.comparisons)


class TestIntervalJudgement(unittest.TestCase):
    """区间包含检验的三种方向 —— 判反了比不算更糟。"""

    B = None

    @classmethod
    def setUpClass(cls):
        benchmarks, meta = load_benchmarks()
        cls.B, cls.META = benchmarks, meta

    def _run(self, ci_low, ci_high, my_de=0.16):
        return compare_to_benchmarks(
            caliber="rolling", my_de=my_de, ci_low=ci_low, ci_high=ci_high,
            my_n_obs=24, benchmarks=self.B, meta=self.META,
        )

    def test_inside_means_indistinguishable(self):
        """文献值 5pp 落在区间内 → 不可分辨（不是「一致」）。"""
        r = self._run(0.01, 0.20)
        c = next(c for c in r.de_comparisons)
        self.assertEqual(c.verdict, "indistinguishable")
        self.assertIn("不足以分辨", c.text)
        # 文本必须显式否定「等于人群」这个误读，防止读者把区间包含当成相同
        self.assertIn("不是", c.text)

    def test_below_means_stronger(self):
        """文献值低于区间下界 → 我比人群强。"""
        r = self._run(0.08, 0.24)
        c = next(c for c in r.de_comparisons)
        self.assertEqual(c.verdict, "stronger")

    def test_above_means_weaker(self):
        """文献值高于区间上界 → 我比人群弱。"""
        r = self._run(-0.10, 0.02)
        c = next(c for c in r.de_comparisons)
        self.assertEqual(c.verdict, "weaker")

    def test_missing_ci_is_unavailable(self):
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.16, benchmarks=self.B, meta=self.META
        )
        c = next(c for c in r.de_comparisons)
        self.assertEqual(c.verdict, "unavailable")


class TestRatioCaliber(unittest.TestCase):
    """比值口径只做描述，且必须在 PLR 过小时自我否定。"""

    @classmethod
    def setUpClass(cls):
        benchmarks, meta = load_benchmarks()
        cls.B, cls.META = benchmarks, meta

    def test_small_plr_flagged_unstable(self):
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.16, ci_low=0.08, ci_high=0.24,
            my_pgr=0.1923, my_plr=0.0314,   # PLR 3.1% < 5%
            benchmarks=self.B, meta=self.META,
        )
        for c in r.ratio_comparisons:
            self.assertEqual(c.verdict, "unstable")
            self.assertIn("分母过小", c.text)

    def test_zero_plr_unavailable(self):
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.16, ci_low=0.08, ci_high=0.24,
            my_pgr=0.5, my_plr=0.0,
            benchmarks=self.B, meta=self.META,
        )
        c = next(c for c in r.ratio_comparisons)
        self.assertEqual(c.verdict, "unavailable")

    def test_healthy_plr_is_descriptive(self):
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.05, ci_low=0.01, ci_high=0.10,
            my_pgr=0.30, my_plr=0.25,
            benchmarks=self.B, meta=self.META,
        )
        c = next(c for c in r.ratio_comparisons)
        self.assertEqual(c.verdict, "descriptive")


class TestPowerWarning(unittest.TestCase):
    """统计功效提示：区间太宽时必须直说。"""

    def test_wide_ci_warns(self):
        # 区间宽 0.40，文献效应量 0.05 → 8 倍，远超 4 倍阈值
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.2, ci_low=0.0, ci_high=0.40,
            my_n_obs=24,
        )
        self.assertGreaterEqual(r.power_ratio, 4.0)
        self.assertTrue(any("分辨力" in n or "必然" in n for n in r.notes))

    def test_narrow_ci_no_warning(self):
        r = compare_to_benchmarks(
            caliber="rolling", my_de=0.06, ci_low=0.03, ci_high=0.09,
            my_n_obs=500,
        )
        self.assertLess(r.power_ratio, 4.0)
        self.assertFalse(any("必然结果" in n for n in r.notes))


class TestDataset(unittest.TestCase):
    """quantile_stance 必须随 report 带出去（这是「不做分位数」的声明）。"""

    def test_quantile_stance_propagated(self):
        r = compare_to_benchmarks(caliber="rolling", my_de=0.16,
                                  ci_low=0.08, ci_high=0.24)
        self.assertFalse(r.meta.get("quantile_stance", {}).get("supported", True))
        self.assertTrue(any("不输出百分位排名" in n for n in r.notes))

    def test_sample_size_note(self):
        r = compare_to_benchmarks(caliber="rolling", my_de=0.16,
                                  ci_low=0.08, ci_high=0.24, my_n_obs=24)
        self.assertTrue(any("卖出日" in n for n in r.notes))


if __name__ == "__main__":
    unittest.main()
