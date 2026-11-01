"""逐日价格序列（prices.py）测试。

钉死的四个点，全是真实数据里最容易翻车的：

1. **缺价不伪造**：`on()` 在无覆盖时返回 None，绝不拿"最近的价格"悄悄顶上 ——
   填充决策必须留给调用方，且要带上"这价格是多少天前的"这层信息。
2. **宽表拒绝**：日期为行、代码为列的宽表必须被明确拒绝，
   而不是猜一个可能错位的列映射 —— 错位会静默算错，比报错危险得多。
3. **编码回退**：券商导出十有八九是 GBK 系，utf-8 解不开要能兜住。
4. **脏行跳过不中断**：行情文件里混进"停牌 --"或空行是常态，
   一行坏数据不该让整份价格序列读不出来。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.prices import (  # noqa: E402
    PriceHistory,
    looks_like_wide,
    read_price_history,
)

D = Decimal


def _write(text: str, encoding: str = "utf-8") -> str:
    """写临时文件返回路径。测试文件用完即弃，放在系统临时目录。"""
    fh = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", encoding=encoding, delete=False, newline=""
    )
    fh.write(text)
    fh.close()
    return fh.name


class TestParse(unittest.TestCase):
    def test_with_header(self):
        p = _write("日期,证券代码,收盘价\n2026-01-05,000001,10.20\n2026-01-06,000001,10.35\n")
        ph = read_price_history(p)
        self.assertFalse(ph.problem)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph.n_points, 2)
        px, pd_ = ph.on("000001", date(2026, 1, 6))
        self.assertEqual(px, D("10.35"))
        self.assertEqual(pd_, date(2026, 1, 6))

    def test_header_aliases(self):
        p = _write("交易日期,证券代码,收盘价(元)\n2026-01-05,600519,1680.00\n")
        ph = read_price_history(p)
        self.assertFalse(ph.problem)
        self.assertEqual(ph.on("600519", date(2026, 1, 5))[0], D("1680.00"))

    def test_no_header(self):
        p = _write("2026-01-05,000001,10.20\n")
        ph = read_price_history(p)
        self.assertFalse(ph.problem)
        self.assertEqual(ph.on("000001", date(2026, 1, 5))[0], D("10.20"))

    def test_compact_date_and_slash(self):
        p = _write("20260105,000001,10.20\n2026/01/06,000001,10.35\n")
        ph = read_price_history(p)
        self.assertEqual(ph.n_points, 2)

    def test_gb18030(self):
        p = _write("日期,证券代码,收盘价\n2026-01-05,600519,1680.00\n", "gb18030")
        ph = read_price_history(p)
        self.assertFalse(ph.problem)

    def test_wide_rejected(self):
        p = _write("日期,600519,000001\n2026-01-05,1680.00,10.20\n")
        ph = read_price_history(p)
        self.assertTrue(ph.problem)
        self.assertIn("宽表", ph.problem)

    def test_missing_column(self):
        p = _write("日期,证券代码\n2026-01-05,000001\n")
        ph = read_price_history(p)
        self.assertTrue(ph.problem)
        self.assertIn("收盘价", ph.problem)

    def test_bad_rows_skipped(self):
        p = _write(
            "日期,证券代码,收盘价\n"
            "2026-01-05,000001,10.20\n"
            "2026-01-06,000001,--\n"      # 停牌占位
            "2026-01-07,000001,0\n"        # 无效价格
            "2026-01-08,,10.5\n"           # 缺代码
            "not-a-date,000001,10.5\n"     # 坏日期
        )
        ph = read_price_history(p)
        self.assertFalse(ph.problem)
        self.assertEqual(ph.n_points, 1)
        self.assertEqual(ph.bad_rows, 4)

    def test_duplicate_overwritten(self):
        p = _write(
            "日期,证券代码,收盘价\n"
            "2026-01-05,000001,10.20\n"
            "2026-01-05,000001,10.30\n"
        )
        ph = read_price_history(p)
        self.assertEqual(ph.dups, 1)
        self.assertEqual(ph.on("000001", date(2026, 1, 5))[0], D("10.30"))

    def test_missing_file(self):
        ph = read_price_history("Z:/definitely/not/here.csv")
        self.assertTrue(ph.problem)
        self.assertFalse(ph)


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.ph = PriceHistory()
        for d, px in [("2026-01-05", "10"), ("2026-01-08", "11")]:
            self.ph.add("A", date.fromisoformat(d), D(px))

    def test_exact_hit(self):
        px, pd_ = self.ph.on("A", date(2026, 1, 8))
        self.assertEqual((px, pd_), (D("11"), date(2026, 1, 8)))

    def test_carry_forward_reports_origin(self):
        """缺价日拿到的是"最近的旧价"，且必须能看出它多旧 —— 这是填充决策的依据。"""
        px, pd_ = self.ph.on("A", date(2026, 1, 9))
        self.assertEqual((px, pd_), (D("11"), date(2026, 1, 8)))

    def test_before_first_point_is_none(self):
        """序列起点之前没有价格可言 —— 绝不能用第一个价格倒灌回更早的日子。"""
        self.assertEqual(self.ph.on("A", date(2026, 1, 4)), (None, None))

    def test_unknown_symbol(self):
        self.assertEqual(self.ph.on("NOPE", date(2026, 1, 8)), (None, None))

    def test_snapshot_uses_last(self):
        self.assertEqual(self.ph.snapshot(), {"A": D("11")})

    def test_span(self):
        self.assertEqual(self.ph.span, (date(2026, 1, 5), date(2026, 1, 8)))

    def test_out_of_order_add(self):
        ph = PriceHistory()
        ph.add("A", date(2026, 1, 8), D("11"))
        ph.add("A", date(2026, 1, 5), D("10"))
        # 乱序插入后 bisect 查询必须仍正确 —— 这是 add() 有序性维护的回归测试
        self.assertEqual(ph.on("A", date(2026, 1, 6)), (D("10"), date(2026, 1, 5)))


class TestWideDetector(unittest.TestCase):
    def test_wide(self):
        self.assertTrue(looks_like_wide(["日期", "600519", "000001", "510300"]))

    def test_long(self):
        self.assertFalse(looks_like_wide(["日期", "证券代码", "收盘价"]))

    def test_too_short_to_judge(self):
        self.assertFalse(looks_like_wide(["日期", "600519"]))


if __name__ == "__main__":
    unittest.main()
