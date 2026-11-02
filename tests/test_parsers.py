"""解析层测试。

重点覆盖四种"真实文件里一定出现、而简单实现一定出错"的情况：
  1. GBK 编码
  2. 表头不在第一行
  3. 卖出数量是负数
  4. 费用拆成多列，需要累加且不能重复计
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from tradeopsy.__main__ import _dedupe
from tradeopsy.load import dedupe as dedupe_tagged
from tradeopsy.model import Event, EventType
from tradeopsy.parsers import detect, parse_file, to_date, to_decimal

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _write_tmp(content: str, encoding: str, suffix: str = ".csv") -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    )
    tmp.write(content.encode(encoding))
    tmp.close()
    return Path(tmp.name)


class TestCleaners(unittest.TestCase):
    def test_to_decimal_variants(self):
        cases = {
            "1,234.56": "1234.56",
            "¥1,234.56": "1234.56",
            "(500.00)": "-500.00",
            "--": "0",
            "": "0",
            " 10.5 ": "10.5",
            "+3.00": "3.00",
        }
        for raw, want in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(to_decimal(raw), Decimal(want))

    def test_to_date_variants(self):
        want = date(2026, 10, 30)
        for raw in ("2026-10-30", "2026/10/30", "2026.10.30", "20261030",
                    "2026-10-30 09:31:25", "2026-10-30T09:31:25"):
            with self.subTest(raw=raw):
                self.assertEqual(to_date(raw), want)

    def test_to_date_invalid(self):
        self.assertIsNone(to_date("2026-13-45"))
        self.assertIsNone(to_date(""))


class TestGenericCsv(unittest.TestCase):
    """fixture 是手写假数据，编码故意存成 GBK、前面故意加两行抬头。"""

    @classmethod
    def setUpClass(cls):
        cls.path = FIXTURES / "sample_ths_gbk.csv"
        cls.result = parse_file(cls.path)

    def test_detected_and_decoded(self):
        self.assertEqual(self.result.meta.get("encoding"), "gb18030")
        self.assertEqual(self.result.meta.get("header_row"), 4)

    def test_header_offset_does_not_break(self):
        # 12 行数据，其中"担保品划转"无法识别 → 11 条事件 + 1 条告警
        self.assertEqual(len(self.result.events), 11)
        self.assertEqual(len(self.result.warnings), 1)

    def test_negative_qty_becomes_sell(self):
        sells = [e for e in self.result.events if e.etype is EventType.SELL]
        self.assertTrue(sells)
        for e in sells:
            self.assertGreater(e.qty, 0, "数量应取绝对值")
        pa = [e for e in sells if e.symbol == "000001"][0]
        self.assertEqual(pa.qty, Decimal("2000"))
        self.assertEqual(pa.price, Decimal("11.60"))

    def test_split_and_dividend(self):
        split = [e for e in self.result.events if e.etype is EventType.SPLIT]
        self.assertEqual(len(split), 1)
        self.assertEqual(split[0].qty, Decimal("50"))
        div = [e for e in self.result.events if e.etype is EventType.DIVIDEND]
        self.assertEqual(len(div), 1)
        self.assertEqual(div[0].cash, Decimal("420.00"))

    def test_unknown_business_warns_instead_of_dropping_silently(self):
        self.assertIn("担保品划转", str(self.result.warnings[0]))


class TestFeeColumns(unittest.TestCase):
    def test_split_fee_columns_are_summed(self):
        content = (
            "成交日期,证券代码,证券名称,业务名称,成交数量,成交价格,净佣金,印花税,过户费\n"
            "2026-03-02,600000,浦发银行,证券买入,1000,10.00,5.00,0,0.02\n"
            "2026-03-05,600000,浦发银行,证券卖出,-1000,10.50,5.00,10.50,0.02\n"
        )
        p = _write_tmp(content, "utf-8")
        r = parse_file(p)
        self.assertEqual(len(r.events), 2)
        buy, sell = r.events
        self.assertEqual(buy.fee, Decimal("5.02"))
        self.assertEqual(sell.fee, Decimal("15.52"))

    def test_single_fee_column_takes_priority(self):
        """同时存在"手续费"和分项列时，只取手续费，不能重复累加。"""
        content = (
            "成交日期,证券代码,证券名称,业务名称,成交数量,成交价格,手续费,佣金,印花税\n"
            "2026-03-02,600000,浦发银行,证券买入,1000,10.00,5.50,5.00,0.50\n"
        )
        p = _write_tmp(content, "utf-8")
        r = parse_file(p)
        self.assertEqual(r.events[0].fee, Decimal("5.50"))


class TestDedupe(unittest.TestCase):
    def test_overlapping_files_are_deduped(self):
        """按月导出时相邻文件常有一天重叠，去重做错会让盈亏翻倍。"""
        e = Event(dt=date(2026, 10, 30), symbol="600519",
                  etype=EventType.BUY, qty=Decimal("100"),
                  price=Decimal("1680"), fee=Decimal("50.40"))
        same = Event(dt=date(2026, 10, 30), symbol="600519",
                     etype=EventType.BUY, qty=Decimal("100"),
                     price=Decimal("1680"), fee=Decimal("50.40"))
        different = Event(dt=date(2026, 10, 30), symbol="600519",
                          etype=EventType.BUY, qty=Decimal("200"),
                          price=Decimal("1680"), fee=Decimal("50.40"))
        out, dup = _dedupe([e, same, different])
        self.assertEqual(len(out), 2)
        self.assertEqual(dup, 1)

    def test_same_file_double_identical_buy_is_kept(self):
        """回归：同一天两笔完全相同的合法买入（拆单/同价补单）不得被去重吞掉。

        旧实现把扁平列表里所有完全相同的记录都当重复，
        同文件里的第二笔会被静默丢弃，盈亏直接少算。
        """
        e1 = Event(dt=date(2026, 10, 30), symbol="510300",
                   etype=EventType.BUY, qty=Decimal("100"),
                   price=Decimal("4.00"), fee=Decimal("5.00"))
        e2 = Event(dt=date(2026, 10, 30), symbol="510300",
                   etype=EventType.BUY, qty=Decimal("100"),
                   price=Decimal("4.00"), fee=Decimal("5.00"))
        # 旧语义（不带来源）：仍去重，兼容老调用方
        out, dup = _dedupe([e1, e2])
        self.assertEqual((len(out), dup), (1, 1))
        # 新语义（带文件来源）：同一文件内的两笔都保留
        out2, dup2 = dedupe_tagged([e1, e2], sources=[0, 0])
        self.assertEqual((len(out2), dup2), (2, 0))

    def test_cross_file_overlap_deduped_with_sources(self):
        """带来源去重后，跨文件的重叠记录仍要去掉。"""
        e = Event(dt=date(2026, 10, 30), symbol="600519",
                  etype=EventType.BUY, qty=Decimal("100"),
                  price=Decimal("1680"), fee=Decimal("50.40"))
        same = Event(dt=date(2026, 10, 30), symbol="600519",
                     etype=EventType.BUY, qty=Decimal("100"),
                     price=Decimal("1680"), fee=Decimal("50.40"))
        other = Event(dt=date(2026, 10, 31), symbol="600519",
                      etype=EventType.SELL, qty=Decimal("100"),
                      price=Decimal("1700"), fee=Decimal("51.00"))
        # e/same 来自两个文件（重叠），other 只在第二个文件
        out, dup = dedupe_tagged([e, other, same], sources=[0, 1, 1])
        self.assertEqual(len(out), 2)
        self.assertEqual(dup, 1)


class TestDividendTaxRow(unittest.TestCase):
    """回归：红利税补扣行金额是负数，旧实现 abs() 会把它变成正的分红收入。

    持股不满一年卖出时，券商补扣红利税，交割单上会出现
    「股息红利差异税扣税」这类业务名，发生金额为负。
    """

    def test_tax_row_keeps_negative_sign(self):
        content = (
            "成交日期,证券代码,证券名称,业务名称,成交数量,成交价格,成交金额\n"
            "2026-03-02,600000,浦发银行,股息红利,0,0.00,300.00\n"
            "2026-04-10,600000,浦发银行,股息红利差异税扣税,0,0.00,-45.00\n"
        )
        p = _write_tmp(content, "utf-8")
        r = parse_file(p)
        self.assertEqual(len(r.events), 2)
        normal, tax = r.events
        self.assertEqual(normal.cash, Decimal("300.00"))
        # 税行：cash 保留负号（冲减分红），fee 记绝对值留痕
        self.assertEqual(tax.cash, Decimal("-45.00"))
        self.assertEqual(tax.fee, Decimal("45.00"))

    def test_tax_row_reduces_total_pnl(self):
        """端到端：税行进账本后，总收益必须减少，而不是倒赚。"""
        from tradeopsy.ledger import PositionLedger
        content = (
            "成交日期,证券代码,证券名称,业务名称,成交数量,成交价格,成交金额\n"
            "2026-03-02,600000,浦发银行,股息入账,0,0.00,100.00\n"
            "2026-04-10,600000,浦发银行,红利税补扣,0,0.00,-20.00\n"
        )
        p = _write_tmp(content, "utf-8")
        r = parse_file(p)
        led = PositionLedger()
        led.apply_all(r.events)
        self.assertEqual(led.total_dividend(), Decimal("80.00"))


class TestDetection(unittest.TestCase):
    def test_non_csv_returns_no_parser(self):
        p = _write_tmp("hello world", "utf-8", suffix=".json")
        parser, score = detect(p)
        self.assertIsNone(parser)
        self.assertEqual(score, 0.0)

    def test_corrupt_file_does_not_raise(self):
        p = _write_tmp("abc,def\n1,2\n", "utf-8")
        r = parse_file(p)
        self.assertEqual(len(r.events), 0)
        self.assertTrue(r.warnings)


if __name__ == "__main__":
    unittest.main()
