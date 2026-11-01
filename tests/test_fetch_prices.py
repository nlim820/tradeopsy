"""tools/fetch_prices.py 的测试。

不联网：所有数据源都被替换成假函数，测的是**分类规则、降级逻辑、落盘格式**。
真接口的连通性验证是一次性的（见脚本头部注释里的实测记录），不进单元测试 ——
否则测试会因为别人家服务器抖动而红，那种红没有信息量。
"""

from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "fetch_prices", ROOT / "tools" / "fetch_prices.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


class TestMarket(unittest.TestCase):
    """交易所前缀。错一个就会拉到别的标的，或者直接 404。"""

    def test_shanghai_main_and_star(self):
        self.assertEqual(fp.market_of("600519"), "sh")
        self.assertEqual(fp.market_of("601318"), "sh")
        self.assertEqual(fp.market_of("688111"), "sh")     # 科创板
        self.assertEqual(fp.market_of("900901"), "sh")     # 沪市 B 股

    def test_shenzhen(self):
        self.assertEqual(fp.market_of("000001"), "sz")
        self.assertEqual(fp.market_of("002594"), "sz")
        self.assertEqual(fp.market_of("300750"), "sz")     # 创业板
        self.assertEqual(fp.market_of("200011"), "sz")     # 深市 B 股

    def test_funds(self):
        self.assertEqual(fp.market_of("510300"), "sh")
        self.assertEqual(fp.market_of("588000"), "sh")     # 科创 50 ETF
        self.assertEqual(fp.market_of("159915"), "sz")

    def test_bonds(self):
        self.assertEqual(fp.market_of("113050"), "sh")     # 沪主板转债
        self.assertEqual(fp.market_of("118077"), "sh")     # 科创板转债 118 段
        self.assertEqual(fp.market_of("123290"), "sz")     # 创业板转债
        self.assertEqual(fp.market_of("127099"), "sz")     # 深主板转债

    def test_beijing(self):
        self.assertEqual(fp.market_of("430047"), "bj")
        self.assertEqual(fp.market_of("830799"), "bj")


class TestKind(unittest.TestCase):
    def test_bond(self):
        self.assertEqual(fp.kind_of("113050"), "bond")
        self.assertEqual(fp.kind_of("123290"), "bond")

    def test_etf(self):
        self.assertEqual(fp.kind_of("510300"), "etf")
        self.assertEqual(fp.kind_of("159915"), "etf")
        self.assertEqual(fp.kind_of("588000"), "etf")

    def test_stock(self):
        self.assertEqual(fp.kind_of("600519"), "stock")
        self.assertEqual(fp.kind_of("000001"), "stock")
        self.assertEqual(fp.kind_of("300750"), "stock")

    def test_every_kind_has_a_source(self):
        """分类结果必须能在 SOURCE_CHAIN 里找到落点，否则该标的永远拉不到。"""
        for code in ("600519", "510300", "113050"):
            self.assertIn(fp.kind_of(code), fp.SOURCE_CHAIN)


class TestDates(unittest.TestCase):
    """行情接口的日期列类型五花八门，这正是最容易静默出错的地方。"""

    def test_iso_string(self):
        self.assertEqual(fp._to_date("2026-10-30"), date(2026, 10, 30))

    def test_slash_and_compact(self):
        self.assertEqual(fp._to_date("2026/10/30"), date(2026, 10, 30))
        self.assertEqual(fp._to_date("20261030"), date(2026, 10, 30))

    def test_datetime_and_date(self):
        from datetime import datetime
        self.assertEqual(fp._to_date(datetime(2026, 10, 30, 15, 0)), date(2026, 10, 30))
        self.assertEqual(fp._to_date(date(2026, 10, 30)), date(2026, 10, 30))

    def test_timestamp_like(self):
        class FakeTs:
            def date(self):
                return date(2026, 10, 30)
        self.assertEqual(fp._to_date(FakeTs()), date(2026, 10, 30))

    def test_garbage(self):
        self.assertIsNone(fp._to_date(None))
        self.assertIsNone(fp._to_date(""))
        self.assertIsNone(fp._to_date("不是日期"))


class TestPrices(unittest.TestCase):
    def test_bytes_from_sina(self):
        """新浪部分接口返回 bytes，不处理会炸在 Decimal 上。"""
        self.assertEqual(fp._to_dec(b"3.856"), Decimal("3.8560"))

    def test_string_and_float(self):
        self.assertEqual(fp._to_dec("10.20"), Decimal("10.2000"))
        self.assertEqual(fp._to_dec(100.0), Decimal("100.0000"))

    def test_keeps_bond_precision(self):
        """可转债 100.001 这种价位，提前截断到 2 位会失真。"""
        self.assertEqual(fp._to_dec("100.001"), Decimal("100.0010"))

    def test_rejects_non_positive_and_garbage(self):
        for bad in (0, -1, "0.00", "abc", None, ""):
            self.assertIsNone(fp._to_dec(bad), f"{bad!r} 应被判无效")


class TestCache(unittest.TestCase):
    def test_roundtrip(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "x.csv"
            rows = [(date(2026, 10, 29), Decimal("10.20")),
                    (date(2026, 10, 30), Decimal("10.35"))]
            fp._write_cache(p, rows)
            back = fp._read_cache(p)
            # 缓存往返要保证数值相等（Decimal 的显示位数会变，值不能变）
            self.assertEqual([(d, v) for d, v in back],
                             [(d, v.quantize(Decimal("0.0001"))) for d, v in rows])

    def test_missing_returns_none(self):
        with TemporaryDirectory() as td:
            self.assertIsNone(fp._read_cache(Path(td) / "nope.csv"))


class TestFallback(unittest.TestCase):
    """降级链：前一个源挂了要能换下一个，全挂了要给出人话原因。"""

    def setUp(self):
        self._orig = fp.SOURCE_CHAIN
        self.addCleanup(lambda: setattr(fp, "SOURCE_CHAIN", self._orig))

    def _run(self, chain, cache_dir=None):
        fp.SOURCE_CHAIN = chain
        return fp.fetch_one("600519", date(2026, 10, 1), date(2026, 10, 30),
                            "", cache_dir, pause=0.0)

    def test_first_source_raises_falls_back(self):
        def boom(code, s, e, adj):
            raise ConnectionError("网络炸了")

        def ok(code, s, e, adj):
            return [(date(2026, 10, 30), Decimal("1400.00"))], 0, 1

        rows, src, _ = self._run({"stock": [("坏源", boom), ("好源", ok)]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(src, "好源")

    def test_all_sources_empty_says_unlisted(self):
        def empty(code, s, e, adj):
            return [], 0, 0

        rows, src, why = self._run({"stock": [("A", empty), ("B", empty)]})
        self.assertEqual(rows, [])
        self.assertIn("尚未上市", why)

    def test_keyerror_means_no_data_not_crash(self):
        """akshare 对未上市/已退市标的抛 KeyError('date')，要归类成「无数据」。"""
        def keyerr(code, s, e, adj):
            raise KeyError("date")

        rows, _, why = self._run({"stock": [("新浪", keyerr)]})
        self.assertEqual(rows, [])
        self.assertIn("无数据", why)

    def test_data_outside_range_is_distinguished(self):
        """有 978 条历史但全在区间外 —— 必须和「接口坏了」区分开。"""
        def outside(code, s, e, adj):
            return [], 0, 978

        _, _, why = self._run({"stock": [("源", outside)]})
        self.assertIn("978", why)
        self.assertIn("之外", why)


class TestOutput(unittest.TestCase):
    def test_format_matches_contract(self):
        """落盘格式必须能被 tradeopsy 的 read_price_history 直接吃。"""
        with TemporaryDirectory() as td:
            out = Path(td) / "p.csv"
            n = fp.write_output([
                (date(2026, 10, 30), "600519", Decimal("1400.5")),
                (date(2026, 10, 29), "600519", Decimal("1390")),
                (date(2026, 10, 30), "113050", Decimal("100.001")),
            ], out)
            self.assertEqual(n, 3)

            raw = out.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "必须带 UTF-8 BOM")

            text = raw.decode("utf-8-sig")
            lines = text.replace("\r\n", "\n").strip().split("\n")
            self.assertEqual(lines[0], "日期,证券代码,收盘价")
            # 按 (代码, 日期) 排序：代码是字符串，113050 排在 600519 之前
            self.assertEqual(lines[1], "2026-10-30,113050,100.001")
            self.assertEqual(lines[2], "2026-10-29,600519,1390.00")
            self.assertEqual(lines[3], "2026-10-30,600519,1400.50")

    def test_output_really_loads_in_tradeopsy(self):
        """端到端：写出来的文件必须能被 prices.read_price_history 解析。"""
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from tradeopsy.prices import read_price_history

        with TemporaryDirectory() as td:
            out = Path(td) / "p.csv"
            fp.write_output([
                (date(2026, 10, 29), "600519", Decimal("1390")),
                (date(2026, 10, 30), "600519", Decimal("1400.5")),
            ], out)
            ph = read_price_history(str(out))
            self.assertEqual(ph.symbols, {"600519"})
            # on() 返回 (价格, 该价格的日期) —— 第二个值是陈旧价检测的依据
            price, px_date = ph.on("600519", date(2026, 10, 30))
            self.assertEqual(price, Decimal("1400.50"))
            self.assertEqual(px_date, date(2026, 10, 30))


class TestFromLedger(unittest.TestCase):
    def test_infers_symbols_and_span(self):
        syms, first, last = fp.symbols_from_ledger(ROOT / "examples" / "sample_account")
        self.assertIn("600519", syms)
        self.assertIn("113050", syms)
        self.assertLess(first, last)
        self.assertEqual(len(syms), 14)

    def test_dry_run_does_not_network(self):
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            rc = fp.main(["--from-ledger", str(ROOT / "examples" / "sample_account"),
                          "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("将请求 14 只", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
