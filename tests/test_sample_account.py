"""示例账户数据集的黄金测试。

这份测试同时验证两件事，所以它比普通单元测试更有价值：

1. **生成器没有跑偏**：记录数、成交笔数、费用合计、送股后的那笔盈利都被钉死。
   以后谁改了 `generate.py` 而忘了同步预期值，这里会立刻红。
2. **解析层真的能吃下两种真实格式**：同花顺风格（UTF-8 BOM + 包干手续费
   + 负数表示卖出）与另一家风格（GB18030 + 费用拆三列 + 正数表示卖出）。
   它等效于一份"真实交割单回归测试"，但数据可以公开进仓库。
"""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradeopsy.analytics.disposition import (  # noqa: E402
    compute_disposition,
    open_positions_from_ledger,
)
from tradeopsy.analytics.significance import bootstrap_disposition  # noqa: E402
from tradeopsy.ledger import PositionLedger  # noqa: E402
from tradeopsy.parsers import parse_file  # noqa: E402

D = Decimal
SAMPLE_DIR = ROOT / "examples" / "sample_account"
PRICES_CSV = ROOT / "examples" / "sample_account_prices.csv"
PART1 = SAMPLE_DIR / "part1_2025-11_2026-03_ths.csv"
PART2 = SAMPLE_DIR / "part2_2026-04_2026-10_ht.csv"


def _load_prices() -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for i, line in enumerate(PRICES_CSV.read_text(encoding="utf-8-sig").splitlines()):
        if i == 0 or not line.strip():
            continue
        sym, px = line.split(",")
        out[sym.strip()] = D(px.strip())
    return out


def _build() -> tuple[PositionLedger, list]:
    events = []
    warnings = []
    for p in (PART1, PART2):
        r = parse_file(p)
        events.extend(r.events)
        warnings.extend(r.warnings)
    ledger = PositionLedger()
    ledger.apply_all(events)
    return ledger, warnings


class TestFilesExist(unittest.TestCase):
    def test_files_present(self):
        for p in (PART1, PART2, PRICES_CSV):
            self.assertTrue(p.exists(), f"缺少示例数据文件 {p}")


class TestParserHandlesBothFormats(unittest.TestCase):
    """两种券商格式必须各自被正确识别 —— 这是数据集的核心卖点。"""

    def test_part1_utf8_bom_with_packed_fee(self):
        meta = parse_file(PART1).meta
        self.assertEqual(meta["encoding"], "utf-8-sig")
        self.assertEqual(meta["header_row"], 4)          # 抬头占 3 行
        self.assertEqual(meta["columns"]["fee"], "手续费")  # 包干列
        self.assertEqual(meta["fee_split_columns"], [])

    def test_part2_gb18030_with_split_fees(self):
        meta = parse_file(PART2).meta
        self.assertEqual(meta["encoding"], "gb18030")
        self.assertEqual(meta["header_row"], 4)
        self.assertIsNone(meta["columns"]["fee"])         # 没有包干列
        self.assertEqual(meta["fee_split_columns"], ["净佣金", "印花税", "过户费"])

    def test_part2_column_aliases_mapped(self):
        cols = parse_file(PART2).meta["columns"]
        self.assertEqual(cols["date"], "发生日期")
        self.assertEqual(cols["code"], "证券编码")
        self.assertEqual(cols["biz"], "摘要")
        self.assertEqual(cols["qty"], "成交股数")


class TestLedgerInvariants(unittest.TestCase):
    """账本层面的不变量。数值全部来自确定性的生成器，改动即为回归。"""

    def setUp(self) -> None:
        self.ledger, self.warnings = _build()

    def test_event_count(self):
        # 68 条原始记录，其中 2 条是无法识别的业务（担保品划转 / 利息归本）
        self.assertEqual(len(self.warnings), 2)
        self.assertEqual(len(self.ledger.chunks), 25)

    def test_unrecognized_rows_warn_but_do_not_abort(self):
        texts = " ".join(w.reason for w in self.warnings)
        self.assertIn("担保品划转", texts)
        self.assertIn("利息归本", texts)
        self.assertEqual(self.ledger.warnings, [])   # 账本层无需再告警

    def test_fee_totals(self):
        self.assertEqual(self.ledger.buy_fees, D("546.80"))
        self.assertEqual(self.ledger.sell_fees, D("1061.16"))

    def test_realized_pnl_and_dividend(self):
        self.assertEqual(self.ledger.realized_pnl(), D("23164.90"))
        self.assertEqual(self.ledger.total_dividend(), D("10500.00"))

    def test_split_makes_losing_trade_profitable(self):
        """五粮液：128 元买入、118 元卖出，含 10 送 3 后其实是赚的。

        这是最反直觉的一笔 —— 只看价格会觉得亏了 10 元/股，
        但股数从 1000 变成 1300、总成本不变，所以每股成本被摊薄到 98.49。
        任何用 `(卖价-买价)*股数` 简单相减的脚本都会在这里算错。
        """
        chunk = [c for c in self.ledger.chunks if c.symbol == "000858"][0]
        self.assertEqual(chunk.qty, D("1300"))
        self.assertEqual(chunk.pnl, D("25250.14"))
        self.assertTrue(chunk.is_win)

    def test_bond_t_plus_zero_same_day(self):
        """可转债 T+0：同一天买入并卖出，持有 0 天。"""
        bonds = [c for c in self.ledger.chunks if c.symbol == "113050"]
        self.assertEqual(len(bonds), 2)
        for c in bonds:
            self.assertEqual(c.hold_days, 0)


class TestDispositionOnSampleAccount(unittest.TestCase):
    """示例账户被设计成"卖盈持亏"，所以处置效应必须为正且显著。"""

    def setUp(self) -> None:
        ledger, _ = _build()
        opens = open_positions_from_ledger(ledger, _load_prices())
        self.rep = compute_disposition(ledger.chunks, opens)
        self.sig = bootstrap_disposition(ledger.chunks, opens)
        self.ledger = ledger

    def test_pgr_greater_than_plr(self):
        self.assertGreater(self.rep.pgr, self.rep.plr)
        self.assertGreater(self.rep.disposition_effect, 0.10)   # 明显处置效应
        self.assertIn("明显处置效应", self.rep.verdict)

    def test_winners_held_shorter_than_losers(self):
        """处置效应的典型指纹：赢家拿得短、输家拿得长。"""
        self.assertLess(self.rep.avg_hold_win, self.rep.avg_hold_loss)

    def test_count_based_effect_also_positive(self):
        self.assertGreater(self.sig.de_count, 0.0)
        self.assertAlmostEqual(self.sig.de_count, self.sig.pgr_count - self.sig.plr_count)

    def test_effect_is_statistically_significant(self):
        self.assertIsNotNone(self.sig.p_value_boot)
        self.assertLess(self.sig.p_value_boot, 0.05)
        self.assertGreater(self.sig.ci_low, 0.0)


if __name__ == "__main__":
    unittest.main()
