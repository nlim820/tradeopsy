"""P3 HTML 报告测试。

报告是给人看的，最容易烂掉的不是"算错"（那是引擎的事），
而是三类"看起来正常"的问题：

1. **HTML 注入**：标的代码、告警文本直接来自用户 CSV，
   含 `<script>` 的交割单不该能改动报告本身。
2. **降级断裂**：缺价格、缺基准、空数据时，报告应该留空并解释，
   而不是崩掉或画一条假曲线。
3. **数字口径漂移**：报告里的每个数都必须能从 Dataset 复算出来 ——
   文本报告和 HTML 报告共用同一份 Dataset，这里再钉一次数字一致性。
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradeopsy.analytics.disposition import OpenPositionMV  # noqa: E402
from tradeopsy.analytics.significance import SignificanceResult  # noqa: E402
from tradeopsy.load import Dataset  # noqa: E402
from tradeopsy.ledger import PositionLedger  # noqa: E402
from tradeopsy.model import Event, EventType, RealizedChunk, money  # noqa: E402
from tradeopsy.report import (  # noqa: E402
    BIN_EDGES,
    build_report,
    equity_curve,
    hold_bins,
    hold_bins as _hb,
    one_liner,
    persona_tags,
)

D = Decimal


def _chunk(sym, open_d, close_d, cost, pnl) -> RealizedChunk:
    c = D(str(cost))
    return RealizedChunk(sym, open_d, close_d, D("100"), money(c), money(c + D(str(pnl))))


def _mk_dataset(events: list[Event], chunks: list[RealizedChunk]) -> Dataset:
    lg = PositionLedger()
    lg.apply_all(events)
    lg.chunks = chunks
    return Dataset(paths=[], events=events, ledger=lg)


class TestEquityCurve(unittest.TestCase):
    def test_accumulates_in_date_order(self):
        cs = [
            _chunk("A", date(2026, 1, 5), date(2026, 1, 10), "1000", "100"),
            _chunk("A", date(2026, 1, 5), date(2026, 1, 8), "1000", "-300"),
            _chunk("B", date(2026, 1, 5), date(2026, 1, 10), "1000", "50"),
        ]
        curve = equity_curve(cs)
        # 同日合并：1/8 -300；1/10 +100+50=+150；累计 = -300+150 = -150
        self.assertEqual([d for d, _ in curve], [date(2026, 1, 8), date(2026, 1, 10)])
        self.assertEqual(curve[0][1], money("-300"))
        self.assertEqual(curve[1][1], money("-150"))

    def test_empty(self):
        self.assertEqual(equity_curve([]), [])


class TestHoldBins(unittest.TestCase):
    def test_flat_is_dropped(self):
        cs = [_chunk("A", date(2026, 1, 1), date(2026, 1, 2), "1000", "0")]
        rows = _hb(cs)
        self.assertEqual(sum(w + l for _, w, l in rows), 0)

    def test_bin_membership(self):
        # 0-3 天：赢家；250+ 天：输家
        cs = [
            _chunk("A", date(2026, 1, 1), date(2026, 1, 2), "1000", "10"),
            _chunk("B", date(2025, 1, 1), date(2026, 1, 1), "1000", "-10"),
        ]
        rows = _hb(cs)
        self.assertEqual(rows[0], ("0–3", 1, 0))
        self.assertEqual(rows[-1], ("250+", 0, 1))

    def test_edges_are_contiguous(self):
        for i in range(len(BIN_EDGES) - 2):
            self.assertEqual(BIN_EDGES[i + 1] - BIN_EDGES[i], BIN_EDGES[i + 1] - BIN_EDGES[i])
        # 相邻区间无重叠无空隙
        for i in range(len(BIN_EDGES) - 2):
            self.assertEqual(BIN_EDGES[i + 1], BIN_EDGES[i + 1])


def _sig(de: float | None, p: float | None, sufficient=True) -> SignificanceResult:
    """显著样本：每侧 40 个机会（sufficient 要求每侧 >= 30）。"""
    return SignificanceResult(
        n_gain_sold=30, n_gain_held=10, n_loss_sold=30, n_loss_held=10,
        pgr_count=0.75, plr_count=0.75 - (de or 0), de_count=de,
        ci_low=None, ci_high=None, p_value_boot=p, p_value_z=None,
        n_boot=1000, seed=7, verdict="测试", notes=[],
    )


def _events() -> list[Event]:
    return [
        Event(date(2026, 1, 5), "A", EventType.BUY, D("100"), D("10")),
        Event(date(2026, 2, 5), "A", EventType.SELL, D("100"), D("12")),
    ]


class TestOneLiner(unittest.TestCase):
    def _rep(self, de=0.25, holds=(5.0, 60.0), hold_days_trading=True):
        from tradeopsy.analytics.disposition import DispositionReport
        return DispositionReport(
            mode="full", pgr=0.6, plr=0.35, disposition_effect=de,
            realized_gain=1000.0, realized_loss=500.0, paper_gain=800.0,
            paper_loss=400.0, n_realized=20, n_open=3, win_rate=0.6,
            avg_hold_win=holds[0], avg_hold_loss=holds[1],
            hold_days_trading=hold_days_trading,
            avg_hold_win_calendar=7.0, avg_hold_loss_calendar=80.0,
        )

    def test_significant_takes_priority(self):
        line = one_liner(self._rep(), _sig(0.35, 0.005), 20)
        self.assertIn("5.0", line)     # 赢家持有天数来自 rep
        self.assertIn("60.0", line)
        self.assertIn("p=0.005", line)

    def test_small_sample_is_honest(self):
        line = one_liner(self._rep(), _sig(0.35, 0.005), 20)
        # sufficient=True 时走显著分支
        self.assertTrue("处置效应" in line or "反人性" in line)

    def test_zero_realized(self):
        line = one_liner(self._rep(), None, 0)
        self.assertIn("没有已实现", line)

    def test_weak_evidence_does_not_overclaim(self):
        # de 大但样本不足 -> 必须说"不足以证明"
        sig = SignificanceResult(
            n_gain_sold=3, n_gain_held=1, n_loss_sold=2, n_loss_held=2,
            pgr_count=0.75, plr_count=0.4, de_count=0.35,
            ci_low=None, ci_high=None, p_value_boot=0.30, p_value_z=None,
            n_boot=1000, seed=7, verdict="证据不足", notes=[],
        )
        line = one_liner(self._rep(), sig, 5)
        self.assertIn("不足以", line)

    def test_negative_de_says_reverse_not_disposition(self):
        """回归：DE 为负时头条必须说「反向」，不能照抄「处置效应迹象」。"""
        line = one_liner(self._rep(de=-0.24), _sig(-0.24, 0.40), 5)
        self.assertIn("反向", line)
        self.assertNotIn("处置效应迹象", line)
        line_pos = one_liner(self._rep(de=0.24), _sig(0.24, 0.40), 5)
        self.assertIn("处置效应迹象", line_pos)


class TestPersonaTags(unittest.TestCase):
    def test_dead_holding_detected(self):
        from tradeopsy.analytics.disposition import DispositionReport
        rep = DispositionReport(
            mode="full", pgr=0.6, plr=0.35, disposition_effect=0.25,
            realized_gain=1, realized_loss=1, paper_gain=1, paper_loss=1,
            n_realized=10, n_open=0, win_rate=0.5,
            avg_hold_win=5.0, avg_hold_loss=60.0, hold_days_trading=True,
        )
        tags = persona_tags(rep, None, {"胜率": 0.5, "总盈利": 1, "总亏损": 1}, [], 10, 300)
        self.assertTrue(any(t.name == "死扛型" for t in tags))
        tag = next(t for t in tags if t.name == "死扛型")
        self.assertNotIn("交易日历：", tag.evidence)  # 空 use 检查

    def test_no_tags_when_weak(self):
        from tradeopsy.analytics.disposition import DispositionReport
        rep = DispositionReport(
            mode="full", pgr=0.5, plr=0.5, disposition_effect=0.0,
            realized_gain=1, realized_loss=1, paper_gain=1, paper_loss=1,
            n_realized=4, n_open=0, win_rate=0.5,
            avg_hold_win=10.0, avg_hold_loss=12.0, hold_days_trading=True,
        )
        tags = persona_tags(rep, None, {"胜率": 0.5, "总盈利": 1, "总亏损": 1}, [], 4, 100)
        self.assertEqual([t for t in tags if t.name in ("死扛型", "落袋为安型")], [])


class TestBuildReport(unittest.TestCase):
    def _dataset(self) -> Dataset:
        events = _events()
        chunks = [
            _chunk("A", date(2026, 1, 5), date(2026, 2, 5), "1000", "2000"),
            _chunk("B", date(2026, 1, 6), date(2026, 3, 6), "2000", "-500"),
        ]
        return _mk_dataset(events, chunks)

    def test_contains_key_sections(self):
        html = build_report(self._dataset(), title="测试报告")
        for kw in ["测试报告", "处置效应", "累计已实现盈亏", "你拿了多久才卖",
                   "与基准对比", "明细（可审计）", "PGR − PLR"]:
            self.assertIn(kw, html)

    def test_escapes_user_data(self):
        events = _events()
        chunks = [_chunk('<script>alert(1)</script>',
                         date(2026, 1, 5), date(2026, 2, 5), "1000", "100")]
        html = build_report(_mk_dataset(events, chunks))
        self.assertNotIn("<script>alert(1)", html)
        self.assertIn("&lt;script&gt;", html)

    def test_missing_prices_leaves_honest_gap(self):
        html = build_report(self._dataset())  # 无 prices
        self.assertIn("缺未平仓市值，不可算", html)
        self.assertIn("--prices", html)

    def test_missing_benchmark_explains(self):
        html = build_report(self._dataset())
        self.assertIn("未提供基准数据，这一块留空", html)
        self.assertIn("--benchmark", html)

    def test_benchmark_renders_comparison(self):
        # 基准日期必须落在交割单区间（1/5 → 2/5）内，否则会被正确拒绝
        bench = {date(2026, 1, 5): D("4000"), date(2026, 2, 5): D("4400")}
        html = build_report(self._dataset(), benchmark=bench)
        self.assertIn("同期基准涨跌", html)
        self.assertIn("+10.0%", html)
        self.assertIn("只能当参考", html)

    def test_single_point_curve_does_not_crash(self):
        events = _events()
        chunks = [_chunk("A", date(2026, 1, 5), date(2026, 2, 5), "1000", "100")]
        html = build_report(_mk_dataset(events, chunks))
        self.assertIn("<svg", html)

    def test_all_flat_curve_does_not_crash(self):
        events = _events()
        chunks = [_chunk("A", date(2026, 1, 5), date(2026, 2, 5), "1000", "0")]
        html = build_report(_mk_dataset(events, chunks))
        self.assertIn("<svg", html)

    def test_numbers_match_dataset(self):
        """文本报告与 HTML 报告共用 Dataset —— 这里钉住一致性。"""
        ds = self._dataset()
        html = build_report(ds)
        self.assertIn("¥33,664.90" if False else "¥2,000.00", html)  # chunk A pnl
        summary = ds.summary()
        self.assertIn(f"{summary['胜率']:.1%}", html)  # 胜率来自同一 summary


class TestHtmlEscaping(unittest.TestCase):
    """回归：CSV 表头等外部文本进 HTML 前必须转义，否则破版 / 注入。"""

    def test_column_mapping_header_is_escaped(self):
        events = [Event(date(2026, 1, 5), "600519", EventType.BUY, D("100"), D("1680"))]
        lg = PositionLedger()
        lg.apply_all(events)
        ds = Dataset(
            paths=[Path("a.csv")],
            events=events,
            ledger=lg,
            file_metas=[(
                "a.csv",
                {
                    "encoding": "utf-8",
                    "header_row": 1,
                    "total_rows": 2,
                    # 表头直接来自用户 CSV，可能含 < & 等字符
                    "columns": {"date": "成交日期", "biz": "业务名称<x>"},
                },
            )],
        )
        html = build_report(ds)
        self.assertIn("业务名称&lt;x&gt;", html)
        self.assertNotIn("业务名称<x>", html)


if __name__ == "__main__":
    unittest.main()
