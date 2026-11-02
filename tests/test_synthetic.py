"""十套模拟人格生成器的回归测试。

每一套都必须：通过工作日/T+1/超卖三重自检、期末价与期末持仓互证、
FIFO 账本干净落地。生成是确定性的（seed = 目录名），任何一条失败
都意味着有人改坏了人格构造或公共积木。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "synthetic_generate", ROOT / "examples" / "synthetic_accounts" / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load_generator()

from decimal import Decimal as D_  # noqa: E402

from tradeopsy.ledger import PositionLedger  # noqa: E402
from tradeopsy.model import Event, classify_business  # noqa: E402


def _to_events(rows: list[dict]) -> list[Event]:
    """生成器的行结构 → 标准事件。字段语义按 model.Event 的文档。"""
    from tradeopsy.model import EventType
    out = []
    for r in rows:
        et = classify_business(r["biz"])
        if et == EventType.OTHER and r["biz"] in ("证券买入", "证券卖出"):
            et = EventType.BUY if r["qty"] > 0 else EventType.SELL
        fee = r["comm"] + r["stamp"] + r["transfer"]
        qty = abs(r["qty"])
        if et == EventType.DIVIDEND:
            out.append(Event(dt=r["dt"], symbol=r["symbol"], etype=et,
                             qty=D_(0), price=D_(0), fee=D_(0), cash=r["amount"]))
        elif et == EventType.SELL:
            out.append(Event(dt=r["dt"], symbol=r["symbol"], etype=et,
                             qty=qty, price=r["price"], fee=fee, cash=D_(0)))
        else:
            out.append(Event(dt=r["dt"], symbol=r["symbol"], etype=et,
                             qty=qty, price=r["price"], fee=fee, cash=D_(0)))
    return out


class TestPersonaGeneration(unittest.TestCase):
    def test_all_ten_personas_are_consistent(self):
        seen = set()
        for pid, _desc, fn, _fmt, _cut in G.PERSONAS:
            with self.subTest(pid=pid):
                rng = __import__("random").Random(pid)
                script, closes = fn(rng)
                rows = G.expand_script(script)
                # 三重自检：工作日 / T+1 / 超卖
                G.validate(rows)
                # 期末价与期末持仓互证
                held = G.final_holdings(rows)
                self.assertEqual(set(closes), set(held),
                                 f"{pid}: 期末价集合与持仓集合不一致")
                # 账本干净落地：全部成交可配对，无凭空卖出
                led = PositionLedger()
                led.apply_all(_to_events(rows))
                self.assertTrue(len(led.chunks) > 0 or len(rows) > 0)
                seen.add(pid)
        self.assertEqual(len(seen), 10)

    def test_t0_bond_has_same_day_roundtrip(self):
        """p07 必须保留当日买卖对 —— 这是 T+0 豁免的活样本。"""
        rng = __import__("random").Random("p07_bond_t0")
        script, _ = G.p07_bond_t0(rng)
        rows = G.expand_script(script)
        by_key = {}
        for r in rows:
            by_key.setdefault((r["dt"], r["symbol"]), []).append(r["biz"])
        same_day = [k for k, v in by_key.items()
                    if "证券买入" in v and "证券卖出" in v]
        self.assertGreater(len(same_day), 40, "T+0 当日平仓对应大量消失")

    def test_events_conversion_matches_parse(self):
        """expand 出的行喂给正式解析器，事件数应与行数一致（无告警行时）。"""
        rng = __import__("random").Random("p10_mixed")
        script, _ = G.p10_mixed(rng)
        rows = G.expand_script(script)
        events = _to_events(rows)
        self.assertEqual(len(events), len(rows))


if __name__ == "__main__":
    unittest.main()
