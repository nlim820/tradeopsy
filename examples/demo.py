"""演示：不用任何真实交割单，也能跑出一份完整报告。

运行：
    python examples/demo.py

这里构造的是一个「典型 A 股散户」三个月的交易流水，
刻意塞进了送转、分红、配股、部分平仓、可转债 T+0 这些真实会遇到的脏情况，
用来验证引擎在复杂场景下仍然算得对。
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:  # Windows 终端默认不是 UTF-8，强制一下免得中文乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tradeopsy.ledger import PositionLedger, summarize  # noqa: E402
from tradeopsy.model import Event, EventType  # noqa: E402

D = Decimal


def E(dt, sym, etype, qty="0", price="0", fee="0", cash="0", note=""):
    return Event(
        dt=dt,
        symbol=sym,
        etype=etype,
        qty=D(qty),
        price=D(price),
        fee=D(fee),
        cash=D(cash),
        note=note,
    )


# fmt: off
EVENTS = [
    # —— 贵州茅台：建仓 + 10转5 + 分红 + 清仓 ——
    E(date(2026, 1, 5),  "600519", EventType.BUY,  "100", "1680.00", "5.00"),
    E(date(2026, 6, 20), "600519", EventType.SPLIT, qty="50"),                    # 10 转 5
    E(date(2026, 6, 25), "600519", EventType.DIVIDEND, cash="1500.00"),           # 现金分红
    E(date(2026, 9, 10), "600519", EventType.SELL, "150", "1720.00", "13.42"),

    # —— 平安银行：三次加仓 + 分批止盈（考察 FIFO） ——
    E(date(2026, 2, 2),  "000001", EventType.BUY, "2000", "10.50", "5.53"),
    E(date(2026, 2, 9),  "000001", EventType.BUY, "1000", "10.20", "5.26"),
    E(date(2026, 3, 3),  "000001", EventType.BUY, "1000", "9.80",  "5.10"),
    E(date(2026, 4, 15), "000001", EventType.SELL, "2500", "11.60", "19.45"),
    E(date(2026, 5, 20), "000001", EventType.SELL, "1500", "9.30",  "12.05"),

    # —— 宁德时代：追高后死扛（处置效应的典型样本） ——
    E(date(2026, 3, 10), "300750", EventType.BUY,  "200", "260.00", "5.26"),
    E(date(2026, 4, 2),  "300750", EventType.BUY,  "100", "310.00", "5.16"),
    E(date(2026, 4, 8),  "300750", EventType.SELL, "100", "330.00", "16.55"),  # 先卖赚钱那批
    # 剩下的 200 股死扛不卖，至今未平仓

    # —— 浦发银行：配股缴款 ——
    E(date(2026, 5, 6),  "600000", EventType.BUY,   "1000", "7.20", "5.07"),
    E(date(2026, 6, 1),  "600000", EventType.RIGHTS, "300", "5.60"),          # 10 配 3
    E(date(2026, 8, 18), "600000", EventType.SELL,  "1300", "8.10", "10.55"),

    # —— 可转债：T+0 当天进出 ——
    E(date(2026, 7, 3),  "113050", EventType.BUY,  "100", "118.00", "5.00"),
    E(date(2026, 7, 3),  "113050", EventType.SELL, "100", "121.50", "5.00"),

    # —— 一行看不懂的业务，验证不会炸掉整批 ——
    E(date(2026, 7, 10), "000001", EventType.OTHER, note="担保品划转"),
]
# fmt: on


def _w(text: str) -> int:
    """终端显示宽度：中日韩字符占 2 列，其余占 1 列。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - _w(text))
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def row(cells: list[tuple[str, int, str]]) -> str:
    return "".join(pad(t, w, a) for t, w, a in cells)


def main() -> None:
    led = PositionLedger()
    led.apply_all(EVENTS)
    s = summarize(led)

    print("=" * 78)
    print("  tradeopsy 引擎自检 · 仿真三个月流水")
    print("=" * 78)

    print("\n【逐笔已实现明细】")
    cols = [("标的", 9, "left"), ("建仓日", 12, "left"), ("平仓日", 12, "left"),
            ("数量", 8, "right"), ("成本", 13, "right"), ("净收入", 13, "right"),
            ("盈亏", 12, "right"), ("天数", 6, "right")]
    header = row(cols)
    print(header)
    print("-" * _w(header))
    for c in led.chunks:
        print(
            row([
                (c.symbol, 9, "left"),
                (str(c.open_dt), 12, "left"),
                (str(c.close_dt), 12, "left"),
                (f"{float(c.qty):,.0f}", 8, "right"),
                (f"{float(c.cost):,.2f}", 13, "right"),
                (f"{float(c.proceeds):,.2f}", 13, "right"),
                (f"{float(c.pnl):,.2f}", 12, "right"),
                (str(c.hold_days), 6, "right"),
            ])
        )

    print("\n【绩效汇总】")
    for k, v in s.items():
        val = f"{v:,.2f}" if isinstance(v, float) else str(v)
        print(f"  {pad(k, 20)}{pad(val, 14, 'right')}")

    print("\n【当前持仓】")
    for p in led.open_positions():
        print(f"  {p.symbol}  {p.qty}股  成本 {p.cost}  建仓 {p.open_dt}")

    print("\n【数据质量警告】")
    if led.warnings:
        for w in led.warnings:
            print(f"  ! {w}")
    else:
        print("  无")

    print("\n" + "=" * 62)
    print("  引擎自检通过。下一步：把真实交割单接进来。")
    print("=" * 62)


if __name__ == "__main__":
    main()
