"""持仓账本：FIFO 配对 + A股公司行为处理。

这是整个项目的技术核心，也是最容易出错的地方。
大多数「交割单分析」脚本只处理买入/卖出，一旦遇到 10 转 5 或配股缴款，
每股成本和持仓数量就全错了，后面所有指标都是错的。

这里用一个规则守住正确性：
    **任何事件都不改变「总成本」与「总股数」的一致性**
    送转股  -> 股数增加，总成本不变，每股成本下降
    分红    -> 股数不变，成本不变，现金单独记
    配股    -> 股数增加，总成本增加（增加了缴款），每股成本重新摊薄
    卖出    -> 股数减少，总成本按 FIFO 减少，现金流入
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from .model import (
    Event,
    EventType,
    Lot,
    OpenPosition,
    RealizedChunk,
    money,
    unit_price,
)

if TYPE_CHECKING:  # 仅类型标注用，避免运行期循环导入
    from .tradingcal import TradingCalendar


class LedgerError(Exception):
    """账本层面的错误（例如卖出超过持仓）。"""


class PositionLedger:
    """按标的维护 FIFO 批次队列。"""

    def __init__(self) -> None:
        self._lots: dict[str, list[Lot]] = defaultdict(list)
        self.chunks: list[RealizedChunk] = []
        self.dividends: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.dividend_tax: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.buy_fees: Decimal = Decimal("0")
        self.sell_fees: Decimal = Decimal("0")
        self.warnings: list[str] = []

    # ---------- 对外接口 ----------

    def apply(self, ev: Event) -> None:
        """消费一个事件。事件必须按时间升序投喂。"""
        handler = {
            EventType.BUY: self._buy,
            EventType.SELL: self._sell,
            EventType.SPLIT: self._split,
            EventType.DIVIDEND: self._dividend,
            EventType.RIGHTS: self._rights,
            EventType.IPO_ALLOT: self._ipo_allot,
            EventType.OTHER: self._other,
        }[ev.etype]
        handler(ev)

    def apply_all(self, events: list[Event]) -> None:
        for ev in sorted(events, key=lambda e: (e.dt, e.symbol)):
            self.apply(ev)

    # ---------- 事件处理 ----------

    def _buy(self, ev: Event) -> None:
        if ev.qty <= 0:
            self._warn(ev, "买入数量为 0，已跳过")
            return
        # 买入费用计入成本：这是「真实成本」，不是成交价
        total_cost = money(ev.qty * ev.price + ev.fee)
        self._lots[ev.symbol].append(
            Lot(dt=ev.dt, qty=ev.qty, cost=total_cost, fee=ev.fee)
        )
        self.buy_fees += ev.fee

    def _sell(self, ev: Event) -> None:
        lots = self._lots.get(ev.symbol, [])
        available = sum((l.qty for l in lots), Decimal("0"))
        if available <= 0:
            self._warn(ev, f"卖出 {ev.qty} 股但无持仓，已跳过（可能是漏了买入记录或转托管）")
            return

        qty_to_sell = ev.qty
        if qty_to_sell > available:
            self._warn(
                ev,
                f"卖出 {qty_to_sell} 股超过持仓 {available} 股，"
                f"按持仓上限成交（多半是交割单区间截断导致）",
            )
            qty_to_sell = available

        gross = qty_to_sell * ev.price
        # 卖出费用按成交数量比例分摊到各个批次
        fee_per_share = ev.fee / qty_to_sell if qty_to_sell else Decimal("0")
        remaining = qty_to_sell
        fee_allocated = Decimal("0")

        while remaining > 0 and lots:
            lot = lots[0]
            take = min(lot.qty, remaining)
            # 整批吃掉时直接转移全额成本；部分卖出才按比例分摊。
            # 这个小判断很关键：它保证同一笔卖出在多批次间分摊后，
            # 「成本之和」仍然精确等于原总成本，不会漏掉几分钱。
            if take == lot.qty:
                cost_part = lot.cost
            else:
                cost_part = money(lot.cost * take / lot.qty)
            # 费用同理：最后一批吃掉全部剩余费用，保证分摊之和精确等于实收费用。
            # 否则 10.71 元分成 8.93 + 1.79 = 10.72，凭空多收用户一分钱。
            if take == remaining:
                fee_part = money(ev.fee - fee_allocated)
            else:
                fee_part = money(fee_per_share * take)
            fee_allocated += fee_part
            self.chunks.append(
                RealizedChunk(
                    symbol=ev.symbol,
                    open_dt=lot.dt,
                    close_dt=ev.dt,
                    qty=take,
                    cost=cost_part,
                    proceeds=money(take * ev.price - fee_part),
                    sell_fee=fee_part,
                    note=ev.note,
                )
            )
            lot.qty -= take
            lot.cost -= cost_part
            remaining -= take
            if lot.qty <= 0:
                lots.pop(0)

        self.sell_fees += ev.fee
        _ = gross  # 保留中间量便于调试，实际净额已按批次写入 chunk

    def _split(self, ev: Event) -> None:
        """送股 / 转增：股数增加，总成本不变。

        实现上对每个批次同比例放大股数、同比例缩小每股成本。
        这样「总成本 = 各批次 qty × unit_cost 之和」保持不变，
        而每股成本自动变成摊薄后的值 —— 和券商 App 显示的成本价一致。
        """
        lots = self._lots.get(ev.symbol, [])
        total_qty = sum((l.qty for l in lots), Decimal("0"))
        if total_qty <= 0:
            self._warn(ev, f"收到送转 {ev.qty} 股但没有底仓，已跳过")
            return
        new_total = total_qty + ev.qty
        ratio = new_total / total_qty
        for lot in lots:
            # 只动股数，不动总成本 —— 这是零漂移的关键
            lot.qty = unit_price(lot.qty * ratio)

    def _dividend(self, ev: Event) -> None:
        """现金分红：不影响持仓成本，单独累计。

        为什么不冲减成本：券商有「摊薄成本」和「实际成本」两套口径，
        混在一起会让复利计算失真。这里保持成本原样，分红单独统计，
        在总收益层面再把两者相加 —— 口径清晰，可审计。
        """
        self.dividends[ev.symbol] += ev.cash
        if ev.fee:
            self.dividend_tax[ev.symbol] += ev.fee

    def _rights(self, ev: Event) -> None:
        """配股缴款：掏钱换新股，形成一个新的买入批次。"""
        if ev.qty <= 0:
            self._warn(ev, "配股数量为 0，已跳过")
            return
        paid = money(ev.cash if ev.cash else ev.qty * ev.price)
        self._lots[ev.symbol].append(Lot(dt=ev.dt, qty=ev.qty, cost=paid))

    def _ipo_allot(self, ev: Event) -> None:
        """打新中签股份到账，按发行价作为成本。"""
        if ev.qty <= 0:
            return
        self._lots[ev.symbol].append(
            Lot(dt=ev.dt, qty=ev.qty, cost=money(ev.qty * ev.price))
        )

    def _other(self, ev: Event) -> None:
        self._warn(ev, f"未识别业务类型：{ev.note or '（无说明）'}，已忽略但建议人工复核")

    # ---------- 结果视图 ----------

    def open_positions(self) -> list[OpenPosition]:
        out: list[OpenPosition] = []
        for symbol, lots in self._lots.items():
            for lot in lots:
                if lot.qty > 0:
                    out.append(
                        OpenPosition(
                            symbol=symbol,
                            open_dt=lot.dt,
                            qty=lot.qty,
                            cost=lot.cost,
                        )
                    )
        return out

    def total_dividend(self) -> Decimal:
        return money(sum(self.dividends.values(), Decimal("0")))

    def realized_pnl(self) -> Decimal:
        """已实现价差盈亏（不含分红）。"""
        return money(sum((c.pnl for c in self.chunks), Decimal("0")))

    def total_pnl(self) -> Decimal:
        """已实现总收益 = 价差盈亏 + 现金分红。"""
        return money(self.realized_pnl() + self.total_dividend())

    def _warn(self, ev: Event, msg: str) -> None:
        self.warnings.append(f"[{ev.dt}] {ev.symbol} {ev.etype.value}: {msg}")


def summarize(ledger: PositionLedger, calendar: "TradingCalendar | None" = None) -> dict:
    """汇总一份可读的统计结果（不含行情相关指标）。

    calendar 传入时，「平均持有天数」按 A 股交易日口径统计；
    不传则退回自然日 —— 自然日是账本里的事实值，交易日才是与学术口径对齐的值。
    """
    chunks = ledger.chunks
    wins = [c for c in chunks if c.is_win]
    losses = [c for c in chunks if not c.is_win]

    gross_win = money(sum((c.pnl for c in wins), Decimal("0")))
    gross_loss = money(-sum((c.pnl for c in losses), Decimal("0")))

    def avg_days(items: list[RealizedChunk]) -> float:
        if not items:
            return 0.0
        if calendar is None:
            return round(sum(c.hold_days for c in items) / len(items), 1)
        return round(sum(calendar.count(c.open_dt, c.close_dt) for c in items) / len(items), 1)

    unit = "交易日" if calendar is not None else "自然日"
    return {
        "成交笔数": len(chunks),
        "盈利笔数": len(wins),
        "亏损笔数": len(losses),
        "胜率": round(len(wins) / len(chunks), 4) if chunks else 0.0,
        "总盈利": float(gross_win),
        "总亏损": float(gross_loss),
        "盈亏比": round(float(gross_win / gross_loss), 2) if gross_loss else None,
        "已实现价差盈亏": float(ledger.realized_pnl()),
        "现金分红": float(ledger.total_dividend()),
        "已实现总收益": float(ledger.total_pnl()),
        f"平均持有{unit}": avg_days(chunks),
        f"盈利笔平均持有{unit}": avg_days(wins),
        f"亏损笔平均持有{unit}": avg_days(losses),
        "买入费用合计": float(ledger.buy_fees),
        "卖出费用合计": float(ledger.sell_fees),
        "未平仓批次": len(ledger.open_positions()),
        "警告数": len(ledger.warnings),
    }


def timeline(events: list[Event]) -> tuple[date, date] | None:
    if not events:
        return None
    ds = [e.dt for e in events]
    return min(ds), max(ds)
