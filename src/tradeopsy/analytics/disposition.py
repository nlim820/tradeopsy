"""处置效应（Disposition Effect）度量 —— 项目的学术内核。

为什么这一块是项目的灵魂：
    市面上大多数"交割单分析"工具只给你胜率、盈亏比、收益率，
    这些是任何交易软件都有的。真正稀缺的是**行为归因**：
    你是不是"赚钱的票拿不住、亏钱的票死扛"？这正是行为金融学最经典的
    处置效应（Odean, 1998），也是散户长期跑不赢宽基指数的核心原因之一。

学术口径（Odean 1998）：
    PGR = 已实现盈利 / (已实现盈利 + 账面盈利)
    PLR = 已实现亏损 / (已实现亏损 + 账面亏损)
    disposition effect DE = PGR - PLR
    若 > 0，说明你更倾向于"落袋盈利、持有亏损" —— 典型的处置效应。

    符号的推导（务必别记反）：
        "落袋盈利" => 盈利机会里被兑现的比例高 => PGR 趋近 1
        "持有亏损" => 亏损机会里被兑现的比例低 => PLR 趋近 0
        两者同时成立时 PGR - PLR > 0，所以 DE 必须是 PGR 减 PLR。
        Odean 原始样本 PGR=0.148 > PLR=0.098，DE=+0.05，方向一致。

A 股本土化要改的三个地方（这是普通脚本做不对的）：
    1. 持有天数必须按**交易日**算，不能按自然日。
       T+1 制度下当天买的票当天卖不掉，自然日会系统性高估持有期；
       美股研究用自然日，直接套到 A 股会失真。
    2. 成本必须含费用。买入佣金计入成本、卖出费用从收入里扣，
       否则 PGR/PLR 的分母就少算了摩擦成本，结论偏乐观。
    3. 必须同时有"已实现"和"账面"两端，单看已实现会退化成无意义值。
       所以本模块把"未平仓市值"交给调用方（行情模块）注入，
       analytics 本身不联网、可独立测试 —— 这是刻意的工程设计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Optional

from ..ledger import PositionLedger
from ..model import RealizedChunk, money
from ..tradingcal import TradingCalendar


@dataclass(frozen=True)
class OpenPositionMV:
    """未平仓头寸 + 它的当前市值。

    analytics 模块不联网，market_value 由调用方用最新行情填入。
    若暂时拿不到行情，用成本 cost 填入即可（此时 paper gain = 0）。
    """

    symbol: str
    cost: Decimal
    market_value: Decimal

    @property
    def paper_gain(self) -> Decimal:
        return money(self.market_value - self.cost)


@dataclass
class DispositionReport:
    mode: str                              # "full" 或 "realized_only"
    pgr: Optional[float]                   # 盈利实现比例
    plr: Optional[float]                   # 亏损实现比例
    disposition_effect: Optional[float]    # pgr - plr（正值 = 处置效应）
    realized_gain: float
    realized_loss: float
    paper_gain: float
    paper_loss: float
    n_realized: int
    n_open: int
    win_rate: float
    avg_hold_win: float
    avg_hold_loss: float
    hold_days_trading: bool
    #: 自然日口径的平均持有期，仅用于和交易日口径对照展示（口径差异的证据）
    avg_hold_win_calendar: float = 0.0
    avg_hold_loss_calendar: float = 0.0
    #: 机会数：盈利端 = 已实现盈利笔数 + 期末浮盈持仓数；亏损端同构。
    #: 机会数决定方向性判定的可信度 —— 任一端太少时 verdict 拒绝给方向。
    n_gain_opp: int = 0
    n_loss_opp: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_full(self) -> bool:
        return self.pgr is not None and self.plr is not None

    @property
    def verdict(self) -> str:
        if not self.has_full:
            # 区分「真的没有那一端」和「没给价格」—— 两者给同一句话会误导。
            if self.pgr is None and self.plr is None:
                return "样本不足或未提供未平仓市值，无法判定处置效应（仅给出已实现画像）。"
            if self.pgr is None:
                return ("观察期内没有任何盈利端机会（从未在盈利状态卖出、期末也无浮盈），"
                        "处置效应无从谈起 —— 这本身是信息：你没有『落袋』行为。")
            return ("观察期内没有任何亏损端机会（从未割肉、期末也无浮亏），"
                    "处置效应无从谈起。")
        small = min(self.n_gain_opp, self.n_loss_opp)
        if small < 8:
            return (f"样本太小（盈利端 {self.n_gain_opp} 个机会 / 亏损端 "
                    f"{self.n_loss_opp} 个），方向性判定不可靠 —— "
                    "请以显著性检验的置信区间为准，不要看这里的方向。")
        de = self.disposition_effect or 0.0
        if de > 0.10:
            return "明显处置效应：倾向于过早卖出盈利、长期持有亏损。"
        if de < -0.10:
            return "反向处置效应：倾向于让盈利继续跑、及时止损（难得）。"
        return "处置效应不明显，盈亏两端的兑现节奏大体均衡。"


def _classify(chunks: list[RealizedChunk]) -> tuple[Decimal, Decimal, list[RealizedChunk], list[RealizedChunk]]:
    """把已实现的每一笔分成赢家 / 输家，并分别累加盈亏。

    平手（pnl == 0）既不算赢也不算输，不参与任何一端 ——
    否则分母会混入零盈亏，扭曲 PGR/PLR。
    """
    rg = Decimal("0")   # realized gain
    rl = Decimal("0")   # realized loss（取正值）
    wins: list[RealizedChunk] = []
    losses: list[RealizedChunk] = []
    for c in chunks:
        if c.pnl > 0:
            rg += c.pnl
            wins.append(c)
        elif c.pnl < 0:
            rl += -c.pnl
            losses.append(c)
    return rg, rl, wins, losses


def _normalize_calendar(
    cal: "TradingCalendar | set[date] | None",
) -> TradingCalendar | None:
    """兼容两种入参：TradingCalendar 直接用，set[date] 包一层，None 透传。

    保留 set 支持不是为了偷懒 —— 老测试与外部调用都按 set 传，
    破坏性改签名会让下游静默出错，这类兼容层成本远低于它的价值。
    """
    if cal is None or isinstance(cal, TradingCalendar):
        return cal
    return TradingCalendar.from_dates(cal, source="explicit")


def _avg_hold(
    items: list[RealizedChunk], calendar: TradingCalendar | None
) -> tuple[float, bool]:
    """平均持有期。返回 (天数, 是否被日历完整覆盖)。

    交易日口径下用二分计数，复杂度 O(n log m)；
    自然日口径下直接用 chunk.hold_days，是底层事实，不加工。
    """
    if not items:
        return 0.0, True
    if calendar is None:
        return round(sum(c.hold_days for c in items) / len(items), 1), False
    counts: list[int] = []
    covered = True
    for c in items:
        n, ok = calendar.count_checked(c.open_dt, c.close_dt)
        counts.append(n)
        covered = covered and ok
    return round(sum(counts) / len(counts), 1), covered


def compute_disposition(
    chunks: Iterable[RealizedChunk],
    open_positions: Optional[Iterable[OpenPositionMV]] = None,
    *,
    trading_calendar: "TradingCalendar | set[date] | None" = None,
) -> DispositionReport:
    """Odean (1998) 处置效应度量。

    参数：
        chunks          已实现盈亏片段（来自 ledger.chunks）
        open_positions  未平仓头寸 + 当前市值（来自行情）；为 None 时退化为
                        realized_only 模式，PGR/PLR 不可计算
        trading_calendar 交易日历（TradingCalendar 或 set[date]）；
                        传入则持有天数按 A 股交易日口径算，不传则按自然日

    返回 DispositionReport，含 PGR/PLR/处置效应/人格判定与辅助统计。
    """
    chunks = list(chunks)
    cal = _normalize_calendar(trading_calendar)
    rg, rl, wins, losses = _classify(chunks)

    pg = Decimal("0")   # paper gain
    pl = Decimal("0")   # paper loss（取正值）
    n_open = 0
    n_open_gain = 0
    n_open_loss = 0
    if open_positions is not None:
        for op in open_positions:
            n_open += 1
            pg += max(op.paper_gain, Decimal("0"))
            pl += max(-op.paper_gain, Decimal("0"))
            if op.paper_gain > 0:
                n_open_gain += 1
            elif op.paper_gain < 0:
                n_open_loss += 1

    mode = "full" if open_positions is not None else "realized_only"

    # realized_only 模式下没有账面盈亏，PGR/PLR 在数学上退化为 1.0 / 1.0，
    # 这个数毫无意义且极易误导，直接置 None，由调用方用 notes 说明退化原因。
    if mode == "realized_only":
        pgr = plr = de = None
    else:
        pgr = float(rg / (rg + pg)) if (rg + pg) > 0 else None
        plr = float(rl / (rl + pl)) if (rl + pl) > 0 else None
        de = (pgr - plr) if (pgr is not None and plr is not None) else None

    avg_win, cov_win = _avg_hold(wins, cal)
    avg_loss, cov_loss = _avg_hold(losses, cal)
    # 自然日口径恒为对照值，用来展示"两种口径差多少"
    cal_win, _ = _avg_hold(wins, None)
    cal_loss, _ = _avg_hold(losses, None)

    notes: list[str] = []
    if mode == "realized_only":
        notes.append("未提供未平仓市值，PGR/PLR 不可用；以下仅为已实现部分画像。")
    if cal is not None:
        notes.append(f"持有天数按交易日计算（A股口径）· {cal.describe()}")
        if not (cov_win and cov_loss):
            notes.append(
                "部分持仓区间超出交易日历覆盖范围，持有期可能偏小，结论请保守看待。"
            )
    else:
        notes.append("持有天数按自然日计算；A股严谨口径应为交易日（P5 技术债）。")

    return DispositionReport(
        mode=mode,
        pgr=pgr,
        plr=plr,
        disposition_effect=de,
        realized_gain=float(rg),
        realized_loss=float(rl),
        paper_gain=float(pg),
        paper_loss=float(pl),
        n_realized=len(chunks),
        n_open=n_open,
        n_gain_opp=len(wins) + n_open_gain,
        n_loss_opp=len(losses) + n_open_loss,
        win_rate=round(len(wins) / len(chunks), 4) if chunks else 0.0,
        avg_hold_win=avg_win,
        avg_hold_loss=avg_loss,
        hold_days_trading=cal is not None,
        avg_hold_win_calendar=cal_win,
        avg_hold_loss_calendar=cal_loss,
        notes=notes,
    )


def open_positions_from_ledger(
    ledger: PositionLedger,
    prices: Optional[dict[str, Decimal]] = None,
) -> list[OpenPositionMV]:
    """从账本取未平仓头寸，并用 prices 注入当前市值。

    prices 缺某只标的时，该头寸按成本计（paper gain = 0），
    不会中断整个计算。
    """
    out: list[OpenPositionMV] = []
    for op in ledger.open_positions():
        px = prices.get(op.symbol) if prices else None
        mv = money(op.qty * px) if px is not None else op.cost
        out.append(OpenPositionMV(op.symbol, op.cost, mv))
    return out
