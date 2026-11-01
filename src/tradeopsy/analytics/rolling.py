"""逐卖出日滚动口径的处置效应（Odean 1998 原文口径）。

和 `disposition.compute_disposition` 的分工：
    disposition.py   **期末快照口径** —— 用"今天还没卖的"头寸算账面端。
                     只需一个当前价，最省数据；但历史上那些
                     "当时浮盈、后来卖掉、当时其实可以不卖"的机会全部丢失，
                     分母系统性偏小 → PGR 偏高。
    rolling.py（本模块）**逐日滚动口径** —— 每个卖出日都重新问一次
                     「此刻还有哪些浮盈 / 浮亏的头寸本可以卖却没卖」。
                     更贴近 Odean 原文，代价是需要逐日价格序列。

两者的差额本身就是一个结论：**口径敏感度**。
    差额大 = 你的账面端历史变化剧烈，只看期末会得出错误画像；
    差额小 = 结论稳健，不依赖用哪一天的价格。

三个必须做对的地方（普通脚本在这里全错）：

    1. **T+1：当日买入的批次不计入账面端。**
       它当天根本卖不掉，把它算进"本可以卖却没卖"的分母是错的。
       这一条对 A 股尤其重要：新股民当天买入的仓位往往是全仓最活跃的部分，
       算进去会把分母撑大、把 PGR/PLR 稀释到看不出效应。

    2. **分母按日累加，同一持仓会被重复计入。**
       这不是 bug，是口径本身的含义：「每个交易日你都面临一次卖出的机会」。
       但它意味着**不同回看期的账户不可直接比较** —— 一年的账户和三个月的账户，
       同一笔浮亏被计入的次数差 4 倍。要横向比就得统一观测窗口。

    3. **陈旧价格必须设上限并披露。**
       停牌股拿三个月前的价格当"今天的市值"，等于用一个假数填分母。
       这里允许有限度的向前沿用（默认 30 个交易日内），
       但把最陈旧的天数写进结果，让人看得见这层妥协。
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Optional

from ..ledger import PositionLedger
from ..model import Event, EventType, money
from ..prices import PriceHistory
from ..tradingcal import TradingCalendar

ZERO = Decimal("0")

__all__ = ["DailyObs", "RollingReport", "rolling_disposition", "replay_by_day"]


@dataclass
class DailyObs:
    """一个卖出日的观测。PGR/PLR 的原子单位。"""

    dt: date
    n_gain_sold: int = 0       # 当日卖出的盈利笔数
    n_loss_sold: int = 0       # 当日卖出的亏损笔数
    n_gain_open: int = 0       # 当日收盘仍持有、且浮盈的批次数
    n_loss_open: int = 0       # 当日收盘仍持有、且浮亏的批次数
    gain_sold: Decimal = ZERO      # 已实现盈利（金额）
    loss_sold: Decimal = ZERO      # 已实现亏损（正数）
    gain_open: Decimal = ZERO      # 账面盈利（金额）
    loss_open: Decimal = ZERO      # 账面亏损（正数）
    stale_max: int = 0             # 当日用到的最陈旧价格（交易日）
    unpriced: int = 0              # 当日因缺价 / 价格过旧而未计入的批次

    @property
    def pgr(self) -> Optional[float]:
        denom = self.n_gain_sold + self.n_gain_open
        return (self.n_gain_sold / denom) if denom else None

    @property
    def plr(self) -> Optional[float]:
        denom = self.n_loss_sold + self.n_loss_open
        return (self.n_loss_sold / denom) if denom else None

    @property
    def de(self) -> Optional[float]:
        p, l = self.pgr, self.plr
        return (p - l) if (p is not None and l is not None) else None

    @property
    def has_paper(self) -> bool:
        """当日有没有形成分母。两侧都为 0 时整日不可用。"""
        return (self.n_gain_open + self.n_loss_open) > 0


@dataclass
class RollingReport:
    """逐日滚动口径的汇总结果。"""

    #: "rolling" 正常 / "no_prices" 没有价格序列 / "insufficient" 有价但无有效观测日
    mode: str
    n_obs: int = 0                    # 有效观测日数
    #: pooled 计数口径（Odean 原文）：Σ分子 ÷ (Σ分子 + Σ账面)
    pgr: Optional[float] = None
    plr: Optional[float] = None
    de: Optional[float] = None
    #: pooled 金额口径（实践变体）
    pgr_amount: Optional[float] = None
    plr_amount: Optional[float] = None
    de_amount: Optional[float] = None
    #: 四项原始计数，让读者能自己复算
    n_gain_sold: int = 0
    n_gain_opp: int = 0               # Σ(已卖盈利 + 浮盈)
    n_loss_sold: int = 0
    n_loss_opp: int = 0
    #: 日度 DE 的简单平均（稳健性对照，不是主结论）
    daily_de_mean: Optional[float] = None
    daily_de_std: Optional[float] = None
    n_daily_de: int = 0
    obs: list[DailyObs] = field(default_factory=list)
    stale_max: int = 0
    stale_median: float = 0.0
    unpriced: int = 0                 # 全期因缺价未计入的批次次数
    skipped_days: int = 0             # 因账面端两侧全 0 被跳过的卖出日
    missing_symbols: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_result(self) -> bool:
        return self.de is not None or self.de_amount is not None

    @property
    def verdict(self) -> str:
        if self.de is None:
            return "逐日滚动口径数据不足，无法判定。"
        if self.de > 0.10:
            return "逐日滚动口径同样显示明显处置效应：多数卖出日都存在「浮盈已了结、浮亏仍在手」。"
        if self.de < -0.10:
            return "逐日滚动口径显示反向处置效应：账面上的亏损被你兑现得更快。"
        return "逐日滚动口径下处置效应不明显 —— 盈亏两端的兑现节奏大体相当。"


def replay_by_day(
    events: Iterable[Event],
) -> list[tuple[date, PositionLedger, list]]:
    """按日推进账本，返回 [(日期, 当日结束后的账本, 当日新增的已实现片段)]。

    为什么逐日重放而不是另写一套 FIFO：
        同一份配对逻辑只应该存在一处。逐日 apply 与一次性 apply_all 的
        事件顺序完全相同（都按 (dt, symbol)），所以结果逐字节等价 ——
        这一点由 test_rolling.py 里的等价性测试守住。

    内存上只保留一个账本实例，快照由调用方在需要的日子取，不缓存全历史。
    """
    grouped: dict[date, list[Event]] = defaultdict(list)
    for ev in events:
        grouped[ev.dt].append(ev)

    ledger = PositionLedger()
    out: list[tuple[date, PositionLedger, list]] = []
    for d in sorted(grouped):
        before = len(ledger.chunks)
        for ev in grouped[d]:
            ledger.apply(ev)
        out.append((d, ledger, ledger.chunks[before:]))
    return out


def rolling_disposition(
    events: Iterable[Event],
    prices: PriceHistory,
    *,
    trading_calendar: Optional[TradingCalendar] = None,
    max_stale_days: int = 30,
) -> RollingReport:
    """逐卖出日滚动口径的 PGR/PLR。

    参数：
        events            标准化事件流（已去重、按时间升序）
        prices            逐日收盘价序列（prices.PriceHistory）
        trading_calendar  交易日历；传入则「价格陈旧度」按交易日计，
                          不传按自然日计（会把长假误判成陈旧）
        max_stale_days    价格最多向前沿用多少个交易日；超出则该批次
                          当日不计入分母（宁可分母少一个，不填假数）

    返回 RollingReport。任何一步数据不足都通过 mode / notes 说明，
    绝不返回一个"看起来算完了"的退化数字。
    """
    events = list(events)
    rep = RollingReport(mode="rolling")

    if not prices:
        rep.mode = "no_prices"
        rep.notes.append(
            "未提供逐日价格序列，滚动口径不可计算。"
            "补法：--price-history 价格序列文件（CSV：日期,代码,收盘价）。"
        )
        return rep

    # 卖出日才有观测价值：没有卖出的日子，分子恒为 0，PGR/PLR 只会是 0，
    # 把它们算进来等于往分母里掺零，会让效应被系统性稀释。
    days = replay_by_day(events)

    stale_seen: list[int] = []
    all_syms: set[str] = {ev.symbol for ev in events}
    covered = prices.symbols
    rep.missing_symbols = sorted(all_syms - covered)

    def stale_of(price_dt: date, dt: date) -> int:
        """价格是几个交易日前收盘的。"""
        if trading_calendar is not None:
            return trading_calendar.count(price_dt, dt)
        return (dt - price_dt).days

    # 预先算出「哪些日子有卖出动作」。用原始事件而不是 chunks：
    # 卖出因无持仓被打回时不会产生 chunk，但那天确实发生了卖出动作，
    # 只是账本层面无法配对 —— 这种日子没有已实现端，纳入只会污染分母。
    sells_by_day: set[date] = {ev.dt for ev in events if ev.etype == EventType.SELL}

    for dt, ledger, new_chunks in days:
        if dt not in sells_by_day:
            continue

        obs = DailyObs(dt=dt)

        # ---- 已实现端：当日新增的 chunks
        for c in new_chunks:
            if c.pnl > 0:
                obs.n_gain_sold += 1
                obs.gain_sold += c.pnl
            elif c.pnl < 0:
                obs.n_loss_sold += 1
                obs.loss_sold += -c.pnl

        # ---- 账面端：当日收盘仍持有的批次
        for op in ledger.open_positions():
            # T+1：当日买入的批次今天就不可卖，不构成"本可以卖却没卖"
            if op.open_dt >= dt:
                continue
            px, price_dt = prices.on(op.symbol, dt)
            if px is None:
                obs.unpriced += 1
                continue
            stale = stale_of(price_dt, dt)
            if stale > max_stale_days:
                obs.unpriced += 1
                continue
            obs.stale_max = max(obs.stale_max, stale)
            gain = money(op.qty * px - op.cost)
            if gain > 0:
                obs.n_gain_open += 1
                obs.gain_open += gain
            elif gain < 0:
                obs.n_loss_open += 1
                obs.loss_open += -gain
            # gain == 0（恰好平价）既不算盈也不算亏，不进任何一端 ——
            # 否则分母会混入零值，把比例往中间拉。

        if not obs.has_paper:
            # 账面端两侧全 0：可能是当日清仓，也可能是全线缺价。
            # 两种情况都无法构成完整观测，整天跳过。
            # unpriced 仍要累加 —— 否则"因价格太旧而被跳过"这件事
            # 会从结果里彻底消失，用户只会看到"数据不足"却不知道缺在哪。
            rep.skipped_days += 1
            rep.unpriced += obs.unpriced
            continue

        rep.obs.append(obs)
        rep.unpriced += obs.unpriced
        if obs.stale_max:
            stale_seen.append(obs.stale_max)

    rep.n_obs = len(rep.obs)

    if rep.n_obs == 0:
        rep.mode = "insufficient"
        missing = rep.missing_symbols
        rep.notes.append(
            "有价格序列，但没有任何一个卖出日能凑齐「已实现 + 账面」两端。"
            + (f"价格文件里没有这些标的：{'、'.join(missing[:8])}"
               + ("…" if len(missing) > 8 else "") if missing else
               "可能是卖出日全部落在价格序列覆盖区间之外。")
        )
        if rep.unpriced:
            rep.notes.append(
                f"另有 {rep.unpriced} 个「持仓批次×日」因缺价或价格超过 "
                f"{max_stale_days} {('交易日' if trading_calendar else '自然日')} "
                f"被剔除。若价格文件确实覆盖了这些日子，请检查日期格式是否被识别。"
            )
        return rep

    # ---- pooled 汇总（Odean 原文口径：把各日观测求和再相除）
    rep.n_gain_sold = sum(o.n_gain_sold for o in rep.obs)
    rep.n_gain_opp = rep.n_gain_sold + sum(o.n_gain_open for o in rep.obs)
    rep.n_loss_sold = sum(o.n_loss_sold for o in rep.obs)
    rep.n_loss_opp = rep.n_loss_sold + sum(o.n_loss_open for o in rep.obs)

    rep.pgr = (rep.n_gain_sold / rep.n_gain_opp) if rep.n_gain_opp else None
    rep.plr = (rep.n_loss_sold / rep.n_loss_opp) if rep.n_loss_opp else None
    rep.de = (rep.pgr - rep.plr) if (rep.pgr is not None and rep.plr is not None) else None

    gs = sum((o.gain_sold for o in rep.obs), ZERO)
    go = sum((o.gain_open for o in rep.obs), ZERO)
    ls = sum((o.loss_sold for o in rep.obs), ZERO)
    lo = sum((o.loss_open for o in rep.obs), ZERO)

    rep.pgr_amount = float(gs / (gs + go)) if (gs + go) > 0 else None
    rep.plr_amount = float(ls / (ls + lo)) if (ls + lo) > 0 else None
    rep.de_amount = (
        rep.pgr_amount - rep.plr_amount
        if (rep.pgr_amount is not None and rep.plr_amount is not None)
        else None
    )

    daily = [o.de for o in rep.obs if o.de is not None]
    rep.n_daily_de = len(daily)
    if daily:
        rep.daily_de_mean = round(statistics.fmean(daily), 4)
        rep.daily_de_std = round(statistics.pstdev(daily), 4) if len(daily) > 1 else 0.0

    rep.stale_max = max(stale_seen) if stale_seen else 0
    rep.stale_median = round(statistics.median(stale_seen), 1) if stale_seen else 0.0

    # ---- 披露
    unit = "交易日" if trading_calendar is not None else "自然日"
    rep.notes.append(
        f"逐日滚动口径：{rep.n_obs} 个卖出日 · 每个卖出日按当日收盘价重估一次账面持仓。"
        f"分母按日累加，同一笔浮亏在持有着的每一天都被计入一次 —— "
        f"这是口径的本意（每天都给你卖出的机会），但也意味着"
        f"不同回看期的账户之间不能直接比较。"
    )
    # 这条必须单独讲：滚动口径的 PGR 绝对值通常远低于期末口径，
    # 那不是"你突然变得更愿意落袋了"，而是分母的定义变了
    # （分子只发生一次，分母随持有天数膨胀）。跨口径能比的只有 DE。
    rep.notes.append(
        "注意：这里的 PGR/PLR 绝对值不能与上一节的期末口径横比 —— "
        "分子（兑现次数）只发生一次，分母（机会）却随持有天数累加，"
        "所以本口径的 PGR 通常明显偏低。两种口径之间只有 DE 可比，"
        "因为 DE 的两端被同一种膨胀同步放大，比值差反而稳定。"
    )
    if rep.unpriced:
        rep.notes.append(
            f"有 {rep.unpriced} 个「持仓批次×日」因缺价或价格超过 {max_stale_days} {unit}"
            f"未计入分母（最陈旧的已用价格距今 {rep.stale_max} {unit}）。"
            f"缺价会让分母偏小、PGR/PLR 偏高，请把价格序列补全后再看结论。"
        )
    elif rep.stale_max > 0:
        rep.notes.append(
            f"部分持仓当日无收盘价，向前沿用了最近一次收盘价"
            f"（最陈旧 {rep.stale_max} {unit}，中位 {rep.stale_median} {unit}）—— "
            f"通常是停牌，属于可接受但需知情的偏差。"
        )
    if rep.skipped_days:
        rep.notes.append(
            f"{rep.skipped_days} 个卖出日因当日账面端为空（当日清仓或全线缺价）被跳过，"
            f"这些日子无法构成完整观测，纳入会得出 PGR=1 的假精确。"
        )
    if rep.missing_symbols:
        rep.notes.append(
            "价格序列里没有这些标的："
            + "、".join(rep.missing_symbols[:8])
            + ("…" if len(rep.missing_symbols) > 8 else "")
            + "（它们的账面端全部缺价）。"
        )
    return rep
