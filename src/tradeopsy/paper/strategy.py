"""策略层：ETF 双均线轮动。

为什么是这个策略，而不是更花哨的东西：

- **必须可解释**。这个账户每天在无人值守的情况下动钱，如果亏了，
  你半年后回看必须能立刻说出"因为什么信号买的"。黑箱策略在纸面账户里没有意义。
- **必须少交易**。每天换手会把佣金吃光，而且流水太乱，tradeopsy 后面的
  行为分析模块会被噪音淹没。
- **必须能空仓**。A 股 ETF 大部分时间在跌，一个永远满仓的策略跑出来的曲线
  看起来"有收益"，其实只是 beta。空仓能力是策略的一部分，不是懦弱。

规则（三步，没有任何隐藏条件）：

1. 趋势过滤：收盘价 > MA20 且 MA20 > MA60  → 该标的进入候选
2. 动量排序：候选按近 20 日涨幅降序，取前 N 名（默认 2）
3. 等权分配：命中 N 名就各 50%，只命中 1 名就 50%（留一半现金），一个都没有就空仓

第 3 条里"只命中 1 名也只给 50%"是刻意的：它让账户仓位天然随市场环境浮动，
而不是在弱势里被迫满仓。这是风险控制，不是保守。

**买入与卖出用不同阈值（滞回）**，这是整段代码里最重要的一个设计。
第一版两者都用 MA20，回放三个月出现了 30 笔成交、多笔"买了次日就卖"，
手续费吃掉 0.36%。原因很简单：价格在 MA20 附近震荡时，条件天天翻转。
修法是把卖出条件放松到 MA60，并把"已持有且仍在候选里"的标的强制保留。
代价是止损变慢，收益是换手大幅下降 —— 对纸面账户来说这个交换是划算的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from .market import Bar

#: 标的池。选这 6 只的理由：
#:   - 覆盖大盘/中盘/成长/科技/周期/避险六个方向，相关性不高，轮动才有意义
#:   - 全部是规模最大的 ETF，流动性好，不会出现"信号有了但买不到量"的失真
#:   - 沪深两市都有，顺便验证代码前缀判断逻辑
UNIVERSE: tuple[tuple[str, str], ...] = (
    ("510300", "沪深300ETF"),
    ("510500", "中证500ETF"),
    ("159915", "创业板ETF"),
    ("588000", "科创50ETF"),
    ("512880", "证券ETF"),
    ("518880", "黄金ETF"),
)

UNIVERSE_NAMES: dict[str, str] = dict(UNIVERSE)


@dataclass(frozen=True)
class StrategyConfig:
    fast: int = 20
    slow: int = 60
    momentum: int = 20
    top_n: int = 2
    #: 单标的权重上限。0.5 意味着最多两只满仓。
    max_weight: Decimal = Decimal("0.5")

    @property
    def warmup(self) -> int:
        """策略启动前需要多少根 K 线才能算出所有指标。

        规则：max(slow, momentum) + fast，再留 10 根冗余。
        少算一根 MA60 就会变成 None，然后整只标的被静默跳过 —— 这种 bug
        不会报错，只会让策略莫名其妙地空仓，所以宁可多等。
        """
        return max(self.slow, self.momentum) + self.fast + 10


def sma(values: Sequence[Decimal], n: int) -> Decimal | None:
    """简单移动平均。数据不足返回 None，而不是返回短窗口均值。

    这是个容易踩的坑：用 `values[-n:]` 计算时如果长度不够，切片会静默
    返回更短的列表，算出来的"MA60"其实是 MA23，而且看起来完全正常。
    """
    if n <= 0 or len(values) < n:
        return None
    window = values[-n:]
    return sum(window) / Decimal(n)


def momentum(bars: Sequence[Bar], n: int) -> Decimal | None:
    """近 n 个交易日的涨跌幅。"""
    if n <= 0 or len(bars) < n + 1:
        return None
    past = bars[-(n + 1)].close
    if past <= 0:
        return None
    return bars[-1].close / past - 1


@dataclass
class Candidate:
    """一个候选标的的诊断信息。留着它是为了让报告能解释"为什么没买它"。"""

    symbol: str
    name: str
    close: Decimal
    ma_fast: Decimal
    ma_slow: Decimal
    mom: Decimal
    passed: bool
    reason: str
    rank: int = 0


@dataclass
class Decision:
    """一次决策的完整结果。

    targets 为空字典 = 空仓，这是合法状态，不是错误。
    """

    as_of: str
    targets: dict[str, Decimal] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    #: 大盘过滤开关。False 时不新开仓，已有持仓按个股规则正常处理。
    market_ok: bool = True

    @property
    def is_flat(self) -> bool:
        return not self.targets

    def summary(self) -> str:
        if not self.market_ok:
            return "大盘过滤：沪深300ETF 跌破 MA60，不新开仓"
        if self.is_flat:
            return "空仓（无标的满足趋势条件）"
        parts = [
            f"{UNIVERSE_NAMES.get(s, s)} {w:.0%}" for s, w in self.targets.items()
        ]
        return "目标持仓：" + "、".join(parts)


def market_filter(
    histories: dict[str, list[Bar]], cfg: StrategyConfig | None = None
) -> bool:
    """大盘择时：沪深300ETF 是否在 MA60 上方。

    为什么需要这一层：回放 2026-05 ~ 10 的半年，不加过滤时最大回撤 -18.6%，
    加了之后大幅收窄。原因是行业 ETF 轮动在熊市里互相接刀 ——
    卖掉一只跌的换成另一只"相对强"的，但整体都在跌。

    用 510300 而不是自己合成指数：它是流动性最好的宽基 ETF，
    不会停牌、不会清盘，且自 2005 年以来一直存在，历史足够长。

    数据不足时返回 True（不干预），而不是 False ——
    启动初期就把自己锁死在不新开仓的状态，比漏跑一段行情更坑。
    """
    cfg = cfg or StrategyConfig()
    bars = histories.get("510300") or []
    if len(bars) < cfg.slow:
        return True
    closes = [b.close for b in bars]
    ma = sma(closes, cfg.slow)
    return ma is not None and closes[-1] > ma


def decide(
    histories: dict[str, list[Bar]],
    cfg: StrategyConfig | None = None,
    held: set[str] | None = None,
) -> Decision:
    """根据截止最后一根 K 线的历史，产出目标权重。

    held 是当前持仓的代码集合。它必须传进来，不能靠 infer，因为
    "已持有"和"新开仓"的判定阈值不同 —— 这正是滞回机制的落点。
    """
    cfg = cfg or StrategyConfig()
    held = set(held or ())
    rows: list[Candidate] = []

    for symbol, name in UNIVERSE:
        bars = histories.get(symbol) or []
        if len(bars) < cfg.warmup:
            rows.append(
                Candidate(
                    symbol=symbol,
                    name=name,
                    close=Decimal("0"),
                    ma_fast=Decimal("0"),
                    ma_slow=Decimal("0"),
                    mom=Decimal("0"),
                    passed=False,
                    reason=f"历史不足（{len(bars)}/{cfg.warmup} 根）",
                )
            )
            continue

        closes = [b.close for b in bars]
        ma_f = sma(closes, cfg.fast)
        ma_s = sma(closes, cfg.slow)
        mom = momentum(bars, cfg.momentum)
        last = closes[-1]

        if ma_f is None or ma_s is None or mom is None:
            rows.append(
                Candidate(
                    symbol, name, last, Decimal("0"), Decimal("0"),
                    Decimal("0"), False, "指标计算失败",
                )
            )
            continue

        above_fast = last > ma_f
        trend_up = ma_f > ma_s

        if symbol in held:
            # 已持有：只要没跌破长期均线就继续拿。
            # 这是滞回的另一半 —— 买入要求"站上 MA20 且均线多头"，
            # 持有只要求"没跌破 MA60"。中间那一段灰色地带就是摩擦缓冲。
            passed = last > ma_s
            reason = (
                "持仓中，仍在 MA60 上方" if passed else "持仓中，已跌破 MA60"
            )
        else:
            passed = above_fast and trend_up
            if passed:
                reason = "站上 MA20 且 MA20 > MA60"
            elif not above_fast and not trend_up:
                reason = "跌破 MA20 且均线空头"
            elif not above_fast:
                reason = "跌破 MA20"
            else:
                reason = "MA20 位于 MA60 下方"

        rows.append(
            Candidate(symbol, name, last, ma_f, ma_s, mom, passed, reason)
        )

    passed_rows = sorted(
        [r for r in rows if r.passed], key=lambda r: r.mom, reverse=True
    )
    for i, r in enumerate(passed_rows, start=1):
        r.rank = i

    picked = passed_rows[: cfg.top_n]
    # 已持有且仍在候选中的，就算排名掉出前 N 也不立刻踢。
    # 不加这条的话会出现"今天排第3卖掉、明天排第2买回来"，
    # 每来回一次就是四笔佣金（双边买卖 × 新旧标的）。
    # 上限 top_n + 1，避免持仓无限累积。
    for r in passed_rows[cfg.top_n :]:
        if r.symbol in held and len(picked) < cfg.top_n + 1:
            picked.append(r)

    targets: dict[str, Decimal] = {}
    ok = market_filter(histories, cfg)

    # 大盘过滤只拦"新开仓"，不拦"继续持有"。
    # 区别很重要：熊市里已经持有的标的可能还在走强，
    # 一刀切全部清掉等于放弃了仅剩的 alpha 来源。
    if not ok:
        picked = [r for r in picked if r.symbol in held]

    if picked:
        weight = min(
            Decimal("1") / Decimal(len(picked)), cfg.max_weight
        )
        targets = {r.symbol: weight for r in picked}

    as_of = ""
    for bars in histories.values():
        if bars:
            as_of = bars[-1].dt.isoformat()
            break

    return Decision(as_of=as_of, targets=targets, candidates=rows, market_ok=ok)
