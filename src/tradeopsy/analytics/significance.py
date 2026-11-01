"""PGR/PLR 的统计显著性：bootstrap 置信区间 + 双比例检验。

为什么需要这一步：
    PGR=25% / PLR=50% 看起来"有处置效应"，但如果只有 4 笔交易，
    这个差异完全可能是随机波动。不做显著性，就是在用噪声下结论 ——
    这是个人量化项目最常见的"伪严谨"。

口径说明（重要）：
    Odean (1998) 原始 PGR/PLR 是**计数口径**（笔数之比），不是金额口径。
    本模块的显著性检验采用计数口径 —— 因为"比例"才是统计上自然的对象，
    金额口径是重尾分布，bootstrap 不稳定。
    （卡片上并列展示的金额口径 PGR/PLR 是另一种常用变体，两者含义不同。）

方法：
    1. 把每个"机会"抽象成一个 0/1 观察：
         gain 机会：卖掉了(1) / 还拿着(0)
         loss 机会：割肉了(1) / 还扛着(0)
    2. PGR = mean(gain flags)，PLR = mean(loss flags)，DE = PGR - PLR
       （Odean 口径：DE 为 PGR 减 PLR，正值表示处置效应）
    3. bootstrap：对 gain、loss 两组各自有放回重抽样，重算 DE，得到分布
       -> 95% 百分位置信区间 + 符号检验 p 值
    4. 双比例 z 检验作为经典方法的交叉验证（近似，不严格独立）

假设与局限（诚实交代）：
    - 把每笔交易/持仓当作独立同分布观察。现实中同一标的的分批买卖相关，
      这会低估标准误、让 p 值偏乐观。所以结论要保守看。
    - 样本很小（每侧 <30）时，bootstrap 区间会很宽，通常得不出显著结论 ——
      这是正确的结果，不是 bug。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..model import RealizedChunk
from .disposition import OpenPositionMV

#: 每侧（盈利机会 / 亏损机会）少于这个数，直接判定"证据不足"。
MIN_OBS = 5
#: 每侧少于这个数，即使算出来也标注"样本偏小，结论仅供参考"。
WEAK_OBS = 30


@dataclass
class SignificanceResult:
    n_gain_sold: int          # 已实现盈利笔数
    n_gain_held: int          # 账面盈利（未平仓浮盈）笔数
    n_loss_sold: int          # 已实现亏损笔数
    n_loss_held: int          # 账面亏损（未平仓浮亏）笔数
    pgr_count: Optional[float]
    plr_count: Optional[float]
    de_count: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value_boot: Optional[float]
    p_value_z: Optional[float]
    n_boot: int
    seed: int
    verdict: str
    notes: list[str] = field(default_factory=list)

    @property
    def n_gain_total(self) -> int:
        return self.n_gain_sold + self.n_gain_held

    @property
    def n_loss_total(self) -> int:
        return self.n_loss_sold + self.n_loss_held

    @property
    def sufficient(self) -> bool:
        return self.n_gain_total >= MIN_OBS and self.n_loss_total >= MIN_OBS


def _percentile(sorted_vals: list[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_prop_p(p1: float, n1: int, p2: float, n2: int) -> Optional[float]:
    """双比例 z 检验（近似）：比较 PGR(p1,n1) 与 PLR(p2,n2) 是否不同。"""
    if n1 <= 0 or n2 <= 0:
        return None
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se <= 0:
        return None
    z = (p2 - p1) / se
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


@dataclass
class DeInterval:
    """两组比例的 DE 区间估计（从原始计数直接构造，供滚动口径复用）。"""

    pgr: Optional[float]
    plr: Optional[float]
    de: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value: Optional[float]
    n_boot: int
    seed: int
    n_gain_total: int
    n_loss_total: int

    @property
    def excludes_zero(self) -> bool:
        return (
            self.ci_low is not None
            and self.ci_high is not None
            and (self.ci_low > 0 or self.ci_high < 0)
        )


def _boot_de(
    gain_flags: list[int],
    loss_flags: list[int],
    n_boot: int,
    seed: int,
    ci_level: float,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """两组 0/1 标志的有放回重抽样，返回 (pgr, plr, de, ci_low, ci_high, p)。

    这是 PGR/PLR 显著性检验的唯一实现 —— 期末快照口径与逐日滚动口径
    都走这里，避免两套口径各自实现导致结论不一致。
    """
    n_gain, n_loss = len(gain_flags), len(loss_flags)
    if n_gain <= 0 or n_loss <= 0:
        return None, None, None, None, None, None

    pgr = sum(gain_flags) / n_gain
    plr = sum(loss_flags) / n_loss

    rng = random.Random(seed)
    de_star: list[float] = []
    for _ in range(n_boot):
        pgr_star = sum(rng.choice(gain_flags) for _ in range(n_gain)) / n_gain
        plr_star = sum(rng.choice(loss_flags) for _ in range(n_loss)) / n_loss
        de_star.append(pgr_star - plr_star)
    de_star.sort()

    half = (1.0 - ci_level) / 2.0
    ci_low = _percentile(de_star, half * 100.0)
    ci_high = _percentile(de_star, (1.0 - half) * 100.0)

    le0 = sum(1 for d in de_star if d <= 0)
    ge0 = sum(1 for d in de_star if d >= 0)
    p_boot = min(1.0, max(2.0 * min(le0, ge0) / n_boot, 1.0 / n_boot))

    return pgr, plr, pgr - plr, ci_low, ci_high, p_boot


def de_interval(
    gain_hits: int,
    gain_total: int,
    loss_hits: int,
    loss_total: int,
    *,
    n_boot: int = 5000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> DeInterval:
    """直接由四项计数得到 DE 的区间估计。

    逐日滚动口径天然产出 pooled 计数（Σ已卖 / Σ机会），不需要逐笔原始记录，
    所以用这个入口；统计核心与 bootstrap_disposition 完全一致。
    """
    if gain_total <= 0 or loss_total <= 0:
        return DeInterval(None, None, None, None, None, None,
                          n_boot, seed, gain_total, loss_total)

    gain_hits = max(0, min(gain_hits, gain_total))
    loss_hits = max(0, min(loss_hits, loss_total))
    gain_flags = [1] * gain_hits + [0] * (gain_total - gain_hits)
    loss_flags = [1] * loss_hits + [0] * (loss_total - loss_hits)

    pgr, plr, de, ci_low, ci_high, p_boot = _boot_de(
        gain_flags, loss_flags, n_boot, seed, ci_level
    )
    return DeInterval(pgr, plr, de, ci_low, ci_high, p_boot,
                      n_boot, seed, gain_total, loss_total)


def bootstrap_disposition(
    chunks: Iterable[RealizedChunk],
    open_positions: Iterable[OpenPositionMV],
    *,
    n_boot: int = 5000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> SignificanceResult:
    """对处置效应 DE 做 bootstrap 显著性检验（计数口径）。

    chunks / open_positions 必须是同一份账本的已实现与未平仓两端。
    返回 SignificanceResult；样本不足时不强行给 p 值，而是明确判"证据不足"。
    """
    chunks = list(chunks)
    open_positions = list(open_positions)

    g_sold = sum(1 for c in chunks if c.pnl > 0)
    g_held = sum(1 for o in open_positions if o.paper_gain > 0)
    l_sold = sum(1 for c in chunks if c.pnl < 0)
    l_held = sum(1 for o in open_positions if o.paper_gain < 0)

    n_gain = g_sold + g_held
    n_loss = l_sold + l_held

    pgr = g_sold / n_gain if n_gain > 0 else None
    plr = l_sold / n_loss if n_loss > 0 else None
    de = (pgr - plr) if (pgr is not None and plr is not None) else None

    notes: list[str] = []

    # ---- 缺一端：无法定义 DE ----
    if n_gain == 0 or n_loss == 0:
        notes.append("缺少盈利或亏损机会（某侧为 0），DE 无法定义。")
        return SignificanceResult(
            g_sold, g_held, l_sold, l_held, pgr, plr, de,
            None, None, None, None, n_boot, seed,
            "证据不足：盈利或亏损机会其中之一为 0，无法判定处置效应。", notes,
        )

    # ---- 样本不足：不强行算 p ----
    if not (n_gain >= MIN_OBS and n_loss >= MIN_OBS):
        notes.append(
            f"样本不足：盈利机会 {n_gain} 个、亏损机会 {n_loss} 个，"
            f"每侧至少需 {MIN_OBS} 个才能检验。"
        )
        return SignificanceResult(
            g_sold, g_held, l_sold, l_held, pgr, plr, de,
            None, None, None, None, n_boot, seed,
            "证据不足：样本量过小，不能断言存在处置效应。", notes,
        )

    # ---- bootstrap（与 de_interval 共用同一实现，口径不会分叉）----
    gain_flags = [1] * g_sold + [0] * g_held
    loss_flags = [1] * l_sold + [0] * l_held
    _b_pgr, _b_plr, _b_de, ci_low, ci_high, p_boot = _boot_de(
        gain_flags, loss_flags, n_boot, seed, ci_level
    )

    p_z = _two_prop_p(pgr, n_gain, plr, n_loss)

    # ---- 结论 ----
    excludes_zero = (ci_low is not None and ci_high is not None and (ci_low > 0 or ci_high < 0))
    if n_gain < WEAK_OBS or n_loss < WEAK_OBS:
        notes.append(
            f"样本偏小（盈利 {n_gain} / 亏损 {n_loss}），区间较宽，结论仅供参考。"
        )
    notes.append("假设各机会独立；分批买卖存在相关性，p 值可能偏乐观，结论应保守看待。")

    if de is None:
        verdict = "无法判定。"
    elif excludes_zero and p_boot < 0.05:
        verdict = (
            "统计显著：存在处置效应（卖盈持亏）。" if de > 0
            else "统计显著：存在反向处置效应（让盈利跑、及时止损）。"
        )
    else:
        verdict = "未达统计显著（95% 置信区间含 0），不能断言存在处置效应。"

    return SignificanceResult(
        g_sold, g_held, l_sold, l_held, pgr, plr, de,
        ci_low, ci_high, p_boot, p_z, n_boot, seed, verdict, notes,
    )
