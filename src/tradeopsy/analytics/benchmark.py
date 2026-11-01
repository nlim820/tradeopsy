"""同行基准对照：给 DE 一个外部坐标系。

为什么会有这个模块，以及它**不做**什么
--------------------------------------
最初的设想是算「你在人群里排第几百分位」。做到一半发现这在数据上不成立：
分位数需要个体层面的横截面分布（每个账户各自的 PGR/PLR 长什么样），
而已核实的文献只公布群体点估计 —— 既没有标准差，也没有分位点。

缺失分布时，任何百分位都必须**假设**一个分布形态。PGR/PLR 有界于 [0,1]
且明显右偏，套正态假设会算出「你在第 78 百分位」这种看起来很精确的假数字。
所以本模块**不输出分位数**，改为区间包含检验：

    文献点估计 是否落在 我的 DE 的 bootstrap 置信区间 内？

这能回答「我和那个人群有没有可分辨的差异」，但回答不了「我排第几」。
两件事不一样，混为一谈就是伪精确。

口径约束（硬拦截，不是提示）
----------------------------
已核实的文献全部采用 Odean (1998) 的「每个卖出日」计数口径。
本项目只有 `--price-history` 的逐日滚动口径与之同构；
期末快照口径的分母是「今天还没卖的」，与文献分母不是同一个东西，
两者对照会得出无意义的结论。因此这里直接拒绝，而不是给个温柔警告。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Optional

#: 比值口径在 PLR 低于此值时视为不稳定（分母趋零，比值会爆炸）。
RATIO_FLOOR = 0.05
#: 置信区间宽度超过文献效应量的这个倍数时，判定检验「几乎没有分辨力」。
POWER_RATIO_LIMIT = 4.0


@dataclass
class Benchmark:
    """一条已核实的文献基准。"""

    id: str
    market: str
    source: str
    sample_period: str
    sample_size: str
    pgr: Optional[float]
    plr: Optional[float]
    de: Optional[float]
    ratio: Optional[float]
    quote: str
    verified: bool
    verified_note: str
    caveat: str = ""
    ratio_note: str = ""

    @property
    def short_source(self) -> str:
        """去掉期刊卷期后的简短出处，表格里用。"""
        head = self.source.split(".")[0]
        return head.strip()

    @property
    def short_market(self) -> str:
        """图表标签用的极短名（括号里的补充说明去掉）。"""
        return self.market.split("（")[0]


@dataclass
class Comparison:
    """一条基准与本人的比对结果。"""

    bench: Benchmark
    metric: str                      # "de"（百分点差）或 "ratio"（PGR/PLR）
    theirs: Optional[float]
    mine: Optional[float]
    delta: Optional[float]
    verdict: str                     # indistinguishable / weaker / stronger / unavailable
    text: str


@dataclass
class BenchmarkReport:
    caliber: str                     # "rolling" / "snapshot" / "none"
    comparable: bool
    blocked_reason: Optional[str]
    comparisons: list[Comparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    power_ratio: Optional[float] = None
    ci_width: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    my_de: Optional[float] = None
    my_n_obs: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def de_comparisons(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.metric == "de"]

    @property
    def ratio_comparisons(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.metric == "ratio"]


# ---------------------------------------------------------------- 数据加载


def load_benchmarks() -> tuple[list[Benchmark], dict]:
    """读取随包分发的 benchmarks.json。

    返回 (基准列表, 元信息)。元信息含口径说明、已撤下条目、分位数立场。
    """
    raw = resources.files("tradeopsy.data").joinpath("benchmarks.json").read_text(
        encoding="utf-8"
    )
    doc = json.loads(raw)

    benchmarks: list[Benchmark] = []
    for item in doc.get("benchmarks", []):
        benchmarks.append(
            Benchmark(
                id=item["id"],
                market=item["market"],
                source=item["source"],
                sample_period=item.get("sample_period", ""),
                sample_size=item.get("sample_size", ""),
                pgr=item.get("pgr"),
                plr=item.get("plr"),
                de=item.get("de"),
                ratio=item.get("ratio"),
                quote=item.get("quote", ""),
                verified=bool(item.get("verified", False)),
                verified_note=item.get("verified_note", ""),
                caveat=item.get("caveat", ""),
                ratio_note=item.get("ratio_note", ""),
            )
        )

    meta = {
        "caliber_note": doc.get("caliber_note", ""),
        "retracted": doc.get("retracted", []),
        "quantile_stance": doc.get("quantile_stance", {}),
    }
    return benchmarks, meta


# ---------------------------------------------------------------- 对照


def _judge_de(
    theirs: float, ci_low: Optional[float], ci_high: Optional[float]
) -> tuple[str, str]:
    """判断文献 DE 落在我的置信区间哪一侧。"""
    if ci_low is None or ci_high is None:
        return "unavailable", "没有置信区间（样本不足），无法判断与文献的距离。"
    if ci_low <= theirs <= ci_high:
        return (
            "indistinguishable",
            "文献值落在你的置信区间内 —— 含义是「样本不足以分辨」，"
            "不是「你和人群一样」。",
        )
    if theirs > ci_high:
        return (
            "weaker",
            "文献值高于你的区间上界：你的处置效应比该人群更弱。",
        )
    return (
        "stronger",
        "文献值低于你的区间下界：你的处置效应比该人群更强。",
    )


def compare_to_benchmarks(
    *,
    caliber: str,
    my_de: Optional[float] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    my_pgr: Optional[float] = None,
    my_plr: Optional[float] = None,
    my_n_obs: int = 0,
    benchmarks: Optional[list[Benchmark]] = None,
    meta: Optional[dict] = None,
) -> BenchmarkReport:
    """把本人的处置效应对照到已发表的同行基准上。

    参数
    ----
    caliber : "rolling" / "snapshot" / "none"
        当前采用的口径。只有 "rolling" 才允许对照（见模块文档）。
    my_de, ci_low, ci_high
        本人的 DE 与 bootstrap 置信区间，用于区间包含检验。
    my_pgr, my_plr
        本人的 PGR/PLR，用于比值口径的描述性对照（不做显著性）。
    """
    if benchmarks is None or meta is None:
        loaded, loaded_meta = load_benchmarks()
        benchmarks = benchmarks if benchmarks is not None else loaded
        meta = meta if meta is not None else loaded_meta

    notes: list[str] = []

    # ---- 口径硬拦截
    if caliber != "rolling":
        if caliber == "snapshot":
            reason = (
                "当前用的是期末快照口径：分母只统计「今天还没卖的」持仓。"
                "已核实的文献（Odean 1998 及后续）分母是「每个卖出日的全部持仓」，"
                "两者分母定义不同 —— 直接对照会得出无意义的结论，因此不做。"
                "要看对照请加上 --price-history 走逐日滚动口径。"
            )
        elif caliber == "none":
            reason = (
                "没有价格序列，无法构造账面端，也就没有可比的口径。"
                "请提供 --price-history（日期,代码,收盘价）。"
            )
        elif caliber == "insufficient":
            reason = (
                "有价格序列，但没有任何一个卖出日能同时凑齐「已实现」与「账面」两端，"
                "算不出可对照的 DE。请检查价格序列是否覆盖了你的持仓区间。"
            )
        else:
            reason = f"未知口径 {caliber!r}，不做对照。"
        return BenchmarkReport(caliber=caliber, comparable=False,
                               blocked_reason=reason, notes=notes)

    report = BenchmarkReport(caliber=caliber, comparable=True,
                             blocked_reason=None, my_n_obs=my_n_obs,
                             ci_low=ci_low, ci_high=ci_high, my_de=my_de)

    my_ratio = None
    if my_pgr is not None and my_plr is not None and my_plr > 0:
        my_ratio = my_pgr / my_plr

    for bench in benchmarks:
        # --- DE 口径（百分点差）：能算的才比
        if bench.de is not None:
            mine = my_de
            delta = (mine - bench.de) if mine is not None else None
            verdict, text = _judge_de(bench.de, ci_low, ci_high)
            report.comparisons.append(
                Comparison(bench, "de", bench.de, mine, delta, verdict, text)
            )

        # --- 比值口径：纯描述，不做显著性
        if bench.ratio is not None:
            usable = my_ratio is not None
            if not usable:
                verdict = "unavailable"
                text = "你的 PLR 为 0 或缺失，比值无定义。"
            elif my_plr is not None and my_plr < RATIO_FLOOR:
                verdict = "unstable"
                text = (
                    f"你的 PLR 仅 {my_plr:.1%}，比值分母过小 —— "
                    "此时 PGR/PLR 对个别交易极敏感，不宜当作稳定指标。"
                )
            else:
                verdict = "descriptive"
                text = "比值口径仅作量级参照，未做显著性检验。"
            report.comparisons.append(
                Comparison(bench, "ratio", bench.ratio, my_ratio,
                           (my_ratio - bench.ratio) if usable else None,
                           verdict, text)
            )

    # ---- 统计功效：把「几乎没有分辨力」量化出来
    if ci_low is not None and ci_high is not None:
        report.ci_width = ci_high - ci_low
        ref = next((b.de for b in benchmarks if b.de is not None), None)
        if ref:
            report.power_ratio = report.ci_width / abs(ref)
            if report.power_ratio >= POWER_RATIO_LIMIT:
                notes.append(
                    f"统计功效提示：你的置信区间宽 {report.ci_width:.1%}，"
                    f"是文献效应量 {ref:.1%} 的 {report.power_ratio:.0f} 倍。"
                    "在这个宽度下，「区间包含文献值」几乎是必然结果，"
                    "不能解读为「我和人群一致」。"
                )
    if my_n_obs:
        notes.append(
            f"样本对比：你 {my_n_obs} 个卖出日 vs 文献的上万账户 —— "
            "样本量差 3~4 个数量级。这类对照只能看量级是否离谱，"
            "不能用来给自己排名。"
        )

    report.meta = meta or {}
    quantile = report.meta.get("quantile_stance", {})
    if quantile and not quantile.get("supported", True):
        notes.append("本工具不输出百分位排名：" + quantile.get("why", ""))

    # 构造时用的是 dataclass 默认值（另一个 list 对象），这里必须显式挂回
    report.notes = notes
    return report
