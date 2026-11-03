"""单文件 HTML 体检报告：把数字变成一份能发出去的东西。

设计约束（全部来自真实使用场景，不是偏好）：

1. **单文件、零外部依赖**。不引 CDN、不引字体、不用 JS 框架。
   原因很实际：报告里有你的持仓和金额，最安全的分享方式是本地打开；
   任何一次外链请求都可能把"打开过这份报告"这件事泄露出去。
   所以 CSS 内联、图表手写 SVG、交互只用原生 `<details>`。

2. **每个数字都能追溯**。所有指标卡片下挂一个可展开的「依据」，
   把参与计算的原值、笔数、公式摆出来。不可追溯的漂亮数字等于没有。

3. **口径写死在同一页**。交易日/自然日、金额口径/计数口径、
   缺失了什么数据 —— 全部印在报告里，而不是留在文档里。
   报告一旦被转发就会脱离上下文，口径必须自带。

4. **不假装有数据**。基准对比需要外部行情，没有就明确说没有、为什么、怎么补，
   绝不画一条看起来像样的假曲线。

配色遵循 A 股惯例：**涨红跌绿**（与欧美相反）。这条写死在 CSS 变量里，
因为默认套用国际配色是这类工具最常见的"一看就不专业"的地方。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from html import escape as _escape

from .analytics.benchmark import BenchmarkReport
from .analytics.disposition import DispositionReport
from .analytics.rolling import DailyObs, RollingReport
from .analytics.significance import SignificanceResult
from .load import Dataset
from .model import RealizedChunk, money
from .prices import PriceHistory

__all__ = ["build_report", "persona_tags", "one_liner", "hold_bins"]


# ---------------------------------------------------------------- 小工具

def _esc(s: object) -> str:
    return _escape(str(s), quote=True)


def _money(x: Decimal | float) -> str:
    v = float(x)
    sign = "-" if v < 0 else ""
    return f"{sign}¥{abs(v):,.2f}"


def _money_short(x: Decimal | float) -> str:
    """大额缩写，用于 KPI 卡片（避免数字太长挤爆手机屏幕）。"""
    v = float(x)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 10000:
        return f"{sign}¥{a / 10000:,.2f}万"
    return f"{sign}¥{a:,.2f}"


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"


def _pct_signed(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.{digits}f}%"


def _tone(x: float) -> str:
    """中国股市惯例：正数红、负数绿、零中性。"""
    if x > 0:
        return "up"
    if x < 0:
        return "down"
    return "flat"


def _sig_ok(sig: SignificanceResult | None, alpha: float = 0.05) -> bool:
    """是否"统计显著"。三个条件缺一不可，缺一个就退回"证据不足"。

    为什么要求 sufficient：样本量不足时 bootstrap 也能吐出一个小 p 值，
    那是数学假象而非发现。先卡样本量，再看 p，顺序不能反。
    """
    if sig is None or sig.de_count is None or sig.p_value_boot is None:
        return False
    return sig.sufficient and sig.p_value_boot < alpha


# ---------------------------------------------------------------- 数据加工

def equity_curve(chunks: list[RealizedChunk]) -> list[tuple[date, Decimal]]:
    """按平仓日累计已实现盈亏（阶梯曲线，同日多笔合并）。

    用「累计」而不是「每日盈亏」，因为单日盈亏在散户账户上噪声极大，
    看图只会看到一堆毛刺；累计曲线才能回答"我这一年到底在往上还是往下"。
    """
    agg: dict[date, Decimal] = {}
    for c in chunks:
        agg[c.close_dt] = agg.get(c.close_dt, Decimal("0")) + c.pnl
    out: list[tuple[date, Decimal]] = []
    total = Decimal("0")
    for d in sorted(agg):
        total += agg[d]
        out.append((d, total))
    return out


#: 持仓天数分箱边界（交易日口径）。选这些边界是因为它们在 A 股语境里有含义：
#: 3 天内 ≈ 短线博弈；1 个月内 ≈ 波段；3 个月以上 ≈ 忘账户了。
BIN_EDGES = (0, 3, 7, 15, 30, 60, 120, 250, 10**9)
BIN_LABELS = ("0–3", "4–7", "8–15", "16–30", "31–60", "61–120", "121–250", "250+")


def hold_bins(
    chunks: list[RealizedChunk],
    calendar=None,
) -> list[tuple[str, int, int]]:
    """持仓天数分布：返回 [(区间标签, 赢家笔数, 输家笔数)]。

    平手（pnl == 0）单独丢弃 —— 它既不是赢家也不是输家，
    塞进任意一边都会让"赢家普遍拿得短"这类结论失真。
    """
    win = [0] * len(BIN_LABELS)
    loss = [0] * len(BIN_LABELS)
    for c in chunks:
        if c.pnl == 0:
            continue
        days = calendar.count(c.open_dt, c.close_dt) if calendar is not None else c.hold_days
        for i in range(len(BIN_LABELS)):
            if BIN_EDGES[i] <= days <= BIN_EDGES[i + 1]:
                (win if c.pnl > 0 else loss)[i] += 1
                break
        else:
            # 超过最大边界（正常不会发生，因为最后一段是 1e9）
            (win if c.pnl > 0 else loss)[-1] += 1
    return [(BIN_LABELS[i], win[i], loss[i]) for i in range(len(BIN_LABELS))]


@dataclass
class Tag:
    name: str
    strength: float      # 0–1，用于排序取前几条
    desc: str
    evidence: str        # 可展开的依据说明


#: 「死扛型」的判定阈值：输家平均持有期 ÷ 赢家平均持有期。
#: 提成常量的原因：同一份数据在 persona_tags（画像标签）和 one_liner
#: （一句话结论）里都要判它。两处阈值一旦分叉，就会出现
#: 「标签说死扛型、结论说没有明显偏向」这种同一屏自相矛盾的输出 ——
#: 这不是排版问题，是读者会直接质疑报告可信度的正确性问题。
_HOLD_RATIO = 1.8


def persona_tags(
    rep: DispositionReport,
    sig: SignificanceResult | None,
    summary: dict,
    chunks: list[RealizedChunk],
    events_total: int,
    span_days: int,
) -> list[Tag]:
    """交易人格判定 —— 只用交割单能自证的数据。

    刻意没做的标签（因为需要行情，硬做就是编）：
      · 追高型：需要"买入价相对 20 日高点分位" → 要 K 线
      · 追涨杀跌：需要买入当日的涨跌 → 要 K 线
    这类标签的失效方式很危险：没有行情时如果拿 0 或拿成本价代替，
    会算出一个看似正常、实则毫无意义的百分位。所以宁可不给。
    """
    tags: list[Tag] = []
    unit = "交易日" if rep.hold_days_trading else "自然日"
    # notes 的顺序随模式变化（realized_only 会多一条），按下标取会取错。
    # 这里按语义找：日历来源那条一定以「持有天数按」开头。
    cal_line = next((n for n in rep.notes if n.startswith("持有天数按")), "")

    # 1) 死扛型：输家拿的时间远超赢家
    if rep.avg_hold_win > 0 and rep.avg_hold_loss > 0:
        ratio = rep.avg_hold_loss / rep.avg_hold_win
        if ratio >= _HOLD_RATIO:
            tags.append(Tag(
                name="死扛型",
                strength=min(ratio / 5, 1.0),
                desc=f"输家平均拿 {rep.avg_hold_loss} {unit}，赢家只拿 {rep.avg_hold_win} {unit}。",
                evidence=(
                    f"亏损笔平均持有 / 盈利笔平均持有 = {rep.avg_hold_loss} / "
                    f"{rep.avg_hold_win} = {ratio:.2f} 倍。判定阈值 ≥{_HOLD_RATIO} 倍。\n"
                    # cal_line 是内部生成的日历说明（非用户数据）；
                    # evidence 由 build_report 统一转义，这里再转一次会变成 &lt;br&gt;
                    f"{cal_line}"
                ),
            ))

    # 2) 落袋型 / 反向自律型：处置效应方向
    de = rep.disposition_effect
    if de is not None and abs(de) > 0.10:
        strong = _sig_ok(sig)
        if de > 0:
            tags.append(Tag(
                name="落袋为安型" if not strong else "典型处置效应",
                strength=min(abs(de) / 0.5, 1.0) + (0.2 if strong else 0),
                desc=f"盈利兑现比例 {_pct(rep.pgr)} 高于亏损兑现比例 {_pct(rep.plr)}。",
                evidence=(
                    f"金额口径 PGR {_pct(rep.pgr)} · PLR {_pct(rep.plr)} · DE {_pct_signed(de)}。"
                    + (f" 计数口径 DE {_pct_signed(sig.de_count)}，"
                       + (f"bootstrap p={sig.p_value_boot:.4f}"
                          f"{'（显著）' if strong else '（不显著，证据不足）'}。"
                          if sig.p_value_boot is not None else
                          "单侧已实现为 0，bootstrap 无法构造，统计上无从谈起。")
                       if sig is not None and sig.de_count is not None else
                       " 未提供未平仓市值，仅有金额口径。")
                ),
            ))
        else:
            tags.append(Tag(
                name="反处置效应 · 少见",
                strength=min(abs(de) / 0.5, 1.0),
                desc=f"亏损兑现比例 {_pct(rep.plr)} 高于盈利兑现比例 {_pct(rep.pgr)}。",
                evidence=(
                    f"DE {_pct_signed(de)} < -10%。这意味着你在做多数人做不到的事："
                    f"让盈利继续跑、及时砍掉亏损。先确认不是样本太小造成的偶然。"
                ),
            ))

    # 3) 高频型：按年均成交笔数判，而不是"换手率"
    years = max(span_days / 365.25, 0.08)
    per_year = events_total / years
    if per_year >= 120:
        tags.append(Tag(
            name="高频交易型",
            strength=min(per_year / 400, 1.0),
            desc=f"平均每年约 {per_year:.0f} 条成交记录。",
            evidence=(
                f"成交 {events_total} 条 ÷ 跨度 {span_days} 天 × 365 ≈ {per_year:.0f} 条/年。"
                f"判定阈值 120 条/年（约每 2 个交易日一笔）。"
                f"注：这里用**笔数**而非换手率 —— 换手率需要持仓市值，"
                f"而交割单不含每日市值，强行算会得到偏高的伪换手率。"
            ),
        ))

    # 4) 集中型：单标的买入成本占比
    buy_cost: dict[str, Decimal] = {}
    for c in chunks:
        buy_cost[c.symbol] = buy_cost.get(c.symbol, Decimal("0")) + c.cost
    total_cost = sum(buy_cost.values(), Decimal("0"))
    if total_cost > 0 and buy_cost:
        top_sym, top_cost = max(buy_cost.items(), key=lambda kv: kv[1])
        share = float(top_cost / total_cost)
        if share >= 0.35:
            tags.append(Tag(
                name="集中持仓型",
                strength=min(share, 1.0),
                desc=f"{top_sym} 占了已实现成本的 {_pct(share, 0)}。",
                evidence=(
                    f"{top_sym} 已实现成本 {_money(top_cost)} ÷ 全部已实现成本 "
                    f"{_money(total_cost)} = {_pct(share)}。判定阈值 ≥35%。"
                    f"注：分母是**已实现**成本，不含未平仓批次 —— 未平仓需要市值才能比较。"
                ),
            ))

    # 5) 小赢大亏型：胜率高但仍是净亏
    win_rate = summary.get("胜率") or 0
    total_win = summary.get("总盈利") or 0
    total_loss = summary.get("总亏损") or 0
    if win_rate >= 0.55 and total_loss > total_win > 0:
        tags.append(Tag(
            name="小赢大亏型",
            strength=min(win_rate, 1.0),
            desc=f"胜率 {_pct(win_rate)} 不低，但总亏损仍大于总盈利。",
            evidence=(
                f"盈利 {_money(total_win)} / 亏损 {_money(total_loss)}，"
                f"盈亏比 {(total_win / total_loss):.2f}。"
                f"典型成因：赢的次数多但每次赚得少，输一次吃掉好几次的利润。"
            ),
        ))

    tags.sort(key=lambda t: -t.strength)
    return tags


def one_liner(rep: DispositionReport, sig: SignificanceResult | None, n_realized: int) -> str:
    """一句话结论 —— 报告里最该被转发的那句话。

    优先级：显著的行为结论 > 方向性结论 > 样本不足的诚实拒绝。
    宁可说"还不能下结论"，也不给一句听起来漂亮但撑不住的判断。
    """
    if n_realized == 0:
        return "这段时间没有已实现的卖出，还看不出行为模式。"
    unit = "交易日" if rep.hold_days_trading else "自然日"
    if _sig_ok(sig) and sig is not None and sig.de_count is not None:
        if sig.de_count > 0:
            return (f"你卖掉的赚钱票平均只拿了 {rep.avg_hold_win} {unit}，"
                    f"亏钱的却扛了 {rep.avg_hold_loss} {unit} —— "
                    f"这是统计显著的处置效应（p={sig.p_value_boot:.3f}）。")
        return (f"你在做反人性的事：亏损实现比例高于盈利实现比例"
                f"（DE {_pct_signed(sig.de_count)}，p={sig.p_value_boot:.3f}）。")
    de = rep.disposition_effect
    # 时间维度是**独立**于 DE 的一套证据：赢家是否跑得比输家快。
    # 它不依赖价格，任何模式下都算得出来 —— 所以不能因为 DE 缺席就当它不存在。
    # 之前这里只看 de，导致「画像标签说死扛型 2.8 倍」和
    # 「一句话结论说没有明显偏向」同时出现在首屏，读者第一眼就会怀疑报告。
    ratio = (rep.avg_hold_loss / rep.avg_hold_win
             if rep.avg_hold_win > 0 and rep.avg_hold_loss > 0 else 0.0)

    if de is not None and abs(de) > 0.10:
        if de > 0:
            return (f"从数据看你有处置效应迹象（DE {_pct_signed(de)}，卖盈持亏），"
                    f"但样本量还不足以证明它不是偶然。")
        return (f"从数据看你有反向迹象（DE {_pct_signed(de)}，卖亏持盈），"
                f"但样本量还不足以证明它不是偶然。")

    if de is None:
        # 缺未平仓市值时严格 DE 算不出来，但「赢家跑得比输家快」这个形状
        # 本身是可观测的方向证据 —— 说「没有明显偏向」就等于丢掉它。
        if ratio >= _HOLD_RATIO:
            return (f"你卖掉的赚钱票平均只拿了 {rep.avg_hold_win} {unit}，"
                    f"亏钱的却扛了 {rep.avg_hold_loss} {unit}（{ratio:.1f} 倍）—— "
                    f"处置效应的典型形状。但缺未平仓市值，严格的 DE 算不出来，"
                    f"这只能算行为迹象，不能算统计结论。")
    elif ratio >= _HOLD_RATIO:
        # 兑现率看不出偏向、持有期却差很多：两个维度不一致，本身就是结论。
        return (f"从兑现率看，盈亏两端差不多（DE {_pct_signed(de)}）；"
                f"但从时间看，赚钱的只拿 {rep.avg_hold_win} {unit}、"
                f"亏钱的扛了 {rep.avg_hold_loss} {unit}（{ratio:.1f} 倍）。"
                f"两个维度不一致 —— 你的「卖盈持亏」更多体现在拖时间，"
                f"而不是体现在卖出的次数上。")

    if n_realized < 5:
        # 1~4 笔时说「没有明显偏向」是假中性 —— 那点观测根本分不出偏向。
        return (f"只有 {n_realized} 笔已实现，样本太少 —— "
                f"现在判断「有没有处置效应」跟瞎猜区别不大，先多攒几个月记录。")

    return (f"已实现 {n_realized} 笔，盈亏两端的兑现节奏没有明显偏向 —— "
            f"样本量也还撑不起更强的结论。")


#: 术语速查。放在「30 秒版」里作为折叠块 —— 一次给全，好过在正文 20 多处
#: 反复插括号解释（那会让懂行的人觉得啰嗦，让不懂的人仍然串联不起来）。
_GLOSSARY = (
    "<b>PGR（盈利兑现比例）</b>：手里赚钱的机会中，你卖掉了多少。"
    "越高 = 越爱落袋为安。<br>"
    "<b>PLR（亏损兑现比例）</b>：手里亏钱的机会中，你割掉了多少。"
    "越低 = 越爱死扛。<br>"
    "<b>DE（处置效应）</b>：PGR 减 PLR。正数 = 卖盈持亏（典型）；负数 = 反向（卖亏持盈）。<br>"
    "<b>金额口径 / 计数口径</b>：前者按钱算，一笔大的能带偏整体；后者按笔算，"
    "一笔一万和一笔一百等价。两个都看，是因为它们有时会在同一批数据上给出相反答案 —— "
    "那本身就是一条信息。<br>"
    "<b>bootstrap 置信区间</b>：把你这批交易反复重抽样上万次，看 DE 会落在什么范围。"
    "区间不跨 0 = 这个方向不是运气。<br>"
    "<b>样本量门槛</b>：任一端机会少于 8 个，程序就拒绝给方向。不是保守 —— "
    "4 个样本算出来的 DE 可以很漂亮，但它是噪声。"
)


def _tldr(
    rep: DispositionReport,
    sig: SignificanceResult | None,
    head_line: str,
    tags: list,
    *,
    n_realized: int,
    n_events: int,
    n_symbols: int,
    span_days: int,
    unit: str,
) -> list[str]:
    """报告顶部「如果你只有 30 秒」。

    为什么单独立这一块，而不是把话说短一点塞进结论里：
        这份报告的默认读者不是量化研究员，而是「愿意花 30 秒判断这东西
        靠不靠谱」的人（老师、面试官、朋友）。他们有判断力，但没有时间
        把一千多行读完。所以这一屏必须自己回答三件事 ——
          ① 它到底在回答什么问题（不知道问题，就无法评价答案）
          ② 答案是什么、有多可信（结论和证据强度一起给，不藏）
          ③ 它不做什么（主动交代边界比多给一个数字更能建立信任）
        下面的专业分析一个字都不删：想深究的人继续往下翻，两拨人各取所需。

    三条依据是**动态**生成的，不是写死的文案 —— 数据不同，说法必须不同；
    数据不足时如实降级，绝不为了版面整齐编一条出来。
    """
    facts: list[str] = []

    # ① 持有时间对比：最直观的一条，不需要任何统计背景就能懂
    hw, hl = rep.avg_hold_win, rep.avg_hold_loss
    if n_realized and hw and hl:
        facts.append(
            f"<b>卖掉的赚钱票平均只拿了 {hw} {unit}，亏钱的却扛了 {hl} {unit}</b>"
            f"（{hl / hw:.1f} 倍）—— 赚了赶紧跑、亏了舍不得割，"
            "这是处置效应最典型的样子。"
        )
    elif n_realized and hw and not hl:
        facts.append(
            f"<b>这期间没有一笔亏损卖出</b>，赚的钱平均拿了 {hw} {unit} 就走。"
            "「死扛亏损」那一端在数据里根本不存在 —— 是没亏过，还是亏了没卖，"
            "本报告分不出来。"
        )

    # ② 兑现率：处置效应的本体
    if rep.has_full:
        facts.append(
            f"<b>盈利时兑现 {_pct(rep.pgr)}，亏损时只兑现 {_pct(rep.plr)}</b>，"
            f"差 {_pct_signed(rep.disposition_effect)}（这个差值就是 DE）。"
        )
    else:
        facts.append(
            "<b>没给未平仓市值，兑现率算不出来</b> —— PGR/PLR 的分母要"
            "「已实现 + 账面」两端，缺账面端硬算会退化成 1.0，"
            "数学成立但完全误导。下面只展示已实现画像。"
            "补法：<code>--prices</code> 给一份「代码,最新价」的 CSV。"
        )

    # ③ 可信度：诚实地告诉读者这个结论能承重多少
    if sig is not None and sig.p_value_boot is not None:
        ok = _sig_ok(sig)
        facts.append(
            f"<b>{n_realized} 笔已实现，bootstrap p={sig.p_value_boot:.3f}，"
            f"{'撑得住统计检验' if ok else '还撑不住统计检验'}</b>"
            + (f"（95% 置信区间 {_pct_signed(sig.ci_low)} ~ "
               f"{_pct_signed(sig.ci_high)}）"
               if sig.ci_low is not None and sig.ci_high is not None else "")
            + "。"
            + ("" if ok else "方向可以参考，但别当成定论。")
        )
    elif sig is not None:
        facts.append(
            f"<b>{n_realized} 笔已实现，但盈亏两端样本不均衡"
            f"（盈利机会 {sig.n_gain_total} / 亏损机会 {sig.n_loss_total}）</b>，"
            "bootstrap 无法构造 —— 程序因此拒绝给方向。"
        )
    else:
        facts.append(
            f"<b>{n_realized} 笔已实现</b>，但没有给未平仓市值，"
            "所以没做显著性检验 —— 下面的结论只能当描述，不能当统计推断。"
        )

    out = [
        '<section class="tldr">',
        '<div class="tldr-eyebrow">如果你只有 30 秒</div>',
        '<p class="tldr-big">这份报告只回答一个问题：<br>'
        "你是不是「赚一点就跑、亏了死扛」？</p>",
        '<p class="tldr-explain">这在行为金融里叫<b>处置效应</b>'
        "（Disposition Effect）：卖掉的赚钱机会多于卖掉的亏损机会，就是有。"
        "它被公认是散户长期跑不赢宽基指数的原因之一 —— "
        "因为它的实际效果是<b>截断盈利、放大亏损</b>，"
        "赢的时候赚一点，输的时候输很多。"
        f"你的样本：{n_events} 条交割记录 / {n_symbols} 只标的 / "
        f"{span_days} 天。</p>",
        '<div class="tldr-divider"></div>',
        '<div class="tldr-eyebrow">你的答案</div>',
        f'<p class="tldr-answer">{_esc(head_line)}</p>',
    ]
    # 画像标签跟着结论走：它们是对同一件事的另外几个侧面，
    # 拆到单独的卡片里会让读者以为是两份不同的判断。
    tag_html = []
    for i, t in enumerate(tags[:3]):
        rev = " rev" if "反处置" in t.name else ""
        cls = f"tag primary{rev}" if i == 0 and t.strength >= 0.25 else "tag"
        tag_html.append(
            f'<span class="{cls}" title="{_esc(t.desc)}">{_esc(t.name)}</span>'
        )
    out.append('<div class="tags">' + "".join(
        tag_html or ['<span class="tag">尚无足够数据形成画像</span>']
    ) + "</div>")
    for t in tags[:3]:
        out.append(
            f'<details><summary>{_esc(t.name)} · {_esc(t.desc)}</summary>'
            f'<div class="body pre">{_esc(t.evidence)}</div></details>'
        )
    out.append('<ul class="tldr-facts">')
    out.extend(f"<li>{f}</li>" for f in facts)
    out.append("</ul>")
    out.append(_details("看不懂 PGR / PLR / DE / 置信区间？展开看这 6 条", _GLOSSARY))
    out.append(
        '<p class="tldr-limit"><b>它不做什么：</b>不预测收益、不推荐标的、不评价未来。'
        "它只把<b>已经发生</b>的交易行为量化 —— 数据是历史，结论是倾向，不是预言。"
        "完整的边界说明见下方「这个报告不做什么」。</p>"
    )
    out.append("</section>")
    return out


# ---------------------------------------------------------------- SVG 图表
#
# 为什么手写 SVG 而不用图表库：
#   报告要能离线双击打开、且不产生任何外部请求（持仓数据不外泄）。
#   引一个 CDN 图表库就同时破坏这两条。而这里只需要两种图，
#   手写的代码量远小于"引入 + 配置 + 处理加载失败"的成本。

_W, _H = 800, 260
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 68, 18, 18, 36


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def _nice_bound(v: float) -> float:
    """把上界抬到"好看"的数值，避免 y 轴出现 23147.83 这种刻度。"""
    if v == 0:
        return 1.0
    a = abs(v)
    mag = 10 ** (len(str(int(a))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if a <= mag * m:
            return mag * m * (1 if v > 0 else -1)
    return mag * 10 * (1 if v > 0 else -1)


def svg_curve(points: list[tuple[date, Decimal]], height: int = _H) -> str:
    """累计已实现盈亏曲线（带面积填充 + 零轴）。

    单点、全零、全负这些退化情形都要能画 —— 报告不能因为数据难看就崩掉。
    """
    if not points:
        return '<p class="empty">没有已实现的卖出，无法绘制收益曲线。</p>'

    vals = [float(v) for _, v in points]
    lo = min(0.0, min(vals))
    hi = max(0.0, max(vals))
    if hi == lo:
        hi, lo = 1.0, -1.0
    top = _nice_bound(hi if hi > 0 else 1.0)
    bot = _nice_bound(lo) if lo < 0 else 0.0
    if top == bot:
        top = bot + 1

    n = len(points)
    inner_w = _W - _PAD_L - _PAD_R
    inner_h = height - _PAD_T - _PAD_B

    def x(i: int) -> float:
        return _PAD_L + (inner_w * i / (n - 1) if n > 1 else inner_w / 2)

    def y(v: float) -> float:
        return _PAD_T + (top - v) / (top - bot) * inner_h

    direction = "curve-up" if vals[-1] >= 0 else "curve-down"
    parts = [f'<svg viewBox="0 0 {_W} {height}" class="chart {direction}" role="img" '
             f'aria-label="累计已实现盈亏曲线，共 {n} 个卖出日，'
             f'最终 {_money(points[-1][1])}">']

    for t in _ticks(bot, top, 4):
        yy = y(t)
        dash = ' stroke-dasharray="3 3"' if abs(t) > 1e-9 else ' stroke-width="1.4"'
        parts.append(f'<line x1="{_PAD_L}" y1="{yy:.1f}" x2="{_W - _PAD_R}" y2="{yy:.1f}" '
                     f'class="grid"{dash}/>')
        parts.append(f'<text x="{_PAD_L - 8}" y="{yy + 4:.1f}" class="ytick" '
                     f'text-anchor="end">{_money_short(t)}</text>')

    coords = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    base = y(0)
    parts.append(
        f'<polygon points="{x(0):.1f},{base:.1f} {coords} {x(n - 1):.1f},{base:.1f}" '
        f'class="area"/>'
    )
    parts.append(f'<polyline points="{coords}" class="line" fill="none"/>')

    # 只标四个日期刻度，密了在手机上会糊成一团
    for i in {0, n // 3, 2 * n // 3, n - 1}:
        parts.append(f'<text x="{x(i):.1f}" y="{height - 12}" class="xtick" '
                     f'text-anchor="middle">{points[i][0].strftime("%y-%m-%d")}</text>')

    lx, ly = x(n - 1), y(vals[-1])
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" class="dot"/>')
    anchor = "end" if lx > _W - 120 else "start"
    dx = -8 if anchor == "end" else 8
    parts.append(f'<text x="{lx + dx:.1f}" y="{ly - 10:.1f}" class="endlabel" '
                 f'text-anchor="{anchor}">{_money_short(vals[-1])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bins(rows: list[tuple[str, int, int]], unit: str, height: int = 250) -> str:
    """持仓天数分布：赢家（红）/ 输家（绿）分组柱状图。"""
    total = sum(w + l for _, w, l in rows)
    if total == 0:
        return '<p class="empty">没有可统计的已实现卖出。</p>'

    max_c = max(max(w, l) for _, w, l in rows) or 1
    inner_w = _W - _PAD_L - _PAD_R
    inner_h = height - _PAD_T - _PAD_B
    slot = inner_w / len(rows)
    bw = min(slot * 0.30, 26)

    def y(v: float) -> float:
        return _PAD_T + inner_h * (1 - v / max_c)

    parts = [f'<svg viewBox="0 0 {_W} {height}" class="chart" role="img" '
             f'aria-label="持仓天数分布，赢家与输家分组对比，共 {total} 笔">']

    for t in range(0, max_c + 1, max(1, max_c // 4) or 1):
        yy = y(t)
        parts.append(f'<line x1="{_PAD_L}" y1="{yy:.1f}" x2="{_W - _PAD_R}" y2="{yy:.1f}" '
                     f'class="grid" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{_PAD_L - 8}" y="{yy + 4:.1f}" class="ytick" '
                     f'text-anchor="end">{t}</text>')

    for i, (label, w, l) in enumerate(rows):
        cx = _PAD_L + slot * (i + 0.5)
        for val, cls, off in ((w, "bar-win", -bw - 1), (l, "bar-loss", 1)):
            if val <= 0:
                continue
            h = inner_h * val / max_c
            parts.append(f'<rect x="{cx + off:.1f}" y="{y(val):.1f}" width="{bw:.1f}" '
                         f'height="{max(h, 1):.1f}" class="{cls}" rx="2"/>')
            if val >= 2 or max_c <= 8:
                parts.append(f'<text x="{cx + off + bw / 2:.1f}" y="{y(val) - 5:.1f}" '
                             f'class="barval" text-anchor="middle">{val}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height - 12}" class="xtick" '
                     f'text-anchor="middle">{label}</text>')

    parts.append(f'<text x="{_PAD_L + inner_w / 2:.0f}" y="{height - 1}" class="axisnote" '
                 f'text-anchor="middle">持有 {unit}（区间）</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_daily_de(obs: list[DailyObs], height: int = 230) -> str:
    """逐日处置效应柱状图。

    为什么值得单独画：pooled 的 DE 是一个数，掩盖了「有些卖出日 DE 是负的」
    这件事。图上能直接看出日度 DE 的离散程度 —— 如果柱子全在同一侧，
    结论就稳；如果上下乱跳，那个 pooled 均值就该被谨慎对待。
    这也是本图不画"平均线 + 置信带"的原因：日度观测不独立，
    画一条带标准误的横线会伪装出根本不存在的精度。
    """
    pts = [(o.dt, o.de) for o in obs if o.de is not None]
    if not pts:
        return '<p class="empty">没有可用的日度观测：每个卖出日都缺账面端数据。</p>'

    vals = [d for _, d in pts]
    # y 轴对称：零轴必须在正中，"向哪边偏"才是第一眼能读到的信息
    top = max(0.2, round(max(abs(v) for v in vals) + 0.05, 1))
    bot = -top

    n = len(pts)
    inner_w = _W - _PAD_L - _PAD_R
    inner_h = height - _PAD_T - _PAD_B
    slot = inner_w / n
    bw = max(1.5, min(slot * 0.62, 15.0))

    def y(v: float) -> float:
        return _PAD_T + (top - v) / (top - bot) * inner_h

    parts = [
        f'<svg viewBox="0 0 {_W} {height}" class="chart" role="img" '
        f'aria-label="逐日处置效应柱状图，共 {n} 个卖出日，'
        f"取值范围 {min(vals):.2f} 到 {max(vals):.2f}\">"
    ]

    for t in _ticks(bot, top, 4):
        yy = y(t)
        dash = ' stroke-dasharray="3 3"' if abs(t) > 1e-9 else ' stroke-width="1.5"'
        parts.append(f'<line x1="{_PAD_L}" y1="{yy:.1f}" x2="{_W - _PAD_R}" y2="{yy:.1f}" '
                     f'class="grid"{dash}/>')
        parts.append(f'<text x="{_PAD_L - 8}" y="{yy + 4:.1f}" class="ytick" '
                     f'text-anchor="end">{t:+.1f}</text>')

    zero = y(0)
    for i, (d, v) in enumerate(pts):
        cx = _PAD_L + slot * (i + 0.5)
        yv = y(v)
        # SVG 的 rect.y 是矩形顶部、height 向下延伸：
        # 正柱顶部在 yv（比零轴靠上）、负柱顶部在零轴。取 min 定顶、取绝对值定高。
        y_top = min(zero, yv)
        h = abs(yv - zero)
        cls = "bar-win" if v >= 0 else "bar-loss"
        parts.append(f'<rect x="{cx - bw / 2:.1f}" y="{y_top:.1f}" width="{bw:.1f}" '
                     f'height="{max(0.8, h):.1f}" class="{cls}"/>')

    for i in sorted({0, n // 2, n - 1}):
        parts.append(f'<text x="{_PAD_L + slot * (i + 0.5):.1f}" y="{height - 12}" '
                     f'class="xtick" text-anchor="middle">{pts[i][0].strftime("%y-%m-%d")}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- 样式
#
# 全部内联。字体用系统栈（不下载 webfont，否则离线打开会闪一下或直接变宋体）。
# 深色模式跟随系统，因为报告常在晚上看。

_CSS = """
*{box-sizing:border-box}
:root{
  --bg:#f6f5f2; --card:#ffffff; --ink:#1b1b1b; --ink2:#565656; --ink3:#8b8b85;
  --line:#e5e3dd; --up:#bf3b2b; --down:#1f7a45; --flat:#6b7280;
  --up-soft:#fbeae7; --down-soft:#e9f4ed; --accent:#8a5a00; --soft:#efece5;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#141418; --card:#1e1e23; --ink:#f1f0ec; --ink2:#a9a7a1; --ink3:#7b7973;
    --line:#2d2d34; --up:#ee7061; --down:#4fc07c; --flat:#9aa0aa;
    --up-soft:#2b1a18; --down-soft:#16291d; --accent:#e2b258; --soft:#26262c;
  }
}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",
    "Microsoft YaHei","Source Han Sans SC",sans-serif;
  font-size:15px;line-height:1.65}
main{max-width:860px;margin:0 auto;padding:28px 18px 64px}
h1{font-size:26px;font-weight:600;margin:6px 0 4px;letter-spacing:-.01em}
h2{font-size:17px;font-weight:600;margin:0 0 14px;letter-spacing:-.005em}
/* 章节标题的「人话主标题 + 专业名副标题」：主标题负责让人愿意读下去，
   副标题负责让懂行的人知道这块的学术名字。两者都不能省。 */
h2 small{display:block;font-size:12.5px;font-weight:400;color:var(--ink3);
  letter-spacing:0;margin-top:5px;line-height:1.5}
h3{font-size:15px;font-weight:600;margin:18px 0 8px}

/* 「如果你只有 30 秒」：给没时间读完这份报告的人。
   视觉上要比 verdict 更抢眼 —— 它是唯一一处「读完就走也不亏」的内容。 */
.tldr{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:24px 24px 20px;margin-bottom:16px}
.tldr-eyebrow{font-size:11px;letter-spacing:.14em;color:var(--ink3);
  font-weight:600;margin-bottom:8px}
.tldr-big{font-size:22px;font-weight:600;line-height:1.42;letter-spacing:-.02em;margin:0 0 10px}
.tldr-explain{font-size:13.5px;color:var(--ink2);line-height:1.78;margin:0}
.tldr-divider{height:1px;background:var(--line);margin:20px 0}
.tldr-answer{font-size:19px;line-height:1.58;font-weight:500;margin:0 0 14px;letter-spacing:-.01em}
.tldr-facts{list-style:none;padding:0;margin:0}
.tldr-facts li{position:relative;padding-left:16px;font-size:13.5px;color:var(--ink2);
  line-height:1.8;margin-bottom:6px}
.tldr-facts li:before{content:"";position:absolute;left:0;top:10px;width:5px;height:5px;
  border-radius:50%;background:var(--ink3)}
.tldr-facts b{color:var(--ink);font-weight:600}
.tldr-limit{font-size:12.5px;color:var(--ink3);line-height:1.78;margin:16px 0 0;
  padding:11px 13px;background:var(--soft);border-radius:9px}
.tldr-limit b{color:var(--ink2)}
@media (max-width:560px){
  .tldr{padding:18px 16px}
  .tldr-big{font-size:19px}
  .tldr-answer{font-size:16.5px}
}
p{margin:0 0 10px}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.sub{color:var(--ink2);font-size:13.5px;margin-bottom:22px}
.up{color:var(--up)} .down{color:var(--down)} .flat{color:var(--flat)}

.verdict{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 22px 18px;margin-bottom:16px}
.verdict-text{font-size:20px;line-height:1.5;font-weight:500;margin:0 0 14px;letter-spacing:-.01em}
@media (max-width:560px){ .verdict-text{font-size:17px} }

.tags{display:flex;flex-wrap:wrap;gap:8px}
.tag{font-size:13px;padding:4px 11px;border-radius:999px;background:var(--soft);
  color:var(--ink2);border:1px solid var(--line);white-space:nowrap}
.tag.primary{background:var(--up-soft);color:var(--up);border-color:transparent;font-weight:500}
.tag.primary.rev{background:var(--down-soft);color:var(--down)}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 15px}
.kpi .k{font-size:12px;color:var(--ink3);margin-bottom:5px;line-height:1.4}
.kpi .v{font-size:22px;font-weight:600;letter-spacing:-.02em;line-height:1.25;
  font-variant-numeric:tabular-nums}
.kpi .n{font-size:12px;color:var(--ink3);margin-top:3px}

.block{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:20px 22px;margin-bottom:16px}
@media (max-width:560px){ .block{padding:16px 14px} }

.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--line);fill:none;stroke-width:1}
.chart .ytick,.chart .xtick{fill:var(--ink3);font-size:15px}
.chart .barval{fill:var(--ink2);font-size:15px}
.chart .axisnote{fill:var(--ink3);font-size:14px}
.chart .endlabel{fill:var(--ink);font-size:16px;font-weight:600}
.curve-up .line{stroke:var(--up);stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
.curve-up .area{fill:var(--up);opacity:.10}
.curve-up .dot{fill:var(--up)}
.curve-down .line{stroke:var(--down);stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
.curve-down .area{fill:var(--down);opacity:.10}
.curve-down .dot{fill:var(--down)}
.bar-win{fill:var(--up);opacity:.85}
.bar-loss{fill:var(--down);opacity:.85}
/* 同行基准刻度尺：误差棒与文献点必须共享同一条轴，否则读法会变成比大小 */
.chart .zero-line{stroke:var(--ink3);stroke-width:1.2;stroke-dasharray:3 3}
.chart .ruler-band{fill:var(--up);opacity:.18}
.chart .ruler-cap{stroke:var(--up);stroke-width:2.4;stroke-linecap:round}
.chart .ruler-me{fill:var(--up);stroke:var(--card);stroke-width:2}
.chart .ruler-peer.in{fill:var(--accent);opacity:.9}
.chart .ruler-peer.out{fill:var(--ink)}
.chart .ruler-label{fill:var(--ink2);font-size:14px;font-weight:600}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.scroll .chart{min-width:520px}

.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--ink2);margin-bottom:10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
  vertical-align:middle}
.empty{color:var(--ink3);font-size:13.5px;padding:16px 0;margin:0}

details{border-top:1px solid var(--line);margin-top:12px;padding-top:10px}
details summary{cursor:pointer;font-size:13px;color:var(--ink2);list-style:none;
  display:flex;align-items:center;gap:6px;user-select:none}
details summary::-webkit-details-marker{display:none}
details summary:before{content:"▸";color:var(--ink3);font-size:11px;transition:transform .15s}
details[open] summary:before{transform:rotate(90deg)}
details .body{font-size:13px;color:var(--ink2);padding:8px 0 2px 16px;line-height:1.7}
details .body.pre{white-space:pre-line}
details .body b{color:var(--ink);font-weight:600}

table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-weight:500;color:var(--ink3);font-size:12px;position:sticky;top:0;background:var(--card)}
tbody tr:last-child td{border-bottom:none}

.bench-row{display:grid;grid-template-columns:1fr;gap:10px;margin:14px 0}
.bench-item{display:flex;align-items:center;gap:12px;font-size:13.5px}
.bench-item .label{width:96px;flex:0 0 96px;color:var(--ink2)}
.bench-item .track{flex:1;height:22px;background:var(--soft);border-radius:5px;position:relative;
  overflow:hidden;min-width:60px}
.bench-item .fill{position:absolute;top:0;bottom:0;border-radius:5px}
.bench-item .num{width:78px;flex:0 0 78px;text-align:right;font-weight:600;
  font-variant-numeric:tabular-nums}

.note{font-size:12.5px;color:var(--ink3);line-height:1.7;margin:10px 0 0;
  padding-left:0}
/* 「不做什么 / 下一步」：两列清单，不用表格 —— 表格的 nowrap 会让长句破版。
   只有条目标题（.bt）独占一行；句中加粗的数字必须保持行内，否则会断版。 */
.bounds{list-style:none;padding:0;margin:0}
.bounds li{padding:10px 0;border-bottom:1px solid var(--line);font-size:13.5px;
  color:var(--ink2);line-height:1.75}
.bounds li:last-child{border-bottom:none;padding-bottom:0}
.bounds li b.bt{display:block;color:var(--ink);font-weight:600;font-size:14px;margin-bottom:2px}
.bounds li b{color:var(--ink);font-weight:600}
.sub-h{font-size:13px;font-weight:600;color:var(--ink2);margin:16px 0 6px;
  letter-spacing:.02em}
.sub-h:first-of-type{margin-top:4px}
.warnbox{background:var(--up-soft);border-radius:10px;padding:12px 14px;font-size:13px;
  color:var(--ink2);line-height:1.7;margin-top:12px}
.warnbox b{color:var(--up)}

footer{border-top:1px solid var(--line);margin-top:26px;padding-top:18px;
  font-size:12.5px;color:var(--ink3);line-height:1.8}
footer b{color:var(--ink2);font-weight:500}
@media print{ body{background:#fff} .block,.verdict,.kpi{break-inside:avoid} }
"""


# ---------------------------------------------------------------- 组装

def _kpi(label: str, value: str, tone: str = "", note: str = "", audit: str = "") -> str:
    cls = f' class="v {tone}"' if tone else ' class="v"'
    note_html = f'<div class="n">{_esc(note)}</div>' if note else ""
    audit_html = (
        f"<details><summary>依据</summary><div class=\"body\">{audit}</div></details>"
        if audit else ""
    )
    return (f'<div class="kpi"><div class="k">{_esc(label)}</div>'
            f"<div{cls}>{value}</div>{note_html}{audit_html}</div>")


def _details(summary: str, body: str, open_: bool = False) -> str:
    op = " open" if open_ else ""
    return f"<details{op}><summary>{_esc(summary)}</summary><div class=\"body\">{body}</div></details>"


def _table(headers: list[str], rows: list[list[str]], tone_col: int | None = None) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            # 数值单元格已在调用侧格式化，这里不再转义，避免 ¥ 与负号被重复处理
            cells.append(f"<td>{c}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _signed_span(v: float) -> str:
    return "up" if v > 0 else ("down" if v < 0 else "flat")


def _compare_line(snap: float | None, roll: float | None) -> str:
    """两种口径的差异判读。与 CLI 里的 _compare_verdict 同一套逻辑，
    分开写是因为一边要纯文本、一边要带标签的 HTML —— 合并只会让两边都别扭。"""
    if snap is None or roll is None:
        return "两种口径无法直接比较（其中一边缺数据）。"
    diff = roll - snap
    if snap * roll < 0:
        return ("<b>方向相反</b>：期末那一天的账面持仓不能代表历史，"
                "只看期末会得出完全错误的结论。<b>以逐日滚动口径为准</b>。")
    if abs(diff) < 0.05:
        return "两种口径基本一致，<b>结论稳健</b>，不依赖用哪一天的价格。"
    if diff > 0:
        return ("逐日滚动更高：历史上有大量「浮盈没卖」的机会被期末口径漏掉了，"
                "期末口径<b>低估</b>了处置效应。")
    return ("逐日滚动更低：历史上有大量「浮亏没割」的机会被期末口径漏掉了。"
            "换句话说，你现在手里的浮亏<b>不代表历史全貌</b>。")


def _rolling_block(roll: RollingReport, rep: DispositionReport) -> list[str]:
    """逐日滚动口径区块。数据不足时也说清楚缺什么，而不是整块消失。"""
    out = [
        '<section class="block">',
        "<h2>换个更严的算法，结论还成立吗？"
        "<small>逐日滚动口径 · 每个卖出日重算一次账面端，而不是只看期末</small></h2>",
    ]

    if roll.mode != "rolling" or not roll.has_result:
        out.append(
            '<p class="note">这个口径需要<b>逐日价格序列</b>，当前算不出来 —— '
            "不是 bug，是数据不足。补法：<b>--price-history</b> "
            "（CSV：日期,证券代码,收盘价）。</p>"
        )
        if roll.notes:
            out.append('<p class="note">' + "<br>".join(_esc(n) for n in roll.notes) + "</p>")
        out.append("</section>")
        return out

    out.append(
        "<p>上一节的期末口径只用「今天还没卖的」头寸算账面端，历史上那些"
        "「当时浮盈、后来卖掉」「当时浮亏、后来割掉」的机会全部丢失。"
        "这一节换成 Odean (1998) 原文的做法：<b>每个卖出日都重新问一次</b> —— "
        "此刻还有哪些浮盈 / 浮亏的头寸本可以卖，却没卖。</p>"
    )

    out.append('<div class="kpis">')
    out.append(_kpi(
        "滚动口径 DE", _pct_signed(roll.de), _signed_span(roll.de or 0.0),
        f"{roll.n_obs} 个卖出日 · 计数口径",
        f"盈利机会 <b>{roll.n_gain_opp}</b> 次中兑现 <b>{roll.n_gain_sold}</b> 次 "
        f"→ PGR {_pct(roll.pgr, 2)}<br>"
        f"亏损机会 <b>{roll.n_loss_opp}</b> 次中兑现 <b>{roll.n_loss_sold}</b> 次 "
        f"→ PLR {_pct(roll.plr, 2)}<br>"
        f"DE = PGR − PLR = <b>{_pct_signed(roll.de, 2)}</b>。<br>"
        f"「机会」是逐日累加的：同一笔浮亏在持有的每一天都被计入一次。",
    ))
    if roll.de_amount is not None:
        out.append(_kpi(
            "滚动 · 金额口径", _pct_signed(roll.de_amount), _signed_span(roll.de_amount),
            "按金额加权",
            f"PGR {_pct(roll.pgr_amount, 2)} − PLR {_pct(roll.plr_amount, 2)} "
            f"= <b>{_pct_signed(roll.de_amount, 2)}</b>。<br>"
            f"与计数口径方向一致时，说明「笔数上」和「钱的体量上」都成立。",
        ))
    if roll.daily_de_mean is not None:
        out.append(_kpi(
            "日度 DE 离散", f"{roll.daily_de_std:.2f}", "",
            f"均值 {roll.daily_de_mean:+.2f} / σ {roll.daily_de_std:.2f}",
            f"{roll.n_daily_de} 个日度观测的标准差。<br>"
            "σ 大 = 你的行为在不同日子差别很大，"
            "那个 pooled 均值只代表<b>平均状态</b>。<br>"
            "（日度观测彼此不独立，这个数不构成标准误，只作离散度参考。）",
        ))
    out.append("</div>")

    out.append(
        '<div class="legend"><span><i style="background:var(--up)"></i>'
        "DE &gt; 0：当日卖出的盈利多于割掉的亏损</span>"
        '<span><i style="background:var(--down)"></i>DE &lt; 0</span></div>'
    )
    out.append(f'<div class="scroll">{svg_daily_de(roll.obs)}</div>')
    out.append(
        f'<p class="note">每根柱子是一个卖出日。柱子的<b>高低差异本身</b>才是这张图的重点：'
        f"如果上下乱跳，pooled 出来的那一个数就该被谨慎对待。"
        f"共 {roll.n_obs} 个观测日，纵轴固定对称以保证零轴居中。</p>"
    )

    if rep.has_full:
        diff = (roll.de - (rep.disposition_effect or 0.0)) * 100
        out.append(
            f'<p class="note"><b>口径对照</b>：期末快照 DE '
            f"{_pct_signed(rep.disposition_effect)} vs 逐日滚动 DE "
            f"{_pct_signed(roll.de)}（差 {diff:+.1f} 个百分点）。"
            f"{_compare_line(rep.disposition_effect, roll.de)}</p>"
        )

    # 披露放在折叠里：重要但不必占据首屏
    out.append(_details(
        "依据与局限：这个口径为什么不能和上一节直接比",
        "<br>".join(_esc(n) for n in roll.notes)
        + "<br><b>成本</b>：需要逐日价格序列，且价格缺失 / 停牌会直接影响分母。"
        + "<br><b>口径差</b>：PGR/PLR 的绝对值在本口径下通常明显低于期末口径，"
        "这不是行为变了，而是分母随持有天数膨胀。跨口径可比的只有 <b>DE</b>。",
    ))
    out.append("</section>")
    return out


def svg_peer_ruler(b: BenchmarkReport, height: int = 190) -> str:
    """处置效应刻度尺：你的区间 + 文献点估计。

    为什么画成一条轴而不是柱状对比：核心问题不是「谁大谁小」，而是
    「文献那个点有没有落在我的区间里」。画成柱状会诱导读者比高低，
    而正确读法是看覆盖关系 —— 所以点必须和误差棒同处一条轴。

    刻意不画"人群分布曲线"：文献只给点估计，没有分布。画一条钟形曲线
    就是在凭空捏造数据，而且看起来特别可信。
    """
    de_comps = [c for c in b.de_comparisons if c.theirs is not None]
    if not de_comps or b.ci_low is None or b.ci_high is None:
        return ""

    vals = [b.ci_low, b.ci_high, 0.0] + [c.theirs for c in de_comps]
    if b.my_de is not None:
        vals.append(b.my_de)

    lo, hi = min(vals), max(vals)
    span = hi - lo or 0.10
    lo -= span * 0.12
    hi += span * 0.12

    pad_l, pad_r = 40, 40
    inner = _W - pad_l - pad_r

    def x(v: float) -> float:
        return pad_l + (v - lo) / (hi - lo) * inner

    mid = height * 0.52

    parts = [f'<svg viewBox="0 0 {_W} {height}" class="chart" role="img" '
             f'aria-label="处置效应刻度尺：本人的置信区间与文献点估计">']

    # 刻度
    for t in _ticks(lo, hi, 5):
        tx = x(t)
        parts.append(f'<line x1="{tx:.1f}" y1="{mid - 22:.1f}" '
                     f'x2="{tx:.1f}" y2="{mid + 22:.1f}" class="grid"/>')
        parts.append(f'<text x="{tx:.1f}" y="{mid + 40:.1f}" '
                     f'text-anchor="middle" class="axisnote">{t:+.0%}</text>')

    # 数轴
    parts.append(f'<line x1="{pad_l}" y1="{mid:.1f}" x2="{_W - pad_r}" '
                 f'y2="{mid:.1f}" class="grid"/>')
    # 零轴强调
    zx = x(0.0)
    parts.append(f'<line x1="{zx:.1f}" y1="{mid - 34:.1f}" x2="{zx:.1f}" '
                 f'y2="{mid + 30:.1f}" class="zero-line"/>')

    # 我的区间：横条 + 端帽
    x1, x2 = x(b.ci_low), x(b.ci_high)
    parts.append(f'<rect x="{x1:.1f}" y="{mid - 13:.1f}" width="{max(2.0, x2 - x1):.1f}" '
                 f'height="26" rx="4" class="ruler-band"/>')
    for xx in (x1, x2):
        parts.append(f'<line x1="{xx:.1f}" y1="{mid - 22:.1f}" x2="{xx:.1f}" '
                     f'y2="{mid + 22:.1f}" class="ruler-cap"/>')

    # 我的点估计
    if b.my_de is not None:
        parts.append(f'<circle cx="{x(b.my_de):.1f}" cy="{mid}" r="7" '
                     f'class="ruler-me"/>')
        parts.append(f'<text x="{x(b.my_de):.1f}" y="{mid - 26:.1f}" '
                     f'text-anchor="middle" class="ruler-label">你 {b.my_de:+.1%}</text>')

    # 文献点：菱形。落在区间内 / 外给不同样式，一眼能看出覆盖关系
    for i, c in enumerate(de_comps):
        cx = x(c.theirs)
        inside = b.ci_low <= c.theirs <= b.ci_high
        cls = "ruler-peer in" if inside else "ruler-peer out"
        dy = mid + 64 + i * 20
        cy = mid
        parts.append(
            f'<polygon points="{cx:.1f},{cy - 8:.1f} {cx + 8:.1f},{cy:.1f} '
            f'{cx:.1f},{cy + 8:.1f} {cx - 8:.1f},{cy:.1f}" class="{cls}"/>'
        )
        parts.append(f'<text x="{cx:.1f}" y="{dy:.1f}" text-anchor="middle" '
                     f'class="ruler-label">{_esc(c.bench.short_market)} '
                     f'{c.theirs:+.1%}</text>')

    # 图例
    parts.append(f'<text x="{pad_l}" y="16" class="axisnote">'
                 f'横轴：PGR − PLR（百分点）· 区间越宽＝样本越少</text>')
    parts.append("</svg>")
    return "".join(parts)


def _peer_block(b: BenchmarkReport) -> list[str]:
    """同行基准区块。"""
    if not b.comparable:
        return [
            '<section class="block">',
            "<h2>这次跟学术研究比不了<small>同行基准 · 缺少可对照的数据，直接说清缺什么</small></h2>",
            f'<p class="note">{_esc(b.blocked_reason or "")}</p>',
            "</section>",
        ]

    out = [
        '<section class="block">',
        "<h2>跟学术研究比，你这算严重吗？"
        "<small>同行基准 · Odean 1998 / 武佳薇 2020，只比量级，不给你排名</small></h2>",
    ]

    out.append(
        '<p class="note">两条轴上的点来自已发表研究，是<b>群体点估计</b>；'
        "红色横条是<b>你自己的 95% 置信区间</b>。正确的读法是看"
        "「文献那个点有没有落进你的条里」，而不是比谁高谁低。</p>"
    )

    ruler = svg_peer_ruler(b)
    if ruler:
        out.append(f'<div class="scroll">{ruler}</div>')

    # 逐条对照
    rows: list[list[str]] = []
    for c in b.de_comparisons:
        bench = c.bench
        rows.append([
            _esc(bench.short_market),
            _esc(bench.sample_size),
            f'{bench.pgr:.1%} / {bench.plr:.1%}',
            f"{bench.de * 100:+.1f}pp",
            f"{c.mine * 100:+.1f}pp" if c.mine is not None else "—",
            _esc(_VERDICT_CN.get(c.verdict, c.verdict)),
        ])
    if rows:
        out.append(_table(
            ["人群", "样本", "PGR / PLR", "文献 DE", "你的 DE", "判读"],
            rows, tone_col=3,
        ))

    # 比值口径：只说量级，不算显著
    ratio_rows: list[list[str]] = []
    for c in b.ratio_comparisons:
        ratio_rows.append([
            _esc(c.bench.short_market),
            f"{c.bench.ratio:.2f}",
            f"{c.mine:.2f}" if c.mine is not None else "—",
            _esc(_VERDICT_CN.get(c.verdict, c.verdict)),
        ])
    if ratio_rows:
        out.append(_details(
            "比值口径（PGR/PLR）对照 —— 为什么只列不用它做检验",
            _table(["人群", "文献比值", "你的比值", "判读"], ratio_rows)
            + '<p class="note">比值 = PGR ÷ PLR，分母是 PLR。当 PLR 很小时'
            "（几十次机会里只割过一两次肉），比值会对个别交易极度敏感，"
            "同一个账户换个样本期就能从 2 跳到 8。所以它只适合看量级，"
            "不适合做显著性检验 —— 判断用上面的 DE 差值口径。</p>",
        ))

    if b.notes:
        out.append('<p class="note">' + "<br>".join(_esc(n) for n in b.notes) + "</p>")

    # 核查记录：撤下的错误基准，主动交代
    retracted = (b.meta or {}).get("retracted", [])
    if retracted:
        items = []
        for r in retracted:
            items.append(
                f'<p class="note"><b>已撤下：{_esc(r.get("market", ""))} '
                f'{_esc(r.get("claimed", ""))}</b>'
                f'（原标注来源：{_esc(r.get("attributed_to", ""))}）<br>'
                f'{_esc(r.get("why", ""))}<br>'
                f'核查依据：{_esc(r.get("evidence", ""))}</p>'
            )
        out.append(_details(
            "核查记录：这两条基准被撤下了",
            "".join(items)
            + '<p class="note">这两条曾出现在本项目早期的路线图里，来自二手文章。'
            "回原文核对时发现原文根本没报告这个数字。留在这里是为了下次"
            "不要又被同一个错误抄回去。</p>",
        ))

    if b.ci_width is not None and b.power_ratio is not None:
        out.append(
            f'<p class="note">你的区间宽度 <b>{b.ci_width:.1%}</b>，'
            f"是文献效应量（{_peer_ref_de(b):.1%}）的 "
            f"<b>{b.power_ratio:.0f} 倍</b> —— 这个比值就是这套对照的"
            "分辨力上限。</p>"
        )

    out.append("</section>")
    return out


_VERDICT_CN = {
    "indistinguishable": "落在区间内（样本不足以分辨）",
    "weaker": "你的更弱",
    "stronger": "你的更强",
    "unstable": "比值不稳定",
    "descriptive": "仅量级参照",
    "unavailable": "无法计算",
}


#: 「这个报告不做什么」。每一条都对应一个**看起来更专业、实际站不住**的做法 ——
#: 写出来是为了让读者知道这些坑作者踩过并且主动绕开了，不是免责声明。
_NOT_DOING = [
    ("不预测收益、不推荐标的",
     "交割单里只有已经发生的事。任何「照这个能赚多少」的说法都是编的，"
     "本项目不提供、也不暗示。"),
    ("不用分位数给你排名",
     "文献只报告群体均值，没有个体分布。硬套一个分布算出「你比 84% 的散户严重」"
     "是伪精确 —— 这个功能原计划要做，查完原文发现做不了，已撤下并写进核查记录。"),
    ("不把「已实现收益率」当业绩",
     "它不含未平仓浮盈浮亏、不含资金的时间价值，和「账户总资产收益率」不是一回事。"
     "真正的业绩归因需要每日持仓市值序列，当前版本不做。"),
    ("不自动判断价格是否复权",
     "账本成本是不复权的成交价。喂进来的如果是前复权价，历史价会被按除权因子下调，"
     "「成本 120 元 vs 现价 92 元」凭空多出一笔浮亏，还会静默进 PGR/PLR 的分母。"
     "口径靠输入保证：取数脚本默认不复权，别改。"),
    ("不覆盖融资融券利息、基金申赎、期权",
     "这些不在「成交」的语义里。硬塞进 FIFO 配对会污染成本，"
     "宁可明确不支持，也不给一个算得出来但算错了的数。"),
    ("数据不出你的电脑",
     "全程本地运行，报告是单文件 HTML、零外链请求。代价是转发出去就等于"
     "交出完整原始数据 —— 分享前请自行确认是否要脱敏。"),
]


def _bounds_block(
    rep: DispositionReport,
    sig: SignificanceResult | None,
    *,
    has_price_history: bool,
    has_prices: bool,
    has_benchmark: bool,
    n_realized: int,
    span_days: int,
) -> list[str]:
    """「这个报告不做什么」+「下一步可以做什么」。

    为什么值得单独占用一块版面：
        这份报告会被拿去判断作者的水平。而判断水平最快的方式不是看做了什么，
        是看知不知道自己没做什么 —— 六条「不做」里任何一条说反了，
        都比少给一个指标更致命。
        「下一步」也一样：诊断完不给可执行的动作，读者只能带走一个情绪。
    """
    out = [
        '<section class="block">',
        "<h2>这个报告不做什么"
        "<small>写在这里是为了让上面的结论更可信，不是免责声明</small></h2>",
        '<ul class="bounds">',
    ]
    for title_, why in _NOT_DOING:
        out.append(f'<li><b class="bt">{title_}</b>{why}</li>')
    out.append("</ul>")
    out.append("</section>")

    # --- 下一步：分成「让报告更准」和「让习惯少亏钱」两类
    todo: list[str] = []
    missing = []
    if not has_price_history:
        missing.append(
            "<code>--price-history</code> 给一份逐日收盘价 —— "
            "这是本报告最严的部分（滚动口径 + 同行基准）唯一的数据来源"
        )
    if not has_prices:
        missing.append(
            "<code>--prices</code> 给未平仓的最新价 —— "
            "PGR/PLR 的分母要「已实现 + 账面」两端，缺了它只能看已实现画像"
        )
    if not has_benchmark:
        missing.append(
            "<code>--benchmark</code> 给同期指数日线 —— "
            "才能回答「有没有跑赢大盘」"
        )
    if missing:
        # 标题要跟着实际缺几样走 —— 只缺一样却说「补三样」，读者会以为漏了
        head = {1: "还差一样数据", 2: "还差两样数据"}.get(len(missing), "补三样数据")
        todo.append(
            f'<div class="sub-h">想让它更准：{head}</div>'
            "<ul class=\"bounds\">"
            + "".join(f"<li>{m}</li>" for m in missing)
            + "</ul>"
        )

    # 样本量提示：时间长度和卖出笔数是两个独立门槛，任一不足都要说
    thin: list[str] = []
    if span_days and span_days < 180:
        thin.append(f"跨度只有 <b>{span_days} 天</b>")
    if n_realized < 20:
        thin.append(f"已实现只有 <b>{n_realized} 笔</b>")
    if thin:
        todo.append(
            '<div class="sub-h">想让它更准：样本还太薄</div>'
            '<ul class="bounds"><li>' + "、".join(thin) + "。"
            "处置效应是习惯性偏差，得攒够时间和足够多的卖出动作才看得出来 —— "
            "几个月内的数据很容易把运气当成习惯。</li></ul>"
        )

    # 行为建议：只在方向站得住的时候给，且明确标注不是投资建议
    advice = ""
    if sig is not None and _sig_ok(sig) and sig.de_count is not None:
        if sig.de_count > 0:
            advice = (
                "你的偏差方向很明确（卖盈持亏，且统计显著）。文献里对这种偏差"
                "最常见的干预不是「下次注意」，而是<b>把卖出决策从临场判断改成事先写好的规则</b>"
                " —— 比如买入时就定下止损线，盈利到某个幅度分批止盈。"
                "规则的价值在于它不参与情绪。"
            )
        else:
            advice = (
                "你在做反人性的事：亏损实现比例高于盈利实现比例，而且统计显著。"
                "这是多数人做不到的方向，但代价通常是<b>止损过于频繁</b> —— "
                "建议顺手看一眼「每笔平均盈亏」和交易频率：止损次数多、单笔盈利小，"
                "说明可能切得太碎。"
            )
    elif rep.has_full and rep.disposition_effect is not None and abs(rep.disposition_effect) > 0.10:
        advice = (
            "方向有迹象，但样本量还不足以排除偶然 —— "
            "现在最该做的不是改交易习惯，是<b>继续积累记录</b>。"
            "按当前笔数，再攒几个月才够下判断。"
        )
    if advice:
        todo.append(
            '<div class="sub-h">想让这个习惯少亏钱：一条建议</div>'
            f'<ul class="bounds"><li>{advice}</li></ul>'
        )
    if not todo:
        todo.append(
            '<div class="sub-h">下一步</div>'
            '<ul class="bounds"><li>数据已齐、观察期也够。'
            "想更进一步只能拉长时间窗口，或者换一批账户做横向对照。</li></ul>"
        )

    out.extend([
        '<section class="block">',
        "<h2>下一步可以做什么"
        "<small>两份清单：让这份报告更准 / 让这个习惯少亏钱</small></h2>",
        *todo,
    ])
    # 只有真给了建议才需要那句免责声明 —— 没给建议还挂着，
    # 读者会以为上面藏了半句投资建议。
    if advice:
        out.append(
            '<p class="note">以上是行为金融文献里常见的干预思路，'
            "用于描述「这类偏差一般怎么处理」，<b>不构成投资建议</b>。"
            "要不要改、怎么改，取决于你自己的资金性质和风险承受能力。</p>"
        )
    out.append("</section>")
    return out


def _peer_ref_de(b: BenchmarkReport) -> float:
    for c in b.de_comparisons:
        if c.theirs:
            return c.theirs
    return 0.05


def build_report(
    ds: Dataset,
    *,
    prices: dict[str, Decimal] | None = None,
    benchmark: dict[date, Decimal] | None = None,
    generated_at: datetime | None = None,
    title: str = "交易体检报告",
    price_history: PriceHistory | None = None,
) -> str:
    """生成完整单文件 HTML。"""
    ledger = ds.ledger
    chunks = ledger.chunks
    summary = ds.summary()
    rep = ds.disposition(prices)
    sig = ds.significance(prices) if prices else None
    roll = ds.rolling(price_history) if price_history else None
    span = ds.span
    unit = "交易日" if rep.hold_days_trading else "自然日"
    now = generated_at or datetime.now()

    span_days = (span[1] - span[0]).days if span else 0
    total_pnl = summary["已实现总收益"]
    n_realized = len(chunks)

    tags = persona_tags(rep, sig, summary, chunks, len(ds.events), span_days)
    head_line = one_liner(rep, sig, n_realized)

    # --- 头部
    scope = "—"
    if span:
        scope = f"{span[0]} → {span[1]}（{span_days} 天）"
    head = [
        '<header>',
        '<div class="eyebrow">tradeopsy · 交易体检</div>',
        f"<h1>{_esc(title)}</h1>",
        f'<p class="sub">{_esc(scope)} · {len(ds.events)} 条记录'
        + (f"（已去重 {ds.dup} 条）" if ds.dup else "")
        + f" · {len(ds.symbols)} 只标的 · {len(ds.paths)} 个文件</p>",
        "</header>",
    ]

    # --- 30 秒版：结论 + 证据 + 边界，一次说完。
    # 它是整份报告唯一一处「读完就走也不亏」的内容，所以放在 KPI 之前。
    tldr = _tldr(
        rep, sig, head_line, tags,
        n_realized=n_realized,
        n_events=len(ds.events),
        n_symbols=len(ds.symbols),
        span_days=span_days,
        unit=unit,
    )

    # --- KPI 卡
    win_rate = summary["胜率"]
    kpis = ['<section class="kpis">']
    kpis.append(_kpi(
        "已实现总收益", _money_short(total_pnl),
        _signed_span(total_pnl),
        f"价差 {_money_short(summary['已实现价差盈亏'])} + 分红 {_money_short(summary['现金分红'])}",
        f"已实现价差 <b>{_money(summary['已实现价差盈亏'])}</b> ＋ 现金分红 "
        f"<b>{_money(summary['现金分红'])}</b> ＝ <b>{_money(total_pnl)}</b>。<br>"
        f"已实现笔数 {n_realized}（一个卖出动作 × 一个买入批次算一笔）。"
        f"未平仓 {summary['未平仓批次']} 个批次不计入 —— 它们没有卖出，盈亏还没兑现。",
    ))
    kpis.append(_kpi(
        "胜率", _pct(win_rate), "",
        f"{summary['盈利笔数']} 赚 / {summary['亏损笔数']} 亏",
        f"盈利 {summary['盈利笔数']} 笔 ÷ 已实现 {n_realized} 笔 = {_pct(win_rate)}。<br>"
        f"盈亏比 {summary['盈亏比'] if summary['盈亏比'] is not None else '—（无亏损笔）'}："
        f"总盈利 {_money(summary['总盈利'])} ÷ 总亏损 {_money(summary['总亏损'])}。",
    ))
    if rep.has_full:
        de = rep.disposition_effect or 0.0
        kpis.append(_kpi(
            "处置效应 DE", _pct_signed(de), _signed_span(de),
            "PGR − PLR · 金额口径",
            f"PGR <b>{_pct(rep.pgr, 2)}</b> － PLR <b>{_pct(rep.plr, 2)}</b> ＝ "
            f"<b>{_pct_signed(de, 2)}</b>。<br>"
            f"分子：已实现盈利 {_money(rep.realized_gain)} · 已实现亏损 {_money(rep.realized_loss)}<br>"
            f"分母追加：账面盈利 {_money(rep.paper_gain)} · 账面亏损 {_money(rep.paper_loss)}"
            f"（来自 {rep.n_open} 个未平仓头寸的市值）<br>"
            f"<b>DE &gt; 0 = 卖盈持亏</b>（处置效应）；DE &lt; 0 = 反向。",
        ))
    else:
        kpis.append(_kpi(
            "处置效应 DE", "—", "",
            "缺未平仓市值，不可算",
            "PGR/PLR 的<b>分母</b>需要「账面盈利 + 账面亏损」，"
            "也就是未平仓头寸的当前市值。<br>"
            "没给价格时这部分无从得知，硬算会让 PGR/PLR 退化成 1.0 —— "
            "数学成立但完全误导，所以这里直接留空。<br>"
            "补法：<b>--prices 价文件</b>（CSV：代码,最新价）。",
        ))
    if sig is not None and sig.de_count is not None:
        kpis.append(_kpi(
            "计数口径 DE", _pct_signed(sig.de_count), _signed_span(sig.de_count),
            f"p={sig.p_value_boot:.3f}" if sig.p_value_boot is not None else "p=—",
            f"按<b>笔数</b>而非金额：盈利机会中兑现 {sig.n_gain_sold}/"
            f"{sig.n_gain_total}，亏损机会中兑现 {sig.n_loss_sold}/{sig.n_loss_total}。<br>"
            f"PGR {_pct(sig.pgr_count, 2)} · PLR {_pct(sig.plr_count, 2)} · "
            f"DE {_pct_signed(sig.de_count, 2)}<br>"
            f"95% 置信区间 [{_pct_signed(sig.ci_low, 2)}, {_pct_signed(sig.ci_high, 2)}]，"
            + (f"bootstrap p={sig.p_value_boot:.4f}，n_boot={sig.n_boot}，seed={sig.seed}。"
               if sig.p_value_boot is not None else
               "bootstrap 未构造（单侧已实现为 0）。")
            + f"<br>"
            f"<b>{_esc(sig.verdict)}</b>",
        ))
    kpis.append(_kpi(
        "赚的 / 亏的各拿多久", f"{rep.avg_hold_win} / {rep.avg_hold_loss}", "",
        f"{unit}",
        f"盈利笔平均持有 <b>{rep.avg_hold_win} {unit}</b>；"
        f"亏损笔平均持有 <b>{rep.avg_hold_loss} {unit}</b>。<br>"
        + (f"同一批数据按自然日算分别是 {rep.avg_hold_win_calendar} / "
           f"{rep.avg_hold_loss_calendar} 天 —— 自然日会系统性高估约 28%。"
           if rep.hold_days_trading and rep.avg_hold_win_calendar else
           "当前为<b>自然日</b>口径；A 股严谨口径应为交易日。"),
    ))
    kpis.append(_kpi(
        "交易频率", f"{len(ds.events)}", "",
        "条交割记录",
        f"共 {len(ds.events)} 条事件，覆盖 {span_days} 天，"
        f"约 {len(ds.events) / max(span_days / 365.25, 0.08):.0f} 条/年。<br>"
        f"类型分布：" + "、".join(
            f"{k} {v}" for k, v in sorted(
                _count_by_type(ds).items(), key=lambda kv: -kv[1])) + "。",
    ))
    kpis.append("</section>")

    # --- 处置效应双口径说明
    disp_block = [
        '<section class="block">',
        "<h2>两套算法，答案一致吗？"
        "<small>处置效应 DE · 金额口径与计数口径：两边都看，不一致本身就是信息</small></h2>",
    ]
    if rep.has_full and sig is not None and sig.de_count is not None:
        disp_block.append(
            f'<p>金额口径 DE <b class="{_signed_span(rep.disposition_effect or 0)}">'
            f"{_pct_signed(rep.disposition_effect)}</b>，"
            f"计数口径 DE <b class=\"{_signed_span(sig.de_count)}\">"
            f"{_pct_signed(sig.de_count)}</b>。</p>"
            "<p class=\"note\">Odean (1998) 原文用的是<b>计数口径</b>（笔数比）；"
            "金额口径是实践变体。两者可能给出方向相反的结论 —— 比如「卖了很多笔小赚的票、"
            "但金额上其实留住了大头」。学术判定以计数口径为准，"
            "金额口径用来判断「钱的体量上有没有同样的倾向」。</p>"
        )
        if sig.notes:
            disp_block.append('<p class="note">' + "<br>".join(_esc(n) for n in sig.notes) + "</p>")
    else:
        disp_block.append(
            '<p class="note">缺少未平仓市值，只能给出已实现画像，无法计算 PGR/PLR。'
            "这不是 bug，是数据不足 —— 报告不会用一个退化的数字假装算完了。</p>"
        )
    disp_block.append("</section>")

    # --- 逐日滚动口径（提供逐日价格序列时才出现）
    roll_block = _rolling_block(roll, rep) if roll is not None else []

    # --- 同行基准对照（只有滚动口径才与文献同构；算不出来就整块不出现，
    #     避免报告里多一个"数据不足"的空板块制造噪音）
    peer_block: list[str] = []
    if roll is not None:
        _peer = ds.benchmark(roll)
        if _peer.comparable:
            peer_block = _peer_block(_peer)

    # --- 收益曲线
    curve = equity_curve(chunks)
    peak = Decimal("0")
    mdd = Decimal("0")
    for _, v in curve:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    curve_block = [
        '<section class="block">',
        "<h2>卖掉的那些，一共赚了多少？"
        "<small>累计已实现盈亏 · 按卖出日累计，不含还在手上的浮盈浮亏</small></h2>",
        '<p class="note">按卖出日累计。单日盈亏噪声太大，累计曲线才能回答'
        "「这一年到底在往上还是往下」。</p>",
        f'<div class="scroll">{svg_curve(curve)}</div>',
    ]
    if curve:
        curve_block.append(
            f'<p class="note">共 {len(curve)} 个卖出日 · 最终 {_money(curve[-1][1])} · '
            f"期间最大回撤 <b>{_money(mdd)}</b>（从历史最高点回落的最大幅度）</p>"
        )
        curve_block.append(_details(
            "依据：曲线怎么来的",
            "把每一笔已实现盈亏按<b>平仓日</b>归组，同日多笔先求和，再沿时间累加。<br>"
            f"首笔 {curve[0][0]}（{_money(curve[0][1])}），末笔 {curve[-1][0]}"
            f"（{_money(curve[-1][1])}）。<br>"
            f"最大回撤：遍历曲线，记录「当前值 − 历史最高值」的最小值 = <b>{_money(mdd)}</b>。<br>"
            "注意：这是<b>已实现</b>口径，不含未平仓浮盈浮亏。"
            "所以它衡量的是「落袋的这部分钱」的画像，不是你账户总资产。",
        ))
    curve_block.append("</section>")

    # --- 持仓分布
    bins = hold_bins(chunks, ds.calendar)
    bins_block = [
        '<section class="block">',
        "<h2>赚钱的和亏钱的，你各拿了多久？"
        "<small>持有期分布 · 赢家与输家分开计</small></h2>",
        '<div class="legend">'
        '<span><i style="background:var(--up)"></i>盈利笔</span>'
        '<span><i style="background:var(--down)"></i>亏损笔</span></div>',
        f'<div class="scroll">{svg_bins(bins, unit)}</div>',
    ]
    peak_bin_w = max(bins, key=lambda r: r[1]) if bins else None
    peak_bin_l = max(bins, key=lambda r: r[2]) if bins else None
    if peak_bin_w and peak_bin_l and (peak_bin_w[1] or peak_bin_l[2]):
        bins_block.append(
            f'<p class="note">盈利笔最集中的区间是 <b>{peak_bin_w[0]} {unit}</b>'
            f"（{peak_bin_w[1]} 笔）；亏损笔最集中在 <b>{peak_bin_l[0]} {unit}</b>"
            f"（{peak_bin_l[2]} 笔）。</p>"
        )
    bins_block.append(_details(
        "依据：分箱规则与原始计数",
        "分箱边界（" + unit + "）：" + "、".join(
            f"{BIN_EDGES[i]}–{BIN_EDGES[i+1]}" if BIN_EDGES[i+1] < 10**9 else f"{BIN_EDGES[i]}+"
            for i in range(len(BIN_LABELS))) + "<br>"
        "平手（盈亏恰好为 0）不计入任何一侧 —— 它既不是赢家也不是输家。<br><br>"
        + "<br>".join(f"{label}：赢 {w} 笔 / 亏 {l} 笔" for label, w, l in bins),
    ))
    bins_block.append("</section>")

    # --- 基准对比
    bench_block = [
        '<section class="block">',
        "<h2>同期大盘涨跌对比"
        "<small>已实现收益率 vs 基准收益率 · 只能当参考，不能当业绩</small></h2>",
    ]
    invested = _invested(ds)
    my_ret = (float(total_pnl) / float(invested)) if invested > 0 else 0.0
    if benchmark and span:
        series = sorted(benchmark.items())
        inside = [(d, v) for d, v in series if span[0] <= d <= span[1]]
        if len(inside) >= 2:
            b_ret = float(inside[-1][1] / inside[0][1] - 1)
            mx = max(abs(my_ret), abs(b_ret), 0.01)
            rows = []
            for label, val, color in (
                ("我的已实现", my_ret, "var(--up)" if my_ret >= 0 else "var(--down)"),
                ("基准同期", b_ret, "var(--up)" if b_ret >= 0 else "var(--down)"),
            ):
                w = abs(val) / mx * 100
                rows.append(
                    f'<div class="bench-item"><div class="label">{label}</div>'
                    f'<div class="track"><div class="fill" style="width:{w:.1f}%;'
                    f'{"left:50%" if val < 0 else "left:50%"};'
                    f'transform:translateX({"-100%" if val < 0 else "0"});'
                    f'background:{color}"></div></div>'
                    f'<div class="num {_signed_span(val)}">{_pct_signed(val)}</div></div>'
                )
            bench_block.append('<div class="bench-row">' + "".join(rows) + "</div>")
            bench_block.append(
                f'<p class="note">同期基准涨跌 <b>{_pct_signed(b_ret)}</b>'
                f"（{inside[0][0]} → {inside[-1][0]}，共 {len(inside)} 个数据点）；"
                f"我的已实现收益率 <b>{_pct_signed(my_ret)}</b>。"
                f"差额 <b>{_pct_signed(my_ret - b_ret)}</b>。</p>"
            )
            bench_block.append(
                '<div class="warnbox"><b>这个对比只能当参考，不能当业绩。</b><br>'
                "「我的已实现收益率」= 已实现总收益 ÷ 动用过的本金，是<b>已实现口径</b>；"
                "它不含未平仓浮盈浮亏、不含资金的时间价值，也和「账户总资产收益率」不是一回事。<br>"
                "真正可比的择时归因需要<b>每日持仓市值序列</b>（要行情），当前版本不做 —— "
                "硬做只会得到一个看似专业、实则站不住的数。</div>"
            )
        else:
            bench_block.append(
                '<p class="empty">基准数据落在交割单区间内的点不足 2 个，无法对比。</p>'
            )
    else:
        bench_block.append(
            '<p class="empty">未提供基准数据，这一块留空。</p>'
            '<p class="note">为什么不能自动算：交割单只记录<b>你自己的成交</b>，'
            "不记录同期大盘涨跌。要算「有没有跑赢指数」，必须外部提供基准的日收盘价序列。<br>"
            "补法：<b>--benchmark 基准.csv</b>（每行 <code>YYYY-MM-DD,收盘价</code>），"
            "例如导出一段沪深300ETF（510300）的日线。</p>"
        )
    bench_block.append("</section>")

    # --- 边界与下一步。
    # 放在明细之前而不是最后：明细又长又枯燥，读者在那里关掉页面的概率最高；
    # 而这两块是最该被看见的自省，不能承担这个风险。
    bounds = _bounds_block(
        rep, sig,
        has_price_history=price_history is not None,
        has_prices=prices is not None,
        has_benchmark=benchmark is not None,
        n_realized=n_realized,
        span_days=span_days,
    )

    # --- 明细
    detail = ['<section class="block">',
              "<h2>每一笔的账，都能对得上"
              "<small>明细 · 可逐行审计回原始交割记录</small></h2>",
              '<p class="note">上面每一个数字都能在这里找到对应原始记录。'
              "点开逐项核对。</p>"]

    if chunks:
        rows = []
        for c in chunks:
            held = (ds.calendar.count(c.open_dt, c.close_dt)
                    if ds.calendar is not None else c.hold_days)
            tone = _tone(float(c.pnl))
            rows.append([
                _esc(c.symbol), _esc(c.open_dt), _esc(c.close_dt),
                f"{float(c.qty):,.0f}",
                _money(c.cost), _money(c.proceeds),
                f'<span class="{tone}">{_money(c.pnl)}</span>',
                f'<span class="{tone}">{_pct(float(c.pnl_pct), 2)}</span>',
                str(held),
            ])
        detail.append(_details(
            f"逐笔已实现（{len(rows)} 笔）",
            _table(["标的", "建仓", "平仓", "数量", "成本", "净收入", "盈亏", "收益率",
                    f"持有({unit})"], rows),
        ))

    by_sym = _by_symbol(chunks)
    if by_sym:
        rows = [[_esc(s), str(n), f"{w}", f'<span class="{_tone(float(p))}">{_money(p)}</span>']
                for s, n, p, w in by_sym]
        detail.append(_details(
            f"按标的归因（{len(rows)} 只，亏得最多的在最上面）",
            _table(["标的", "笔数", "胜", "已实现盈亏"], rows),
        ))

    allw = ds.all_warnings
    if allw:
        detail.append(_details(
            f"告警（{len(allw)} 条，未中断解析）",
            "<br>".join(_esc(w) for w in allw[:60])
            + (f"<br>… 另 {len(allw) - 60} 条" if len(allw) > 60 else ""),
        ))

    files_txt = "、".join(
        f"{_esc(name)}" + (f"（{_esc(m.get('encoding'))}）" if m.get("encoding") else "")
        for name, m in ds.file_metas
    )
    detail.append(_details(
        "数据来源与解析方式",
        f"文件：{files_txt}<br>"
        + "".join(
            f"{_esc(name)}：编码 {_esc(m.get('encoding'))} · 分隔符 {_esc(m.get('delimiter'))} · "
            f"表头第 {_esc(m.get('header_row'))} 行 · {_esc(m.get('total_rows'))} 行"
            + (f" · 列映射 {' / '.join(f'{_esc(k)}={_esc(v)}' for k, v in (m.get('columns') or {}).items() if v)}"
               if m.get("columns") else "")
            + "<br>"
            for name, m in ds.file_metas
        )
        + (f"跨文件去重 {ds.dup} 条。" if ds.dup else ""),
    ))
    detail.append("</section>")

    # --- 页脚
    cal_note = (ds.calendar.describe() if ds.calendar
                else "按自然日计算（非 A 股严谨口径）")
    footer = [
        "<footer>",
        f"<b>生成时间</b> {now.strftime('%Y-%m-%d %H:%M')} · "
        f"<b>数据</b> {len(ds.paths)} 个文件 / {len(ds.events)} 条记录<br>",
        f"<b>持有期口径</b> {_esc(cal_note)}<br>",
        "<b>处置效应口径</b> PGR = 已实现盈利/(已实现盈利+账面盈利)，"
        "PLR = 已实现亏损/(已实现亏损+账面亏损)，DE = PGR − PLR（Odean 1998）<br>",
        "<b>已知局限</b> 未平仓头寸用成本或给定价格计市值；"
        "同一标的的分批买卖存在相关性，p 值可能偏乐观；"
        "未处理融资融券利息与基金申赎。<br>",
        '<div class="warnbox">这份报告包含你的持仓标的与金额。'
        "<b>分享前请确认是否要脱敏</b> —— 它是自包含的单文件 HTML，"
        "不含任何外链请求，但也意味着转发出去就是完整的原始数据。</div>",
        "</footer>",
    ]

    doc = (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(title)} · tradeopsy</title>"
        f"<style>{_CSS}</style></head><body><main>"
        + "".join(head) + "".join(tldr) + "".join(kpis)
        + "".join(disp_block) + "".join(roll_block) + "".join(peer_block)
        + "".join(curve_block) + "".join(bins_block)
        + "".join(bench_block) + "".join(bounds) + "".join(detail) + "".join(footer)
        + "</main></body></html>"
    )
    return doc


def _count_by_type(ds: Dataset) -> dict:
    from collections import Counter
    from .model import EventType

    cn = {
        EventType.BUY: "买入", EventType.SELL: "卖出", EventType.DIVIDEND: "分红",
        EventType.SPLIT: "送转", EventType.RIGHTS: "配股",
        EventType.IPO_ALLOT: "打新", EventType.OTHER: "其他",
    }
    c = Counter(e.etype for e in ds.events)
    return {cn.get(k, str(k)): v for k, v in c.items()}


def _invested(ds: Dataset) -> Decimal:
    """「动用过的本金」= 所有买入与配股缴款的金额（含费用）。

    为什么用它做收益率分母：交割单没有每日持仓市值，
    用「动用过的本金」是唯一能从流水自证的分母。
    它会让收益率看起来偏低（因为资金不是全程占用），所以报告里必须写明口径。
    """
    from .model import EventType

    total = Decimal("0")
    for e in ds.events:
        if e.etype == EventType.BUY:
            total += e.qty * e.price + e.fee
        elif e.etype == EventType.RIGHTS:
            total += e.cash if e.cash > 0 else e.qty * e.price
    return money(total)


def _by_symbol(chunks: list[RealizedChunk]) -> list[tuple[str, int, Decimal, int]]:
    agg: dict[str, dict] = {}
    for c in chunks:
        d = agg.setdefault(c.symbol, {"n": 0, "pnl": Decimal("0"), "win": 0})
        d["n"] += 1
        d["pnl"] += c.pnl
        if c.is_win:
            d["win"] += 1
    rows = [(s, d["n"], d["pnl"], d["win"]) for s, d in agg.items()]
    rows.sort(key=lambda r: r[2])
    return rows

