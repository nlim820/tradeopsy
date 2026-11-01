"""交易日历 —— 把「持有天数」从自然日口径纠正为 A 股交易日口径。

为什么必须有这个模块（P5 技术债，本轮偿还）：

    自然日算持有期会**系统性高估**，而且是双向失真：
      - 周五买、周一卖，自然日是 3 天，实际只持有 1 个交易日；
      - 春节前买、节后卖，自然日 9 天，实际交易日可能只有 1 天。
    散户多数持仓就在几天到几周，周末与长假占比很高，
    用自然日算出来的"平均持有 52 天"可能实际只有 36 个交易日。
    在 Odean 口径的论文里，持有期是解释处置效应的常用控制变量，
    口径错了，"我严格按学术口径算的"这句话就站不住。

三层数据源，可信度从高到低（务必按此优先级选）：

    1. **行情 bars 的日期序列**（最权威）
       指数（如沪深300）不会停牌，所以它的每一根日线就是一个交易日。
       有行情时一律用这个 —— 这是我们唯一能自证的交易日定义。
    2. **工作日近似 + 可选节假日补丁**
       周一至周五，减去传入的 holidays。适合完全离线的场景。
       误差来源于法定节假日：每年约 11 天休市，占全年交易日 ~4.4%，
       即持有期会被轻微高估；比自然日的 ~30% 高估好一个数量级。
    3. ~~交割单自身的日期~~（**绝不能用**）
       这是个常见陷阱：交割单只在你**有交易**的日子才有记录。
       三个月不交易，日历里就缺三个月，持有期会被严重低估。
       本模块不提供、也故意不提供这种构造方式。

设计上仍然零第三方依赖、零联网：日历由调用方构造后注入，
`analytics` 与 `ledger` 都只消费这个对象，可独立测试。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

__all__ = ["TradingCalendar", "weekdays_between"]


def weekdays_between(start: date, end: date) -> list[date]:
    """[start, end] 闭区间内的所有工作日（周一至周五），不含节假日判断。"""
    if end < start:
        return []
    out: list[date] = []
    cur = start
    step = timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += step
    return out


@dataclass(frozen=True)
class TradingCalendar:
    """一个不可变的交易日集合，按升序保存，支持 O(log n) 区间计数。

    语义约定（与 Odean 口径一致）：
        count(start, end) 统计的是 **(start, end] 半开区间**内的交易日数，
        即"建仓日之后、平仓日当天（含）"有多少个交易日。
        建仓日当天不计入 —— 因为那天你只是买入，还没开始"持有"。
        当日买当日卖（T+0 品种）因此得到 0 天，符合直觉。
    """

    days: tuple[date, ...]
    source: str  # "bars" / "weekday_approx" / "explicit"

    # ---------- 构造 ----------

    @classmethod
    def from_dates(cls, dates: Iterable[date], source: str = "explicit") -> "TradingCalendar":
        """从一组日期构造（自动去重升序）。

        典型用法：把行情 bars 的 `.dt` 收集起来传进来。
        """
        return cls(days=tuple(sorted(set(dates))), source=source)

    @classmethod
    def weekday_approx(
        cls,
        start: date,
        end: date,
        holidays: Iterable[date] | None = None,
    ) -> "TradingCalendar":
        """工作日近似：周一至周五，减去 holidays。

        holidays 传法定休市日（可跨年）。不传时只剔周末，
        误差约 4%（每年 11 天节假日 / 250 交易日），已在 source 中标注。
        """
        hol = set(holidays or ())
        days = [d for d in weekdays_between(start, end) if d not in hol]
        return cls(days=tuple(days), source="weekday_approx")

    # ---------- 查询 ----------

    def __len__(self) -> int:
        return len(self.days)

    @property
    def span(self) -> tuple[date, date] | None:
        """日历覆盖的 [首日, 末日]；空日历返回 None。"""
        if not self.days:
            return None
        return (self.days[0], self.days[-1])

    def is_trading_day(self, d: date) -> bool:
        i = bisect_right(self.days, d)
        return i > 0 and self.days[i - 1] == d

    def count(self, start: date, end: date) -> int:
        """(start, end] 内的交易日数。无交易日历覆盖时返回 0，由调用方判断是否可信。"""
        if end <= start:
            return 0
        return bisect_right(self.days, end) - bisect_right(self.days, start)

    def covers(self, start: date, end: date) -> bool:
        """该区间是否被日历完整覆盖（用于识别"日历比数据短"的降级情形）。

        容差设计：平仓日落在日历末日之后，但中间全是周末时，视为完整覆盖。
        否则「周五买、周日才平仓」这种（合法的周末跨期）会被误报为数据缺失，
        噪音提示多了，用户就会开始无视所有提示。
        """
        sp = self.span
        if sp is None:
            return False
        if start < sp[0]:
            return False
        if end <= sp[1]:
            return True
        tail = weekdays_between(sp[1] + timedelta(days=1), end)
        return not tail

    def count_checked(self, start: date, end: date) -> tuple[int, bool]:
        """返回 (交易日数, 是否完整覆盖)。

        不完整覆盖时数字可能偏小（日历截断），调用方应据此降低结论强度，
        而不是拿一个偏小的数当真。
        """
        return self.count(start, end), self.covers(start, end)

    # ---------- 展示 ----------

    def describe(self) -> str:
        sp = self.span
        if sp is None:
            return "空交易日历"
        label = {
            "bars": "行情K线日期（权威）",
            "weekday_approx": "工作日近似（未剔法定节假日，约 +4% 误差）",
            "explicit": "显式给定日期",
        }.get(self.source, self.source)
        return f"{sp[0]} → {sp[1]} 共 {len(self.days)} 个交易日 · 来源：{label}"
