"""数据装配层：从「磁盘上的交割单」到「可用的分析对象」。

为什么单独一层，而不让 `__main__` 和 `report` 各自读文件：

    文本报告和 HTML 报告必须是同一批数字。一旦两边各自解析、各自去重，
    迟早会出现"命令行说赚 2.3 万、网页说赚 2.4 万"这类无法解释的差异，
    而且这种 bug 极难被发现 —— 两个输出看起来都"合理"。

    所以解析、去重、建账、日历、处置效应全部收敛到这里，
    两个渲染层只做一件事：把同一份 Dataset 画成不同形状。

这层刻意不联网、不读环境变量、不写文件，纯函数式：给路径，返回结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from .analytics.benchmark import BenchmarkReport, compare_to_benchmarks
from .analytics.disposition import (
    DispositionReport,
    OpenPositionMV,
    compute_disposition,
    open_positions_from_ledger,
)
from .analytics.rolling import RollingReport, rolling_disposition
from .analytics.significance import (
    SignificanceResult,
    bootstrap_disposition,
    de_interval,
)
from .ledger import PositionLedger, summarize, timeline
from .model import Event
from .parsers import parse_file
from .parsers.base import ParseResult
from .prices import PriceHistory
from .tradingcal import TradingCalendar

__all__ = ["Dataset", "load_dataset", "read_calendar", "read_prices", "dedupe"]

DATA_SUFFIXES = (".csv", ".txt", ".tsv")


def dedupe(events: list[Event]) -> tuple[list[Event], int]:
    """跨文件去重。

    真实场景：用户按月导出，相邻两个文件的区间可能重叠几天
    （比如 3 月文件带上了 4 月 1 日的成交）。不去重就是把同一笔算两遍，
    而盈亏会直接翻倍 —— 这种错误看起来很"合理"，最难被发现。
    """
    seen: set[tuple] = set()
    out: list[Event] = []
    dup = 0
    for e in sorted(events, key=lambda x: (x.dt, x.symbol, x.etype.value)):
        key = (e.dt, e.symbol, e.etype, e.qty, e.price, e.fee, e.cash)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(e)
    return out, dup


def collect_inputs(target: Path) -> list[Path]:
    """把一个目录或单个文件展开成待解析文件列表。"""
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and not p.name.startswith(".")
        )
    return [target] if target.exists() else []


def read_prices(path: str) -> dict[str, Decimal]:
    """读未平仓标的最新价文件。

    支持带表头（证券代码/代码/symbol + 最新价/价格）或不带表头（第 1 列代码、第 2 列价）。
    行内若混入千分位逗号先去掉。
    """
    import csv

    out: dict[str, Decimal] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return out
    header = [c.strip() for c in rows[0]]
    data = rows[1:] if header and header[0] in ("证券代码", "代码", "symbol", "Symbol") else rows
    for r in data:
        if len(r) < 2:
            continue
        sym = r[0].strip()
        try:
            val = Decimal(r[1].strip().replace(",", ""))
        except Exception:
            continue
        if sym:
            out[sym] = val
    return out


def read_calendar(path: str) -> TradingCalendar | None:
    """读交易日历文件：每行一个 YYYY-MM-DD（可带表头，非日期行自动跳过）。

    为什么用「日期列表」而不是内置节假日表：
        A 股休市安排每年由交易所公告，硬编码一张表必然过期，
        而过期的日历会静默算错持有期 —— 比没有日历更危险。
        行情 bars 的日期序列是最权威的交易日定义（指数不会停牌）。
    """
    p = Path(path)
    if not p.exists():
        return None
    dates: list[date] = []
    with p.open(encoding="utf-8-sig") as fh:
        for raw in fh:
            tok = raw.strip().split(",")[0].strip().strip('"').strip("'")
            if not tok:
                continue
            try:
                dates.append(date.fromisoformat(tok[:10]))
            except ValueError:
                continue
    if not dates:
        return None
    return TradingCalendar.from_dates(dates, source="bars")


def read_benchmark(path: str) -> dict[date, Decimal]:
    """读基准净值/收盘价序列：每行 `YYYY-MM-DD,数值`（可带表头）。

    基准是**唯一**一个交割单无法自证的东西 —— 交割单只记录你自己的成交，
    不记录同期大盘涨跌。所以它必须由外部提供，缺了就明确说缺，不估。
    """
    import csv

    out: dict[date, Decimal] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 2:
                continue
            try:
                d = date.fromisoformat(r[0].strip()[:10])
                v = Decimal(r[1].strip().replace(",", ""))
            except Exception:
                continue
            out[d] = v
    return out


@dataclass
class Dataset:
    """一次分析的完整结果。两个渲染层（文本 / HTML）共用它。"""

    paths: list[Path]
    events: list[Event]
    ledger: PositionLedger
    file_metas: list[tuple[str, dict]] = field(default_factory=list)
    dup: int = 0
    #: 解析层告警（账本告警单独从 ledger.warnings 取，两者用途不同，不在这里合并）
    warnings: list[str] = field(default_factory=list)
    calendar: TradingCalendar | None = None

    @property
    def all_warnings(self) -> list[str]:
        """解析层 + 账本层的全部告警。"""
        return list(self.warnings) + list(self.ledger.warnings)

    @property
    def span(self) -> tuple[date, date] | None:
        return timeline(self.events)

    @property
    def symbols(self) -> set[str]:
        return {e.symbol for e in self.events}

    def summary(self) -> dict:
        return summarize(self.ledger, self.calendar)

    def open_positions(self, prices: dict[str, Decimal] | None = None) -> list[OpenPositionMV]:
        return open_positions_from_ledger(self.ledger, prices)

    def disposition(
        self,
        prices: dict[str, Decimal] | None = None,
        with_paper: bool | None = None,
    ) -> DispositionReport:
        """处置效应。with_paper=None 时按"有没有给价格"自动决定。"""
        opens = self.open_positions(prices)
        if with_paper is None:
            with_paper = bool(prices)
        return compute_disposition(
            self.ledger.chunks,
            opens if with_paper else None,
            trading_calendar=self.calendar,
        )

    def significance(self, prices: dict[str, Decimal] | None = None) -> SignificanceResult:
        return bootstrap_disposition(self.ledger.chunks, self.open_positions(prices))

    def rolling(
        self,
        price_history: "PriceHistory | None",
        *,
        max_stale_days: int = 30,
    ) -> RollingReport | None:
        """逐卖出日滚动口径。price_history 为空时返回 None（与"算了但无结果"区分开）。

        为什么返回 None 而不是空报告：调用方要能区分
        「用户没给价格序列」和「给了但算不出来」—— 前者不该在报告里
        出现一个"数据不足"的板块，那只会制造噪音。
        """
        if not price_history:
            return None
        return rolling_disposition(
            self.events,
            price_history,
            trading_calendar=self.calendar,
            max_stale_days=max_stale_days,
        )

    def benchmark(self, roll: RollingReport | None) -> BenchmarkReport:
        """同行基准对照。

        只有逐日滚动口径与已发表文献同构（都是「每个卖出日」计数口径），
        所以这里由 roll 的成败决定能否对照 —— 期末快照口径一律拒绝，
        理由见 analytics/benchmark.py 的模块文档。
        """
        if roll is None:
            return compare_to_benchmarks(caliber="none")
        if roll.mode != "rolling":
            return compare_to_benchmarks(caliber="insufficient")

        # 复用同一套 bootstrap 核心，区间与 DE 来自同一批计数
        interval = de_interval(
            roll.n_gain_sold, roll.n_gain_opp, roll.n_loss_sold, roll.n_loss_opp
        )
        return compare_to_benchmarks(
            caliber="rolling",
            my_de=roll.de,
            ci_low=interval.ci_low,
            ci_high=interval.ci_high,
            my_pgr=roll.pgr,
            my_plr=roll.plr,
            my_n_obs=roll.n_obs,
        )


def load_dataset(
    target: Path,
    *,
    calendar_path: str | None = None,
    natural_days: bool = False,
) -> Dataset:
    """读目录/文件 → 事件流 → 账本 → 交易日历。

    返回的 Dataset 即使 0 条事件也合法（调用方据此给出诊断信息）。
    """
    paths = collect_inputs(target)
    all_events: list[Event] = []
    warnings: list[str] = []
    metas: list[tuple[str, dict]] = []
    for p in paths:
        r = parse_file(p)
        all_events.extend(r.events)
        warnings.extend(r.warnings)
        metas.append((p.name, dict(r.meta)))

    merged = ParseResult(events=all_events, warnings=warnings)
    events, dup = dedupe(merged.events)

    ledger = PositionLedger()
    ledger.apply_all(events)

    span = timeline(events)
    calendar: TradingCalendar | None = None
    if not natural_days:
        if calendar_path:
            calendar = read_calendar(calendar_path)
        if calendar is None and span is not None:
            calendar = TradingCalendar.weekday_approx(span[0], span[1])

    return Dataset(
        paths=paths,
        events=events,
        ledger=ledger,
        file_metas=metas,
        dup=dup,
        warnings=warnings,
        calendar=calendar,
    )
