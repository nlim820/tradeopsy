"""行情获取层。

三个设计约束，都是为了能在 GitHub Actions 的免费 Runner 上活下去：

1. **零第三方依赖**。只用了标准库 urllib。在 Actions 上 `pip install akshare`
   要 1-2 分钟，而且 akshare 本身又依赖一堆东西，失败率不低。
   我们每天只需要 8 只 ETF 的日线，一个 GET 请求就够，没必要拖这么重的船。

2. **多源回退**。腾讯（qfq 前复权）为主，新浪（不复权）兜底，东财排最后。
   理由很实际：Actions 的共享 Runner IP 段经常被财经站点限流，
   单源方案会在某个周二早上突然挂掉，而你不会立刻发现。

   顺序不是拍脑袋定的，是实测结果：2026-11-01 在本机网络下
   东财三个 K 线域名全部连接即断（同域名的实时接口却正常，说明是路径级限制），
   网易 chddata 返回 502。腾讯和新浪稳定可用，所以它们排前面。

   **兜底源是不复权数据**，所以 Bar 带了 adjusted 标记，
   runner 会据此启动除权跳空检测 —— 不复权序列里的一次分红除权，
   在均线眼里就是一根 -3% 的假阴线，足够把策略骗去清仓。

3. **统一返回前复权序列**。
   这条最容易被忽略：不复权价在除权日会跳空（510300 分红后价格直接掉一截），
   双均线会把这次跳空当成"暴跌"从而发出错误卖出信号。
   而前复权序列是连续的，均线才有意义。

   **为什么成交也能用前复权价**：前复权是以"最新交易日"为基准往前调整的，
   所以序列最后一天的 OHLC 就等于当天的真实成交价，不会失真。
   这是很多人搞混的一点 —— 前复权只改历史，不改当下。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TIMEOUT = 15
_RETRY = 2


class FetchError(RuntimeError):
    """所有数据源都失败时抛出。runner 应该捕获它并让本次运行整体失败、不落账。"""


@dataclass(frozen=True)
class Bar:
    """一根日线。价格全部转成 Decimal —— 撮合要算钱，float 会给你 0.1+0.2 的惊喜。"""

    dt: date
    open: Decimal
    close: Decimal
    high: Decimal
    low: Decimal
    volume: Decimal = Decimal("0")
    #: 是否为前复权价。False 时必须依赖除权跳空检测，否则均线会被除权扭曲。
    adjusted: bool = True

    def __post_init__(self) -> None:
        for f in ("open", "close", "high", "low", "volume"):
            object.__setattr__(self, f, Decimal(str(getattr(self, f))))

    @property
    def is_flat(self) -> bool:
        """一字板 / 停牌价不动。这种日子撮合必须跳过，否则你会买到根本买不到的票。"""
        return self.high == self.low


def _http_json(url: str, retry: int = _RETRY) -> dict:
    last: Exception | None = None
    for attempt in range(retry + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": _UA,
                    "Referer": "https://quote.eastmoney.com/",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retry:
                time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"请求失败: {url[:80]}... ({last})")


def _secid(symbol: str) -> str:
    """东财的证券 ID：沪市前缀 1，深市前缀 0。

    ETF 代码规律：5 开头（510300/512880/518880/588000）在沪市，
    1 开头（159915）在深市。股票同理（6 → 沪，0/3 → 深）。
    """
    s = symbol.strip()
    if s.startswith(("5", "6", "9")):
        return f"1.{s}"
    return f"0.{s}"


def _fetch_eastmoney(symbol: str, start: date, end: date) -> list[Bar]:
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={_secid(symbol)}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
        "&klt=101&fqt=1"  # klt=101 日线, fqt=1 前复权
        f"&beg={start:%Y%m%d}&end={end:%Y%m%d}"
        "&lmt=1000000"
    )
    payload = _http_json(url)
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    bars: list[Bar] = []
    for line in klines:
        p = line.split(",")
        if len(p) < 6:
            continue
        bars.append(
            Bar(
                dt=datetime.strptime(p[0], "%Y-%m-%d").date(),
                open=p[1],
                close=p[2],
                high=p[3],
                low=p[4],
                volume=p[5],
            )
        )
    return bars


def _fetch_tencent(symbol: str, start: date, end: date) -> list[Bar]:
    s = symbol.strip()
    prefix = "sh" if s.startswith(("5", "6", "9")) else "sz"
    # 腾讯这个接口一次最多给 ~640 根，多要几次没有意义，日线场景够用
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={prefix}{s},day,{start:%Y-%m-%d},{end:%Y-%m-%d},640,qfq"
    )
    payload = _http_json(url)
    node = (payload.get("data") or {}).get(f"{prefix}{s}") or {}
    rows = node.get("qfqday") or node.get("day") or []
    bars: list[Bar] = []
    for p in rows:
        if len(p) < 6:
            continue
        # 腾讯顺序：日期, 开, 收, 高, 低, 成交量
        bars.append(
            Bar(
                dt=datetime.strptime(p[0], "%Y-%m-%d").date(),
                open=p[1],
                close=p[2],
                high=p[3],
                low=p[4],
                volume=p[5],
            )
        )
    return bars


def _fetch_sina(symbol: str, start: date, end: date) -> list[Bar]:
    """新浪日线。**不复权**，仅作兜底。

    datalen 固定给 1000（实测上限约 1000 根，约四年日线），
    够覆盖 500 自然日的回看窗口还留足余量。
    """
    s = symbol.strip()
    prefix = "sh" if s.startswith(("5", "6", "9")) else "sz"
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={prefix}{s}&scale=240&ma=no&datalen=1000"
    )
    rows = _http_json(url)
    if not isinstance(rows, list):
        return []
    bars: list[Bar] = []
    for r in rows:
        try:
            d = datetime.strptime(r["day"][:10], "%Y-%m-%d").date()
        except (KeyError, ValueError, TypeError):
            continue
        if d < start or d > end:
            continue
        bars.append(
            Bar(
                dt=d,
                open=r["open"],
                close=r["close"],
                high=r["high"],
                low=r["low"],
                volume=r.get("volume", 0),
                adjusted=False,
            )
        )
    return bars


#: 数据源按顺序尝试。加新源只需往这里塞一个 (名字, 函数)。
#: 第一个返回非空结果者胜出，后面的不再请求。
SOURCES: tuple[tuple[str, object], ...] = (
    ("tencent", _fetch_tencent),
    ("sina", _fetch_sina),
    ("eastmoney", _fetch_eastmoney),
)

#: 记录最近一次实际使用的数据源，供 runner 打印诊断信息。
LAST_SOURCE: dict[str, str] = {}


def fetch_bars(symbol: str, start: date, end: date) -> list[Bar]:
    """按 SOURCES 顺序取日线，第一个成功且非空的结果胜出。

    注意"非空"这个条件：东财在限流时不会报错，而是返回 `data: null`，
    看起来请求成功了但里面什么都没有。只判断异常会漏掉这种情况。
    """
    errors: list[str] = []
    for name, fn in SOURCES:
        try:
            bars = fn(symbol, start, end)  # type: ignore[operator]
        except FetchError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if bars:
            LAST_SOURCE[symbol] = name
            return bars
        errors.append(f"{name}: 返回空数据")
    raise FetchError(f"{symbol} 全部数据源失败 -> {'; '.join(errors)}")


def fetch_many(
    symbols: list[str], start: date, end: date, pause: float = 0.35
) -> dict[str, list[Bar]]:
    """批量取数，带节流。

    pause 默认 0.35 秒：8 只 ETF 就是 3 秒，慢一点但不容易被限流。
    免费接口上"快"不是优点，能稳定跑完一年才是。
    """
    out: dict[str, list[Bar]] = {}
    for i, sym in enumerate(symbols):
        out[sym] = fetch_bars(sym, start, end)
        if i < len(symbols) - 1:
            time.sleep(pause)
    return out


def suspicious_gaps(
    bars: list[Bar], threshold: Decimal = Decimal("0.11")
) -> list[tuple[date, Decimal]]:
    """找出单日跌幅超过阈值的日子，作为"疑似除权"的预警。

    为什么需要这个：如果走到不复权的兜底源，一次 2% 的分红除权在均线上
    看起来就是一根阴线，均线会缓慢向下偏移，策略可能因此提前清仓 ——
    而且整个过程没有任何报错，你只会觉得"策略最近不太行"。

    阈值的取值：ETF 涨跌幅限制 10%（创业板/科创板 ETF 20%），
    所以 11% 以上的单日跌幅在宽基 ETF 上几乎不可能是正常行情。

    不做自动修补。原因：猜错复权因子比不修更危险，
    而这类事件一年也就一两次，人工确认成本极低。
    """
    out: list[tuple[date, Decimal]] = []
    for prev, cur in zip(bars, bars[1:]):
        if prev.close <= 0:
            continue
        chg = cur.close / prev.close - Decimal("1")
        if chg < -threshold:
            out.append((cur.dt, chg))
    return out
