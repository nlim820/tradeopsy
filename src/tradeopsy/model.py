"""数据模型。

一条券商交割单的原始记录，先被翻译成一个 `Event`；
账本（ledger.py）消费 Event，产出 `RealizedChunk`（已实现片段）。

为什么要分 Event 和 Chunk 两层：
    Event   = 发生了什么（买入/卖出/送转/分红/配股）
    Chunk   = 卖出的钱是怎么赚的（对应到哪一批买入，持有多少天）
一个卖出动作可能横跨多个买入批次，所以一对多。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

CENT = Decimal("0.01")
UNIT = Decimal("0.0001")


def money(x) -> Decimal:
    """金额保留 2 位小数（到分）。

    显式指定 ROUND_HALF_UP 而不是默认的 ROUND_HALF_EVEN。
    中国财务惯例是「四舍五入」，而 Python 的默认银行家舍入会把 8.925 变成 8.92，
    这种差异会让你的报告和券商 App 对不上几分钱，进而被用户当成 bug。
    """
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def unit_price(x) -> Decimal:
    """每股成本保留 4 位小数（仅用于展示）。

    注意：每股成本**不是**权威数据。权威是「总成本」和「总股数」。
    原因见 Lot 的注释 —— 用每股成本做乘法会累积舍入误差。
    """
    return Decimal(x).quantize(UNIT, rounding=ROUND_HALF_UP)


class EventType(str, Enum):
    """交割单里能出现的业务类型。

    A 股交割单的「业务名称」列远不止买卖，下面这些是最容易让简单脚本算错账的。
    """

    BUY = "buy"                 # 证券买入
    SELL = "sell"               # 证券卖出
    DIVIDEND = "dividend"       # 股息入账 / 红利入账
    SPLIT = "split"             # 红股入账 / 送股 / 转增（数量增加，总成本不变）
    RIGHTS = "rights"           # 配股缴款（掏钱换新股）
    IPO_ALLOT = "ipo_allot"     # 新股入账 / 中签（打新中签后股份到账）
    OTHER = "other"             # 其它（担保品划转、基金申赎等，先归到这里）


#: 交割单「业务名称」中文关键词 -> 事件类型。
#: 顺序有意义：先匹配更具体的词，避免「红股入账」被「股份」泛匹配掉。
BUSINESS_NAME_RULES: tuple[tuple[tuple[str, ...], EventType], ...] = (
    (("红股入账", "送股入账", "转增", "红股"), EventType.SPLIT),
    (("股息入账", "红利入账", "股息红利", "派息"), EventType.DIVIDEND),
    (("配股缴款", "配股"), EventType.RIGHTS),
    (("新股入账", "中签", "申购中签"), EventType.IPO_ALLOT),
    (("证券买入", "买入", "担保品买入"), EventType.BUY),
    (("证券卖出", "卖出", "担保品卖出"), EventType.SELL),
)


def classify_business(raw: str) -> EventType:
    """把交割单上的「业务名称」翻译成事件类型。

    匹配不上时返回 OTHER 而不是抛异常 —— 因为真实交割单里总会有你没见过的条目，
    整批数据不应该因为一行看不懂就全丢掉。未识别条目会被记进 warnings 供人工复核。
    """
    text = (raw or "").strip()
    for keywords, etype in BUSINESS_NAME_RULES:
        if any(k in text for k in keywords):
            return etype
    return EventType.OTHER


@dataclass(frozen=True)
class Event:
    """标准化后的单个交易事件。

    qty / price / fee / cash 四个字段的语义按 etype 不同而不同：
        BUY       qty=股数, price=成交价, fee=买入费用(佣金+过户费), cash=0
        SELL      qty=股数, price=成交价, fee=卖出费用,               cash=0
        DIVIDEND  qty=0,   price=0,   fee=红利税(若有),              cash=到账金额
        SPLIT     qty=新增股数, price=0, fee=0,                      cash=0
        RIGHTS    qty=配股股数, price=配股价, fee=0,                 cash=缴款金额
        IPO_ALLOT qty=股数,   price=发行价, fee=0,                   cash=0
    """

    dt: date
    symbol: str
    etype: EventType
    qty: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    note: str = ""

    def __post_init__(self) -> None:
        # frozen=True 的对象只能用 object.__setattr__ 赋值
        object.__setattr__(self, "qty", Decimal(self.qty))
        object.__setattr__(self, "price", Decimal(self.price))
        object.__setattr__(self, "fee", Decimal(self.fee))
        object.__setattr__(self, "cash", Decimal(self.cash))
        object.__setattr__(self, "symbol", str(self.symbol).strip())


@dataclass
class Lot:
    """一个买入批次。这是 FIFO 配对的原子单位。

    关键设计：以 (qty, cost) 为权威，unit_cost 只作展示。

    为什么不用 unit_cost 做权威：
        10 转 5 之后每股成本变成 10.0051 / 1.5 = 6.6700666...，
        保留 4 位后是 6.6701，再乘回 1500 股得到 10005.15，
        凭空多出 0.05 元。股数越多、送转次数越多，漂移越大。
        而以「总成本 10005.10 不变」为准，则永远不会漂移。
    """

    dt: date
    qty: Decimal
    cost: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")

    @property
    def unit_cost(self) -> Decimal:
        """摊薄后的每股成本（仅供展示，不参与后续计算）。"""
        if self.qty <= 0:
            return Decimal("0.0000")
        return unit_price(self.cost / self.qty)


@dataclass
class RealizedChunk:
    """一笔已实现盈亏。

    粒度是「一个卖出动作 × 一个买入批次」，所以一次卖出可能产生多个 Chunk。
    选择这个粒度而不是「从建仓到清仓算一个回合」，原因是：
      - 分批建仓、分批止盈在 A 股散户里非常普遍，
        强行合成一个回合会掩盖「哪一批亏、哪一批赚」；
      - Odean (1998) 的 PGR/PLR 本身就是按「卖出动作」定义的，粒度对齐。
    """

    symbol: str
    open_dt: date
    close_dt: date
    qty: Decimal
    cost: Decimal          # 已实现部分对应的买入成本
    proceeds: Decimal      # 卖出净额（已扣卖出费用）
    sell_fee: Decimal = Decimal("0")
    note: str = ""

    @property
    def pnl(self) -> Decimal:
        return money(self.proceeds - self.cost)

    @property
    def pnl_pct(self) -> Decimal:
        if self.cost == 0:
            return Decimal("0")
        return (self.proceeds / self.cost - 1).quantize(Decimal("0.000001"))

    @property
    def hold_days(self) -> int:
        return (self.close_dt - self.open_dt).days

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class OpenPosition:
    """当前仍持有的某一批次（仅用于输出持仓快照）。"""

    symbol: str
    open_dt: date
    qty: Decimal
    cost: Decimal

    @property
    def unit_cost(self) -> Decimal:
        if self.qty <= 0:
            return Decimal("0.0000")
        return unit_price(self.cost / self.qty)
