"""撮合与账户状态。

这个模块是整条管道里唯一"会算错钱"的地方，所以规则写得很死，每条都注明理由。

四条硬规则：
1. **T+1**：当日买入的份额当日不可卖。用 available（可卖）与 qty（持仓）两个字段分开记。
2. **一字板不成交**：开盘=最高=最低的日子，撮合直接跳过并记一条 rejected 理由。
   如果不做这个判断，账户会在根本买不进去的日子"成功买入"，
   净值曲线会比真实情况好看，而你还以为策略很神。
3. **佣金最低 5 元**：这是真实券商规则，不是保守假设。
   高频小额调仓在 5 元保底价下会被吃得非常惨 —— 让账户真实承担这个成本，
   你才会对"少交易"有切身体感。
4. **ETF 免印花税、免过户费**：所以费用只有佣金一项。
   这也是选纯 ETF 做标的池的附带好处 —— 费用模型简单到不会算错。

还有一个顺序规则：**先卖后买**。买入前必须先拿到卖出释放的现金，
否则目标仓位切换时会因为"钱不够"而卡住，需要等第二天。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

from ..model import Event, EventType, money
from .market import Bar

#: 佣金费率（万 2.5）与最低收费。券商之间差异不大，这个值属于行业常见档。
COMMISSION_RATE = Decimal("0.00025")
MIN_COMMISSION = Decimal("5.00")

#: A 股 ETF 最小交易单位。卖出时可以不足 100 股（一次性清掉零股），
#: 但买入必须是 100 的整数倍。
LOT_SIZE = 100

#: 调仓带宽：差额小于总资产这个比例时不动手。
#: 不设带宽的后果是每天为了对齐权重反复下百元碎单，
#: 而每笔碎单都要付 5 元最低佣金 —— 一万块本金一年能被吃走 1%。
REBALANCE_BAND = Decimal("0.03")


def _rebalance_threshold(total: Decimal, price: Decimal) -> Decimal:
    """低于这个金额的差额不调仓。至少一手，且不低于总资产的 3%。"""
    return max(money(total * REBALANCE_BAND), money(price * LOT_SIZE))


def commission(amount: Decimal) -> Decimal:
    """佣金 = max(成交额 × 万2.5, 5元)。

    最低 5 元这条在纸面账户里特别重要：
    如果你只买 5000 元，费率实际是 0.1%，是名义费率的 4 倍。
    很多回测把这些成本忽略掉，得出的结论和真实操作完全两回事。
    """
    if amount <= 0:
        return Decimal("0.00")
    return max(money(amount * COMMISSION_RATE), MIN_COMMISSION)


@dataclass
class Position:
    """一个持仓。qty 与 available 分开是为了 T+1。"""

    symbol: str
    name: str
    qty: Decimal = Decimal("0")
    available: Decimal = Decimal("0")
    last_price: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for f in ("qty", "available", "last_price"):
            object.__setattr__(self, f, Decimal(str(getattr(self, f))))

    @property
    def market_value(self) -> Decimal:
        return money(self.qty * self.last_price)


@dataclass
class Fill:
    """一次撮合的结果。reason 为空表示成功。"""

    dt: date
    symbol: str
    name: str
    side: str          # "buy" / "sell"
    qty: Decimal
    price: Decimal
    fee: Decimal
    amount: Decimal
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.reason

    def to_event(self) -> Event:
        return Event(
            dt=self.dt,
            symbol=self.symbol,
            etype=EventType.BUY if self.side == "buy" else EventType.SELL,
            qty=self.qty,
            price=self.price,
            fee=self.fee,
            note="paper",
        )


@dataclass
class Account:
    """纸面账户状态。整个对象就是 state.json 的内容。"""

    cash: Decimal = Decimal("100000.00")
    initial_cash: Decimal = Decimal("100000.00")
    as_of: str = ""
    positions: dict[str, Position] = field(default_factory=dict)
    #: 上一交易日收盘后定下、等待今日开盘执行的目标权重
    pending_from: str = ""
    pending_targets: dict[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash", Decimal(str(self.cash)))
        object.__setattr__(self, "initial_cash", Decimal(str(self.initial_cash)))

    # ---------- 估值 ----------

    def total_value(self) -> Decimal:
        return money(self.cash + sum(p.market_value for p in self.positions.values()))

    def equity(self) -> Decimal:
        return self.total_value() - self.initial_cash

    # ---------- 序列化 ----------
    # 用 JSON 而不是 pickle：state 要 commit 进仓库、要能被 diff 看得懂。
    # pickle 存进去的是二进制，出问题时你连哪一天账户不对都查不出来。

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "as_of": self.as_of,
            "cash": str(money(self.cash)),
            "initial_cash": str(money(self.initial_cash)),
            "positions": {
                s: {
                    "name": p.name,
                    "qty": str(p.qty),
                    "available": str(p.available),
                    "last_price": str(p.last_price),
                }
                for s, p in self.positions.items()
            },
            "pending_from": self.pending_from,
            "pending_targets": {s: str(w) for s, w in self.pending_targets.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Account":
        acc = cls(
            cash=Decimal(str(raw.get("cash", "100000.00"))),
            initial_cash=Decimal(str(raw.get("initial_cash", "100000.00"))),
            as_of=raw.get("as_of", ""),
            pending_from=raw.get("pending_from", ""),
            pending_targets={
                s: Decimal(str(w))
                for s, w in (raw.get("pending_targets") or {}).items()
            },
        )
        for sym, p in (raw.get("positions") or {}).items():
            if Decimal(str(p.get("qty", "0"))) > 0:
                acc.positions[sym] = Position(
                    symbol=sym,
                    name=p.get("name", sym),
                    qty=p.get("qty", "0"),
                    available=p.get("available", "0"),
                    last_price=p.get("last_price", "0"),
                )
        return acc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, initial_cash: str = "100000.00") -> "Account":
        if not path.exists():
            return cls(cash=Decimal(initial_cash), initial_cash=Decimal(initial_cash))
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def roll_day(account: Account) -> None:
    """推进到新交易日：把昨日持仓全部解锁为可卖（T+1 到期）。

    为什么单独抽一个函数，而不是写在 execute_targets 开头：
    后来的是“每日一次”的语义，而撮合可能被调多次。
    若把解锁藏在撮合里，同一天第二次撮合就会把当日刚买入的份额当成可卖，
     T+1 静默失效 —— 而这种错误在净值曲线上看不出来，只会在某天你
    “当日买入又当日卖出”时才发现，那时已经无从查证。
    """
    for pos in account.positions.values():
        pos.available = pos.qty


def execute_targets(
    account: Account,
    targets: dict[str, Decimal],
    bars_today: dict[str, Bar],
    names: dict[str, str],
    today: date,
) -> list[Fill]:
    """按目标权重在今日开盘价上撮合。

    参数 bars_today 是"今天的 K 线"，撮合价格用它的 open。
    如果某只标的今天没有 K 线（停牌 / 未上市），它会从目标里被剔除，
    并且已有的持仓保持不动 —— 不能因为取不到数据就当它该清仓。
    """
    fills: list[Fill] = []
    bars = bars_today

    # ---- 1. 用今日开盘价给整个账户估值 ----
    for sym, pos in account.positions.items():
        bar = bars.get(sym)
        if bar is not None:
            pos.last_price = bar.open

    total = account.total_value()

    # ---- 2. 计算每只标的的目标市值与差额 ----
    plan: list[tuple[str, Decimal]] = []   # (symbol, 差额市值) 正买负卖
    target_syms: set[str] = set()
    for sym, weight in targets.items():
        bar = bars.get(sym)
        if bar is None:
            # 今日无行情（停牌/数据缺失）：既不买也不卖，保持原状。
            # 不能因为取不到数据就当成“该清仓”。
            continue
        target_syms.add(sym)
        target_value = money(total * weight)
        cur = account.positions.get(sym)
        cur_value = money(cur.qty * bar.open) if cur else Decimal("0")
        diff = target_value - cur_value
        if abs(diff) < _rebalance_threshold(total, bar.open):
            continue
        plan.append((sym, diff))

    # ---- 2b. 持仓但不在目标里的，必须清掉 ----
    # 这一段绝不能省。targets 为空字典就是“空仓”信号，
    # 如果只遍历 targets，现有持仓永远不会产生卖单 ——
    # 不报错、不告警，账户默默退化成买入持有，几个月后你才发现。
    # 清仓不走调仓带宽（不能用“差额太小”的理由无视一个卖出信号）。
    for sym, pos in list(account.positions.items()):
        if sym in target_syms:
            continue
        bar = bars.get(sym)
        if bar is None:
            continue
        plan.append((sym, -money(pos.qty * bar.open)))

    # ---- 3. 先卖：释放现金 ----
    for sym, diff in [p for p in plan if p[1] < 0]:
        bar = bars[sym]
        pos = account.positions.get(sym)
        if pos is None:
            continue
        want = min(pos.available, (-diff / bar.open))
        lots = (want / LOT_SIZE).to_integral_value(rounding=ROUND_FLOOR)
        qty = min(pos.available, lots * LOT_SIZE)
        # 允许一次性卖掉剩余零股（真实规则）
        if pos.available - qty < LOT_SIZE:
            qty = pos.available
        if qty <= 0:
            continue
        if bar.is_flat:
            fills.append(
                Fill(today, sym, pos.name, "sell", qty, bar.open,
                     Decimal("0"), Decimal("0"), "一字板，无法卖出")
            )
            continue
        amount = money(qty * bar.open)
        fee = commission(amount)
        account.cash += amount - fee
        pos.qty -= qty
        pos.available -= qty
        if pos.qty <= 0:
            account.positions.pop(sym, None)
        fills.append(
            Fill(today, sym, pos.name, "sell", qty, bar.open, fee, amount)
        )

    # ---- 4. 后买：用含卖出回款的现金 ----
    for sym, diff in [p for p in plan if p[1] > 0]:
        bar = bars[sym]
        budget = min(diff, account.cash)
        if budget <= 0:
            continue
        if bar.is_flat:
            fills.append(
                Fill(today, sym, names.get(sym, sym), "buy", Decimal("0"),
                     bar.open, Decimal("0"), Decimal("0"), "一字板，无法买入")
            )
            continue

        # 预算里要先扣掉佣金，否则会超支。反复减一手直到装得下为止。
        lots = (budget / (bar.open * LOT_SIZE)).to_integral_value(
            rounding=ROUND_FLOOR
        )
        qty = lots * LOT_SIZE
        while qty > 0 and money(qty * bar.open) + commission(money(qty * bar.open)) > account.cash:
            qty -= LOT_SIZE
        if qty <= 0:
            continue
        amount = money(qty * bar.open)
        fee = commission(amount)
        account.cash -= amount + fee
        pos = account.positions.get(sym) or Position(
            symbol=sym, name=names.get(sym, sym)
        )
        # 当日买入不增加 available（T+1）
        pos.qty += qty
        pos.last_price = bar.open
        account.positions[sym] = pos
        fills.append(
            Fill(today, sym, pos.name, "buy", qty, bar.open, fee, amount)
        )

    account.cash = money(account.cash)
    return fills


def mark_to_market(account: Account, closes: dict[str, Decimal]) -> None:
    """收盘后按收盘价重估持仓。这一步决定 nav.csv 的每日净值。"""
    for sym, pos in account.positions.items():
        px = closes.get(sym)
        if px is not None and px > 0:
            pos.last_price = px
