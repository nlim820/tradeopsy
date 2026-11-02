"""生成十套「模拟交割单」—— 每套围绕一种真实存在的散户行为画像构造。

**全部是合成数据。** 代码是真实存在的证券代码，但名称、价格、成交记录
均为构造值，只求结构逼真，不代表任何真实行情或真实交易。

为什么不是"随机生成"：
    完全随机的买卖记录跑出来的 PGR/PLR 会收敛到无意义的中间值，
    十份报告长得一样，既测不出引擎对行为结构的敏感度，也没有展示价值。
    所以每套数据围绕一种**行为画像**构造：赢家的持有期、输家的持有期、
    期末持仓的盈亏状态都被刻意设计，报告结论各有各的故事。

十种画像一览（详细见本目录 README.md）：
    p01 追涨杀跌型    强处置效应，短炒，笔数最多
    p02 反向处置型    截亏持盈，DE 为负（受过训练的样子）
    p03 长线价值型    极少交易，小样本 → 报告应输出"证据不足"
    p04 学生小账户    金额小、笔数少 → 同样"证据不足"
    p05 频繁拆单型    同标的分批建仓出货，压测 FIFO 配对链
    p06 ETF 定投型    只有 ETF，月定投 + 偶尔止盈
    p07 转债 T+0 型   当日买卖，压测 T+0 与 11/12 两个市场前缀
    p08 事件驱动型    送转/配股/分红/打新/担保品划转，结构最脏
    p09 熊市死扛型    重仓被套，几乎不割，账面深亏
    p10 无模式基线    有赚有亏无明显倾向，作为对照组

一致性保证（生成时自检，不通过直接报错）：
    1. 全部成交落在工作日；
    2. 股票/ETF 不违反 T+1（可转债 11/12 开头豁免）；
    3. 任何一笔卖出不超过当时持仓；
    4. 有期末价的标的期末仍有持仓、无期末价的标的已全部了结
       —— 否则价格序列与账本会自相矛盾。

用法：
    python examples/synthetic_accounts/generate.py    # 固定 seed，可复现

产物：
    pXX_名字/*.csv                    该人格的交割单（券商风格见 README）
    pXX_名字_price_history.csv        该人格的逐日收盘价（供 --price-history）
    reports/pXX_名字.html             （由外部脚本批量生成，见 README）
"""

from __future__ import annotations

import importlib.util
import math
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_shared():
    """加载 sample_account 的生成器，复用其格式写入与校验逻辑。

    用 importlib 按路径加载而不是 sys.path 导入：两个目录都有 generate.py，
    按名字导入会撞模块名。
    """
    spec = importlib.util.spec_from_file_location(
        "sa_generate", HERE.parent / "sample_account" / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SA = _load_shared()

D = Decimal
STOCK, ETF, BOND = SA.STOCK, SA.ETF, SA.BOND
expand_script = SA.expand_script
validate = SA.validate
write_ths = SA.write_ths
write_ht = SA.write_ht
money = SA.money

#: 统一的交易窗口与"今天"。所有人格共用，方便横向比较。
START = date(2025, 11, 3)    # 周一
END = date(2026, 10, 30)     # 周五

#: 标的池。代码真实存在，价格与名称均为构造。
POOL: dict[str, tuple[str, str]] = {
    # 股票
    "600519": ("贵州茅台", STOCK), "000858": ("五粮液", STOCK),
    "600036": ("招商银行", STOCK), "000001": ("平安银行", STOCK),
    "300750": ("宁德时代", STOCK), "002594": ("比亚迪", STOCK),
    "601318": ("中国平安", STOCK), "600030": ("中信证券", STOCK),
    "601899": ("紫金矿业", STOCK), "000651": ("格力电器", STOCK),
    "601728": ("中国电信", STOCK), "688981": ("中芯国际", STOCK),
    "002415": ("海康威视", STOCK), "600900": ("长江电力", STOCK),
    "601088": ("中国神华", STOCK), "600276": ("恒瑞医药", STOCK),
    "000333": ("美的集团", STOCK), "002714": ("牧原股份", STOCK),
    # ETF
    "510300": ("沪深300ETF", ETF), "588000": ("科创50ETF", ETF),
    "512100": ("中证1000ETF", ETF), "518880": ("黄金ETF", ETF),
    "159915": ("创业板ETF", ETF),
    # 可转债（11 沪 / 12 深，均 T+0）
    "113050": ("南银转债", BOND), "113065": ("路桥转债", BOND),
    "127098": ("创白转债", BOND), "123140": ("苏行转债", BOND),
    "110093": ("荆门转债", BOND), "118041": ("科创转债", BOND),
}

#: 构造基准价。成交价在基准价 ±6% 内浮动，避免所有人都在同一价位成交。
BASE_PX: dict[str, str] = {
    "600519": "1650", "000858": "125", "600036": "41", "000001": "10.3",
    "300750": "350", "002594": "255", "601318": "52", "600030": "26",
    "601899": "19", "000651": "44", "601728": "6.8", "688981": "95",
    "002415": "31", "600900": "29", "601088": "41", "600276": "48",
    "000333": "72", "002714": "45",
    "510300": "4.15", "588000": "0.98", "512100": "2.3", "518880": "7.8",
    "159915": "2.45",
    "113050": "120", "113065": "118", "127098": "112", "123140": "105",
    "110093": "101", "118041": "130",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def scale(base: Decimal, pct: float) -> Decimal:
    """base × (1 + pct)。pct 是 float，转 Decimal 再乘，避免 Decimal×float。"""
    return base * (D(1) + D(str(pct)))

class Book:
    """同标的买卖日期簿记：买入避开卖出日、卖出避开已占用的日子。

    A 股的 T+1 校验是"卖出当天只要存在同标的买入就报警"——哪怕卖的
    不是这一笔买入的股份。所以拆单/多往返的账户必须让买卖日期错开。
    """

    def __init__(self) -> None:
        self.bought: dict[str, set] = {}
        self.sold: dict[str, set] = {}

    def record_buy(self, sym: str, d) -> None:
        self.bought.setdefault(sym, set()).add(d)

    def record_sell(self, sym: str, d) -> None:
        self.sold.setdefault(sym, set()).add(d)


_BOOK: Book | None = None


def new_book() -> Book:
    global _BOOK
    _BOOK = Book()
    return _BOOK


def _free(d, *forbidden):
    """d 若撞上禁止日期就顺延到下一个交易日。"""
    while any(d in s for s in forbidden):
        d = add_td(d, 1)
    return d


def add_td(d0: date, n: int) -> date:
    """d0 之后第 n 个交易日（忽略节假日，只跳周末）。n >= 1。"""
    d, left = d0, n
    while left > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            left -= 1
    return d


def rand_wd(rng: random.Random, a: date, b: date) -> date:
    """[a, b] 内随机取一个工作日。"""
    if a < START:
        a = START
    if b > END:
        b = END
    while True:
        d = a + timedelta(days=rng.randrange((b - a).days + 1))
        if d.weekday() < 5:
            return d


def px(x, asset: str) -> Decimal:
    """按资产类型定价位。股票 2 位、ETF/转债 3 位。"""
    dp = 2 if asset == STOCK else 3
    return D(str(x)).quantize(D(1).scaleb(-dp), rounding=ROUND_HALF_UP)


def base_px(sym: str, rng: random.Random) -> Decimal:
    """基准价 ±6% 内浮动后的成交价。"""
    return px(D(BASE_PX[sym]) * D(1 + rng.uniform(-0.06, 0.06)), POOL[sym][1])


def lot_qty(amount, price: Decimal, asset: str) -> int:
    """按单笔金额换算数量：股票/ETF 整百，转债整十张。"""
    raw = D(amount) / price
    if asset == BOND:
        return max(int(raw // 10) * 10, 10)
    return max(int(raw // 100) * 100, 100)


def script_of(pairs: list[tuple[str, list[tuple]]]) -> list[tuple]:
    """[(code, legs)] → expand_script 需要的 (code, name, asset, legs) 形式。"""
    return [(sym, POOL[sym][0], POOL[sym][1], legs) for sym, legs in pairs]


# ---------------------------------------------------------------------------
# 交易模式积木。每个函数返回 (legs, 需要的期末价)
# ---------------------------------------------------------------------------

def win_round(rng, sym, d0, *, hold_td, gain, amount):
    """一次"买后获利了结"。返回 legs（不含期末价——已全部了结）。"""
    global _BOOK
    if _BOOK is None:
        new_book()
    asset = POOL[sym][1]
    cost = base_px(sym, rng)
    q = lot_qty(amount, cost, asset)
    d0 = _free(d0, _BOOK.sold.get(sym, ()))
    d1 = add_td(d0, max(hold_td, 1))
    d1 = _free(d1, _BOOK.sold.get(sym, ()), _BOOK.bought.get(sym, ()))
    _BOOK.record_buy(sym, d0)
    _BOOK.record_sell(sym, d1)
    sell = px(scale(cost, gain), asset)
    return [("B", d0.isoformat(), q, str(cost)),
            ("S", d1.isoformat(), q, str(sell))]


def loss_round(rng, sym, d0, *, hold_td, loss, amount):
    """一次"割肉离场"。"""
    global _BOOK
    if _BOOK is None:
        new_book()
    asset = POOL[sym][1]
    cost = base_px(sym, rng)
    q = lot_qty(amount, cost, asset)
    d0 = _free(d0, _BOOK.sold.get(sym, ()))
    d1 = add_td(d0, max(hold_td, 1))
    d1 = _free(d1, _BOOK.sold.get(sym, ()), _BOOK.bought.get(sym, ()))
    _BOOK.record_buy(sym, d0)
    _BOOK.record_sell(sym, d1)
    sell = px(scale(cost, -loss), asset)
    return [("B", d0.isoformat(), q, str(cost)),
            ("S", d1.isoformat(), q, str(sell))]


def open_pos(rng, sym, d0, *, chg, amount):
    """开仓后拿到期末（浮盈 chg>0 / 浮亏 chg<0）。返回 (legs, 期末价)。"""
    global _BOOK
    if _BOOK is None:
        new_book()
    asset = POOL[sym][1]
    cost = base_px(sym, rng)
    q = lot_qty(amount, cost, asset)
    d0 = _free(d0, _BOOK.sold.get(sym, ()))
    _BOOK.record_buy(sym, d0)
    end = px(scale(cost, chg), asset)
    return [("B", d0.isoformat(), q, str(cost))], str(end)


def dividend(day, amount) -> tuple:
    return ("D", day, amount)


# ---------------------------------------------------------------------------
# 十种人格。每个函数返回 (script, closes)：
#   script —— expand_script 的输入
#   closes —— {code: 期末价}，只列期末仍有持仓的标的
# ---------------------------------------------------------------------------

def p01_chase(rng) -> tuple:
    """p01 追涨杀跌型：小赚快跑、大亏死扛。全库最强的正向处置效应。"""
    legs: list[tuple[str, list[tuple]]] = []
    closes: dict[str, str] = {}

    # 6 只“落袋”标的：≤22 个来回/只，持有 3~12 个交易日，赚 2.5~6% 就跑
    for sym in ["000001", "601318", "600030", "000651", "601899", "002415"]:
        cur = START
        for _ in range(22):
            if cur + timedelta(days=25) > END:
                break                       # 窗口尾部塞不下一个来回了
            d0 = rand_wd(rng, cur + timedelta(days=1),
                         cur + timedelta(days=8))
            legs.append((sym, win_round(
                rng, sym, d0, hold_td=rng.randint(3, 12),
                gain=rng.uniform(0.025, 0.06), amount=rng.randint(8, 25) * 1000)))
            cur = date.fromisoformat(legs[-1][1][-1][1])

    # 3 只“割过的输家”：扛 4~7 个月，亏 12~22% 才认输
    for sym in ["300750", "002594", "688981"]:
        for _ in range(3):
            d0 = rand_wd(rng, date(2025, 11, 10), date(2026, 1, 31))
            legs.append((sym, loss_round(
                rng, sym, d0, hold_td=rng.randint(80, 150),
                loss=rng.uniform(0.12, 0.22), amount=rng.randint(15, 30) * 1000)))

    # 6 只"还扛着的输家"：浮亏 15~35%，一股未卖
    for i, sym in enumerate(["600519", "300750", "002594",
                             "688981", "588000", "601728"]):
        d0 = rand_wd(rng, date(2025, 11, 5) , date(2026, 4, 30))
        l, end = open_pos(rng, sym, d0, chg=-rng.uniform(0.15, 0.35),
                          amount=(12 + i * 3) * 1000)
        legs.append((sym, l))
        closes[sym] = end

    # 4 笔"刚买就赚"的浮盈持仓（9~10 月买入；open_pos 自动避开卖出日）
    for i, sym in enumerate(["000001", "600030", "000651", "512100"]):
        d0 = rand_wd(rng, date(2026, 9, 1), date(2026, 10, 12))
        l, end = open_pos(rng, sym, d0, chg=rng.uniform(0.03, 0.08),
                          amount=(8 + i) * 1000)
        legs.append((sym, l))
        closes[sym] = end
    return script_of(legs), closes


def p02_reverse(rng) -> tuple:
    """p02 反向处置型：亏 3~5% 立刻砍，赚的拿到 3~9 个月。DE 应为负。"""
    legs, closes = [], {}

    # 8 只止损标的 × ≤5 次：持有 2~6 个交易日，亏 2.5~5% 立刻走
    for sym in ["000001", "600030", "000651", "601899", "002415",
                "600900", "601088", "000333"]:
        cur = START
        for _ in range(5):
            if cur + timedelta(days=50) > END:
                break
            d0 = rand_wd(rng, cur + timedelta(days=2),
                         cur + timedelta(days=45))
            legs.append((sym, loss_round(
                rng, sym, d0, hold_td=rng.randint(2, 6),
                loss=rng.uniform(0.025, 0.05), amount=rng.randint(10, 20) * 1000)))
            cur = date.fromisoformat(legs[-1][1][-1][1])

    # 8 次长线兑现：持有 3~7 个月，赚 15~40%（买入日期收窄，保证卖出落在窗口内）
    for sym in ["600036", "600519", "300750", "000858",
                "601318", "600276", "510300", "518880"]:
        d0 = rand_wd(rng, date(2025, 11, 10), date(2026, 2, 27))
        legs.append((sym, win_round(
            rng, sym, d0, hold_td=rng.randint(90, 150),
            gain=rng.uniform(0.15, 0.40), amount=rng.randint(15, 30) * 1000)))

    # 12 笔仍在赚的持仓：浮盈 25~70%
    for i, sym in enumerate(["600036", "600519", "300750", "000858",
                             "601318", "600276", "510300", "518880",
                             "600900", "000333", "588000", "159915"]):
        d0 = rand_wd(rng, date(2025, 12, 1), date(2026, 6, 30))
        l, end = open_pos(rng, sym, d0, chg=rng.uniform(0.25, 0.70),
                          amount=(10 + i) * 1000)
        legs.append((sym, l))
        closes[sym] = end

    # 20 笔小幅浮亏的持仓（-1~-5%，还在止损线上方忍着）
    for i, sym in enumerate(["000001", "600030", "000651", "601899",
                             "002415", "601088", "002714", "601728",
                             "688981", "512100"] * 2):
        d0 = rand_wd(rng, date(2026, 5, 1), date(2026, 9, 30))
        l, end = open_pos(rng, sym, d0, chg=-rng.uniform(0.01, 0.05),
                          amount=(6 + i) * 1000)
        legs.append((sym, l))
        if sym in closes:
            # 同标的两笔持仓取更低的那笔期末价（保守处理，不会高估浮盈）
            closes[sym] = str(min(D(closes[sym]), D(end)))
        else:
            closes[sym] = end
    return script_of(legs), closes


def p03_longterm(rng) -> tuple:
    """p03 长线价值型：一年十几笔。分母大分子小 → 应输出"证据不足"。"""
    script: list[tuple[str, str, str, list[tuple]]] = []
    closes: dict[str, str] = {}

    # 五笔买入后拿到期末，盈亏互现（chg = 期末相对成本的变动）
    plans = [("600036", "2025-11-06", 0.18), ("600900", "2025-12-15", 0.10),
             ("000333", "2026-01-20", -0.06), ("600519", "2026-03-09", -0.14),
             ("510300", "2026-05-11", 0.06)]
    for sym, d, chg in plans:
        asset = POOL[sym][1]
        cost = base_px(sym, rng)
        q = lot_qty(rng.randint(20, 40) * 1000, cost, asset)
        legs = [("B", d, q, str(cost))]
        if sym == "600036":
            legs.append(("S", "2026-08-14", q // 2, str(px(scale(cost, 0.09), asset))))
            legs.append(dividend("2026-04-20", "1650.00"))
        elif sym == "600900":
            legs.append(dividend("2026-06-18", "900.00"))
        script.append((sym, POOL[sym][0], asset, legs))
        closes[sym] = str(px(scale(cost, chg), asset))
    return script, closes


def p04_student(rng) -> tuple:
    """p04 学生小账户：几千块一笔，7 个来回 + 1 笔被套。应“证据不足”。"""
    legs, closes = [], {}
    cur = date(2025, 11, 5)
    for i in range(7):
        if cur > date(2026, 8, 15):
            break                           # 窗口尾部留足持有期
        sym = ["000001", "510300", "600030", "159915"][i % 4]
        d0 = rand_wd(rng, cur, cur + timedelta(days=15))
        if i % 2 == 0:
            r = win_round(rng, sym, d0, hold_td=rng.randint(4, 20),
                          gain=rng.uniform(0.02, 0.07), amount=rng.randint(3, 8) * 1000)
        else:
            r = loss_round(rng, sym, d0, hold_td=rng.randint(10, 60),
                           loss=rng.uniform(0.03, 0.10), amount=rng.randint(3, 8) * 1000)
        legs.append((sym, r))
        cur = date.fromisoformat(r[-1][1]) + timedelta(days=rng.randint(5, 25))
    # 一笔被套的
    l, end = open_pos(rng, "300750", date(2026, 6, 8), chg=-0.16, amount=6000)
    legs.append(("300750", l))
    closes["300750"] = end
    return script_of(legs), closes


def p05_split(rng) -> tuple:
    """p05 频繁拆单型：建仓 4~8 笔、出货 3~6 笔，压测 FIFO 配对链。

    两只刻意留底仓到期末（一赚一亏），其余全部了结。
    """
    script: list[tuple[str, str, str, list[tuple]]] = []
    closes: dict[str, str] = {}
    for i, sym in enumerate(["000001", "600036", "601318", "600030",
                             "000651", "601899", "002415", "000333",
                             "600900", "601088"]):
        asset = POOL[sym][1]
        lot = 10 if asset == BOND else 100
        cost = base_px(sym, rng)
        d0 = rand_wd(rng, START + timedelta(days=i * 15), date(2026, 5, 25))
        nb, ns = rng.randint(4, 8), rng.randint(3, 6)
        legs: list[tuple] = []
        q_total = 0
        for k in range(nb):
            q = lot_qty(rng.randint(5, 10) * 1000, cost, asset) // max(1, nb)
            q = max(q // lot * lot, lot)
            d = add_td(d0, k * rng.randint(2, 5))
            legs.append(("B", d.isoformat(), q,
                         str(px(scale(cost, rng.uniform(-0.015, 0.015)), asset))))
            q_total += q

        keep = q_total // 3 if i in (0, 5) else 0    # 两只留底仓
        keep -= keep % lot
        win = (i % 3 != 2)                            # 七只赚出、三只亏出
        move = rng.uniform(0.02, 0.07) if win else rng.uniform(-0.10, -0.03)
        sell_px = scale(cost, move)
        n_sell = ns if keep == 0 else max(ns - 1, 1)
        per = (q_total - keep) // n_sell
        per -= per % lot
        # 出货起点必须在最后一笔买入之后：买入最晚在 (nb-1)*5 个交易日后，
        # 这里从 nb*5+2 起算，留足间隔（同日先买后卖会触发 T+1 校验）
        for k in range(n_sell):
            q = per if k < n_sell - 1 else (q_total - keep) - per * (n_sell - 1)
            d = add_td(d0, nb * 5 + 2 + k * rng.randint(2, 6))
            legs.append(("S", d.isoformat(), q,
                         str(px(scale(sell_px, rng.uniform(-0.008, 0.008)), asset))))
        script.append((sym, POOL[sym][0], asset, legs))
        if keep > 0:
            chg = 0.05 if i == 0 else -0.12
            closes[sym] = str(px(scale(cost, chg), asset))
    return script, closes


def p06_etf_dca(rng) -> tuple:
    """p06 ETF 定投型：4 只 × 12 个月定投，期末前部分止盈。"""
    script: list[tuple[str, str, str, list[tuple]]] = []
    closes: dict[str, str] = {}
    totals: dict[str, int] = {}
    for sym in ["510300", "588000", "512100", "518880"]:
        asset = POOL[sym][1]
        legs = []
        for m in range(12):
            y, mo = (2025, 11 + m) if m < 2 else (2026, m - 1)
            d = date(y, mo, 1)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            cost = px(D(BASE_PX[sym]) * D(1 + rng.uniform(-0.05, 0.05)), asset)
            q = lot_qty(4000, cost, asset)
            legs.append(("B", d.isoformat(), q, str(cost)))
            totals[sym] = totals.get(sym, 0) + q
        script.append((sym, POOL[sym][0], asset, legs))

    # 止盈：前两只各卖约 1/3（一赚一亏），其余继续持有
    for idx, sym, move in [(0, "510300", 0.09), (1, "588000", -0.12)]:
        asset = POOL[sym][1]
        last_buy = script[idx][3][-1]
        sell_d = add_td(date.fromisoformat(last_buy[1]), 5)
        q_sell = totals[sym] // 3 // 100 * 100
        cost0 = px(D(BASE_PX[sym]), asset)
        script[idx] = (sym, POOL[sym][0], asset,
                       script[idx][3] + [("S", sell_d.isoformat(), q_sell,
                                          str(px(scale(cost0, move), asset)))])
        chg = 0.12 if move > 0 else -0.10
        closes[sym] = str(px(scale(cost0, chg), asset))
    closes["512100"] = str(px(scale(D(BASE_PX["512100"]), 0.03), ETF))
    closes["518880"] = str(px(scale(D(BASE_PX["518880"]), 0.18), ETF))
    return script, closes


def p07_bond_t0(rng) -> tuple:
    """p07 转债 T+0 型：当日买卖 ×6 只，压测 T+0 豁免与沪深前缀。"""
    legs, script = [], []
    for sym in ["113050", "113065", "127098", "123140", "110093", "118041"]:
        asset = POOL[sym][1]
        l = []
        cur = START
        for _ in range(25):
            d0 = rand_wd(rng, cur, cur + timedelta(days=12))
            cost = px(D(BASE_PX[sym]) * D(1 + rng.uniform(-0.04, 0.04)), asset)
            q = lot_qty(rng.randint(15, 60) * 100, cost, asset)
            sell = px(scale(cost, rng.uniform(0.003, 0.015)), asset)
            l.append(("B", d0.isoformat(), q, str(cost)))
            l.append(("S", d0.isoformat(), q, str(sell)))       # T+0：当日平
            cur = d0 + timedelta(days=1)
        # 2 笔隔夜仓：今天买明天卖
        for _ in range(2):
            d0 = rand_wd(rng, date(2026, 7, 1), date(2026, 9, 30))
            cost = px(D(BASE_PX[sym]) * D(1 + rng.uniform(-0.03, 0.03)), asset)
            q = lot_qty(30000, cost, asset)
            l.append(("B", d0.isoformat(), q, str(cost)))
            l.append(("S", add_td(d0, 1).isoformat(), q,
                      str(px(scale(cost, rng.uniform(0.005, 0.02)), asset))))
        script.append((sym, POOL[sym][0], asset, l))
    return script, {}


def p08_events(rng) -> tuple:
    """p08 事件驱动型：送转/配股/分红/打新/担保品划转，结构最脏。"""
    script: list[tuple[str, str, str, list[tuple]]] = []
    closes: dict[str, str] = {}

    # 10 送 3 + 分红 + 卖出：全库最反直觉的一笔（原样保留自 sample_account）
    script.append(("000858", "五粮液", STOCK, [
        ("B", "2025-11-13", 800, "129.60"),
        ("D", "2026-05-22", "2400.00"),
        ("N", "2026-06-05", 240),
        ("S", "2026-09-23", 1040, "119.20"),
    ]))
    # 配股缴款 + 持有到期末
    script.append(("600036", "招商银行", STOCK, [
        ("B", "2025-11-06", 3000, "40.20"),
        ("R", "2026-05-15", 900, "33.10"),
        ("D", "2026-04-20", "1350.00"),
    ]))
    closes["600036"] = "42.80"
    # 打新中签 → 上市即卖
    script.append(("601728", "中国电信", STOCK, [
        ("I", "2026-04-09", 1000, "6.50"),
        ("S", "2026-04-16", 1000, "7.20"),
    ]))
    # 黄金 ETF：买持有，期末浮盈
    script.append(("518880", "黄金ETF", ETF, [
        ("B", "2026-01-08", 5000, "7.42"),
    ]))
    closes["518880"] = "8.65"
    # 普通来回 ×2
    script.append(("000001", "平安银行", STOCK, [
        ("B", "2025-12-08", 2000, "10.45"),
        ("S", "2026-01-15", 2000, "10.92"),
        ("B", "2026-06-10", 3000, "9.95"),
        ("S", "2026-08-20", 3000, "10.55"),
    ]))
    # 三条"认不出来"的业务：正确行为是告警并继续
    script.append(("000001", "平安银行", STOCK, [("O", "2026-03-18", "担保品划转")]))
    script.append(("518880", "黄金ETF", ETF, [("O", "2026-08-12", "利息归本")]))
    script.append(("600036", "招商银行", STOCK, [("O", "2026-09-01", "证券转托管")]))
    return script, closes


def p09_bear_hold(rng) -> tuple:
    """p09 熊市死扛型：重仓 3 只，深套不出。应"证据不足"+ 账面深亏。"""
    script: list[tuple[str, str, str, list[tuple]]] = []
    closes = {"300750": "238.00", "688981": "68.50", "002594": "188.00"}
    script.append(("300750", "宁德时代", STOCK, [
        ("B", "2025-11-18", 300, "392.00"),
        ("B", "2026-02-10", 200, "330.00"),
        ("S", "2026-08-19", 100, "282.00"),     # 恐慌割了一小笔
    ]))
    script.append(("688981", "中芯国际", STOCK, [
        ("B", "2025-12-03", 900, "96.50"),
    ]))
    script.append(("002594", "比亚迪", STOCK, [
        ("B", "2025-11-25", 400, "262.00"),
    ]))
    return script, closes


def p10_mixed(rng) -> tuple:
    """p10 无模式基线：胜负各半、持有期随机。DE 应接近 0。"""
    legs, closes = [], {}
    syms = ["000001", "600036", "601318", "600030", "000651", "601899",
            "002415", "000333", "600900", "601088", "510300", "512100"]
    cur = START
    for i in range(60):
        sym = syms[i % len(syms)]
        win = rng.random() < 0.5
        d0 = rand_wd(rng, cur, cur + timedelta(days=10))
        if win:
            r = win_round(rng, sym, d0, hold_td=rng.randint(4, 40),
                          gain=rng.uniform(0.02, 0.08), amount=rng.randint(8, 18) * 1000)
        else:
            r = loss_round(rng, sym, d0, hold_td=rng.randint(4, 40),
                           loss=rng.uniform(0.02, 0.08), amount=rng.randint(8, 18) * 1000)
        legs.append((sym, r))
        cur = date.fromisoformat(r[-1][1]) + timedelta(days=rng.randint(1, 10))
        if cur > END - timedelta(days=50):
            cur = rand_wd(rng, START, START + timedelta(days=60))
    # 6 笔期末持仓：3 赚 3 亏
    for i, sym in enumerate(["600519", "300750", "002594",
                             "588000", "600276", "002714"]):
        d0 = rand_wd(rng, date(2026, 4, 1), date(2026, 9, 30))
        chg = rng.uniform(0.04, 0.15) if i < 3 else rng.uniform(-0.15, -0.04)
        l, end = open_pos(rng, sym, d0, chg=chg, amount=rng.randint(10, 20) * 1000)
        legs.append((sym, l))
        closes[sym] = end
    return script_of(legs), closes


# ---------------------------------------------------------------------------
# 组装与落盘
# ---------------------------------------------------------------------------

#: (目录名, 一句话, 生成函数, 格式, 切分日期或 None)
PERSONAS: list[tuple[str, str, object, str, date | None]] = [
    ("p01_chase_sell", "追涨杀跌型：小赚快跑、大亏死扛（强处置效应）", p01_chase, "ths", None),
    ("p02_reverse", "反向处置型：亏了立刻砍、赚的拿得久（DE 为负）", p02_reverse, "ht", None),
    ("p03_longterm", "长线价值型：一年只交易几笔（小样本）", p03_longterm, "ths", None),
    ("p04_student", "学生小账户：几千块一笔（小样本）", p04_student, "ths", None),
    ("p05_split_orders", "频繁拆单型：分批建仓出货（压测 FIFO）", p05_split, "ht", None),
    ("p06_etf_dca", "ETF 定投型：月定投 + 偶尔止盈", p06_etf_dca, "mixed", date(2026, 5, 1)),
    ("p07_bond_t0", "可转债 T+0 型：当日买卖（压测 T+0 与沪深前缀）", p07_bond_t0, "ths", None),
    ("p08_events", "事件驱动型：送转/配股/打新/担保品划转", p08_events, "ht", None),
    ("p09_bear_hold", "熊市死扛型：重仓被套、几乎不割", p09_bear_hold, "ths", None),
    ("p10_mixed", "无模式基线：胜负各半（对照组）", p10_mixed, "mixed", date(2026, 6, 1)),
]

ASSET_DP = {STOCK: 2, ETF: 3, BOND: 3}
WOBBLE = SA.WOBBLE if hasattr(SA, "WOBBLE") else 0.012
_weekdays_between = SA._weekdays_between


def asset_of(sym: str) -> str:
    return POOL[sym][1]


def build_history(rows: list[dict], closes: dict[str, str]) -> list[tuple[date, str, Decimal]]:
    """逐日价格序列。与 sample_account 同一算法：成交价当锚点、线性插值、
    端点归零正弦扰动；期末持仓补 END 收盘价锚点。"""
    anchors: dict[str, dict[date, Decimal]] = {}
    for r in rows:
        if r["price"] <= 0:
            continue
        anchors.setdefault(r["symbol"], {})[r["dt"]] = r["price"]

    out: list[tuple[date, str, Decimal]] = []
    for sym, by_day in anchors.items():
        pts = sorted(by_day.items())
        if sym in closes and END > pts[-1][0]:
            pts.append((END, D(closes[sym])))
        dp = ASSET_DP[asset_of(sym)]
        q = D(1).scaleb(-dp)
        for i in range(len(pts) - 1):
            d0, p0 = pts[i]
            d1, p1 = pts[i + 1]
            span = (d1 - d0).days
            for d in _weekdays_between(d0, d1):
                t = D((d - d0).days) / D(span)
                base = p0 + (p1 - p0) * t
                wob = D(str(WOBBLE * math.sin(math.pi * float(t))
                            * math.sin(2.7 * math.pi * float(t))))
                out.append((d, sym, (base * (D(1) + wob)).quantize(q, rounding=ROUND_HALF_UP)))
        out.append((pts[-1][0], sym, pts[-1][1]))
    return sorted(out)


def final_holdings(rows: list[dict]) -> dict[str, Decimal]:
    """重放账本得期末持仓（与 validate 同一规则）。"""
    holdings: dict[str, Decimal] = {}
    order = {"证券买入": 0, "新股入账": 0, "红股入账": 0, "配股缴款": 0,
             "股息入账": 1, "证券卖出": 2}
    for r in sorted(rows, key=lambda x: (x["dt"], x["symbol"], order.get(x["biz"], 3))):
        sym, biz = r["symbol"], r["biz"]
        qty = abs(r["qty"])
        if biz in ("证券买入", "新股入账", "配股缴款"):
            holdings[sym] = holdings.get(sym, D(0)) + qty
        elif biz == "红股入账":
            total = holdings.get(sym, D(0))
            if total > 0:
                holdings[sym] = total * (total + qty) / total
        elif biz == "证券卖出":
            holdings[sym] = holdings.get(sym, D(0)) - qty
    return {s: q for s, q in holdings.items() if q > 0}


def write_history(hist: list[tuple[date, str, Decimal]], path: Path) -> None:
    lines = ["日期,证券代码,收盘价"]
    for d, sym, p in hist:
        lines.append(f"{d.isoformat()},{sym},{p}")
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8-sig")


def main() -> int:
    total_rows = 0
    for pid, desc, fn, fmt, cut in PERSONAS:
        rng = random.Random(pid)                       # seed = 目录名，稳定可复现
        script, closes = fn(rng)
        rows = expand_script(script)
        validate(rows)

        # 一致性第 4 条：期末价与期末持仓必须互相印证
        held = final_holdings(rows)
        for sym in closes:
            if sym not in held:
                raise ValueError(f"{pid}: {sym} 给了期末价但期末已清仓")
        for sym in held:
            if sym not in closes:
                raise ValueError(f"{pid}: {sym} 期末仍有 {held[sym]} 股但没给期末价")

        out_dir = HERE / pid
        out_dir.mkdir(exist_ok=True)
        files: list[Path] = []
        rng_text = f"{START:%Y%m%d}-{END:%Y%m%d}"
        acct = pid[1:3] + "0000" + pid[1:3]
        if fmt == "mixed" and cut is not None:
            # 两种券商格式年中拼接（演示分月导出后合并的真实形态）
            part1 = [r for r in rows if r["dt"] < cut]
            part2 = [r for r in rows if r["dt"] >= cut]
            p1 = out_dir / f"{pid}_part1_ths.csv"
            p2 = out_dir / f"{pid}_part2_ht.csv"
            write_ths(part1, p1, account=acct,
                      range_text=f"{START:%Y%m%d}-{cut - timedelta(days=1):%Y%m%d}")
            write_ht(part2, p2, account=acct,
                     range_text=f"{cut:%Y%m%d}-{END:%Y%m%d}")
            files += [p1, p2]
        elif fmt == "ht":
            p = out_dir / f"{pid}_{rng_text}_ht.csv"
            write_ht(rows, p, account=acct, range_text=rng_text)
            files.append(p)
        else:
            p = out_dir / f"{pid}_{rng_text}_ths.csv"
            write_ths(rows, p, account=acct, range_text=rng_text)
            files.append(p)

        hist = build_history(rows, closes)
        hp = HERE / f"{pid}_price_history.csv"
        write_history(hist, hp)

        # 人格说明卡
        (out_dir / "README.md").write_text(
            f"# {pid} —— {desc.split('：')[0]}\n\n{desc}。\n\n"
            f"**模拟数据**：代码真实存在，名称/价格/成交均为构造值。\n\n"
            f"- 记录 {len(rows)} 条，标的 {len({r['symbol'] for r in rows})} 只\n"
            f"- 格式：{'同花顺风格 + 另一家风格拼接' if fmt == 'mixed' else ('同花顺风格(UTF-8 BOM)' if fmt == 'ths' else '华泰风格(GB18030)')}\n"
            f"- 运行：`PYTHONPATH=src python -m tradeopsy report examples/synthetic_accounts/{pid} "
            f"--price-history examples/synthetic_accounts/{pid}_price_history.csv -o report.html`\n",
            encoding="utf-8")

        total_rows += len(rows)
        print(f"{pid:<18} {len(rows):>4} 条 · {len(held):>2} 只期末持仓 · "
              f"价格 {len(hist):>5} 条 · {fmt}")
    print(f"\n十套合计 {total_rows} 条记录。seed 固定，重跑逐字节一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
