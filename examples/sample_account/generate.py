"""生成「示例账户」交割单（**合成数据，不含任何真实交易信息**）。

为什么需要这份数据：
    项目定位是"体检你自己的交割单"，但真实交割单涉及隐私、不能进仓库。
    没有可公开的输入，任何打开仓库的人都只能看到一句"请把你的 CSV 放进来"，
    无法验证这个工具真的能处理券商导出的脏数据。

    所以这里造一个"规模、结构、脏法"都尽可能逼真的示例账户：
    12 个月、14 只标的、60+ 条记录，覆盖 A 股几乎所有会让简单脚本算错账的场景。

刻意设计的两件事：
    1. **两种券商格式**：上半段用同花顺风格（包干手续费列、负数表示卖出、
       BOM 头、表头不在第一行）；下半段用另一家风格（GB18030 编码、
       费用拆成净佣金/印花税/过户费三列、正数表示卖出、列顺序不同）。
       目的是让 `tradeopsy analyze` 一次性演示编码嗅探、表头定位、
       列名模糊匹配、费用分项合并这一整条链路。
    2. **行为上是一个真实的处置效应样本**：赚了就跑、亏了死扛。
       所以跑出来的 PGR 会明显大于 PLR，报告能讲出故事，而不是一堆随机数。

用法：
    python examples/sample_account/generate.py      # 重新生成（结果确定，无随机数）

文件产物：
    part1_2025-11_2026-03_ths.csv     同花顺风格（UTF-8 BOM）
    part2_2026-04_2026-10_ht.csv      另一家风格（GB18030，费用分项）
    ../sample_account_prices.csv      2026-10-30 收盘价，供 --prices 使用
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICES_OUT = HERE.parent / "sample_account_prices.csv"

D = Decimal
CENT = D("0.01")

#: 资产类型决定费率。A 股真实差异很大，这里按最常见档位设定，并在 README 里写明。
STOCK, ETF, BOND = "stock", "etf", "bond"

#: 分段日期：交割单按月导出后拼接，是真实用户的常见形态。
CUT = date(2026, 4, 1)

#: 2026-10-30 收盘价（构造值）。刻意让"死扛的那几只"处在浮亏，
#: 让"早早止盈的那几只"处在浮盈 —— 这样账面端才能配合已实现端讲出同一个故事。
CLOSE_2026_10_30: dict[str, str] = {
    "600519": "1450.00",   # 茅台，浮亏
    "002594": "195.00",    # 比亚迪，浮亏
    "600036": "44.00",     # 招行，浮盈
    "300750": "305.00",    # 宁德，浮亏
    "510300": "4.48",      # 沪深300ETF，浮盈
    "588000": "0.88",      # 科创50ETF，浮亏
    "601899": "21.90",     # 紫金，浮亏
    "600030": "26.30",     # 中信证券，浮盈
}


def money(x) -> Decimal:
    """金额落到分，四舍五入（与账本口径一致）。"""
    return D(x).quantize(CENT, rounding=ROUND_HALF_UP)


def fee_parts(amount: Decimal, asset: str, side: str) -> tuple[Decimal, Decimal, Decimal]:
    """返回 (净佣金, 印花税, 过户费)，单位元。

    A 股现行规则（2022-04 沪深过户费统一降至成交金额 0.001% 双向收取）：
        股票 买入：佣金（万 2.5，最低 5 元）+ 过户费 万 0.1
        股票 卖出：佣金 + 印花税 千 0.5 + 过户费
        ETF       ：佣金（万 1，最低 5 元），**免印花税、免过户费**
        可转债    ：佣金（万 1，无 5 元最低），**免印花税**
    把 ETF 的免印花税做对，是这个字段最容易露怯的地方 ——
    ETF 卖出也按千 0.5 收税，一年多算的成本足够让收益率对不上。
    """
    if amount <= 0:
        return D("0"), D("0"), D("0")

    if asset == ETF:
        return money(max(D("5"), amount * D("0.0001"))), D("0"), D("0")
    if asset == BOND:
        return money(max(D("0.10"), amount * D("0.0001"))), D("0"), D("0")

    comm = money(max(D("5"), amount * D("0.00025")))
    stamp = money(amount * D("0.0005")) if side == "S" else D("0")
    transfer = money(amount * D("0.00001"))
    return comm, stamp, transfer


# ---------------------------------------------------------------------------
# 交易脚本：一行一笔，按标的组织。注释写的是"这笔交易在讲什么故事"。
# ---------------------------------------------------------------------------

#: 每条 leg 的格式：
#:   ("B", 日期, 数量, 价格)   买入
#:   ("S", 日期, 数量, 价格)   卖出
#:   ("D", 日期, 到账金额)     股息入账
#:   ("N", 日期, 股数)         红股入账（10 送 X 转增）
#:   ("R", 日期, 股数, 价格)   配股缴款
#:   ("I", 日期, 股数, 价格)   新股入账（中签）
#:   ("O", 日期, 摘要)         其它业务（用来演示"告警但不中断"）
SCRIPT: list[tuple[str, str, str, list[tuple]]] = [
    # 死扛的大输家：买在高点，一年不动，账面浮亏 2.3 万，一股没卖
    ("600519", "贵州茅台", STOCK, [
        ("B", "2025-11-12", 100, "1680.00"),
        ("D", "2026-06-18", "3000.00"),
    ]),

    # 最典型的"落袋型"：一年 6 个来回，每次赚 3~5% 就跑，从不恋战
    ("000001", "平安银行", STOCK, [
        ("B", "2025-11-05", 2000, "10.20"),
        ("S", "2025-11-18", 2000, "10.72"),
        ("B", "2025-12-03", 3000, "10.05"),
        ("S", "2025-12-12", 3000, "10.55"),
        ("B", "2026-01-14", 2000, "9.80"),
        ("S", "2026-01-23", 2000, "10.15"),
        ("B", "2026-03-04", 5000, "9.65"),
        ("S", "2026-03-13", 5000, "10.02"),
        ("D", "2026-06-18", "800.00"),
        ("B", "2026-07-08", 3000, "9.90"),
        ("S", "2026-07-16", 3000, "10.28"),
        ("B", "2026-09-09", 4000, "10.10"),
        ("S", "2026-09-17", 4000, "10.45"),
    ]),

    # 先亏后补仓，补完继续跌；年底只割掉补仓那一笔，原始仓位还扛着
    ("002594", "比亚迪", STOCK, [
        ("B", "2025-11-20", 500, "265.00"),
        ("B", "2026-02-10", 300, "240.00"),
        ("S", "2026-09-15", 300, "205.00"),
    ]),

    # 含配股缴款 + 两次分红：配股会让成本重新摊薄，是最容易算错的一类
    ("600036", "招商银行", STOCK, [
        ("D", "2025-12-18", "1500.00"),
        ("B", "2025-11-06", 5000, "38.50"),
        ("S", "2026-01-08", 2500, "41.20"),
        ("D", "2026-04-20", "2200.00"),
        ("R", "2026-05-15", 1200, "33.00"),
    ]),

    # 含 10 送 3 + 分红：送出转只增加股数、不增加总成本，
    # 所以"看起来 118 元卖 128 元买"其实是赚的 —— 这是全库最反直觉的一笔
    ("000858", "五粮液", STOCK, [
        ("B", "2025-11-14", 1000, "128.00"),
        ("D", "2026-05-22", "3000.00"),
        ("N", "2026-06-05", 300),
        ("S", "2026-09-23", 1300, "118.00"),
    ]),

    # 三个快进快出的赢家
    ("601318", "中国平安", STOCK, [
        ("B", "2025-12-10", 1000, "52.30"),
        ("S", "2025-12-19", 1000, "54.80"),
        ("B", "2026-04-08", 2000, "49.60"),
        ("S", "2026-04-17", 2000, "51.90"),
        ("B", "2026-07-28", 1500, "55.20"),
        ("S", "2026-08-07", 1500, "56.10"),
    ]),

    # 越跌越补、补完继续跌，年底只割了一小笔，大仓位还扛着
    ("300750", "宁德时代", STOCK, [
        ("B", "2025-11-25", 200, "420.00"),
        ("B", "2026-03-12", 100, "330.00"),
        ("S", "2026-08-19", 100, "290.00"),
    ]),

    # ETF 定投 5 期后止盈一部分；ETF 免印花税，费用明显低于股票
    ("510300", "沪深300ETF", ETF, [
        ("B", "2025-12-01", 10000, "4.05"),
        ("B", "2026-01-05", 10000, "3.95"),
        ("B", "2026-02-02", 10000, "4.10"),
        ("B", "2026-04-01", 10000, "3.88"),
        ("B", "2026-06-01", 10000, "4.02"),
        ("S", "2026-09-01", 15000, "4.35"),
    ]),

    # 连补两笔、一路阴跌、一次没卖 —— 增强"持亏"这一端
    ("588000", "科创50ETF", ETF, [
        ("B", "2025-12-15", 20000, "1.05"),
        ("B", "2026-03-20", 20000, "0.92"),
    ]),

    # 可转债 T+0：当天买当天卖，不受 T+1 限制，是校验撮合规则的好样本
    ("113050", "南银转债", BOND, [
        ("B", "2026-01-21", 20, "118.50"),
        ("S", "2026-01-21", 20, "121.30"),
        ("B", "2026-05-13", 30, "115.20"),
        ("S", "2026-05-13", 30, "116.90"),
    ]),

    # 有赚有亏的来回选手：2 胜 2 负，用来拉高"已实现"两端的样本量
    ("601899", "紫金矿业", STOCK, [
        ("B", "2025-11-18", 3000, "18.20"),
        ("S", "2026-01-06", 3000, "17.45"),
        ("B", "2026-02-25", 4000, "19.80"),
        ("S", "2026-03-06", 4000, "21.30"),
        ("B", "2026-05-27", 5000, "17.60"),
        ("S", "2026-06-30", 5000, "17.05"),
        ("B", "2026-07-21", 3000, "20.40"),
        ("S", "2026-08-04", 3000, "21.55"),
        ("B", "2026-09-28", 2000, "22.80"),
    ]),

    ("600030", "中信证券", STOCK, [
        ("B", "2026-01-12", 3000, "26.50"),
        ("S", "2026-02-03", 3000, "28.10"),
        ("B", "2026-06-11", 4000, "24.80"),
    ]),

    # 赚一笔就跑、亏一笔扛三个月 —— 单只标的内部就能看出处置效应
    ("000651", "格力电器", STOCK, [
        ("B", "2025-12-22", 2000, "42.60"),
        ("S", "2026-01-09", 2000, "44.15"),
        ("B", "2026-04-14", 3000, "46.80"),
        ("S", "2026-07-13", 3000, "43.20"),
    ]),

    # 打新中签后上市即卖
    ("601728", "中国电信", STOCK, [
        ("I", "2026-04-09", 1000, "6.50"),
        ("S", "2026-04-16", 1000, "7.20"),
    ]),

    # 两类"认不出来"的业务：正确行为是告警并继续，而不是整批失败
    ("000001", "平安银行", STOCK, [
        ("O", "2026-03-18", "担保品划转"),
    ]),
    ("510300", "沪深300ETF", ETF, [
        ("O", "2026-08-12", "利息归本"),
    ]),
]


def build_rows() -> list[dict]:
    """把 SCRIPT 展开成扁平的成交记录，并逐条算出手续费。"""
    rows: list[dict] = []
    for symbol, name, asset, legs in SCRIPT:
        for leg in legs:
            kind, day = leg[0], date.fromisoformat(leg[1])
            if kind in ("B", "S"):
                qty, price = D(leg[2]), D(leg[3])
                amount = money(qty * price)
                comm, stamp, transfer = fee_parts(amount, asset, kind)
                rows.append(dict(
                    dt=day, symbol=symbol, name=name, biz="证券买入" if kind == "B" else "证券卖出",
                    qty=qty if kind == "B" else -qty, price=price, amount=amount,
                    comm=comm, stamp=stamp, transfer=transfer,
                ))
            elif kind == "D":
                rows.append(dict(
                    dt=day, symbol=symbol, name=name, biz="股息入账",
                    qty=D("0"), price=D("0"), amount=money(leg[2]),
                    comm=D("0"), stamp=D("0"), transfer=D("0"),
                ))
            elif kind == "N":
                rows.append(dict(
                    dt=day, symbol=symbol, name=name, biz="红股入账",
                    qty=D(leg[2]), price=D("0"), amount=D("0"),
                    comm=D("0"), stamp=D("0"), transfer=D("0"),
                ))
            elif kind == "R":
                qty, price = D(leg[2]), D(leg[3])
                rows.append(dict(
                    dt=day, symbol=symbol, name=name, biz="配股缴款",
                    qty=qty, price=price, amount=-money(qty * price),
                    comm=D("0"), stamp=D("0"), transfer=D("0"),
                ))
            elif kind == "I":
                qty, price = D(leg[2]), D(leg[3])
                rows.append(dict(
                    dt=day, symbol=symbol, name=name, biz="新股入账",
                    qty=qty, price=price, amount=D("0"),
                    comm=D("0"), stamp=D("0"), transfer=D("0"),
                ))
            elif kind == "O":
                rows.append(dict(
                    dt=day, symbol=symbol, name=name, biz=str(leg[2]),
                    qty=D("0"), price=D("0"), amount=D("0"),
                    comm=D("0"), stamp=D("0"), transfer=D("0"),
                ))
            else:
                raise ValueError(f"未知 leg 类型: {kind}")
    rows.sort(key=lambda r: (r["dt"], r["symbol"], r["biz"]))
    return rows


def validate(rows: list[dict]) -> None:
    """自检：日期必须是工作日；股票/ETF 不能违反 T+1；卖出不能超过持仓。

    这三条只要有一条不成立，生成出来的账本就会出现"凭空卖出"或"成本漂移"，
    而那种错误极难在报告里看出来 —— 所以在源头卡死。
    """
    for r in rows:
        if r["dt"].weekday() >= 5:
            raise ValueError(f"非工作日: {r['dt']} {r['symbol']} {r['biz']}")

    holdings: dict[str, list[list]] = {}
    # 按 (日期, 业务类型优先级) 排序：同日买入必须排在卖出之前（可转债 T+0 依赖这一点）
    order = {"证券买入": 0, "新股入账": 0, "红股入账": 0, "配股缴款": 0,
             "股息入账": 1, "证券卖出": 2}
    for r in sorted(rows, key=lambda x: (x["dt"], x["symbol"], order.get(x["biz"], 3))):
        sym, biz = r["symbol"], r["biz"]
        qty = abs(r["qty"])
        if biz in ("证券买入", "新股入账", "配股缴款"):
            holdings.setdefault(sym, []).append([r["dt"], qty])
        elif biz == "红股入账":
            lots = holdings.get(sym, [])
            total = sum(l[1] for l in lots)
            if total <= 0:
                raise ValueError(f"{r['dt']} {sym} 红股入账时无底仓")
            ratio = (total + qty) / total
            for lot in lots:
                lot[1] = lot[1] * ratio
        elif biz == "证券卖出":
            lots = holdings.get(sym, [])
            available = sum(l[1] for l in lots)
            if available < qty:
                raise ValueError(
                    f"{r['dt']} {sym} 卖出 {qty} 超过持仓 {available}"
                )
            remaining = qty
            for lot in lots:
                if remaining <= 0:
                    break
                take = min(lot[1], remaining)
                lot[1] -= take
                remaining -= take
            holdings[sym] = [l for l in lots if l[1] > 0]
            # T+1：股票/ETF 当日买入当日不可卖；可转债（11 开头）豁免
            same_day_buy = any(
                r2["dt"] == r["dt"] and r2["symbol"] == sym
                and r2["biz"] in ("证券买入", "新股入账", "配股缴款")
                for r2 in rows
            )
            if same_day_buy and not sym.startswith("11"):
                raise ValueError(f"{r['dt']} {sym} 涉嫌 T+1 违规：当日买入当日卖出")


def _fmt(x: Decimal) -> str:
    return f"{x:.2f}"


def write_ths(rows: list[dict], path: Path) -> None:
    """同花顺风格：UTF-8 BOM、抬头两行、负数量表示卖出、单列"手续费"包干。"""
    lines: list[str] = [
        "资金账号,88888888",
        "查询区间,20251101-20260331 人民币",
        "",
        "成交日期,证券代码,证券名称,业务名称,成交数量,成交价格,成交金额,手续费",
    ]
    for r in rows:
        fee = money(r["comm"] + r["stamp"] + r["transfer"])
        lines.append(",".join([
            r["dt"].isoformat(), r["symbol"], r["name"], r["biz"],
            f"{r['qty']:.0f}", _fmt(r["price"]), _fmt(r["amount"]), _fmt(fee),
        ]))
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8-sig")


def write_ht(rows: list[dict], path: Path) -> None:
    """另一家风格：GB18030、抬头三行、正数量表示卖出、费用拆三列、列顺序不同。"""
    lines: list[str] = [
        "华泰证券股份有限公司 客户对账单",
        "资金账号:66666666",
        "查询区间:20260401-20261030",
        "发生日期,证券编码,证券简称,摘要,成交股数,成交均价,发生金额,净佣金,印花税,过户费",
    ]
    for r in rows:
        lines.append(",".join([
            r["dt"].isoformat(), r["symbol"], r["name"], r["biz"],
            f"{abs(r['qty']):.0f}", _fmt(r["price"]), _fmt(r["amount"]),
            _fmt(r["comm"]), _fmt(r["stamp"]), _fmt(r["transfer"]),
        ]))
    path.write_text("\r\n".join(lines) + "\r\n", encoding="gb18030")


def write_prices(path: Path) -> None:
    lines = ["证券代码,最新价"]
    for sym, px in CLOSE_2026_10_30.items():
        lines.append(f"{sym},{px}")
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8-sig")


def main() -> int:
    rows = build_rows()
    validate(rows)

    part1 = [r for r in rows if r["dt"] < CUT]
    part2 = [r for r in rows if r["dt"] >= CUT]

    p1 = HERE / "part1_2025-11_2026-03_ths.csv"
    p2 = HERE / "part2_2026-04_2026-10_ht.csv"
    write_ths(part1, p1)
    write_ht(part2, p2)
    write_prices(PRICES_OUT)

    total_fee = sum((r["comm"] + r["stamp"] + r["transfer"] for r in rows), D("0"))
    print(f"生成完成：共 {len(rows)} 条记录（前段 {len(part1)} / 后段 {len(part2)}）"
          f"，标的 {len({r['symbol'] for r in rows})} 只，费用合计 {money(total_fee)} 元")
    for p in (p1, p2, PRICES_OUT):
        print(f"  {p.relative_to(HERE.parent)}  ({p.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
