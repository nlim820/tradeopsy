"""每日编排。

这里是整个自动化系统的"进程"，负责把 取数 → 撮合 → 估值 → 决策 串成一条链。

两个必须做对的工程点：

**① 逐日回放，而不是"只处理今天"。**
    如果你有三天没跑（周末、出门、Actions 挂了），系统必须能把这三天的
    撮合和估值全部补上，且结果和每天都跑完全一致。这是幂等性的定义。
    做法是把"当前日期"从状态里读出来，然后从它的次日起逐日推进。

**② 决策与执行隔一个交易日。**
    `pending_targets` 是 T 日收盘后算出来的，到 T+1 日开盘才执行。
    这个"隔一天"的设计看起来只是个小细节，实际上是整个项目价值观的落点：
    不允许任何当天信息影响当天成交。交易体检工具自己必须先不作弊。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from ..model import money
from . import broker, market, strategy
from .broker import Account, Fill
from .market import Bar

#: 交易流水列名。刻意和真实券商交割单对齐（中文业务名称），
#: 这样这份 CSV 可以直接被 tradeopsy 的解析层吃，不需要写专用适配器。
TRADE_COLUMNS = [
    "成交日期", "证券代码", "证券名称", "业务名称",
    "成交数量", "成交价格", "成交金额", "费用", "备注",
]
NAV_COLUMNS = [
    "日期", "总资产", "现金", "持仓市值", "当日收益率", "累计收益率", "持仓数",
]

#: 回看窗口（自然日）。策略需要 MA60 + 动量 20 + 冗余，回看 500 个自然日约 340 根 K 线，
#: 刚够冷启动。第一次运行会把这 340 根全部拉下来，之后每天只是增量。
LOOKBACK_DAYS = 500

SIDE_CN = {"buy": "证券买入", "sell": "证券卖出"}


@dataclass
class DayResult:
    dt: date
    fills: list[Fill] = field(default_factory=list)
    total: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    positions_value: Decimal = Decimal("0")
    decision: strategy.Decision | None = None


def _index(bars: dict[str, list[Bar]]) -> dict[str, dict[date, Bar]]:
    return {sym: {b.dt: b for b in lst} for sym, lst in bars.items()}


def trading_calendar(bars: dict[str, list[Bar]]) -> list[date]:
    """用沪深300ETF 的交易日作为日历。

    为什么不自己维护节假日表：A 股调休规则每年都变（春节、国庆前后常有
    周末补班），硬编码一份日历等于每年给自己埋一个 bug。用一只从不
    停牌的宽基 ETF 的实际 K 线日期，天然就是权威日历。
    """
    anchor = bars.get("510300")
    if not anchor:
        anchor = max(bars.values(), key=len, default=[])
    return sorted({b.dt for b in anchor})


def _append_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(columns)
        w.writerows(rows)


def _rewrite_nav(path: Path, rows: list[dict]) -> None:
    """nav.csv 整表重写。

    为什么不追加：逐日回放时如果某天算错了、修完重跑，追加会造成重复行。
    净值表很小（一年 250 行），重写成本可以忽略，换来的是永远不可能有重复。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NAV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_nav(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {r["日期"]: r for r in csv.DictReader(fh)}


def run(
    root: Path,
    *,
    initial_cash: str = "100000.00",
    today: date | None = None,
    dry_run: bool = False,
    cfg: strategy.StrategyConfig | None = None,
    verbose: bool = True,
) -> dict:
    """跑一次。返回本次运行的摘要（给 CLI 和 Actions 打印用）。

    today 参数存在的意义是可测试性：所有涉及"今天"的逻辑都必须能注入，
    否则测试就得依赖系统时间，跑一年后自动失效。
    """
    cfg = cfg or strategy.StrategyConfig()
    state_path = root / "paper" / "state.json"
    nav_path = root / "paper" / "nav.csv"
    trade_path = root / "paper" / "trades.csv"

    account = Account.load(state_path, initial_cash)
    end = today or date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    if verbose:
        print(f"[1/4] 取数 {start} → {end}（{len(strategy.UNIVERSE)} 只 ETF）")
    bars = market.fetch_many([s for s, _ in strategy.UNIVERSE], start, end)
    idx = _index(bars)
    calendar = trading_calendar(bars)

    if not calendar:
        raise market.FetchError("交易日历为空，取数异常")

    # ---- 数据健康检查 ----
    warnings: list[str] = []
    srcs = set(market.LAST_SOURCE.values())
    if srcs and srcs != {"tencent"}:
        warnings.append(
            f"未使用主源（实际：{', '.join(sorted(srcs))}），价格可能未复权"
        )
    if not all(b.adjusted for lst in bars.values() for b in lst):
        warnings.append("部分标的返回不复权数据，已启动除权跳空检测")
    for sym, lst in bars.items():
        for d, chg in market.suspicious_gaps(lst):
            warnings.append(f"{sym} {d} 单日 {chg:.2%}，疑似除权或极端行情")
    for w in warnings:
        print(f"[warn] {w}")

    last_done = account.as_of
    if last_done:
        todo = [d for d in calendar if d.isoformat() > last_done]
    else:
        # 首次运行：不撮合，只以最近一个交易日为起点建立基线
        todo = calendar[-1:]

    if not todo:
        if verbose:
            print(f"[=] 无新交易日（当前 {last_done}），跳过。")
        return {"new_days": 0, "as_of": last_done, "total": str(account.total_value())}

    if verbose:
        print(f"[2/4] 回放 {len(todo)} 个交易日")

    nav = _load_nav(nav_path)
    results: list[DayResult] = []
    prev_total = account.total_value()
    new_trade_rows: list[list[str]] = []

    for d in todo:
        bars_today = {s: m[d] for s, m in idx.items() if d in m}

        # ---- T+1 解锁：新的一天，昨日持仓变为可卖 ----
        broker.roll_day(account)

        # ---- 执行上一交易日收盘时定下的目标 ----
        fills = broker.execute_targets(
            account,
            account.pending_targets,
            bars_today,
            strategy.UNIVERSE_NAMES,
            d,
        )

        # ---- 收盘估值 ----
        closes = {s: m[d].close for s, m in idx.items() if d in m}
        broker.mark_to_market(account, closes)
        total = account.total_value()
        pos_value = money(total - account.cash)

        # ---- 用截至今日的历史做决策，存入 pending 供下一交易日开盘执行 ----
        hist = {
            s: [b for b in lst if b.dt <= d]
            for s, lst in bars.items()
        }
        # held 必须传当日收盘后的实际持仓，策略靠它区分"持有"与"新开仓"，
        # 两者的判定阈值不同（滞回机制）。
        decision = strategy.decide(hist, cfg, held=set(account.positions))
        account.pending_from = d.isoformat()
        account.pending_targets = dict(decision.targets)
        account.as_of = d.isoformat()

        # ---- 记录 ----
        day_ret = (total / prev_total - 1) if prev_total > 0 else Decimal("0")
        cum_ret = (total / account.initial_cash - 1) if account.initial_cash > 0 else Decimal("0")
        nav[d.isoformat()] = {
            "日期": d.isoformat(),
            "总资产": f"{total:.2f}",
            "现金": f"{account.cash:.2f}",
            "持仓市值": f"{pos_value:.2f}",
            "当日收益率": f"{day_ret:.6f}",
            "累计收益率": f"{cum_ret:.6f}",
            "持仓数": str(len(account.positions)),
        }
        prev_total = total
        results.append(
            DayResult(d, fills, total, account.cash, pos_value, decision)
        )

        for f in fills:
            if not f.ok:
                continue
            new_trade_rows.append([
                f.dt.isoformat(), f.symbol, f.name, SIDE_CN[f.side],
                f"{f.qty:.0f}", f"{f.price:.4f}", f"{f.amount:.2f}",
                f"{f.fee:.2f}", "paper-trading",
            ])

    if not dry_run:
        account.save(state_path)
        _append_csv(trade_path, TRADE_COLUMNS, new_trade_rows)
        _rewrite_nav(nav_path, [nav[k] for k in sorted(nav)])
        _append_decisions(root / "paper" / "decisions.jsonl", results)

    last = results[-1]
    if verbose:
        print(f"[3/4] 成交 {sum(1 for r in results for f in r.fills if f.ok)} 笔")
        for r in results:
            for f in r.fills:
                if not f.ok:
                    print(f"      {r.dt} 跳过 {f.symbol} {f.reason}")
                    continue
                print(
                    f"      {r.dt} {SIDE_CN[f.side]} {f.symbol} {f.name} "
                    f"{f.qty:.0f}股 @{f.price:.3f}  金额{f.amount:,.2f} 佣金{f.fee:.2f}"
                )
        print(f"[4/4] 期末 {last.dt}: 总资产 {last.total:,.2f}  现金 {last.cash:,.2f}")
        print(f"      {last.decision.summary() if last.decision else ''}")

    return {
        "new_days": len(todo),
        "as_of": account.as_of,
        "total": f"{account.total_value():.2f}",
        "cash": f"{account.cash:.2f}",
        "fills": sum(1 for r in results for f in r.fills if f.ok),
        "targets": {s: f"{w}" for s, w in account.pending_targets.items()},
        "warnings": warnings,
        "sources": sorted(market.LAST_SOURCE.values()),
    }


def _append_decisions(path: Path, results: list[DayResult]) -> None:
    """每天决策落一行 JSON，供复盘。

    单独存一份的理由：nav.csv 只有净值，trades.csv 只有成交，
    但"那天为什么空仓"这个信息两者都回答不了。出问题时最想知道的恰恰是它。
    """
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in results:
            if r.decision is None:
                continue
            fh.write(json.dumps({
                "date": r.dt.isoformat(),
                "targets": {s: str(w) for s, w in r.decision.targets.items()},
                "candidates": [
                    {
                        "symbol": c.symbol,
                        "name": c.name,
                        "close": str(c.close),
                        "ma20": str(c.ma_fast),
                        "ma60": str(c.ma_slow),
                        "mom": str(c.mom)[:8],
                        "passed": c.passed,
                        "reason": c.reason,
                        "rank": c.rank,
                    }
                    for c in r.decision.candidates
                ],
            }, ensure_ascii=False) + "\n")
