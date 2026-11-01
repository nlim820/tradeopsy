"""从公开行情接口拉日线收盘价，生成 tradeopsy 能吃的逐日价格序列。

    为什么需要这个脚本
    ------------------
    `--price-history` 要的是「日期,证券代码,收盘价」长格式 CSV。
    券商不提供这个文件，行情软件导出的格式又各不相同。
    没有它，处置效应只能算「已实现」一半 —— 分母里的浮盈浮亏全缺，
    PGR/PLR 结论是残缺的。

    ！！！复权口径：默认必须是「不复权」，不要改成前复权 ！！！
    ---------------------------------------------------------
    账本里的成本是**当时实际成交价**（送转只摊薄每股成本、总成本不变，
    见 ledger.py 的 _split）。前复权会把历史价按除权因子往下调，
    于是「成本 120 元 vs 前复权现价 92 元」→ 凭空多出一笔巨额浮亏，
    而这笔浮亏会静默进入 PGR/PLR 的分母，分母一错整份报告就废了。
    以本仓示例账户为例：五粮液 2026-06-05 有 10 送 3，
    不复权是 120.60 →（除权后约 92.8），账本同时把成本摊薄到约 92.8，
    两边才对得上。用前复权就永远对不上。

    数据源选择（2026-11-02 实测）
    ----------------------------
        腾讯 stock_zh_a_hist_tx   股票✅ ETF✅ 可转债✗  支持日期区间，最快
        新浪 stock_zh_a_daily     股票✅ ETF✗ 可转债✗  支持日期区间
        新浪 fund_etf_hist_sina   ETF✅              不支持日期区间，返全历史
        新浪 bond_zh_hs_cov_daily 可转债✅            不支持日期区间，返全历史
        东财 stock_zh_a_hist      ——本机实测被限流（Push2RateLimitError），不作源

    用法
    ----
        # 最省事：直接从交割单推断标的与区间
        python tools/fetch_prices.py --from-ledger private/

        # 手动指定标的
        python tools/fetch_prices.py --symbols 600519,000858,510300

        # 先看会请求什么，不发请求
        python tools/fetch_prices.py --from-ledger private/ --dry-run

        # 生成后直接出报告
        PYTHONPATH=src python -m tradeopsy report private \\
            --price-history private/price_history.csv -o report.html

    akshare 不是 tradeopsy 的依赖（项目核心保持零第三方依赖），
    只有这个脚本需要，用前先 `pip install akshare`。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

# akshare 内部用 tqdm 打进度条，会把输出搅乱。tqdm>=4.66 认这个变量。
os.environ.setdefault("TQDM_DISABLE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"

#: 刻意留空 = 不复权。理由见模块文档，改它之前先读完那段。
DEFAULT_ADJUST = ""


# ---------------------------------------------------------------- 代码分类


def market_of(code: str) -> str:
    """6 位代码 → 交易所前缀。

    代码段（这块错了会拉到别的标的或直接 404）：
        沪市  60xxxx 主板 · 68xxxx 科创板 · 90xxxx B股
              50/51/52/56/58xxxx ETF · 110/111/113 主板转债 · 118 科创板转债
        深市  00xxxx 主板 · 30xxxx 创业板 · 20xxxx B股
              15/16/18xxxx 基金 · 123/127/128xxxx 转债
        北交所 43/83/87/88/92xxxx
    """
    c = code.strip()
    if c.startswith(("60", "68", "90", "50", "51", "52", "56", "58", "11", "13")):
        return "sh"
    if c.startswith(("00", "30", "20", "15", "16", "18", "12", "39")):
        return "sz"
    if c.startswith(("4", "8", "9")):
        return "bj"
    return "sh" if c[:1] in "56" else "sz"


def kind_of(code: str) -> str:
    """6 位代码 → stock / etf / bond。决定用哪个接口。"""
    c = code.strip()
    if c.startswith(("11", "12")):
        return "bond"
    if c.startswith(("15", "16", "18", "50", "51", "52", "56", "58")):
        return "etf"
    return "stock"


# ---------------------------------------------------------------- 值转换


def _to_date(v) -> date | None:
    """行情接口的日期列类型五花八门，统一成 date。"""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if hasattr(v, "date") and not isinstance(v, str):   # pandas.Timestamp
        try:
            return v.date()
        except Exception:
            pass
    s = str(v).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _to_dec(v) -> Decimal | None:
    """收盘价统一成 Decimal。新浪部分接口会返回 bytes，要单独处理。"""
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    try:
        d = Decimal(str(v).strip())
    except (InvalidOperation, ValueError):
        return None
    if d <= 0:
        return None
    # 先多留一位，落盘时再按标的类型定小数位（股票 2 位，ETF/转债 3 位）。
    # 在这里就截断会把可转债的 0.001 精度提前丢掉。
    return d.quantize(Decimal("0.0001"))


def _rows_from_df(df, date_key: str = "date", close_key: str = "close"):
    """DataFrame → [(date, Decimal)]，坏行静默丢弃（记数）。"""
    if df is None or len(df) == 0:
        return [], 0
    cols = {str(c).strip().lower(): c for c in df.columns}
    dc = cols.get(date_key) or cols.get("日期") or cols.get("trade_date")
    cc = cols.get(close_key) or cols.get("收盘") or cols.get("收盘价")
    if dc is None or cc is None:
        raise KeyError(f"返回列里找不到日期/收盘价：{list(df.columns)}")
    out, bad = [], 0
    for d_raw, c_raw in zip(df[dc].tolist(), df[cc].tolist()):
        d = _to_date(d_raw)
        p = _to_dec(c_raw)
        if d is None or p is None:
            bad += 1
            continue
        out.append((d, p))
    return out, bad


# ---------------------------------------------------------------- 各数据源


def _src_tx(code: str, start: date, end: date, adjust: str):
    """腾讯。股票与 ETF 都吃这个接口，且支持日期区间，首选。"""
    import akshare as ak

    df = ak.stock_zh_a_hist_tx(
        symbol=f"{market_of(code)}{code}",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    rows, bad = _rows_from_df(df)
    return rows, bad, len(rows)


def _src_sina_stock(code: str, start: date, end: date, adjust: str):
    """新浪股票日线。接口对日期参数生效，但偶尔返回超出区间的行，统一本地过滤。"""
    import akshare as ak

    df = ak.stock_zh_a_daily(
        symbol=f"{market_of(code)}{code}",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    rows, bad = _rows_from_df(df)
    keep = [r for r in rows if start <= r[0] <= end]
    return keep, bad, len(rows)


def _src_sina_etf(code: str, start: date, end: date, adjust: str):
    """新浪 ETF / LOF。不支持日期参数，返回全历史，本地裁剪。"""
    import akshare as ak

    df = ak.fund_etf_hist_sina(symbol=f"{market_of(code)}{code}")
    rows, bad = _rows_from_df(df)
    keep = [r for r in rows if start <= r[0] <= end]
    return keep, bad, len(rows)


def _src_sina_bond(code: str, start: date, end: date, adjust: str):
    """新浪沪深可转债。同样返回全历史，本地裁剪。

    可转债没有复权概念，adjust 参数在调用层就被忽略 —— 传 qfq 进来也按不复权取。

    注意：标的尚未上市或已退市时，akshare 内部会抛 KeyError('date')
    而不是返回空表，这个异常由 fetch_one 归类成「接口无数据」。
    """
    import akshare as ak

    df = ak.bond_zh_hs_cov_daily(symbol=f"{market_of(code)}{code}")
    rows, bad = _rows_from_df(df)
    keep = [r for r in rows if start <= r[0] <= end]
    return keep, bad, len(rows)


#: 每类标的的源优先级链。前一个失败或返空就试下一个。
SOURCE_CHAIN = {
    "stock": [("腾讯", _src_tx), ("新浪", _src_sina_stock)],
    "etf": [("腾讯", _src_tx), ("新浪ETF", _src_sina_etf)],
    "bond": [("新浪转债", _src_sina_bond)],
}


# ---------------------------------------------------------------- 缓存


def _cache_path(cache_dir: Path, code: str, start: date, end: date) -> Path:
    return cache_dir / f"{code}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def _read_cache(path: Path):
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = []
            for r in csv.DictReader(fh):
                d = _to_date(r.get("日期"))
                p = _to_dec(r.get("收盘价"))
                if d and p:
                    rows.append((d, p))
    except Exception:
        return None
    return rows or None


def _write_cache(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["日期", "收盘价"])
        for d, p in rows:
            w.writerow([d.isoformat(), f"{p}"])


# ---------------------------------------------------------------- 主流程


def fetch_one(code: str, start: date, end: date, adjust: str,
              cache_dir: Path | None, pause: float):
    """取一只标的。返回 (rows, source_name, note)。

    note 在两个场景下有值：
        成功时 —— 脏行提示
        失败时 —— 人话版的失败原因（rows 为空）
    """
    if cache_dir is not None:
        cp = _cache_path(cache_dir, code, start, end)
        cached = _read_cache(cp)
        if cached:
            dmin, dmax = cached[0][0], cached[-1][0]
            # 缓存必须覆盖请求区间，否则可能是上次被截断的结果
            if dmin <= start + timedelta(days=5) and dmax >= end - timedelta(days=5):
                return cached, "缓存", ""

    kind = kind_of(code)
    errs: list[str] = []
    reason = ""
    for name, fn in SOURCE_CHAIN[kind]:
        try:
            rows, bad, raw = fn(code, start, end, adjust)
        except KeyError:
            # akshare 在接口返回空结构时抛 KeyError('date') —— 不是我们的 bug，
            # 是这只标的根本没有行情（未上市 / 已退市 / 代码有误）。
            errs.append(f"{name}: 空结构")
            reason = "接口无数据：该标的可能尚未上市、已退市，或代码有误"
            time.sleep(pause)
            continue
        except Exception as exc:                    # 网络、限流、接口改名
            errs.append(f"{name}: {type(exc).__name__} {str(exc)[:50]}")
            time.sleep(pause)
            continue
        time.sleep(pause)
        if rows:
            rows.sort(key=lambda r: r[0])
            if cache_dir is not None:
                _write_cache(_cache_path(cache_dir, code, start, end), rows)
            note = f"{bad} 行脏数据已丢弃" if bad else ""
            return rows, name, note
        if raw:
            reason = (f"接口有 {raw} 条历史，但全部落在 {start} ~ {end} 之外"
                      f"（该标的在这个区间未上市，或期间已退市）")
        else:
            reason = "接口无数据：该标的可能尚未上市、已退市，或代码有误"
        errs.append(f"{name}: 区间内无数据")
    return [], "", reason or "；".join(errs)


def write_output(rows, out: Path) -> int:
    """按 tradeopsy 能吃的格式落盘。返回写入行数。

    格式对齐 examples/sample_account_price_history.csv：
        UTF-8 BOM（Excel 直接打开不乱码）· CRLF · 日期,证券代码,收盘价
    按 (代码, 日期) 排序，方便人工核对某只标的。
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r[1], r[0]))
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["日期", "证券代码", "收盘价"])
        for d, code, p in rows:
            # 小数位按标的类型定：股票最小报价单位 0.01，ETF/转债 0.001。
            # 统一 3 位不会错，但和券商导出对不上，人工核对时容易看花眼。
            dp = 2 if kind_of(code) == "stock" else 3
            w.writerow([d.isoformat(), code, f"{p:.{dp}f}"])
    return len(rows)


def symbols_from_ledger(path: Path):
    """从交割单里推断标的与日期区间。"""
    sys.path.insert(0, str(SRC))
    from tradeopsy.load import load_dataset

    ds = load_dataset(path)
    if not ds.events:
        raise SystemExit(f"交割单里没解析出任何成交：{path}")
    syms = sorted({ev.symbol for ev in ds.events})
    ds_dates = sorted(ev.dt for ev in ds.events)
    return syms, ds_dates[0], ds_dates[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="拉日线收盘价，生成 tradeopsy 的 --price-history 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-ledger", metavar="路径",
                     help="交割单文件或目录，自动推断标的与区间（推荐）")
    src.add_argument("--symbols", metavar="代码",
                     help="手动指定，逗号分隔，如 600519,000858,510300")

    ap.add_argument("--start", help="起始日期 YYYY-MM-DD（--symbols 时必填）")
    ap.add_argument("--end", help="结束日期 YYYY-MM-DD，默认今天")
    ap.add_argument("-o", "--out", default="private/price_history.csv",
                    help="输出文件，默认 private/price_history.csv（该目录已被 git 忽略）")
    ap.add_argument("--adjust", default=DEFAULT_ADJUST,
                    choices=["", "qfq", "hfq"],
                    help="复权方式。默认不复权 —— 改成 qfq/hfq 会与账本成本口径错开，见脚本文档")
    ap.add_argument("--pause", type=float, default=0.8,
                    help="每次请求后的等待秒数，防限流，默认 0.8")
    ap.add_argument("--cache-dir", default="private/.cache",
                    help="原始结果缓存目录，默认 private/.cache，传空字符串关闭")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要请求的内容，不发请求")
    args = ap.parse_args(argv)

    # ---- 确定标的与区间
    if args.from_ledger:
        syms, d_min, d_max = symbols_from_ledger(Path(args.from_ledger))
        # 起始日往前推 7 天：卖出日若逢停牌要沿用前一交易日价，多拉几天不亏
        start = d_min - timedelta(days=7)
        end = date.today()
        print(f"从交割单推断：{len(syms)} 只标的，成交区间 {d_min} ~ {d_max}")
    else:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if not args.start:
            ap.error("--symbols 需要同时给 --start")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else date.today()
        print(f"手动指定：{len(syms)} 只标的，区间 {start} ~ {end}")

    bad_codes = [s for s in syms if not (len(s) == 6 and s.isdigit())]
    if bad_codes:
        print(f"以下代码不是 6 位数字，已跳过：{'、'.join(bad_codes)}", file=sys.stderr)
        syms = [s for s in syms if s not in bad_codes]
    if not syms:
        raise SystemExit("没有可请求的标的")

    if args.adjust:
        print("\n⚠ 你指定了复权口径 " + args.adjust + "：")
        print("  账本成本是原始成交价，复权价会让「成本 vs 现价」错开，")
        print("  浮盈浮亏系统性失真。除非你清楚自己在做什么，否则用默认的不复权。\n")

    if args.dry_run:
        print(f"\n将请求 {len(syms)} 只（不发请求）：")
        for s in syms:
            print(f"  {s}  {kind_of(s):5s} {market_of(s)}{s}")
        return 0

    try:
        import akshare  # noqa: F401
    except ImportError:
        raise SystemExit(
            "缺 akshare。这是本脚本唯一的第三方依赖（tradeopsy 核心仍是零依赖）：\n"
            "    pip install akshare"
        )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # ---- 逐只拉取
    all_rows: list[tuple[date, str, Decimal]] = []
    ok, failed, notes = 0, [], []
    t0 = time.time()
    for i, code in enumerate(syms, 1):
        rows, src_name, note = fetch_one(code, start, end, args.adjust, cache_dir, args.pause)
        if rows:
            ok += 1
            all_rows.extend((d, code, p) for d, p in rows)
            tag = f"  [{src_name}]"
            extra = f"  {note}" if note else ""
            print(f"({i}/{len(syms)}) {code}  {len(rows):>4d} 条{tag}{extra}")
            if note:
                notes.append(f"{code}: {note}")
        else:
            failed.append((code, note))
            print(f"({i}/{len(syms)}) {code}  失败：{note}", file=sys.stderr)

    if not all_rows:
        raise SystemExit("一只都没取到。检查代码是否正确、网络是否可用。")

    # ---- 落盘
    out = Path(args.out)
    write_output(all_rows, out)
    span = f"{min(r[0] for r in all_rows)} ~ {max(r[0] for r in all_rows)}"
    print(f"\n已写入 {out}：{len(all_rows)} 条 · {ok}/{len(syms)} 只成功 · "
          f"{span} · 耗时 {time.time() - t0:.0f}s")

    if notes:
        print("\n数据质量：")
        for n in notes:
            print(f"  {n}")

    if failed:
        print(f"\n{len(failed)} 只没取到，这些标的的浮盈浮亏不会进分母：")
        for code, why in failed:
            print(f"  {code}  {why}")
        print("  它们仍会参与 FIFO 配对（已实现照样算），只是账面端缺价 → 从 PGR/PLR 分母里缺席。")
        print("  若失败标的都是在很早以前交易的，先确认交割单的日期区间是否覆盖完整。")

    if args.adjust:
        print("\n⚠ 本次用的是复权价，报告里的浮盈浮亏会和券商 App 对不上。")

    print("\n下一步：")
    target = args.from_ledger or "private"
    print(f"  PYTHONPATH=src python -m tradeopsy report {target} \\\n"
          f"      --price-history {out} -o report.html")
    print("提醒：这个文件暴露你持有过哪些标的，别随手放公共仓库。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
