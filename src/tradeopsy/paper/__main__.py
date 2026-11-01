"""命令行入口：`python -m tradeopsy.paper`

为什么给日更任务单独做 CLI：
GitHub Actions 里要跑的是一条确定性命令，不能依赖任何交互。
所以这里只接受参数、只输出纯文本、只用退出码表达成败（0 成功 / 1 失败）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from .runner import run


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tradeopsy.paper",
        description="纸面交易日更任务：取数 → 撮合 → 估值 → 决策",
    )
    ap.add_argument(
        "--root",
        default=".",
        help="仓库根目录（默认当前目录），账户数据写到 <root>/paper/",
    )
    ap.add_argument("--cash", default="100000.00", help="初始资金（仅首次生效）")
    ap.add_argument("--today", default=None, help="截止日期 YYYY-MM-DD，默认今天")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算不落盘，用来验证逻辑或做回放试验",
    )
    ap.add_argument("--quiet", action="store_true", help="静默模式")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        summary = run(
            root,
            initial_cash=args.cash,
            today=_parse_date(args.today),
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except Exception as exc:  # noqa: BLE001
        # Actions 里失败必须响。静默吞掉异常 = 账本悄悄停更而你毫不知情。
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        print(f"{summary['as_of']} total={summary['total']} fills={summary['fills']}")
    if args.dry_run:
        print("(dry-run：未写入任何文件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
