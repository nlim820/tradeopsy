"""纸面交易引擎（paper trading）。

定位重申一遍，避免后面越做越歪：
这是一个**研究用**的无人值守账户，不是交易系统。
它不下任何真实委托，不接任何券商接口，产出物是一份 CSV 流水，
交给 tradeopsy 的体检引擎去分析。

子模块：
    market    行情获取（东财 → 腾讯 双源回退，零第三方依赖）
    strategy  双均线轮动，产出目标权重
    broker    撮合（T+1 / 一字板 / 最低佣金）与账户状态
    runner    逐日回放编排，幂等、可补跑
"""

from .broker import Account, Fill, Position, commission, execute_targets
from .market import Bar, FetchError, fetch_bars, fetch_many
from .runner import run
from .strategy import UNIVERSE, UNIVERSE_NAMES, Decision, StrategyConfig, decide

__all__ = [
    "Account",
    "Bar",
    "Decision",
    "FetchError",
    "Fill",
    "Position",
    "StrategyConfig",
    "UNIVERSE",
    "UNIVERSE_NAMES",
    "commission",
    "decide",
    "execute_targets",
    "fetch_bars",
    "fetch_many",
    "run",
]
