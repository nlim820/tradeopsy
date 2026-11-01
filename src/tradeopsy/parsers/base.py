"""解析器协议。

为什么用"协议 + 注册表"而不是在主流程里写 `if 券商 == "同花顺"`：

真实交割单格式会变 —— 券商改版、不同开户年份导出模板不同、
同一个券商 App 和网页版列名都不一样。每加一种格式就改一次主流程，
是最容易引入回归的做法。

注册表把"怎么识别"和"怎么解析"都收敛进单个文件，
新增一种格式 = 新增一个文件 + 注册一行，主流程零改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..model import Event, EventType


@dataclass
class ParseWarning:
    """一行解析不成功时的记录。

    设计原则：**单行失败不能中断整批**。
    真实交割单里总有你没见过的业务条目（担保品划转、基金申赎、利息归本），
    整批报错会让用户觉得工具坏了；跳过并列出，让用户自己判断才合理。
    """

    row: int
    reason: str
    raw: str = ""

    def __str__(self) -> str:
        preview = self.raw[:60].replace("\n", " ")
        return f"第{self.row}行: {self.reason}" + (f" | {preview}" if preview else "")


@dataclass
class ParseResult:
    """一次解析的完整产物。"""

    events: list[Event] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)
    #: 解析元信息：编码、分隔符、列映射、总行数、识别出的券商风格。
    #: 存下来是为了排障 —— 用户说"我这份跑出来不对"时，第一件事就是看它。
    meta: dict = field(default_factory=dict)

    @property
    def ok_count(self) -> int:
        return len(self.events)

    def counts_by_type(self) -> dict[EventType, int]:
        out: dict[EventType, int] = {}
        for e in self.events:
            out[e.etype] = out.get(e.etype, 0) + 1
        return out

    def symbols(self) -> set[str]:
        return {e.symbol for e in self.events if e.symbol}


class Parser:
    """解析器基类。

    sniff() 返回 0~1 的置信度，注册表按置信度排序取最高者。
    不用布尔值的原因：将来可能有两种格式长得七成像，
    这时需要按"更像哪个"来选，而不是"先匹配上的赢"。
    """

    name: str = "base"
    #: 人类可读的说明，会出现在 CLI 的 --list-formats 里
    description: str = ""

    def sniff(self, path: Path, head: str) -> float:
        """看文件开头判断像不像自己能处理的格式。head 是已解码的前若干行。"""
        return 0.0

    def parse(self, path: Path) -> ParseResult:
        raise NotImplementedError
