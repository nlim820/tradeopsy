"""解析器注册表与自动探测。

新增一种券商格式 = 写一个 Parser 子类 + 在这里加一行。主流程不动。
"""

from __future__ import annotations

from pathlib import Path

from .base import ParseResult, ParseWarning, Parser
from .csv_generic import GenericCsvParser, read_text

#: 按优先级排列。将来若有"某某券商专用解析器"，要放在通用解析器之前，
#: 因为专用解析器的置信度更高，逻辑也更准。
PARSERS: tuple[Parser, ...] = (GenericCsvParser(),)


def detect(path: Path) -> tuple[Parser | None, float]:
    """返回 (最匹配的解析器, 置信度)。

    只读文件前 8KB 做判断。大文件（几年流水）不必要为了嗅探全读一遍。
    """
    head, _ = read_text_head(path, limit=8192)
    best: tuple[Parser | None, float] = (None, 0.0)
    for p in PARSERS:
        score = p.sniff(path, head)
        if score > best[1]:
            best = (p, score)
    return best


def read_text_head(path: Path, limit: int = 8192) -> tuple[str, str]:
    """只解码文件头部。用于嗅探，避免为判断格式把整份文件读进来。"""
    raw = path.read_bytes()[:limit]
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def parse_file(path: Path) -> ParseResult:
    """自动探测并解析。探测不到时返回带告警的空结果，而不是抛异常。

    理由：用户可能一次给了 10 个文件，其中一个格式不认识。
    因为一个坏文件就整批失败，体验很差。
    """
    parser, score = detect(path)
    if parser is None or score <= 0:
        return ParseResult(
            warnings=[ParseWarning(0, "无法识别文件格式，可能不是 CSV 交割单")],
            meta={"parser": None, "file": path.name},
        )
    result = parser.parse(path)
    result.meta["file"] = path.name
    result.meta["confidence"] = round(score, 2)
    return result


__all__ = [
    "PARSERS",
    "ParseResult",
    "ParseWarning",
    "Parser",
    "detect",
    "parse_file",
    "read_text",
]
