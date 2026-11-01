"""解析层入口。"""

from .base import ParseResult, ParseWarning, Parser
from .csv_generic import GenericCsvParser, detect_delimiter, read_text, to_date, to_decimal
from .registry import PARSERS, detect, parse_file

__all__ = [
    "PARSERS",
    "GenericCsvParser",
    "ParseResult",
    "ParseWarning",
    "Parser",
    "detect",
    "detect_delimiter",
    "parse_file",
    "read_text",
    "to_date",
    "to_decimal",
]
