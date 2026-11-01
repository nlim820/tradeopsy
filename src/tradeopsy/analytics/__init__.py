"""交易行为归因子包。

目前只装了一件事：处置效应（Disposition Effect）度量。
这是整个项目最有学术含量的部分，也是简历上最拿得出手的一点。
"""

from .disposition import (
    DispositionReport,
    OpenPositionMV,
    compute_disposition,
    open_positions_from_ledger,
)
from .significance import SignificanceResult, bootstrap_disposition

__all__ = [
    "DispositionReport",
    "OpenPositionMV",
    "compute_disposition",
    "open_positions_from_ledger",
    "SignificanceResult",
    "bootstrap_disposition",
]
