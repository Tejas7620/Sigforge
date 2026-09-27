"""Pipeline — orchestrator, DAG scheduler, parameter ledger."""

from .data_types import (
    PipelineResult,
    DemodResult,
    SignalProfile,
    ParameterEstimates,
    FECResult,
)
from .config import PipelineConfig


def __getattr__(name: str):
    if name == "PipelineOrchestrator":
        from .orchestrator import PipelineOrchestrator
        return PipelineOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineResult",
    "DemodResult",
    "SignalProfile",
    "ParameterEstimates",
    "FECResult",
    "PipelineConfig",
    "PipelineOrchestrator",
]
