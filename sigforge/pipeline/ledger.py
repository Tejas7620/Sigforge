"""
Parameter Ledger — centralized parameter store with provenance tracking.

Every parameter in SigForge carries a status tag indicating how it was determined.
The ledger is the single source of truth for all parameters across the pipeline.
Downstream confidence is capped by the weakest input status.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ParameterStatus(enum.Enum):
    """How a parameter value was determined.

    Ordered by reliability (KNOWN is strongest, UNKNOWN is weakest).
    """
    KNOWN = "KNOWN"          # Read directly from trustworthy metadata (WAV header, SigMF)
    USER = "USER"            # Explicitly set/overridden by the analyst
    INFERRED = "INFERRED"    # Computed from signal statistics with a confidence score
    ASSUMED = "ASSUMED"      # System default used because nothing better is available
    UNKNOWN = "UNKNOWN"      # Not determinable; explicitly declared

    @property
    def reliability_rank(self) -> int:
        """Higher = more reliable."""
        return {
            ParameterStatus.KNOWN: 5,
            ParameterStatus.USER: 4,
            ParameterStatus.INFERRED: 3,
            ParameterStatus.ASSUMED: 2,
            ParameterStatus.UNKNOWN: 1,
        }[self]


@dataclass
class ParameterEntry:
    """A single parameter with its value, status, and metadata."""
    name: str
    value: Any
    status: ParameterStatus
    confidence: float = 1.0          # 0.0-1.0, meaningful only for INFERRED
    unit: str = ""                   # e.g. "Hz", "dB", "samples"
    source: str = ""                 # e.g. "wav_header", "autocorrelation", "user_input"
    timestamp: float = field(default_factory=time.time)
    alternatives: List[Any] = field(default_factory=list)  # other candidate values

    def __repr__(self) -> str:
        return (
            f"Param({self.name}={self.value} [{self.status.value}] "
            f"conf={self.confidence:.2f} {self.unit})"
        )


class ParameterLedger:
    """
    Centralized parameter store with provenance tracking.

    Usage:
        ledger = ParameterLedger()
        ledger.set("sample_rate", 2048000, ParameterStatus.KNOWN, source="wav_header", unit="Hz")
        ledger.set("symbol_rate", 2400, ParameterStatus.INFERRED, confidence=0.85, source="autocorrelation", unit="Hz")

        sr = ledger.get("sample_rate")         # → ParameterEntry
        sr.value                                # → 2048000
        sr.status                               # → ParameterStatus.KNOWN

        ledger.override("symbol_rate", 4800)    # Sets status to USER
    """

    def __init__(self) -> None:
        self._params: Dict[str, ParameterEntry] = {}
        self._history: List[Dict[str, Any]] = []  # audit trail

    def set(
        self,
        name: str,
        value: Any,
        status: ParameterStatus,
        confidence: float = 1.0,
        unit: str = "",
        source: str = "",
        alternatives: Optional[List[Any]] = None,
    ) -> None:
        """Set or update a parameter with full provenance."""
        entry = ParameterEntry(
            name=name,
            value=value,
            status=status,
            confidence=confidence,
            unit=unit,
            source=source,
            alternatives=alternatives or [],
        )
        # Record history
        if name in self._params:
            self._history.append({
                "action": "update",
                "name": name,
                "old": self._params[name],
                "new": entry,
                "timestamp": time.time(),
            })
        else:
            self._history.append({
                "action": "create",
                "name": name,
                "new": entry,
                "timestamp": time.time(),
            })
        self._params[name] = entry

    def get(self, name: str) -> Optional[ParameterEntry]:
        """Get a parameter entry, or None if not set."""
        return self._params.get(name)

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get just the value of a parameter, with a default fallback."""
        entry = self._params.get(name)
        if entry is None:
            return default
        return entry.value

    def get_status(self, name: str) -> ParameterStatus:
        """Get the status of a parameter. Returns UNKNOWN if not set."""
        entry = self._params.get(name)
        if entry is None:
            return ParameterStatus.UNKNOWN
        return entry.status

    def override(self, name: str, value: Any, source: str = "user_input") -> None:
        """Override a parameter value (sets status to USER)."""
        existing = self._params.get(name)
        unit = existing.unit if existing else ""
        self.set(name, value, ParameterStatus.USER, confidence=1.0, unit=unit, source=source)

    def mark_unknown(self, name: str, unit: str = "") -> None:
        """Explicitly mark a parameter as unknown."""
        self.set(name, None, ParameterStatus.UNKNOWN, confidence=0.0, unit=unit, source="system")

    @property
    def weakest_status(self) -> ParameterStatus:
        """Return the weakest (least reliable) status among all parameters."""
        if not self._params:
            return ParameterStatus.UNKNOWN
        return min(self._params.values(), key=lambda e: e.status.reliability_rank).status

    @property
    def max_allowed_confidence(self) -> float:
        """Maximum downstream confidence, capped by weakest input status.

        If any critical parameter is ASSUMED or UNKNOWN, downstream confidence
        is capped to prevent false certainty.
        """
        _cap_map = {
            ParameterStatus.KNOWN: 1.0,
            ParameterStatus.USER: 1.0,
            ParameterStatus.INFERRED: 0.85,
            ParameterStatus.ASSUMED: 0.60,
            ParameterStatus.UNKNOWN: 0.30,
        }
        return _cap_map.get(self.weakest_status, 0.30)

    def all_params(self) -> Dict[str, ParameterEntry]:
        """Return a copy of all parameters."""
        return dict(self._params)

    def summary(self) -> str:
        """Return a human-readable summary of all parameters."""
        lines = ["Parameter Ledger:"]
        for name, entry in sorted(self._params.items()):
            badge = {
                ParameterStatus.KNOWN: "✓",
                ParameterStatus.USER: "👤",
                ParameterStatus.INFERRED: "~",
                ParameterStatus.ASSUMED: "?",
                ParameterStatus.UNKNOWN: "✗",
            }.get(entry.status, "?")
            val_str = f"{entry.value}" if entry.value is not None else "N/A"
            if entry.unit:
                val_str += f" {entry.unit}"
            conf_str = f" (conf={entry.confidence:.2f})" if entry.status == ParameterStatus.INFERRED else ""
            lines.append(f"  {badge} {name}: {val_str} [{entry.status.value}]{conf_str}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, dict]:
        """Serialize ledger to a dict for JSON export."""
        result = {}
        for name, entry in self._params.items():
            result[name] = {
                "value": entry.value if not isinstance(entry.value, complex) else str(entry.value),
                "status": entry.status.value,
                "confidence": entry.confidence,
                "unit": entry.unit,
                "source": entry.source,
            }
        return result

    def __repr__(self) -> str:
        return f"ParameterLedger({len(self._params)} params, weakest={self.weakest_status.value})"
