"""
Signal Hypothesis — end-to-end signal description with evidence scoring.

A SignalHypothesis represents one complete interpretation of the input signal:
modulation + symbol rate + FEC + interleaver + frame structure, with per-evidence
confidence scores and a combined badge.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sigforge.classify.modulation_types import ModulationType


class ConfidenceBadge(enum.Enum):
    """Color-coded confidence badge for GUI display."""
    GREEN = "GREEN"      # ≥0.85: CRC pass / syndrome=0 / re-encode match >99%
    YELLOW = "YELLOW"    # 0.60–0.84: Multiple independent indicators agree
    ORANGE = "ORANGE"    # 0.30–0.59: Single weak indicator, inconclusive
    GREY = "GREY"        # <0.30: Insufficient evidence

    @staticmethod
    def from_confidence(confidence: float, has_hard_validation: bool = False) -> "ConfidenceBadge":
        """Assign badge based on confidence score and hard validation flags."""
        if has_hard_validation:
            return ConfidenceBadge.GREEN
        if confidence >= 0.85:
            return ConfidenceBadge.GREEN
        elif confidence >= 0.60:
            return ConfidenceBadge.YELLOW
        elif confidence >= 0.30:
            return ConfidenceBadge.ORANGE
        else:
            return ConfidenceBadge.GREY

    @property
    def emoji(self) -> str:
        return {
            ConfidenceBadge.GREEN: "🟢",
            ConfidenceBadge.YELLOW: "🟡",
            ConfidenceBadge.ORANGE: "🟠",
            ConfidenceBadge.GREY: "⚪",
        }[self]

    @property
    def color_hex(self) -> str:
        return {
            ConfidenceBadge.GREEN: "#22c55e",
            ConfidenceBadge.YELLOW: "#eab308",
            ConfidenceBadge.ORANGE: "#f97316",
            ConfidenceBadge.GREY: "#9ca3af",
        }[self]


# Default evidence weights (tunable)
DEFAULT_EVIDENCE_WEIGHTS: Dict[str, float] = {
    "cumulants": 0.15,
    "ml_classifier": 0.15,
    "constellation_quality": 0.10,
    "timing_carrier_lock": 0.10,
    "symbol_rate_agreement": 0.10,
    "spectrum_bandwidth": 0.10,
    "fec_syndrome": 0.10,
    "crc_pass": 0.05,
    "re_encoding_match": 0.05,
    "frame_sync_correlation": 0.05,
    "snr_consistency": 0.05,
}


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting or contradicting a hypothesis."""
    source: str           # evidence source key (matches weight table)
    score: float          # 0.0 to 1.0 (1.0 = strong support)
    weight: float         # weight for this source
    detail: str = ""      # human-readable explanation
    raw_value: Any = None # underlying measurement (e.g., EVM in dB)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class SignalHypothesis:
    """
    A complete end-to-end interpretation of the input signal.

    Represents one path through the hypothesis tree:
    modulation + symbol rate + FEC + interleaver + frame structure.
    """
    # Signal parameters
    modulation: ModulationType = ModulationType.UNKNOWN
    symbol_rate: Optional[float] = None        # Hz
    samples_per_symbol: Optional[float] = None
    carrier_offset: Optional[float] = None     # Hz

    # FEC parameters
    fec_type: Optional[str] = None             # e.g., "Conv(K=7, R=1/2)"
    fec_codec: Optional[str] = None            # codec identifier

    # Interleaver parameters
    interleaver_type: Optional[str] = None     # e.g., "Block(16x32)"
    interleaver_params: Dict[str, Any] = field(default_factory=dict)

    # Frame structure
    frame_length: Optional[int] = None         # bits
    sync_word: Optional[str] = None            # hex string
    crc_type: Optional[str] = None             # e.g., "CRC-16-CCITT"

    # Evidence and confidence
    evidence: List[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    badge: ConfidenceBadge = ConfidenceBadge.GREY
    has_hard_validation: bool = False          # CRC pass, syndrome=0, etc.

    # Intermediate results (for GUI display)
    evm_db: Optional[float] = None
    ber_estimate: Optional[float] = None
    decoded_bits: Optional[int] = None         # count of decoded bits
    warnings: List[str] = field(default_factory=list)

    def compute_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute combined confidence from all evidence items."""
        if not self.evidence:
            self.confidence = 0.0
            self.badge = ConfidenceBadge.GREY
            return 0.0

        w = weights or DEFAULT_EVIDENCE_WEIGHTS
        total_score = 0.0
        total_weight = 0.0

        for item in self.evidence:
            weight = w.get(item.source, item.weight)
            total_score += item.score * weight
            total_weight += weight

        self.confidence = total_score / total_weight if total_weight > 0 else 0.0
        self.badge = ConfidenceBadge.from_confidence(self.confidence, self.has_hard_validation)
        return self.confidence

    def add_evidence(
        self,
        source: str,
        score: float,
        detail: str = "",
        raw_value: Any = None,
    ) -> None:
        """Add an evidence item using default weights."""
        weight = DEFAULT_EVIDENCE_WEIGHTS.get(source, 0.05)
        self.evidence.append(EvidenceItem(
            source=source,
            score=max(0.0, min(1.0, score)),
            weight=weight,
            detail=detail,
            raw_value=raw_value,
        ))

    @property
    def description(self) -> str:
        """Human-readable one-line description of this hypothesis."""
        parts = [self.modulation.value]
        if self.fec_type:
            parts.append(f"+ {self.fec_type}")
        if self.interleaver_type:
            parts.append(f"+ {self.interleaver_type}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        """Serialize for JSON report."""
        return {
            "modulation": self.modulation.value,
            "symbol_rate": self.symbol_rate,
            "fec_type": self.fec_type,
            "interleaver_type": self.interleaver_type,
            "frame_length": self.frame_length,
            "sync_word": self.sync_word,
            "crc_type": self.crc_type,
            "confidence": round(self.confidence, 4),
            "badge": self.badge.value,
            "has_hard_validation": self.has_hard_validation,
            "evm_db": self.evm_db,
            "ber_estimate": self.ber_estimate,
            "evidence": [
                {
                    "source": e.source,
                    "score": round(e.score, 4),
                    "weight": e.weight,
                    "detail": e.detail,
                }
                for e in self.evidence
            ],
            "warnings": self.warnings,
        }

    def __repr__(self) -> str:
        return (
            f"Hypothesis({self.badge.emoji} {self.description} "
            f"conf={self.confidence:.3f})"
        )


@dataclass
class HypothesisGraph:
    """
    Collection of ranked signal hypotheses with evidence fusion.

    The graph holds all candidate interpretations of the signal,
    ranked by combined confidence, and supports backward propagation
    of validation results.
    """
    hypotheses: List[SignalHypothesis] = field(default_factory=list)
    max_candidates: int = 20

    def add(self, hypothesis: SignalHypothesis) -> None:
        """Add a hypothesis, maintaining rank order."""
        hypothesis.compute_confidence()
        self.hypotheses.append(hypothesis)
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        # Prune to max candidates
        if len(self.hypotheses) > self.max_candidates:
            self.hypotheses = self.hypotheses[: self.max_candidates]

    @property
    def best(self) -> Optional[SignalHypothesis]:
        """Return the highest-confidence hypothesis, or None."""
        return self.hypotheses[0] if self.hypotheses else None

    @property
    def top_n(self) -> List[SignalHypothesis]:
        """Return top candidates (those within 50% of best confidence)."""
        if not self.hypotheses:
            return []
        threshold = self.hypotheses[0].confidence * 0.5
        return [h for h in self.hypotheses if h.confidence >= threshold]

    def promote_validated(self) -> None:
        """Promote hypotheses that have hard validation (CRC, syndrome)."""
        for h in self.hypotheses:
            if h.has_hard_validation:
                h.badge = ConfidenceBadge.GREEN
        self.hypotheses.sort(key=lambda h: (h.has_hard_validation, h.confidence), reverse=True)

    def summary(self) -> str:
        """Return a human-readable ranking summary."""
        lines = ["Hypothesis Ranking:"]
        for i, h in enumerate(self.hypotheses):
            lines.append(f"  {i+1}. {h.badge.emoji} {h.description}  {h.confidence:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON report."""
        return {
            "count": len(self.hypotheses),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }
