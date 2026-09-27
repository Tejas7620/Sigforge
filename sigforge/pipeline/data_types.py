"""
Signal Profile and Analysis Results — shared data structures.

These dataclasses carry intermediate analysis results between pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SignalProfile:
    """
    Output of the signal analysis stage — spectral, statistical, and
    cumulant features computed from the cleaned IQ data.
    """
    # Spectral
    psd_freqs: Optional[np.ndarray] = None        # frequency axis (Hz)
    psd_power: Optional[np.ndarray] = None         # power spectral density (dB)
    spectrogram: Optional[np.ndarray] = None       # 2D time-frequency (power)
    spectrogram_times: Optional[np.ndarray] = None # time axis for spectrogram
    spectrogram_freqs: Optional[np.ndarray] = None # freq axis for spectrogram

    # Bandwidth and SNR
    occupied_bandwidth: Optional[float] = None     # Hz
    snr_db: Optional[float] = None                 # estimated SNR in dB
    noise_floor_db: Optional[float] = None         # noise floor in dB

    # Carrier
    center_freq_offset: Optional[float] = None     # Hz, estimated CFO

    # Higher-order cumulants
    cumulants: Dict[str, complex] = field(default_factory=dict)
    # Keys: "C20", "C21", "C40", "C41", "C42"

    # Instantaneous statistics
    inst_amplitude_std: Optional[float] = None
    inst_phase_std: Optional[float] = None
    inst_freq_std: Optional[float] = None
    kurtosis: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize for JSON (excludes large arrays)."""
        return {
            "occupied_bandwidth": self.occupied_bandwidth,
            "snr_db": self.snr_db,
            "noise_floor_db": self.noise_floor_db,
            "center_freq_offset": self.center_freq_offset,
            "cumulants": {k: [v.real, v.imag] for k, v in self.cumulants.items()},
            "inst_amplitude_std": self.inst_amplitude_std,
            "inst_phase_std": self.inst_phase_std,
            "inst_freq_std": self.inst_freq_std,
            "kurtosis": self.kurtosis,
        }


@dataclass
class ParameterEstimates:
    """Output of the parameter estimation stage."""
    symbol_rate: Optional[float] = None            # Hz
    symbol_rate_confidence: float = 0.0            # 0.0-1.0
    samples_per_symbol: Optional[float] = None     # Fs / symbol_rate
    timing_offset: Optional[float] = None          # fractional sample offset
    cfo_hz: Optional[float] = None                 # carrier frequency offset (Hz)

    # Individual method results for fusion
    symbol_rate_autocorr: Optional[float] = None
    symbol_rate_spectral: Optional[float] = None
    symbol_rate_cyclo: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "symbol_rate": self.symbol_rate,
            "symbol_rate_confidence": self.symbol_rate_confidence,
            "samples_per_symbol": self.samples_per_symbol,
            "timing_offset": self.timing_offset,
            "cfo_hz": self.cfo_hz,
        }


@dataclass
class DemodResult:
    """Output of the demodulation stage."""
    symbols: Optional[np.ndarray] = None           # complex symbol stream
    hard_bits: Optional[np.ndarray] = None         # uint8 bit stream (hard decision)
    soft_bits: Optional[np.ndarray] = None         # float32 LLRs (for soft Viterbi)
    evm_db: Optional[float] = None                 # Error Vector Magnitude (dB)
    carrier_lock_quality: float = 0.0              # 0.0-1.0
    timing_lock_quality: float = 0.0               # 0.0-1.0
    phase_error_variance: Optional[float] = None   # rad^2
    constellation_scatter: Optional[np.ndarray] = None  # raw symbol samples for plotting

    def to_dict(self) -> dict:
        return {
            "num_symbols": len(self.symbols) if self.symbols is not None else 0,
            "num_bits": len(self.hard_bits) if self.hard_bits is not None else 0,
            "evm_db": self.evm_db,
            "carrier_lock_quality": self.carrier_lock_quality,
            "timing_lock_quality": self.timing_lock_quality,
        }


@dataclass
class InterleaverCandidate:
    """A candidate interleaver detected by the search engine."""
    il_type: str                                   # "block", "convolutional", "diagonal", "none"
    params: Dict[str, Any] = field(default_factory=dict)
    deinterleaved_bits: Optional[np.ndarray] = None
    score: float = 0.0                             # quality score from FEC-assisted validation

    def __repr__(self) -> str:
        return f"IL({self.il_type}, {self.params}, score={self.score:.3f})"


@dataclass
class FECResult:
    """Result of FEC decoding attempt."""
    codec_type: str                                # "convolutional", "reed_solomon", "concatenated", "ldpc"
    codec_id: str                                  # e.g., "k7_r12", "RS(255,223)"
    rate: Optional[str] = None                     # e.g., "r12", "r34"
    decoded_bits: Optional[np.ndarray] = None
    syndrome_zero: bool = False
    re_encode_match: float = 0.0                   # 0.0-1.0 fraction matching
    ber_improvement: float = 0.0                   # ratio of BER_before / BER_after
    errors_corrected: int = 0
    confidence: float = 0.0                        # combined FEC confidence

    def __repr__(self) -> str:
        valid = "✓" if self.syndrome_zero else "✗"
        return f"FEC({self.codec_type}/{self.codec_id} {valid} conf={self.confidence:.3f})"


@dataclass
class FrameAnalysis:
    """Result of bitstream/frame analysis."""
    sync_word: Optional[str] = None                # hex string of detected sync word
    sync_positions: List[int] = field(default_factory=list)  # bit positions
    frame_length: Optional[int] = None             # bits
    frame_count: int = 0
    header_range: Optional[Tuple[int, int]] = None # (start_bit, end_bit) within frame
    payload_range: Optional[Tuple[int, int]] = None
    crc_type: Optional[str] = None                 # detected CRC polynomial
    crc_position: Optional[Tuple[int, int]] = None # bit range of CRC field
    crc_valid_count: int = 0                       # number of frames with valid CRC
    entropy_profile: Optional[np.ndarray] = None   # per-bit-position entropy
    counters_detected: List[Dict[str, Any]] = field(default_factory=list)  # incrementing fields

    @property
    def crc_pass_rate(self) -> float:
        """Fraction of frames with valid CRC."""
        if self.frame_count == 0:
            return 0.0
        return self.crc_valid_count / self.frame_count

    def to_dict(self) -> dict:
        return {
            "sync_word": self.sync_word,
            "frame_length": self.frame_length,
            "frame_count": self.frame_count,
            "crc_type": self.crc_type,
            "crc_pass_rate": self.crc_pass_rate,
        }


@dataclass
class PipelineResult:
    """Complete result of the analysis pipeline — passed to the GUI."""
    # Raw data
    iq_data: Optional[np.ndarray] = None
    sample_rate: Optional[float] = None

    # Stage results
    signal_profile: Optional[SignalProfile] = None
    parameter_estimates: Optional[ParameterEstimates] = None
    demod_results: List[DemodResult] = field(default_factory=list)
    interleaver_candidates: List[InterleaverCandidate] = field(default_factory=list)
    fec_results: List[FECResult] = field(default_factory=list)
    frame_analysis: Optional[FrameAnalysis] = None

    # Hypothesis
    hypothesis_graph: Optional[Any] = None  # HypothesisGraph (avoid circular import)

    # Metadata
    file_path: str = ""
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
