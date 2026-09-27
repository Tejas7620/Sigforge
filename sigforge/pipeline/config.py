"""
Pipeline configuration — default parameters and search bounds.

All tunable constants in one place. Loaded at startup; overridable via
config YAML or GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class PreprocessConfig:
    """Preprocessing stage configuration."""
    dc_removal_block_size: int = 4096              # samples per block for DC removal
    agc_target_rms: float = 1.0                    # target RMS amplitude after AGC
    agc_window_size: int = 1024                    # AGC smoothing window
    iq_correction_enabled: bool = True             # enable IQ imbalance correction
    default_rolloff: float = 0.35                  # RRC filter rolloff factor


@dataclass
class AnalysisConfig:
    """Signal analysis stage configuration."""
    psd_nfft: int = 4096                           # FFT size for PSD
    psd_overlap_frac: float = 0.5                  # PSD segment overlap
    spectrogram_nfft: int = 1024                   # FFT size for spectrogram
    spectrogram_overlap_frac: float = 0.75         # spectrogram overlap
    bandwidth_threshold_db: float = 10.0           # x-dB bandwidth threshold
    cumulant_segment_length: int = 8192            # samples per cumulant segment
    cumulant_num_segments: int = 50                # number of segments to average


@dataclass
class SymbolRateConfig:
    """Symbol rate estimation configuration."""
    autocorr_max_sps: float = 100.0                # max samples/symbol to search
    autocorr_min_sps: float = 2.0                  # min samples/symbol
    spectral_method_m_values: List[int] = field(default_factory=lambda: [2, 4, 8])
    confidence_agreement_threshold: float = 0.02   # max relative disagreement for high confidence


@dataclass
class ClassifierConfig:
    """Modulation classifier configuration."""
    num_iq_segments: int = 100                     # number of IQ segments for CNN
    iq_segment_length: int = 128                   # samples per IQ segment
    cnn_model_path: str = "models/cnn_modclass_v1.pt"
    classical_weight: float = 0.5                  # weight of classical classifier in fusion
    ml_weight: float = 0.5                         # weight of ML classifier in fusion
    top_n_candidates: int = 3                      # max modulation hypotheses to forward


@dataclass
class DemodConfig:
    """Demodulation configuration."""
    rrc_span_symbols: int = 10                     # RRC filter span in symbols
    rrc_rolloff: float = 0.35                      # RRC filter rolloff
    costas_loop_bw: float = 0.01                   # Costas loop bandwidth (normalized)
    gardner_mu: float = 0.01                       # Gardner TED step size
    mueller_muller_mu: float = 0.01                # M&M TED step size
    pll_bandwidth: float = 0.01                    # PLL loop bandwidth
    cma_mu: float = 0.001                          # CMA step size
    dd_lms_mu: float = 0.0005                      # DD-LMS step size
    cma_warmup_symbols: int = 500                  # CMA warmup before switching to DD-LMS


@dataclass
class InterleaverSearchConfig:
    """Interleaver search configuration."""
    block_row_candidates: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24, 32, 48, 64]
    )
    block_col_candidates: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24, 32, 48, 64]
    )
    conv_b_candidates: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 32]
    )
    conv_m_candidates: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 17, 34]
    )
    max_il_candidates: int = 5                     # top candidates to forward
    score_threshold: float = 0.1                   # minimum score to consider


@dataclass
class FECSearchConfig:
    """FEC search configuration."""
    # Convolutional code search
    conv_codes: Dict[str, Dict] = field(default_factory=lambda: {
        "k3_r12": {"K": 3, "generators": [0o7, 0o5]},
        "k5_r12": {"K": 5, "generators": [0o23, 0o35]},
        "k7_r12": {"K": 7, "generators": [0o171, 0o133]},
        "k9_r12": {"K": 9, "generators": [0o753, 0o561]},
    })
    puncture_patterns: Dict[str, List[int]] = field(default_factory=lambda: {
        "r23": [1, 1, 0, 1],
        "r34": [1, 1, 0, 1, 1, 0],
    })
    # Reed-Solomon search
    rs_codes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (255, 223),
        (255, 239),
        (204, 188),
    ])
    max_fec_candidates: int = 3


@dataclass
class FrameConfig:
    """Frame detection configuration."""
    autocorr_max_frame_bits: int = 65536           # max frame length to search
    sync_min_length: int = 8                       # min sync word length (bits)
    sync_max_length: int = 32                      # max sync word length
    sync_hamming_tolerance: int = 2                # max Hamming distance for sync match
    crc_polynomials: Dict[str, int] = field(default_factory=lambda: {
        "CRC-8": 0x107,
        "CRC-16-CCITT": 0x11021,
        "CRC-16-IBM": 0x18005,
        "CRC-32": 0x104C11DB7,
    })
    entropy_window_bits: int = 32                  # sliding window for entropy analysis


@dataclass
class PipelineConfig:
    """Master configuration combining all stage configs."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    symbol_rate: SymbolRateConfig = field(default_factory=SymbolRateConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    demod: DemodConfig = field(default_factory=DemodConfig)
    interleaver: InterleaverSearchConfig = field(default_factory=InterleaverSearchConfig)
    fec: FECSearchConfig = field(default_factory=FECSearchConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)

    # Global settings
    max_hypothesis_branches: int = 45              # N_mod × M_il × P_fec limit
    prune_confidence_threshold: float = 0.10       # drop branches below this
    max_signal_duration_sec: float = 60.0          # warn if signal longer than this
    chunk_size_samples: int = 1_000_000            # process in chunks for memory
