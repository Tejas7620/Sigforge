"""
Modulation Classifier.

Uses higher-order cumulants (C20, C40, C42), envelope statistics,
and spectral features to classify unknown signals into modulation types.
Supports: BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM, 2-FSK, GFSK, MSK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from sigforge.classify.modulation_types import ModulationType, ModulationFamily


@dataclass
class ClassificationResult:
    """Result of modulation classification."""
    predicted_type: ModulationType
    confidence: float
    ranked_candidates: List[Tuple[ModulationType, float]] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)


class ModulationClassifier:
    """
    Blind modulation classifier using higher-order statistics (cumulants)
    and envelope features.
    """

    def __init__(self):
        pass

    def extract_features(self, iq_data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Extract statistical and spectral features from IQ samples."""
        if len(iq_data) == 0:
            return {}

        # Remove DC
        x = iq_data - np.mean(iq_data)
        
        # Energy normalize
        p_avg = np.mean(np.abs(x) ** 2)
        if p_avg < 1e-15:
            return {}
        x_norm = x / np.sqrt(p_avg)

        # Segment for reliable statistics
        N = min(len(x_norm), 16384)
        x_seg = x_norm[:N]

        # Moments
        m20 = np.mean(x_seg ** 2)
        m21 = np.mean(np.abs(x_seg) ** 2)  # should be ~1
        m40 = np.mean(x_seg ** 4)
        m42 = np.mean((np.abs(x_seg) ** 2) * (x_seg ** 2))
        m_abs4 = np.mean(np.abs(x_seg) ** 4)

        # Cumulants
        c20 = m20
        c40 = m40 - 3.0 * (m20 ** 2)
        c42 = m_abs4 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)

        # Envelope statistics
        amp = np.abs(x_seg)
        gamma_max = np.max(np.abs(np.fft.fft(x_seg - np.mean(x_seg)))) ** 2 / (N * p_avg + 1e-12)
        sigma_aa = float(np.std(amp) / (np.mean(amp) + 1e-12))
        kurtosis = float(np.mean(amp ** 4) / ((np.mean(amp ** 2) + 1e-12) ** 2) - 2.0)

        # Phase / Frequency statistics
        phase = np.unwrap(np.angle(x_seg))
        freq = np.diff(phase)
        sigma_af = float(np.std(freq))

        # Check for FSK spectral peaks
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft(x_seg, n=2048)))
        fft_db = 20 * np.log10(fft_mag + 1e-12)
        peak_diff = float(np.max(fft_db) - np.median(fft_db))

        features = {
            "abs_c20": float(np.abs(c20)),
            "abs_c40": float(np.abs(c40)),
            "abs_c42": float(np.abs(c42)),
            "c42_real": float(np.real(c42)),
            "sigma_aa": sigma_aa,       # normalized envelope variance
            "kurtosis": kurtosis,
            "sigma_af": sigma_af,       # frequency dispersion
            "gamma_max": float(gamma_max),
            "peak_diff": peak_diff,
        }
        return features

    def classify(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        sps: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classify modulation type and return ranked candidates with confidence scores.
        """
        if sps is not None and sps >= 1.5:
            from sigforge.synchronize.sync import SymbolSynchronizer
            sync = SymbolSynchronizer(int(round(sps)))
            analysis_data = sync.synchronize(iq_data)
        else:
            analysis_data = iq_data

        features = self.extract_features(analysis_data, sample_rate)
        if not features:
            return ClassificationResult(
                predicted_type=ModulationType.UNKNOWN,
                confidence=0.0,
                ranked_candidates=[(ModulationType.UNKNOWN, 0.0)],
                features={},
            )

        scores: Dict[ModulationType, float] = {}

        abs_c20 = features["abs_c20"]
        abs_c40 = features["abs_c40"]
        c42_val = features["c42_real"]
        sigma_aa = features["sigma_aa"]
        kurtosis = features["kurtosis"]

        # --- Decision metrics ---
        # 1. BPSK: |C20| close to 1.0, |C40| close to 2.0, C42 ~ -2.0
        # Distance to ideal (|C20|=1.0, |C40|=2.0)
        dist_bpsk = np.sqrt((abs_c20 - 1.0) ** 2 + (abs_c40 - 2.0) ** 2 * 0.25)
        scores[ModulationType.BPSK] = float(np.exp(-dist_bpsk * 3.0))

        # 2. QPSK: |C20| close to 0, |C40| close to 1.0, C42 ~ -1.0, envelope nearly constant
        dist_qpsk = np.sqrt(abs_c20 ** 2 * 4.0 + (abs_c40 - 1.0) ** 2 * 2.0 + (c42_val - (-1.0)) ** 2)
        scores[ModulationType.QPSK] = float(np.exp(-dist_qpsk * 2.5))

        # 3. 8-PSK: |C20| close to 0, |C40| close to 0, C42 ~ -1.0, envelope constant
        dist_psk8 = np.sqrt(abs_c20 ** 2 * 4.0 + abs_c40 ** 2 * 3.0 + (c42_val - (-1.0)) ** 2)
        scores[ModulationType.PSK8] = float(np.exp(-dist_psk8 * 2.5))

        # 4. 16-QAM: |C20| close to 0, |C40| ~ 0.68, C42 ~ -0.68, envelope has higher variance (sigma_aa > 0.25)
        dist_qam16 = np.sqrt(
            abs_c20 ** 2 * 3.0 +
            (abs_c40 - 0.68) ** 2 * 2.0 +
            (c42_val - (-0.68)) ** 2 * 2.0 +
            (max(0.0, 0.20 - sigma_aa) * 5.0) ** 2
        )
        scores[ModulationType.QAM16] = float(np.exp(-dist_qam16 * 2.0))

        # 5. 64-QAM: |C20| close to 0, |C40| ~ 0.62, C42 ~ -0.62
        dist_qam64 = np.sqrt(
            abs_c20 ** 2 * 3.0 +
            (abs_c40 - 0.62) ** 2 * 2.0 +
            (c42_val - (-0.62)) ** 2 * 2.0 +
            (max(0.0, 0.22 - sigma_aa) * 5.0) ** 2
        )
        scores[ModulationType.QAM64] = float(np.exp(-dist_qam64 * 1.8))

        # 6. FSK / GFSK / MSK: constant envelope (low sigma_aa), low |C20|, frequency modulation
        # In FSK, signal is FM, |C20| and |C40| are low, envelope variation is strictly from noise/filtering
        is_fsk_like = sigma_aa < 0.25 and abs_c20 < 0.4 and features["sigma_af"] > 0.1
        fsk_bonus = 1.0 if is_fsk_like else 0.2
        dist_fsk = (sigma_aa * 3.0) + (abs_c20 * 2.0)
        scores[ModulationType.FSK2] = float(np.exp(-dist_fsk * 2.0) * fsk_bonus)
        scores[ModulationType.GFSK] = scores[ModulationType.FSK2] * 0.9
        scores[ModulationType.MSK] = scores[ModulationType.FSK2] * 0.95

        # Normalize scores to pseudo-probabilities
        total = sum(scores.values()) + 1e-12
        ranked = sorted(
            [(mod, score / total) for mod, score in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        top_mod, top_conf = ranked[0]
        return ClassificationResult(
            predicted_type=top_mod,
            confidence=top_conf,
            ranked_candidates=ranked,
            features=features,
        )
