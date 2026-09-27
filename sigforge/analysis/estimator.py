"""
Parameter Estimator Module.

Performs blind estimation of:
  - Carrier Frequency Offset (CFO) via M-th power FFT
  - Symbol Rate (Rs) via envelope power spectrum / cyclic analysis
  - Signal-to-Noise Ratio (SNR) and Noise Floor
  - Occupied Bandwidth
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import scipy.signal


@dataclass
class EstimationResult:
    """Estimated signal parameters with confidence indicators."""
    symbol_rate: float
    samples_per_symbol: float
    cfo_hz: float
    snr_db: float
    noise_floor_db: float
    occupied_bw_hz: float
    symbol_rate_confidence: float = 0.5


class ParameterEstimator:
    """Blind parameter estimator for digital communications signals."""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def estimate_cfo(self, iq_data: np.ndarray, m_order: int = 4) -> float:
        """
        Estimate Carrier Frequency Offset (CFO) using M-th power method.
        For QPSK/QAM, m_order=4 cancels data modulation: (s(t) e^{j 2pi f_c t})^4 -> e^{j 8pi f_c t}.
        """
        N = min(len(iq_data), 16384)
        x = iq_data[:N] - np.mean(iq_data[:N])
        
        # M-th power
        xm = x ** m_order
        nfft = 16384
        fft_res = np.fft.fft(xm, n=nfft)
        freqs = np.fft.fftfreq(nfft, 1.0 / self.sample_rate)

        # Ignore DC
        zero_idx = nfft // 2
        fft_mag = np.abs(fft_res)
        fft_mag[0] = 0

        peak_idx = int(np.argmax(fft_mag))
        est_cfo = float(freqs[peak_idx] / m_order)
        return est_cfo

    def estimate_symbol_rate(
        self,
        iq_data: np.ndarray,
        cfo_corrected: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        Estimate Symbol Rate (Rs) via delay-and-multiply cyclostationary analysis:
        y(n) = x(n) * conj(x(n-1)), whose spectrum contains a discrete peak at Rs.
        
        Returns:
            (symbol_rate_hz, samples_per_symbol, confidence)
        """
        x = cfo_corrected if cfo_corrected is not None else iq_data
        N = min(len(x), 32768)
        if N < 4:
            return float(self.sample_rate / 4.0), 4.0, 0.1
        x_seg = x[:N]

        # Delay-and-multiply cyclostationary line generation
        y = x_seg[1:] * np.conj(x_seg[:-1])
        y = y - np.mean(y)

        nfft = 32768
        Y = np.abs(np.fft.fft(y, n=nfft))
        freqs = np.fft.fftfreq(nfft, 1.0 / self.sample_rate)

        # Search range: between 1 kHz and Fs / 2
        min_freq = 1000.0
        max_freq = self.sample_rate / 2.0
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)

        valid_fft = Y[valid_mask]
        valid_freqs = freqs[valid_mask]

        if len(valid_fft) > 0:
            median_level = np.median(valid_fft)
            peak_idx = int(np.argmax(valid_fft))
            peak_val = valid_fft[peak_idx]
            peak_freq = valid_freqs[peak_idx]

            prominence = float(peak_val / (median_level + 1e-12))
            if prominence > 3.0:
                sps = self.sample_rate / peak_freq
                if abs(sps - round(sps)) < 0.20 and round(sps) >= 2:
                    sps = float(round(sps))
                    peak_freq = self.sample_rate / sps

                conf = min(1.0, float(prominence / 12.0))
                return float(peak_freq), float(sps), conf

        # Fallback: 4 samples per symbol
        fallback_sps = 4.0
        fallback_rs = self.sample_rate / fallback_sps
        return float(fallback_rs), float(fallback_sps), 0.25

    def estimate_all(
        self,
        iq_data: np.ndarray,
        cfo_order: int = 4,
    ) -> EstimationResult:
        """Run all estimators on IQ data block."""
        # 1. Coarse CFO
        cfo = self.estimate_cfo(iq_data, m_order=cfo_order)

        # 2. De-rotate by coarse CFO for symbol rate estimation
        t = np.arange(len(iq_data)) / self.sample_rate
        cfo_corrected = iq_data * np.exp(-2j * np.pi * cfo * t)

        # 3. Symbol Rate
        sym_rate, sps, sym_conf = self.estimate_symbol_rate(iq_data, cfo_corrected=cfo_corrected)

        # 4. PSD, SNR and Occupied BW
        freqs, psd = scipy.signal.welch(
            iq_data, self.sample_rate,
            nperseg=min(2048, len(iq_data)),
            return_onesided=False
        )
        psd_shift = np.fft.fftshift(psd)
        freqs_shift = np.fft.fftshift(freqs)
        psd_db = 10 * np.log10(psd_shift + 1e-12)

        peak_pwr = float(np.max(psd_db))
        noise_pwr = float(np.percentile(psd_db, 10))
        snr = float(peak_pwr - noise_pwr)

        thresh = peak_pwr - 10.0
        occ_freqs = freqs_shift[psd_db > thresh]
        if len(occ_freqs) > 1:
            bw = float(occ_freqs[-1] - occ_freqs[0])
        else:
            bw = float(sym_rate * 1.35)

        return EstimationResult(
            symbol_rate=sym_rate,
            samples_per_symbol=sps,
            cfo_hz=cfo,
            snr_db=snr,
            noise_floor_db=noise_pwr,
            occupied_bw_hz=bw,
            symbol_rate_confidence=sym_conf,
        )
