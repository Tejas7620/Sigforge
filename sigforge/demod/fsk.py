"""
FSK Demodulators.

Implements demodulation for 2-FSK, GFSK, and MSK.
Uses frequency discrimination (instantaneous frequency) for bit recovery.
"""
import numpy as np
from typing import Optional

from sigforge.pipeline.data_types import DemodResult
from sigforge.classify.modulation_types import ModulationType


class FSKDemodulator:
    """Demodulator for Frequency Shift Keying schemes."""

    def __init__(self, mod_type: ModulationType, freq_deviation: Optional[float] = None):
        """
        Args:
            mod_type: Must be FSK2, GFSK, or MSK.
            freq_deviation: Estimated frequency deviation in Hz. If None,
                            the demodulator will auto-estimate from the signal.
        """
        self.mod_type = mod_type
        self.freq_deviation = freq_deviation
        if mod_type not in [ModulationType.FSK2, ModulationType.GFSK, ModulationType.MSK]:
            raise ValueError(f"Unsupported modulation for FSKDemodulator: {mod_type}")

    def _estimate_freq_deviation(self, inst_freq: np.ndarray) -> float:
        """Estimate frequency deviation from instantaneous frequency histogram."""
        # Use histogram to find the two dominant frequency peaks
        hist, bin_edges = np.histogram(inst_freq, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram
        from scipy.ndimage import uniform_filter1d
        hist_smooth = uniform_filter1d(hist.astype(float), size=5)

        # Find peaks (two main clusters for binary FSK)
        mid = len(hist_smooth) // 2
        peak_low_idx = np.argmax(hist_smooth[:mid])
        peak_high_idx = mid + np.argmax(hist_smooth[mid:])

        freq_low = bin_centers[peak_low_idx]
        freq_high = bin_centers[peak_high_idx]

        return abs(freq_high - freq_low) / 2.0

    def demodulate(self, symbols: np.ndarray) -> DemodResult:
        """
        Demodulate FSK signal using frequency discrimination.

        For FSK, 'symbols' here are actually IQ samples at 1 sample/symbol
        (i.e., already timing-recovered). We compute the instantaneous
        frequency and threshold it.
        """
        if len(symbols) < 2:
            return DemodResult()

        # Compute instantaneous frequency via phase derivative
        # diff(angle(x[n])) gives the instantaneous frequency
        phase = np.unwrap(np.angle(symbols))
        inst_freq = np.diff(phase)

        # Auto-estimate deviation if not provided
        if self.freq_deviation is None:
            self.freq_deviation = self._estimate_freq_deviation(inst_freq)

        # Threshold at zero (after removing DC offset from inst_freq)
        dc_offset = np.mean(inst_freq)
        inst_freq_centered = inst_freq - dc_offset

        # Binary decision: positive freq → bit 1, negative freq → bit 0
        hard_bits = (inst_freq_centered > 0).astype(np.uint8)

        # Compute a quality metric analogous to EVM for FSK:
        # Use the "modulation index" clarity — how well-separated the two clusters are
        bits_1_mask = hard_bits == 1
        bits_0_mask = hard_bits == 0

        if np.any(bits_1_mask) and np.any(bits_0_mask):
            mean_high = np.mean(inst_freq_centered[bits_1_mask])
            mean_low = np.mean(inst_freq_centered[bits_0_mask])
            std_high = np.std(inst_freq_centered[bits_1_mask])
            std_low = np.std(inst_freq_centered[bits_0_mask])

            # Signal-to-noise ratio of frequency discrimination
            separation = abs(mean_high - mean_low)
            noise = (std_high + std_low) / 2.0 + 1e-12
            fsk_snr = separation / noise
            evm_db = -20 * np.log10(fsk_snr) if fsk_snr > 0 else 0.0
        else:
            evm_db = 0.0

        # For constellation scatter, plot I vs inst_freq (not standard, but useful)
        scatter_len = min(len(symbols) - 1, 1000)
        # Create a pseudo-constellation: real part vs instantaneous frequency
        scatter = inst_freq_centered[:scatter_len] + 1j * np.real(symbols[:scatter_len])

        return DemodResult(
            symbols=symbols,
            hard_bits=hard_bits,
            evm_db=evm_db,
            constellation_scatter=scatter,
        )
