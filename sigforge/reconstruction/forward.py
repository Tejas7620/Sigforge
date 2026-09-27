"""
Forward RF Reconstruction Engine.

This module is responsible for the core differentiator of SigForge:
taking recovered bits, a hypothesis, and estimated parameters, and reconstructing
the expected RF IQ waveform to be compared against the observed signal.
"""
import numpy as np
from scipy import signal
from typing import Optional

from sigforge.classify.modulation_types import ModulationType


def rrcosfilter(N: int, alpha: float, Ts: float, Fs: float) -> np.ndarray:
    """
    Generates a Root Raised Cosine (RRC) filter (FIR).
    
    Args:
        N: Length of the filter (samples).
        alpha: Roll-off factor.
        Ts: Symbol period (seconds).
        Fs: Sampling frequency (Hz).
        
    Returns:
        np.ndarray: The RRC filter coefficients.
    """
    T_delta = 1 / float(Fs)
    time_idx = ((np.arange(N) - N / 2)) * T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                    (np.sin(np.pi / (4 * alpha)))) + ((1 - 2 / np.pi) * \
                    (np.cos(np.pi / (4 * alpha)))))
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                    (np.sin(np.pi / (4 * alpha)))) + ((1 - 2 / np.pi) * \
                    (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) + \
                    4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                    (np.pi * t * (1 - (4 * alpha * t / Ts) ** 2) / Ts)

    # Normalize filter energy
    energy = np.sqrt(np.sum(h_rrc ** 2))
    if energy > 0:
        h_rrc = h_rrc / energy
    return h_rrc


class ForwardReconstructor:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def modulate(self, bits: np.ndarray, mod_type: ModulationType) -> np.ndarray:
        """Map bits to complex symbols."""
        bits = np.asarray(bits, dtype=np.float64)

        if mod_type == ModulationType.BPSK:
            # 0 -> -1, 1 -> +1
            return 2.0 * bits - 1.0 + 0j
        elif mod_type == ModulationType.QPSK:
            # Group bits by 2
            if len(bits) % 2 != 0:
                bits = np.pad(bits, (0, 1))
            syms = (bits[0::2] * 2.0 - 1.0) + 1j * (bits[1::2] * 2.0 - 1.0)
            return syms / np.sqrt(2)
        elif mod_type == ModulationType.OQPSK:
            # Group bits by 2
            if len(bits) % 2 != 0:
                bits = np.pad(bits, (0, 1))
            syms = (bits[0::2] * 2.0 - 1.0) + 1j * (bits[1::2] * 2.0 - 1.0)
            return syms / np.sqrt(2)
        elif mod_type == ModulationType.PSK8:
            # Simple 8PSK mapping
            if len(bits) % 3 != 0:
                bits = np.pad(bits, (0, 3 - len(bits) % 3))
            vals = bits[0::3] * 4.0 + bits[1::3] * 2.0 + bits[2::3]
            phases = vals * (np.pi / 4.0)
            return np.exp(1j * phases)
        elif mod_type == ModulationType.QAM16:
            # 16-QAM: 4 bits per symbol, map to 4x4 grid
            if len(bits) % 4 != 0:
                bits = np.pad(bits, (0, 4 - len(bits) % 4))
            levels = np.array([-3, -1, 1, 3], dtype=np.float64)
            i_idx = (bits[0::4] * 2 + bits[1::4]).astype(int)
            q_idx = (bits[2::4] * 2 + bits[3::4]).astype(int)
            syms = levels[i_idx] + 1j * levels[q_idx]
            rms = np.sqrt(np.mean(np.abs(syms) ** 2)) + 1e-12
            return syms / rms
        elif mod_type == ModulationType.QAM64:
            # 64-QAM: 6 bits per symbol, map to 8x8 grid
            if len(bits) % 6 != 0:
                bits = np.pad(bits, (0, 6 - len(bits) % 6))
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
            i_idx = (bits[0::6] * 4 + bits[1::6] * 2 + bits[2::6]).astype(int)
            q_idx = (bits[3::6] * 4 + bits[4::6] * 2 + bits[5::6]).astype(int)
            syms = levels[i_idx] + 1j * levels[q_idx]
            rms = np.sqrt(np.mean(np.abs(syms) ** 2)) + 1e-12
            return syms / rms
        elif mod_type in (ModulationType.FSK2, ModulationType.GFSK, ModulationType.MSK):
            # FSK family: bits -> +/- frequency deviation as phase steps
            freq_dev = 0.25  # normalized to symbol rate
            phases = np.cumsum((2.0 * bits - 1.0) * np.pi * freq_dev)
            return np.exp(1j * phases)
        else:
            raise NotImplementedError(f"Modulation {mod_type.value} reconstruction not implemented.")

    def reconstruct(self, 
                    bits: np.ndarray, 
                    mod_type: ModulationType, 
                    symbol_rate: float, 
                    cfo_hz: float = 0.0,
                    phase_offset: float = 0.0) -> np.ndarray:
        """
        Reconstruct the expected IQ waveform from bits.
        
        Flow:
        Bits -> Symbols -> Pulse Shaping -> Carrier Frequency Offset -> Output
        """
        # 1. Modulate
        symbols = self.modulate(bits, mod_type)
        
        # 2. Pulse shaping (RRC)
        sps = int(self.sample_rate / symbol_rate)
        if sps < 1:
            sps = 1
        
        # Create RRC filter
        # Alpha = 0.35 is typical, N = 11 symbols
        rrc_filter = rrcosfilter(11 * sps, 0.35, 1.0/symbol_rate, self.sample_rate)
        
        # Upsample and filter
        upsampled = np.zeros(len(symbols) * sps, dtype=complex)
        upsampled[::sps] = symbols
        baseband_iq = signal.lfilter(rrc_filter, 1.0, upsampled)
        
        # 3. Apply CFO and Phase offset
        t = np.arange(len(baseband_iq)) / self.sample_rate
        carrier = np.exp(1j * (2 * np.pi * cfo_hz * t + phase_offset))
        
        reconstructed_iq = baseband_iq * carrier
        
        return reconstructed_iq
