"""
PSK Demodulators.

Implements demodulation for BPSK, QPSK, and 8PSK.
Takes synchronized complex symbols and maps them to bits.
"""
import numpy as np
from typing import Tuple

from sigforge.pipeline.data_types import DemodResult
from sigforge.classify.modulation_types import ModulationType


class PSKDemodulator:
    """Demodulator for Phase Shift Keying schemes."""

    def __init__(self, mod_type: ModulationType):
        self.mod_type = mod_type
        if mod_type not in [ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8]:
            raise ValueError(f"Unsupported modulation for PSKDemodulator: {mod_type}")

    def demodulate(self, symbols: np.ndarray) -> DemodResult:
        """Demodulate complex symbols into hard bits."""
        if len(symbols) == 0:
            return DemodResult()

        phases = np.angle(symbols)
        
        if self.mod_type == ModulationType.BPSK:
            # BPSK: >0 is 1, <0 is 0
            hard_bits = (symbols.real > 0).astype(np.uint8)
            ideal_syms = np.where(hard_bits, 1.0, -1.0)
            
        elif self.mod_type == ModulationType.QPSK:
            # QPSK: standard Gray mapping (pi/4, 3pi/4, -3pi/4, -pi/4)
            # Quadrant decoding based on signs of real and imag parts
            bit0 = (symbols.real > 0).astype(np.uint8)
            bit1 = (symbols.imag > 0).astype(np.uint8)
            hard_bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
            hard_bits[0::2] = bit0
            hard_bits[1::2] = bit1
            
            # Ideal constellation points for EVM (convert to float to avoid uint8 underflow)
            ideal_real = (bit0.astype(float) * 2.0 - 1.0) / np.sqrt(2)
            ideal_imag = (bit1.astype(float) * 2.0 - 1.0) / np.sqrt(2)
            ideal_syms = ideal_real + 1j * ideal_imag
            
        elif self.mod_type == ModulationType.PSK8:
            # 8PSK: sector decoding (pi/4 per sector)
            # Shift phases to [0, 2pi)
            phases_norm = np.mod(phases + 2*np.pi, 2*np.pi)
            # Sector index 0 to 7
            sector = np.round(phases_norm / (np.pi / 4)).astype(int) % 8
            
            # Natural mapping for MVP (can be extended to Gray mapping)
            bit0 = (sector & 4) >> 2
            bit1 = (sector & 2) >> 1
            bit2 = (sector & 1)
            
            hard_bits = np.zeros(len(symbols) * 3, dtype=np.uint8)
            hard_bits[0::3] = bit0
            hard_bits[1::3] = bit1
            hard_bits[2::3] = bit2
            
            ideal_syms = np.exp(1j * sector * (np.pi / 4))

        # Energy normalize symbols for true constellation EVM calculation
        sym_rms = np.sqrt(np.mean(np.abs(symbols) ** 2)) + 1e-12
        symbols_norm = symbols / sym_rms
        ideal_rms = np.sqrt(np.mean(np.abs(ideal_syms) ** 2)) + 1e-12
        ideal_norm = ideal_syms / ideal_rms

        # Compute EVM (Error Vector Magnitude)
        error_vector = symbols_norm - ideal_norm
        mse = float(np.mean(np.abs(error_vector) ** 2))
        
        if mse > 1e-12:
            evm_rms = np.sqrt(mse)
            evm_db = float(20.0 * np.log10(evm_rms))
        else:
            evm_db = -60.0

        return DemodResult(
            symbols=symbols,
            hard_bits=hard_bits,
            evm_db=evm_db,
            constellation_scatter=symbols[:1000] if len(symbols) > 1000 else symbols
        )
