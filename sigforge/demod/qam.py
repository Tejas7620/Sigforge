"""
QAM Demodulators.

Implements demodulation for 16-QAM and 64-QAM.
Uses nearest-neighbor constellation decoding with Gray-coded bit mapping.
"""
import numpy as np
from typing import Optional

from sigforge.pipeline.data_types import DemodResult
from sigforge.classify.modulation_types import ModulationType, get_constellation, gray_encode


class QAMDemodulator:
    """Demodulator for Quadrature Amplitude Modulation schemes."""

    def __init__(self, mod_type: ModulationType):
        self.mod_type = mod_type
        if mod_type not in [ModulationType.QAM16, ModulationType.QAM64]:
            raise ValueError(f"Unsupported modulation for QAMDemodulator: {mod_type}")

        self.constellation = get_constellation(mod_type)
        self.bps = mod_type.bits_per_symbol
        self.M = mod_type.order

        # Build Gray-coded bit map
        self._build_bit_map()

    def _build_bit_map(self):
        """Build the mapping from constellation index to Gray-coded bits."""
        if self.mod_type == ModulationType.QAM16:
            # 4x4 grid, each axis has 4 levels with Gray coding: 0->00, 1->01, 3->11, 2->10
            gray_2bit = [0, 1, 3, 2]  # Gray code for 2 bits
            self.bit_table = np.zeros((self.M, self.bps), dtype=np.uint8)
            idx = 0
            for i, gi in enumerate(gray_2bit):
                for q, gq in enumerate(gray_2bit):
                    # i-axis bits (MSB pair), q-axis bits (LSB pair)
                    self.bit_table[idx, 0] = (gi >> 1) & 1
                    self.bit_table[idx, 1] = gi & 1
                    self.bit_table[idx, 2] = (gq >> 1) & 1
                    self.bit_table[idx, 3] = gq & 1
                    idx += 1

        elif self.mod_type == ModulationType.QAM64:
            # 8x8 grid, 3-bit Gray per axis
            gray_3bit = [0, 1, 3, 2, 6, 7, 5, 4]  # Gray code for 3 bits
            self.bit_table = np.zeros((self.M, self.bps), dtype=np.uint8)
            idx = 0
            for i, gi in enumerate(gray_3bit):
                for q, gq in enumerate(gray_3bit):
                    self.bit_table[idx, 0] = (gi >> 2) & 1
                    self.bit_table[idx, 1] = (gi >> 1) & 1
                    self.bit_table[idx, 2] = gi & 1
                    self.bit_table[idx, 3] = (gq >> 2) & 1
                    self.bit_table[idx, 4] = (gq >> 1) & 1
                    self.bit_table[idx, 5] = gq & 1
                    idx += 1

    def demodulate(self, symbols: np.ndarray) -> DemodResult:
        """Demodulate complex symbols using nearest-neighbor decoding."""
        if len(symbols) == 0:
            return DemodResult()

        # Nearest-neighbor decoding: find closest constellation point for each symbol
        # Shape: (num_symbols, 1) vs (1, M) -> (num_symbols, M) distance matrix
        distances = np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :])
        nearest_idx = np.argmin(distances, axis=1)

        # Map indices to bits
        hard_bits = self.bit_table[nearest_idx].flatten()

        # Ideal symbols for EVM calculation
        ideal_syms = self.constellation[nearest_idx]

        # Compute EVM
        error_vector = symbols - ideal_syms
        mse = np.mean(np.abs(error_vector) ** 2)
        mean_power = np.mean(np.abs(ideal_syms) ** 2)

        if mean_power > 0 and mse > 0:
            evm_rms = np.sqrt(mse / mean_power)
            evm_db = 20 * np.log10(evm_rms)
        else:
            evm_db = float('-inf')

        return DemodResult(
            symbols=symbols,
            hard_bits=hard_bits,
            evm_db=evm_db,
            constellation_scatter=symbols[:1000] if len(symbols) > 1000 else symbols,
        )
