"""
Modulation type definitions and constellation maps.

Central registry of all modulation types SigForge can identify and demodulate.
Each modulation type has an ideal constellation, bits-per-symbol, and modulation family.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class ModulationFamily(enum.Enum):
    """Broad modulation family for routing to the correct demodulator."""
    FSK = "FSK"
    PSK = "PSK"
    QAM = "QAM"
    UNKNOWN = "UNKNOWN"


class ModulationType(enum.Enum):
    """Specific modulation scheme identifiers."""
    FSK2 = "2-FSK"
    GFSK = "GFSK"
    MSK = "MSK"
    BPSK = "BPSK"
    QPSK = "QPSK"
    OQPSK = "OQPSK"
    PSK8 = "8-PSK"
    QAM16 = "16-QAM"
    QAM64 = "64-QAM"
    UNKNOWN = "UNKNOWN"

    @property
    def family(self) -> ModulationFamily:
        """Return the modulation family for demodulator routing."""
        _family_map = {
            ModulationType.FSK2: ModulationFamily.FSK,
            ModulationType.GFSK: ModulationFamily.FSK,
            ModulationType.MSK: ModulationFamily.FSK,
            ModulationType.BPSK: ModulationFamily.PSK,
            ModulationType.QPSK: ModulationFamily.PSK,
            ModulationType.OQPSK: ModulationFamily.PSK,
            ModulationType.PSK8: ModulationFamily.PSK,
            ModulationType.QAM16: ModulationFamily.QAM,
            ModulationType.QAM64: ModulationFamily.QAM,
            ModulationType.UNKNOWN: ModulationFamily.UNKNOWN,
        }
        return _family_map[self]

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits carried per symbol."""
        _bps_map = {
            ModulationType.FSK2: 1,
            ModulationType.GFSK: 1,
            ModulationType.MSK: 1,
            ModulationType.BPSK: 1,
            ModulationType.QPSK: 2,
            ModulationType.OQPSK: 2,
            ModulationType.PSK8: 3,
            ModulationType.QAM16: 4,
            ModulationType.QAM64: 6,
            ModulationType.UNKNOWN: 0,
        }
        return _bps_map[self]

    @property
    def order(self) -> int:
        """Modulation order M (number of constellation points)."""
        return 2 ** self.bits_per_symbol if self.bits_per_symbol > 0 else 0

    @property
    def mth_power(self) -> int:
        """Exponent M for M-th power CFO estimation."""
        _m_map = {
            ModulationType.FSK2: 2,
            ModulationType.GFSK: 2,
            ModulationType.MSK: 2,
            ModulationType.BPSK: 2,
            ModulationType.QPSK: 4,
            ModulationType.OQPSK: 4,
            ModulationType.PSK8: 8,
            ModulationType.QAM16: 4,   # not ideal for QAM, but usable
            ModulationType.QAM64: 4,
            ModulationType.UNKNOWN: 4,
        }
        return _m_map[self]


def get_constellation(mod_type: ModulationType) -> Optional[np.ndarray]:
    """
    Return the ideal constellation points (complex64 array) for a modulation type.

    Points are unit-energy normalized. Returns None for FSK/UNKNOWN types
    (FSK doesn't have a meaningful complex constellation).
    """
    if mod_type == ModulationType.BPSK:
        return np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)

    elif mod_type in (ModulationType.QPSK, ModulationType.OQPSK):
        # Gray-coded QPSK: 00→+1+1j, 01→-1+1j, 11→-1-1j, 10→+1-1j
        pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64)
        return pts / np.sqrt(2)  # unit energy

    elif mod_type == ModulationType.PSK8:
        # 8-PSK equally spaced on unit circle
        angles = np.arange(8) * (2 * np.pi / 8)
        return np.exp(1j * angles).astype(np.complex64)

    elif mod_type == ModulationType.QAM16:
        # 16-QAM: 4×4 grid, Gray-coded
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        grid = np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex64)
        rms = np.sqrt(np.mean(np.abs(grid) ** 2))
        return grid / rms  # unit energy

    elif mod_type == ModulationType.QAM64:
        # 64-QAM: 8×8 grid, Gray-coded
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
        grid = np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex64)
        rms = np.sqrt(np.mean(np.abs(grid) ** 2))
        return grid / rms

    else:
        return None  # FSK, UNKNOWN


# ---------------------------------------------------------------------------
# Gray code tables for bit mapping
# ---------------------------------------------------------------------------

def gray_encode(n: int) -> int:
    """Convert binary integer to Gray code."""
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    """Convert Gray code to binary integer."""
    n = g
    mask = g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def get_gray_map(mod_type: ModulationType) -> Optional[np.ndarray]:
    """
    Return Gray code mapping table: index = Gray code, value = natural binary.

    For a modulation with M=2^k points, returns an array of length M where
    gray_map[gray_index] = natural_index.
    """
    M = mod_type.order
    if M == 0:
        return None
    return np.array([gray_decode(i) for i in range(M)], dtype=np.int32)


@dataclass
class ModulationHypothesis:
    """A single modulation classification hypothesis with confidence."""
    mod_type: ModulationType
    confidence: float  # 0.0 to 1.0
    features: dict = field(default_factory=dict)  # supporting feature values
    source: str = "unknown"  # "classical", "cnn", "fused"

    def __repr__(self) -> str:
        return f"ModHyp({self.mod_type.value}, conf={self.confidence:.3f}, src={self.source})"
