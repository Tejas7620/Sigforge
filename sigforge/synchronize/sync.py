"""
Carrier and Timing Synchronization.

Implements carrier frequency/phase recovery (Costas loop)
and symbol timing recovery (minimum envelope variance / eye opening).
"""
import numpy as np


class CostasLoop:
    """Costas loop for carrier phase and frequency tracking."""
    
    def __init__(self, loop_bw: float = 0.02, order: int = 4):
        """
        Args:
            loop_bw: Loop bandwidth normalized to symbol rate (e.g., 0.01 - 0.05)
            order: 2 for BPSK, 4 for QPSK/QAM, 8 for 8PSK
        """
        self.order = order
        # Calculate loop filter coefficients (damping factor = 0.707)
        damping = 0.707
        theta = loop_bw / (damping + 0.25 / damping)
        self.alpha = (4 * damping * theta) / (1 + 2 * damping * theta + theta**2)
        self.beta = (4 * theta**2) / (1 + 2 * damping * theta + theta**2)
        
        self.phase = 0.0
        self.freq = 0.0
        
    def synchronize(self, samples: np.ndarray) -> np.ndarray:
        """Apply Costas loop to an array of symbol-spaced samples."""
        out = np.zeros_like(samples, dtype=complex)
        
        for i in range(len(samples)):
            # Rotate sample by current phase estimate
            out[i] = samples[i] * np.exp(-1j * self.phase)
            
            # Phase error detector
            if self.order == 2:
                # BPSK: phase error ~ real * imag
                error = np.real(out[i]) * np.imag(out[i])
            elif self.order == 4:
                # QPSK: error ~ sign(real)*imag - sign(imag)*real
                error = np.sign(np.real(out[i])) * np.imag(out[i]) - np.sign(np.imag(out[i])) * np.real(out[i])
            elif self.order == 8:
                # 8PSK: compute error from 8th power
                error = np.imag(out[i]**8)
            else:
                error = 0.0
            
            # Limit error
            error = np.clip(error, -1.0, 1.0)
            
            # Update loop filter
            self.freq += self.beta * error
            self.phase += self.freq + self.alpha * error
            
            # Wrap phase
            self.phase = (self.phase + np.pi) % (2 * np.pi) - np.pi
            
        return out


class SymbolSynchronizer:
    """Symbol timing recovery using minimum envelope variance (eye opening)."""
    
    def __init__(self, sps: int):
        self.sps = max(1, int(sps))
        
    def synchronize(self, samples: np.ndarray) -> np.ndarray:
        """Extract symbols from samples at the optimal sampling point."""
        if len(samples) < self.sps * 2 or self.sps == 1:
            return samples
            
        # Minimum normalized envelope variance selects the center of the eye opening
        variances = []
        for offset in range(self.sps):
            sub = samples[offset::self.sps]
            amp = np.abs(sub)
            norm_std = float(np.std(amp) / (np.mean(amp) + 1e-12))
            variances.append(norm_std)
            
        best_offset = int(np.argmin(variances))
        return samples[best_offset::self.sps]
