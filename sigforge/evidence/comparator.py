"""
Evidence Engine Comparator.

Evaluates hypotheses by comparing the original observed RF signal with
the forward-reconstructed IQ generated from the recovered bitstream.
"""
import numpy as np
from scipy import signal


class EvidenceComparator:
    """Compares observed IQ against reconstructed IQ to generate a support score."""
    
    def align_signals(self, observed: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Align signals using cross-correlation to handle any timing offsets."""
        # Normalize for correlation
        obs_norm = observed / (np.linalg.norm(observed) + 1e-12)
        rec_norm = reconstructed / (np.linalg.norm(reconstructed) + 1e-12)
        
        # Cross correlate
        corr = signal.correlate(obs_norm, rec_norm, mode='full')
        delay = np.argmax(np.abs(corr)) - (len(reconstructed) - 1)
        
        # Align via overlap slicing rather than zero-padding
        if delay >= 0:
            obs_aligned = observed[delay:]
            rec_aligned = reconstructed[:len(obs_aligned)]
            min_len = min(len(obs_aligned), len(rec_aligned))
            obs_aligned = obs_aligned[:min_len]
            rec_aligned = rec_aligned[:min_len]
        else:
            delay_pos = -delay
            obs_aligned = observed[:max(0, len(reconstructed) - delay_pos)]
            rec_aligned = reconstructed[delay_pos: delay_pos + len(obs_aligned)]
            min_len = min(len(obs_aligned), len(rec_aligned))
            obs_aligned = obs_aligned[:min_len]
            rec_aligned = rec_aligned[:min_len]
            
        return obs_aligned, rec_aligned

    def compare(self, observed: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
        """
        Compare the observed and reconstructed signals.
        
        Returns:
            (support_score, nmse)
            support_score: 0.0 to 1.0 (1.0 = perfect match)
            nmse: Normalized Mean Square Error
        """
        if len(observed) == 0 or len(reconstructed) == 0:
            return 0.0, float('inf')
            
        obs_align, rec_align = self.align_signals(observed, reconstructed)
        
        # Scale to same power level
        p_obs = np.mean(np.abs(obs_align)**2)
        p_rec = np.mean(np.abs(rec_align)**2)
        
        if p_rec == 0:
            return 0.0, float('inf')
            
        rec_scaled = rec_align * np.sqrt(p_obs / p_rec)
        
        # Fine frequency and phase alignment between observed and reconstructed
        prod = obs_align * np.conj(rec_scaled)
        if len(prod) > 1:
            dphase = np.angle(prod[1:] * np.conj(prod[:-1]))
            residual_w = float(np.median(dphase))
            t_idx = np.arange(len(rec_scaled))
            rec_scaled = rec_scaled * np.exp(1j * residual_w * t_idx)

        # Find optimal static phase shift theta
        cross_term = np.sum(obs_align * np.conj(rec_scaled))
        theta = np.angle(cross_term)
        rec_scaled = rec_scaled * np.exp(1j * theta)
        
        # Compute NMSE (Normalized Mean Square Error)
        error = obs_align - rec_scaled
        mse = np.mean(np.abs(error)**2)
        nmse = float(mse / (p_obs + 1e-12))
        
        # Convert NMSE to a confidence score (0.0 to 1.0)
        # Perfect match: nmse = 0 -> score = 1.0
        # Complete mismatch (orthogonal/uncorrelated): nmse ~ 1.0 or higher -> score = 0.0
        support_score = np.exp(-nmse * 2.0) # Empirical scaling
        
        return support_score, nmse

