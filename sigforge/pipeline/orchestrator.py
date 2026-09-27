"""
Pipeline Orchestrator.

Ties together all analysis stages into the core hypothesis-and-verification loop.
Predict -> Recover -> Reconstruct -> Verify
"""
from __future__ import annotations

import time
import logging
from typing import Callable, Dict, List, Optional
import numpy as np

from sigforge.pipeline.config import PipelineConfig
from sigforge.pipeline.data_types import PipelineResult, ParameterEstimates, SignalProfile, DemodResult
from sigforge.hypothesis.hypothesis import SignalHypothesis, HypothesisGraph
from sigforge.classify.modulation_types import ModulationType, ModulationFamily
from sigforge.classify.classifier import ModulationClassifier
from sigforge.analysis.estimator import ParameterEstimator

# Demodulators
from sigforge.demod.psk import PSKDemodulator
from sigforge.demod.fsk import FSKDemodulator
from sigforge.demod.qam import QAMDemodulator

# Synchronization & Verification
from sigforge.synchronize.sync import CostasLoop, SymbolSynchronizer
from sigforge.reconstruction.forward import ForwardReconstructor
from sigforge.evidence.comparator import EvidenceComparator
from sigforge.fec.decoder import FECSearchEngine

logger = logging.getLogger("sigforge.pipeline")


class PipelineOrchestrator:
    """Executes the core SigForge analysis loop on a block of IQ data."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.fec_engine = FECSearchEngine()
        self.classifier = ModulationClassifier()

    def _get_demodulator(self, mod_type: ModulationType):
        """Factory: return the correct demodulator for a modulation type."""
        family = mod_type.family
        if family == ModulationFamily.PSK:
            return PSKDemodulator(mod_type=mod_type)
        elif family == ModulationFamily.FSK:
            return FSKDemodulator(mod_type=mod_type)
        elif family == ModulationFamily.QAM:
            return QAMDemodulator(mod_type=mod_type)
        else:
            raise ValueError(f"No demodulator for {mod_type.value}")

    def _get_costas_order(self, mod_type: ModulationType) -> int:
        """Return the Costas loop order for a modulation type."""
        order_map = {
            ModulationType.BPSK: 2,
            ModulationType.QPSK: 4,
            ModulationType.OQPSK: 4,
            ModulationType.PSK8: 8,
            ModulationType.QAM16: 4,
            ModulationType.QAM64: 4,
        }
        return order_map.get(mod_type, 4)

    def run(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        symbol_rate_override: Optional[float] = None,
        mod_type_override: Optional[ModulationType] = None,
        stage_callback: Optional[Callable[[str, str], None]] = None,
    ) -> PipelineResult:
        """
        Run the full analysis pipeline.
        
        Args:
            iq_data: Complex IQ samples
            sample_rate: Sample rate in Hz
            symbol_rate_override: Optional user override for symbol rate in Hz
            mod_type_override: Optional user override for modulation type
            stage_callback: Optional callback func(stage_id: str, status: str)
        """
        def notify(stage: str, status: str):
            if stage_callback:
                stage_callback(stage, status)

        start_time = time.time()
        result = PipelineResult(iq_data=iq_data, sample_rate=sample_rate)
        result.hypothesis_graph = HypothesisGraph(max_candidates=self.config.max_hypothesis_branches)

        # Stage 2: Preprocessing (DC offset removal & power normalization)
        notify("preprocess", "running")
        x_centered = iq_data - np.mean(iq_data)
        p_avg = np.mean(np.abs(x_centered) ** 2)
        norm_factor = np.sqrt(p_avg) if p_avg > 1e-12 else 1.0
        iq_norm = x_centered / norm_factor
        notify("preprocess", "done")

        # Stage 3: Signal Analysis & Parameter Estimation
        notify("analysis", "running")
        notify("param_est", "running")
        estimator = ParameterEstimator(sample_rate)
        est = estimator.estimate_all(iq_norm)

        # Build Signal Profile
        profile = SignalProfile()
        profile.snr_db = est.snr_db
        profile.noise_floor_db = est.noise_floor_db
        profile.occupied_bandwidth = est.occupied_bw_hz
        profile.center_freq_offset = est.cfo_hz

        # Coarse CFO de-rotation for accurate feature extraction & carrier sync
        t = np.arange(len(iq_norm)) / sample_rate
        cfo_hz = est.cfo_hz
        iq_derotated = iq_norm * np.exp(-2j * np.pi * cfo_hz * t)

        # Extract features & cumulants on derotated IQ
        features = self.classifier.extract_features(iq_derotated, sample_rate)
        profile.cumulants["C20"] = complex(features.get("abs_c20", 0.0))
        profile.cumulants["C40"] = complex(features.get("abs_c40", 0.0))
        profile.cumulants["C42"] = complex(features.get("c42_real", 0.0))
        profile.inst_amplitude_std = features.get("sigma_aa", 0.0)
        profile.kurtosis = features.get("kurtosis", 0.0)
        result.signal_profile = profile

        # Use overrides or estimated parameters
        est_symbol_rate = symbol_rate_override if symbol_rate_override is not None else est.symbol_rate
        est_sps = sample_rate / est_symbol_rate if est_symbol_rate > 0 else 4.0

        result.parameter_estimates = ParameterEstimates(
            symbol_rate=est_symbol_rate,
            samples_per_symbol=est_sps,
            cfo_hz=est.cfo_hz,
            symbol_rate_confidence=est.symbol_rate_confidence,
        )
        notify("analysis", "done")
        notify("param_est", "done")

        # Stage 4: Modulation Classification (using derotated signal + symbol-spaced features)
        notify("classify", "running")
        classification = self.classifier.classify(iq_derotated, sample_rate, sps=est_sps)
        notify("classify", "done")

        # Determine candidates to test
        if mod_type_override is not None and mod_type_override != ModulationType.UNKNOWN:
            candidates = [mod_type_override]
        else:
            # Test all major supported candidates, with top classified first
            top_candidate = classification.predicted_type
            candidates = [top_candidate]
            for m in [
                ModulationType.QPSK,
                ModulationType.BPSK,
                ModulationType.PSK8,
                ModulationType.FSK2,
                ModulationType.QAM16,
                ModulationType.MSK,
            ]:
                if m not in candidates:
                    candidates.append(m)

        notify("sync", "running")
        notify("demod", "running")

        comparator = EvidenceComparator()

        for mod_type in candidates:
            logger.info(f"Testing hypothesis: {mod_type.value}")
            sps = int(round(est_sps))
            if sps < 1:
                sps = 1

            hypo = SignalHypothesis(
                modulation=mod_type,
                symbol_rate=est_symbol_rate,
                samples_per_symbol=est_sps,
                carrier_offset=cfo_hz,
            )

            # Add classification / cumulant evidence
            mod_score = 0.2
            for ranked_mod, score in classification.ranked_candidates:
                if ranked_mod == mod_type:
                    mod_score = score
                    break
            hypo.add_evidence(
                "cumulants",
                mod_score,
                detail=f"cumulant/classifier match ({mod_score:.1%})"
            )

            try:
                # --- RECOVER ---
                if mod_type.family == ModulationFamily.FSK:
                    sync_timing = SymbolSynchronizer(sps=sps)
                    sym_samples = sync_timing.synchronize(iq_norm)
                    demod = self._get_demodulator(mod_type)
                    demod_res = demod.demodulate(sym_samples)
                else:
                    if sps > 1:
                        import scipy.signal
                        from sigforge.reconstruction.forward import rrcosfilter
                        rrc_len = min(11 * sps, 64)
                        rrc_rx = rrcosfilter(rrc_len, 0.35, 1.0 / est_symbol_rate, sample_rate)
                        rx_signal = scipy.signal.lfilter(rrc_rx, 1.0, iq_derotated)
                    else:
                        rx_signal = iq_derotated

                    sync_timing = SymbolSynchronizer(sps=sps)
                    sym_samples = sync_timing.synchronize(rx_signal)

                    order = self._get_costas_order(mod_type)
                    sync_carrier = CostasLoop(
                        loop_bw=self.config.demod.costas_loop_bw,
                        order=order,
                    )
                    locked_syms = sync_carrier.synchronize(sym_samples)

                    demod = self._get_demodulator(mod_type)
                    demod_res = demod.demodulate(locked_syms)

                hypo.evm_db = demod_res.evm_db
                result.demod_results.append(demod_res)

                recovered_bits = demod_res.hard_bits
                if recovered_bits is None or len(recovered_bits) == 0:
                    continue

                notify("bits", "done")

                # --- FEC SEARCH ---
                notify("fec", "running")
                fec_results = self.fec_engine.search(recovered_bits, max_results=1)
                if fec_results:
                    best_fec = fec_results[0]
                    result.fec_results.append(best_fec)
                    if best_fec.confidence > 0.5:
                        hypo.fec_type = f"{best_fec.codec_type}({best_fec.codec_id})"
                        hypo.add_evidence(
                            "fec_syndrome",
                            best_fec.confidence,
                            detail=f"{best_fec.codec_id} match={best_fec.re_encode_match:.2%}"
                        )
                        if best_fec.syndrome_zero:
                            hypo.has_hard_validation = True
                notify("fec", "done")

                # --- RECONSTRUCT ---
                notify("validate", "running")
                try:
                    reconstructor = ForwardReconstructor(sample_rate=sample_rate)
                    reconstructed_iq = reconstructor.reconstruct(
                        bits=recovered_bits,
                        mod_type=mod_type,
                        symbol_rate=est_symbol_rate,
                        cfo_hz=0.0,
                    )

                    # --- VERIFY ---
                    support_score, nmse = comparator.compare(
                        observed=iq_derotated,
                        reconstructed=reconstructed_iq,
                    )

                    hypo.add_evidence(
                        "re_encoding_match",
                        support_score,
                        detail=f"NMSE: {nmse:.4f}"
                    )
                except NotImplementedError:
                    hypo.warnings.append(f"Reconstruction not available for {mod_type.value}")

                # Evidence from EVM (lower EVM dB = higher quality)
                if hypo.evm_db is not None:
                    if hypo.evm_db <= 0:
                        evm_score = float(np.clip((-hypo.evm_db) / 25.0, 0.0, 1.0))
                    else:
                        evm_score = 0.0
                    hypo.add_evidence(
                        "constellation_quality",
                        evm_score,
                        detail=f"EVM: {hypo.evm_db:.1f} dB"
                    )

                # Evidence from SNR
                if result.signal_profile.snr_db is not None:
                    snr_score = np.clip(result.signal_profile.snr_db / 30.0, 0.0, 1.0)
                    hypo.add_evidence(
                        "snr_consistency",
                        snr_score,
                        detail=f"SNR: {result.signal_profile.snr_db:.1f} dB"
                    )

                # Add hypothesis to graph
                result.hypothesis_graph.add(hypo)

            except Exception as e:
                logger.warning(f"Pipeline failed for {mod_type.value}: {e}")
                result.errors.append(f"Pipeline failed for {mod_type.value}: {str(e)}")

        notify("sync", "done")
        notify("demod", "done")
        notify("validate", "done")

        # Stage: Hypothesis Fusion
        notify("hypothesis", "running")
        result.hypothesis_graph.promote_validated()
        notify("hypothesis", "done")

        # Finalize
        result.processing_time_sec = time.time() - start_time
        logger.info(f"Pipeline complete in {result.processing_time_sec:.2f}s")
        return result
