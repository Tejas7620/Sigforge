"""Analysis Worker Thread for PyQt GUI."""
from __future__ import annotations

from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

from sigforge.pipeline.orchestrator import PipelineOrchestrator
from sigforge.pipeline.config import PipelineConfig
from sigforge.io.loader import SignalLoader
from sigforge.classify.modulation_types import ModulationType


class AnalysisWorker(QThread):
    """Runs the PipelineOrchestrator in a background thread."""

    # Signals
    progress = pyqtSignal(int, str)       # value (0-100), message
    stage_update = pyqtSignal(str, str)   # stage_id, status (running, done, error)
    finished = pyqtSignal(object)         # PipelineResult
    error = pyqtSignal(str)

    def __init__(
        self,
        file_path: Optional[str] = None,
        iq_data: Optional[np.ndarray] = None,
        sample_rate_override: Optional[float] = None,
        symbol_rate_override: Optional[float] = None,
        mod_type_override: Optional[ModulationType] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.iq_data = iq_data
        self.sample_rate_override = sample_rate_override
        self.symbol_rate_override = symbol_rate_override
        self.mod_type_override = mod_type_override
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.stage_update.emit("ingest", "running")

            if self.iq_data is not None:
                self.progress.emit(10, "Ingesting Live SDR IQ Data...")
                iq_data = np.asarray(self.iq_data, dtype=np.complex64)
                sample_rate = self.sample_rate_override if self.sample_rate_override is not None else 1e6
            elif self.file_path:
                self.progress.emit(5, "Loading File...")
                load_sr = self.sample_rate_override if self.sample_rate_override is not None else 1e6
                iq_data, sample_rate = SignalLoader.load(self.file_path, sample_rate=load_sr)
            else:
                raise ValueError("No file_path or iq_data provided to AnalysisWorker.")

            self.stage_update.emit("ingest", "done")

            if not self._is_running:
                return

            self.progress.emit(20, "Analyzing Signal...")

            stage_progress = {
                "preprocess": 25,
                "analysis": 35,
                "param_est": 45,
                "classify": 55,
                "sync": 65,
                "demod": 75,
                "bits": 80,
                "interleave": 85,
                "fec": 90,
                "validate": 95,
                "hypothesis": 98,
            }

            def stage_cb(stage_id: str, status: str):
                if not self._is_running:
                    return
                self.stage_update.emit(stage_id, status)
                if status == "running" and stage_id in stage_progress:
                    self.progress.emit(stage_progress[stage_id], f"{stage_id.replace('_', ' ').title()}...")

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config)

            result = orchestrator.run(
                iq_data=iq_data,
                sample_rate=sample_rate,
                symbol_rate_override=self.symbol_rate_override,
                mod_type_override=self.mod_type_override,
                stage_callback=stage_cb,
            )

            if not self._is_running:
                return

            self.progress.emit(100, "Analysis Complete")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
            self.stage_update.emit("ingest", "error")
