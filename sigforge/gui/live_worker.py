"""
Live SDR & GNU Radio Background Stream Worker.

Connects to a live GNU Radio TCP/UDP stream, handles real-time IQ buffering,
computes FFT spectrum / waterfall slices at 30 FPS, and captures snapshots
for instant SigForge GNI pipeline analysis.
"""
from __future__ import annotations

import time
import socket
from typing import Optional
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from sigforge.io.gnuradio_source import GNURadioSource, GNURadioMockServer


class LiveStreamWorker(QThread):
    """
    Background worker thread managing GNU Radio SDR socket streaming.
    """

    # Signals
    connected_signal = pyqtSignal(bool, str)        # status, message
    fft_update = pyqtSignal(object, object)          # freqs (np.ndarray), psd_db (np.ndarray)
    waterfall_row = pyqtSignal(object)               # psd_row (np.ndarray)
    stats_update = pyqtSignal(dict)                  # dict with throughput, buffer stats, peak freq
    capture_complete = pyqtSignal(object, float)     # iq_data (np.ndarray), sample_rate (float)
    error_signal = pyqtSignal(str)                   # error message

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12345,
        protocol: str = "TCP",
        data_format: str = "cf32",
        sample_rate: float = 1_000_000.0,
        center_freq: float = 433_920_000.0,
        fft_size: int = 1024,
        parent=None,
    ):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.protocol = protocol
        self.data_format = data_format
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.fft_size = fft_size

        self.source: Optional[GNURadioSource] = None
        self.mock_server: Optional[GNURadioMockServer] = None

        self._is_running = False
        self._capture_requested = False
        self._capture_samples_count = 131072

        # Precompute FFT window
        self._fft_window = np.hanning(self.fft_size).astype(np.float32)

    def set_connection_params(
        self,
        host: str,
        port: int,
        protocol: str,
        data_format: str,
        sample_rate: float,
        center_freq: float,
    ) -> None:
        """Update connection parameters before starting."""
        self.host = host
        self.port = port
        self.protocol = protocol
        self.data_format = data_format
        self.sample_rate = sample_rate
        self.center_freq = center_freq

    def start_mock_simulator(self, modulation: str = "QPSK") -> None:
        """Launch the built-in GNU Radio simulator server."""
        if self.mock_server and self.mock_server.is_running:
            self.mock_server.stop()
        self.mock_server = GNURadioMockServer(
            host=self.host,
            port=self.port,
            sample_rate=self.sample_rate,
            modulation=modulation,
        )
        self.mock_server.start()

    def stop_mock_simulator(self) -> None:
        """Stop the built-in GNU Radio simulator server."""
        if self.mock_server:
            self.mock_server.stop()
            self.mock_server = None

    def request_capture(self, num_samples: int = 131072) -> None:
        """Request immediate snapshot capture from the live ring buffer."""
        self._capture_samples_count = num_samples
        self._capture_requested = True

    def stop(self) -> None:
        """Signal thread to stop."""
        self._is_running = False
        if self.source:
            self.source.disconnect()

    def run(self) -> None:
        """Thread execution: connect, stream, compute FFT, and handle captures."""
        self._is_running = True

        try:
            self.source = GNURadioSource(
                host=self.host,
                port=self.port,
                protocol=self.protocol,
                data_format=self.data_format,
                sample_rate=self.sample_rate,
                center_freq=self.center_freq,
                buffer_capacity=1_048_576,
            )
            self.source.connect(timeout=3.0)
            self.source.start_stream()
            self.connected_signal.emit(True, f"Connected to {self.host}:{self.port} ({self.protocol})")
        except Exception as e:
            self.connected_signal.emit(False, str(e))
            self.error_signal.emit(f"Connection failed: {e}")
            self._is_running = False
            return

        last_fft_time = 0.0
        min_fft_interval = 1.0 / 30.0  # Limit GUI spectrum updates to ~30 FPS

        # Precompute frequency axis
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.sample_rate)) + self.center_freq

        while self._is_running and self.source and self.source.is_connected:
            now = time.time()

            # Handle capture request
            if self._capture_requested:
                self._capture_requested = False
                captured_iq = self.source.capture_snapshot(self._capture_samples_count)
                if len(captured_iq) > 0:
                    self.capture_complete.emit(captured_iq, self.sample_rate)

            # Check if it's time to refresh FFT spectrum & waterfall
            if (now - last_fft_time) >= min_fft_interval:
                recent_samples = self.source.capture_snapshot(self.fft_size)
                if len(recent_samples) == self.fft_size:
                    # Windowed FFT
                    windowed = recent_samples * self._fft_window
                    fft_vals = np.fft.fftshift(np.fft.fft(windowed))
                    psd = (np.abs(fft_vals) ** 2) / (self.fft_size * np.sum(self._fft_window ** 2) + 1e-12)
                    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

                    self.fft_update.emit(freqs, psd_db)
                    self.waterfall_row.emit(psd_db)

                    # Compute statistics
                    peak_idx = int(np.argmax(psd_db))
                    peak_freq = float(freqs[peak_idx])
                    peak_power = float(psd_db[peak_idx])

                    fill_samples = min(self.source.ring_buffer.total_written, self.source.ring_buffer.capacity)
                    stats = {
                        "rate_mbps": self.source.current_data_rate_bps / 1_000_000.0,
                        "total_samples": self.source.total_samples_received,
                        "buffer_fill": fill_samples,
                        "buffer_capacity": self.source.ring_buffer.capacity,
                        "peak_freq": peak_freq,
                        "peak_power": peak_power,
                    }
                    self.stats_update.emit(stats)
                    last_fft_time = now

            time.sleep(0.005)

        if self.source:
            self.source.disconnect()
        self._is_running = False
        self.connected_signal.emit(False, "Stream disconnected")
