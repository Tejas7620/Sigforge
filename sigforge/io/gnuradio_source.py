"""
SigForge GNU Radio Source & Live SDR Network Streamer.

Provides:
- GNURadioSource: TCP/UDP client that connects to GNU Radio flowgraphs
  (e.g., Osmocom/RTL-SDR/HackRF/USRP -> TCP/UDP Sink) or rtl_tcp.
- RingBuffer: Thread-safe circular buffer for live IQ samples.
- GNURadioMockServer: Built-in SDR simulator server for offline testing and demos.
"""
from __future__ import annotations

import socket
import threading
import time
import os
from typing import Optional, Callable, Tuple
import numpy as np


class IQRingBuffer:
    """Thread-safe circular ring buffer for complex IQ samples."""

    def __init__(self, capacity: int = 1_048_576):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.complex64)
        self.write_idx = 0
        self.total_written = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Write incoming complex IQ samples into the ring buffer."""
        if len(samples) == 0:
            return
        samples = np.asarray(samples, dtype=np.complex64)
        n = len(samples)

        with self._lock:
            if n >= self.capacity:
                # Samples exceed capacity, take the most recent
                self.buffer[:] = samples[-self.capacity:]
                self.write_idx = 0
            else:
                end_idx = self.write_idx + n
                if end_idx <= self.capacity:
                    self.buffer[self.write_idx:end_idx] = samples
                else:
                    first_part = self.capacity - self.write_idx
                    self.buffer[self.write_idx:] = samples[:first_part]
                    self.buffer[:end_idx % self.capacity] = samples[first_part:]
                self.write_idx = end_idx % self.capacity
            self.total_written += n

    def get_latest(self, count: int) -> np.ndarray:
        """Retrieve the most recent N samples from the ring buffer."""
        with self._lock:
            available = min(self.total_written, self.capacity)
            actual_count = min(count, available)
            if actual_count <= 0:
                return np.zeros(0, dtype=np.complex64)

            # Reconstruct contiguous slice
            if self.total_written < self.capacity:
                # Buffer hasn't wrapped yet
                start = max(0, self.write_idx - actual_count)
                return self.buffer[start:self.write_idx].copy()
            else:
                # Wrapped
                idx = self.write_idx
                indices = (np.arange(idx - actual_count, idx)) % self.capacity
                return self.buffer[indices].copy()

    def clear(self) -> None:
        with self._lock:
            self.buffer.fill(0)
            self.write_idx = 0
            self.total_written = 0


class GNURadioSource:
    """
    Connects to a GNU Radio TCP/UDP stream or SDR server.

    Supported formats:
    - 'cf32': Complex Float32 (pair of 32-bit floats, GNU Radio native complex)
    - 'cs16': Complex Int16 (pair of 16-bit signed ints, HackRF / bladeRF)
    - 'cu8':  Complex UInt8 (pair of 8-bit unsigned ints, rtl_tcp)
    - 'cs8':  Complex Int8 (pair of 8-bit signed ints)
    """

    FORMAT_MAP = {
        "cf32": (np.float32, 8, True),   # dtype, bytes_per_complex_sample, is_float
        "cs16": (np.int16,   4, False),
        "cu8":  (np.uint8,   2, False),
        "cs8":  (np.int8,    2, False),
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12345,
        protocol: str = "TCP",
        data_format: str = "cf32",
        sample_rate: float = 1_000_000.0,
        center_freq: float = 433_920_000.0,
        buffer_capacity: int = 1_048_576,
    ):
        self.host = host
        self.port = port
        self.protocol = protocol.upper()
        self.data_format = data_format.lower()
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)

        self.ring_buffer = IQRingBuffer(capacity=buffer_capacity)

        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._chunk_callbacks: list[Callable[[np.ndarray], None]] = []
        self._stats_callback: Optional[Callable[[dict], None]] = None

        self.total_bytes_received = 0
        self.total_samples_received = 0
        self.current_data_rate_bps = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def add_chunk_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        if cb not in self._chunk_callbacks:
            self._chunk_callbacks.append(cb)

    def remove_chunk_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        if cb in self._chunk_callbacks:
            self._chunk_callbacks.remove(cb)

    def set_stats_callback(self, cb: Callable[[dict], None]) -> None:
        self._stats_callback = cb

    def connect(self, timeout: float = 3.0) -> bool:
        """Connect to the GNU Radio / SDR network stream."""
        if self._connected:
            return True

        try:
            if self.protocol == "TCP":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((self.host, self.port))
                sock.settimeout(1.0)
                self._socket = sock
            elif self.protocol == "UDP":
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((self.host, self.port))
                sock.settimeout(1.0)
                self._socket = sock
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

            self._connected = True
            return True
        except Exception as e:
            self._connected = False
            self._socket = None
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port} via {self.protocol}: {e}")

    def disconnect(self) -> None:
        """Disconnect and stop streaming thread."""
        self.stop_stream()
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False

    def start_stream(self) -> None:
        """Start reading stream in background thread."""
        if not self._connected:
            self.connect()

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        """Stop background streaming thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _convert_bytes_to_iq(self, raw_bytes: bytes) -> np.ndarray:
        """Convert raw received bytes to numpy complex64 array based on data_format."""
        raw_dtype, bytes_per_sample, is_float = self.FORMAT_MAP.get(
            self.data_format, (np.float32, 8, True)
        )

        # Ensure byte length matches alignment
        item_size = np.dtype(raw_dtype).itemsize * 2
        valid_len = (len(raw_bytes) // item_size) * item_size
        if valid_len == 0:
            return np.zeros(0, dtype=np.complex64)

        if valid_len < len(raw_bytes):
            raw_bytes = raw_bytes[:valid_len]

        raw_arr = np.frombuffer(raw_bytes, dtype=raw_dtype)
        if len(raw_arr) < 2:
            return np.zeros(0, dtype=np.complex64)

        # Separate interleaved I and Q
        i_data = raw_arr[0::2]
        q_data = raw_arr[1::2]
        min_len = min(len(i_data), len(q_data))
        i_data = i_data[:min_len]
        q_data = q_data[:min_len]

        if is_float:
            # Float32 normalized [-1.0, 1.0]
            iq = (i_data + 1j * q_data).astype(np.complex64)
        else:
            if raw_dtype == np.int16:
                iq = ((i_data.astype(np.float32) / 32768.0) +
                      1j * (q_data.astype(np.float32) / 32768.0)).astype(np.complex64)
            elif raw_dtype == np.uint8:
                iq = (((i_data.astype(np.float32) - 128.0) / 128.0) +
                      1j * ((q_data.astype(np.float32) - 128.0) / 128.0)).astype(np.complex64)
            elif raw_dtype == np.int8:
                iq = ((i_data.astype(np.float32) / 128.0) +
                      1j * (q_data.astype(np.float32) / 128.0)).astype(np.complex64)
            else:
                iq = (i_data.astype(np.float32) + 1j * q_data.astype(np.float32)).astype(np.complex64)

        return iq

    def _stream_loop(self) -> None:
        """Continuous background loop receiving network packets."""
        chunk_bytes = 16384  # 16 KB per recv
        last_stat_time = time.time()
        stat_bytes = 0

        while self._running and self._connected and self._socket:
            try:
                if self.protocol == "TCP":
                    raw_data = self._socket.recv(chunk_bytes)
                    if not raw_data:
                        # Server closed connection
                        self._connected = False
                        break
                else:  # UDP
                    raw_data, _ = self._socket.recvfrom(chunk_bytes)

                if not raw_data:
                    continue

                iq_chunk = self._convert_bytes_to_iq(raw_data)
                if len(iq_chunk) > 0:
                    self.ring_buffer.write(iq_chunk)
                    self.total_bytes_received += len(raw_data)
                    self.total_samples_received += len(iq_chunk)
                    stat_bytes += len(raw_data)

                    # Trigger callbacks
                    for cb in self._chunk_callbacks:
                        try:
                            cb(iq_chunk)
                        except Exception:
                            pass

                # Stats update every 0.5s
                now = time.time()
                dt = now - last_stat_time
                if dt >= 0.5:
                    self.current_data_rate_bps = (stat_bytes / dt) * 8
                    if self._stats_callback:
                        self._stats_callback({
                            "rate_bps": self.current_data_rate_bps,
                            "rate_mbps": (self.current_data_rate_bps / 1_000_000.0),
                            "total_samples": self.total_samples_received,
                            "total_bytes": self.total_bytes_received,
                            "ring_buffer_fill": min(self.ring_buffer.total_written, self.ring_buffer.capacity),
                            "ring_buffer_capacity": self.ring_buffer.capacity,
                        })
                    stat_bytes = 0
                    last_stat_time = now

            except socket.timeout:
                continue
            except Exception:
                self._connected = False
                break

        self._running = False

    def capture_snapshot(self, num_samples: int = 131072) -> np.ndarray:
        """Capture the most recent N samples from the ring buffer."""
        return self.ring_buffer.get_latest(num_samples)

    def save_to_file(self, file_path: str, samples: np.ndarray, format_type: str = "cf32") -> None:
        """Save captured IQ samples to disk."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".wav":
            from sigforge.io.loader import SignalLoader
            import scipy.io.wavfile as wavfile
            # Interleave real and imag as stereo float32 or int16
            audio = np.column_stack([samples.real, samples.imag])
            audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            wavfile.write(file_path, int(self.sample_rate), audio_int16)
        else:
            # Raw binary interleaved float32
            interleaved = np.empty(len(samples) * 2, dtype=np.float32)
            interleaved[0::2] = samples.real.astype(np.float32)
            interleaved[1::2] = samples.imag.astype(np.float32)
            interleaved.tofile(file_path)


class GNURadioMockServer:
    """
    Built-in high-performance GNU Radio SDR Simulator server.

    Streams live synthetic signals (QPSK, BPSK, 2FSK, Tone + Noise) over TCP
    at standard SDR rates (1 MSPS), allowing complete testing without hardware.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12345,
        sample_rate: float = 1_000_000.0,
        modulation: str = "QPSK",
    ):
        self.host = host
        self.port = port
        self.sample_rate = float(sample_rate)
        self.modulation = modulation.upper()
        self._server_sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start mock SDR server in background."""
        if self._running:
            return

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(5)
        self._server_sock.setblocking(False)

        self._running = True
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop server and disconnect clients."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

    def _generate_iq_block(self, num_samples: int = 4096) -> bytes:
        """Generate a block of synthetic IQ samples formatted as cf32 (interleaved float32)."""
        sr = self.sample_rate
        t = np.arange(num_samples) / sr

        if self.modulation == "QPSK":
            # Symbol rate: 250 kBd (sps = 4)
            sps = 4
            num_syms = (num_samples // sps) + 1
            constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
            syms = np.random.choice(constellation, size=num_syms)
            iq_upsampled = np.repeat(syms, sps)[:num_samples]
            # Add small CFO (e.g. 1.5 kHz) and phase noise
            cfo_term = np.exp(1j * 2 * np.pi * 1500.0 * t).astype(np.complex64)
            signal = iq_upsampled * cfo_term
        elif self.modulation == "BPSK":
            sps = 4
            num_syms = (num_samples // sps) + 1
            syms = np.random.choice(np.array([-1.0, 1.0], dtype=np.complex64), size=num_syms)
            iq_upsampled = np.repeat(syms, sps)[:num_samples]
            cfo_term = np.exp(1j * 2 * np.pi * 800.0 * t).astype(np.complex64)
            signal = iq_upsampled * cfo_term
        elif self.modulation == "2FSK":
            # Deviation 50 kHz
            f_dev = 50_000.0
            sps = 8
            num_syms = (num_samples // sps) + 1
            bits = np.random.choice([-1.0, 1.0], size=num_syms)
            freqs = np.repeat(bits * f_dev, sps)[:num_samples]
            phase = 2 * np.pi * np.cumsum(freqs) / sr
            signal = np.exp(1j * phase).astype(np.complex64)
        else:
            # CW Tone at 100 kHz
            signal = np.exp(1j * 2 * np.pi * 100_000.0 * t).astype(np.complex64)

        # Add AWGN (SNR ~ 18 dB)
        noise_std = 0.05
        noise = (np.random.normal(0, noise_std, num_samples) +
                 1j * np.random.normal(0, noise_std, num_samples)).astype(np.complex64)
        rx = signal + noise

        # Interleave I and Q as float32
        out = np.empty(num_samples * 2, dtype=np.float32)
        out[0::2] = rx.real.astype(np.float32)
        out[1::2] = rx.imag.astype(np.float32)
        return out.tobytes()

    def _server_loop(self) -> None:
        """Accept clients and stream synthetic IQ blocks."""
        block_samples = 8192
        block_duration = block_samples / self.sample_rate
        next_block_time = time.time()

        while self._running:
            # Accept new connections non-blocking
            if self._server_sock:
                try:
                    conn, _ = self._server_sock.accept()
                    conn.setblocking(True)
                    with self._lock:
                        self._clients.append(conn)
                except (BlockingIOError, socket.error):
                    pass
                except Exception:
                    break

            # If we have clients, generate and send blocks
            with self._lock:
                active_clients = list(self._clients)

            if active_clients:
                raw_bytes = self._generate_iq_block(block_samples)

                dead_clients = []
                for client in active_clients:
                    try:
                        client.sendall(raw_bytes)
                    except Exception:
                        dead_clients.append(client)

                if dead_clients:
                    with self._lock:
                        for dc in dead_clients:
                            if dc in self._clients:
                                self._clients.remove(dc)
                            try:
                                dc.close()
                            except Exception:
                                pass

                next_block_time += block_duration
                now = time.time()
                sleep_t = next_block_time - now
                if sleep_t > 0.002:
                    time.sleep(sleep_t)
                elif sleep_t < -0.1:
                    next_block_time = now
            else:
                time.sleep(0.02)
                next_block_time = time.time()

