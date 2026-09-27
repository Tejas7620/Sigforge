"""
Signal File Loader.

Supports loading .wav (real or complex) and raw binary .iq / .bin files.
"""
import numpy as np
import scipy.io.wavfile as wavfile
import os
from typing import Tuple


class SignalLoader:
    """Loads RF signal files into complex numpy arrays."""
    
    @staticmethod
    def load_wav(file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load a .wav file.
        If stereo, assumes channel 0 is I and channel 1 is Q.
        If mono, returns real signal as complex (imaginary part = 0).
        """
        sample_rate, data = wavfile.read(file_path)
        
        # Normalize to [-1.0, 1.0]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
            
        if len(data.shape) == 2:
            # Stereo: I + jQ
            iq_data = data[:, 0] + 1j * data[:, 1]
        else:
            # Mono
            iq_data = data.astype(np.complex64)
            
        return iq_data, float(sample_rate)
        
    @staticmethod
    def load_raw(file_path: str, sample_rate: float, dtype=np.complex64) -> Tuple[np.ndarray, float]:
        """
        Load a raw binary IQ file.
        Assumes interleaved floats (I, Q, I, Q) for complex64.
        """
        data = np.fromfile(file_path, dtype=dtype)
        return data, sample_rate
        
    @classmethod
    def load(cls, file_path: str, sample_rate: float = 1e6) -> Tuple[np.ndarray, float]:
        """
        Auto-detects format and loads the file.
        For raw files, a default sample_rate must be provided or guessed.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.wav':
            return cls.load_wav(file_path)
        elif ext in ['.iq', '.bin', '.raw']:
            return cls.load_raw(file_path, sample_rate)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
