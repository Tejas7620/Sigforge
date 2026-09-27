"""
FEC (Forward Error Correction) Decoder Module.

Implements:
  - Reed-Solomon decoder (using reedsolo library)
  - Simple Convolutional code decoder (Viterbi, pure NumPy)
  - FEC search engine that tries multiple codec candidates

The FEC module is used in the hypothesis verification loop:
recovered bits are fed through candidate FEC decoders. If a decoder
produces a valid syndrome (or re-encoding match), it provides strong
evidence for the hypothesis.
"""
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from sigforge.pipeline.data_types import FECResult


# ── Reed-Solomon Decoder ──────────────────────────────────────────────

class ReedSolomonDecoder:
    """Wrapper around reedsolo for RS decoding with syndrome checking."""

    def __init__(self, n: int = 255, k: int = 223, prim: int = 0x11d):
        """
        Args:
            n: Codeword length (symbols).
            k: Message length (symbols).
            prim: Primitive polynomial for GF(2^8).
        """
        self.n = n
        self.k = k
        self.nsym = n - k  # number of parity symbols
        self.codec_id = f"RS({n},{k})"
        self._rs = None

        try:
            import reedsolo
            self._rs = reedsolo.RSCodec(self.nsym, prim=prim)
        except ImportError:
            pass  # reedsolo not available; decode will return failure

    def decode(self, bits: np.ndarray) -> FECResult:
        """
        Attempt RS decoding on the bitstream.

        The bitstream is grouped into 8-bit symbols, then fed to the RS decoder.
        """
        result = FECResult(
            codec_type="reed_solomon",
            codec_id=self.codec_id,
        )

        if self._rs is None:
            result.confidence = 0.0
            return result

        # Convert bits to bytes
        num_bytes = len(bits) // 8
        if num_bytes < self.n:
            result.confidence = 0.0
            return result

        byte_data = np.packbits(bits[:num_bytes * 8])

        # Try decoding blocks of n bytes
        decoded_bytes = bytearray()
        blocks_ok = 0
        blocks_total = 0
        errors_corrected_total = 0

        for start in range(0, len(byte_data) - self.n + 1, self.n):
            block = bytes(byte_data[start:start + self.n])
            blocks_total += 1
            try:
                decoded, _, errata_pos = self._rs.decode(block)
                decoded_bytes.extend(decoded)
                blocks_ok += 1
                errors_corrected_total += len(errata_pos)
            except Exception:
                # RS decode failure — block has too many errors
                decoded_bytes.extend(block[:self.k])

        if blocks_total == 0:
            result.confidence = 0.0
            return result

        # Convert decoded bytes back to bits
        decoded_bits = np.unpackbits(np.frombuffer(bytes(decoded_bytes), dtype=np.uint8))

        result.decoded_bits = decoded_bits
        result.syndrome_zero = (blocks_ok == blocks_total)
        result.errors_corrected = errors_corrected_total
        result.re_encode_match = blocks_ok / blocks_total if blocks_total > 0 else 0.0

        # Confidence scoring
        if result.syndrome_zero and blocks_total >= 2:
            result.confidence = 0.95  # Very strong evidence
        elif blocks_ok / blocks_total > 0.8:
            result.confidence = 0.7
        elif blocks_ok / blocks_total > 0.5:
            result.confidence = 0.4
        else:
            result.confidence = 0.1

        return result


# ── Convolutional Decoder (Viterbi) ───────────────────────────────────

class ViterbiDecoder:
    """
    Simple rate-1/2 convolutional code Viterbi decoder.

    Supports constraint lengths K=3, 5, 7 with standard generator polynomials.
    Pure NumPy implementation — no external dependencies.
    """

    def __init__(self, K: int = 7, generators: Tuple[int, ...] = (0o171, 0o133)):
        """
        Args:
            K: Constraint length.
            generators: Generator polynomials in octal.
        """
        self.K = K
        self.generators = generators
        self.rate_inv = len(generators)  # rate = 1/rate_inv
        self.num_states = 2 ** (K - 1)
        self.codec_id = f"k{K}_r1{self.rate_inv}"

        # Precompute state transitions and outputs
        self._build_trellis()

    def _build_trellis(self):
        """Precompute the trellis transitions and output bits."""
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.output = np.zeros((self.num_states, 2, self.rate_inv), dtype=np.uint8)

        for state in range(self.num_states):
            for input_bit in range(2):
                # Shift register: input_bit enters from the left
                reg = (input_bit << (self.K - 1)) | state
                next_s = reg >> 1
                self.next_state[state, input_bit] = next_s

                # Compute output for each generator polynomial
                for g_idx, gen in enumerate(self.generators):
                    # Count number of 1s in (reg & gen) — XOR sum
                    val = reg & gen
                    parity = 0
                    while val:
                        parity ^= (val & 1)
                        val >>= 1
                    self.output[state, input_bit, g_idx] = parity

    def decode(self, bits: np.ndarray) -> FECResult:
        """
        Viterbi decode a bitstream assumed to be rate-1/2 encoded.

        Args:
            bits: Encoded bitstream (length must be even for rate-1/2).

        Returns:
            FECResult with decoded bits and confidence.
        """
        result = FECResult(
            codec_type="convolutional",
            codec_id=self.codec_id,
            rate=f"r1{self.rate_inv}",
        )

        # Cap to 512 bits for fast hypothesis verification
        if len(bits) > 512:
            bits = bits[:512]

        n_coded_bits = len(bits)
        if n_coded_bits < self.rate_inv * 10:
            result.confidence = 0.0
            return result

        # Group into code symbols (rate_inv bits per symbol)
        n_symbols = n_coded_bits // self.rate_inv
        coded = bits[:n_symbols * self.rate_inv].reshape(n_symbols, self.rate_inv)

        # Viterbi algorithm
        INF = 1e9
        path_metric = np.full(self.num_states, INF)
        path_metric[0] = 0.0  # Start in state 0

        # Survivor paths stored as indices
        survivors = np.zeros((n_symbols, self.num_states), dtype=np.int32)
        decisions = np.zeros((n_symbols, self.num_states), dtype=np.uint8)

        for t in range(n_symbols):
            new_metric = np.full(self.num_states, INF)

            for state in range(self.num_states):
                if path_metric[state] >= INF:
                    continue

                for input_bit in range(2):
                    next_s = self.next_state[state, input_bit]
                    expected = self.output[state, input_bit]

                    # Hamming distance between received and expected
                    dist = np.sum(coded[t] != expected)
                    candidate_metric = path_metric[state] + dist

                    if candidate_metric < new_metric[next_s]:
                        new_metric[next_s] = candidate_metric
                        survivors[t, next_s] = state
                        decisions[t, next_s] = input_bit

            path_metric = new_metric

        # Traceback from the state with minimum metric
        final_state = np.argmin(path_metric)
        min_metric = path_metric[final_state]

        decoded = np.zeros(n_symbols, dtype=np.uint8)
        state = final_state

        for t in range(n_symbols - 1, -1, -1):
            decoded[t] = decisions[t, state]
            state = survivors[t, state]

        result.decoded_bits = decoded

        # Re-encode to check match
        re_encoded = self._encode(decoded)
        original_coded = bits[:len(re_encoded)]
        if len(re_encoded) > 0 and len(original_coded) > 0:
            match_rate = np.mean(re_encoded == original_coded[:len(re_encoded)])
            result.re_encode_match = float(match_rate)
            result.syndrome_zero = match_rate > 0.99

            # BER improvement estimation
            result.errors_corrected = int(np.sum(re_encoded != original_coded[:len(re_encoded)]))

        # Confidence based on re-encode match
        if result.re_encode_match > 0.98:
            result.confidence = 0.9
        elif result.re_encode_match > 0.90:
            result.confidence = 0.6
        elif result.re_encode_match > 0.80:
            result.confidence = 0.3
        else:
            result.confidence = 0.05

        return result

    def _encode(self, data_bits: np.ndarray) -> np.ndarray:
        """Encode data bits using the convolutional code (for re-encode check)."""
        state = 0
        output = []
        for bit in data_bits:
            reg = (int(bit) << (self.K - 1)) | state
            state = reg >> 1

            for gen in self.generators:
                val = reg & gen
                parity = 0
                while val:
                    parity ^= (val & 1)
                    val >>= 1
                output.append(parity)

        return np.array(output, dtype=np.uint8)


# ── FEC Search Engine ─────────────────────────────────────────────────

class FECSearchEngine:
    """
    Tries multiple FEC codec candidates against a bitstream.

    Returns the best-matching FEC result, or a null result if none match.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Build candidate decoders
        self.candidates: List = []

        # Viterbi candidates (different constraint lengths)
        viterbi_specs = self.config.get("conv_codes", {
            "k3_r12": {"K": 3, "generators": (0o7, 0o5)},
            "k5_r12": {"K": 5, "generators": (0o23, 0o35)},
            "k7_r12": {"K": 7, "generators": (0o171, 0o133)},
        })
        for name, spec in viterbi_specs.items():
            self.candidates.append(
                ViterbiDecoder(K=spec["K"], generators=tuple(spec["generators"]))
            )

        # Reed-Solomon candidates
        rs_specs = self.config.get("rs_codes", [
            (255, 223),
            (255, 239),
        ])
        for n, k in rs_specs:
            self.candidates.append(ReedSolomonDecoder(n=n, k=k))

    def search(self, bits: np.ndarray, max_results: int = 3) -> List[FECResult]:
        """
        Try all FEC candidates and return the top matches ranked by confidence.
        """
        results = []

        for decoder in self.candidates:
            try:
                result = decoder.decode(bits)
                if result.confidence > 0.05:
                    results.append(result)
            except Exception:
                continue

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:max_results]
