import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sigforge.io.gnuradio_source import GNURadioSource, GNURadioMockServer
from sigforge.pipeline.orchestrator import PipelineOrchestrator
from sigforge.pipeline.config import PipelineConfig

def test_live_sdr_streaming():
    port = 12399
    host = "127.0.0.1"
    sample_rate = 1_000_000.0

    print("1. Starting GNURadioMockServer...")
    server = GNURadioMockServer(host=host, port=port, sample_rate=sample_rate, modulation="QPSK")
    server.start()
    time.sleep(0.3)

    print("2. Connecting GNURadioSource client...")
    client = GNURadioSource(host=host, port=port, sample_rate=sample_rate, protocol="TCP", data_format="cf32")
    client.connect()
    client.start_stream()

    print("3. Receiving live IQ samples for 1.0 second...")
    time.sleep(1.0)

    print(f"   Total bytes received: {client.total_bytes_received:,} bytes")
    print(f"   Total samples received: {client.total_samples_received:,} samples")
    print(f"   Throughput: {client.current_data_rate_bps/1e6:.2f} Mbps")

    assert client.total_samples_received > 10000, "Should have received > 10000 samples"

    print("4. Capturing snapshot from live ring buffer (65,536 samples)...")
    captured = client.capture_snapshot(65536)
    print(f"   Captured shape: {captured.shape}, dtype: {captured.dtype}")
    assert len(captured) == 65536

    print("5. Stopping stream and server...")
    client.disconnect()
    server.stop()

    print("6. Feeding live captured IQ data directly into SigForge GNI Pipeline...")
    config = PipelineConfig()
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(iq_data=captured, sample_rate=sample_rate)

    print("7. Pipeline completed successfully!")
    if result.signal_profile:
        print(f"   SNR: {result.signal_profile.snr_db:.1f} dB")
        print(f"   Occupied BW: {result.signal_profile.occupied_bandwidth/1e3:.1f} kHz")
    if result.parameter_estimates:
        print(f"   Symbol Rate: {result.parameter_estimates.symbol_rate:.1f} Bd")
    if result.hypothesis_graph and result.hypothesis_graph.best:
        print(f"   Best Hypothesis: {result.hypothesis_graph.best.description}")
        print(f"   Confidence: {result.hypothesis_graph.best.compute_confidence():.2f}")


    print("\nALL LIVE SDR & GNU RADIO INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    test_live_sdr_streaming()
