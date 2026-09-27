import sys
import numpy as np
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from sigforge.reconstruction.forward import ForwardReconstructor
from sigforge.classify.modulation_types import ModulationType
from sigforge.pipeline.orchestrator import PipelineOrchestrator
from sigforge.pipeline.config import PipelineConfig

def generate_synthetic_qpsk(num_symbols: int = 1000, sample_rate: float = 1e6, symbol_rate: float = 250e3):
    """Generate a clean synthetic QPSK signal for testing."""
    bits = np.random.randint(0, 2, num_symbols * 2).astype(np.uint8)
    reconstructor = ForwardReconstructor(sample_rate)
    
    # Generate clean IQ
    clean_iq = reconstructor.reconstruct(
        bits=bits, 
        mod_type=ModulationType.QPSK, 
        symbol_rate=symbol_rate,
        cfo_hz=500.0,
        phase_offset=np.pi/8
    )
    
    # Add AWGN
    noise_power = 0.01
    noise = np.random.normal(0, np.sqrt(noise_power/2), len(clean_iq)) + \
            1j * np.random.normal(0, np.sqrt(noise_power/2), len(clean_iq))
            
    return clean_iq + noise, bits

def test_pipeline():
    print("Generating Synthetic QPSK...")
    sample_rate = 1e6
    iq_data, true_bits = generate_synthetic_qpsk(sample_rate=sample_rate)
    
    config = PipelineConfig()
    orchestrator = PipelineOrchestrator(config)
    
    print("Running Pipeline Orchestrator (Predict -> Recover -> Reconstruct -> Verify)...")
    result = orchestrator.run(iq_data, sample_rate)
    
    print("\n--- HYPOTHESIS RESULTS ---")
    print(result.hypothesis_graph.summary())
    
    best = result.hypothesis_graph.best
    print("\n--- BEST HYPOTHESIS EVIDENCE ---")
    assert best is not None, "Pipeline must produce at least one hypothesis"
    assert best.modulation == ModulationType.QPSK, f"Expected QPSK as best hypothesis, got {best.modulation}"
    for ev in best.evidence:
        print(f"[{ev.source}] Score: {ev.score:.3f} (Weight: {ev.weight}) -> {ev.detail}")

if __name__ == "__main__":
    test_pipeline()

