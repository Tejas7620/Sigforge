import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sigforge.reconstruction.forward import ForwardReconstructor
from sigforge.classify.modulation_types import ModulationType

def generate_and_save():
    print("Generating Synthetic QPSK Signal...")
    sample_rate = 1e6
    symbol_rate = 250e3
    num_symbols = 4000
    
    bits = np.random.randint(0, 2, num_symbols * 2).astype(np.uint8)
    reconstructor = ForwardReconstructor(sample_rate)
    
    clean_iq = reconstructor.reconstruct(
        bits=bits, 
        mod_type=ModulationType.QPSK, 
        symbol_rate=symbol_rate,
        cfo_hz=200.0,
        phase_offset=np.pi/4
    )
    
    # Add AWGN (SNR ~ 10dB)
    noise_power = np.var(clean_iq) / 10.0
    noise = np.random.normal(0, np.sqrt(noise_power/2), len(clean_iq)) + \
            1j * np.random.normal(0, np.sqrt(noise_power/2), len(clean_iq))
            
    noisy_iq = clean_iq + noise
    
    # Save as complex64 binary
    out_file = "test_qpsk.bin"
    noisy_iq.astype(np.complex64).tofile(out_file)
    print(f"Saved {len(noisy_iq)} samples to {out_file}")

if __name__ == "__main__":
    generate_and_save()
