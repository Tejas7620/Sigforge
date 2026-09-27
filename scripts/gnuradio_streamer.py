#!/usr/bin/env python3
"""
SigForge GNU Radio SDR Live IQ Streamer.

This script acts as the SDR server bridge for SigForge:
1. If GNU Radio and SDR hardware (RTL-SDR / HackRF / USRP / Osmosdr) are available,
   it runs an official GNU Radio top_block streaming complex IQ samples to a TCP sink.
2. If GNU Radio is not installed or no SDR device is plugged in, it runs in
   high-performance standalone SDR emulation mode, streaming live modulated IQ frames
   (QPSK, BPSK, 2FSK, or tone) over TCP socket to SigForge.

Usage:
    python scripts/gnuradio_streamer.py [--host 127.0.0.1] [--port 12345] [--rate 1000000] [--freq 433.92e6] [--mod QPSK]
"""
import sys
import os
import time
import argparse
import signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def run_gnuradio_flowgraph(host: str, port: int, samp_rate: float, freq: float, gain: float, device_args: str):
    """Run GNU Radio flowgraph if gnuradio is installed."""
    try:
        from gnuradio import gr, blocks
        try:
            import osmosdr
            has_osmosdr = True
        except ImportError:
            has_osmosdr = False

        print(f"[GNU Radio] Initializing GNU Radio top_block (samp_rate={samp_rate/1e6:.2f}MSPS, freq={freq/1e6:.2f}MHz)...")
        tb = gr.top_block("sigforge_sdr_streamer")

        if has_osmosdr and device_args != "synth":
            print(f"[GNU Radio] Using Osmocom SDR Source ({device_args})...")
            src = osmosdr.source(args=device_args)
            src.set_sample_rate(samp_rate)
            src.set_center_freq(freq, 0)
            src.set_gain_mode(False, 0)
            src.set_gain(gain, 0)
        else:
            print("[GNU Radio] Osmosdr hardware block not detected; using GNU Radio analog signal source...")
            from gnuradio import analog
            src = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 10000, 1.0, 0)

        # TCP Server Sink: port, host, vec_len=1
        tcp_sink = blocks.tcp_server_sink(
            itemsize=gr.sizeof_gr_complex,
            host=host,
            port=port,
            server=True
        )

        tb.connect(src, tcp_sink)
        print(f"[GNU Radio] Server listening on {host}:{port}. Ready for SigForge connection.")
        tb.run()
        return True
    except ImportError as e:
        print(f"[GNU Radio] GNU Radio Python bindings not available: {e}")
        return False
    except Exception as e:
        print(f"[GNU Radio] Flowgraph runtime error: {e}")
        return False


def run_standalone_emulator(host: str, port: int, samp_rate: float, modulation: str):
    """Run high-performance standalone SDR emulator."""
    from sigforge.io.gnuradio_source import GNURadioMockServer

    print("=" * 65)
    print("  SigForge GNU Radio SDR Live Streamer (Emulation Mode)")
    print(f"  Streaming:   {modulation.upper()} @ {samp_rate/1e6:.2f} MSPS")
    print(f"  TCP Server:  {host}:{port}")
    print("=" * 65)
    print("[Streamer] Starting TCP server...")

    server = GNURadioMockServer(
        host=host,
        port=port,
        sample_rate=samp_rate,
        modulation=modulation,
    )
    server.start()
    print(f"[Streamer] Server active! Connect SigForge to {host}:{port} via TCP.")
    print("[Streamer] Press Ctrl+C to terminate.")

    def handle_sigint(signum, frame):
        print("\n[Streamer] Shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description="SigForge GNU Radio SDR Live IQ Streamer")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TCP listen host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12345, help="TCP listen port (default: 12345)")
    parser.add_argument("--rate", type=float, default=1_000_000.0, help="Sample rate in Hz (default: 1000000)")
    parser.add_argument("--freq", type=float, default=433_920_000.0, help="Center freq in Hz (default: 433.92 MHz)")
    parser.add_argument("--gain", type=float, default=30.0, help="RF gain in dB (default: 30.0)")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "rtlsdr", "hackrf", "usrp", "synth"],
                        help="SDR Hardware source or synthetic fallback")
    parser.add_argument("--mod", type=str, default="QPSK", choices=["QPSK", "BPSK", "2FSK", "TONE"],
                        help="Modulation for synthetic signal (default: QPSK)")
    args = parser.parse_args()

    if args.source != "synth":
        device_args = ""
        if args.source == "rtlsdr":
            device_args = "rtl=0"
        elif args.source == "hackrf":
            device_args = "hackrf=0"
        elif args.source == "usrp":
            device_args = "uhd"

        success = run_gnuradio_flowgraph(
            host=args.host,
            port=args.port,
            samp_rate=args.rate,
            freq=args.freq,
            gain=args.gain,
            device_args=device_args,
        )
        if success:
            return

    # Fallback to standalone SDR emulator
    run_standalone_emulator(
        host=args.host,
        port=args.port,
        samp_rate=args.rate,
        modulation=args.mod,
    )


if __name__ == "__main__":
    main()
