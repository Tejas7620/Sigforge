"""
Report Exporter Module.

Generates HTML and JSON analysis reports from PipelineResult and HypothesisGraph.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
import numpy as np

from sigforge.pipeline.data_types import PipelineResult


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ReportExporter:
    """Exports signal analysis results to HTML and JSON format."""

    @staticmethod
    def to_json(result: PipelineResult, file_path: str) -> None:
        """Export analysis result to structured JSON file."""
        data = {
            "sigforge_version": "0.1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "processing_time_sec": result.processing_time_sec,
            "sample_rate": result.sample_rate,
            "signal_profile": {
                "snr_db": result.signal_profile.snr_db,
                "noise_floor_db": result.signal_profile.noise_floor_db,
                "occupied_bandwidth": result.signal_profile.occupied_bandwidth,
                "center_freq_offset": result.signal_profile.center_freq_offset,
                "kurtosis": result.signal_profile.kurtosis,
            } if result.signal_profile else {},
            "parameter_estimates": {
                "symbol_rate": result.parameter_estimates.symbol_rate,
                "samples_per_symbol": result.parameter_estimates.samples_per_symbol,
                "cfo_hz": result.parameter_estimates.cfo_hz,
                "symbol_rate_confidence": result.parameter_estimates.symbol_rate_confidence,
            } if result.parameter_estimates else {},
            "hypotheses": [h.to_dict() for h in result.hypothesis_graph.hypotheses],
            "fec_results": [
                {
                    "codec_type": f.codec_type,
                    "codec_id": f.codec_id,
                    "confidence": f.confidence,
                    "syndrome_zero": f.syndrome_zero,
                }
                for f in result.fec_results
            ],
            "warnings": result.warnings,
            "errors": result.errors,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    @staticmethod
    def to_html(result: PipelineResult, file_path: str, filename_hint: str = "Unknown") -> None:
        """Export analysis result to a polished, standalone HTML report."""
        best = result.hypothesis_graph.best
        best_desc = best.description if best else "Inconclusive"
        best_conf = f"{best.confidence:.1%}" if best else "0%"
        best_badge = best.badge.value if best else "GREY"
        best_color = best.badge.color_hex if best else "#9ca3af"

        # Hypotheses rows
        hyp_rows = ""
        for i, h in enumerate(result.hypothesis_graph.hypotheses, 1):
            hyp_rows += f"""
            <tr>
                <td><strong>{i}</strong></td>
                <td><span class="badge" style="background:{h.badge.color_hex}22; color:{h.badge.color_hex}; border: 1px solid {h.badge.color_hex}44;">{h.badge.emoji} {h.badge.value}</span></td>
                <td><strong>{h.description}</strong></td>
                <td>{h.symbol_rate:,.0f} Hz</td>
                <td>{h.confidence:.1%}</td>
                <td>{h.evm_db:.1f} dB</td>
                <td>{"✔ Validated" if h.has_hard_validation else "—"}</td>
            </tr>
            """

        # Evidence rows for best hypothesis
        ev_rows = ""
        if best:
            for ev in best.evidence:
                score_pct = int(ev.score * 100)
                ev_rows += f"""
                <tr>
                    <td><code>{ev.source}</code></td>
                    <td>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: {score_pct}%;"></div>
                        </div>
                        <span style="font-size:12px; color:#888;">{ev.score:.3f}</span>
                    </td>
                    <td>{ev.weight:.2f}</td>
                    <td>{ev.detail}</td>
                </tr>
                """

        # Bitstream hex preview
        hex_dump = ""
        if result.demod_results and result.demod_results[0].hard_bits is not None:
            bits = result.demod_results[0].hard_bits[:512]
            lines = []
            for b_idx in range(0, len(bits), 64):
                chunk = bits[b_idx:b_idx+64]
                hex_part = ""
                ascii_part = ""
                for byte_idx in range(0, len(chunk), 8):
                    byte_bits = chunk[byte_idx:byte_idx+8]
                    val = 0
                    for bit in byte_bits:
                        val = (val << 1) | int(bit)
                    hex_part += f"{val:02X} "
                    ascii_part += chr(val) if 32 <= val <= 126 else "."
                lines.append(f"{b_idx//8:04X}  {hex_part:<24} |{ascii_part}|")
            hex_dump = "\n".join(lines)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SigForge Analysis Report — {filename_hint}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 30px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid #30363d;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            color: #58a6ff;
            margin: 0;
            font-size: 26px;
        }}
        .meta-tag {{
            background: #161b22;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            color: #8b949e;
            border: 1px solid #30363d;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        h2 {{
            color: #f0f6fc;
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 16px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 8px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 16px;
        }}
        .stat-box {{
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #21262d;
        }}
        .stat-label {{
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: 600;
            color: #58a6ff;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            color: #8b949e;
            font-weight: 600;
        }}
        .badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
        }}
        .progress-bar-bg {{
            background: #21262d;
            border-radius: 4px;
            height: 8px;
            width: 100px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
        }}
        .progress-bar-fill {{
            background: #238636;
            height: 100%;
            border-radius: 4px;
        }}
        pre {{
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: monospace;
            font-size: 12px;
            color: #79c0ff;
            border: 1px solid #21262d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>SigForge Signal Analysis Report</h1>
                <div style="color: #8b949e; font-size: 13px; margin-top: 4px;">
                    Target: <strong>{filename_hint}</strong> &bull; Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC")}
                </div>
            </div>
            <div>
                <span class="meta-tag">Processed in {result.processing_time_sec:.2f}s</span>
            </div>
        </header>

        <!-- Executive Summary -->
        <div class="card" style="border-left: 4px solid {best_color};">
            <h2>Verdict: {best_desc}</h2>
            <div class="grid-4">
                <div class="stat-box">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value" style="color:{best_color};">{best_conf}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Estimated Symbol Rate</div>
                    <div class="stat-value">{(result.parameter_estimates.symbol_rate or 0)/1000:,.1f} kHz</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Carrier Frequency Offset</div>
                    <div class="stat-value">{result.parameter_estimates.cfo_hz or 0:+.1f} Hz</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Signal-to-Noise Ratio</div>
                    <div class="stat-value">{result.signal_profile.snr_db or 0:.1f} dB</div>
                </div>
            </div>
        </div>

        <!-- Hypothesis Ranking -->
        <div class="card">
            <h2>Hypothesis Ranking & Validation</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Status</th>
                        <th>Interpretation</th>
                        <th>Symbol Rate</th>
                        <th>Confidence</th>
                        <th>EVM</th>
                        <th>Hard Validation</th>
                    </tr>
                </thead>
                <tbody>
                    {hyp_rows}
                </tbody>
            </table>
        </div>

        <!-- Evidence Breakdown -->
        <div class="card">
            <h2>Evidence Breakdown (Best Hypothesis)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Support Score</th>
                        <th>Weight</th>
                        <th>Measurement Detail</th>
                    </tr>
                </thead>
                <tbody>
                    {ev_rows}
                </tbody>
            </table>
        </div>

        <!-- Bitstream Dump -->
        <div class="card">
            <h2>Decoded Bitstream Preview (First 512 bits)</h2>
            <pre>{hex_dump if hex_dump else "No bits decoded"}</pre>
        </div>
    </div>
</body>
</html>
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
