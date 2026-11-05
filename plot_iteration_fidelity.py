"""Preview full 8x8 fidelity against a fixed final target, without saving files.

Run from an IDE or terminal:
    python -B plot_iteration_fidelity.py

The default run is DataUsedinPaper/experiment_results/Unitary_6x6.
Its 30 P_current CSV files are iteration-START measurements across layers 2-4,
verified against cosine_before_iter. Its final_power_matrix.csv is verified
against the last iteration-end metric and appended as a labeled final endpoint.
The loader supports both single-layer and multilayer CSV formats and checks
snapshot timing against the recorded loss or cosine metrics.

The fidelity comparison uses ALL 64 entries of the measured 8x8 power matrix
and the fixed final 8x8 target: physical inputs and outputs 1..8. Any configured
optimization subregion is used only to validate the log's snapshot timing.
This script calculates classical power-distribution fidelity:

    F = (sum(sqrt(P * Q)))**2 / (sum(P) * sum(Q)).

This is squared Bhattacharyya overlap of the normalized, flattened 8x8
matrices; it does not assess measured phases or absolute transmission.
Definition: https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#hellinger_fidelity

No plots, tables, or logs are saved. The default opens an interactive preview.
--preview-base64 emits an in-memory PNG to stdout for an embedded preview.
"""

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent
    / "DataUsedinPaper" / "experiment_results" / "Unitary_6x6"
)


def power_fidelity(current, reference):
    """Classical fidelity, invariant under positive overall power scaling."""
    current = np.asarray(current, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if current.shape != reference.shape or current.ndim != 2:
        raise ValueError("Current and reference must have matching 2-D shapes")
    distributions = []
    for matrix in (current, reference):
        if not np.isfinite(matrix).all() or np.any(matrix < -1e-12):
            raise ValueError("Power matrices must be finite and nonnegative")
        matrix = np.maximum(matrix, 0.0)
        total = float(matrix.sum())
        if total <= 0:
            raise ValueError("Fidelity is undefined for an all-zero power block")
        distributions.append(matrix / total)
    overlap = np.sqrt(distributions[0] * distributions[1]).sum()
    return float(np.clip(overlap**2, 0.0, 1.0))


def select_block(matrix, row_start, col_start, size):
    """Select a square region using physical channel numbers (1-based)."""
    row, col = int(row_start) - 1, int(col_start) - 1
    if matrix.ndim != 2 or row < 0 or col < 0 or size < 1:
        raise ValueError("Invalid matrix shape or selected region")
    if row + size > matrix.shape[0] or col + size > matrix.shape[1]:
        raise ValueError("Selected region extends beyond the matrix")
    return matrix[row:row + size, col:col + size]


def matches_log(matrix, stage_target, record, config, timing):
    """Use the original log's metric/region solely to identify snapshot timing."""
    size = config.get("focus_size")
    if size is not None:
        row = int(config.get("focus_row_start", 1))
        col = int(config.get("focus_col_start", 1))
        matrix = select_block(matrix, row, col, int(size))
        stage_target = select_block(stage_target, row, col, int(size))
    cosine_key = "cosine_before_iter" if timing == "start" else "cosine_after_iter"
    if record.get(cosine_key):
        norm = float(np.linalg.norm(matrix) * np.linalg.norm(stage_target))
        if not np.isfinite(norm) or norm <= 0:
            return False
        measured = float(np.vdot(matrix, stage_target).real / norm)
        expected = float(record[cosine_key])
    else:
        legacy = "iter" in record
        key = ("loss" if timing == "start" else "accepted_loss") if legacy else (
            "loss_before_iter" if timing == "start" else "loss_after_iter"
        )
        if not record.get(key):
            return False
        measured = float(np.square(matrix - stage_target).sum() / (np.square(stage_target).sum() + 1e-12))
        expected = float(record[key])
    return bool(np.isclose(measured, expected, atol=1e-9, rtol=0))


def load_history(data_dir=DEFAULT_DATA_DIR, reference="target"):
    data_dir = Path(data_dir)
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    legacy_target_path = data_dir / "target_power_matrix.csv"
    if reference == "target":
        reference_path = legacy_target_path if legacy_target_path.exists() else (
            data_dir / f"target_power_matrix_layer{int(config['layer']):02d}.csv"
        )
        reference_label = "final theoretical target"
    elif reference == "final-measured":
        reference_path = data_dir / "final_power_matrix.csv"
        reference_label = "final measured matrix"
    else:
        raise ValueError(f"Unknown reference: {reference}")
    target = np.loadtxt(reference_path, delimiter=",", ndmin=2)
    if target.shape != (8, 8):
        raise ValueError(f"{reference_path}: expected a complete 8x8 reference matrix")
    rows, seen = [], set()
    timing_candidates = {"start", "end"}
    with (data_dir / "iter_log.csv").open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            legacy = "iter" in record
            if not legacy and record["layer_status"] != "optimized":
                continue
            layer = int(config["layer"]) if legacy else int(float(record["current_layer"]))
            iteration = int(float(record["iter"] if legacy else record["layer_iter"]))
            key = (layer, iteration)
            if key in seen:
                raise ValueError(f"Repeated iteration {key}; cannot identify its unique saved matrix")
            seen.add(key)
            path = data_dir / (f"P_current_iter{iteration:03d}.csv" if legacy else
                               f"P_current_layer{layer:02d}_iter{iteration:03d}.csv")
            full_current = np.loadtxt(path, delimiter=",")
            stage_target_path = legacy_target_path if legacy else data_dir / f"target_power_matrix_layer{layer:02d}.csv"
            stage_target = np.loadtxt(stage_target_path, delimiter=",")
            if full_current.shape != (8, 8) or full_current.shape != stage_target.shape:
                raise ValueError(f"{path}: shape differs from the corresponding layer target")
            timing_candidates &= {timing for timing in ("start", "end")
                                  if matches_log(full_current, stage_target, record, config, timing)}
            if not timing_candidates:
                raise ValueError(f"{path}: snapshot timing is inconsistent with the iteration log")
            rows.append({
                "global_iteration": len(rows) + 1,
                "layer": layer,
                "layer_iteration": iteration,
                "fidelity": power_fidelity(full_current, target),
                "matrix_path": path,
                "experiment": data_dir.name,
            })
            last_record, last_stage_target = record, stage_target
    if not rows:
        raise ValueError(f"No iteration snapshots found in {data_dir}")
    if len(timing_candidates) != 1:
        raise ValueError("The log does not uniquely identify iteration-start or iteration-end snapshots")
    timing = timing_candidates.pop()
    for row in rows:
        row["snapshot_timing"] = timing
    final_path = data_dir / "final_power_matrix.csv"
    if timing == "start" and final_path.exists():
        final = np.loadtxt(final_path, delimiter=",")
        if final.shape != (8, 8) or not matches_log(final, last_stage_target, last_record, config, "end"):
            raise ValueError("Final matrix does not match the last iteration-end metric in the log")
        rows.append({
            **rows[-1], "global_iteration": len(rows) + 1,
            "fidelity": power_fidelity(final, target), "matrix_path": final_path,
            "snapshot_timing": "final",
        })
    return rows, reference_path, reference_label


def create_figure(rows, reference_label):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator

    x = np.array([row["global_iteration"] for row in rows])
    y = np.array([row["fidelity"] for row in rows])
    layers = np.array([row["layer"] for row in rows])
    timing = rows[0]["snapshot_timing"]
    has_final = rows[-1]["snapshot_timing"] == "final"
    snapshot_count = len(rows) - int(has_final)
    experiment = rows[0]["experiment"]
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BCC7D5", "axes.labelcolor": "#334155",
        "xtick.color": "#64748B", "ytick.color": "#64748B",
    }):
        fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
        fig.canvas.manager.set_window_title(f"Full 8x8 fidelity - {experiment}")
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.10, right=0.96, top=0.77, bottom=0.22)
        fig.text(0.10, 0.94, "Full 8 x 8 matrix: fidelity through optimization",
                 fontsize=19, fontweight="bold", color="#0F172A")
        fig.text(0.10, 0.885,
                 f"{experiment}  |  Fixed {reference_label}",
                 fontsize=10.5, color="#64748B")

        colors = ["#2563EB", "#0D9488", "#7C3AED", "#D97706", "#DB2777", "#0284C7"]
        ax.plot(x, y, color="#64748B", linewidth=1.5, zorder=2)
        boundaries = np.r_[0, np.flatnonzero(np.diff(layers)) + 1, len(rows)]
        for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            color = colors[index % len(colors)]
            ax.axvspan(x[start] - 0.5, x[stop - 1] + 0.5, color=color, alpha=0.045)
            ax.plot(x[start:stop], y[start:stop], color=color, linewidth=2.2,
                    marker="o", markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3)
            midpoint = (x[start] + x[stop - 1]) / 2
            count = sum(row["snapshot_timing"] != "final" for row in rows[start:stop])
            description = f"Layer {layers[start]}\n{count} {timing} snapshots"
            if has_final and stop == len(rows):
                description += " + final"
            ax.text(midpoint, 1.045, description,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=color)
            if start:
                ax.axvline(x[start] - 0.5, color="#94A3B8", linestyle="--", linewidth=1)

        ax.annotate(f"First  {y[0]:.2%}", xy=(x[0], y[0]), xytext=(12, -28),
                    textcoords="offset points", fontsize=11, color="#2563EB")
        if has_final:
            ax.scatter(x[-1], y[-1], marker="D", s=65, color="#0D9488",
                       edgecolor="white", linewidth=0.8, zorder=5)
        endpoint_label = "Final" if has_final or timing == "end" else "Latest start"
        ax.annotate(f"{endpoint_label}  {y[-1]:.2%}", xy=(x[-1], y[-1]), xytext=(-8, -32),
                    textcoords="offset points", ha="right", fontsize=12,
                    fontweight="bold", color="#0F766E",
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F0FDFA", "edgecolor": "#99F6E4"})
        ax.set_xlim(0.5, len(rows) + 0.8)
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if len(rows) <= 12 and len(set(layers)) == 1:
            ax.set_xticks(x, ["Final\nend" if row["snapshot_timing"] == "final" else
                             f"{row['layer_iteration']}\n{timing}" for row in rows])
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.set_xlabel("Snapshot index across layers (last point: final measurement)" if has_final else
                      f"Iteration-{timing} snapshot (all layers, in recorded order)", labelpad=12)
        ax.set_ylabel("Power-distribution fidelity", labelpad=12)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.text(0.10, 0.075,
                 r"$F=(\sum_{ij}\sqrt{P_{ij}Q_{ij}})^2\,/\,[(\sum_{ij}P_{ij})(\sum_{ij}Q_{ij})]$",
                 fontsize=12, color="#334155")
        fig.text(0.10, 0.035,
                 f"Exact CSVs: {snapshot_count} iteration-{timing} matrices"
                 + (" + separately labeled final endpoint; " if has_final else "; ")
                 + "all 64 entries, physical inputs and outputs 1-8.",
                 fontsize=8.5, color="#64748B")
        fig.text(0.10, 0.01, "Normalized full 8 x 8 powers; measured phase and absolute transmission are not assessed.",
                 fontsize=8.5, color="#64748B")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference", choices=("target", "final-measured"), default="target")
    parser.add_argument("--preview-base64", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preview_base64:
        import matplotlib
        matplotlib.use("Agg")
    rows, reference_path, reference_label = load_history(args.data_dir, args.reference)
    fig = create_figure(rows, reference_label)
    import matplotlib.pyplot as plt
    if args.preview_base64:
        # Render straight to memory: no PNG file or temporary image is created.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        buffer = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buffer)
        print(base64.b64encode(buffer.getvalue()).decode("ascii"))
        plt.close(fig)
    else:
        print(f"Reference: {reference_path}")
        print(f"Exact snapshots: {len(rows)}; timing: {rows[0]['snapshot_timing']}; last point: {rows[-1]['snapshot_timing']}")
        print(f"First fidelity: {rows[0]['fidelity']:.6%}")
        print(f"Last plotted fidelity: {rows[-1]['fidelity']:.6%}")
        print("Preview only; no files are saved.")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
