"""Calibrate image-derived input vectors with the MNIST closed-loop procedure.

The physical image CSV contains input1..input9 as port1..port9 columns. The
eight input switch MZIs map directly to physical inputs 1..8; input 9 is an
unused all-zero port. Each 0..1 coefficient is kept at its original scale:
1.0 means 120 uW and x means x * 120 uW at every input MZI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE = SCRIPT_DIR / "input" / "04_single_port_input_intensities_physical9.csv"
DEFAULT_PREPARED = SCRIPT_DIR / "prepared" / "04_single_port_input_coefficients_mzi8.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "InputCalibration"
PHYSICAL_PORT_COLUMNS = [f"port{port}_intensity" for port in range(1, 9)]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def prepare_input_coefficients(source_csv: Path, prepared_csv: Path) -> pd.DataFrame:
    source_csv = source_csv.resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(f"Input vector CSV not found: {source_csv}")

    frame = pd.read_csv(source_csv)
    required = ["unique_vector_id_zero_based", *PHYSICAL_PORT_COLUMNS, "port9_intensity"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_csv} is missing columns: {missing}")

    port9 = pd.to_numeric(frame["port9_intensity"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(port9).all():
        raise ValueError("port9_intensity contains non-finite values.")
    if not np.allclose(port9, 0.0, atol=1e-12, rtol=0.0):
        bad = np.flatnonzero(np.abs(port9) > 1e-12)[:10].tolist()
        raise ValueError(
            "Physical input 9 is ignored and therefore must remain zero; "
            f"nonzero rows include {bad}."
        )

    intensities = frame[PHYSICAL_PORT_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(intensities).all():
        bad = np.argwhere(~np.isfinite(intensities))[:10].tolist()
        raise ValueError(f"Physical input intensities contain non-finite values at {bad}.")
    if np.any(intensities < -1e-12):
        bad = np.argwhere(intensities < -1e-12)[:10].tolist()
        raise ValueError(f"Physical input intensities contain negative values at {bad}.")
    intensities = np.clip(intensities, 0.0, None)
    if np.any(intensities > 1.0 + 1e-12):
        bad = np.argwhere(intensities > 1.0 + 1e-12)[:10].tolist()
        raise ValueError(f"Physical input intensities exceed 1 at {bad}.")
    intensities = np.clip(intensities, 0.0, 1.0)

    active_counts = np.sum(intensities > 1e-12, axis=1)
    invalid_active = np.flatnonzero(active_counts != 1)
    if invalid_active.size:
        raise ValueError(
            "Every image vector must activate exactly one of physical inputs 1..8; "
            f"invalid rows include {invalid_active[:20].tolist()}."
        )

    intensity_sums = intensities.sum(axis=1)
    zero_rows = np.flatnonzero(intensity_sums <= 1e-15)
    if zero_rows.size:
        raise ValueError(f"Input vectors with zero total intensity: {zero_rows[:20].tolist()}")
    over_unity = np.flatnonzero(intensity_sums > 1.0 + 1e-12)
    if over_unity.size:
        raise ValueError(
            "Single-source input coefficient exceeds the 120-uW physical limit; "
            f"invalid rows include {over_unity[:20].tolist()}."
        )
    metadata_columns = [
        column
        for column in (
            "unique_vector_id_zero_based",
            "first_window_row",
            "first_window_col",
            "source_pixel_index_one_based",
            "physical_input_port_one_based",
            "occurrence_count",
        )
        if column in frame.columns
    ]
    prepared = frame[metadata_columns].copy()
    prepared.insert(0, "sample_index", frame["unique_vector_id_zero_based"].astype(int))
    prepared["source_intensity_sum"] = intensity_sums
    for index in range(8):
        prepared[f"feature_{index}"] = intensities[:, index]

    prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(prepared_csv, index=False)
    return prepared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set the 9-mode backend mesh to all-Bar, map physical input ports 1..8 to the eight "
            "input switch MZIs, map coefficient x to x*120 uW, and continuously correct input "
            "voltages until measured output powers match their absolute baseline targets."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepared-csv", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--retry-nonconverged", type=parse_bool, default=False)
    parser.add_argument("--prepare-only", type=parse_bool, default=False)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--input-upload-mode",
        choices=["scan-leak-inverse", "scan-leak-fixed-power"],
        default="scan-leak-fixed-power",
    )
    parser.add_argument("--input-reference-power-uw", type=float, default=120.0)
    parser.add_argument("--closed-loop-max-iters", type=int, default=20)
    parser.add_argument("--closed-loop-lr", type=float, default=0.35)
    parser.add_argument("--closed-loop-tol", type=float, default=0.02)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=5.5)
    parser.add_argument("--switch-current-check", type=parse_bool, default=True)
    parser.add_argument("--fail-on-switch-current-failure", type=parse_bool, default=False)
    parser.add_argument("--print-each-sample", type=parse_bool, default=True)
    parser.add_argument("--dry-run", type=parse_bool, default=True)
    parser.add_argument("--record-dry-run", type=parse_bool, default=False)
    parser.add_argument("--confirm-hardware", type=parse_bool, default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepared = prepare_input_coefficients(args.source_csv, args.prepared_csv)
    feature_columns = [f"feature_{index}" for index in range(8)]
    coefficients = prepared[feature_columns].to_numpy(dtype=float)
    print(f"Prepared {len(prepared)} vectors: {args.prepared_csv.resolve()}")
    print("Input MZI mapping: MZI1..MZI8 <- physical inputs 1..8; input 9 is ignored")
    print("Unified physical scale: coefficient 1.0 = 120 uW; x = x * 120 uW")
    print(f"Coefficient range: [{coefficients.min():.12g}, {coefficients.max():.12g}]")
    print("Strict input contract: exactly one active source per vector; total target input <= 120 uW")

    if args.prepare_only:
        print("Prepare-only mode completed; hardware was not accessed.")
        return 0
    if not args.dry_run and not args.confirm_hardware:
        raise RuntimeError("Hardware mode requires --confirm-hardware true.")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "MNISTInputPassThroughTest.py"),
        "--features-csv",
        str(args.prepared_csv.resolve()),
        "--out-dir",
        str(args.out_dir.resolve()),
        "--N",
        "9",
        "--network-mode",
        "pass-through",
        "--closed-loop-input",
        "true",
        "--closed-loop-objective",
        "absolute-total-power",
        "--measure-switch-on",
        "false",
        "--feature-validation-mode",
        "bounded",
        "--fail-on-invalid-feature",
        "false",
        "--input-upload-mode",
        args.input_upload_mode,
        "--input-reference-power-uw",
        str(args.input_reference_power_uw),
        "--closed-loop-max-iters",
        str(args.closed_loop_max_iters),
        "--closed-loop-lr",
        str(args.closed_loop_lr),
        "--closed-loop-tol",
        str(args.closed_loop_tol),
        "--settle-time",
        str(args.settle_time),
        "--v-min",
        str(args.v_min),
        "--v-max",
        str(args.v_max),
        "--switch-current-check",
        str(args.switch_current_check).lower(),
        "--fail-on-switch-current-failure",
        str(args.fail_on_switch_current_failure).lower(),
        "--print-each-sample",
        str(args.print_each_sample).lower(),
        "--dry-run",
        str(args.dry_run).lower(),
        "--record-dry-run",
        str(args.record_dry_run).lower(),
        "--confirm-hardware",
        str(args.confirm_hardware).lower(),
        "--run-inference",
        "false",
    ]
    if args.sample_limit is not None:
        command.extend(["--sample-limit", str(args.sample_limit)])
    if args.sample_offset:
        command.extend(["--sample-offset", str(args.sample_offset)])
    if args.resume_run_dir is not None:
        command.extend(["--resume-run-dir", str(args.resume_run_dir.resolve())])
    if args.retry_nonconverged:
        if args.resume_run_dir is None:
            raise ValueError("--retry-nonconverged true requires --resume-run-dir.")
        command.extend(["--retry-nonconverged", "true"])

    print("Starting all-Bar fixed-120-uW input calibration pass...")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
