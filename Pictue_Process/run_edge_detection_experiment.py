"""Run the edge-detection mesh with calibrated voltages for every image vector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MNISTInputPassThroughTest as mnist
import upload_matrix as um
import utils.communication as cu


DEFAULT_FEATURES = SCRIPT_DIR / "prepared" / "04_single_port_input_coefficients_mzi8.csv"
DEFAULT_NETWORK_VOLTAGE = SCRIPT_DIR / "edge detection kernel" / "final_voltage.csv"
DEFAULT_CALIBRATION_ROOT = SCRIPT_DIR / "results" / "InputCalibration"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "EdgeDetectionExperiment"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--input-voltage-file", type=Path)
    parser.add_argument("--network-voltage-file", type=Path, default=DEFAULT_NETWORK_VOLTAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=5.5)
    parser.add_argument("--ser-address", default="COM3")
    parser.add_argument("--opm-address", default="TCPIP0::192.168.0.7::inst0::INSTR")
    parser.add_argument("--switch-current-check", type=parse_bool, default=True)
    parser.add_argument("--switch-current-tolerance", type=float, default=0.10)
    parser.add_argument("--max-upload-attempts", type=int, default=3)
    parser.add_argument("--input-reference-power-uw", type=float, default=120.0)
    # Kept as ignored compatibility options so older run commands still work.
    parser.add_argument("--max-output-total-uw", type=float, default=120.0, help=argparse.SUPPRESS)
    parser.add_argument("--output-power-tolerance-uw", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--fail-on-output-power-limit", type=parse_bool, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--off-voltage-tolerance-v", type=float, default=0.001)
    parser.add_argument("--repair-csv", type=Path)
    parser.add_argument(
        "--repair-sample-indices",
        help="Comma-separated sample_index values to remeasure and replace in --repair-csv.",
    )
    parser.add_argument("--repair-residual-threshold-uw", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--confirm-hardware", type=parse_bool, default=False)
    return parser


def find_latest_complete_voltage_file(features_csv: Path) -> Path:
    features = pd.read_csv(features_csv)
    expected_ids = set(pd.to_numeric(features["sample_index"], errors="raise").astype(int))
    candidates = sorted(
        DEFAULT_CALIBRATION_ROOT.glob("run_*/closed_loop_input_final_voltage.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        frame = pd.read_csv(path)
        required = {"sample_index", *(f"final_voltage_{idx}" for idx in range(8))}
        if required.issubset(frame.columns):
            ids = set(pd.to_numeric(frame["sample_index"], errors="raise").astype(int))
            if ids == expected_ids and len(frame) == len(features):
                return path.resolve()
    raise FileNotFoundError(
        f"No complete calibrated voltage file matching {len(features)} samples was found under "
        f"{DEFAULT_CALIBRATION_ROOT}. Use --input-voltage-file explicitly."
    )


def select_samples(features_csv: Path, voltage_file: Path, offset: int, limit: int | None) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    voltages = pd.read_csv(voltage_file)
    voltage_columns = [f"final_voltage_{idx}" for idx in range(8)]
    required_features = {"sample_index", *(f"feature_{idx}" for idx in range(8))}
    required_voltages = {"sample_index", *voltage_columns}
    if not required_features.issubset(features.columns):
        raise ValueError(f"{features_csv} is missing columns: {sorted(required_features - set(features.columns))}")
    if not required_voltages.issubset(voltages.columns):
        raise ValueError(f"{voltage_file} is missing columns: {sorted(required_voltages - set(voltages.columns))}")
    if voltages["sample_index"].duplicated().any():
        raise ValueError(f"{voltage_file} contains duplicate sample_index values.")

    offset = int(offset)
    if offset < 0:
        raise ValueError("--sample-offset must be >= 0.")
    features = features.iloc[offset:]
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--sample-limit must be positive.")
        features = features.head(int(limit))

    merged = features.merge(
        voltages[["sample_index", *voltage_columns]],
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    if merged[voltage_columns].isna().any().any():
        missing = merged.loc[merged[voltage_columns].isna().any(axis=1), "sample_index"].tolist()
        raise ValueError(f"Missing calibrated voltages for sample IDs: {missing[:20]}")
    if merged.empty:
        raise ValueError("No samples selected.")
    return merged.reset_index(drop=True)


def select_repair_samples(
    samples: pd.DataFrame,
    repair_csv: Path,
    voltage_file: Path,
    threshold_uw: float,
    explicit_sample_indices: str | None = None,
):
    previous = pd.read_csv(repair_csv)
    required = {
        "sample_index",
        "switch_current_failure_count",
        *(f"feature_{idx}" for idx in range(8)),
        *(f"output_power_uw_{idx}" for idx in range(8)),
    }
    if not required.issubset(previous.columns):
        raise ValueError(f"{repair_csv} is missing repair-analysis columns: {sorted(required - set(previous.columns))}")
    if previous["sample_index"].duplicated().any():
        raise ValueError(f"{repair_csv} contains duplicate sample_index values.")

    if explicit_sample_indices:
        try:
            requested = {int(value.strip()) for value in explicit_sample_indices.split(",") if value.strip()}
        except ValueError as exc:
            raise ValueError("--repair-sample-indices must be comma-separated integers.") from exc
        if not requested:
            raise ValueError("--repair-sample-indices did not contain any sample IDs.")
        available = set(samples["sample_index"].astype(int))
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested repair samples are absent from the selected feature range: {missing}")
        selected = samples.loc[samples["sample_index"].astype(int).isin(requested)].copy()
        selected["repair_reason"] = "explicit_power_anomaly_remeasure"
        reasons = {sample_id: {"explicit_power_anomaly_remeasure"} for sample_id in requested}
        return selected.sort_values("sample_index").reset_index(drop=True), previous, reasons

    feature_cols = [f"feature_{idx}" for idx in range(8)]
    power_cols = [f"output_power_uw_{idx}" for idx in range(8)]
    x = previous[feature_cols].to_numpy(dtype=float)
    y = previous[power_cols].to_numpy(dtype=float)
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    residual_mae_uw = np.mean(np.abs(y - fitted), axis=1)

    reasons: dict[int, set[str]] = {}
    for sample_id in previous.loc[previous["switch_current_failure_count"] > 0, "sample_index"]:
        reasons.setdefault(int(sample_id), set()).add("current_check_failure")
    for row_idx in np.flatnonzero(residual_mae_uw > float(threshold_uw)):
        sample_id = int(previous.iloc[int(row_idx)]["sample_index"])
        reasons.setdefault(sample_id, set()).add(
            f"power_residual_mae>{float(threshold_uw):g}uW"
        )

    calibration = pd.read_csv(voltage_file)
    if "sample_index" in calibration.columns and "converged" in calibration.columns:
        converged = calibration["converged"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        for sample_id in calibration.loc[~converged, "sample_index"]:
            reasons.setdefault(int(sample_id), set()).add("input_calibration_not_converged")

    if not reasons:
        raise ValueError("No problematic samples were found; repair run is unnecessary.")
    selected = samples.loc[samples["sample_index"].astype(int).isin(reasons)].copy()
    selected["repair_reason"] = selected["sample_index"].astype(int).map(
        lambda sample_id: ";".join(sorted(reasons[int(sample_id)]))
    )
    if len(selected) != len(reasons):
        missing = sorted(set(reasons) - set(selected["sample_index"].astype(int)))
        raise ValueError(f"Repair samples are absent from the selected feature range: {missing}")
    return selected.reset_index(drop=True), previous, reasons


def close_connection(connection) -> None:
    if connection is not None:
        try:
            connection.close()
        except Exception as exc:
            print(f"Warning: failed to close hardware connection: {exc}")


def validate_single_source_contract(samples, switch_rows, reference_power_uw, off_tolerance_v):
    feature_cols = [f"feature_{idx}" for idx in range(8)]
    voltage_cols = [f"final_voltage_{idx}" for idx in range(8)]
    features = samples[feature_cols].to_numpy(dtype=float)
    voltages = samples[voltage_cols].to_numpy(dtype=float)
    active = features > 1e-12
    active_counts = active.sum(axis=1)
    if np.any(active_counts != 1):
        bad = samples.loc[active_counts != 1, "sample_index"].astype(int).tolist()
        raise ValueError(f"Every sample must activate exactly one input source; invalid samples: {bad[:20]}")
    if np.any(features < -1e-12) or np.any(features > 1.0 + 1e-12):
        raise ValueError("All input coefficients must remain in the physical [0, 1] range.")
    target_input_uw = features.sum(axis=1) * float(reference_power_uw)
    if np.any(target_input_uw > float(reference_power_uw) + 1e-9):
        bad = samples.loc[target_input_uw > float(reference_power_uw) + 1e-9, "sample_index"].astype(int).tolist()
        raise ValueError(f"Target input power exceeds {reference_power_uw:g} uW: {bad[:20]}")

    off_voltages = np.asarray([float(switch_rows[idx + 1]["OFF"]) for idx in range(8)])
    inactive_voltage_error = np.abs(voltages - off_voltages.reshape(1, -1))
    bad_inactive = (~active) & (inactive_voltage_error > float(off_tolerance_v) + 1e-12)
    if np.any(bad_inactive):
        locations = np.argwhere(bad_inactive)
        details = [
            {
                "sample_index": int(samples.iloc[row]["sample_index"]),
                "input_channel": int(col + 1),
                "voltage": float(voltages[row, col]),
                "off_voltage": float(off_voltages[col]),
            }
            for row, col in locations[:20]
        ]
        raise ValueError(f"Inactive input MZIs are not at OFF voltage: {details}")
    return target_input_uw


def main() -> int:
    args = build_parser().parse_args()
    if not args.confirm_hardware:
        raise SystemExit("Refusing to upload voltages: add --confirm-hardware true.")
    if args.v_min > args.v_max:
        raise ValueError("--v-min must be <= --v-max.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    if args.max_upload_attempts <= 0:
        raise ValueError("--max-upload-attempts must be positive.")
    if args.repair_residual_threshold_uw <= 0:
        raise ValueError("--repair-residual-threshold-uw must be positive.")
    if args.resume_run_dir is not None and args.repair_csv is not None:
        raise ValueError("Use either --resume-run-dir or --repair-csv, not both.")
    if args.repair_sample_indices is not None and args.repair_csv is None:
        raise ValueError("--repair-sample-indices requires --repair-csv.")
    for name in ("input_reference_power_uw",):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.off_voltage_tolerance_v < 0:
        raise ValueError("OFF-voltage tolerance must be nonnegative.")

    features_csv = args.features_csv.resolve()
    network_voltage_file = args.network_voltage_file.resolve()
    input_voltage_file = (
        args.input_voltage_file.resolve()
        if args.input_voltage_file is not None
        else find_latest_complete_voltage_file(features_csv)
    )
    if not network_voltage_file.is_file():
        raise FileNotFoundError(f"Network voltage file not found: {network_voltage_file}")
    samples = select_samples(features_csv, input_voltage_file, args.sample_offset, args.sample_limit)
    previous_results = None
    repair_reasons = None
    if args.repair_csv is not None:
        repair_csv = args.repair_csv.resolve()
        samples, previous_results, repair_reasons = select_repair_samples(
            samples,
            repair_csv,
            input_voltage_file,
            args.repair_residual_threshold_uw,
            args.repair_sample_indices,
        )
        print(f"Repair mode selected {len(samples)} problematic samples from {repair_csv}.")

    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume-run-dir does not exist: {run_dir}")
    else:
        run_dir = args.out_dir.resolve() / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=False)
    output_csv = run_dir / "edge_detection_output_power.csv"
    previously_completed = set()
    if args.resume_run_dir is not None and output_csv.is_file():
        checkpoint = pd.read_csv(output_csv)
        if "sample_index" not in checkpoint.columns:
            raise ValueError(f"Resume CSV has no sample_index column: {output_csv}")
        previously_completed = set(checkpoint["sample_index"].astype(int))
        samples = samples.loc[~samples["sample_index"].astype(int).isin(previously_completed)].reset_index(drop=True)
        print(
            f"Resume checkpoint: {len(previously_completed)} samples already complete; "
            f"{len(samples)} remain."
        )
        if samples.empty:
            print(f"All samples are already complete: {output_csv}")
            return 0
    config = vars(args).copy()
    config.update(
        {
            "features_csv": str(features_csv),
            "input_voltage_file": str(input_voltage_file),
            "network_voltage_file": str(network_voltage_file),
            "sample_count": int(len(samples)),
            "repair_mode": bool(args.repair_csv is not None),
            "repair_reason_by_sample": (
                {str(key): sorted(value) for key, value in repair_reasons.items()}
                if repair_reasons is not None
                else None
            ),
            "output_csv": str(output_csv),
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    working_data = mnist.make_zero_working_data(128)
    mnist.load_voltage_state_file(network_voltage_file, working_data, args.v_min, args.v_max)
    mnist.set_all_switches(working_data, 8, "OFF")
    mnist.validate_all_voltages(working_data, args.v_min, args.v_max)
    switch_rows = mnist.load_switch_rows_by_mzi("IN_MZI.txt", 8, args.v_min, args.v_max)
    current_table = um.load_switch_mzi_table("IN_MZI.txt")
    target_input_uw = validate_single_source_contract(
        samples, switch_rows, args.input_reference_power_uw, args.off_voltage_tolerance_v
    )
    samples["target_input_power_uw"] = target_input_uw

    opm = None
    mcv = None
    wrote_header = output_csv.is_file() and output_csv.stat().st_size > 0
    completed = 0
    try:
        opm = cu.open_VISA_connection(args.opm_address)
        mcv = cu.open_ser_connection(args.ser_address)
        if opm is None or mcv is None:
            raise RuntimeError("Failed to open OPM or voltage-controller connection.")
        um.upload_v_checked(mcv, working_data, args.v_min, args.v_max)

        for local_idx, sample in samples.iterrows():
            voltages = np.asarray([sample[f"final_voltage_{idx}"] for idx in range(8)], dtype=float)
            mnist.set_switch_voltage_vector(working_data, switch_rows, voltages, args.v_min, args.v_max)
            mnist.validate_all_voltages(working_data, args.v_min, args.v_max)
            failures = []
            attempts_used = 0
            powers_w = None
            powers_uw = None
            for upload_attempt in range(1, int(args.max_upload_attempts) + 1):
                attempts_used = upload_attempt
                um.upload_v_checked(mcv, working_data, args.v_min, args.v_max)
                time.sleep(float(args.settle_time))
                if args.switch_current_check:
                    failures = um.verify_switch_mzi_currents(
                        mcv, working_data, current_table, tolerance=float(args.switch_current_tolerance)
                    )
                else:
                    failures = []
                if failures:
                    print(
                        f"sample {int(sample['sample_index'])}: current check failed on upload "
                        f"{upload_attempt}/{int(args.max_upload_attempts)} ({len(failures)} heaters).",
                        flush=True,
                    )
                    continue

                powers_w = mnist.read_opm_powers(opm, 8)
                powers_uw = np.asarray(powers_w, dtype=float) * 1e6
                break
            if powers_w is None or powers_uw is None:
                # Preserve a measurement even if every current check failed.
                powers_w = mnist.read_opm_powers(opm, 8)
                powers_uw = np.asarray(powers_w, dtype=float) * 1e6
            row = {
                "local_measurement_index": int(local_idx),
                "sample_index": int(sample["sample_index"]),
                "output_power_sum_w": float(np.sum(powers_w)),
                "output_power_sum_uw": float(np.sum(powers_uw)),
                "switch_current_failure_count": int(len(failures)),
                "upload_attempts": int(attempts_used),
                "target_input_power_uw": float(sample["target_input_power_uw"]),
                # Legacy columns retained so an interrupted older run can be resumed safely.
                # Output power is now recorded as measured and is never thresholded.
                "output_power_limit_uw": np.nan,
                "output_power_limit_exceeded": False,
                "repair_reason": str(sample.get("repair_reason", "")),
            }
            for idx in range(8):
                row[f"feature_{idx}"] = float(sample[f"feature_{idx}"])
                row[f"input_voltage_{idx}"] = float(voltages[idx])
                row[f"output_power_w_{idx}"] = float(powers_w[idx])
                row[f"output_power_uw_{idx}"] = float(powers_uw[idx])
            pd.DataFrame([row]).to_csv(output_csv, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            completed += 1
            if completed % int(args.print_every) == 0 or completed == len(samples):
                print(
                    f"[{completed}/{len(samples)}] sample={int(sample['sample_index'])}, "
                    f"output_uW={np.array2string(powers_uw, precision=3)}, "
                    f"current_failures={len(failures)}",
                    flush=True,
                )
    finally:
        if mcv is not None:
            try:
                mnist.set_all_switches(working_data, 8, "OFF")
                um.upload_v_checked(mcv, working_data, args.v_min, args.v_max)
                print("All input switches returned to OFF.")
            except Exception as exc:
                print(f"Warning: failed to return all input switches to OFF: {exc}")
        close_connection(opm)
        close_connection(mcv)

    print(f"Completed {completed} samples. Output CSV: {output_csv}")
    if previous_results is not None:
        repaired = pd.read_csv(output_csv)
        repaired_ids = set(repaired["sample_index"].astype(int))
        unchanged = previous_results.loc[~previous_results["sample_index"].astype(int).isin(repaired_ids)]
        merged = pd.concat([unchanged, repaired], ignore_index=True, sort=False)
        merged = merged.sort_values("sample_index").reset_index(drop=True)
        merged["upload_attempts"] = merged["upload_attempts"].fillna(1).astype(int)
        merged["repair_reason"] = merged["repair_reason"].fillna("")
        merged_path = run_dir / "edge_detection_output_power_merged.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Merged repaired 924-sample CSV: {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
