import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import upload_matrix as um
import utils.communication as cu


DEFAULT_SER_ADDRESS = "COM3"
DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
DEFAULT_INFERENCE_VOLTAGE_FILE = (
    r"results\LayerwiseLossOptimize\run_20260716_224053\final_voltage.csv"
)
DEFAULT_SWITCH_LEAK_SCAN_DIR = r"Scandata-backup-04\1Col\power_data"
DEFAULT_INPUT_REFERENCE_POWER_UW = 120.0


def is_fixed_reference_power_mode(mode):
    return str(mode).strip().lower() == "scan-leak-fixed-power"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def create_run_dir(out_dir):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def should_record_results(args):
    return bool(getattr(args, "record_results", True))


def result_path(args, run_dir, filename):
    if not should_record_results(args) or run_dir is None:
        return None
    return os.fspath(run_dir / filename)


def atomic_write_rows(path, rows):
    """Checkpoint rows without leaving a partially written CSV destination."""
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def load_checkpoint_rows(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def make_zero_working_data(channel_count=128):
    return pd.DataFrame({0: np.zeros(int(channel_count), dtype=float)})


def validate_all_voltages(working_data, v_min, v_max):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        bad = voltages[voltages.isna()].index.tolist()
        raise ValueError(f"working_data contains non-numeric voltages at rows: {bad}")
    bad_mask = (voltages < float(v_min)) | (voltages > float(v_max))
    if bad_mask.any():
        bad_rows = []
        for idx in voltages[bad_mask].index.tolist():
            bad_rows.append(
                {
                    "row": int(idx),
                    "port": int(idx) + 1,
                    "voltage": float(voltages.iloc[idx]),
                }
            )
        raise ValueError(f"Voltage out of range [{v_min}, {v_max}]: {bad_rows}")


def assert_voltage_in_range(label, voltage, v_min, v_max):
    voltage = float(voltage)
    if not np.isfinite(voltage):
        raise ValueError(f"{label} voltage is not finite: {voltage!r}")
    if voltage < float(v_min) or voltage > float(v_max):
        raise ValueError(f"{label} voltage {voltage:.6f} V is out of safe range [{v_min}, {v_max}] V")
    return voltage


def save_voltage_state(path, working_data):
    if path is None:
        return
    pd.DataFrame(
        {
            "port": np.arange(1, len(working_data) + 1),
            "voltage": working_data.iloc[:, 0].to_numpy(dtype=float),
        }
    ).to_csv(path, index=False)


def load_voltage_state_file(path, working_data, v_min, v_max):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Voltage state file not found: {path}")
    df = pd.read_csv(path)
    loaded_rows = []

    if {"port", "voltage"}.issubset(df.columns):
        for _, row in df.iterrows():
            port = int(row["port"])
            voltage = assert_voltage_in_range(f"{path} port {port}", row["voltage"], v_min, v_max)
            cu.write_port_voltage(port, voltage, working_data)
            loaded_rows.append({"port": int(port), "voltage": round(float(voltage), 3)})
    else:
        if "voltage" in df.columns:
            values = pd.to_numeric(df["voltage"], errors="coerce").to_numpy(dtype=float)
        else:
            values = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        if len(values) > len(working_data):
            raise ValueError(f"{path} has {len(values)} voltages, working_data has {len(working_data)} rows.")
        for idx, voltage in enumerate(values):
            port = int(idx + 1)
            voltage = assert_voltage_in_range(f"{path} port {port}", voltage, v_min, v_max)
            cu.write_port_voltage(port, voltage, working_data)
            loaded_rows.append({"port": port, "voltage": round(float(voltage), 3)})

    validate_all_voltages(working_data, v_min, v_max)
    return loaded_rows


def load_features(features_csv, input_count, sample_limit=None, sample_offset=0):
    df = pd.read_csv(features_csv)
    feature_cols = [f"feature_{idx}" for idx in range(input_count)]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{features_csv} missing feature columns: {missing}")

    sample_offset = int(sample_offset)
    if sample_offset < 0:
        raise ValueError("--sample-offset must be >= 0.")
    if sample_offset:
        df = df.iloc[sample_offset:].copy()

    if sample_limit is not None:
        if int(sample_limit) <= 0:
            raise ValueError("--sample-limit must be positive when provided.")
        df = df.head(int(sample_limit)).copy()

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return df, feature_cols, features


def validate_features(features, sum_tol=1e-3, negative_tol=1e-12, mode="normalized"):
    mode = str(mode).strip().lower()
    if mode not in {"normalized", "bounded"}:
        raise ValueError("feature validation mode must be 'normalized' or 'bounded'.")
    sums = np.sum(features, axis=1)
    finite = np.isfinite(features).all(axis=1)
    nonnegative = np.min(features, axis=1) >= -float(negative_tol)
    upper_bound_ok = np.max(features, axis=1) <= 1.0 + float(negative_tol)
    nonzero = sums > float(negative_tol)
    normalized_sum_ok = np.abs(sums - 1.0) <= float(sum_tol)
    if mode == "normalized":
        sum_ok = normalized_sum_ok
        valid = finite & nonnegative & sum_ok
    else:
        # Absolute-intensity vectors need not sum to one. Each coefficient is
        # independently meaningful on the physical [0, 1] scale.
        sum_ok = np.ones(len(sums), dtype=bool)
        valid = finite & nonnegative & upper_bound_ok & nonzero
    return {
        "valid": valid,
        "finite": finite,
        "nonnegative": nonnegative,
        "sum_ok": sum_ok,
        "normalized_sum_ok": normalized_sum_ok,
        "upper_bound_ok": upper_bound_ok,
        "nonzero": nonzero,
        "mode": mode,
        "sums": sums,
        "min_values": np.min(features, axis=1),
        "max_values": np.max(features, axis=1),
    }


def set_network_to_bar_pass_through(working_data, mzi_table, mzi_count, v_min, v_max):
    records = []
    for mzi_id in range(1, int(mzi_count) + 1):
        entry = um.get_mzi_entry(mzi_table, mzi_id)
        ports = entry.get("ports", [])
        if not ports:
            raise ValueError(f"MZI {mzi_id} has no ports in MZI table.")

        upper_port = int(ports[0])
        upper_voltage = float(um.get_mzi_state_voltage(mzi_table, mzi_id, "BAR", arm_index=0))
        assert_voltage_in_range(f"MZI{mzi_id}_arm0_BAR port {upper_port}", upper_voltage, v_min, v_max)
        cu.write_port_voltage(upper_port, upper_voltage, working_data)
        records.append(
            {
                "mzi_id": int(mzi_id),
                "arm_index": 0,
                "port": upper_port,
                "state": "BAR_PASS_THROUGH",
                "voltage": round(upper_voltage, 3),
            }
        )

        if len(ports) > 1:
            lower_port = int(ports[1])
            lower_voltage = 0.0
            assert_voltage_in_range(f"MZI{mzi_id}_arm1_ZERO port {lower_port}", lower_voltage, v_min, v_max)
            cu.write_port_voltage(lower_port, lower_voltage, working_data)
            records.append(
                {
                    "mzi_id": int(mzi_id),
                    "arm_index": 1,
                    "port": lower_port,
                    "state": "LOWER_ARM_ZERO",
                    "voltage": lower_voltage,
                }
            )

    validate_all_voltages(working_data, v_min, v_max)
    return records


def set_all_switches(working_data, input_count, state):
    records = []
    for switch_id in range(1, int(input_count) + 1):
        voltage = um.switch_IN(switch_id, state, working_data)
        records.append(
            {
                "switch_mzi": int(switch_id),
                "state": str(state).upper(),
                "voltage": float(voltage),
            }
        )
    return records


def validate_switch_voltage_table(table, path, input_count, v_min, v_max):
    checked_rows = []
    for _, row in table.iterrows():
        mzi = int(row["MZI"])
        if not (1 <= mzi <= int(input_count)):
            continue
        port = int(row["PORT"])
        on_v = assert_voltage_in_range(f"{path} MZI{mzi} ON port {port}", row["ON"], v_min, v_max)
        off_v = assert_voltage_in_range(f"{path} MZI{mzi} OFF port {port}", row["OFF"], v_min, v_max)
        checked_rows.append(
            {
                "switch_mzi": int(mzi),
                "port": int(port),
                "on_voltage": float(on_v),
                "off_voltage": float(off_v),
                "safe_range_min": float(v_min),
                "safe_range_max": float(v_max),
            }
        )
    return checked_rows


def load_switch_rows_by_mzi(path, input_count, v_min, v_max):
    table = um.load_switch_mzi_table(path)
    validate_switch_voltage_table(table, path, input_count, v_min, v_max)
    rows = {}
    for _, row in table.iterrows():
        mzi = int(row["MZI"])
        if 1 <= mzi <= int(input_count):
            rows[mzi] = row
    missing = [idx for idx in range(1, int(input_count) + 1) if idx not in rows]
    if missing:
        raise ValueError(f"{path} missing switch MZI rows: {missing}")
    return rows


def read_switch_leak_scan(scan_path):
    scan_path = Path(scan_path)
    if not scan_path.exists():
        raise FileNotFoundError(f"Cannot find switch leak scan file: {scan_path}")
    df = pd.read_csv(scan_path)
    lower_cols = {str(col).strip().lower(): col for col in df.columns}

    voltage_col = None
    for key in ("v", "voltage_v", "voltage"):
        if key in lower_cols:
            voltage_col = lower_cols[key]
            break
    if voltage_col is None:
        for col in df.columns:
            if str(col).strip().lower().startswith("v"):
                voltage_col = col
                break

    power_col = None
    for col in df.columns:
        if "pow" in str(col).strip().lower():
            power_col = col
            break

    if voltage_col is None or power_col is None:
        raise ValueError(f"{scan_path} must contain voltage and power columns, got {list(df.columns)}")

    voltage = pd.to_numeric(df[voltage_col], errors="coerce").to_numpy(dtype=float)
    leak_power = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(voltage) & np.isfinite(leak_power)
    voltage = voltage[valid]
    leak_power = leak_power[valid]
    if voltage.size < 2:
        raise ValueError(f"{scan_path} has fewer than two valid scan points.")

    order = np.argsort(voltage)
    return {
        "path": os.fspath(scan_path),
        "voltage": voltage[order],
        "leak_power": leak_power[order],
        "voltage_col": str(voltage_col),
        "power_col": str(power_col),
    }


def build_switch_leak_inverse_table_for_row(row, scan_dir, v_min, v_max):
    mzi = int(row["MZI"])
    port = int(row["PORT"])
    v_on = assert_voltage_in_range(f"switch MZI{mzi} ON", row["ON"], v_min, v_max)
    v_off = assert_voltage_in_range(f"switch MZI{mzi} OFF", row["OFF"], v_min, v_max)
    if np.isclose(v_on, v_off):
        raise ValueError(f"switch MZI{mzi} ON and OFF voltages are identical; cannot build inverse lookup.")

    scan = read_switch_leak_scan(Path(scan_dir) / f"{port}.txt")
    voltage = scan["voltage"]
    leak_power = scan["leak_power"]
    if v_on < float(np.min(voltage)) or v_on > float(np.max(voltage)):
        raise ValueError(f"switch MZI{mzi} ON voltage {v_on:.6f} is outside scan voltage range.")
    if v_off < float(np.min(voltage)) or v_off > float(np.max(voltage)):
        raise ValueError(f"switch MZI{mzi} OFF voltage {v_off:.6f} is outside scan voltage range.")

    p_on = float(np.interp(v_on, voltage, leak_power))
    p_off = float(np.interp(v_off, voltage, leak_power))
    if not np.isfinite(p_on) or not np.isfinite(p_off):
        raise ValueError(f"switch MZI{mzi} ON/OFF leak power interpolation failed.")
    if p_off <= p_on:
        raise ValueError(
            f"switch MZI{mzi} leak scan direction invalid: expected OFF leak power > ON leak power, "
            f"got P_OFF={p_off:.12g}, P_ON={p_on:.12g}."
        )

    v_lo = min(v_on, v_off)
    v_hi = max(v_on, v_off)
    branch_mask = (voltage >= v_lo) & (voltage <= v_hi)
    branch_v = voltage[branch_mask]
    branch_p = leak_power[branch_mask]
    order = np.argsort(branch_v)
    branch_v = branch_v[order]
    branch_p = branch_p[order]

    # The scan measured the other/leak port:
    #   feature=1 means ON -> minimum leak -> light enters the chip.
    #   feature=0 means OFF -> maximum leak -> light is rejected/leaked.
    raw_fraction = (p_off - branch_p) / (p_off - p_on)
    finite_internal = (
        np.isfinite(raw_fraction)
        & np.isfinite(branch_v)
        & np.isfinite(branch_p)
        & (raw_fraction > 0.0)
        & (raw_fraction < 1.0)
    )
    # Put exact ON/OFF anchors first. Stable sorting + unique keeps these anchors
    # when nearby scan points clip to the same endpoint coefficient.
    chip_fraction = np.concatenate(
        [
            np.array([0.0, 1.0], dtype=float),
            raw_fraction[finite_internal].astype(float),
        ]
    )
    lookup_voltage = np.concatenate(
        [
            np.array([v_off, v_on], dtype=float),
            branch_v[finite_internal].astype(float),
        ]
    )
    lookup_leak = np.concatenate(
        [
            np.array([p_off, p_on], dtype=float),
            branch_p[finite_internal].astype(float),
        ]
    )
    if chip_fraction.size < 2:
        raise ValueError(f"switch MZI{mzi} has too few valid inverse lookup points.")

    frac_order = np.argsort(chip_fraction, kind="mergesort")
    frac_sorted = chip_fraction[frac_order]
    voltage_sorted = lookup_voltage[frac_order]
    leak_sorted = lookup_leak[frac_order]
    unique_frac, unique_idx = np.unique(np.round(frac_sorted, 12), return_index=True)
    unique_voltage = voltage_sorted[unique_idx]
    unique_leak = leak_sorted[unique_idx]
    if unique_frac.size < 2:
        raise ValueError(f"switch MZI{mzi} inverse lookup collapsed to fewer than two unique feature points.")

    return {
        "mzi": int(mzi),
        "port": int(port),
        "scan_path": scan["path"],
        "v_on": float(v_on),
        "v_off": float(v_off),
        "p_on_leak": float(p_on),
        "p_off_leak": float(p_off),
        "chip_power_max_uw": float(p_off - p_on),
        "feature_points": unique_frac.astype(float),
        "voltage_points": unique_voltage.astype(float),
        "leak_power_points": unique_leak.astype(float),
        "branch_point_count": int(len(branch_v)),
        "voltage_min": float(np.min(unique_voltage)),
        "voltage_max": float(np.max(unique_voltage)),
    }


def build_switch_leak_inverse_tables(switch_rows_by_mzi, scan_dir, input_count, v_min, v_max):
    tables = {}
    summary_rows = []
    for mzi in range(1, int(input_count) + 1):
        entry = build_switch_leak_inverse_table_for_row(
            switch_rows_by_mzi[mzi],
            scan_dir,
            v_min,
            v_max,
        )
        tables[int(mzi)] = entry
        summary_rows.append(
            {
                "switch_mzi": int(mzi),
                "port": int(entry["port"]),
                "scan_path": entry["scan_path"],
                "v_on": entry["v_on"],
                "v_off": entry["v_off"],
                "p_on_leak": entry["p_on_leak"],
                "p_off_leak": entry["p_off_leak"],
                "chip_power_max_uw": entry["chip_power_max_uw"],
                "branch_point_count": entry["branch_point_count"],
                "lookup_voltage_min": entry["voltage_min"],
                "lookup_voltage_max": entry["voltage_max"],
            }
        )
    return tables, summary_rows


def build_switch_leak_inverse_lookup_rows(tables, step=0.01):
    step = float(step)
    if step <= 0.0 or step > 1.0:
        raise ValueError("--switch-leak-lookup-step must be in (0, 1].")
    coeffs = np.arange(0.0, 1.0 + step * 0.5, step, dtype=float)
    coeffs = np.unique(np.clip(np.concatenate([coeffs, np.array([1.0])]), 0.0, 1.0))
    rows = []
    for mzi in sorted(tables):
        table = tables[mzi]
        for coeff in coeffs:
            voltage = float(np.interp(coeff, table["feature_points"], table["voltage_points"]))
            target_leak_power = float(table["p_off_leak"] - coeff * (table["p_off_leak"] - table["p_on_leak"]))
            target_chip_power_uw = float(coeff * table["chip_power_max_uw"])
            rows.append(
                {
                    "switch_mzi": int(mzi),
                    "port": int(table["port"]),
                    "feature_coefficient": float(coeff),
                    "upload_voltage": round(voltage, 3),
                    "target_chip_power_uw": target_chip_power_uw,
                    "target_leak_power": target_leak_power,
                    "p_on_leak": float(table["p_on_leak"]),
                    "p_off_leak": float(table["p_off_leak"]),
                    "chip_power_max_uw": float(table["chip_power_max_uw"]),
                    "scan_path": table["scan_path"],
                }
            )
    return rows


def validate_fixed_reference_power(tables, reference_power_uw):
    reference_power_uw = float(reference_power_uw)
    if not np.isfinite(reference_power_uw) or reference_power_uw <= 0.0:
        raise ValueError(f"--input-reference-power-uw must be positive and finite, got {reference_power_uw!r}")

    rows = []
    unreachable = []
    for mzi in sorted(tables):
        table = tables[mzi]
        chip_power_max_uw = float(table["chip_power_max_uw"])
        ok = bool(chip_power_max_uw + 1e-9 >= reference_power_uw)
        row = {
            "switch_mzi": int(mzi),
            "port": int(table["port"]),
            "input_reference_power_uw": reference_power_uw,
            "chip_power_max_uw": chip_power_max_uw,
            "reference_fraction_of_max": float(reference_power_uw / chip_power_max_uw),
            "reference_reachable": ok,
            "scan_path": table["scan_path"],
        }
        rows.append(row)
        if not ok:
            unreachable.append(row)

    if unreachable:
        raise ValueError(
            "Requested fixed input reference power is not reachable by all switch MZIs: "
            f"{unreachable}"
        )
    return rows


def switch_voltage_from_scan_leak_inverse(row, coefficient, inverse_table, v_min, v_max):
    coeff = float(np.clip(float(coefficient), 0.0, 1.0))
    mzi = int(row["MZI"])
    if inverse_table is None or mzi not in inverse_table:
        raise ValueError(f"No switch leak inverse table for MZI{mzi}.")
    table = inverse_table[mzi]
    voltage = float(np.interp(coeff, table["feature_points"], table["voltage_points"]))
    target_leak_power = float(table["p_off_leak"] - coeff * (table["p_off_leak"] - table["p_on_leak"]))
    assert_voltage_in_range(
        f"scan inverse feature coefficient {coeff:.6g} mapped switch MZI{mzi} port {int(row['PORT'])}",
        voltage,
        v_min,
        v_max,
    )
    return round(voltage, 3), {
        "target_leak_power": target_leak_power,
        "target_chip_power_uw": float(coeff * table["chip_power_max_uw"]),
        "p_on_leak": float(table["p_on_leak"]),
        "p_off_leak": float(table["p_off_leak"]),
        "chip_power_max_uw": float(table["chip_power_max_uw"]),
        "scan_path": table["scan_path"],
    }


def switch_voltage_from_scan_leak_fixed_power(
    row,
    coefficient,
    inverse_table,
    input_reference_power_uw,
    v_min,
    v_max,
):
    coeff = float(np.clip(float(coefficient), 0.0, 1.0))
    mzi = int(row["MZI"])
    if inverse_table is None or mzi not in inverse_table:
        raise ValueError(f"No switch leak inverse table for MZI{mzi}.")
    table = inverse_table[mzi]
    reference_power_uw = float(input_reference_power_uw)
    if not np.isfinite(reference_power_uw) or reference_power_uw <= 0.0:
        raise ValueError(f"--input-reference-power-uw must be positive and finite, got {reference_power_uw!r}")

    chip_power_max_uw = float(table["chip_power_max_uw"])
    target_chip_power_uw = float(coeff * reference_power_uw)
    if target_chip_power_uw > chip_power_max_uw + 1e-9:
        raise ValueError(
            f"switch MZI{mzi} target chip power {target_chip_power_uw:.6g} uW exceeds "
            f"reachable max {chip_power_max_uw:.6g} uW."
        )

    target_fraction = float(np.clip(target_chip_power_uw / chip_power_max_uw, 0.0, 1.0))
    voltage = float(np.interp(target_fraction, table["feature_points"], table["voltage_points"]))
    target_leak_power = float(table["p_off_leak"] - target_chip_power_uw)
    assert_voltage_in_range(
        (
            f"fixed-power feature coefficient {coeff:.6g} mapped switch MZI{mzi} "
            f"port {int(row['PORT'])}"
        ),
        voltage,
        v_min,
        v_max,
    )
    return round(voltage, 3), {
        "input_reference_power_uw": reference_power_uw,
        "target_chip_power_uw": target_chip_power_uw,
        "target_leak_power": target_leak_power,
        "reference_fraction_of_max": float(reference_power_uw / chip_power_max_uw),
        "target_fraction_of_max": target_fraction,
        "p_on_leak": float(table["p_on_leak"]),
        "p_off_leak": float(table["p_off_leak"]),
        "chip_power_max_uw": chip_power_max_uw,
        "scan_path": table["scan_path"],
    }


def switch_voltage_from_coefficient(
    row,
    coefficient,
    mode="voltage-linear",
    v_min=0.0,
    v_max=5.5,
    inverse_table=None,
    input_reference_power_uw=DEFAULT_INPUT_REFERENCE_POWER_UW,
):
    coeff = float(coefficient)
    if not np.isfinite(coeff):
        raise ValueError(f"Feature coefficient is not finite: {coefficient!r}")
    coeff_tol = 1e-9
    if coeff < -coeff_tol or coeff > 1.0 + coeff_tol:
        raise ValueError(f"Feature coefficient {coeff:.12g} is outside [0, 1].")
    coeff = float(np.clip(coeff, 0.0, 1.0))
    on_v = float(row["ON"])
    off_v = float(row["OFF"])
    mode_norm = str(mode).strip().lower()
    info = {}
    if mode_norm == "scan-leak-inverse":
        voltage, info = switch_voltage_from_scan_leak_inverse(row, coeff, inverse_table, v_min, v_max)
    elif mode_norm == "scan-leak-fixed-power":
        voltage, info = switch_voltage_from_scan_leak_fixed_power(
            row,
            coeff,
            inverse_table,
            input_reference_power_uw,
            v_min,
            v_max,
        )
    elif mode_norm == "voltage-linear":
        voltage = off_v + coeff * (on_v - off_v)
    elif mode_norm == "binary-threshold":
        voltage = on_v if coeff >= 0.5 else off_v
    else:
        raise ValueError(f"Unsupported --input-upload-mode: {mode!r}")
    assert_voltage_in_range(
        f"feature coefficient {coeff:.6g} mapped switch MZI{int(row['MZI'])} port {int(row['PORT'])}",
        voltage,
        v_min,
        v_max,
    )
    return round(float(voltage), 3), info


def set_switch_feature_vector(
    working_data,
    switch_rows_by_mzi,
    coefficients,
    input_upload_mode="voltage-linear",
    v_min=0.0,
    v_max=5.5,
    inverse_table=None,
    input_reference_power_uw=DEFAULT_INPUT_REFERENCE_POWER_UW,
):
    records = []
    for idx, coeff in enumerate(np.asarray(coefficients, dtype=float), start=1):
        row = switch_rows_by_mzi[int(idx)]
        port = int(row["PORT"])
        voltage, mapping_info = switch_voltage_from_coefficient(
            row,
            coeff,
            mode=input_upload_mode,
            v_min=v_min,
            v_max=v_max,
            inverse_table=inverse_table,
            input_reference_power_uw=input_reference_power_uw,
        )
        cu.write_port_voltage(port, voltage, working_data)
        record = {
            "input_channel": int(idx),
            "switch_mzi": int(idx),
            "port": port,
            "feature_coefficient": float(coeff),
            "upload_voltage": float(voltage),
            "mode": str(input_upload_mode),
        }
        record.update(mapping_info)
        records.append(record)
    return records


def load_sample_input_voltage_table(path, source_df, input_count, v_min, v_max):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--sample-input-voltage-file not found: {path}")
    df = pd.read_csv(path)

    final_cols = [f"final_voltage_{idx}" for idx in range(int(input_count))]
    upload_cols = [f"upload_voltage_{idx}" for idx in range(int(input_count))]
    if all(col in df.columns for col in final_cols):
        voltage_cols = final_cols
    elif all(col in df.columns for col in upload_cols):
        voltage_cols = upload_cols
    else:
        raise ValueError(
            f"{path} must contain either {final_cols} or {upload_cols}."
        )

    use_sample_index = "sample_index" in df.columns and "sample_index" in source_df.columns
    if use_sample_index:
        if df["sample_index"].duplicated().any():
            duplicates = df.loc[df["sample_index"].duplicated(), "sample_index"].tolist()
            raise ValueError(f"{path} contains duplicate sample_index values: {duplicates}")
        by_sample_index = {int(row["sample_index"]): row for _, row in df.iterrows()}
    else:
        if len(df) < len(source_df):
            raise ValueError(
                f"{path} has {len(df)} rows, but {len(source_df)} samples are requested."
            )
        by_sample_index = None

    rows = []
    for local_idx in range(len(source_df)):
        metadata = metadata_for_sample(source_df, local_idx)
        if by_sample_index is not None:
            sample_id = int(metadata["sample_index"])
            if sample_id not in by_sample_index:
                raise ValueError(f"{path} has no row for sample_index={sample_id}")
            table_row = by_sample_index[sample_id]
        else:
            table_row = df.iloc[int(local_idx)]

        voltages = []
        for ch, col in enumerate(voltage_cols, start=1):
            voltage = assert_voltage_in_range(
                f"{path} sample {metadata['sample_index']} input channel {ch}",
                table_row[col],
                v_min,
                v_max,
            )
            voltages.append(float(voltage))

        row = dict(metadata)
        row["source_voltage_file"] = os.fspath(path)
        row["source_voltage_row"] = int(table_row.name)
        append_vector_columns(row, "input_voltage", voltages)
        rows.append(row)
    return rows


def set_switch_voltage_vector(working_data, switch_rows_by_mzi, voltages, v_min=0.0, v_max=5.5):
    records = []
    for idx, voltage in enumerate(np.asarray(voltages, dtype=float), start=1):
        row = switch_rows_by_mzi[int(idx)]
        port = int(row["PORT"])
        voltage = assert_voltage_in_range(
            f"voltage-table input channel {idx} switch MZI{int(row['MZI'])} port {port}",
            voltage,
            v_min,
            v_max,
        )
        cu.write_port_voltage(port, voltage, working_data)
        records.append(
            {
                "input_channel": int(idx),
                "switch_mzi": int(idx),
                "port": port,
                "upload_voltage": round(float(voltage), 3),
                "mode": "voltage-table",
            }
        )
    return records


def measure_sequential_input_sum(
    args,
    working_data,
    hardware,
    switch_rows_by_mzi,
    switch_current_table,
    metadata,
    voltages,
    baseline_matrix,
    coefficients=None,
):
    summed_powers = np.zeros(int(args.output_count), dtype=float)
    voltage_records = []
    current_check_rows = []
    component_rows = []
    voltages = np.asarray(voltages, dtype=float)
    if coefficients is None:
        coefficients = np.zeros(int(args.input_count), dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)

    for active_idx, voltage in enumerate(voltages, start=1):
        set_all_switches(working_data, args.input_count, "OFF")
        row = switch_rows_by_mzi[int(active_idx)]
        port = int(row["PORT"])
        voltage = assert_voltage_in_range(
            f"sequential input sample {metadata.get('sample_index')} input channel {active_idx}",
            voltage,
            args.v_min,
            args.v_max,
        )
        cu.write_port_voltage(port, voltage, working_data)

        record = dict(metadata)
        record.update(
            {
                "input_channel": int(active_idx),
                "switch_mzi": int(active_idx),
                "port": port,
                "upload_voltage": round(float(voltage), 3),
                "mode": "sequential-input-sum",
            }
        )
        voltage_records.append(record)
        validate_all_voltages(working_data, args.v_min, args.v_max)

        if hardware is None:
            if baseline_matrix is not None and np.asarray(baseline_matrix).ndim == 2:
                powers = np.asarray(baseline_matrix, dtype=float)[:, active_idx - 1] * float(
                    coefficients[active_idx - 1]
                )
            else:
                powers = np.zeros(int(args.output_count), dtype=float)
        else:
            um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
            time.sleep(float(args.settle_time))

            if args.switch_current_check:
                failures = um.verify_switch_mzi_currents(
                    hardware["mcv"],
                    working_data,
                    switch_current_table,
                    tolerance=float(args.switch_current_tolerance),
                )
                for item in failures:
                    failure_row = dict(item)
                    failure_row.update(metadata)
                    failure_row["active_input_channel"] = int(active_idx)
                    current_check_rows.append(failure_row)

            powers = read_opm_powers(hardware["opm2"], args.output_count)

        powers = np.asarray(powers, dtype=float)
        summed_powers += powers

        component_row = dict(metadata)
        component_row.update(
            {
                "active_input_channel": int(active_idx),
                "upload_voltage": round(float(voltage), 3),
                "component_output_power_sum": float(np.sum(powers)),
            }
        )
        for out_idx, power in enumerate(powers):
            component_row[f"component_output_power_{out_idx}"] = float(power)
        component_rows.append(component_row)

        if args.print_each_sample:
            print(
                f"sample {metadata.get('sample_index')} input {active_idx}: "
                f"component_sum={float(np.sum(powers)):.6g}, "
                f"argmax_output={int(np.argmax(powers))}",
                flush=True,
            )

    set_all_switches(working_data, args.input_count, "OFF")
    return summed_powers, voltage_records, current_check_rows, component_rows


def measure_samples_from_voltage_table(args, working_data, hardware, run_dir, source_df, features, baseline_matrix):
    switch_rows_by_mzi = load_switch_rows_by_mzi(args.switch_table, args.input_count, args.v_min, args.v_max)
    voltage_rows = load_sample_input_voltage_table(
        args.sample_input_voltage_file,
        source_df,
        args.input_count,
        args.v_min,
        args.v_max,
    )
    if should_record_results(args):
        pd.DataFrame(voltage_rows).to_csv(run_dir / "sample_input_voltage_from_table.csv", index=False)

    sample_outputs = []
    voltage_records = []
    current_check_rows = []
    component_rows = []
    switch_current_table = um.load_switch_mzi_table(args.switch_table)

    for local_idx, voltage_row in enumerate(voltage_rows):
        metadata = {
            key: voltage_row[key]
            for key in ("local_sample_row", "sample_index", "label")
            if key in voltage_row
        }
        voltages = np.asarray(
            [voltage_row[f"input_voltage_{idx}"] for idx in range(int(args.input_count))],
            dtype=float,
        )
        if args.sequential_input_sum:
            powers, records, failures, components = measure_sequential_input_sum(
                args,
                working_data,
                hardware,
                switch_rows_by_mzi,
                switch_current_table,
                metadata,
                voltages,
                baseline_matrix,
                coefficients=np.asarray(features[local_idx], dtype=float),
            )
            voltage_records.extend(records)
            current_check_rows.extend(failures)
            component_rows.extend(components)
        else:
            records = set_switch_voltage_vector(
                working_data,
                switch_rows_by_mzi,
                voltages,
                v_min=args.v_min,
                v_max=args.v_max,
            )
            for record in records:
                record.update(metadata)
                voltage_records.append(record)

            validate_all_voltages(working_data, args.v_min, args.v_max)
            if hardware is None:
                powers = np.asarray(baseline_matrix, dtype=float) @ np.asarray(features[local_idx], dtype=float)
            else:
                um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
                time.sleep(float(args.settle_time))

                if args.switch_current_check:
                    failures = um.verify_switch_mzi_currents(
                        hardware["mcv"],
                        working_data,
                        switch_current_table,
                        tolerance=float(args.switch_current_tolerance),
                    )
                    for item in failures:
                        row = dict(item)
                        row.update(metadata)
                        current_check_rows.append(row)

                powers = read_opm_powers(hardware["opm2"], args.output_count)

        sample_outputs.append(np.asarray(powers, dtype=float))
        if args.run_inference:
            print_inference_progress(
                args,
                metadata,
                powers,
                local_idx=local_idx,
                total_count=len(voltage_rows),
            )
        elif args.print_each_sample:
            print(
                f"sample {metadata['sample_index']}: voltage-table raw_total={float(np.sum(powers)):.6g}, "
                f"argmax_output={int(np.argmax(powers)) + 1}"
            )

    if hardware is not None:
        set_all_switches(working_data, args.input_count, "OFF")
        um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)

    if args.sequential_input_sum and component_rows and should_record_results(args):
        pd.DataFrame(component_rows).to_csv(run_dir / "sequential_input_component_outputs.csv", index=False)

    return np.asarray(sample_outputs, dtype=float), voltage_records, current_check_rows


def initialize_hardware(args):
    if args.dry_run:
        return None
    if not args.confirm_hardware:
        raise RuntimeError("Refusing hardware access: set --confirm-hardware true with --dry-run false.")
    hardware = {
        "mcv": cu.open_ser_connection(args.ser_address),
        "opm2": cu.open_VISA_connection(args.opm2_address),
    }
    if hardware["mcv"] is None or hardware["opm2"] is None:
        raise RuntimeError("Failed to initialize mcv or opm2.")
    um.mcv = hardware["mcv"]
    um.opm2 = hardware["opm2"]
    return hardware


def read_opm_powers(opm2, output_count):
    power_str_list = cu.read_pow(opm2)
    powers = []
    for idx, value in enumerate(power_str_list[:output_count]):
        try:
            powers.append(float(value))
        except ValueError as exc:
            raise ValueError(f"Invalid power at OPM channel {idx + 1}: {value!r}") from exc
    if len(powers) != int(output_count):
        raise ValueError(f"Expected {output_count} OPM powers, got {len(powers)}")
    return np.asarray(powers, dtype=float)


def normalize_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    sums = np.sum(matrix, axis=0, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0.0)


def measure_switch_on_transmission(args, working_data, hardware, run_dir):
    raw_cols = []
    current_check_rows = []
    switch_table = um.load_switch_mzi_table(args.switch_table)

    for input_ch in range(1, int(args.input_count) + 1):
        set_all_switches(working_data, args.input_count, "OFF")
        um.switch_IN(input_ch, "ON", working_data)
        validate_all_voltages(working_data, args.v_min, args.v_max)
        um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
        time.sleep(float(args.settle_time))

        if args.switch_current_check:
            failures = um.verify_switch_mzi_currents(
                hardware["mcv"],
                working_data,
                switch_table,
                tolerance=float(args.switch_current_tolerance),
            )
            for item in failures:
                row = dict(item)
                row["input_channel"] = int(input_ch)
                current_check_rows.append(row)

        powers = read_opm_powers(hardware["opm2"], args.output_count)
        raw_cols.append(powers)
        print(
            f"switch input {input_ch}: raw total={float(np.sum(powers)):.6g}, "
            f"argmax output={int(np.argmax(powers)) + 1}"
        )

    set_all_switches(working_data, args.input_count, "OFF")
    um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)

    raw_matrix = np.column_stack(raw_cols)
    norm_matrix = normalize_columns(raw_matrix)
    if should_record_results(args):
        np.savetxt(run_dir / "switch_on_power_matrix_raw.csv", raw_matrix, delimiter=",")
        np.savetxt(run_dir / "switch_on_power_matrix_norm.csv", norm_matrix, delimiter=",")

    route_rows = []
    for col_idx in range(norm_matrix.shape[1]):
        expected_output = col_idx + 1
        argmax_output = int(np.argmax(norm_matrix[:, col_idx])) + 1
        diagonal_power = float(norm_matrix[expected_output - 1, col_idx])
        route_rows.append(
            {
                "input_channel": int(col_idx + 1),
                "expected_output_channel": int(expected_output),
                "argmax_output_channel": int(argmax_output),
                "diagonal_normalized_power": diagonal_power,
                "raw_total_power": float(np.sum(raw_matrix[:, col_idx])),
                "identity_route_ok": bool(
                    argmax_output == expected_output
                    and diagonal_power >= float(args.identity_power_threshold)
                ),
            }
        )
    if should_record_results(args):
        pd.DataFrame(route_rows).to_csv(run_dir / "switch_on_route_check.csv", index=False)

    if current_check_rows:
        if should_record_results(args):
            pd.DataFrame(current_check_rows).to_csv(run_dir / "switch_current_check_failures.csv", index=False)
        if args.fail_on_switch_current_failure:
            raise RuntimeError(
                "Switch current check failed; see switch_current_check_failures.csv in the run directory."
            )

    if args.fail_on_routing_mismatch:
        bad_routes = [row for row in route_rows if not row["identity_route_ok"]]
        if bad_routes:
            raise RuntimeError(f"Pass-through route check failed: {bad_routes}")

    return raw_matrix, norm_matrix, route_rows


def build_route_check_rows(matrix, identity_power_threshold):
    matrix = np.asarray(matrix, dtype=float)
    norm_matrix = normalize_columns(matrix)
    route_rows = []
    for col_idx in range(norm_matrix.shape[1]):
        expected_output = col_idx + 1
        argmax_output = int(np.argmax(norm_matrix[:, col_idx])) + 1
        diagonal_power = float(norm_matrix[expected_output - 1, col_idx])
        route_rows.append(
            {
                "input_channel": int(col_idx + 1),
                "expected_output_channel": int(expected_output),
                "argmax_output_channel": int(argmax_output),
                "diagonal_normalized_power": diagonal_power,
                "raw_total_power": float(np.sum(matrix[:, col_idx])),
                "identity_route_ok": bool(
                    argmax_output == expected_output
                    and diagonal_power >= float(identity_power_threshold)
                ),
            }
        )
    return route_rows


def measure_reference_power_transmission(args, working_data, hardware, run_dir):
    raw_cols = []
    current_check_rows = []
    voltage_records = []
    switch_rows_by_mzi = load_switch_rows_by_mzi(args.switch_table, args.input_count, args.v_min, args.v_max)
    switch_table = um.load_switch_mzi_table(args.switch_table)
    inverse_table = getattr(args, "switch_leak_inverse_tables", None)

    for input_ch in range(1, int(args.input_count) + 1):
        set_all_switches(working_data, args.input_count, "OFF")
        coeffs = np.zeros(int(args.input_count), dtype=float)
        coeffs[input_ch - 1] = 1.0
        records = set_switch_feature_vector(
            working_data,
            switch_rows_by_mzi,
            coeffs,
            input_upload_mode=args.input_upload_mode,
            v_min=args.v_min,
            v_max=args.v_max,
            inverse_table=inverse_table,
            input_reference_power_uw=args.input_reference_power_uw,
        )
        for record in records:
            record["reference_input_channel"] = int(input_ch)
            voltage_records.append(record)

        validate_all_voltages(working_data, args.v_min, args.v_max)
        um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
        time.sleep(float(args.settle_time))

        if args.switch_current_check:
            failures = um.verify_switch_mzi_currents(
                hardware["mcv"],
                working_data,
                switch_table,
                tolerance=float(args.switch_current_tolerance),
            )
            for item in failures:
                row = dict(item)
                row["reference_input_channel"] = int(input_ch)
                current_check_rows.append(row)

        powers = read_opm_powers(hardware["opm2"], args.output_count)
        raw_cols.append(powers)
        print(
            f"reference input {input_ch}: target={float(args.input_reference_power_uw):.6g} uW, "
            f"raw total={float(np.sum(powers)):.6g}, argmax output={int(np.argmax(powers)) + 1}"
        )

    set_all_switches(working_data, args.input_count, "OFF")
    um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)

    raw_matrix = np.column_stack(raw_cols)
    norm_matrix = normalize_columns(raw_matrix)
    if should_record_results(args):
        np.savetxt(run_dir / "switch_reference_power_matrix_raw.csv", raw_matrix, delimiter=",")
        np.savetxt(run_dir / "switch_reference_power_matrix_norm.csv", norm_matrix, delimiter=",")
        pd.DataFrame(build_route_check_rows(raw_matrix, args.identity_power_threshold)).to_csv(
            run_dir / "switch_reference_route_check.csv",
            index=False,
        )
    if voltage_records:
        if should_record_results(args):
            pd.DataFrame(voltage_records).to_csv(run_dir / "switch_reference_voltage_plan.csv", index=False)
    if current_check_rows:
        if should_record_results(args):
            pd.DataFrame(current_check_rows).to_csv(run_dir / "switch_reference_current_check_failures.csv", index=False)
        if args.fail_on_switch_current_failure:
            raise RuntimeError(
                "Reference switch current check failed; see switch_reference_current_check_failures.csv."
            )

    return raw_matrix, norm_matrix, build_route_check_rows(raw_matrix, args.identity_power_threshold)


def build_dry_run_baseline(input_count):
    return np.eye(int(input_count), dtype=float)


def build_sample_output_dataframe(sample_outputs, source_df):
    metadata_cols = [col for col in ("sample_index", "label", "prediction", "correct") if col in source_df.columns]
    out = source_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=source_df.index)
    sample_outputs = np.asarray(sample_outputs, dtype=float)
    for idx in range(sample_outputs.shape[1]):
        out[f"measured_output_power_{idx}"] = sample_outputs[:, idx]
    out["measured_output_power_sum"] = np.sum(sample_outputs, axis=1)
    return out


def build_inference_result_table(source_df, sample_outputs, class_index_offset=0):
    metadata_cols = [col for col in ("sample_index", "label", "prediction", "correct") if col in source_df.columns]
    out = source_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=source_df.index)
    sample_outputs = np.asarray(sample_outputs, dtype=float)
    for idx in range(sample_outputs.shape[1]):
        out[f"output_power_{idx}"] = sample_outputs[:, idx]

    total = np.sum(sample_outputs, axis=1)
    norm_outputs = np.divide(
        sample_outputs,
        total.reshape(-1, 1),
        out=np.zeros_like(sample_outputs),
        where=total.reshape(-1, 1) != 0.0,
    )
    for idx in range(norm_outputs.shape[1]):
        out[f"output_power_norm_{idx}"] = norm_outputs[:, idx]

    pred_zero_based = np.argmax(sample_outputs, axis=1).astype(int)
    out["predicted_output_index"] = pred_zero_based
    out["predicted_class"] = pred_zero_based + int(class_index_offset)
    out["predicted_output_power"] = sample_outputs[np.arange(sample_outputs.shape[0]), pred_zero_based]
    out["output_power_sum"] = total

    if "label" in source_df.columns:
        labels = pd.to_numeric(source_df["label"], errors="coerce").to_numpy(dtype=float)
        out["inference_correct"] = (out["predicted_class"].to_numpy(dtype=float) == labels)
    return out


def print_inference_progress(args, metadata, powers, local_idx=None, total_count=None):
    if not getattr(args, "run_inference", False):
        return

    if not hasattr(args, "_inference_progress_seen"):
        args._inference_progress_seen = 0
        args._inference_progress_labeled = 0
        args._inference_progress_correct = 0

    powers = np.asarray(powers, dtype=float)
    pred_output_index = int(np.argmax(powers))
    pred_class = int(pred_output_index + int(args.class_index_offset))
    pred_power = float(powers[pred_output_index])
    output_sum = float(np.sum(powers))

    sample_index = metadata.get("sample_index", local_idx if local_idx is not None else args._inference_progress_seen)
    label = metadata.get("label")
    args._inference_progress_seen += 1

    correct_text = "N/A"
    running_acc_text = "N/A"
    if label is not None and pd.notna(label):
        label_int = int(label)
        correct = bool(pred_class == label_int)
        args._inference_progress_labeled += 1
        args._inference_progress_correct += int(correct)
        correct_text = "True" if correct else "False"
        running_acc = args._inference_progress_correct / args._inference_progress_labeled
        running_acc_text = (
            f"{args._inference_progress_correct}/{args._inference_progress_labeled}="
            f"{running_acc:.4f}"
        )
        label_text = str(label_int)
    else:
        label_text = "N/A"

    progress_text = (
        f"{int(local_idx) + 1}/{int(total_count)}"
        if local_idx is not None and total_count is not None
        else str(args._inference_progress_seen)
    )
    print(
        "inference "
        f"{progress_text}: sample_index={sample_index}, "
        f"label={label_text}, pred_class={pred_class}, pred_output={pred_output_index}, "
        f"correct={correct_text}, running_acc={running_acc_text}, "
        f"output_sum={output_sum:.6g}, pred_power={pred_power:.6g}",
        flush=True,
    )


def estimate_coefficients_from_outputs(sample_outputs, baseline_matrix, diagonal_floor=1e-12):
    sample_outputs = np.asarray(sample_outputs, dtype=float)
    baseline_matrix = np.asarray(baseline_matrix, dtype=float)
    if baseline_matrix.ndim != 2:
        raise ValueError("baseline_matrix must be 2D")
    if sample_outputs.ndim != 2:
        raise ValueError("sample_outputs must be 2D")
    if sample_outputs.shape[1] != baseline_matrix.shape[0]:
        raise ValueError(
            f"sample output width {sample_outputs.shape[1]} does not match baseline output count {baseline_matrix.shape[0]}"
        )

    diag = np.diag(baseline_matrix)
    diag_est = np.divide(
        sample_outputs[:, : len(diag)],
        diag.reshape(1, -1),
        out=np.full((sample_outputs.shape[0], len(diag)), np.nan, dtype=float),
        where=np.abs(diag.reshape(1, -1)) > float(diagonal_floor),
    )

    # y = B @ x, where B rows are output channels and columns are input channels.
    # Solve B x ~= y for every measured sample y.
    lstsq_est = np.linalg.lstsq(baseline_matrix, sample_outputs.T, rcond=None)[0].T
    expected_from_lstsq = (baseline_matrix @ lstsq_est.T).T
    return diag_est, lstsq_est, expected_from_lstsq


def build_input_comparison_table(source_df, features, sample_outputs, baseline_matrix, diagonal_floor=1e-12):
    metadata_cols = [col for col in ("sample_index", "label", "prediction", "correct") if col in source_df.columns]
    out = source_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=source_df.index)
    features = np.asarray(features, dtype=float)
    sample_outputs = np.asarray(sample_outputs, dtype=float)
    baseline_matrix = np.asarray(baseline_matrix, dtype=float)
    expected_outputs = (baseline_matrix @ features.T).T
    diag_est, lstsq_est, expected_from_lstsq = estimate_coefficients_from_outputs(
        sample_outputs,
        baseline_matrix,
        diagonal_floor=diagonal_floor,
    )

    for idx in range(features.shape[1]):
        out[f"feature_csv_{idx}"] = features[:, idx]
    for idx in range(sample_outputs.shape[1]):
        out[f"measured_output_power_{idx}"] = sample_outputs[:, idx]
    for idx in range(expected_outputs.shape[1]):
        out[f"expected_output_power_from_csv_{idx}"] = expected_outputs[:, idx]
    for idx in range(diag_est.shape[1]):
        out[f"estimated_feature_diag_{idx}"] = diag_est[:, idx]
        out[f"feature_error_diag_{idx}"] = diag_est[:, idx] - features[:, idx]
    for idx in range(lstsq_est.shape[1]):
        out[f"estimated_feature_lstsq_{idx}"] = lstsq_est[:, idx]
        out[f"feature_error_lstsq_{idx}"] = lstsq_est[:, idx] - features[:, idx]
    for idx in range(expected_from_lstsq.shape[1]):
        out[f"fitted_output_power_lstsq_{idx}"] = expected_from_lstsq[:, idx]

    output_error = sample_outputs - expected_outputs
    out["feature_mae_diag"] = np.nanmean(np.abs(diag_est - features), axis=1)
    out["feature_max_abs_error_diag"] = np.nanmax(np.abs(diag_est - features), axis=1)
    out["feature_mae_lstsq"] = np.mean(np.abs(lstsq_est - features), axis=1)
    out["feature_max_abs_error_lstsq"] = np.max(np.abs(lstsq_est - features), axis=1)
    out["output_power_mae_vs_csv_expected"] = np.mean(np.abs(output_error), axis=1)
    out["output_power_max_abs_error_vs_csv_expected"] = np.max(np.abs(output_error), axis=1)
    return out, {
        "expected_outputs": expected_outputs,
        "diag_est": diag_est,
        "lstsq_est": lstsq_est,
        "output_error": output_error,
    }


def measure_feature_samples(args, working_data, hardware, run_dir, source_df, features, baseline_matrix):
    switch_rows_by_mzi = load_switch_rows_by_mzi(args.switch_table, args.input_count, args.v_min, args.v_max)
    sample_outputs = []
    voltage_records = []
    current_check_rows = []
    component_rows = []
    switch_current_table = um.load_switch_mzi_table(args.switch_table)
    inverse_table = getattr(args, "switch_leak_inverse_tables", None)

    if args.dry_run:
        for local_idx, coeffs in enumerate(np.asarray(features, dtype=float)):
            metadata = metadata_for_sample(source_df, local_idx)
            sample_id = int(metadata["sample_index"])
            records = set_switch_feature_vector(
                working_data,
                switch_rows_by_mzi,
                coeffs,
                input_upload_mode=args.input_upload_mode,
                v_min=args.v_min,
                v_max=args.v_max,
                inverse_table=inverse_table,
                input_reference_power_uw=args.input_reference_power_uw,
            )
            if args.sequential_input_sum:
                voltages = records_to_voltage_array(records, args.input_count)
                powers, seq_records, failures, components = measure_sequential_input_sum(
                    args,
                    working_data,
                    hardware=None,
                    switch_rows_by_mzi=switch_rows_by_mzi,
                    switch_current_table=switch_current_table,
                    metadata=metadata,
                    voltages=voltages,
                    baseline_matrix=baseline_matrix,
                    coefficients=coeffs,
                )
                sample_outputs.append(powers)
                voltage_records.extend(seq_records)
                current_check_rows.extend(failures)
                component_rows.extend(components)
            else:
                for record in records:
                    record["local_sample_row"] = int(local_idx)
                    record["sample_index"] = int(sample_id)
                    voltage_records.append(record)
            validate_all_voltages(working_data, args.v_min, args.v_max)
        if args.sequential_input_sum:
            sample_outputs = np.asarray(sample_outputs, dtype=float)
            if component_rows and should_record_results(args):
                pd.DataFrame(component_rows).to_csv(run_dir / "sequential_input_component_outputs.csv", index=False)
        else:
            sample_outputs = (np.asarray(baseline_matrix, dtype=float) @ np.asarray(features, dtype=float).T).T
        if args.run_inference:
            for local_idx, powers in enumerate(sample_outputs):
                print_inference_progress(
                    args,
                    metadata_for_sample(source_df, local_idx),
                    powers,
                    local_idx=local_idx,
                    total_count=len(source_df),
                )
        return sample_outputs, voltage_records, current_check_rows

    for local_idx, coeffs in enumerate(np.asarray(features, dtype=float)):
        metadata = metadata_for_sample(source_df, local_idx)
        sample_id = int(metadata["sample_index"])
        records = set_switch_feature_vector(
            working_data,
            switch_rows_by_mzi,
            coeffs,
            input_upload_mode=args.input_upload_mode,
            v_min=args.v_min,
            v_max=args.v_max,
            inverse_table=inverse_table,
            input_reference_power_uw=args.input_reference_power_uw,
        )
        if args.sequential_input_sum:
            voltages = records_to_voltage_array(records, args.input_count)
            powers, seq_records, failures, components = measure_sequential_input_sum(
                args,
                working_data,
                hardware,
                switch_rows_by_mzi,
                switch_current_table,
                metadata,
                voltages,
                baseline_matrix,
                coefficients=coeffs,
            )
            voltage_records.extend(seq_records)
            current_check_rows.extend(failures)
            component_rows.extend(components)
        else:
            for record in records:
                record["local_sample_row"] = int(local_idx)
                record["sample_index"] = int(sample_id)
                voltage_records.append(record)

            validate_all_voltages(working_data, args.v_min, args.v_max)
            um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
            time.sleep(float(args.settle_time))

            if args.switch_current_check:
                failures = um.verify_switch_mzi_currents(
                    hardware["mcv"],
                    working_data,
                    switch_current_table,
                    tolerance=float(args.switch_current_tolerance),
                )
                for item in failures:
                    row = dict(item)
                    row["local_sample_row"] = int(local_idx)
                    row["sample_index"] = int(sample_id)
                    current_check_rows.append(row)

            powers = read_opm_powers(hardware["opm2"], args.output_count)
        sample_outputs.append(powers)
        if args.run_inference:
            print_inference_progress(
                args,
                metadata,
                powers,
                local_idx=local_idx,
                total_count=len(source_df),
            )
        elif args.print_each_sample:
            print(
                f"sample {sample_id}: raw_total={float(np.sum(powers)):.6g}, "
                f"argmax_output={int(np.argmax(powers)) + 1}"
            )

    set_all_switches(working_data, args.input_count, "OFF")
    um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
    if args.sequential_input_sum and component_rows and should_record_results(args):
        pd.DataFrame(component_rows).to_csv(run_dir / "sequential_input_component_outputs.csv", index=False)
    return np.asarray(sample_outputs, dtype=float), voltage_records, current_check_rows


def normalize_vector(values, eps=1e-15):
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= float(eps):
        return np.zeros_like(values, dtype=float), total
    return values / total, total


def metadata_for_sample(source_df, local_idx):
    source_row = source_df.iloc[int(local_idx)]
    sample_id = (
        int(source_row["sample_index"])
        if "sample_index" in source_df.columns and pd.notna(source_row["sample_index"])
        else int(local_idx)
    )
    metadata = {
        "local_sample_row": int(local_idx),
        "sample_index": int(sample_id),
    }
    if "label" in source_df.columns and pd.notna(source_row["label"]):
        metadata["label"] = int(source_row["label"])
    return metadata


def records_to_voltage_array(records, input_count):
    voltages = np.full(int(input_count), np.nan, dtype=float)
    for record in records:
        idx = int(record["input_channel"]) - 1
        if 0 <= idx < int(input_count):
            voltages[idx] = float(record["upload_voltage"])
    if not np.isfinite(voltages).all():
        raise ValueError(f"Missing switch voltages in records: {records}")
    return voltages


def append_vector_columns(row, prefix, values):
    for idx, value in enumerate(np.asarray(values, dtype=float)):
        row[f"{prefix}_{idx}"] = float(value)


def closed_loop_loss(target_ratio, measured_powers, eps=1e-15):
    measured_ratio, output_sum = normalize_vector(measured_powers, eps=eps)
    error = measured_ratio - np.asarray(target_ratio, dtype=float)
    return {
        "loss_mae": float(np.mean(np.abs(error))),
        "loss_mse": float(np.mean(error**2)),
        "max_abs_ratio_error": float(np.max(np.abs(error))),
        "measured_ratio": measured_ratio,
        "output_sum": float(output_sum),
        "error": error,
    }


def closed_loop_absolute_baseline_loss(
    target_coefficients,
    target_powers,
    measured_powers,
    baseline_matrix,
    power_to_uw=1e6,
    optimization_mask=None,
):
    """Evaluate absolute output power error through the measured 120-uW baseline matrix."""
    target_coefficients = np.asarray(target_coefficients, dtype=float)
    target_powers = np.asarray(target_powers, dtype=float)
    measured_powers = np.asarray(measured_powers, dtype=float)
    estimated_coefficients = np.linalg.pinv(np.asarray(baseline_matrix, dtype=float)) @ measured_powers
    coefficient_error = estimated_coefficients - target_coefficients
    power_error = measured_powers - target_powers
    if optimization_mask is None:
        optimization_mask = np.ones_like(target_coefficients, dtype=bool)
    optimization_mask = np.asarray(optimization_mask, dtype=bool)
    if optimization_mask.shape != target_coefficients.shape or not optimization_mask.any():
        raise ValueError("optimization_mask must select at least one target coefficient.")
    optimized_error = coefficient_error[optimization_mask]
    return {
        "loss_mae": float(np.mean(np.abs(optimized_error))),
        "loss_mse": float(np.mean(optimized_error**2)),
        "max_abs_ratio_error": float(np.max(np.abs(optimized_error))),
        # Keep these legacy keys so existing result readers remain compatible.
        "measured_ratio": estimated_coefficients,
        "output_sum": float(np.sum(measured_powers)),
        "error": coefficient_error,
        "optimization_mask": optimization_mask,
        "target_powers": target_powers,
        "power_error": power_error,
        "power_to_uw": float(power_to_uw),
        "power_mae_uw": float(np.mean(np.abs(power_error)) * float(power_to_uw)),
        "power_max_abs_uw": float(np.max(np.abs(power_error)) * float(power_to_uw)),
    }


def closed_loop_absolute_total_power_loss(
    target_coefficients, measured_powers, reference_power_uw, optimization_mask, power_to_uw=1e6
):
    """Match the sum of all measured outputs to coefficient * reference power."""
    target_coefficients = np.asarray(target_coefficients, dtype=float)
    optimization_mask = np.asarray(optimization_mask, dtype=bool)
    if optimization_mask.shape != target_coefficients.shape or not optimization_mask.any():
        raise ValueError("optimization_mask must select at least one target coefficient.")
    target_total_uw = float(np.sum(target_coefficients[optimization_mask]) * float(reference_power_uw))
    measured_total_uw = float(np.sum(np.asarray(measured_powers, dtype=float)) * float(power_to_uw))
    normalized_total_error = float((measured_total_uw - target_total_uw) / float(reference_power_uw))
    estimated_coefficients = np.zeros_like(target_coefficients, dtype=float)
    # Image vectors are required to contain exactly one active source. Keeping
    # this representation general still makes the update deterministic.
    estimated_coefficients[optimization_mask] = (
        np.sum(target_coefficients[optimization_mask]) + normalized_total_error
    ) / int(np.sum(optimization_mask))
    coefficient_error = estimated_coefficients - target_coefficients
    optimized_error = coefficient_error[optimization_mask]
    return {
        "loss_mae": float(np.mean(np.abs(optimized_error))),
        "loss_mse": float(np.mean(optimized_error**2)),
        "max_abs_ratio_error": float(np.max(np.abs(optimized_error))),
        "measured_ratio": estimated_coefficients,
        "output_sum": float(np.sum(measured_powers)),
        "error": coefficient_error,
        "optimization_mask": optimization_mask,
        "target_total_power_uw": target_total_uw,
        "measured_total_power_uw": measured_total_uw,
        "total_power_error_uw": measured_total_uw - target_total_uw,
        "power_mae_uw": abs(measured_total_uw - target_total_uw),
        "power_max_abs_uw": abs(measured_total_uw - target_total_uw),
    }


def upload_coefficients_and_measure(
    args,
    working_data,
    hardware,
    switch_rows_by_mzi,
    coefficients,
    baseline_matrix,
):
    records = set_switch_feature_vector(
        working_data,
        switch_rows_by_mzi,
        coefficients,
        input_upload_mode=args.input_upload_mode,
        v_min=args.v_min,
        v_max=args.v_max,
        inverse_table=getattr(args, "switch_leak_inverse_tables", None),
        input_reference_power_uw=args.input_reference_power_uw,
    )
    validate_all_voltages(working_data, args.v_min, args.v_max)

    current_failures = []
    if hardware is None:
        powers = np.asarray(baseline_matrix, dtype=float) @ np.asarray(coefficients, dtype=float)
        return np.asarray(powers, dtype=float), records, current_failures

    um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
    time.sleep(float(args.settle_time))

    if args.switch_current_check:
        switch_current_table = um.load_switch_mzi_table(args.switch_table)
        current_failures = um.verify_switch_mzi_currents(
            hardware["mcv"],
            working_data,
            switch_current_table,
            tolerance=float(args.switch_current_tolerance),
        )

    powers = read_opm_powers(hardware["opm2"], args.output_count)
    return np.asarray(powers, dtype=float), records, current_failures


def build_closed_loop_iter_row(
    metadata,
    iteration,
    accepted,
    learning_rate,
    coefficients,
    voltages,
    measured_powers,
    target_ratio,
    metrics,
):
    row = dict(metadata)
    row.update(
        {
            "iteration": int(iteration),
            "accepted": bool(accepted),
            "learning_rate": float(learning_rate),
            "loss_mae": float(metrics["loss_mae"]),
            "loss_mse": float(metrics["loss_mse"]),
            "max_abs_ratio_error": float(metrics["max_abs_ratio_error"]),
            "output_power_sum": float(metrics["output_sum"]),
        }
    )
    append_vector_columns(row, "target_ratio", target_ratio)
    append_vector_columns(row, "measured_ratio", metrics["measured_ratio"])
    append_vector_columns(row, "ratio_error", metrics["error"])
    append_vector_columns(row, "optimized_coefficient", coefficients)
    append_vector_columns(row, "upload_voltage", voltages)
    append_vector_columns(row, "output_power", measured_powers)
    if "target_powers" in metrics:
        append_vector_columns(row, "target_output_power", metrics["target_powers"])
        append_vector_columns(row, "output_power_error", metrics["power_error"])
        row["output_power_mae_uw"] = float(metrics["power_mae_uw"])
        row["output_power_max_abs_uw"] = float(metrics["power_max_abs_uw"])
    if "target_total_power_uw" in metrics:
        row["target_total_power_uw"] = float(metrics["target_total_power_uw"])
        row["measured_total_power_uw"] = float(metrics["measured_total_power_uw"])
        row["total_power_error_uw"] = float(metrics["total_power_error_uw"])
    return row


def print_closed_loop_live(metadata, iteration, accepted, voltages, measured_powers, metrics):
    status = "accepted" if accepted else "rejected"
    print(f"\n[sample {metadata['sample_index']} iter {iteration:02d} {status}]")
    if "target_total_power_uw" in metrics:
        measured_uw = np.asarray(measured_powers, dtype=float) * float(metrics.get("power_to_uw", 1e6))
        print(f"  target_total_uW  : {metrics['target_total_power_uw']:.4f}")
        print(f"  measured_total_uW: {metrics['measured_total_power_uw']:.4f}")
        print(f"  total_delta_uW   : {metrics['total_power_error_uw']:.4f}")
        print("  output_channels_uW: " + np.array2string(measured_uw, precision=3, suppress_small=True))
    elif "target_powers" in metrics:
        power_to_uw = float(metrics.get("power_to_uw", 1e6))
        target_uw = np.asarray(metrics["target_powers"], dtype=float) * power_to_uw
        measured_uw = np.asarray(measured_powers, dtype=float) * power_to_uw
        error_uw = measured_uw - target_uw
        print("  target_uW  : " + np.array2string(target_uw, precision=3, suppress_small=True))
        print("  measured_uW: " + np.array2string(measured_uw, precision=3, suppress_small=True))
        print("  delta_uW   : " + np.array2string(error_uw, precision=3, suppress_small=True))
        print(
            f"  power MAE={metrics['power_mae_uw']:.4f} uW, "
            f"max|delta|={metrics['power_max_abs_uw']:.4f} uW"
        )
    else:
        print("  target     : " + np.array2string(np.asarray(metrics["measured_ratio"]) - metrics["error"], precision=4))
        print("  measured   : " + np.array2string(np.asarray(metrics["measured_ratio"]), precision=4))
        print("  delta      : " + np.array2string(np.asarray(metrics["error"]), precision=4))
    print("  voltage_V  : " + np.array2string(np.asarray(voltages), precision=3))
    print(f"  coefficient MAE={metrics['loss_mae']:.6g}, max|error|={metrics['max_abs_ratio_error']:.6g}")


def optimize_input_vectors_closed_loop(args, working_data, hardware, run_dir, source_df, features, baseline_matrix):
    switch_rows_by_mzi = load_switch_rows_by_mzi(args.switch_table, args.input_count, args.v_min, args.v_max)
    sample_outputs = []
    final_voltage_rows = []
    final_voltage_long_rows = []
    iter_rows = []
    current_check_rows = []
    power_to_uw = 1.0 if hardware is None else 1e6
    completed_sample_ids = set()
    completed_rows_by_id = {}
    if should_record_results(args) and bool(getattr(args, "resume_run_dir", "")):
        final_voltage_rows = load_checkpoint_rows(run_dir / "closed_loop_input_final_voltage.csv")
        final_voltage_long_rows = load_checkpoint_rows(run_dir / "closed_loop_input_final_voltage_long.csv")
        iter_rows = load_checkpoint_rows(run_dir / "closed_loop_input_iter_log.csv")
        current_check_rows = load_checkpoint_rows(run_dir / "closed_loop_input_current_check_failures.csv")
        if bool(getattr(args, "retry_nonconverged", False)):
            retry_sample_ids = {
                int(row["sample_index"])
                for row in final_voltage_rows
                if not parse_bool(row.get("converged", False))
            }
            final_voltage_rows = [
                row for row in final_voltage_rows if int(row["sample_index"]) not in retry_sample_ids
            ]
            final_voltage_long_rows = [
                row for row in final_voltage_long_rows if int(row["sample_index"]) not in retry_sample_ids
            ]
            iter_rows = [row for row in iter_rows if int(row["sample_index"]) not in retry_sample_ids]
            current_check_rows = [
                row for row in current_check_rows if int(row["sample_index"]) not in retry_sample_ids
            ]
            print(
                f"Retry-nonconverged mode: retained {len(final_voltage_rows)} converged vectors; "
                f"will recalibrate {len(retry_sample_ids)} vectors: {sorted(retry_sample_ids)}"
            )
        completed_sample_ids = {int(row["sample_index"]) for row in final_voltage_rows}
        completed_rows_by_id = {int(row["sample_index"]): row for row in final_voltage_rows}
        print(f"Resume checkpoint: {len(completed_sample_ids)} input vectors already complete.")

    for local_idx, feature_row in enumerate(np.asarray(features, dtype=float)):
        metadata = metadata_for_sample(source_df, local_idx)
        if int(metadata["sample_index"]) in completed_sample_ids:
            checkpoint_row = completed_rows_by_id[int(metadata["sample_index"])]
            sample_outputs.append(
                np.asarray(
                    [checkpoint_row[f"final_output_power_{idx}"] for idx in range(int(args.output_count))],
                    dtype=float,
                )
            )
            continue
        target_ratio, target_sum = normalize_vector(feature_row, eps=args.closed_loop_output_epsilon)
        if target_sum <= float(args.closed_loop_output_epsilon):
            raise ValueError(f"sample {metadata['sample_index']} has zero feature-vector sum.")

        objective = getattr(args, "closed_loop_objective", "ratio")
        absolute_baseline_objective = objective == "absolute-baseline"
        absolute_total_objective = objective == "absolute-total-power"
        absolute_objective = absolute_baseline_objective or absolute_total_objective
        target_coefficients = np.clip(np.asarray(feature_row, dtype=float), 0.0, 1.0)
        optimization_mask = target_coefficients > float(args.closed_loop_output_epsilon)
        if absolute_objective and not optimization_mask.any():
            raise ValueError(f"sample {metadata['sample_index']} has no nonzero input channel to optimize.")
        target_powers = np.asarray(baseline_matrix, dtype=float) @ target_coefficients
        logged_target = target_coefficients if absolute_objective else target_ratio
        best_coefficients = (target_coefficients if absolute_objective else target_ratio).copy()
        lr = float(args.closed_loop_lr)

        measured_powers, records, failures = upload_coefficients_and_measure(
            args,
            working_data,
            hardware,
            switch_rows_by_mzi,
            best_coefficients,
            baseline_matrix,
        )
        for item in failures:
            row = dict(item)
            row.update(metadata)
            row["iteration"] = 0
            current_check_rows.append(row)

        best_voltages = records_to_voltage_array(records, args.input_count)
        if absolute_total_objective:
            best_metrics = closed_loop_absolute_total_power_loss(
                target_coefficients,
                measured_powers,
                args.input_reference_power_uw,
                optimization_mask,
                power_to_uw=power_to_uw,
            )
            best_metrics["power_to_uw"] = power_to_uw
        elif absolute_baseline_objective:
            best_metrics = closed_loop_absolute_baseline_loss(
                target_coefficients,
                target_powers,
                measured_powers,
                baseline_matrix,
                power_to_uw=power_to_uw,
                optimization_mask=optimization_mask,
            )
        else:
            best_metrics = closed_loop_loss(target_ratio, measured_powers, eps=args.closed_loop_output_epsilon)
        current_state_is_best = True
        initial_loss_mae = float(best_metrics["loss_mae"])
        best_powers = np.asarray(measured_powers, dtype=float)
        initial_coefficients = best_coefficients.copy()
        initial_voltages = best_voltages.copy()
        iter_rows.append(
            build_closed_loop_iter_row(
                metadata,
                iteration=0,
                accepted=True,
                learning_rate=lr,
                coefficients=best_coefficients,
                voltages=best_voltages,
                measured_powers=best_powers,
                target_ratio=logged_target,
                metrics=best_metrics,
            )
        )
        if args.print_each_sample:
            print_closed_loop_live(metadata, 0, True, best_voltages, best_powers, best_metrics)

        iterations_executed = 0
        converged = bool(best_metrics["loss_mae"] <= float(args.closed_loop_tol))

        for iteration in range(1, int(args.closed_loop_max_iters) + 1):
            if converged:
                break
            iterations_executed = int(iteration)
            update_target = target_coefficients if absolute_objective else target_ratio
            error_for_update = update_target - best_metrics["measured_ratio"]
            if absolute_objective:
                proposal = best_coefficients.copy()
                proposal[optimization_mask] = np.clip(
                    best_coefficients[optimization_mask] + lr * error_for_update[optimization_mask],
                    0.0,
                    1.0,
                )
                # A zero target means that input MZI must remain physically OFF;
                # it must never be used to compensate another channel's error.
                proposal[~optimization_mask] = 0.0
            else:
                proposal = np.clip(best_coefficients + lr * error_for_update, 0.0, 1.0)
            if np.allclose(proposal, best_coefficients, atol=1e-12, rtol=0.0):
                lr *= float(args.closed_loop_lr_shrink)
                if lr < float(args.closed_loop_min_lr):
                    break
                continue

            measured_powers, records, failures = upload_coefficients_and_measure(
                args,
                working_data,
                hardware,
                switch_rows_by_mzi,
                proposal,
                baseline_matrix,
            )
            for item in failures:
                row = dict(item)
                row.update(metadata)
                row["iteration"] = int(iteration)
                current_check_rows.append(row)

            voltages = records_to_voltage_array(records, args.input_count)
            if absolute_total_objective:
                metrics = closed_loop_absolute_total_power_loss(
                    target_coefficients,
                    measured_powers,
                    args.input_reference_power_uw,
                    optimization_mask,
                    power_to_uw=power_to_uw,
                )
                metrics["power_to_uw"] = power_to_uw
            elif absolute_baseline_objective:
                metrics = closed_loop_absolute_baseline_loss(
                    target_coefficients,
                    target_powers,
                    measured_powers,
                    baseline_matrix,
                    power_to_uw=power_to_uw,
                    optimization_mask=optimization_mask,
                )
            else:
                metrics = closed_loop_loss(target_ratio, measured_powers, eps=args.closed_loop_output_epsilon)
            accepted = bool(
                metrics["loss_mae"] <= best_metrics["loss_mae"] - float(args.closed_loop_min_improvement)
            )
            iter_rows.append(
                build_closed_loop_iter_row(
                    metadata,
                    iteration=iteration,
                    accepted=accepted,
                    learning_rate=lr,
                    coefficients=proposal,
                    voltages=voltages,
                    measured_powers=measured_powers,
                    target_ratio=logged_target,
                    metrics=metrics,
                )
            )
            if args.print_each_sample:
                print_closed_loop_live(metadata, iteration, accepted, voltages, measured_powers, metrics)

            if accepted:
                best_coefficients = proposal
                best_voltages = voltages
                best_metrics = metrics
                best_powers = np.asarray(measured_powers, dtype=float)
                current_state_is_best = True
                lr = min(float(args.closed_loop_max_lr), lr * float(args.closed_loop_lr_grow))
                converged = bool(best_metrics["loss_mae"] <= float(args.closed_loop_tol))
            else:
                current_state_is_best = False
                lr *= float(args.closed_loop_lr_shrink)
                if lr < float(args.closed_loop_min_lr):
                    break

        # If the last uploaded state was rejected, restore the accepted best
        # state before recording the final voltages and moving to the next sample.
        if not current_state_is_best:
            final_powers, final_records, final_failures = upload_coefficients_and_measure(
                args,
                working_data,
                hardware,
                switch_rows_by_mzi,
                best_coefficients,
                baseline_matrix,
            )
            for item in final_failures:
                row = dict(item)
                row.update(metadata)
                row["iteration"] = "final_restore"
                current_check_rows.append(row)
            best_voltages = records_to_voltage_array(final_records, args.input_count)
            best_powers = np.asarray(final_powers, dtype=float)
            if absolute_total_objective:
                best_metrics = closed_loop_absolute_total_power_loss(
                    target_coefficients,
                    best_powers,
                    args.input_reference_power_uw,
                    optimization_mask,
                    power_to_uw=power_to_uw,
                )
                best_metrics["power_to_uw"] = power_to_uw
            elif absolute_baseline_objective:
                best_metrics = closed_loop_absolute_baseline_loss(
                    target_coefficients,
                    target_powers,
                    best_powers,
                    baseline_matrix,
                    power_to_uw=power_to_uw,
                    optimization_mask=optimization_mask,
                )
            else:
                best_metrics = closed_loop_loss(target_ratio, best_powers, eps=args.closed_loop_output_epsilon)
        converged = bool(best_metrics["loss_mae"] <= float(args.closed_loop_tol))
        if absolute_objective and np.any(np.abs(best_coefficients[~optimization_mask]) > 1e-12):
            raise RuntimeError(
                f"sample {metadata['sample_index']} attempted to activate a zero-target input: "
                f"{best_coefficients.tolist()}"
            )

        sample_outputs.append(best_powers)
        final_row = dict(metadata)
        final_row.update(
            {
                "converged": bool(converged),
                "iterations_executed": int(iterations_executed),
                "final_loss_mae": float(best_metrics["loss_mae"]),
                "final_loss_mse": float(best_metrics["loss_mse"]),
                "final_max_abs_ratio_error": float(best_metrics["max_abs_ratio_error"]),
                "final_output_power_sum": float(best_metrics["output_sum"]),
                "initial_loss_mae": initial_loss_mae,
                "final_learning_rate": float(lr),
            }
        )
        append_vector_columns(final_row, "target_ratio", logged_target)
        append_vector_columns(final_row, "final_ratio", best_metrics["measured_ratio"])
        append_vector_columns(final_row, "ratio_error", best_metrics["error"])
        append_vector_columns(final_row, "initial_coefficient", initial_coefficients)
        append_vector_columns(final_row, "optimized_coefficient", best_coefficients)
        append_vector_columns(final_row, "initial_voltage", initial_voltages)
        append_vector_columns(final_row, "final_voltage", best_voltages)
        append_vector_columns(final_row, "final_output_power", best_powers)
        if absolute_objective:
            if absolute_baseline_objective:
                append_vector_columns(final_row, "target_output_power", target_powers)
                append_vector_columns(final_row, "output_power_error", best_metrics["power_error"])
            if absolute_total_objective:
                final_row["target_total_power_uw"] = float(best_metrics["target_total_power_uw"])
                final_row["measured_total_power_uw"] = float(best_metrics["measured_total_power_uw"])
                final_row["total_power_error_uw"] = float(best_metrics["total_power_error_uw"])
            final_row["final_output_power_mae_uw"] = float(best_metrics["power_mae_uw"])
            final_row["final_output_power_max_abs_uw"] = float(best_metrics["power_max_abs_uw"])
        final_voltage_rows.append(final_row)

        for ch in range(int(args.input_count)):
            long_row = dict(metadata)
            long_row.update(
                {
                    "input_channel": int(ch + 1),
                    "target_ratio": float(logged_target[ch]),
                    "final_ratio": float(best_metrics["measured_ratio"][ch]),
                    "ratio_error": float(best_metrics["error"][ch]),
                    "initial_coefficient": float(initial_coefficients[ch]),
                    "optimized_coefficient": float(best_coefficients[ch]),
                    "initial_voltage": float(initial_voltages[ch]),
                    "final_voltage": float(best_voltages[ch]),
                    "final_output_power": float(best_powers[ch]),
                    "converged": bool(converged),
                    "final_loss_mae": float(best_metrics["loss_mae"]),
                }
            )
            final_voltage_long_rows.append(long_row)

        if args.print_each_sample:
            print(
                f"sample {metadata['sample_index']}: closed-loop loss={best_metrics['loss_mae']:.6g}, "
                f"max_err={best_metrics['max_abs_ratio_error']:.6g}, converged={converged}"
            )

        if should_record_results(args):
            atomic_write_rows(run_dir / "closed_loop_input_final_voltage_long.csv", final_voltage_long_rows)
            atomic_write_rows(run_dir / "closed_loop_input_iter_log.csv", iter_rows)
            if current_check_rows:
                atomic_write_rows(run_dir / "closed_loop_input_current_check_failures.csv", current_check_rows)
            # Write the completion marker last. Resume skips a sample only when
            # its final-voltage row has been durably committed.
            atomic_write_rows(run_dir / "closed_loop_input_final_voltage.csv", final_voltage_rows)
            print(f"Checkpoint saved: {len(final_voltage_rows)}/{len(features)} complete.", flush=True)

    if hardware is not None:
        set_all_switches(working_data, args.input_count, "OFF")
        um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)

    if should_record_results(args):
        atomic_write_rows(run_dir / "closed_loop_input_final_voltage.csv", final_voltage_rows)
        atomic_write_rows(run_dir / "closed_loop_input_final_voltage_long.csv", final_voltage_long_rows)
        atomic_write_rows(run_dir / "closed_loop_input_iter_log.csv", iter_rows)
    if current_check_rows:
        if should_record_results(args):
            pd.DataFrame(current_check_rows).to_csv(run_dir / "closed_loop_input_current_check_failures.csv", index=False)
        if args.fail_on_switch_current_failure:
            raise RuntimeError(
                "Closed-loop switch current check failed; see closed_loop_input_current_check_failures.csv."
            )

    return (
        np.asarray(sample_outputs, dtype=float),
        final_voltage_long_rows,
        current_check_rows,
        final_voltage_rows,
        iter_rows,
    )


def build_expected_input_table(
    source_df,
    features,
    feature_validation,
    input_count,
    renormalize_features=False,
    input_unit_power=1.0,
):
    features_for_expected = np.array(features, dtype=float, copy=True)
    if renormalize_features:
        sums = np.sum(features_for_expected, axis=1, keepdims=True)
        features_for_expected = np.divide(
            features_for_expected,
            sums,
            out=np.zeros_like(features_for_expected),
            where=sums != 0.0,
        )

    switch_on_unit = np.ones(int(input_count), dtype=float) * float(input_unit_power)
    expected_input_power = features_for_expected * switch_on_unit.reshape(1, -1)

    metadata_cols = [col for col in ("sample_index", "label", "prediction", "correct") if col in source_df.columns]
    out = source_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=source_df.index)
    for idx in range(int(input_count)):
        out[f"feature_{idx}"] = features[:, idx]
    for idx in range(int(input_count)):
        out[f"expected_input_power_{idx}"] = expected_input_power[:, idx]

    out["feature_sum"] = feature_validation["sums"]
    out["expected_input_power_sum"] = np.sum(expected_input_power, axis=1)
    out["feature_min"] = feature_validation["min_values"]
    out["feature_max"] = feature_validation["max_values"]
    out["feature_valid"] = feature_validation["valid"]
    out["finite_ok"] = feature_validation["finite"]
    out["nonnegative_ok"] = feature_validation["nonnegative"]
    out["sum_ok"] = feature_validation["sum_ok"]
    out["upper_bound_ok"] = feature_validation["upper_bound_ok"]
    out["nonzero_ok"] = feature_validation["nonzero"]
    out["feature_validation_mode"] = feature_validation["mode"]
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Step-1 MNIST input test. In pass-through mode the backend network MZIs are set to Bar. "
            "The input switch MZIs can be driven by ON-normalized, voltage-linear, binary, or fixed-uW "
            "reference power mapping."
        )
    )
    parser.add_argument("--features-csv", default="features_test.csv")
    parser.add_argument("--out-dir", default=os.path.join("results", "MNISTInputPassThroughTest"))
    parser.add_argument(
        "--resume-run-dir",
        default="",
        help="Resume an interrupted recorded run and skip samples already checkpointed.",
    )
    parser.add_argument(
        "--retry-nonconverged",
        type=parse_bool,
        default=False,
        help="With --resume-run-dir, discard and recalibrate only rows whose converged flag is false.",
    )
    parser.add_argument("--N", type=int, default=9)
    parser.add_argument("--mzi-table", default=os.path.join("Scandata", "MZI_table.json"))
    parser.add_argument("--switch-table", default="IN_MZI.txt")
    parser.add_argument("--inference-voltage-file", default=DEFAULT_INFERENCE_VOLTAGE_FILE)
    parser.add_argument(
        "--network-mode",
        choices=["pass-through", "inference-file"],
        default="pass-through",
        help="pass-through sets all network MZIs to Bar; inference-file loads --inference-voltage-file for the backend network.",
    )
    parser.add_argument(
        "--run-inference",
        type=parse_bool,
        default=False,
        help="Save MNIST inference predictions from measured sample outputs. Prediction is argmax output power.",
    )
    parser.add_argument(
        "--class-index-offset",
        type=int,
        default=0,
        help="Predicted class = argmax output index + this offset. Use 0 for output_power_0..7 labels.",
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many rows from --features-csv before applying --sample-limit.",
    )
    parser.add_argument("--sum-tol", type=float, default=1e-3)
    parser.add_argument(
        "--feature-validation-mode",
        choices=["normalized", "bounded"],
        default="normalized",
        help=(
            "normalized requires each vector sum to one; bounded accepts absolute-intensity "
            "vectors whose elements are in [0, 1] and whose total is nonzero."
        ),
    )
    parser.add_argument("--negative-tol", type=float, default=1e-12)
    parser.add_argument("--renormalize-features", type=parse_bool, default=False)
    parser.add_argument("--fail-on-invalid-feature", type=parse_bool, default=True)
    parser.add_argument("--dry-run", type=parse_bool, default=True)
    parser.add_argument(
        "--record-dry-run",
        type=parse_bool,
        default=False,
        help="When false, dry-run files are written only to a temporary directory and deleted at exit.",
    )
    parser.add_argument("--confirm-hardware", type=parse_bool, default=False)
    parser.add_argument("--measure-switch-on", type=parse_bool, default=True)
    parser.add_argument(
        "--measure-feature-samples",
        type=parse_bool,
        default=True,
        help="After ON baseline, upload each feature vector to the input switches and measure output powers.",
    )
    parser.add_argument(
        "--sequential-input-sum",
        type=parse_bool,
        default=False,
        help=(
            "For each sample, measure one input channel at a time with all other inputs OFF, "
            "then sum the 8 measured output-power vectors before inference."
        ),
    )
    parser.add_argument(
        "--input-upload-mode",
        choices=[
            "scan-leak-inverse",
            "scan-leak-fixed-power",
            "voltage-linear",
            "binary-threshold",
            "voltage-table",
        ],
        default="scan-leak-inverse",
        help=(
            "Map feature coefficient to switch voltage. scan-leak-inverse uses old other-port leak scans: "
            "feature=1 -> ON/min leak, feature=0 -> OFF/max leak. scan-leak-fixed-power uses "
            "--input-reference-power-uw as feature=1 target chip input power. voltage-table uploads per-sample "
            "voltages from --sample-input-voltage-file."
        ),
    )
    parser.add_argument(
        "--sample-input-voltage-file",
        default="",
        help=(
            "CSV containing per-sample input voltages, normally closed_loop_input_final_voltage.csv "
            "with final_voltage_0..final_voltage_7 columns."
        ),
    )
    parser.add_argument(
        "--input-reference-power-uw",
        type=float,
        default=DEFAULT_INPUT_REFERENCE_POWER_UW,
        help=(
            "For --input-upload-mode scan-leak-fixed-power, feature coefficient c maps to "
            "target chip input power c * this value in uW."
        ),
    )
    parser.add_argument(
        "--switch-leak-scan-dir",
        default=DEFAULT_SWITCH_LEAK_SCAN_DIR,
        help="Directory containing old switch-MZI other-port leak scans named by port, e.g. 43.txt.",
    )
    parser.add_argument("--switch-leak-lookup-step", type=float, default=0.01)
    parser.add_argument("--print-each-sample", type=parse_bool, default=False)
    parser.add_argument("--diagonal-floor", type=float, default=1e-12)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=5.5)
    parser.add_argument("--ser-address", default=DEFAULT_SER_ADDRESS)
    parser.add_argument("--opm2-address", default=DEFAULT_OPM2_ADDRESS)
    parser.add_argument("--switch-current-check", type=parse_bool, default=True)
    parser.add_argument("--switch-current-tolerance", type=float, default=0.10)
    parser.add_argument("--fail-on-switch-current-failure", type=parse_bool, default=False)
    parser.add_argument("--identity-power-threshold", type=float, default=0.5)
    parser.add_argument("--fail-on-routing-mismatch", type=parse_bool, default=False)
    parser.add_argument(
        "--closed-loop-input",
        type=parse_bool,
        default=False,
        help=(
            "Optimize input-switch voltages sample by sample with backend network fixed to Bar. "
            "The objective is measured output-power ratio ~= feature vector."
        ),
    )
    parser.add_argument("--closed-loop-max-iters", type=int, default=20)
    parser.add_argument(
        "--closed-loop-objective",
        choices=["ratio", "absolute-baseline", "absolute-total-power"],
        default="ratio",
        help=(
            "ratio matches normalized output proportions; absolute-baseline preserves feature amplitude "
            "and matches output powers predicted by the measured fixed-reference baseline matrix; "
            "absolute-total-power matches sum(outputs) to coefficient * reference uW."
        ),
    )
    parser.add_argument("--closed-loop-lr", type=float, default=0.35)
    parser.add_argument("--closed-loop-max-lr", type=float, default=1.0)
    parser.add_argument("--closed-loop-min-lr", type=float, default=0.01)
    parser.add_argument("--closed-loop-lr-grow", type=float, default=1.05)
    parser.add_argument("--closed-loop-lr-shrink", type=float, default=0.5)
    parser.add_argument(
        "--closed-loop-tol",
        type=float,
        default=0.02,
        help="Stop a sample when mean absolute ratio error is at or below this value.",
    )
    parser.add_argument(
        "--closed-loop-min-improvement",
        type=float,
        default=0.0,
        help="Accept an iteration only if loss decreases by at least this value.",
    )
    parser.add_argument("--closed-loop-output-epsilon", type=float, default=1e-15)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.N < 2:
        raise ValueError("--N must be >= 2")
    if args.v_min > args.v_max:
        raise ValueError("--v-min must be <= --v-max")
    if args.run_inference and not args.measure_feature_samples:
        raise ValueError("--run-inference requires --measure-feature-samples true")
    if args.network_mode == "inference-file" and not Path(args.inference_voltage_file).exists():
        raise FileNotFoundError(f"--inference-voltage-file not found: {args.inference_voltage_file}")
    if args.input_upload_mode == "voltage-table":
        if not args.sample_input_voltage_file:
            raise ValueError("--input-upload-mode voltage-table requires --sample-input-voltage-file.")
        if not Path(args.sample_input_voltage_file).exists():
            raise FileNotFoundError(f"--sample-input-voltage-file not found: {args.sample_input_voltage_file}")
    if args.closed_loop_input:
        if args.network_mode != "pass-through":
            raise ValueError("--closed-loop-input requires --network-mode pass-through because backend must be all Bar.")
        if args.run_inference:
            raise ValueError("--closed-loop-input records input voltages only; do not combine it with --run-inference.")
        if args.input_upload_mode == "voltage-table":
            raise ValueError("--closed-loop-input generates a voltage table; do not combine it with voltage-table upload.")
        if args.input_upload_mode not in {"scan-leak-inverse", "scan-leak-fixed-power"}:
            raise ValueError(
                "--closed-loop-input requires --input-upload-mode scan-leak-inverse or scan-leak-fixed-power."
            )
        if int(args.closed_loop_max_iters) < 0:
            raise ValueError("--closed-loop-max-iters must be >= 0")
        for name in (
            "closed_loop_lr",
            "closed_loop_max_lr",
            "closed_loop_min_lr",
            "closed_loop_lr_grow",
            "closed_loop_lr_shrink",
            "closed_loop_tol",
            "closed_loop_output_epsilon",
        ):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
        if float(args.closed_loop_lr_shrink) >= 1.0:
            raise ValueError("--closed-loop-lr-shrink must be < 1.")
        if float(args.closed_loop_min_improvement) < 0.0:
            raise ValueError("--closed-loop-min-improvement must be >= 0.")

    args.input_count = args.N - 1
    args.output_count = args.N - 1
    mzi_count = args.N * (args.N - 1) // 2
    input_unit_power = float(args.input_reference_power_uw) if is_fixed_reference_power_mode(args.input_upload_mode) else 1.0

    args.record_results = bool(not args.dry_run or args.record_dry_run)
    if args.record_results:
        if args.resume_run_dir:
            run_dir = Path(args.resume_run_dir).resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"--resume-run-dir does not exist: {run_dir}")
        else:
            run_dir = create_run_dir(args.out_dir)
    else:
        run_dir = None
    args.run_dir = str(run_dir) if run_dir is not None else ""
    config = vars(args).copy()
    config["mzi_count"] = int(mzi_count)
    config["dry_run_recorded"] = bool(args.record_results)
    config["step_note"] = (
        "inference_voltage_file is recorded for traceability only; it is not loaded in this step."
    )
    if args.record_results:
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    feature_df, feature_cols, features = load_features(
        args.features_csv,
        args.input_count,
        sample_limit=args.sample_limit,
        sample_offset=args.sample_offset,
    )
    feature_validation = validate_features(
        features,
        sum_tol=args.sum_tol,
        negative_tol=args.negative_tol,
        mode=args.feature_validation_mode,
    )
    expected_table = build_expected_input_table(
        feature_df,
        features,
        feature_validation,
        args.input_count,
        renormalize_features=args.renormalize_features,
        input_unit_power=input_unit_power,
    )
    if args.record_results:
        expected_table.to_csv(run_dir / "expected_input_power.csv", index=False)

    invalid_table = expected_table.loc[~expected_table["feature_valid"]].copy()
    if not invalid_table.empty:
        if args.record_results:
            invalid_table.to_csv(run_dir / "invalid_feature_rows.csv", index=False)
        if args.fail_on_invalid_feature:
            if args.record_results:
                raise ValueError(
                    f"Found {len(invalid_table)} invalid feature rows; see {run_dir / 'invalid_feature_rows.csv'}"
                )
            raise ValueError(f"Found {len(invalid_table)} invalid feature rows; dry-run is not recorded.")

    mzi_table = um.load_mzi_table(args.mzi_table)
    working_data = make_zero_working_data()
    switch_table = um.load_switch_mzi_table(args.switch_table)
    switch_safety_records = validate_switch_voltage_table(
        switch_table,
        args.switch_table,
        args.input_count,
        args.v_min,
        args.v_max,
    )
    if args.record_results:
        pd.DataFrame(switch_safety_records).to_csv(run_dir / "switch_voltage_safety_check.csv", index=False)
    switch_rows_by_mzi = load_switch_rows_by_mzi(args.switch_table, args.input_count, args.v_min, args.v_max)
    args.switch_leak_inverse_tables = {}
    inverse_summary_rows = []
    reference_feasibility_rows = []
    if args.input_upload_mode in {"scan-leak-inverse", "scan-leak-fixed-power"}:
        args.switch_leak_inverse_tables, inverse_summary_rows = build_switch_leak_inverse_tables(
            switch_rows_by_mzi,
            args.switch_leak_scan_dir,
            args.input_count,
            args.v_min,
            args.v_max,
        )
        if args.record_results:
            pd.DataFrame(inverse_summary_rows).to_csv(run_dir / "switch_leak_inverse_summary.csv", index=False)
        inverse_lookup_rows = build_switch_leak_inverse_lookup_rows(
            args.switch_leak_inverse_tables,
            step=args.switch_leak_lookup_step,
        )
        if args.record_results:
            pd.DataFrame(inverse_lookup_rows).to_csv(run_dir / "switch_leak_inverse_lookup.csv", index=False)
        if is_fixed_reference_power_mode(args.input_upload_mode):
            reference_feasibility_rows = validate_fixed_reference_power(
                args.switch_leak_inverse_tables,
                args.input_reference_power_uw,
            )
            if args.record_results:
                pd.DataFrame(reference_feasibility_rows).to_csv(
                    run_dir / "switch_fixed_power_reference_feasibility.csv",
                    index=False,
                )

    network_records = []
    if args.network_mode == "pass-through":
        network_records = set_network_to_bar_pass_through(
            working_data,
            mzi_table,
            mzi_count,
            args.v_min,
            args.v_max,
        )
        if args.record_results:
            pd.DataFrame(network_records).to_csv(run_dir / "pass_through_mzi_plan.csv", index=False)
    elif args.network_mode == "inference-file":
        network_records = load_voltage_state_file(
            args.inference_voltage_file,
            working_data,
            args.v_min,
            args.v_max,
        )
        if args.record_results:
            pd.DataFrame(network_records).to_csv(run_dir / "inference_voltage_loaded.csv", index=False)
    else:
        raise ValueError(f"Unsupported network mode: {args.network_mode!r}")

    # The loaded inference voltage file may contain stale switch states. Always
    # force all input switches OFF before per-sample feature voltages are applied.
    set_all_switches(working_data, args.input_count, "OFF")
    validate_all_voltages(working_data, args.v_min, args.v_max)
    initial_network_voltage_path = (
        run_dir / ("pass_through_voltage.csv" if args.network_mode == "pass-through" else "inference_network_voltage.csv")
        if args.record_results
        else None
    )
    save_voltage_state(initial_network_voltage_path, working_data)

    hardware = initialize_hardware(args)
    route_rows = []
    reference_route_rows = []
    baseline_raw = build_dry_run_baseline(args.input_count)
    baseline_norm = build_dry_run_baseline(args.input_count)
    comparison_baseline_raw = baseline_raw
    if hardware is not None:
        um.upload_v_checked(hardware["mcv"], working_data, args.v_min, args.v_max)
        if args.measure_switch_on:
            baseline_raw, baseline_norm, route_rows = measure_switch_on_transmission(args, working_data, hardware, run_dir)
    else:
        if args.record_results:
            np.savetxt(run_dir / "switch_on_power_matrix_raw.csv", baseline_raw, delimiter=",")
            np.savetxt(run_dir / "switch_on_power_matrix_norm.csv", baseline_norm, delimiter=",")

    total_power_objective = bool(
        args.closed_loop_input and args.closed_loop_objective == "absolute-total-power"
    )
    if is_fixed_reference_power_mode(args.input_upload_mode) and not total_power_objective:
        if hardware is not None:
            reference_raw, reference_norm, reference_route_rows = measure_reference_power_transmission(
                args,
                working_data,
                hardware,
                run_dir,
            )
        else:
            reference_raw = build_dry_run_baseline(args.input_count) * float(args.input_reference_power_uw)
            reference_norm = normalize_columns(reference_raw)
            reference_route_rows = build_route_check_rows(reference_raw, args.identity_power_threshold)
            if args.record_results:
                np.savetxt(run_dir / "switch_reference_power_matrix_raw.csv", reference_raw, delimiter=",")
                np.savetxt(run_dir / "switch_reference_power_matrix_norm.csv", reference_norm, delimiter=",")
                pd.DataFrame(reference_route_rows).to_csv(run_dir / "switch_reference_route_check.csv", index=False)
        comparison_baseline_raw = reference_raw
    elif total_power_objective:
        # The total-power loop has an explicit physical target in uW and must
        # not derive its target from a measured baseline whose reference plane
        # may use a different scale.
        comparison_baseline_raw = (
            build_dry_run_baseline(args.input_count)
            * float(args.input_reference_power_uw)
            * (1.0 if hardware is None else 1e-6)
        )
    else:
        comparison_baseline_raw = baseline_raw

    sample_outputs = None
    comparison_metrics = None
    closed_loop_final_rows = []
    closed_loop_iter_rows = []
    if args.closed_loop_input:
        (
            sample_outputs,
            voltage_records,
            sample_current_failures,
            closed_loop_final_rows,
            closed_loop_iter_rows,
        ) = optimize_input_vectors_closed_loop(
            args,
            working_data,
            hardware,
            run_dir,
            feature_df,
            features,
            comparison_baseline_raw,
        )
        if args.record_results:
            build_sample_output_dataframe(sample_outputs, feature_df).to_csv(
                run_dir / "sample_output_power_raw.csv",
                index=False,
            )
        comparison_table, comparison_metrics = build_input_comparison_table(
            feature_df,
            features,
            sample_outputs,
            comparison_baseline_raw,
            diagonal_floor=args.diagonal_floor,
        )
        if args.record_results:
            comparison_table.to_csv(run_dir / "input_vector_comparison.csv", index=False)
        if voltage_records and args.record_results:
            pd.DataFrame(voltage_records).to_csv(run_dir / "sample_switch_voltage_plan.csv", index=False)
    elif args.input_upload_mode == "voltage-table":
        sample_outputs, voltage_records, sample_current_failures = measure_samples_from_voltage_table(
            args,
            working_data,
            hardware,
            run_dir,
            feature_df,
            features,
            comparison_baseline_raw,
        )
        if args.record_results:
            build_sample_output_dataframe(sample_outputs, feature_df).to_csv(
                run_dir / "sample_output_power_raw.csv",
                index=False,
            )
        if args.network_mode == "pass-through":
            comparison_table, comparison_metrics = build_input_comparison_table(
                feature_df,
                features,
                sample_outputs,
                comparison_baseline_raw,
                diagonal_floor=args.diagonal_floor,
            )
            if args.record_results:
                comparison_table.to_csv(run_dir / "input_vector_comparison.csv", index=False)
        if voltage_records and args.record_results:
            pd.DataFrame(voltage_records).to_csv(run_dir / "sample_switch_voltage_plan.csv", index=False)
        if sample_current_failures:
            if args.record_results:
                pd.DataFrame(sample_current_failures).to_csv(
                    run_dir / "sample_switch_current_check_failures.csv",
                    index=False,
                )
            if args.fail_on_switch_current_failure:
                raise RuntimeError(
                    "Sample switch current check failed; see sample_switch_current_check_failures.csv."
                )
    elif args.measure_feature_samples:
        sample_outputs, voltage_records, sample_current_failures = measure_feature_samples(
            args,
            working_data,
            hardware,
            run_dir,
            feature_df,
            features,
            comparison_baseline_raw,
        )
        if args.record_results:
            build_sample_output_dataframe(sample_outputs, feature_df).to_csv(
                run_dir / "sample_output_power_raw.csv",
                index=False,
            )
        comparison_table, comparison_metrics = build_input_comparison_table(
            feature_df,
            features,
            sample_outputs,
            comparison_baseline_raw,
            diagonal_floor=args.diagonal_floor,
        )
        if args.record_results:
            comparison_table.to_csv(run_dir / "input_vector_comparison.csv", index=False)
        if voltage_records and args.record_results:
            pd.DataFrame(voltage_records).to_csv(run_dir / "sample_switch_voltage_plan.csv", index=False)
        if sample_current_failures:
            if args.record_results:
                pd.DataFrame(sample_current_failures).to_csv(
                    run_dir / "sample_switch_current_check_failures.csv",
                    index=False,
                )
            if args.fail_on_switch_current_failure:
                raise RuntimeError(
                    "Sample switch current check failed; see sample_switch_current_check_failures.csv."
                )

    summary = {
        "features_csv": os.fspath(args.features_csv),
        "sample_count": int(len(feature_df)),
        "feature_columns": feature_cols,
        "feature_validation_mode": str(feature_validation["mode"]),
        "invalid_feature_rows": int((~feature_validation["valid"]).sum()),
        "feature_sum_min": float(np.min(feature_validation["sums"])),
        "feature_sum_max": float(np.max(feature_validation["sums"])),
        "expected_input_power_rule": (
            f"expected_input_power_i = feature_i * {input_unit_power:.12g}"
            + (" uW" if is_fixed_reference_power_mode(args.input_upload_mode) else "")
        ),
        "input_reference_power_uw": (
            float(args.input_reference_power_uw)
            if is_fixed_reference_power_mode(args.input_upload_mode)
            else None
        ),
        "closed_loop_input": bool(args.closed_loop_input),
        "closed_loop_max_iters": int(args.closed_loop_max_iters),
        "closed_loop_lr": float(args.closed_loop_lr),
        "closed_loop_tol": float(args.closed_loop_tol),
        "network_mode": str(args.network_mode),
        "inference_voltage_file": os.fspath(args.inference_voltage_file),
        "inference_voltage_file_loaded": bool(args.network_mode == "inference-file"),
        "initial_network_voltage_file": (
            os.fspath(initial_network_voltage_path) if initial_network_voltage_path is not None else None
        ),
        "switch_voltage_safety_check_file": result_path(args, run_dir, "switch_voltage_safety_check.csv"),
        "input_upload_mode": str(args.input_upload_mode),
        "sequential_input_sum": bool(args.sequential_input_sum),
        "sequential_input_component_outputs_file": (
            result_path(args, run_dir, "sequential_input_component_outputs.csv")
            if args.sequential_input_sum
            else None
        ),
        "sample_input_voltage_file": (
            os.fspath(args.sample_input_voltage_file)
            if args.input_upload_mode == "voltage-table"
            else None
        ),
        "switch_leak_scan_dir": os.fspath(args.switch_leak_scan_dir),
        "switch_leak_inverse_summary_file": (
            result_path(args, run_dir, "switch_leak_inverse_summary.csv")
            if args.input_upload_mode in {"scan-leak-inverse", "scan-leak-fixed-power"}
            else None
        ),
        "switch_leak_inverse_lookup_file": (
            result_path(args, run_dir, "switch_leak_inverse_lookup.csv")
            if args.input_upload_mode in {"scan-leak-inverse", "scan-leak-fixed-power"}
            else None
        ),
        "switch_fixed_power_reference_feasibility_file": (
            result_path(args, run_dir, "switch_fixed_power_reference_feasibility.csv")
            if is_fixed_reference_power_mode(args.input_upload_mode)
            else None
        ),
        "switch_reference_power_matrix_file": (
            result_path(args, run_dir, "switch_reference_power_matrix_raw.csv")
            if is_fixed_reference_power_mode(args.input_upload_mode)
            else None
        ),
        "comparison_baseline_matrix_file": (
            result_path(args, run_dir, "switch_reference_power_matrix_raw.csv")
            if is_fixed_reference_power_mode(args.input_upload_mode)
            else result_path(args, run_dir, "switch_on_power_matrix_raw.csv")
        ),
        "safe_voltage_range_v": [float(args.v_min), float(args.v_max)],
        "hardware_mode": bool(not args.dry_run),
        "dry_run_recorded": bool(not args.dry_run or args.record_dry_run),
        "switch_on_measured": bool(hardware is not None and args.measure_switch_on),
        "route_identity_ok_count": int(sum(1 for row in route_rows if row.get("identity_route_ok"))),
        "route_identity_total": int(len(route_rows)),
        "reference_route_identity_ok_count": int(
            sum(1 for row in reference_route_rows if row.get("identity_route_ok"))
        ),
        "reference_route_identity_total": int(len(reference_route_rows)),
        "feature_samples_measured": bool(args.measure_feature_samples),
    }
    if args.input_upload_mode == "voltage-table":
        summary["sample_input_voltage_from_table_file"] = result_path(
            args,
            run_dir,
            "sample_input_voltage_from_table.csv",
        )
    if args.closed_loop_input:
        final_df = pd.DataFrame(closed_loop_final_rows)
        summary.update(
            {
                "feature_samples_measured": bool(sample_outputs is not None),
                "closed_loop_final_voltage_file": result_path(args, run_dir, "closed_loop_input_final_voltage.csv"),
                "closed_loop_final_voltage_long_file": result_path(
                    args,
                    run_dir,
                    "closed_loop_input_final_voltage_long.csv",
                ),
                "closed_loop_iter_log_file": result_path(args, run_dir, "closed_loop_input_iter_log.csv"),
                "closed_loop_converged_count": int(final_df["converged"].sum()) if not final_df.empty else 0,
                "closed_loop_total": int(len(final_df)),
                "closed_loop_final_loss_mae_mean": (
                    float(final_df["final_loss_mae"].mean()) if not final_df.empty else None
                ),
                "closed_loop_final_loss_mae_max": (
                    float(final_df["final_loss_mae"].max()) if not final_df.empty else None
                ),
                "closed_loop_final_max_abs_ratio_error": (
                    float(final_df["final_max_abs_ratio_error"].max()) if not final_df.empty else None
                ),
            }
        )
    if sample_outputs is not None and comparison_metrics is not None:
        diag_error = comparison_metrics["diag_est"] - features
        lstsq_error = comparison_metrics["lstsq_est"] - features
        output_error = comparison_metrics["output_error"]
        summary.update(
            {
                "sample_output_power_file": result_path(args, run_dir, "sample_output_power_raw.csv"),
                "input_vector_comparison_file": result_path(args, run_dir, "input_vector_comparison.csv"),
                "diag_feature_mae": float(np.nanmean(np.abs(diag_error))),
                "diag_feature_max_abs_error": float(np.nanmax(np.abs(diag_error))),
                "lstsq_feature_mae": float(np.mean(np.abs(lstsq_error))),
                "lstsq_feature_max_abs_error": float(np.max(np.abs(lstsq_error))),
                "output_power_mae_vs_csv_expected": float(np.mean(np.abs(output_error))),
                "output_power_max_abs_error_vs_csv_expected": float(np.max(np.abs(output_error))),
            }
        )
    if args.run_inference and sample_outputs is not None:
        inference_table = build_inference_result_table(
            feature_df,
            sample_outputs,
            class_index_offset=args.class_index_offset,
        )
        inference_path = run_dir / "mnist_inference_results.csv" if args.record_results else None
        if args.record_results:
            inference_table.to_csv(inference_path, index=False)
        summary["mnist_inference_results_file"] = (
            os.fspath(inference_path) if inference_path is not None else None
        )
        if "inference_correct" in inference_table.columns:
            summary["mnist_inference_accuracy"] = float(inference_table["inference_correct"].mean())
            summary["mnist_inference_correct_count"] = int(inference_table["inference_correct"].sum())
            summary["mnist_inference_total"] = int(len(inference_table))
    if args.record_results:
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if args.dry_run and not args.record_dry_run:
        print("Dry-run completed; no result directory recorded.")
    else:
        print(f"Saved run directory: {run_dir}")
    print(f"Samples checked: {len(feature_df)}")
    print(f"Invalid feature rows: {summary['invalid_feature_rows']}")
    if args.network_mode == "inference-file":
        print(f"Inference voltage matrix loaded: {args.inference_voltage_file}")
    else:
        print("Inference voltage matrix was not loaded in this step.")
    print(f"Expected input rule: {summary['expected_input_power_rule']}")
    if is_fixed_reference_power_mode(args.input_upload_mode):
        print(f"Fixed input reference power: {float(args.input_reference_power_uw):.6g} uW")
    if args.closed_loop_input:
        print(
            f"Closed-loop input convergence: {summary['closed_loop_converged_count']}/"
            f"{summary['closed_loop_total']}, mean loss={summary['closed_loop_final_loss_mae_mean']:.6g}"
        )
    if args.run_inference and "mnist_inference_accuracy" in summary:
        print(
            f"MNIST inference accuracy: {summary['mnist_inference_correct_count']}/"
            f"{summary['mnist_inference_total']} = {summary['mnist_inference_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
