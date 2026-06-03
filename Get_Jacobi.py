import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import utils.communication as cu
import utils.AllDecompositionUtils as du
from inter_calibration import find_Bmzi_path, switch_IN, write_port_voltage


DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
DEFAULT_SER_ADDRESS = "COM3"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv_list(text, item_type=str):
    if text is None or str(text).strip() == "":
        return []
    return [item_type(item.strip()) for item in str(text).split(",") if item.strip()]


def validate_heater_order(heaters, mzi_ids):
    normalized = []
    errors = []
    allowed_mzis = {int(mzi_id) for mzi_id in mzi_ids}
    for heater in heaters:
        text = str(heater).strip().lower()
        if len(text) < 2 or text[-1] not in {"u", "d"}:
            errors.append(text)
            continue
        try:
            mzi_id = int(text[:-1])
        except ValueError:
            errors.append(text)
            continue
        if mzi_id not in allowed_mzis:
            errors.append(text)
            continue
        normalized.append(text)
    if errors:
        raise ValueError(
            f"Invalid --heaters entries {errors}. "
            "Use heater labels like 5u,5d,6u,6d,7u,7d,8u,8d."
        )
    duplicates = sorted({heater for heater in normalized if normalized.count(heater) > 1})
    if duplicates:
        raise ValueError(f"Duplicate --heaters entries: {duplicates}")
    expected = [f"{int(mzi_id)}{arm}" for mzi_id in mzi_ids for arm in ("u", "d")]
    missing = [heater for heater in expected if heater not in normalized]
    if missing:
        raise ValueError(
            f"--heaters is missing {missing}. "
            f"Expected all second-column heaters: {','.join(expected)}"
        )
    return normalized


def generate_delta_offsets(args):
    text = str(args.delta_probe_points).strip()
    if "," in text:
        return np.asarray(parse_csv_list(text, float), dtype=float)
    points = int(text)
    if points < 2:
        raise ValueError("--delta_probe_points must be at least 2.")
    return np.linspace(-float(args.delta_probe_half_width_w), float(args.delta_probe_half_width_w), points)


def generate_sigma_phase_points(args):
    phase_text = str(getattr(args, "sigma_phase_points", "")).strip()
    if phase_text:
        return np.asarray(parse_csv_list(phase_text, float), dtype=float)
    points = int(args.sigma_points)
    if points < 2:
        raise ValueError("--sigma_points must be at least 2.")
    return np.linspace(0.0, 2.0 * np.pi, points)


def load_mzi_table(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_voltage_range(working_data, v_min, v_max):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltages.")
    bad = (voltages < float(v_min)) | (voltages > float(v_max))
    if bad.any():
        raise ValueError(f"Voltage out of range [{v_min}, {v_max}] at rows: {bad[bad].index.tolist()}")


def voltage_range_summary(working_data):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltages.")
    min_idx = int(voltages.idxmin())
    max_idx = int(voltages.idxmax())
    return {
        "min_v": float(voltages.iloc[min_idx]),
        "max_v": float(voltages.iloc[max_idx]),
        "min_row": min_idx,
        "max_row": max_idx,
    }


def upload_voltage_checked(mcv, working_data, args, label):
    validate_voltage_range(working_data, 0.0, float(args.voltage_limit_v))
    summary = voltage_range_summary(working_data)
    print(
        "[Get_Jacobi] "
        f"{label}: voltage protection passed, "
        f"min={summary['min_v']:.6f} V(row {summary['min_row']}), "
        f"max={summary['max_v']:.6f} V(row {summary['max_row']}), "
        f"limit=[0.000000, {float(args.voltage_limit_v):.6f}] V"
    )
    cu.upload_voltage(mcv, working_data)


def get_mzi_arm_info(mzi_table, mzi_id, arm):
    mzi_id = int(mzi_id)
    arm = str(arm).lower()
    idx = 0 if arm == "u" else 1
    entry = mzi_table[str(mzi_id)]
    ports = entry.get("ports", [])
    heater_r = entry.get("heater_R", [])
    ppi = entry.get("Ppi", [])
    if len(ports) <= idx or len(heater_r) <= idx:
        raise ValueError(f"MZI {mzi_id} missing port/heater_R for arm {arm}.")
    return {
        "mzi_id": mzi_id,
        "arm": arm,
        "arm_index": idx,
        "arm_name": "upper" if arm == "u" else "lower",
        "port": int(ports[idx]),
        "resistance": float(heater_r[idx]),
        "ppi": float(ppi[idx]) if len(ppi) > idx else np.nan,
    }


def power_to_voltage(power_w, resistance):
    return float(np.sqrt(max(0.0, float(power_w)) * float(resistance)))


def voltage_to_power_w(voltage, resistance):
    return float(float(voltage) ** 2 / float(resistance))


def get_port_voltage(working_data, port):
    return float(working_data.iloc[int(port) - 1, 0])


def get_left_upper_bar_channel(mzi_id, n_value):
    matrix = du.Clements_matrix(int(n_value))
    rows, _ = np.where(matrix == int(mzi_id))
    if rows.size == 0:
        raise ValueError(f"MZI {mzi_id} not found in Clements matrix.")
    return int(rows[0]) + 1


def read_output_power_uW(opm, output_channel):
    values = cu.read_pow(opm)
    idx = int(output_channel) - 1
    if idx < 0 or idx >= len(values):
        raise IndexError(f"Output channel {output_channel} is not available from OPM readout.")
    return float(values[idx]) * 1e6


def fold_power_to_limit(power_w, period_w, power_limit_w):
    power = float(power_w)
    period = float(period_w)
    if period <= 0.0:
        raise ValueError("fold period must be positive.")
    folds = 0
    while power > float(power_limit_w):
        power -= period
        folds += 1
    return max(0.0, power), folds


def parse_probe_map(text, mzi_ids):
    result = {int(mzi_id): "u" for mzi_id in mzi_ids}
    if text is None or str(text).strip() == "":
        return result
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid probe_map item {item!r}; expected like 5:u.")
        mzi_text, arm_text = item.split(":", 1)
        arm = arm_text.strip().lower()
        if arm not in {"u", "d"}:
            raise ValueError(f"Unsupported probe arm {arm!r}; expected u or d.")
        result[int(mzi_text)] = arm
    missing = [int(mzi_id) for mzi_id in mzi_ids if int(mzi_id) not in result]
    if missing:
        raise ValueError(f"probe_map missing MZI ids: {missing}")
    return result


def parse_sigma_bmzi_map(text, mzi_ids):
    if text is None or str(text).strip() == "":
        text = "5:0,6:5,7:6,8:7"
    parsed = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid sigma_bmzi_map item {item!r}; expected like 6:5.")
        mzi_text, bmzi_text = item.split(":", 1)
        parsed[str(int(mzi_text.strip()))] = int(bmzi_text.strip())
    missing = [str(int(mzi_id)) for mzi_id in mzi_ids if str(int(mzi_id)) not in parsed]
    if missing:
        raise ValueError(f"sigma_bmzi_map missing MZI ids: {missing}")
    return {str(int(mzi_id)): int(parsed[str(int(mzi_id))]) for mzi_id in mzi_ids}


def read_power_file(path, heater_order):
    df = pd.read_csv(path)
    if not {"heater", "power_w"}.issubset(df.columns):
        raise ValueError(f"{path} must contain heater,power_w columns.")
    values = {str(row.heater).strip().lower(): float(row.power_w) for row in df.itertuples()}
    missing = [str(heater).strip() for heater in heater_order if str(heater).strip().lower() not in values]
    if missing:
        raise ValueError(
            f"{path} is missing requested heaters {missing}. "
            f"Available heaters: {sorted(values)}. "
            "Use --heaters like 5u,5d,6u,6d,7u,7d,8u,8d, not just MZI ids."
        )
    return {heater: values[str(heater).strip().lower()] for heater in heater_order}


def read_opm_repeated(opm, output_channel, args):
    values = []
    reads = max(1, int(args.opm_reads_per_point))
    for idx in range(reads):
        values.append(read_output_power_uW(opm, int(output_channel)))
        if idx + 1 < reads:
            time.sleep(float(args.opm_read_interval_s))
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    median = float(np.median(arr))
    return {
        "opm_raw_uW": json.dumps([float(v) for v in arr]),
        "opm_mean_uW": mean,
        "opm_std_uW": std,
        "opm_median_uW": median,
        "opm_relative_std": float(std / max(abs(mean), 1e-9)),
        "opm_read_count": int(arr.size),
    }


def read_stable_opm(opm, output_channel, args, reupload_callback=None):
    result = None
    warning = ""
    retry_count = 0
    upload_retry_count = 0
    for attempt in range(int(args.opm_max_retry_per_point) + 1):
        result = read_opm_repeated(opm, output_channel, args)
        if result["opm_relative_std"] <= float(args.opm_relative_std_threshold):
            warning = ""
            break
        warning = "unstable OPM reading"
        retry_count = attempt + 1
        if attempt < int(args.opm_max_retry_per_point) and reupload_callback is not None and parse_bool(args.reupload_on_unstable_point):
            upload_retry_count += 1
            reupload_callback()
            time.sleep(float(args.point_settle_time_s))
    result["warning"] = warning
    result["point_warning"] = warning
    result["opm_retry_count"] = int(retry_count)
    result["upload_retry_count"] = int(upload_retry_count)
    return result


def write_port_power(working_data, port, resistance, power_w):
    voltage = float(power_to_voltage(float(max(0.0, power_w)), float(resistance)))
    write_port_voltage(int(port), voltage, working_data)
    return voltage


QUALITY_COLUMNS = [
    "stage",
    "perturb_label",
    "scan_kind",
    "observed_mzi",
    "scan_file",
    "valid",
    "try_count",
    "point_count",
    "median_power_uW",
    "min_power_uW",
    "max_power_uW",
    "curve_range_uW",
    "unstable_point_count",
    "near_zero_point_count",
    "neighbor_jump_count",
    "edge_jump_count",
    "route_lower_policy",
    "route_lower_zero_applied",
    "warning",
    "failure_reason",
]


FAILED_COLUMNS = [
    "stage",
    "perturb_label",
    "scan_kind",
    "observed_mzi",
    "failed_files",
    "final_failure_reason",
    "suggested_action",
]


def append_csv_row(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{col: row.get(col, "") for col in columns}]).to_csv(
        path,
        mode="a",
        index=False,
        header=not path.exists(),
    )


def first_nonempty_df_value(df, column, default=""):
    if column not in df.columns:
        return default
    for value in df[column].tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value
    return default


def parse_bool_cell(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def build_quality_row(stage, perturb_label, scan_kind, observed_mzi, scan_file, try_count, df, valid, warning, failure_reason, counts):
    y = pd.to_numeric(df["optical_power_uW"], errors="coerce").to_numpy(dtype=float)
    if y.size == 0 or np.isnan(y).all():
        median_y = min_y = max_y = curve_range = np.nan
    else:
        median_y = float(np.nanmedian(y))
        min_y = float(np.nanmin(y))
        max_y = float(np.nanmax(y))
        curve_range = float(max_y - min_y)
    route_lower_policy = first_nonempty_df_value(df, "route_lower_policy", "")
    route_lower_zero_applied = ""
    if "route_lower_zero_applied" in df.columns:
        parsed = [parse_bool_cell(value) for value in df["route_lower_zero_applied"].tolist()]
        parsed = [value for value in parsed if value is not None]
        if parsed:
            route_lower_zero_applied = bool(all(parsed))
    route_warning = str(first_nonempty_df_value(df, "route_warning", "")).strip()
    merged_warning = "; ".join([text for text in (str(warning).strip(), route_warning) if text])
    return {
        "stage": stage,
        "perturb_label": perturb_label,
        "scan_kind": scan_kind,
        "observed_mzi": int(observed_mzi),
        "scan_file": str(scan_file),
        "valid": bool(valid),
        "try_count": int(try_count),
        "point_count": int(len(df)),
        "median_power_uW": median_y,
        "min_power_uW": min_y,
        "max_power_uW": max_y,
        "curve_range_uW": curve_range,
        "unstable_point_count": int(counts.get("unstable_point_count", 0)),
        "near_zero_point_count": int(counts.get("near_zero_point_count", 0)),
        "neighbor_jump_count": int(counts.get("neighbor_jump_count", 0)),
        "edge_jump_count": int(counts.get("edge_jump_count", 0)),
        "route_lower_policy": route_lower_policy,
        "route_lower_zero_applied": route_lower_zero_applied,
        "warning": merged_warning,
        "failure_reason": failure_reason,
    }


def validate_delta_scan_df(df, args, expected_points):
    y = pd.to_numeric(df["optical_power_uW"], errors="coerce").to_numpy(dtype=float)
    reasons = []
    warnings = []
    if y.size != int(expected_points) or np.isnan(y).any():
        reasons.append("scan file incomplete")
    median_y = float(np.nanmedian(y)) if y.size else 0.0
    near_zero_mask = (y < float(args.near_zero_absolute_uW)) & (median_y > float(args.near_zero_median_min_uW))
    near_zero_count = int(np.sum(near_zero_mask))
    if near_zero_count:
        reasons.append("near-zero isolated point")
    neighbor_count = 0
    if y.size >= 3:
        for idx in range(1, y.size - 1):
            neighbor_mean = (float(y[idx - 1]) + float(y[idx + 1])) / 2.0
            if abs(float(y[idx]) - neighbor_mean) > float(args.delta_neighbor_jump_ratio) * max(abs(neighbor_mean), 1e-9):
                neighbor_count += 1
        if neighbor_count:
            reasons.append("neighbor jump")
    edge_count = 0
    if y.size >= 2:
        if abs(float(y[0]) - float(y[1])) > float(args.delta_edge_jump_ratio) * max(abs(float(y[1])), 1e-9):
            edge_count += 1
        if abs(float(y[-1]) - float(y[-2])) > float(args.delta_edge_jump_ratio) * max(abs(float(y[-2])), 1e-9):
            edge_count += 1
        if edge_count:
            reasons.append("edge jump")
    rel_std = pd.to_numeric(df.get("opm_relative_std", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    unstable_count = int(np.sum(rel_std.to_numpy(dtype=float) > float(args.opm_relative_std_threshold)))
    if unstable_count > int(args.max_unstable_points_per_scan):
        reasons.append("too many unstable OPM points")
    curve_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    if curve_range < float(args.delta_min_curve_range_uW):
        warnings.append("low delta curve range")
    counts = {
        "unstable_point_count": unstable_count,
        "near_zero_point_count": near_zero_count,
        "neighbor_jump_count": neighbor_count,
        "edge_jump_count": edge_count,
    }
    return len(reasons) == 0, "; ".join(warnings), "; ".join(reasons), counts


def validate_sigma_scan_df(df, args, expected_points):
    y = pd.to_numeric(df["optical_power_uW"], errors="coerce").to_numpy(dtype=float)
    reasons = []
    warnings = []
    if y.size != int(expected_points) or np.isnan(y).any():
        reasons.append("scan file incomplete")
    median_y = float(np.nanmedian(y)) if y.size else 0.0
    near_zero_count = int(np.sum(y < float(args.sigma_min_median_power_uW))) if y.size else 0
    if median_y < float(args.sigma_min_median_power_uW):
        warnings.append("low sigma median power")
    rel_std = pd.to_numeric(df.get("opm_relative_std", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    unstable_count = int(np.sum(rel_std.to_numpy(dtype=float) > float(args.opm_relative_std_threshold)))
    if unstable_count > int(args.max_unstable_points_per_scan):
        reasons.append("too many unstable OPM points")
    curve_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    if curve_range < float(args.delta_min_curve_range_uW):
        warnings.append("very low sigma visibility")
    counts = {
        "unstable_point_count": unstable_count,
        "near_zero_point_count": near_zero_count,
        "neighbor_jump_count": 0,
        "edge_jump_count": 0,
    }
    return len(reasons) == 0, "; ".join(warnings), "; ".join(reasons), counts


def suggested_action_for_failure(reason):
    if "unstable OPM" in reason:
        return "increase settle time"
    if "near-zero" in reason:
        return "check optical path"
    if "jump" in reason:
        return "check voltage upload"
    if "incomplete" in reason:
        return "rerun this perturb"
    return "check input/output switching"


def get_mzi_state_voltage(entry, state):
    state = str(state).upper()
    values = entry.get("dtheta_Bar", entry.get("dtheta", [])) if state == "B" else entry.get("dtheta_Cross", entry.get("dtheta", []))
    if not values:
        raise ValueError(f"MZI entry missing {state} voltage.")
    return float(values[0 if state == "B" else min(1, len(values) - 1)])


def normalize_route_lower_policy(policy):
    policy = str(policy or "zero").strip().lower()
    if policy not in {"zero", "keep_base"}:
        raise ValueError("--route_lower_policy must be zero or keep_base.")
    return policy


def get_route_state_voltage(entry, state):
    state = str(state).upper()
    if state in {"B", "C"}:
        return get_mzi_state_voltage(entry, state)
    if state == "H":
        half_values = entry.get("half_power", [])
        if not half_values:
            raise ValueError("MZI entry missing H half_power voltage.")
        return float(half_values[0])
    raise ValueError(f"Unsupported route MZI state {state!r}.")


def set_route_mzi_state(mzi_id, entry, state, working_data, lower_policy="zero"):
    lower_policy = normalize_route_lower_policy(lower_policy)
    ports = [int(port) for port in entry.get("ports", [])]
    if not ports:
        raise ValueError(f"MZI {int(mzi_id)} missing ports.")

    state = str(state).upper()
    upper_voltage = float(get_route_state_voltage(entry, state))
    write_port_voltage(int(ports[0]), upper_voltage, working_data)

    lower_port = int(ports[1]) if len(ports) >= 2 else None
    lower_voltage = None
    lower_zero_applied = True
    warnings = []
    if lower_port is None:
        warnings.append(f"MZI{int(mzi_id)} has no lower route port for state {state}")
    elif lower_policy == "zero":
        lower_voltage = 0.0
        write_port_voltage(lower_port, lower_voltage, working_data)
    else:
        lower_voltage = float(get_port_voltage(working_data, lower_port))
        lower_zero_applied = False
        warnings.append(
            f"route_lower_policy=keep_base kept lower arm for MZI{int(mzi_id)} at {lower_voltage:.6f} V"
        )

    return {
        "mzi_id": int(mzi_id),
        "state": state,
        "upper_port": int(ports[0]),
        "lower_port": lower_port,
        "upper_voltage": upper_voltage,
        "lower_voltage": lower_voltage,
        "lower_zero_applied": bool(lower_zero_applied),
        "warning": "; ".join(warnings),
    }


def route_records_to_fields(records, lower_policy):
    records = list(records)
    warnings = [str(record.get("warning", "")).strip() for record in records if str(record.get("warning", "")).strip()]
    route_lower_zero_applied = bool(all(record.get("lower_zero_applied", True) for record in records))
    return {
        "route_mzi_ids": json.dumps([int(record["mzi_id"]) for record in records]),
        "route_states": json.dumps([str(record["state"]) for record in records]),
        "route_upper_ports": json.dumps([int(record["upper_port"]) for record in records]),
        "route_lower_ports": json.dumps([record["lower_port"] for record in records]),
        "route_upper_voltages": json.dumps([float(record["upper_voltage"]) for record in records]),
        "route_lower_voltages": json.dumps(
            [None if record["lower_voltage"] is None else float(record["lower_voltage"]) for record in records]
        ),
        "route_lower_policy": normalize_route_lower_policy(lower_policy),
        "route_lower_zero_applied": route_lower_zero_applied,
        "route_warning": "; ".join(warnings),
    }


def build_bmzi_state_no_upload(
    path,
    input_idx,
    state,
    bmzi,
    working_data,
    mzi_table,
    n_value,
    route_lower_policy="zero",
    target_mzi=None,
):
    route_lower_policy = normalize_route_lower_policy(route_lower_policy)
    route_records = []
    for idx, mzi_value in enumerate(path):
        mzi_id = int(mzi_value)
        entry = mzi_table[str(mzi_id)]
        route_state = str(state[idx]).upper()
        ports = entry.get("ports", [])
        if not ports:
            raise ValueError(f"MZI {mzi_id} missing ports.")
        if target_mzi is not None and mzi_id == int(target_mzi):
            write_port_voltage(int(ports[0]), get_route_state_voltage(entry, route_state), working_data)
            continue
        route_records.append(set_route_mzi_state(mzi_id, entry, route_state, working_data, route_lower_policy))
    for channel in range(1, int(n_value)):
        switch_IN(channel, "OFF", working_data)
    switch_IN(int(input_idx) + 1, "ON", working_data)
    route_fields = route_records_to_fields(route_records, route_lower_policy)
    if int(bmzi) == 0:
        route_fields["bmzi_route_note"] = "bmzi=0 top straight reference"
    else:
        route_fields["bmzi_route_note"] = f"bmzi={int(bmzi)} metadata only; no separate B/C/H route state applied"
    return route_fields


def apply_second_column_powers(working_data, mzi_table, mzi_ids, heater_order, powers):
    for heater in heater_order:
        mzi_id = int(heater[:-1])
        arm = heater[-1]
        info = get_mzi_arm_info(mzi_table, mzi_id, arm)
        write_port_power(working_data, info["port"], info["resistance"], powers[heater])


def scan_delta_probe_once(observed_mzi, probe_arm, out_path, base_working_data, hardware, mzi_table, args, progress_label="", try_index=1):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = get_mzi_arm_info(mzi_table, observed_mzi, probe_arm)
    input_channel = get_left_upper_bar_channel(int(observed_mzi), int(args.N))
    output_channel = input_channel
    baseline_v = get_port_voltage(base_working_data, info["port"])
    baseline_power = voltage_to_power_w(baseline_v, info["resistance"])
    offsets = generate_delta_offsets(args)
    rows = []
    total_points = int(len(offsets))
    for point_idx, offset in enumerate(offsets, start=1):
        scan_data = base_working_data.copy(deep=True)
        target_power = max(0.0, float(baseline_power) + float(offset))
        voltage = write_port_power(scan_data, info["port"], info["resistance"], target_power)
        for channel in range(1, int(args.N)):
            switch_IN(channel, "OFF", scan_data)
        switch_IN(input_channel, "ON", scan_data)
        label = (
            f"{progress_label} delta obs{int(observed_mzi)} "
            f"point {point_idx}/{total_points}, offset={float(offset):.9f} W"
        ).strip()
        upload_voltage_checked(hardware["mcv"], scan_data, args, label)
        upload_timestamp = datetime.now().isoformat(timespec="seconds")
        time.sleep(float(args.point_settle_time_s))
        opm = read_stable_opm(
            hardware["opm2"],
            output_channel,
            args,
            reupload_callback=lambda sd=scan_data, lbl=label: upload_voltage_checked(hardware["mcv"], sd, args, f"{lbl} reupload"),
        )
        read_timestamp = datetime.now().isoformat(timespec="seconds")
        rows.append(
            {
                "target": int(observed_mzi),
                "observed_mzi": int(observed_mzi),
                "probe_arm": probe_arm,
                "arm_name": info["arm_name"],
                "arm_index": info["arm_index"],
                "port": int(info["port"]),
                "probe_axis_power_w": float(offset),
                "target_power_w": float(target_power),
                "measured_power_w": float(target_power),
                "voltage_v": float(voltage),
                "optical_power_uW": float(opm["opm_median_uW"]),
                **opm,
                "scan_type": "delta_probe",
                "output_channel": int(output_channel),
                "scan_try_index": int(try_index),
                "upload_timestamp": upload_timestamp,
                "read_timestamp": read_timestamp,
                "point_settle_time_s": float(args.point_settle_time_s),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.12f")
    return out_path, df


def scan_sigma_inter_once(observed_mzi, out_path, base_working_data, hardware, mzi_table, args, sigma_bmzi_map, progress_label="", try_index=1):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = int(observed_mzi)
    scan_data = base_working_data.copy(deep=True)
    path, input_idx, output_idx, state, bmzi = find_Bmzi_path(target, int(args.N))
    route_fields = build_bmzi_state_no_upload(
        path,
        input_idx,
        state,
        bmzi,
        scan_data,
        mzi_table,
        int(args.N),
        route_lower_policy=getattr(args, "route_lower_policy", "zero"),
        target_mzi=target,
    )
    entry = mzi_table[str(target)]
    ports = [int(v) for v in entry.get("ports", [])[:2]]
    heater_r = [float(v) for v in entry.get("heater_R", [])[:2]]
    ppi = [float(v) for v in entry.get("Ppi", [])[:2]]
    if len(ports) != 2 or len(heater_r) != 2 or len(ppi) != 2:
        raise ValueError(f"MZI {target} requires two ports, heater_R, and Ppi for sigma scan.")
    p_upper_base = voltage_to_power_w(get_port_voltage(scan_data, ports[0]), heater_r[0])
    p_lower_base = voltage_to_power_w(get_port_voltage(scan_data, ports[1]), heater_r[1])
    phase_points = generate_sigma_phase_points(args)
    rows = []
    total_points = int(len(phase_points))
    for point_idx, dp in enumerate(phase_points, start=1):
        p_upper_unfolded = p_upper_base + float(dp) / np.pi * ppi[0]
        p_lower_unfolded = p_lower_base + float(dp) / np.pi * ppi[1]
        p_upper, upper_folds = fold_power_to_limit(p_upper_unfolded, 2.0 * ppi[0], args.power_limit_w)
        p_lower, lower_folds = fold_power_to_limit(p_lower_unfolded, 2.0 * ppi[1], args.power_limit_w)
        v_upper = write_port_power(scan_data, ports[0], heater_r[0], p_upper)
        v_lower = write_port_power(scan_data, ports[1], heater_r[1], p_lower)
        label = (
            f"{progress_label} sigma obs{target} "
            f"point {point_idx}/{total_points}, dp={float(dp):.9f} rad"
        ).strip()
        upload_voltage_checked(hardware["mcv"], scan_data, args, label)
        upload_timestamp = datetime.now().isoformat(timespec="seconds")
        time.sleep(float(args.point_settle_time_s))
        opm = read_stable_opm(
            hardware["opm2"],
            int(output_idx) + 1,
            args,
            reupload_callback=lambda sd=scan_data, lbl=label: upload_voltage_checked(hardware["mcv"], sd, args, f"{lbl} reupload"),
        )
        route_warning = str(route_fields.get("route_warning", "")).strip()
        combined_warning = "; ".join([text for text in (str(opm.get("warning", "")).strip(), route_warning) if text])
        opm["warning"] = combined_warning
        opm["point_warning"] = combined_warning
        read_timestamp = datetime.now().isoformat(timespec="seconds")
        rows.append(
            {
                "target": target,
                "observed_mzi": target,
                "bmzi": int(sigma_bmzi_map.get(str(target), bmzi)),
                "input_channel": int(input_idx) + 1,
                "output_channel": int(output_idx) + 1,
                "path": json.dumps([int(v) for v in path]),
                "state": json.dumps([str(v) for v in state]),
                **route_fields,
                "dp": float(dp),
                "v_primary": float(v_upper),
                "v_secondary": float(v_lower),
                "p_primary": float(p_upper),
                "p_secondary": float(p_lower),
                "p_primary_unfolded": float(p_upper_unfolded),
                "p_secondary_unfolded": float(p_lower_unfolded),
                "upper_fold_count": int(upper_folds),
                "lower_fold_count": int(lower_folds),
                "optical_power_uW": float(opm["opm_median_uW"]),
                "pow(uW)": float(opm["opm_median_uW"]),
                **opm,
                "scan_type": "sigma_inter",
                "scan_try_index": int(try_index),
                "upload_timestamp": upload_timestamp,
                "read_timestamp": read_timestamp,
                "point_settle_time_s": float(args.point_settle_time_s),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.12f")
    return out_path, df


def mark_and_save_scan_df(df, out_path, valid, warning, failure_reason, quality_score):
    df = df.copy()
    df["scan_valid"] = bool(valid)
    df["scan_failure_reason"] = failure_reason
    df["scan_warning"] = warning
    df["quality_score"] = float(quality_score)
    df.to_csv(out_path, index=False, float_format="%.12f")


def scan_delta_probe(observed_mzi, probe_arm, save_dir, base_working_data, hardware, mzi_table, args, run_dir, stage, perturb_label, progress_label=""):
    save_dir = Path(save_dir)
    final_path = save_dir / f"obs{int(observed_mzi)}_probe.txt"
    failed_files = []
    last_reason = ""
    expected_points = len(generate_delta_offsets(args))
    for try_index in range(1, int(args.max_scan_retry) + 1):
        try_path = save_dir / f"obs{int(observed_mzi)}_probe_try{try_index}.txt"
        try_path, df = scan_delta_probe_once(
            observed_mzi,
            probe_arm,
            try_path,
            base_working_data,
            hardware,
            mzi_table,
            args,
            progress_label=progress_label,
            try_index=try_index,
        )
        valid, warning, failure_reason, counts = validate_delta_scan_df(df, args, expected_points)
        last_reason = failure_reason
        quality_score = 1.0 if valid else 0.0
        mark_and_save_scan_df(df, try_path, valid, warning, failure_reason, quality_score)
        quality_row = build_quality_row(stage, perturb_label, "delta", observed_mzi, try_path, try_index, df, valid, warning, failure_reason, counts)
        append_csv_row(Path(run_dir) / "scan_quality_summary.csv", quality_row, QUALITY_COLUMNS)
        if valid:
            shutil.copyfile(try_path, final_path)
            print(f"[Get_Jacobi] delta obs{int(observed_mzi)} passed on try {try_index}: {final_path}")
            return final_path
        failed_files.append(str(try_path))
        print(f"[Get_Jacobi] delta obs{int(observed_mzi)} failed try {try_index}: {failure_reason}")
        if try_index < int(args.max_scan_retry):
            time.sleep(float(args.rescan_wait_s))
    failed_row = {
        "stage": stage,
        "perturb_label": perturb_label,
        "scan_kind": "delta",
        "observed_mzi": int(observed_mzi),
        "failed_files": json.dumps(failed_files),
        "final_failure_reason": last_reason,
        "suggested_action": suggested_action_for_failure(last_reason),
    }
    append_csv_row(Path(run_dir) / "failed_scans.csv", failed_row, FAILED_COLUMNS)
    return None


def scan_sigma_inter(observed_mzi, save_dir, base_working_data, hardware, mzi_table, args, sigma_bmzi_map, run_dir, stage, perturb_label, progress_label=""):
    save_dir = Path(save_dir)
    final_path = save_dir / f"obs{int(observed_mzi)}_inter_scan.txt"
    failed_files = []
    last_reason = ""
    expected_points = len(generate_sigma_phase_points(args))
    for try_index in range(1, int(args.max_scan_retry) + 1):
        try_path = save_dir / f"obs{int(observed_mzi)}_inter_scan_try{try_index}.txt"
        try_path, df = scan_sigma_inter_once(
            observed_mzi,
            try_path,
            base_working_data,
            hardware,
            mzi_table,
            args,
            sigma_bmzi_map,
            progress_label=progress_label,
            try_index=try_index,
        )
        valid, warning, failure_reason, counts = validate_sigma_scan_df(df, args, expected_points)
        last_reason = failure_reason
        quality_score = 1.0 if valid else 0.0
        mark_and_save_scan_df(df, try_path, valid, warning, failure_reason, quality_score)
        quality_row = build_quality_row(stage, perturb_label, "sigma", observed_mzi, try_path, try_index, df, valid, warning, failure_reason, counts)
        append_csv_row(Path(run_dir) / "scan_quality_summary.csv", quality_row, QUALITY_COLUMNS)
        if valid:
            shutil.copyfile(try_path, final_path)
            print(f"[Get_Jacobi] sigma obs{int(observed_mzi)} passed on try {try_index}: {final_path}")
            return final_path
        failed_files.append(str(try_path))
        print(f"[Get_Jacobi] sigma obs{int(observed_mzi)} failed try {try_index}: {failure_reason}")
        if try_index < int(args.max_scan_retry):
            time.sleep(float(args.rescan_wait_s))
    failed_row = {
        "stage": stage,
        "perturb_label": perturb_label,
        "scan_kind": "sigma",
        "observed_mzi": int(observed_mzi),
        "failed_files": json.dumps(failed_files),
        "final_failure_reason": last_reason,
        "suggested_action": suggested_action_for_failure(last_reason),
    }
    append_csv_row(Path(run_dir) / "failed_scans.csv", failed_row, FAILED_COLUMNS)
    return None


def collect_all_scans(save_root, base_working_data, hardware, mzi_table, args, probe_map, sigma_bmzi_map, run_dir, stage, perturb_label, progress_label=""):
    delta_dir = Path(save_root) / "delta"
    sigma_dir = Path(save_root) / "sigma"
    total_mzis = int(len(args._mzi_ids))
    for mzi_idx, mzi_id in enumerate(args._mzi_ids, start=1):
        print(f"[Get_Jacobi] {progress_label} delta scan {mzi_idx}/{total_mzis}: obs{int(mzi_id)}")
        scan_delta_probe(
            mzi_id,
            probe_map[int(mzi_id)],
            delta_dir,
            base_working_data,
            hardware,
            mzi_table,
            args,
            run_dir,
            stage,
            perturb_label,
            progress_label=progress_label,
        )
    for mzi_idx, mzi_id in enumerate(args._mzi_ids, start=1):
        print(f"[Get_Jacobi] {progress_label} sigma scan {mzi_idx}/{total_mzis}: obs{int(mzi_id)}")
        scan_sigma_inter(
            mzi_id,
            sigma_dir,
            base_working_data,
            hardware,
            mzi_table,
            args,
            sigma_bmzi_map,
            run_dir,
            stage,
            perturb_label,
            progress_label=progress_label,
        )


def write_dry_run_placeholders(run_dir, args):
    for group in ["baseline", *[f"perturb_{h}" for h in args._heaters]]:
        for sub in ("delta", "sigma"):
            (run_dir / group / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=QUALITY_COLUMNS).to_csv(run_dir / "scan_quality_summary.csv", index=False)
    pd.DataFrame(columns=FAILED_COLUMNS).to_csv(run_dir / "failed_scans.csv", index=False)
    for heater in args._heaters:
        metadata_path = run_dir / f"perturb_{heater}" / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "perturbed_heater": heater,
                    "delta_power_w": float(args.delta_power_w),
                    "baseline_power_w": None,
                    "perturbed_power_w": None,
                    "heater_order": args._heaters,
                    "mzi_ids": args._mzi_ids,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dry_run": True,
                },
                f,
                indent=2,
            )


def measure(args):
    args._mzi_ids = parse_csv_list(args.mzi_ids, int)
    args._heaters = parse_csv_list(args.heaters, str)
    args._heaters = validate_heater_order(args._heaters, args._mzi_ids)
    args._delta_offsets = generate_delta_offsets(args)
    args._sigma_phase_points = generate_sigma_phase_points(args)
    probe_map = parse_probe_map(args.probe_map, args._mzi_ids)
    sigma_bmzi_map = parse_sigma_bmzi_map(args.sigma_bmzi_map, args._mzi_ids)
    run_dir = Path(args.out_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mzi_ids": args._mzi_ids,
        "heater_order": args._heaters,
        "probe_map": {str(k): v for k, v in probe_map.items()},
        "sigma_bmzi_map": sigma_bmzi_map,
        "sigma_reference_mode": args.sigma_reference_mode,
        "route_lower_policy": normalize_route_lower_policy(args.route_lower_policy),
        "delta_probe_points": [float(v) for v in args._delta_offsets],
        "delta_probe_half_width_w": float(args.delta_probe_half_width_w),
        "delta_probe_step_w": float(args.delta_probe_step_w),
        "sigma_points": int(args.sigma_points),
        "sigma_phase_points": [float(v) for v in args._sigma_phase_points],
        "opm_reads_per_point": int(args.opm_reads_per_point),
        "max_scan_retry": int(args.max_scan_retry),
        "power_limit_w": float(args.power_limit_w),
        "voltage_limit_v": float(args.voltage_limit_v),
        "point_settle_time_s": float(args.point_settle_time_s),
        "baseline_restore_wait_s": float(args.baseline_restore_wait_s),
        "perturb_settle_time_s": float(args.perturb_settle_time_s),
        "initial_state": args.initial_state,
        "initial_power_file": args.initial_power_file,
        "dry_run": bool(args.dry_run),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))
    pd.DataFrame(columns=QUALITY_COLUMNS).to_csv(run_dir / "scan_quality_summary.csv", index=False)
    pd.DataFrame(columns=FAILED_COLUMNS).to_csv(run_dir / "failed_scans.csv", index=False)

    if args.dry_run:
        write_dry_run_placeholders(run_dir, args)
        print(f"[Get_Jacobi] dry run: created directory skeleton at {run_dir}")
        return run_dir
    if not parse_bool(args.confirm_hardware):
        raise RuntimeError("Refusing hardware write: set --confirm_hardware true when --dry_run false.")

    mzi_table = load_mzi_table(args.mzi_table)
    baseline_powers = read_power_file(args.initial_power_file, args._heaters)
    mcv = cu.open_ser_connection(args.ser_address)
    opm2 = cu.open_VISA_connection(args.opm2_address)
    if mcv is None:
        raise RuntimeError(f"Failed to open serial port {args.ser_address}.")
    if opm2 is None:
        raise RuntimeError(f"Failed to open OPM2 {args.opm2_address}.")
    hardware = {"mcv": mcv, "opm2": opm2}
    try:
        total_groups = 1 + int(len(args._heaters))
        working_data = cu.generate_working_data()
        apply_second_column_powers(working_data, mzi_table, args._mzi_ids, args._heaters, baseline_powers)
        upload_voltage_checked(mcv, working_data, args, f"group 1/{total_groups} baseline initial state")
        time.sleep(float(args.settle_time))
        collect_all_scans(
            run_dir / "baseline",
            working_data,
            hardware,
            mzi_table,
            args,
            probe_map,
            sigma_bmzi_map,
            run_dir,
            "baseline",
            "baseline",
            progress_label=f"group 1/{total_groups} baseline",
        )
        for group_idx, heater in enumerate(args._heaters, start=2):
            pert_dir = run_dir / f"perturb_{heater}"
            pert_dir.mkdir(parents=True, exist_ok=True)
            upload_voltage_checked(mcv, working_data, args, f"group {group_idx}/{total_groups} restore baseline before {heater}")
            time.sleep(float(args.baseline_restore_wait_s))
            pert_powers = dict(baseline_powers)
            baseline_power = float(pert_powers[heater])
            pert_powers[heater] = baseline_power + float(args.delta_power_w)
            pert_working = working_data.copy(deep=True)
            apply_second_column_powers(pert_working, mzi_table, args._mzi_ids, args._heaters, pert_powers)
            validate_voltage_range(pert_working, 0.0, float(args.voltage_limit_v))
            upload_voltage_checked(mcv, pert_working, args, f"group {group_idx}/{total_groups} apply perturb {heater}")
            time.sleep(float(args.perturb_settle_time_s))
            with (pert_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "perturbed_heater": heater,
                        "delta_power_w": float(args.delta_power_w),
                        "baseline_power_w": baseline_power,
                        "perturbed_power_w": float(pert_powers[heater]),
                        "baseline_power_vector_w": {str(k): float(v) for k, v in baseline_powers.items()},
                        "perturb_power_vector_w": {str(k): float(v) for k, v in pert_powers.items()},
                        "restored_before_measurement": True,
                        "heater_order": args._heaters,
                        "mzi_ids": args._mzi_ids,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                    f,
                    indent=2,
                )
            collect_all_scans(
                pert_dir,
                pert_working,
                hardware,
                mzi_table,
                args,
                probe_map,
                sigma_bmzi_map,
                run_dir,
                "perturb",
                heater,
                progress_label=f"group {group_idx}/{total_groups} perturb {heater}",
            )
            upload_voltage_checked(mcv, working_data, args, f"group {group_idx}/{total_groups} restore baseline after {heater}")
            time.sleep(float(args.baseline_restore_wait_s))
    finally:
        for handle in (mcv, opm2):
            close = getattr(handle, "close", None)
            if callable(close):
                close()
    print(f"[Get_Jacobi] saved raw measurements to {run_dir}")
    return run_dir


def prepare_measure_args(args):
    args._mzi_ids = parse_csv_list(args.mzi_ids, int)
    args._heaters = parse_csv_list(args.heaters, str)
    args._heaters = validate_heater_order(args._heaters, args._mzi_ids)
    args._delta_offsets = generate_delta_offsets(args)
    args._sigma_phase_points = generate_sigma_phase_points(args)
    return parse_probe_map(args.probe_map, args._mzi_ids), parse_sigma_bmzi_map(args.sigma_bmzi_map, args._mzi_ids)


def rescan_one(args):
    probe_map, sigma_bmzi_map = prepare_measure_args(args)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if int(args.observed_mzi) not in args._mzi_ids:
        raise ValueError(f"observed_mzi {args.observed_mzi} is not in --mzi_ids {args._mzi_ids}")
    if args.stage == "perturb" and str(args.perturb_heater) not in args._heaters:
        raise ValueError(f"perturb_heater {args.perturb_heater} is not in --heaters {args._heaters}")

    config = {
        "mode": "rescan_one",
        "run_dir": str(run_dir),
        "stage": args.stage,
        "perturb_heater": args.perturb_heater,
        "scan_kind": args.scan_kind,
        "observed_mzi": int(args.observed_mzi),
        "route_lower_policy": normalize_route_lower_policy(args.route_lower_policy),
        "dry_run": bool(args.dry_run),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        target_dir = run_dir / ("baseline" if args.stage == "baseline" else f"perturb_{args.perturb_heater}") / args.scan_kind
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Get_Jacobi] dry run: would rescan into {target_dir}")
        return target_dir
    if not parse_bool(args.confirm_hardware):
        raise RuntimeError("Refusing hardware write: set --confirm_hardware true when --dry_run false.")

    mzi_table = load_mzi_table(args.mzi_table)
    baseline_powers = read_power_file(args.initial_power_file, args._heaters)
    mcv = cu.open_ser_connection(args.ser_address)
    opm2 = cu.open_VISA_connection(args.opm2_address)
    if mcv is None:
        raise RuntimeError(f"Failed to open serial port {args.ser_address}.")
    if opm2 is None:
        raise RuntimeError(f"Failed to open OPM2 {args.opm2_address}.")
    hardware = {"mcv": mcv, "opm2": opm2}
    try:
        working_data = cu.generate_working_data()
        apply_second_column_powers(working_data, mzi_table, args._mzi_ids, args._heaters, baseline_powers)
        upload_voltage_checked(mcv, working_data, args, "rescan restore baseline")
        time.sleep(float(args.baseline_restore_wait_s))

        stage_dir = run_dir / "baseline"
        stage = "baseline"
        perturb_label = "baseline"
        scan_working = working_data
        if args.stage == "perturb":
            stage_dir = run_dir / f"perturb_{args.perturb_heater}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            perturb_powers = dict(baseline_powers)
            perturb_powers[str(args.perturb_heater)] = float(perturb_powers[str(args.perturb_heater)]) + float(args.delta_power_w)
            scan_working = working_data.copy(deep=True)
            apply_second_column_powers(scan_working, mzi_table, args._mzi_ids, args._heaters, perturb_powers)
            upload_voltage_checked(mcv, scan_working, args, f"rescan apply perturb {args.perturb_heater}")
            time.sleep(float(args.perturb_settle_time_s))
            stage = "perturb"
            perturb_label = str(args.perturb_heater)
            metadata_path = stage_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8-sig") as f:
                    metadata = json.load(f)
            metadata.update(
                {
                    "rescan_updated_at": datetime.now().isoformat(timespec="seconds"),
                    "baseline_power_vector_w": {str(k): float(v) for k, v in baseline_powers.items()},
                    "perturb_power_vector_w": {str(k): float(v) for k, v in perturb_powers.items()},
                    "restored_before_measurement": True,
                }
            )
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        if args.scan_kind == "sigma":
            scan_sigma_inter(
                int(args.observed_mzi),
                stage_dir / "sigma",
                scan_working,
                hardware,
                mzi_table,
                args,
                sigma_bmzi_map,
                run_dir,
                stage,
                perturb_label,
                progress_label=f"rescan {stage} {perturb_label}",
            )
        else:
            scan_delta_probe(
                int(args.observed_mzi),
                probe_map[int(args.observed_mzi)],
                stage_dir / "delta",
                scan_working,
                hardware,
                mzi_table,
                args,
                run_dir,
                stage,
                perturb_label,
                progress_label=f"rescan {stage} {perturb_label}",
            )
        upload_voltage_checked(mcv, working_data, args, "rescan restore baseline after scan")
        time.sleep(float(args.baseline_restore_wait_s))
    finally:
        for handle in (mcv, opm2):
            close = getattr(handle, "close", None)
            if callable(close):
                close()
    print(f"[Get_Jacobi] rescan saved under {run_dir}")
    return run_dir


def add_common_measure_args(p):
    p.add_argument("--mzi_ids", default="5,6,7,8")
    p.add_argument("--heaters", default="5u,5d,6u,6d,7u,7d,8u,8d")
    p.add_argument("--mzi_table", default="Scandata/MZI_table.json")
    p.add_argument("--initial_power_file", default="current_power_second_column.csv")
    p.add_argument("--probe_map", default="5:u,6:u,7:u,8:u")
    p.add_argument("--sigma_bmzi_map", default="5:0,6:5,7:6,8:7")
    p.add_argument("--sigma_reference_mode", default="chained_bmzi", choices=["chained_bmzi", "direct_reference"])
    p.add_argument("--route_lower_policy", default="zero", choices=["zero", "keep_base"])
    p.add_argument("--delta_power_w", type=float, default=0.001)
    p.add_argument("--delta_probe_points", default="9")
    p.add_argument("--delta_probe_half_width_w", type=float, default=0.001)
    p.add_argument("--delta_probe_step_w", type=float, default=0.00025)
    p.add_argument("--sigma_points", type=int, default=9)
    p.add_argument("--sigma_phase_points", default="")
    p.add_argument("--opm_reads_per_point", type=int, default=5)
    p.add_argument("--opm_read_interval_s", type=float, default=0.1)
    p.add_argument("--opm_relative_std_threshold", type=float, default=0.05)
    p.add_argument("--opm_max_retry_per_point", type=int, default=2)
    p.add_argument("--reupload_on_unstable_point", type=parse_bool, default=True)
    p.add_argument("--point_settle_time_s", type=float, default=2.0)
    p.add_argument("--near_zero_absolute_uW", type=float, default=1.0)
    p.add_argument("--near_zero_median_min_uW", type=float, default=10.0)
    p.add_argument("--delta_neighbor_jump_ratio", type=float, default=0.3)
    p.add_argument("--delta_edge_jump_ratio", type=float, default=0.3)
    p.add_argument("--max_unstable_points_per_scan", type=int, default=1)
    p.add_argument("--delta_min_curve_range_uW", type=float, default=1.0)
    p.add_argument("--sigma_min_median_power_uW", type=float, default=1.0)
    p.add_argument("--max_scan_retry", type=int, default=3)
    p.add_argument("--rescan_wait_s", type=float, default=1.0)
    p.add_argument("--baseline_restore_wait_s", type=float, default=2.0)
    p.add_argument("--perturb_settle_time_s", type=float, default=2.0)
    p.add_argument("--power_limit_w", type=float, default=0.055)
    p.add_argument("--voltage_limit_v", type=float, default=6.0)
    p.add_argument("--settle_time", type=float, default=2.0)
    p.add_argument("--initial_state", default="voltage_pair")
    p.add_argument("--N", type=int, default=9)
    p.add_argument("--dry_run", type=parse_bool, default=True)
    p.add_argument("--confirm_hardware", type=parse_bool, default=False)
    p.add_argument("--ser_address", default=DEFAULT_SER_ADDRESS)
    p.add_argument("--opm2_address", default=DEFAULT_OPM2_ADDRESS)


def build_parser():
    parser = argparse.ArgumentParser(description="Collect raw scans for second-column Jacobian measurement.")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("measure")
    p.add_argument("--mzi_ids", default="5,6,7,8")
    p.add_argument("--heaters", default="5u,5d,6u,6d,7u,7d,8u,8d")
    p.add_argument("--out_root", default="Scandata/J_remeasure")
    p.add_argument("--mzi_table", default="Scandata/MZI_table.json")
    p.add_argument("--initial_power_file", default="current_power_second_column.csv")
    p.add_argument("--probe_map", default="5:u,6:u,7:u,8:u")
    p.add_argument("--sigma_bmzi_map", default="5:0,6:5,7:6,8:7")
    p.add_argument("--sigma_reference_mode", default="chained_bmzi", choices=["chained_bmzi", "direct_reference"])
    p.add_argument("--route_lower_policy", default="zero", choices=["zero", "keep_base"])
    p.add_argument("--delta_power_w", type=float, default=0.001)
    p.add_argument("--delta_probe_points", default="9")
    p.add_argument("--delta_probe_half_width_w", type=float, default=0.001)
    p.add_argument("--delta_probe_step_w", type=float, default=0.00025)
    p.add_argument("--sigma_points", type=int, default=9)
    p.add_argument("--sigma_phase_points", default="")
    p.add_argument("--opm_reads_per_point", type=int, default=5)
    p.add_argument("--opm_read_interval_s", type=float, default=0.1)
    p.add_argument("--opm_relative_std_threshold", type=float, default=0.05)
    p.add_argument("--opm_max_retry_per_point", type=int, default=2)
    p.add_argument("--reupload_on_unstable_point", type=parse_bool, default=True)
    p.add_argument("--point_settle_time_s", type=float, default=2.0)
    p.add_argument("--near_zero_absolute_uW", type=float, default=1.0)
    p.add_argument("--near_zero_median_min_uW", type=float, default=10.0)
    p.add_argument("--delta_neighbor_jump_ratio", type=float, default=0.3)
    p.add_argument("--delta_edge_jump_ratio", type=float, default=0.3)
    p.add_argument("--max_unstable_points_per_scan", type=int, default=1)
    p.add_argument("--delta_min_curve_range_uW", type=float, default=1.0)
    p.add_argument("--sigma_min_median_power_uW", type=float, default=1.0)
    p.add_argument("--max_scan_retry", type=int, default=3)
    p.add_argument("--rescan_wait_s", type=float, default=1.0)
    p.add_argument("--baseline_restore_wait_s", type=float, default=2.0)
    p.add_argument("--perturb_settle_time_s", type=float, default=2.0)
    p.add_argument("--power_limit_w", type=float, default=0.055)
    p.add_argument("--voltage_limit_v", type=float, default=6.0)
    p.add_argument("--settle_time", type=float, default=2.0)
    p.add_argument("--initial_state", default="voltage_pair")
    p.add_argument("--N", type=int, default=9)
    p.add_argument("--dry_run", type=parse_bool, default=True)
    p.add_argument("--confirm_hardware", type=parse_bool, default=False)
    p.add_argument("--ser_address", default=DEFAULT_SER_ADDRESS)
    p.add_argument("--opm2_address", default=DEFAULT_OPM2_ADDRESS)
    r = sub.add_parser("rescan_one")
    add_common_measure_args(r)
    r.add_argument("--run_dir", required=True)
    r.add_argument("--stage", default="perturb", choices=["baseline", "perturb"])
    r.add_argument("--perturb_heater", default="7u")
    r.add_argument("--scan_kind", default="sigma", choices=["delta", "sigma"])
    r.add_argument("--observed_mzi", type=int, default=8)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "measure":
        measure(args)
    elif args.mode == "rescan_one":
        rescan_one(args)


if __name__ == "__main__":
    main()
