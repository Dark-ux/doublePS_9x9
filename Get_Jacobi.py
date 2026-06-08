import argparse
from collections import Counter
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import utils.communication as cu
import utils.AllDecompositionUtils as du
from inter_calibration import find_Bmzi_path, switch_IN, write_port_voltage


DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
DEFAULT_SER_ADDRESS = "COM3"
DEFAULT_HEATER_ORDER = ["5u", "5d", "6u", "6d", "7u", "7d", "8u", "8d"]
PROJECT_NAME = "doublePS_9x9"
RUN_PURPOSE = "second_column_Jtheta_measurement"
ZERO_VOLTAGE_THRESHOLD_V = 0.01
MAX_PORT_CURRENT_A = None
MAX_PORT_CURRENT_JUMP_A = None
ZERO_VOLTAGE_MAX_CURRENT_A = None


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


def sine_model(x, A, w, phi, b):
    return A * np.sin(w * x + phi) + b


def choose_sigma_short_center(phi, w, scan_min=0.0, scan_max=2.0 * np.pi):
    """
    Choose a quadrature point where the fitted baseline sigma curve has maximum slope.

    For I=A*sin(w*dp+phi)+b, maximum slope occurs when w*dp+phi=k*pi.
    The selected candidate is the one inside [scan_min, scan_max] closest to the
    full-scan center.
    """
    try:
        phi = float(phi)
        w = float(w)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(phi) or not np.isfinite(w) or abs(w) <= 1e-12:
        return None
    scan_min = float(scan_min)
    scan_max = float(scan_max)
    center = 0.5 * (scan_min + scan_max)
    k_min = int(np.floor((w * scan_min + phi) / np.pi)) - 8
    k_max = int(np.ceil((w * scan_max + phi) / np.pi)) + 8
    candidates = []
    for k in range(k_min, k_max + 1):
        dp = (float(k) * np.pi - phi) / w
        if scan_min <= dp <= scan_max:
            candidates.append(float(dp))
    if not candidates:
        return None
    return float(min(candidates, key=lambda value: abs(value - center)))


def generate_sigma_short_phase_points(dp0, half_width_rad, num_points):
    warnings = []
    try:
        points = int(num_points)
    except (TypeError, ValueError):
        points = 3
        warnings.append("short_scan_points_invalid_adjusted")
    if points < 3:
        points = 3
        warnings.append("short_scan_points_too_small_adjusted")
    half_width = float(half_width_rad)
    phase_points = np.asarray(dp0 + np.linspace(-half_width, half_width, points), dtype=float)
    if np.any((phase_points < 0.0) | (phase_points > 2.0 * np.pi)):
        warnings.append("short_scan_outside_baseline_range")
    return phase_points, warnings


def fit_sigma_baseline_for_short_center(scan_path):
    scan_path = Path(scan_path)
    if not scan_path.exists():
        return None, "short_scan_center_unavailable_fallback_full"
    try:
        df = pd.read_csv(scan_path, sep=None, engine="python")
        y_col = "optical_power_uW" if "optical_power_uW" in df.columns else "pow(uW)"
        if "dp" not in df.columns or y_col not in df.columns:
            return None, "short_scan_center_unavailable_fallback_full"
        x = pd.to_numeric(df["dp"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 4:
            return None, "short_scan_center_unavailable_fallback_full"
        x = x[valid]
        y = y[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        span = float(np.nanmax(y) - np.nanmin(y))
        A0 = max(0.5 * span, 1e-9)
        w0 = 1.0
        phi0 = 0.0
        b0 = float(np.nanmean(y))
        popt, _ = curve_fit(
            sine_model,
            x,
            y,
            p0=[A0, w0, phi0, b0],
            bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        return {"A": float(popt[0]), "w": float(popt[1]), "phi": float(popt[2]), "b": float(popt[3])}, ""
    except Exception:
        return None, "short_scan_center_unavailable_fallback_full"


def resolve_sigma_phase_points_for_scan(observed_mzi, args, run_dir, stage):
    requested_mode = str(getattr(args, "sigma_scan_mode", "full")).strip().lower()
    if requested_mode not in {"full", "short"}:
        requested_mode = "full"
    metadata = {
        "sigma_scan_mode": "full",
        "sigma_short_center_dp": np.nan,
        "sigma_short_half_width_rad": float(getattr(args, "sigma_short_half_width_rad", 0.25)),
        "sigma_short_points": int(getattr(args, "sigma_short_points", 5)),
        "sigma_short_center_policy": str(getattr(args, "sigma_short_center_policy", "quadrature")),
        "short_scan_fallback_full": False,
    }
    if str(stage).strip().lower() != "perturb" or requested_mode == "full":
        return generate_sigma_phase_points(args), metadata, ""

    baseline_path = Path(run_dir) / "baseline" / "sigma" / f"obs{int(observed_mzi)}_inter_scan.txt"
    fit, warning = fit_sigma_baseline_for_short_center(baseline_path)
    if warning or fit is None:
        metadata["short_scan_fallback_full"] = True
        return generate_sigma_phase_points(args), metadata, "short_scan_center_unavailable_fallback_full"
    dp0 = choose_sigma_short_center(fit["phi"], fit["w"])
    if dp0 is None:
        metadata["short_scan_fallback_full"] = True
        return generate_sigma_phase_points(args), metadata, "short_scan_center_unavailable_fallback_full"
    phase_points, warnings = generate_sigma_short_phase_points(
        dp0,
        getattr(args, "sigma_short_half_width_rad", 0.25),
        getattr(args, "sigma_short_points", 5),
    )
    metadata.update(
        {
            "sigma_scan_mode": "short",
            "sigma_short_center_dp": float(dp0),
            "sigma_short_points": int(len(phase_points)),
            "short_scan_fallback_full": False,
        }
    )
    return phase_points, metadata, merge_warning_text(["short_scan_used", *warnings])


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


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def sanitize_column_name(name):
    text = str(name).strip()
    safe_chars = []
    for char in text:
        if char.isalnum() or char == "_":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "unknown"


def format_port_label(port):
    try:
        return f"{int(port):03d}"
    except (TypeError, ValueError):
        return sanitize_column_name(port)


def infer_port_list_from_working_data(working_data, port_list=None):
    if port_list is not None:
        return [int(port) if str(port).strip().isdigit() else str(port).strip() for port in port_list]
    return list(range(1, int(len(working_data)) + 1))


def _warning_items(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            items.extend(_warning_items(item))
        return items
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text.strip("[]")
        text = text.replace("'", "").replace('"', "")
        return [part.strip() for part in text.split(",") if part.strip()]
    return [part.strip() for part in text.split(";") if part.strip()]


def normalize_warning_field(value):
    seen = set()
    normalized = []
    for item in _warning_items(value):
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return "; ".join(normalized)


def join_warnings(warning_list):
    return normalize_warning_field(warning_list)


def append_warning(existing_warning, new_warning):
    return join_warnings([existing_warning, new_warning])


def merge_warning_text(*parts):
    return join_warnings(parts)


def scan_group_name(stage, perturb_label):
    if str(stage).strip().lower() == "perturb":
        return f"perturb_{str(perturb_label).strip().lower()}"
    return "baseline"


def perturb_heater_name(stage, perturb_label):
    if str(stage).strip().lower() == "perturb":
        return str(perturb_label).strip().lower()
    return "baseline"


def make_scan_point_id(scan_group, scan_kind, observed_mzi, point_index):
    safe_group = sanitize_column_name(scan_group)
    safe_kind = sanitize_column_name(scan_kind)
    return f"{safe_group}_{safe_kind}_obs{int(observed_mzi)}_point{int(point_index) - 1:03d}"


def heater_snapshot_columns(heater_order=DEFAULT_HEATER_ORDER):
    columns = []
    for heater in heater_order:
        heater_key = str(heater).strip().lower()
        columns.extend([f"P_{heater_key}_w", f"V_{heater_key}_v", f"I_{heater_key}_a"])
    return columns


def port_set_voltage_columns(port_list):
    return [f"port_{format_port_label(port)}_set_voltage_v" for port in port_list]


def port_current_columns(port_list):
    return [f"port_{format_port_label(port)}_current_a" for port in port_list]


def split_warning_tokens(value):
    return [item for item in normalize_warning_field(value).split("; ") if item]


def warning_count_key(token):
    token = str(token).strip()
    if not token:
        return ""
    if ":" in token:
        token = token.split(":", 1)[0].strip()
    if token.startswith("port_") and token.endswith("_current_read_failed"):
        return "current_read_failed"
    if token.startswith("port_") and token.endswith("_current_nan"):
        return "current_value_nan"
    if token.startswith("port_") and token.endswith("_current_non_numeric"):
        return "current_value_non_numeric"
    if token.startswith("port_voltage_missing_"):
        return "port_voltage_missing"
    if token.startswith("heater_snapshot_missing_"):
        return "heater_snapshot_missing"
    if token.startswith("heater_resistance_missing_"):
        return "heater_resistance_missing"
    if token.startswith("heater_port_missing_"):
        return "heater_port_missing"
    if token.startswith("heater_voltage_missing_"):
        return "heater_voltage_missing"
    return token


def normalize_scan_row_warnings(row):
    row = dict(row)
    row["row_warning"] = normalize_warning_field(row.get("row_warning", ""))
    row["current_read_warning"] = normalize_warning_field(row.get("current_read_warning", ""))
    return row


def validate_required_timestamps(row):
    missing = []
    for name in ("upload_timestamp", "current_read_timestamp", "opm_read_timestamp"):
        if not str(row.get(name, "")).strip():
            missing.append(name)
    if missing:
        row["row_warning"] = append_warning(row.get("row_warning", ""), "timestamp_missing")
    return row


def _is_missing_row_value(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() == ""


def validate_required_scan_metadata(row):
    common_fields = [
        "run_id",
        "scan_group",
        "scan_type",
        "observed_mzi",
        "perturb_heater",
        "point_index",
        "scan_point_id",
    ]
    scan_type = str(row.get("scan_type", "")).strip()
    if scan_type == "delta_probe":
        required_fields = [
            *common_fields,
            "probe_arm",
            "probe_heater",
            "output_channel",
            "settle_time_s",
        ]
    elif scan_type == "sigma_inter":
        required_fields = [
            *common_fields,
            "bmzi",
            "dp",
            "input_channel",
            "output_channel",
            "path",
            "state",
            "settle_time_s",
            "fold_happened",
        ]
    else:
        required_fields = common_fields
    if any(_is_missing_row_value(row.get(field, "")) for field in required_fields):
        row["row_warning"] = append_warning(row.get("row_warning", ""), "scan_row_incomplete")
    return row


def finalize_scan_row(row):
    row = normalize_scan_row_warnings(row)
    row = validate_required_timestamps(row)
    row = validate_required_scan_metadata(row)
    return normalize_scan_row_warnings(row)


def _collect_warning_strings(*parts):
    warnings = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            warnings.extend(str(item).strip() for item in part if str(item).strip())
        else:
            text = str(part).strip()
            if text and text.lower() != "nan":
                warnings.append(text)
    return warnings


def get_all_heater_snapshot(working_data, mzi_table, heater_order=DEFAULT_HEATER_ORDER):
    snapshot = {}
    warnings = []
    for heater in heater_order:
        heater_key = str(heater).strip().lower()
        for prefix in ("P", "V", "I"):
            suffix = "w" if prefix == "P" else "v" if prefix == "V" else "a"
            snapshot[f"{prefix}_{heater_key}_{suffix}"] = np.nan
        if len(heater_key) < 2 or heater_key[-1] not in {"u", "d"}:
            warnings.append(f"heater_snapshot_missing_{heater_key}")
            continue
        try:
            mzi_id = int(heater_key[:-1])
            arm = heater_key[-1]
        except ValueError:
            warnings.append(f"heater_snapshot_missing_{heater_key}")
            continue
        entry = mzi_table.get(str(mzi_id), {})
        arm_idx = 0 if arm == "u" else 1
        ports = entry.get("ports", [])
        heater_r = entry.get("heater_R", [])
        if len(ports) <= arm_idx:
            warnings.append(f"heater_port_missing_{heater_key}")
            continue
        if len(heater_r) <= arm_idx:
            warnings.append(f"heater_resistance_missing_{heater_key}")
            continue
        try:
            port = int(ports[arm_idx])
            resistance = float(heater_r[arm_idx])
        except (TypeError, ValueError):
            warnings.append(f"heater_snapshot_missing_{heater_key}")
            continue
        try:
            voltage = float(get_port_voltage(working_data, port))
        except Exception:
            warnings.append(f"heater_voltage_missing_{heater_key}")
            continue
        if not np.isfinite(resistance) or resistance <= 0:
            warnings.append(f"heater_resistance_missing_{heater_key}")
            continue
        if not np.isfinite(voltage):
            warnings.append(f"heater_voltage_missing_{heater_key}")
            continue
        snapshot[f"P_{heater_key}_w"] = float(voltage**2 / resistance)
        snapshot[f"V_{heater_key}_v"] = float(voltage)
        snapshot[f"I_{heater_key}_a"] = float(voltage / resistance)
    return snapshot, warnings


def get_all_port_set_voltages(working_data, port_list=None):
    ports = infer_port_list_from_working_data(working_data, port_list)
    voltages = {}
    warnings = []
    for port in ports:
        label = format_port_label(port)
        col = f"port_{label}_set_voltage_v"
        try:
            port_int = int(port)
            if port_int < 1 or port_int > len(working_data):
                raise IndexError(f"port {port_int} outside working_data length {len(working_data)}")
            value = float(get_port_voltage(working_data, port_int))
            if not np.isfinite(value):
                warnings.append(f"port_voltage_missing_{label}")
                value = np.nan
        except Exception as exc:
            warnings.append(f"port_voltage_missing_{label}")
            value = np.nan
        voltages[col] = value
    if not ports:
        warnings.append("port_list_missing")
    return voltages, warnings


def read_all_port_currents(instrument, port_list=None, allow_missing_current_read=False):
    if port_list is None:
        channel_count = int(getattr(cu, "CHANNEL_NUM", 128))
        port_list = list(range(1, channel_count + 1))
    ports = [int(port) if str(port).strip().isdigit() else str(port).strip() for port in port_list]
    currents = {f"port_{format_port_label(port)}_current_a": np.nan for port in ports}
    warnings = []

    if not ports:
        warnings.append("port_list_missing")
        if not allow_missing_current_read:
            raise RuntimeError("; ".join(warnings))
        return currents, warnings
    if instrument is None:
        warnings.append("current_read_failed")
        if not allow_missing_current_read:
            raise RuntimeError("; ".join(warnings))
        warnings.append("current_read_missing_allowed")
        return currents, warnings

    try:
        raw_currents_ma = cu.read_current(instrument)
    except Exception as exc:
        warnings.append("current_read_failed")
        if not allow_missing_current_read:
            raise RuntimeError("; ".join(warnings)) from exc
        warnings.append("current_read_missing_allowed")
        return currents, warnings

    if raw_currents_ma is None:
        warnings.append("current_read_failed")
        if not allow_missing_current_read:
            raise RuntimeError("; ".join(warnings))
        warnings.append("current_read_missing_allowed")
        return currents, warnings

    for port in ports:
        label = format_port_label(port)
        col = f"port_{label}_current_a"
        try:
            port_int = int(port)
            value_ma = raw_currents_ma[port_int - 1]
            if value_ma is None:
                warnings.append(f"port_{label}_current_nan")
                currents[col] = np.nan
                continue
            try:
                value_a = float(value_ma) * 1e-3
            except (TypeError, ValueError):
                warnings.append(f"port_{label}_current_non_numeric")
                currents[col] = np.nan
                continue
            if not np.isfinite(value_a):
                warnings.append(f"port_{label}_current_nan")
                currents[col] = np.nan
                continue
            currents[col] = value_a
        except Exception as exc:
            warnings.append(f"port_{label}_current_read_failed")
            currents[col] = np.nan

    if warnings:
        warnings.append("current_read_partial_missing")
        if allow_missing_current_read:
            warnings.append("current_read_missing_allowed")
    return currents, warnings


def collect_scan_point_state(working_data, mzi_table, args, instrument):
    port_list = infer_port_list_from_working_data(working_data, getattr(args, "_all_ports", None))
    heater_order = getattr(args, "_heaters", DEFAULT_HEATER_ORDER)
    heater_snapshot, heater_warnings = get_all_heater_snapshot(working_data, mzi_table, heater_order)
    port_set_voltages, voltage_warnings = get_all_port_set_voltages(working_data, port_list)
    current_warnings = []
    if parse_bool(getattr(args, "read_all_port_currents", True)):
        port_currents, current_warnings = read_all_port_currents(
            instrument,
            port_list=port_list,
            allow_missing_current_read=parse_bool(getattr(args, "allow_missing_current_read", False)),
        )
    else:
        port_currents = {f"port_{format_port_label(port)}_current_a": np.nan for port in port_list}
        current_warnings.append("current_read_missing_allowed")
    fields = {}
    fields.update(heater_snapshot)
    fields.update(port_set_voltages)
    fields.update(port_currents)
    warnings = [*heater_warnings, *voltage_warnings, *current_warnings]
    return fields, warnings, merge_warning_text(current_warnings)


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
        warning = "opm_read_failed"
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
    "run_id",
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
    "run_id",
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
    merged_warning = merge_warning_text(warning, route_warning)
    return {
        "run_id": str(first_nonempty_df_value(df, "run_id", "")),
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
        "failure_reason": normalize_warning_field(failure_reason),
    }


def validate_delta_scan_df(df, args, expected_points):
    y = pd.to_numeric(df["optical_power_uW"], errors="coerce").to_numpy(dtype=float)
    reasons = []
    warnings = []
    if y.size != int(expected_points) or np.isnan(y).any():
        reasons.append("scan_row_incomplete")
    median_y = float(np.nanmedian(y)) if y.size else 0.0
    near_zero_mask = (y < float(args.near_zero_absolute_uW)) & (median_y > float(args.near_zero_median_min_uW))
    near_zero_count = int(np.sum(near_zero_mask))
    if near_zero_count:
        reasons.append("near_zero_point")
    neighbor_count = 0
    if y.size >= 3:
        for idx in range(1, y.size - 1):
            neighbor_mean = (float(y[idx - 1]) + float(y[idx + 1])) / 2.0
            if abs(float(y[idx]) - neighbor_mean) > float(args.delta_neighbor_jump_ratio) * max(abs(neighbor_mean), 1e-9):
                neighbor_count += 1
        if neighbor_count:
            reasons.append("neighbor_jump")
    edge_count = 0
    if y.size >= 2:
        if abs(float(y[0]) - float(y[1])) > float(args.delta_edge_jump_ratio) * max(abs(float(y[1])), 1e-9):
            edge_count += 1
        if abs(float(y[-1]) - float(y[-2])) > float(args.delta_edge_jump_ratio) * max(abs(float(y[-2])), 1e-9):
            edge_count += 1
        if edge_count:
            reasons.append("edge_jump")
    rel_std = pd.to_numeric(df.get("opm_relative_std", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    unstable_count = int(np.sum(rel_std.to_numpy(dtype=float) > float(args.opm_relative_std_threshold)))
    if unstable_count > int(args.max_unstable_points_per_scan):
        reasons.append("opm_read_failed")
    curve_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    if curve_range < float(args.delta_min_curve_range_uW):
        warnings.append("low_delta_curve_range")
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
        reasons.append("scan_row_incomplete")
    median_y = float(np.nanmedian(y)) if y.size else 0.0
    near_zero_count = int(np.sum(y < float(args.sigma_min_median_power_uW))) if y.size else 0
    if median_y < float(args.sigma_min_median_power_uW):
        warnings.append("low_sigma_median_power")
    rel_std = pd.to_numeric(df.get("opm_relative_std", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    unstable_count = int(np.sum(rel_std.to_numpy(dtype=float) > float(args.opm_relative_std_threshold)))
    if unstable_count > int(args.max_unstable_points_per_scan):
        reasons.append("opm_read_failed")
    curve_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    if curve_range < float(args.delta_min_curve_range_uW):
        warnings.append("low_sigma_visibility")
    counts = {
        "unstable_point_count": unstable_count,
        "near_zero_point_count": near_zero_count,
        "neighbor_jump_count": 0,
        "edge_jump_count": 0,
    }
    return len(reasons) == 0, "; ".join(warnings), "; ".join(reasons), counts


def suggested_action_for_failure(reason):
    if "opm_read_failed" in reason or "unstable OPM" in reason:
        return "increase settle time"
    if "near_zero_point" in reason or "near-zero" in reason:
        return "check optical path"
    if "jump" in reason:
        return "check voltage upload"
    if "scan_row_incomplete" in reason or "incomplete" in reason:
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
        warnings.append(f"route_lower_port_missing_mzi{int(mzi_id)}")
    elif lower_policy == "zero":
        lower_voltage = 0.0
        write_port_voltage(lower_port, lower_voltage, working_data)
    else:
        lower_voltage = float(get_port_voltage(working_data, lower_port))
        lower_zero_applied = False
        warnings.append(
            f"route_lower_policy_keep_base_mzi{int(mzi_id)}"
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


def scan_delta_probe_once(
    observed_mzi,
    probe_arm,
    out_path,
    base_working_data,
    hardware,
    mzi_table,
    args,
    progress_label="",
    try_index=1,
    stage="",
    perturb_label="",
):
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
    run_id = getattr(args, "_run_id", "")
    scan_group = scan_group_name(stage, perturb_label)
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
        upload_timestamp = now_timestamp()
        time.sleep(float(args.point_settle_time_s))
        if float(getattr(args, "current_read_settle_s", 0.0)) > 0.0:
            time.sleep(float(args.current_read_settle_s))
        point_state, state_warnings, current_read_warning = collect_scan_point_state(
            scan_data,
            mzi_table,
            args,
            hardware.get("mcv"),
        )
        current_read_timestamp = now_timestamp()
        opm = read_stable_opm(
            hardware["opm2"],
            output_channel,
            args,
            reupload_callback=lambda sd=scan_data, lbl=label: upload_voltage_checked(hardware["mcv"], sd, args, f"{lbl} reupload"),
        )
        read_timestamp = now_timestamp()
        row_warning = merge_warning_text(state_warnings, opm.get("warning", ""))
        perturb_heater = perturb_heater_name(stage, perturb_label)
        perturb_power_w = float(args.delta_power_w) if str(stage).strip().lower() == "perturb" else 0.0
        row = {
                "run_id": run_id,
                "scan_group": scan_group,
                "scan_point_id": make_scan_point_id(scan_group, "delta", observed_mzi, point_idx),
                "target": int(observed_mzi),
                "observed_mzi": int(observed_mzi),
                "probe_arm": probe_arm,
                "probe_heater": f"{int(observed_mzi)}{str(probe_arm).lower()}",
                "perturb_heater": perturb_heater,
                "perturb_power_w": perturb_power_w,
                "point_index": int(point_idx),
                "arm_name": info["arm_name"],
                "arm_index": info["arm_index"],
                "port": int(info["port"]),
                "probe_axis_power_w": float(offset),
                "target_power_w": float(target_power),
                "measured_power_w": float(target_power),
                "voltage_v": float(voltage),
                "target_voltage_v": float(voltage),
                "optical_power_uW": float(opm["opm_median_uW"]),
                **opm,
                **point_state,
                "scan_type": "delta_probe",
                "output_channel": int(output_channel),
                "scan_try_index": int(try_index),
                "upload_timestamp": upload_timestamp,
                "current_read_timestamp": current_read_timestamp,
                "read_timestamp": read_timestamp,
                "opm_read_timestamp": read_timestamp,
                "point_settle_time_s": float(args.point_settle_time_s),
                "settle_time_s": float(args.point_settle_time_s),
                "current_read_warning": current_read_warning,
                "row_warning": row_warning,
                "timestamp": read_timestamp,
            }
        rows.append(finalize_scan_row(row))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.12f")
    return out_path, df


def scan_sigma_inter_once(
    observed_mzi,
    out_path,
    base_working_data,
    hardware,
    mzi_table,
    args,
    sigma_bmzi_map,
    progress_label="",
    try_index=1,
    stage="",
    perturb_label="",
    phase_points=None,
    sigma_scan_metadata=None,
    sigma_scan_warning="",
):
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
    if phase_points is None:
        phase_points = generate_sigma_phase_points(args)
    phase_points = np.asarray(phase_points, dtype=float)
    sigma_scan_metadata = dict(sigma_scan_metadata or {})
    sigma_scan_metadata.setdefault("sigma_scan_mode", "full")
    sigma_scan_metadata.setdefault("sigma_short_center_dp", np.nan)
    sigma_scan_metadata.setdefault("sigma_short_half_width_rad", float(getattr(args, "sigma_short_half_width_rad", 0.25)))
    sigma_scan_metadata.setdefault("sigma_short_points", int(getattr(args, "sigma_short_points", 5)))
    sigma_scan_metadata.setdefault("sigma_short_center_policy", str(getattr(args, "sigma_short_center_policy", "quadrature")))
    sigma_scan_metadata.setdefault("short_scan_fallback_full", False)
    rows = []
    total_points = int(len(phase_points))
    run_id = getattr(args, "_run_id", "")
    scan_group = scan_group_name(stage, perturb_label)
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
        upload_timestamp = now_timestamp()
        time.sleep(float(args.point_settle_time_s))
        if float(getattr(args, "current_read_settle_s", 0.0)) > 0.0:
            time.sleep(float(args.current_read_settle_s))
        point_state, state_warnings, current_read_warning = collect_scan_point_state(
            scan_data,
            mzi_table,
            args,
            hardware.get("mcv"),
        )
        current_read_timestamp = now_timestamp()
        opm = read_stable_opm(
            hardware["opm2"],
            int(output_idx) + 1,
            args,
            reupload_callback=lambda sd=scan_data, lbl=label: upload_voltage_checked(hardware["mcv"], sd, args, f"{lbl} reupload"),
        )
        route_warning = str(route_fields.get("route_warning", "")).strip()
        combined_warning = merge_warning_text(opm.get("warning", ""), route_warning)
        opm["warning"] = combined_warning
        opm["point_warning"] = combined_warning
        read_timestamp = now_timestamp()
        row_warning = merge_warning_text(state_warnings, combined_warning, sigma_scan_warning)
        perturb_heater = perturb_heater_name(stage, perturb_label)
        perturb_power_w = float(args.delta_power_w) if str(stage).strip().lower() == "perturb" else 0.0
        fold_happened = bool(upper_folds or lower_folds)
        fold_detail = ""
        if fold_happened:
            fold_detail = (
                f"upper {p_upper_unfolded:.12f}->{p_upper:.12f} W ({upper_folds}); "
                f"lower {p_lower_unfolded:.12f}->{p_lower:.12f} W ({lower_folds})"
            )
            row_warning = append_warning(row_warning, "fold_happened")
            if not fold_detail:
                row_warning = append_warning(row_warning, "fold_detail_missing")
        row = {
                "run_id": run_id,
                "scan_group": scan_group,
                "scan_point_id": make_scan_point_id(scan_group, "sigma", observed_mzi, point_idx),
                "target": target,
                "observed_mzi": target,
                "bmzi": int(sigma_bmzi_map.get(str(target), bmzi)),
                "perturb_heater": perturb_heater,
                "perturb_power_w": perturb_power_w,
                "point_index": int(point_idx),
                "input_channel": int(input_idx) + 1,
                "output_channel": int(output_idx) + 1,
                "path": json.dumps([int(v) for v in path]),
                "state": json.dumps([str(v) for v in state]),
                **route_fields,
                **sigma_scan_metadata,
                "dp": float(dp),
                "primary_heater": f"{target}u",
                "secondary_heater": f"{target}d",
                "v_primary": float(v_upper),
                "v_secondary": float(v_lower),
                "v_primary_v": float(v_upper),
                "v_secondary_v": float(v_lower),
                "p_primary": float(p_upper),
                "p_secondary": float(p_lower),
                "p_primary_w": float(p_upper),
                "p_secondary_w": float(p_lower),
                "p_primary_unfolded": float(p_upper_unfolded),
                "p_secondary_unfolded": float(p_lower_unfolded),
                "upper_fold_count": int(upper_folds),
                "lower_fold_count": int(lower_folds),
                "fold_happened": fold_happened,
                "fold_detail": fold_detail,
                "optical_power_uW": float(opm["opm_median_uW"]),
                "pow(uW)": float(opm["opm_median_uW"]),
                **opm,
                **point_state,
                "scan_type": "sigma_inter",
                "scan_try_index": int(try_index),
                "upload_timestamp": upload_timestamp,
                "current_read_timestamp": current_read_timestamp,
                "read_timestamp": read_timestamp,
                "opm_read_timestamp": read_timestamp,
                "point_settle_time_s": float(args.point_settle_time_s),
                "settle_time_s": float(args.point_settle_time_s),
                "current_read_warning": current_read_warning,
                "row_warning": row_warning,
                "timestamp": read_timestamp,
            }
        rows.append(finalize_scan_row(row))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.12f")
    return out_path, df


def mark_and_save_scan_df(df, out_path, valid, warning, failure_reason, quality_score):
    df = df.copy()
    df["scan_valid"] = bool(valid)
    df["scan_failure_reason"] = normalize_warning_field(failure_reason)
    df["scan_warning"] = normalize_warning_field(warning)
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
            stage=stage,
            perturb_label=perturb_label,
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
        "run_id": getattr(args, "_run_id", ""),
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
    phase_points, sigma_scan_metadata, sigma_scan_warning = resolve_sigma_phase_points_for_scan(
        observed_mzi,
        args,
        run_dir,
        stage,
    )
    expected_points = len(phase_points)
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
            stage=stage,
            perturb_label=perturb_label,
            phase_points=phase_points,
            sigma_scan_metadata=sigma_scan_metadata,
            sigma_scan_warning=sigma_scan_warning,
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
        "run_id": getattr(args, "_run_id", ""),
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


def _relative_path(path, root):
    return str(Path(path).relative_to(Path(root))).replace("\\", "/")


def collect_run_data_files(run_dir):
    run_dir = Path(run_dir)
    groups = {
        "baseline_delta": [],
        "baseline_sigma": [],
        "perturb_delta": [],
        "perturb_sigma": [],
    }
    scan_files = []
    for path in sorted(run_dir.rglob("*.txt")):
        name = path.name
        if "_try" in name:
            continue
        rel = _relative_path(path, run_dir)
        parts = Path(rel).parts
        if len(parts) < 3:
            continue
        scan_group, scan_kind = parts[0], parts[1]
        if scan_group == "baseline" and scan_kind == "delta" and name.endswith("_probe.txt"):
            groups["baseline_delta"].append(rel)
            scan_files.append(path)
        elif scan_group == "baseline" and scan_kind == "sigma" and name.endswith("_inter_scan.txt"):
            groups["baseline_sigma"].append(rel)
            scan_files.append(path)
        elif scan_group.startswith("perturb_") and scan_kind == "delta" and name.endswith("_probe.txt"):
            groups["perturb_delta"].append(rel)
            scan_files.append(path)
        elif scan_group.startswith("perturb_") and scan_kind == "sigma" and name.endswith("_inter_scan.txt"):
            groups["perturb_sigma"].append(rel)
            scan_files.append(path)
    return groups, scan_files


def read_scan_frames(scan_files):
    frames = []
    for path in scan_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    return frames


def summarize_warnings(frames):
    total_rows = 0
    rows_with_warning = 0
    counts = Counter()
    for df in frames:
        total_rows += int(len(df))
        for row in df.to_dict("records"):
            tokens = []
            for col in ("row_warning", "current_read_warning"):
                tokens.extend(split_warning_tokens(row.get(col, "")))
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                rows_with_warning += 1
            for token in tokens:
                key = warning_count_key(token)
                if key:
                    counts[key] += 1
    return {
        "total_rows": total_rows,
        "rows_with_warning": rows_with_warning,
        "warning_counts": dict(sorted(counts.items())),
    }


def summarize_current_reads(frames, args, current_columns):
    total_rows = 0
    rows_with_warning = 0
    nan_values = 0
    numeric_values = 0
    for df in frames:
        total_rows += int(len(df))
        if "current_read_warning" in df.columns:
            warnings = df["current_read_warning"].fillna("").astype(str).map(normalize_warning_field)
            rows_with_warning += int(np.sum(warnings != ""))
        present_cols = [col for col in current_columns if col in df.columns]
        if present_cols:
            values = df[present_cols].apply(pd.to_numeric, errors="coerce")
            nan_values += int(values.isna().sum().sum())
            numeric_values += int(np.isfinite(values.to_numpy(dtype=float)).sum())
    mode = "normal"
    if parse_bool(getattr(args, "allow_missing_current_read", False)) and nan_values:
        mode = "offline_or_missing_current_allowed"
    if not parse_bool(getattr(args, "read_all_port_currents", True)):
        mode = "current_read_disabled"
    return {
        "read_all_port_currents": bool(parse_bool(getattr(args, "read_all_port_currents", True))),
        "allow_missing_current_read": bool(parse_bool(getattr(args, "allow_missing_current_read", False))),
        "mode": mode,
        "total_rows": total_rows,
        "rows_with_current_read_success": max(0, total_rows - rows_with_warning),
        "rows_with_current_read_warning": rows_with_warning,
        "current_columns_count": len(current_columns),
        "current_columns": current_columns,
        "numeric_current_values": numeric_values,
        "nan_current_values": nan_values,
    }


def summarize_port_current_anomalies(frames, current_columns, set_voltage_columns):
    nan_counts = Counter()
    non_numeric_counts = Counter()
    over_abs_limit_counts = Counter()
    jump_counts = Counter()
    zero_voltage_nonzero_current_counts = Counter()
    for df in frames:
        for col in current_columns:
            if col not in df.columns:
                continue
            raw = df[col]
            numeric = pd.to_numeric(raw, errors="coerce")
            non_empty = raw.notna() & (raw.astype(str).str.strip() != "")
            non_numeric_counts[col] += int(np.sum(non_empty & numeric.isna()))
            nan_counts[col] += int(numeric.isna().sum())
            if MAX_PORT_CURRENT_A is not None:
                over_abs_limit_counts[col] += int(np.sum(np.abs(numeric.fillna(0.0)) > float(MAX_PORT_CURRENT_A)))
            if MAX_PORT_CURRENT_JUMP_A is not None and len(numeric) >= 2:
                jump_counts[col] += int(np.sum(np.abs(np.diff(numeric.to_numpy(dtype=float))) > float(MAX_PORT_CURRENT_JUMP_A)))
            if ZERO_VOLTAGE_MAX_CURRENT_A is not None:
                port_label = col[len("port_") : -len("_current_a")]
                voltage_col = f"port_{port_label}_set_voltage_v"
                if voltage_col in set_voltage_columns and voltage_col in df.columns:
                    voltage = pd.to_numeric(df[voltage_col], errors="coerce")
                    zero_mask = voltage.abs() <= float(ZERO_VOLTAGE_THRESHOLD_V)
                    current_mask = numeric.abs() > float(ZERO_VOLTAGE_MAX_CURRENT_A)
                    zero_voltage_nonzero_current_counts[col] += int(np.sum(zero_mask & current_mask))
    return {
        "nan_counts": dict(sorted((k, int(v)) for k, v in nan_counts.items() if v)),
        "non_numeric_counts": dict(sorted((k, int(v)) for k, v in non_numeric_counts.items() if v)),
        "over_abs_limit_counts": dict(sorted((k, int(v)) for k, v in over_abs_limit_counts.items() if v)),
        "jump_counts": dict(sorted((k, int(v)) for k, v in jump_counts.items() if v)),
        "zero_voltage_nonzero_current_counts": dict(
            sorted((k, int(v)) for k, v in zero_voltage_nonzero_current_counts.items() if v)
        ),
        "thresholds": {
            "MAX_PORT_CURRENT_A": MAX_PORT_CURRENT_A,
            "MAX_PORT_CURRENT_JUMP_A": MAX_PORT_CURRENT_JUMP_A,
            "ZERO_VOLTAGE_THRESHOLD_V": ZERO_VOLTAGE_THRESHOLD_V,
            "ZERO_VOLTAGE_MAX_CURRENT_A": ZERO_VOLTAGE_MAX_CURRENT_A,
        },
    }


def scan_schema_from_frames(frames, heater_columns, current_columns, set_voltage_columns):
    delta_columns = []
    sigma_columns = []
    for df in frames:
        if "scan_type" not in df.columns:
            continue
        scan_types = set(df["scan_type"].dropna().astype(str).tolist())
        if "delta_probe" in scan_types and not delta_columns:
            delta_columns = list(df.columns)
        if "sigma_inter" in scan_types and not sigma_columns:
            sigma_columns = list(df.columns)
    return {
        "delta_scan_columns": delta_columns,
        "sigma_scan_columns": sigma_columns,
        "heater_snapshot_columns": heater_columns,
        "port_current_columns": current_columns,
        "port_set_voltage_columns": set_voltage_columns,
        "warning_fields": ["row_warning", "current_read_warning", "warning", "point_warning", "scan_warning"],
        "timestamp_fields": ["upload_timestamp", "current_read_timestamp", "opm_read_timestamp", "timestamp"],
    }


def write_scan_column_schema(run_dir, schema):
    with (Path(run_dir) / "scan_column_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def write_run_manifest(run_dir, args, probe_map, sigma_bmzi_map, timestamp_start, timestamp_end="", note=""):
    run_dir = Path(run_dir)
    port_list = list(range(1, int(getattr(cu, "CHANNEL_NUM", 128)) + 1))
    heater_columns = heater_snapshot_columns(getattr(args, "_heaters", DEFAULT_HEATER_ORDER))
    current_columns = port_current_columns(port_list)
    set_voltage_columns = port_set_voltage_columns(port_list)
    data_files, scan_files = collect_run_data_files(run_dir)
    frames = read_scan_frames(scan_files)
    warning_summary = summarize_warnings(frames)
    current_read_summary = summarize_current_reads(frames, args, current_columns)
    anomaly_summary = summarize_port_current_anomalies(frames, current_columns, set_voltage_columns)
    schema = scan_schema_from_frames(frames, heater_columns, current_columns, set_voltage_columns)
    write_scan_column_schema(run_dir, schema)

    manifest = {
        "run_id": getattr(args, "_run_id", run_dir.name),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "project": PROJECT_NAME,
        "purpose": RUN_PURPOSE,
        "mzi_ids": [int(mzi_id) for mzi_id in getattr(args, "_mzi_ids", [])],
        "heater_order": list(getattr(args, "_heaters", DEFAULT_HEATER_ORDER)),
        "delta_power_w": float(getattr(args, "delta_power_w", np.nan)),
        "probe_map": {str(k): v for k, v in probe_map.items()},
        "sigma_bmzi_map": {str(k): int(v) for k, v in sigma_bmzi_map.items()},
        "sigma_scan_mode": str(getattr(args, "sigma_scan_mode", "full")),
        "baseline_sigma_scan_mode": "full",
        "perturb_sigma_scan_mode": "short" if str(getattr(args, "sigma_scan_mode", "full")).strip().lower() == "short" else "full",
        "sigma_short_points": int(getattr(args, "sigma_short_points", 5)),
        "sigma_short_half_width_rad": float(getattr(args, "sigma_short_half_width_rad", 0.25)),
        "sigma_short_center_policy": str(getattr(args, "sigma_short_center_policy", "quadrature")),
        "route_lower_policy": normalize_route_lower_policy(getattr(args, "route_lower_policy", "zero")),
        "read_all_port_currents": bool(parse_bool(getattr(args, "read_all_port_currents", True))),
        "allow_missing_current_read": bool(parse_bool(getattr(args, "allow_missing_current_read", False))),
        "current_read_settle_s": float(getattr(args, "current_read_settle_s", 0.0)),
        "voltage_limit_v": float(getattr(args, "voltage_limit_v", np.nan)),
        "power_limit_w": float(getattr(args, "power_limit_w", np.nan)),
        "data_files": data_files,
        "current_columns": current_columns,
        "set_voltage_columns": set_voltage_columns,
        "heater_snapshot_columns": heater_columns,
        "warning_summary": warning_summary,
        "current_read_summary": current_read_summary,
        "port_current_anomaly_summary": anomaly_summary,
        "schema_file": "scan_column_schema.json",
        "note": note,
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


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
                    "run_id": getattr(args, "_run_id", ""),
                    "perturbed_heater": heater,
                    "delta_power_w": float(args.delta_power_w),
                    "baseline_power_w": None,
                    "perturbed_power_w": None,
                    "heater_order": args._heaters,
                    "mzi_ids": args._mzi_ids,
                    "timestamp": now_timestamp(),
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
    args._all_ports = list(range(1, int(getattr(cu, "CHANNEL_NUM", 128)) + 1))
    probe_map = parse_probe_map(args.probe_map, args._mzi_ids)
    sigma_bmzi_map = parse_sigma_bmzi_map(args.sigma_bmzi_map, args._mzi_ids)
    run_dir = Path(args.out_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args._run_id = run_dir.name
    timestamp_start = now_timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": args._run_id,
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
        "sigma_scan_mode": str(args.sigma_scan_mode),
        "baseline_sigma_scan_mode": "full",
        "perturb_sigma_scan_mode": "short" if str(args.sigma_scan_mode).strip().lower() == "short" else "full",
        "sigma_short_points": int(args.sigma_short_points),
        "sigma_short_half_width_rad": float(args.sigma_short_half_width_rad),
        "sigma_short_center_policy": str(args.sigma_short_center_policy),
        "opm_reads_per_point": int(args.opm_reads_per_point),
        "max_scan_retry": int(args.max_scan_retry),
        "power_limit_w": float(args.power_limit_w),
        "voltage_limit_v": float(args.voltage_limit_v),
        "point_settle_time_s": float(args.point_settle_time_s),
        "read_all_port_currents": bool(parse_bool(args.read_all_port_currents)),
        "allow_missing_current_read": bool(parse_bool(args.allow_missing_current_read)),
        "current_read_settle_s": float(args.current_read_settle_s),
        "baseline_restore_wait_s": float(args.baseline_restore_wait_s),
        "perturb_settle_time_s": float(args.perturb_settle_time_s),
        "initial_state": args.initial_state,
        "initial_power_file": args.initial_power_file,
        "dry_run": bool(args.dry_run),
        "timestamp": timestamp_start,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))
    pd.DataFrame(columns=QUALITY_COLUMNS).to_csv(run_dir / "scan_quality_summary.csv", index=False)
    pd.DataFrame(columns=FAILED_COLUMNS).to_csv(run_dir / "failed_scans.csv", index=False)

    if args.dry_run:
        write_dry_run_placeholders(run_dir, args)
        write_run_manifest(
            run_dir,
            args,
            probe_map,
            sigma_bmzi_map,
            timestamp_start=timestamp_start,
            timestamp_end=now_timestamp(),
            note="dry_run directory skeleton only; no scan rows collected",
        )
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
                        "run_id": getattr(args, "_run_id", ""),
                        "delta_power_w": float(args.delta_power_w),
                        "baseline_power_w": baseline_power,
                        "perturbed_power_w": float(pert_powers[heater]),
                        "baseline_power_vector_w": {str(k): float(v) for k, v in baseline_powers.items()},
                        "perturb_power_vector_w": {str(k): float(v) for k, v in pert_powers.items()},
                        "restored_before_measurement": True,
                        "heater_order": args._heaters,
                        "mzi_ids": args._mzi_ids,
                        "timestamp": now_timestamp(),
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
    write_run_manifest(
        run_dir,
        args,
        probe_map,
        sigma_bmzi_map,
        timestamp_start=timestamp_start,
        timestamp_end=now_timestamp(),
    )
    print(f"[Get_Jacobi] saved raw measurements to {run_dir}")
    return run_dir


def prepare_measure_args(args):
    args._mzi_ids = parse_csv_list(args.mzi_ids, int)
    args._heaters = parse_csv_list(args.heaters, str)
    args._heaters = validate_heater_order(args._heaters, args._mzi_ids)
    args._delta_offsets = generate_delta_offsets(args)
    args._sigma_phase_points = generate_sigma_phase_points(args)
    args._all_ports = list(range(1, int(getattr(cu, "CHANNEL_NUM", 128)) + 1))
    return parse_probe_map(args.probe_map, args._mzi_ids), parse_sigma_bmzi_map(args.sigma_bmzi_map, args._mzi_ids)


def rescan_one(args):
    probe_map, sigma_bmzi_map = prepare_measure_args(args)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    args._run_id = run_dir.name
    timestamp_start = now_timestamp()
    if int(args.observed_mzi) not in args._mzi_ids:
        raise ValueError(f"observed_mzi {args.observed_mzi} is not in --mzi_ids {args._mzi_ids}")
    if args.stage == "perturb" and str(args.perturb_heater) not in args._heaters:
        raise ValueError(f"perturb_heater {args.perturb_heater} is not in --heaters {args._heaters}")

    config = {
        "mode": "rescan_one",
        "run_id": args._run_id,
        "run_dir": str(run_dir),
        "stage": args.stage,
        "perturb_heater": args.perturb_heater,
        "scan_kind": args.scan_kind,
        "observed_mzi": int(args.observed_mzi),
        "route_lower_policy": normalize_route_lower_policy(args.route_lower_policy),
        "sigma_scan_mode": str(args.sigma_scan_mode),
        "baseline_sigma_scan_mode": "full",
        "perturb_sigma_scan_mode": "short" if str(args.sigma_scan_mode).strip().lower() == "short" else "full",
        "sigma_short_points": int(args.sigma_short_points),
        "sigma_short_half_width_rad": float(args.sigma_short_half_width_rad),
        "sigma_short_center_policy": str(args.sigma_short_center_policy),
        "read_all_port_currents": bool(parse_bool(args.read_all_port_currents)),
        "allow_missing_current_read": bool(parse_bool(args.allow_missing_current_read)),
        "current_read_settle_s": float(args.current_read_settle_s),
        "dry_run": bool(args.dry_run),
        "timestamp": timestamp_start,
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        target_dir = run_dir / ("baseline" if args.stage == "baseline" else f"perturb_{args.perturb_heater}") / args.scan_kind
        target_dir.mkdir(parents=True, exist_ok=True)
        write_run_manifest(
            run_dir,
            args,
            probe_map,
            sigma_bmzi_map,
            timestamp_start=timestamp_start,
            timestamp_end=now_timestamp(),
            note="rescan_one dry_run; no scan rows collected",
        )
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
                    "rescan_updated_at": now_timestamp(),
                    "run_id": getattr(args, "_run_id", ""),
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
    write_run_manifest(
        run_dir,
        args,
        probe_map,
        sigma_bmzi_map,
        timestamp_start=timestamp_start,
        timestamp_end=now_timestamp(),
        note=f"manifest refreshed by rescan_one {args.stage} {args.scan_kind} obs{int(args.observed_mzi)}",
    )
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
    p.add_argument("--sigma_scan_mode", default="full", choices=["full", "short"])
    p.add_argument("--sigma_short_points", type=int, default=5)
    p.add_argument("--sigma_short_half_width_rad", type=float, default=0.25)
    p.add_argument("--sigma_short_center_policy", default="quadrature", choices=["quadrature"])
    p.add_argument("--opm_reads_per_point", type=int, default=5)
    p.add_argument("--opm_read_interval_s", type=float, default=0.1)
    p.add_argument("--opm_relative_std_threshold", type=float, default=0.05)
    p.add_argument("--opm_max_retry_per_point", type=int, default=2)
    p.add_argument("--reupload_on_unstable_point", type=parse_bool, default=True)
    p.add_argument("--point_settle_time_s", type=float, default=2.0)
    p.add_argument("--read_all_port_currents", type=parse_bool, default=True)
    p.add_argument("--allow_missing_current_read", type=parse_bool, default=False)
    p.add_argument("--current_read_settle_s", type=float, default=0.0)
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
    p.add_argument("--sigma_scan_mode", default="full", choices=["full", "short"])
    p.add_argument("--sigma_short_points", type=int, default=5)
    p.add_argument("--sigma_short_half_width_rad", type=float, default=0.25)
    p.add_argument("--sigma_short_center_policy", default="quadrature", choices=["quadrature"])
    p.add_argument("--opm_reads_per_point", type=int, default=5)
    p.add_argument("--opm_read_interval_s", type=float, default=0.1)
    p.add_argument("--opm_relative_std_threshold", type=float, default=0.05)
    p.add_argument("--opm_max_retry_per_point", type=int, default=2)
    p.add_argument("--reupload_on_unstable_point", type=parse_bool, default=True)
    p.add_argument("--point_settle_time_s", type=float, default=2.0)
    p.add_argument("--read_all_port_currents", type=parse_bool, default=True)
    p.add_argument("--allow_missing_current_read", type=parse_bool, default=False)
    p.add_argument("--current_read_settle_s", type=float, default=0.0)
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
