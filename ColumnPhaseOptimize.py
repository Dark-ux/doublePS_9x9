import argparse
import json
import shutil
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


DEFAULT_MZI_IDS = [5, 6, 7, 8]
DEFAULT_ALPHA = 0.3
DEFAULT_LAMBDA_REG = 1e-4
DEFAULT_POWER_LIMIT_W = 0.055
DEFAULT_STEP_LIMIT_W = 0.002
DEFAULT_THETA_TOL = 0.05
DEFAULT_MAX_ITER = 10
DEFAULT_SETTLE_TIME = 2.0
DEFAULT_VOLTAGE_LIMIT_V = 6.0
DEFAULT_BW_PHASES = "-1.08057563,-1.63657319,-1.40683010"
DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
RUN_START_TIME = time.perf_counter()


@dataclass
class DirectRunConfig:
    mode: str = "iterate"
    j_theta: str = "results/J_full/J_theta_rad_per_w.csv"
    mzi_table: str = "Scandata/MZI_table.json"
    mzi_ids: str = "5,6,7,8"
    out_dir: str = "results/ColumnPhaseOptimize"

    current_power: str = "current_power_second_column.csv"
    current_theta: str | None = "current_theta_second_column.csv"
    reference_dir: str = "Scandata/current_theta_reference"
    current_dir: str = "Scandata/current_theta_measurement"
    delta_reference_source_dir: str = "jacobian_measurements/baseline"
    sigma_reference_source_dir: str = "Scandata/J_sigma/baseline"
    sigma_sign_path: str = "Scandata/J_sigma/sign_check/sigma_sign.json"
    auto_init_current_power_from_bar: bool = True
    assume_reference_zero: bool = False

    alpha: float = DEFAULT_ALPHA
    lambda_reg: float = DEFAULT_LAMBDA_REG
    power_limit_w: float = DEFAULT_POWER_LIMIT_W
    step_limit_w: float = DEFAULT_STEP_LIMIT_W
    theta_tol: float = DEFAULT_THETA_TOL
    voltage_limit_v: float = DEFAULT_VOLTAGE_LIMIT_V
    max_iter: int = DEFAULT_MAX_ITER
    settle_time: float = DEFAULT_SETTLE_TIME
    dry_run: bool = False
    ser_address: str = "COM3"
    opm2_address: str = DEFAULT_OPM2_ADDRESS

    enable_branch_search: bool = False
    branch_candidates: str = "0,1"
    pause_for_manual_theta_update: bool = False
    auto_measure_current_theta: bool = True
    probe_map: str = "5:u,6:u,7:u,8:u"
    probe_half_width_w: float = 0.001
    probe_step_w: float = 0.00025
    sigma_phase_points: str = "0,0.785398,1.570796,2.356194,3.141593,3.926991,4.712389,5.497787,6.283185"
    N: int = 9
    bw_phases: str = DEFAULT_BW_PHASES
    other_theta_file: str | None = None
    output_tol: float = 0.02
    measured_output: str | None = None
    enable_line_search: bool = False
    line_search_min_alpha: float = 0.05
    line_search_shrink: float = 0.5


DIRECT_RUN_CONFIG = DirectRunConfig()


def parse_csv_list(text, item_type=str):
    if text is None or str(text).strip() == "":
        return []
    return [item_type(item.strip()) for item in str(text).split(",") if item.strip()]


def elapsed_time_text():
    elapsed = time.perf_counter() - RUN_START_TIME
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((elapsed - int(elapsed)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def run_log(message):
    print(f"[elapsed {elapsed_time_text()}] {message}")


def heater_labels_for_mzis(mzi_ids):
    return [f"{mzi}{arm}" for mzi in mzi_ids for arm in ("u", "d")]


def theta_labels_for_mzis(mzi_ids):
    return [f"theta{mzi}{arm}" for mzi in mzi_ids for arm in ("u", "d")]


def normalize_theta_label(label):
    text = str(label).strip()
    lower = text.lower()
    if lower.startswith("theta"):
        return "theta" + lower[5:]
    if lower.startswith("phi_u"):
        return "theta" + lower[5:] + "u"
    if lower.startswith("phi_d"):
        return "theta" + lower[5:] + "d"
    if lower.startswith("thetau"):
        return "theta" + lower[6:] + "u"
    if lower.startswith("thetad"):
        return "theta" + lower[6:] + "d"
    return lower


def normalize_heater_label(label):
    text = str(label).strip()
    return text[1:] if text.lower().startswith("p") else text.lower()


def parse_heater_label(label):
    text = normalize_heater_label(label)
    if len(text) < 2 or text[-1] not in {"u", "d"}:
        raise ValueError(f"Invalid heater label {label!r}; expected like 5u or 5d.")
    return int(text[:-1]), text[-1]


def wrap_to_pi(angle):
    values = np.asarray(angle, dtype=float)
    wrapped = (values + np.pi) % (2 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped


def parse_probe_map(text, mzi_ids):
    probe_map = {str(int(mzi_id)): "u" for mzi_id in mzi_ids}
    if text is None or str(text).strip() == "":
        return probe_map
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid probe_map item {item!r}; expected like 5:u.")
        mzi_text, arm_text = item.split(":", 1)
        arm = arm_text.strip().lower()
        if arm not in {"u", "d"}:
            raise ValueError(f"Unsupported probe arm {arm_text!r}; expected u or d.")
        probe_map[str(int(mzi_text))] = arm
    missing = [str(int(mzi_id)) for mzi_id in mzi_ids if str(int(mzi_id)) not in probe_map]
    if missing:
        raise ValueError(f"probe_map missing MZI ids: {missing}")
    return probe_map


def sine_model(x, A, w, phi, b):
    return A * np.sin(w * x + phi) + b


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_sine_curve(x, y, fix_w=None, init_params=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        raise ValueError("Need at least four valid points for sine fitting.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    span = float(np.max(y) - np.min(y))
    A0 = max(0.5 * span, 1e-9)
    b0 = float(np.mean(y))
    w0 = 1.0 if float(np.ptp(x)) == 0.0 else float(2 * np.pi / max(np.ptp(x), 1e-9))
    phi0 = 0.0
    if init_params:
        A0 = max(abs(float(init_params.get("A", A0))), 1e-9)
        w0 = max(abs(float(init_params.get("w", w0))), 1e-9)
        phi0 = float(init_params.get("phi", init_params.get("phase", phi0)))
        b0 = float(init_params.get("b", b0))

    if fix_w is None:
        popt, _ = curve_fit(
            sine_model,
            x,
            y,
            p0=[A0, w0, phi0, b0],
            bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        A, w, phi, b = popt
    else:
        w = float(fix_w)
        popt, _ = curve_fit(
            lambda x_value, A, phi, b: sine_model(x_value, A, w, phi, b),
            x,
            y,
            p0=[A0, phi0, b0],
            bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        A, phi, b = popt

    fitted = sine_model(x, A, w, phi, b)
    return {
        "A": float(A),
        "w": float(w),
        "phi": wrap_to_pi(phi),
        "b": float(b),
        "rmse_uW": rmse(y, fitted),
        "x": x,
        "y": y,
        "fitted": fitted,
    }


def load_scan_file(path):
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    if "dp" in df.columns:
        x_col = "dp"
    elif "probe_axis_power_w" in df.columns:
        x_col = "probe_axis_power_w"
    elif "measured_power_w" in df.columns:
        x_col = "measured_power_w"
    elif "target_power_w" in df.columns:
        x_col = "target_power_w"
    else:
        raise ValueError(f"{path} must contain dp, probe_axis_power_w, measured_power_w, or target_power_w.")

    if "pow(uW)" in df.columns:
        y_col = "pow(uW)"
    elif "optical_power_uW" in df.columns:
        y_col = "optical_power_uW"
    else:
        raise ValueError(f"{path} must contain pow(uW) or optical_power_uW.")

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    return {"path": path, "df": df, "x": x, "y": y, "x_col": x_col, "y_col": y_col}


def fit_delta_probe_phase(scan_file, init_w=None, init_params=None):
    scan = load_scan_file(scan_file)
    fit = fit_sine_curve(scan["x"], scan["y"], fix_w=init_w, init_params=init_params)
    fit["scan_file"] = str(scan_file)
    return fit


def fit_sigma_inter_phase(scan_file, fix_w=None, init_params=None):
    scan = load_scan_file(scan_file)
    fit = fit_sine_curve(scan["x"], scan["y"], fix_w=fix_w, init_params=init_params)
    fit["scan_file"] = str(scan_file)
    return fit


def load_matrix_csv(path, expected_rows, expected_cols):
    path = Path(path)
    matrix = pd.read_csv(path, index_col=0)
    if matrix.shape != (len(expected_rows), len(expected_cols)):
        raise ValueError(
            f"{path} must be {len(expected_rows)}x{len(expected_cols)}, got {matrix.shape}."
        )

    normalized_rows = [normalize_theta_label(idx) for idx in matrix.index]
    normalized_cols = [normalize_heater_label(col) for col in matrix.columns]
    expected_rows_norm = [normalize_theta_label(row) for row in expected_rows]
    expected_cols_norm = [normalize_heater_label(col) for col in expected_cols]

    missing_rows = [row for row in expected_rows_norm if row not in normalized_rows]
    missing_cols = [col for col in expected_cols_norm if col not in normalized_cols]
    if missing_rows or missing_cols:
        raise ValueError(f"{path} missing rows={missing_rows}, cols={missing_cols}.")

    matrix.index = normalized_rows
    matrix.columns = normalized_cols
    matrix = matrix.loc[expected_rows_norm, expected_cols_norm]
    if matrix.isna().any().any():
        raise ValueError(f"{path} contains NaN values.")
    return matrix.astype(float)


def load_current_power_csv(path, heater_labels):
    df = pd.read_csv(path)
    if not {"heater", "power_w"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: heater,power_w")
    values = {normalize_heater_label(row.heater): float(row.power_w) for row in df.itertuples()}
    missing = [label for label in heater_labels if label not in values]
    if missing:
        raise ValueError(f"{path} missing heater powers: {missing}")
    return np.array([values[label] for label in heater_labels], dtype=float)


def load_current_theta_csv(path, theta_labels):
    df = pd.read_csv(path)
    if not {"theta", "value_rad"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: theta,value_rad")
    values = {normalize_theta_label(row.theta): float(row.value_rad) for row in df.itertuples()}
    missing = [label for label in theta_labels if label not in values]
    if missing:
        raise ValueError(f"{path} missing theta values: {missing}")
    return np.array([values[label] for label in theta_labels], dtype=float)


def save_zero_current_theta_csv(path, theta_labels):
    path = Path(path)
    theta = np.zeros(len(theta_labels), dtype=float)
    pd.DataFrame({"theta": theta_labels, "value_rad": theta}).to_csv(path, index=False)
    print(
        f"[ColumnPhaseOptimize] created {path} with zero phases. "
        "This assumes the current hardware state is the theta=0 reference state."
    )
    return theta


def load_initial_theta(args, theta_labels, run_dir=None):
    current_theta_path = Path(args.current_theta) if args.current_theta else None
    if current_theta_path is not None and current_theta_path.exists():
        return load_current_theta_csv(current_theta_path, theta_labels)

    ref_ready = bool(args.reference_dir) and (Path(args.reference_dir) / "theta_reference.json").exists()
    cur_ready = bool(args.current_dir) and Path(args.current_dir).exists()
    if ref_ready and cur_ready:
        out_csv = Path(run_dir) / "current_theta_second_column.csv" if run_dir else "current_theta_second_column.csv"
        measure_current_theta(args.reference_dir, args.current_dir, out_csv=out_csv)
        return load_current_theta_csv(out_csv, theta_labels)

    if getattr(args, "assume_reference_zero", False):
        out_csv = current_theta_path if current_theta_path is not None else Path("current_theta_second_column.csv")
        return save_zero_current_theta_csv(out_csv, theta_labels)

    raise FileNotFoundError(
        "Cannot get current theta. Provide current_theta CSV, provide reference_dir/current_dir scans, "
        "or explicitly set assume_reference_zero=True if the current state is the theta=0 reference."
    )


def load_theta_reference(reference_dir):
    path = Path(reference_dir) / "theta_reference.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run init_reference first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_theta_reference(reference_dir, data):
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    path = reference_dir / "theta_reference.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_sigma_coeff(reference_dir, sigma_sign_path=None, mzi_ids=DEFAULT_MZI_IDS):
    coeff = {}
    warnings = []
    sign_data = {}
    if sigma_sign_path and Path(sigma_sign_path).exists():
        with Path(sigma_sign_path).open("r", encoding="utf-8") as f:
            sign_data = json.load(f)
    else:
        default_sign = Path(reference_dir) / "sign_check" / "sigma_sign.json"
        if default_sign.exists():
            with default_sign.open("r", encoding="utf-8") as f:
                sign_data = json.load(f)

    for mzi_id in mzi_ids:
        key = str(int(mzi_id))
        entry = sign_data.get(key, {})
        if "c" in entry:
            coeff[key] = float(entry["c"])
        elif "s" in entry:
            coeff[key] = 2.0 * float(entry["s"])
        else:
            coeff[key] = 2.0
            warnings.append(f"MZI {key}: sigma coeff missing; default c=2.0")
    return coeff, warnings


def load_j_theta(path, theta_labels, heater_labels):
    return load_matrix_csv(path, theta_labels, heater_labels)


def load_mzi_table(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_second_column_heater_info(mzi_table, mzi_ids=DEFAULT_MZI_IDS):
    labels = []
    ports = []
    resistances = []
    for mzi in mzi_ids:
        entry = mzi_table.get(str(int(mzi)))
        if entry is None:
            raise KeyError(f"MZI {mzi} not found in MZI table.")
        entry_ports = entry.get("ports", [])
        heater_r = entry.get("heater_R", [])
        if len(entry_ports) != 2 or len(heater_r) != 2:
            raise ValueError(f"MZI {mzi} must have exactly two ports and heater_R values.")
        for arm, idx in (("u", 0), ("d", 1)):
            labels.append(f"{mzi}{arm}")
            ports.append(int(entry_ports[idx]))
            r_value = float(heater_r[idx])
            if not np.isfinite(r_value) or r_value <= 0:
                raise ValueError(f"MZI {mzi}{arm} has invalid heater_R={r_value}.")
            resistances.append(r_value)
    return {
        "heater_labels": labels,
        "ports": np.array(ports, dtype=int),
        "resistances": np.array(resistances, dtype=float),
    }


def get_mzi_arm_info(mzi_table, mzi_id, arm):
    mzi_id = int(mzi_id)
    arm = str(arm).lower()
    if arm not in {"u", "d"}:
        raise ValueError(f"arm must be u or d, got {arm!r}.")
    entry = mzi_table.get(str(mzi_id))
    if entry is None:
        raise KeyError(f"MZI {mzi_id} not found in MZI table.")
    ports = entry.get("ports", [])
    heater_r = entry.get("heater_R", [])
    ppi = entry.get("Ppi", [])
    idx = 0 if arm == "u" else 1
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


def get_mzi_state_voltage(entry, state):
    state = str(state).upper()
    if state == "B":
        candidates = [
            entry.get("dtheta_Bar", []),
            entry.get("dtheta", []),
        ]
        fit_key = "bar_voltage_v"
        preferred_index = 0
    elif state == "C":
        candidates = [
            entry.get("dtheta_Cross", []),
            entry.get("dtheta", []),
        ]
        fit_key = "cross_voltage_v"
        preferred_index = 1
    else:
        raise ValueError(f"Unsupported MZI state {state!r}.")

    for values in candidates:
        if not isinstance(values, list) or not values:
            continue
        if len(values) > preferred_index:
            return float(values[preferred_index])
        if len(values) == 1:
            return float(values[0])

    fit_params = entry.get("fit_params", [])
    if isinstance(fit_params, list) and fit_params and fit_key in fit_params[0]:
        return float(fit_params[0][fit_key])
    raise ValueError(f"MZI entry missing {state} voltage.")


def get_second_column_bar_voltages(mzi_table, mzi_ids=DEFAULT_MZI_IDS):
    voltages = []
    labels = []
    for mzi in mzi_ids:
        key = str(int(mzi))
        entry = mzi_table.get(key)
        if entry is None:
            raise KeyError(f"MZI {mzi} not found in MZI table.")
        bar_values = entry.get("dtheta_Bar", entry.get("dtheta", []))
        if isinstance(bar_values, list) and len(bar_values) >= 2:
            upper_v, lower_v = float(bar_values[0]), float(bar_values[1])
        else:
            fit_params = entry.get("fit_params", [])
            if len(fit_params) < 2:
                raise ValueError(f"MZI {mzi} has no dtheta_Bar or fit_params bar voltages.")
            upper_v = float(fit_params[0].get("bar_voltage_v"))
            lower_v = float(fit_params[1].get("bar_voltage_v"))
        labels.extend([f"{mzi}u", f"{mzi}d"])
        voltages.extend([upper_v, lower_v])
    return labels, np.array(voltages, dtype=float)


def save_current_power_from_bar_state(path, mzi_table, heater_info, mzi_ids=DEFAULT_MZI_IDS):
    labels, bar_voltages = get_second_column_bar_voltages(mzi_table, mzi_ids)
    expected = list(heater_info["heater_labels"])
    normalized = [normalize_heater_label(label) for label in labels]
    if normalized != expected:
        raise ValueError(f"Bar voltage order mismatch: got {normalized}, expected {expected}.")

    powers = voltage_to_power(bar_voltages, heater_info["resistances"])
    df = pd.DataFrame(
        {
            "heater": expected,
            "power_w": powers,
            "bar_voltage_v": bar_voltages,
            "R_ohm": heater_info["resistances"],
            "source": "MZI_table Bar state",
        }
    )
    path = Path(path)
    df.to_csv(path, index=False)
    print(f"[ColumnPhaseOptimize] created {path} from MZI_table Bar voltages.")
    return powers


def power_to_voltage(power_w, resistance_ohm):
    power = np.asarray(power_w, dtype=float)
    resistance = np.asarray(resistance_ohm, dtype=float)
    if np.any(power < 0):
        raise ValueError("power_w must be non-negative.")
    return np.sqrt(power * resistance)


def voltage_to_power(voltage_v, resistance_ohm):
    voltage = np.asarray(voltage_v, dtype=float)
    resistance = np.asarray(resistance_ohm, dtype=float)
    return voltage**2 / resistance


def validate_voltage_vector(voltages, voltage_limit_v):
    voltages = np.asarray(voltages, dtype=float)
    if np.any(~np.isfinite(voltages)):
        raise ValueError("Voltage vector contains non-finite values.")
    if np.any(voltages < 0.0) or np.any(voltages > float(voltage_limit_v)):
        bad = np.where((voltages < 0.0) | (voltages > float(voltage_limit_v)))[0].tolist()
        raise ValueError(f"Voltage out of safe range [0, {voltage_limit_v}] at indices {bad}.")


def validate_working_data_voltages(working_data, voltage_limit_v):
    values = working_data.iloc[:, 0].to_numpy(dtype=float)
    validate_voltage_vector(values, voltage_limit_v)


def voltage_upload_summary(working_data, context_label):
    values = working_data.iloc[:, 0].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        min_v = max_v = np.nan
        min_port = max_port = -1
    else:
        min_idx = int(np.argmin(values))
        max_idx = int(np.argmax(values))
        min_v = float(values[min_idx])
        max_v = float(values[max_idx])
        min_port = min_idx + 1
        max_port = max_idx + 1
    run_log(
        f"[ColumnPhaseOptimize] upload voltage | test={context_label} | "
        f"min={min_v:.3f} V (port {min_port}) | max={max_v:.3f} V (port {max_port})"
    )


def upload_second_column_voltages(working_data, update_df, cu, write_port_voltage, mcv, voltage_limit_v, context_label="second-column update"):
    validate_voltage_vector(update_df["V_next_v"].to_numpy(dtype=float), voltage_limit_v)
    for row in update_df.itertuples():
        write_port_voltage(int(row.port), float(row.V_next_v), working_data)
    validate_working_data_voltages(working_data, voltage_limit_v)
    voltage_upload_summary(working_data, context_label)
    cu.upload_voltage(mcv, working_data)


def mzi_transfer(theta_u, theta_d):
    delta = (float(theta_u) - float(theta_d)) / 2.0
    sigma = (float(theta_u) + float(theta_d)) / 2.0
    local = np.array(
        [
            [np.sin(delta), np.cos(delta)],
            [np.cos(delta), -np.sin(delta)],
        ],
        dtype=complex,
    )
    return 1j * np.exp(1j * sigma) * local


class MZI:
    def __init__(self, theta1=None, theta2=None):
        self.theta1 = 0.0 if theta1 is None else float(theta1)
        self.theta2 = 0.0 if theta2 is None else float(theta2)

    def forward(self):
        return mzi_transfer(self.theta1, self.theta2)


def build_clements_matrix(N):
    try:
        import utils.AllDecompositionUtils as du
    except Exception as exc:
        raise ImportError("Failed to import utils.AllDecompositionUtils for Clements_matrix.") from exc
    return du.Clements_matrix(int(N))


def embed_2x2_local(A, i, j, N):
    result = np.eye(int(N), dtype=complex)
    result[int(i) - 1, int(i) - 1] = A[0, 0]
    result[int(i) - 1, int(j) - 1] = A[0, 1]
    result[int(j) - 1, int(i) - 1] = A[1, 0]
    result[int(j) - 1, int(j) - 1] = A[1, 1]
    return result


def net_T(mzi_net, mzis_param, bw_phases=None):
    mzi_net = np.asarray(mzi_net, dtype=int)
    mzis_param = np.asarray(mzis_param, dtype=float)
    N = mzi_net.shape[0] + 1
    T = np.eye(N, dtype=complex)
    bw_values = [] if bw_phases is None else list(np.asarray(bw_phases, dtype=float).reshape(-1))
    bw_index = 0
    for col in range(mzi_net.shape[1]):
        C = np.eye(N, dtype=complex)
        for row in range(mzi_net.shape[0]):
            mzi_id = int(mzi_net[row, col])
            if mzi_id > 0:
                theta_u, theta_d = mzis_param[mzi_id - 1]
                B = embed_2x2_local(mzi_transfer(theta_u, theta_d), row + 1, row + 2, N)
                C = C @ B
            elif mzi_id < 0:
                phase = bw_values[bw_index] if bw_index < len(bw_values) else 0.0
                C[N - 1, N - 1] *= np.exp(1j * phase)
                bw_index += 1
        T = C @ T
    return T


def build_full_theta_array(
    second_column_theta,
    N=9,
    mzi_ids=DEFAULT_MZI_IDS,
    default_theta=(np.pi, 0.0),
    optional_theta_file=None,
):
    count = int(N) * (int(N) - 1) // 2
    thetas = np.tile(np.asarray(default_theta, dtype=float), (count, 1))
    optional_path = Path(optional_theta_file) if optional_theta_file else None
    if optional_path and optional_path.exists():
        df = pd.read_csv(optional_path)
        required = {"mzi_id", "theta_u", "theta_d"}
        if not required.issubset(df.columns):
            raise ValueError(f"{optional_path} must contain columns: mzi_id,theta_u,theta_d")
        for row in df.itertuples():
            mzi_id = int(row.mzi_id)
            if not 1 <= mzi_id <= count:
                raise ValueError(f"mzi_id {mzi_id} out of range 1..{count}")
            thetas[mzi_id - 1] = [float(row.theta_u), float(row.theta_d)]

    second = np.asarray(second_column_theta, dtype=float).reshape(len(mzi_ids), 2)
    for idx, mzi_id in enumerate(mzi_ids):
        thetas[int(mzi_id) - 1] = second[idx]
    return thetas


def compute_theory_transfer(second_column_theta, N=9, mzi_ids=DEFAULT_MZI_IDS, bw_phases=None, other_theta_file=None):
    cm = build_clements_matrix(N)
    bw_values = parse_csv_list(bw_phases, float) if isinstance(bw_phases, str) else bw_phases
    if bw_values is not None and len(bw_values) > 0:
        negative_cols = [2 + 2 * idx for idx in range(len(bw_values))]
        for col in negative_cols:
            if col < cm.shape[1]:
                cm[-1, col] = -1
    full_theta = build_full_theta_array(
        second_column_theta,
        N=N,
        mzi_ids=mzi_ids,
        default_theta=(np.pi, 0.0),
        optional_theta_file=other_theta_file,
    )
    T = net_T(cm, full_theta, bw_phases=bw_values)
    return T[: int(N) - 1, : int(N) - 1]


def compute_theory_power_matrix(second_column_theta, N=9, mzi_ids=DEFAULT_MZI_IDS, bw_phases=None, other_theta_file=None):
    T = compute_theory_transfer(
        second_column_theta,
        N=N,
        mzi_ids=mzi_ids,
        bw_phases=bw_phases,
        other_theta_file=other_theta_file,
    )
    return np.abs(T) ** 2


def compute_theory_output_from_second_column_theta(
    second_column_theta,
    N=9,
    mzi_ids=DEFAULT_MZI_IDS,
    bw_phases=None,
    other_theta_file=None,
):
    T = compute_theory_transfer(
        second_column_theta,
        N=N,
        mzi_ids=mzi_ids,
        bw_phases=bw_phases,
        other_theta_file=other_theta_file,
    )
    return T, np.abs(T) ** 2


def save_matrix_csv(matrix, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.asarray(matrix)).to_csv(path)


def load_measured_output_matrix(path):
    df = pd.read_csv(path, header=None)
    values = df.apply(pd.to_numeric, errors="coerce")
    if values.shape == (9, 9):
        values = values.iloc[1:, 1:]
    elif values.shape[0] == 8 and values.shape[1] == 9:
        values = values.iloc[:, 1:]
    elif values.shape[0] == 9 and values.shape[1] == 8:
        values = values.iloc[1:, :]
    if values.shape != (8, 8):
        df2 = pd.read_csv(path, index_col=0)
        values = df2.apply(pd.to_numeric, errors="coerce")
    if values.shape != (8, 8):
        raise ValueError(f"{path} must contain an 8x8 output matrix, got {values.shape}.")
    matrix = values.to_numpy(dtype=float)
    if np.isnan(matrix).any():
        raise ValueError(f"{path} contains non-numeric or NaN entries.")
    return matrix


def compute_output_error_metrics(P_current, P_target):
    error = np.asarray(P_current, dtype=float) - np.asarray(P_target, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def save_output_error_outputs(out_dir, prefix, P_current, P_target):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error = np.asarray(P_current) - np.asarray(P_target)
    save_matrix_csv(P_current, out_dir / f"{prefix}_current_model_power_matrix.csv")
    save_matrix_csv(P_target, out_dir / f"{prefix}_target_power_matrix.csv")
    save_matrix_csv(error, out_dir / f"{prefix}_output_error_matrix.csv")
    return compute_output_error_metrics(P_current, P_target)


def plot_output_compare(P_target, P_current, out_path):
    diff = np.abs(np.asarray(P_current) - np.asarray(P_target))
    vmax = max(float(np.max(P_target)), float(np.max(P_current)), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, matrix, title in zip(
        axes,
        [P_target, P_current, diff],
        ["P_target", "P_current_model", "abs error"],
    ):
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax if title != "abs error" else None)
        ax.set_title(title)
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_output_rmse_history(log_df, out_path):
    if "model_output_rmse" not in log_df.columns:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["iter"], log_df["model_output_rmse"], "o-", label="model output RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def compute_iteration_errors(theta_current, theta_target, P_current_model, P_target):
    theta_error = compute_phase_error(theta_target, theta_current)
    output_metrics = compute_output_error_metrics(P_current_model, P_target)
    return theta_error, output_metrics


def build_target_theta(mzi_ids=DEFAULT_MZI_IDS):
    target = []
    for mzi in mzi_ids:
        target.extend([np.pi if int(mzi) == 8 else np.pi / 3, 0.0])
    return np.array(target, dtype=float)


def init_theta_reference(reference_dir, probe_map, sigma_sign_path=None, mzi_ids=DEFAULT_MZI_IDS, dry_run=False):
    reference_dir = Path(reference_dir)
    delta_dir = reference_dir / "delta_baseline"
    sigma_dir = reference_dir / "sigma_baseline"
    missing = []
    for mzi_id in mzi_ids:
        if not (delta_dir / f"obs{int(mzi_id)}_probe.txt").exists():
            missing.append(str(delta_dir / f"obs{int(mzi_id)}_probe.txt"))
        if not (sigma_dir / f"obs{int(mzi_id)}_inter_scan.txt").exists():
            missing.append(str(sigma_dir / f"obs{int(mzi_id)}_inter_scan.txt"))
    if dry_run:
        print("init_reference dry-run:")
        if missing:
            print("Missing reference scans. Prepare these files first:")
            for item in missing:
                print(f"  {item}")
        else:
            print("All required baseline files exist.")
        return None
    if missing:
        raise FileNotFoundError(
            "Missing reference scans. Create/copy baseline scans first:\n" + "\n".join(missing)
        )

    sigma_coeff, coeff_warnings = load_sigma_coeff(reference_dir, sigma_sign_path, mzi_ids)
    delta_eta_ref = {}
    delta_w_ref = {}
    delta_fit = {}
    sigma_beta_ref = {}
    sigma_w_ref = {}
    sigma_fit = {}
    warnings = list(coeff_warnings)

    for mzi_id in mzi_ids:
        key = str(int(mzi_id))
        delta_fit_i = fit_delta_probe_phase(delta_dir / f"obs{key}_probe.txt")
        sigma_fit_i = fit_sigma_inter_phase(sigma_dir / f"obs{key}_inter_scan.txt")
        delta_eta_ref[key] = float(delta_fit_i["phi"])
        delta_w_ref[key] = float(delta_fit_i["w"])
        delta_fit[key] = {
            "A": float(delta_fit_i["A"]),
            "w": float(delta_fit_i["w"]),
            "eta": float(delta_fit_i["phi"]),
            "b": float(delta_fit_i["b"]),
            "rmse_uW": float(delta_fit_i["rmse_uW"]),
        }
        sigma_beta_ref[key] = float(sigma_fit_i["phi"])
        sigma_w_ref[key] = float(sigma_fit_i["w"])
        sigma_fit[key] = {
            "A": float(sigma_fit_i["A"]),
            "w": float(sigma_fit_i["w"]),
            "beta": float(sigma_fit_i["phi"]),
            "b": float(sigma_fit_i["b"]),
            "rmse_uW": float(sigma_fit_i["rmse_uW"]),
        }

    data = {
        "mzi_ids": [int(x) for x in mzi_ids],
        "probe_map": {str(k): v for k, v in probe_map.items()},
        "delta_eta_ref": delta_eta_ref,
        "delta_w_ref": delta_w_ref,
        "delta_fit": delta_fit,
        "sigma_beta_ref": sigma_beta_ref,
        "sigma_w_ref": sigma_w_ref,
        "sigma_fit": sigma_fit,
        "sigma_coeff": sigma_coeff,
        "theta_order": theta_labels_for_mzis(mzi_ids),
        "warnings": warnings,
    }
    path = save_theta_reference(reference_dir, data)
    print(f"Saved theta reference to {path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    return data


def ensure_theta_reference(reference_dir, args, mzi_ids=DEFAULT_MZI_IDS):
    reference_dir = Path(reference_dir)
    reference_json = reference_dir / "theta_reference.json"
    if reference_json.exists():
        return reference_json

    delta_ref_dir = reference_dir / "delta_baseline"
    sigma_ref_dir = reference_dir / "sigma_baseline"
    delta_ref_dir.mkdir(parents=True, exist_ok=True)
    sigma_ref_dir.mkdir(parents=True, exist_ok=True)

    delta_source_dir = Path(getattr(args, "delta_reference_source_dir", "jacobian_measurements/baseline"))
    sigma_source_dir = Path(getattr(args, "sigma_reference_source_dir", "Scandata/J_sigma/baseline"))
    missing = []
    for mzi_id in mzi_ids:
        key = str(int(mzi_id))
        delta_src = delta_source_dir / f"obs{key}_probe.txt"
        delta_dst = delta_ref_dir / f"obs{key}_probe.txt"
        sigma_src = sigma_source_dir / f"obs{key}_inter_scan.txt"
        sigma_dst = sigma_ref_dir / f"obs{key}_inter_scan.txt"
        if not delta_dst.exists():
            if delta_src.exists():
                shutil.copy2(delta_src, delta_dst)
            else:
                missing.append(str(delta_src))
        if not sigma_dst.exists():
            if sigma_src.exists():
                shutil.copy2(sigma_src, sigma_dst)
            else:
                missing.append(str(sigma_src))

    if missing:
        raise FileNotFoundError(
            "Cannot initialize theta reference. Missing baseline scans:\n" + "\n".join(missing)
        )

    probe_map = parse_probe_map(getattr(args, "probe_map", "5:u,6:u,7:u,8:u"), mzi_ids)
    init_theta_reference(
        reference_dir=reference_dir,
        probe_map=probe_map,
        sigma_sign_path=getattr(args, "sigma_sign_path", "Scandata/J_sigma/sign_check/sigma_sign.json"),
        mzi_ids=mzi_ids,
        dry_run=False,
    )
    return reference_json


def combine_delta_sigma_to_theta(delta_dict, sigma_dict, mzi_ids=DEFAULT_MZI_IDS):
    theta_values = []
    for mzi_id in mzi_ids:
        key = str(int(mzi_id))
        delta = float(delta_dict[key])
        sigma = float(sigma_dict[key])
        theta_values.extend([(sigma + delta) / 2.0, (sigma - delta) / 2.0])
    return np.array(theta_values, dtype=float)


def save_current_theta_outputs(out_csv, theta_labels, theta_values, details, summary):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True) if out_csv.parent != Path("") else None
    pd.DataFrame({"theta": theta_labels, "value_rad": theta_values}).to_csv(out_csv, index=False)
    detail_path = out_csv.with_name("current_theta_details.csv")
    summary_path = out_csv.with_name("current_theta_summary.json")
    pd.DataFrame(details).to_csv(detail_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_csv, detail_path, summary_path


def measure_current_theta_from_files(reference_dir, current_dir, out_csv="current_theta_second_column.csv"):
    reference = load_theta_reference(reference_dir)
    mzi_ids = [int(x) for x in reference["mzi_ids"]]
    probe_map = {str(k): v for k, v in reference.get("probe_map", {}).items()}
    theta_labels = reference.get("theta_order", theta_labels_for_mzis(mzi_ids))
    current_dir = Path(current_dir)
    delta_dir = current_dir / "delta_current"
    sigma_dir = current_dir / "sigma_current"
    delta_dict = {}
    sigma_dict = {}
    details = []
    warnings = []

    for mzi_id in mzi_ids:
        key = str(int(mzi_id))
        probe_arm = probe_map.get(key, "u")
        row_warning = []
        delta_delta = np.nan
        delta_eta = np.nan
        eta_cur = np.nan
        delta_rmse = np.nan
        beta_cur = np.nan
        delta_beta = np.nan
        sigma_rmse = np.nan
        delta_sigma = np.nan

        try:
            delta_ref = reference["delta_fit"][key]
            delta_fit = fit_delta_probe_phase(
                delta_dir / f"obs{key}_probe.txt",
                init_w=float(reference["delta_w_ref"][key]),
                init_params={"A": delta_ref["A"], "phi": delta_ref["eta"], "b": delta_ref["b"]},
            )
            eta_cur = float(delta_fit["phi"])
            delta_rmse = float(delta_fit["rmse_uW"])
            delta_eta = wrap_to_pi(eta_cur - float(reference["delta_eta_ref"][key]))
            delta_delta = delta_eta if probe_arm == "u" else -delta_eta
            if abs(delta_eta) > np.pi / 2:
                row_warning.append("Delta phase jump may be too large")
            if delta_fit["A"] < 0.3 * float(delta_ref["A"]):
                row_warning.append("Delta low visibility")
        except Exception as exc:
            row_warning.append(f"Delta fit failed: {exc}")

        try:
            sigma_ref = reference["sigma_fit"][key]
            sigma_fit = fit_sigma_inter_phase(
                sigma_dir / f"obs{key}_inter_scan.txt",
                fix_w=float(reference["sigma_w_ref"][key]),
                init_params={"A": sigma_ref["A"], "phi": sigma_ref["beta"], "b": sigma_ref["b"]},
            )
            beta_cur = float(sigma_fit["phi"])
            sigma_rmse = float(sigma_fit["rmse_uW"])
            delta_beta = wrap_to_pi(beta_cur - float(reference["sigma_beta_ref"][key]))
            coeff = float(reference.get("sigma_coeff", {}).get(key, 2.0))
            if key not in reference.get("sigma_coeff", {}):
                row_warning.append("sigma_coeff missing; default c=2")
            delta_sigma = coeff * delta_beta
            if abs(delta_beta) > np.pi / 2:
                row_warning.append("Sigma phase jump may be too large")
            if sigma_fit["A"] < 0.3 * float(sigma_ref["A"]):
                row_warning.append("Sigma low visibility")
        except Exception as exc:
            coeff = float(reference.get("sigma_coeff", {}).get(key, 2.0))
            row_warning.append(f"Sigma fit failed: {exc}")

        delta_dict[key] = delta_delta
        sigma_dict[key] = delta_sigma
        theta_u = (delta_sigma + delta_delta) / 2.0 if np.isfinite(delta_sigma) and np.isfinite(delta_delta) else np.nan
        theta_d = (delta_sigma - delta_delta) / 2.0 if np.isfinite(delta_sigma) and np.isfinite(delta_delta) else np.nan
        warning_text = "; ".join(row_warning)
        if warning_text:
            warnings.append(f"MZI {key}: {warning_text}")
        details.append(
            {
                "mzi_id": int(mzi_id),
                "probe_arm": probe_arm,
                "eta_ref": reference["delta_eta_ref"].get(key, np.nan),
                "eta_cur": eta_cur,
                "delta_eta": delta_eta,
                "delta_Delta": delta_delta,
                "beta_ref": reference["sigma_beta_ref"].get(key, np.nan),
                "beta_cur": beta_cur,
                "delta_beta": delta_beta,
                "sigma_coeff": coeff,
                "delta_Sigma": delta_sigma,
                "theta_u": theta_u,
                "theta_d": theta_d,
                "delta_fit_rmse_uW": delta_rmse,
                "sigma_fit_rmse_uW": sigma_rmse,
                "warning": warning_text,
            }
        )

    theta_values = combine_delta_sigma_to_theta(delta_dict, sigma_dict, mzi_ids)
    summary = {
        "max_abs_theta": float(np.nanmax(np.abs(theta_values))) if np.isfinite(theta_values).any() else np.nan,
        "mzi_ids": mzi_ids,
        "reference_dir": str(reference_dir),
        "current_dir": str(current_dir),
        "warnings": warnings,
    }
    save_current_theta_outputs(out_csv, theta_labels, theta_values, details, summary)
    print(f"Saved current theta to {out_csv}")
    return theta_values, details, summary


def compute_phase_error(theta_target, theta_current):
    return wrap_to_pi(np.asarray(theta_target, dtype=float) - np.asarray(theta_current, dtype=float))


def regularized_solve(J, error, lambda_reg, weights=None):
    J = np.asarray(J, dtype=float)
    error = np.asarray(error, dtype=float)
    if weights is None:
        W = np.eye(J.shape[0])
    else:
        W = np.diag(np.asarray(weights, dtype=float))
    lhs = J.T @ W @ J + float(lambda_reg) * np.eye(J.shape[1])
    rhs = J.T @ W @ error
    return np.linalg.solve(lhs, rhs)


def apply_step_limits(delta_P, alpha, step_limit_w):
    raw_step = float(alpha) * np.asarray(delta_P, dtype=float)
    if step_limit_w is None or step_limit_w <= 0:
        return raw_step, 1.0
    max_abs = float(np.max(np.abs(raw_step)))
    if max_abs <= step_limit_w:
        return raw_step, 1.0
    scale = float(step_limit_w) / max_abs
    return raw_step * scale, scale


def apply_power_limits(P_current, delta_P_applied, power_limit_w):
    P_current = np.asarray(P_current, dtype=float)
    proposed = P_current + np.asarray(delta_P_applied, dtype=float)
    clipped = np.clip(proposed, 0.0, float(power_limit_w))
    return clipped, clipped - P_current, np.abs(clipped - proposed) > 1e-15


def branch_targets(theta_target, branch_candidates):
    candidates = list(branch_candidates)
    grids = np.meshgrid(*([candidates] * len(theta_target)), indexing="ij")
    shifts = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    for shift in shifts:
        yield theta_target + 2 * np.pi * shift


def select_branch_update(J, theta_current, theta_target, P_current, args):
    warnings = []
    base_error = compute_phase_error(theta_target, theta_current)
    best = None
    branch_candidates = parse_csv_list(args.branch_candidates, int)
    targets = [theta_target]
    if args.enable_branch_search:
        targets = list(branch_targets(theta_target, branch_candidates))

    for candidate_target in targets:
        error = candidate_target - theta_current if args.enable_branch_search else base_error
        delta_raw = regularized_solve(J, error, args.lambda_reg)
        step_limited, step_scale = apply_step_limits(delta_raw, args.alpha, args.step_limit_w)
        P_next, delta_applied, clipped = apply_power_limits(P_current, step_limited, args.power_limit_w)
        violation = float(np.sum(np.maximum(0.0, P_current + step_limited - args.power_limit_w)))
        violation += float(np.sum(np.maximum(0.0, -(P_current + step_limited))))
        norm_step = float(np.linalg.norm(delta_applied))
        score = (np.any(clipped), violation, norm_step)
        if best is None or score < best["score"]:
            best = {
                "target": candidate_target,
                "error": error,
                "delta_raw": delta_raw,
                "delta_applied": delta_applied,
                "P_next": P_next,
                "clipped": clipped,
                "step_scale": step_scale,
                "score": score,
            }

    if args.enable_branch_search and best is not None and np.any(best["clipped"]):
        warnings.append("branch search could not find a fully in-range update; selected least-violating branch")
    return best, warnings


def maybe_line_search_update(J, P_current, theta_current, theta_target, result, args, P_target=None, mzi_ids=None):
    if not getattr(args, "enable_line_search", False):
        result["alpha_used"] = float(args.alpha)
        return result, []

    warnings = []
    mzi_ids = DEFAULT_MZI_IDS if mzi_ids is None else mzi_ids
    if P_target is None:
        P_target = compute_theory_power_matrix(
            theta_target,
            N=getattr(args, "N", 9),
            mzi_ids=mzi_ids,
            bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
            other_theta_file=getattr(args, "other_theta_file", None),
        )

    current_theta_error = compute_phase_error(theta_target, theta_current)
    current_theta_norm = float(np.linalg.norm(current_theta_error))
    current_model = compute_theory_power_matrix(
        theta_current,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )
    current_rmse = compute_output_error_metrics(current_model, P_target)["rmse"]

    alpha_try = float(args.alpha)
    best = None
    min_alpha = float(getattr(args, "line_search_min_alpha", 0.05))
    shrink = float(getattr(args, "line_search_shrink", 0.5))
    while alpha_try >= min_alpha:
        step_limited, step_scale = apply_step_limits(result["delta_raw"], alpha_try, args.step_limit_w)
        P_next, delta_applied, clipped = apply_power_limits(P_current, step_limited, args.power_limit_w)
        theta_pred = theta_current + np.asarray(J, dtype=float) @ delta_applied
        pred_theta_norm = float(np.linalg.norm(compute_phase_error(theta_target, theta_pred)))
        pred_model = compute_theory_power_matrix(
            theta_pred,
            N=getattr(args, "N", 9),
            mzi_ids=mzi_ids,
            bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
            other_theta_file=getattr(args, "other_theta_file", None),
        )
        pred_rmse = compute_output_error_metrics(pred_model, P_target)["rmse"]
        candidate = {
            **result,
            "delta_applied": delta_applied,
            "P_next": P_next,
            "clipped": clipped,
            "step_scale": step_scale,
            "alpha_used": alpha_try,
            "pred_theta_norm": pred_theta_norm,
            "pred_output_rmse": pred_rmse,
        }
        best = candidate
        if pred_theta_norm <= current_theta_norm and pred_rmse <= current_rmse:
            return candidate, warnings
        alpha_try *= shrink

    warnings.append("line search reached minimum alpha without improving both theta norm and model output RMSE")
    if best is not None:
        return best, warnings
    result["alpha_used"] = float(args.alpha)
    return result, warnings


def plan_update(J, P_current, theta_current, theta_target, args, P_target=None, mzi_ids=None):
    result, warnings = select_branch_update(J, theta_current, theta_target, P_current, args)
    result, line_warnings = maybe_line_search_update(
        J,
        P_current,
        theta_current,
        theta_target,
        result,
        args,
        P_target=P_target,
        mzi_ids=mzi_ids,
    )
    warnings.extend(line_warnings)
    cond = float(np.linalg.cond(J))
    if cond > 1e6:
        warnings.append(f"J_theta condition number is large: {cond:.3e}")
    if np.linalg.norm(result["delta_raw"]) > 0.02:
        warnings.append("large delta_P; consider increasing lambda_reg or decreasing alpha")
    result["condition_number"] = cond
    result["warnings"] = warnings
    result.setdefault("alpha_used", float(args.alpha))
    return result


def save_plan_outputs(
    out_dir,
    heater_labels,
    theta_labels,
    heater_info,
    P_current,
    theta_current,
    plan,
    args,
    P_target=None,
    P_current_model=None,
    output_metrics=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voltages = power_to_voltage(plan["P_next"], heater_info["resistances"])
    voltage_clipped = voltages > args.voltage_limit_v
    voltages_safe = np.minimum(voltages, args.voltage_limit_v)
    warnings = []
    if np.any(voltage_clipped):
        warnings.append("one or more voltages exceeded voltage_limit_v and were clipped in output")

    update_df = pd.DataFrame(
        {
            "heater": heater_labels,
            "P_current_w": P_current,
            "delta_P_raw_w": plan["delta_raw"],
            "delta_P_applied_w": plan["delta_applied"],
            "P_next_w": plan["P_next"],
            "R_ohm": heater_info["resistances"],
            "port": heater_info["ports"],
            "V_next_v": voltages_safe,
            "clipped": plan["clipped"] | voltage_clipped,
            "warning": ["voltage clipped" if flag else "" for flag in voltage_clipped],
        }
    )
    update_df.to_csv(out_dir / "predicted_update.csv", index=False)

    error = compute_phase_error(plan["target"], theta_current)
    phase_df = pd.DataFrame(
        {
            "theta_name": theta_labels,
            "theta_current_rad": theta_current,
            "theta_target_rad": plan["target"],
            "error_rad": error,
            "abs_error_rad": np.abs(error),
        }
    )
    phase_df.to_csv(out_dir / "phase_error.csv", index=False)

    summary = {
        "max_abs_error_rad": float(np.max(np.abs(error))),
        "norm_error": float(np.linalg.norm(error)),
        "alpha": args.alpha,
        "alpha_used": float(plan.get("alpha_used", args.alpha)),
        "lambda_reg": args.lambda_reg,
        "power_limit_w": args.power_limit_w,
        "step_limit_w": args.step_limit_w,
        "condition_number_J": plan["condition_number"],
        "used_branch_search": bool(args.enable_branch_search),
        "warnings": plan["warnings"] + warnings,
    }
    if P_target is not None and P_current_model is not None:
        metrics = output_metrics or save_output_error_outputs(out_dir, "model", P_current_model, P_target)
        save_matrix_csv(P_target, out_dir / "target_power_matrix.csv")
        save_matrix_csv(P_current_model, out_dir / "current_model_power_matrix.csv")
        save_matrix_csv(np.asarray(P_current_model) - np.asarray(P_target), out_dir / "model_output_error_matrix.csv")
        summary.update(
            {
                "model_output_rmse": metrics["rmse"],
                "model_output_mae": metrics["mae"],
                "model_output_max_abs": metrics["max_abs"],
            }
        )
    if getattr(args, "measured_output", None) and P_target is not None:
        measured = load_measured_output_matrix(args.measured_output)
        measured_metrics = compute_output_error_metrics(measured, P_target)
        save_matrix_csv(measured - np.asarray(P_target), out_dir / "measured_output_error_matrix.csv")
        summary.update(
            {
                "measured_output_rmse": measured_metrics["rmse"],
                "measured_output_mae": measured_metrics["mae"],
                "measured_output_max_abs": measured_metrics["max_abs"],
            }
        )
    with (out_dir / "plan_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Recommended voltages:")
    print(update_df[["heater", "port", "P_next_w", "V_next_v", "clipped"]].to_string(index=False))
    return update_df, phase_df, summary


def plot_error_history(log_df, out_path):
    plt.figure(figsize=(7, 4))
    max_col = "max_abs_theta_error_rad" if "max_abs_theta_error_rad" in log_df.columns else "max_abs_error_rad"
    norm_col = "norm_theta_error" if "norm_theta_error" in log_df.columns else "norm_error"
    plt.plot(log_df["iter"], log_df[max_col], "o-", label="max abs theta error")
    plt.plot(log_df["iter"], log_df[norm_col], "s-", label="theta error norm")
    plt.xlabel("Iteration")
    plt.ylabel("Error (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_power_history(power_df, out_path):
    plt.figure(figsize=(8, 4.5))
    for col in power_df.columns:
        if col != "iter":
            plt.plot(power_df["iter"], power_df[col], marker="o", label=col)
    plt.xlabel("Iteration")
    plt.ylabel("Power (W)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_theta_target_vs_current(theta_labels, theta_current, theta_target, out_path):
    x = np.arange(len(theta_labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.18, theta_current, width=0.36, label="current")
    plt.bar(x + 0.18, theta_target, width=0.36, label="target")
    plt.xticks(x, theta_labels, rotation=45)
    plt.ylabel("Theta (rad)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def load_inputs(args):
    mzi_ids = parse_csv_list(args.mzi_ids, int)
    heater_labels = heater_labels_for_mzis(mzi_ids)
    theta_labels = theta_labels_for_mzis(mzi_ids)
    J_df = load_matrix_csv(args.j_theta, theta_labels, heater_labels)
    mzi_table = load_mzi_table(args.mzi_table)
    heater_info = get_second_column_heater_info(mzi_table, mzi_ids)
    theta_target = build_target_theta(mzi_ids)
    return mzi_ids, heater_labels, theta_labels, J_df, heater_info, theta_target


def run_plan(args):
    mzi_ids, heater_labels, theta_labels, J_df, heater_info, theta_target = load_inputs(args)
    P_current = load_current_power_csv(args.current_power, heater_labels)
    theta_current = load_initial_theta(args, theta_labels)
    P_target = compute_theory_power_matrix(
        theta_target,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )
    P_current_model = compute_theory_power_matrix(
        theta_current,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )
    output_metrics = compute_output_error_metrics(P_current_model, P_target)
    plan = plan_update(
        J_df.to_numpy(dtype=float),
        P_current,
        theta_current,
        theta_target,
        args,
        P_target=P_target,
        mzi_ids=mzi_ids,
    )
    save_plan_outputs(
        args.out_dir,
        heater_labels,
        theta_labels,
        heater_info,
        P_current,
        theta_current,
        plan,
        args,
        P_target=P_target,
        P_current_model=P_current_model,
        output_metrics=output_metrics,
    )


def simulate(args):
    mzi_ids, heater_labels, theta_labels, J_df, heater_info, theta_target = load_inputs(args)
    J = J_df.to_numpy(dtype=float)
    P = load_current_power_csv(args.current_power, heater_labels)
    theta = load_initial_theta(args, theta_labels)
    P_target = compute_theory_power_matrix(
        theta_target,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    theta_rows = []
    power_rows = []

    for k in range(args.max_iter + 1):
        P_current_model = compute_theory_power_matrix(
            theta,
            N=getattr(args, "N", 9),
            mzi_ids=mzi_ids,
            bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
            other_theta_file=getattr(args, "other_theta_file", None),
        )
        output_metrics = compute_output_error_metrics(P_current_model, P_target)
        error = compute_phase_error(theta_target, theta)
        theta_converged = bool(np.max(np.abs(error)) < args.theta_tol)
        output_converged = bool(output_metrics["rmse"] < getattr(args, "output_tol", 0.02))
        log_rows.append(
            {
                "iter": k,
                "max_abs_theta_error_rad": float(np.max(np.abs(error))),
                "norm_theta_error": float(np.linalg.norm(error)),
                "model_output_rmse": output_metrics["rmse"],
                "model_output_mae": output_metrics["mae"],
                "model_output_max_abs": output_metrics["max_abs"],
                "alpha_used": args.alpha,
                "lambda_reg": args.lambda_reg,
                "step_scale": 1.0,
                "num_clipped": 0,
                "theta_converged": theta_converged,
                "output_converged": output_converged,
                "converged": theta_converged and output_converged,
                "warning": "",
            }
        )
        theta_rows.append({"iter": k, **dict(zip(theta_labels, theta))})
        power_rows.append({"iter": k, **dict(zip(heater_labels, P))})
        if log_rows[-1]["converged"] or k == args.max_iter:
            break

        plan = plan_update(J, P, theta, theta_target, args, P_target=P_target, mzi_ids=mzi_ids)
        P_next = plan["P_next"]
        delta_P = P_next - P
        theta = theta + J @ delta_P
        P = P_next
        log_rows[-1]["step_scale"] = plan["step_scale"]
        log_rows[-1]["num_clipped"] = int(np.sum(plan["clipped"]))
        log_rows[-1]["alpha_used"] = float(plan.get("alpha_used", args.alpha))
        log_rows[-1]["warning"] = "; ".join(plan["warnings"])

    log_df = pd.DataFrame(log_rows)
    theta_df = pd.DataFrame(theta_rows)
    power_df = pd.DataFrame(power_rows)
    log_df.to_csv(out_dir / "simulate_iteration_log.csv", index=False)
    theta_df.to_csv(out_dir / "simulate_theta_history.csv", index=False)
    power_df.to_csv(out_dir / "simulate_power_history.csv", index=False)
    plot_error_history(log_df, out_dir / "simulate_error_plot.png")
    plot_output_rmse_history(log_df, out_dir / "simulate_output_rmse_plot.png")
    plot_power_history(power_df, out_dir / "power_history.png")
    plot_theta_target_vs_current(theta_labels, theta, theta_target, out_dir / "final_theta_compare.png")
    final_model = compute_theory_power_matrix(
        theta,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )
    plot_output_compare(P_target, final_model, out_dir / "final_output_compare.png")


def measure_current_theta(reference_dir, current_dir, out_csv="current_theta_second_column.csv"):
    return measure_current_theta_from_files(reference_dir, current_dir, out_csv=out_csv)


def read_current_power_from_working_data(file_data, heater_info):
    voltages = []
    for port in heater_info["ports"]:
        voltages.append(float(file_data.iloc[int(port) - 1, 0]))
    return voltage_to_power(np.array(voltages), heater_info["resistances"])


def apply_power_vector_to_working_data(working_data, heater_info, powers_w, write_port_voltage):
    powers = np.asarray(powers_w, dtype=float)
    for port, resistance, power_w in zip(heater_info["ports"], heater_info["resistances"], powers):
        write_port_power(working_data, int(port), float(resistance), float(power_w), write_port_voltage)


def get_port_voltage(working_data, port):
    return float(working_data.iloc[int(port) - 1, 0])


def write_port_power(working_data, port, resistance, power_w, write_port_voltage):
    voltage = float(power_to_voltage(float(max(0.0, power_w)), float(resistance)))
    write_port_voltage(int(port), voltage, working_data)
    return voltage


def read_opm_power_uW(cu, opm, output_channel):
    values = cu.read_pow(opm)
    idx = int(output_channel) - 1
    if idx < 0 or idx >= len(values):
        raise IndexError(f"Output channel {output_channel} not available from OPM readout.")
    return float(values[idx]) * 1e6


def upload_working_data_checked(cu, mcv, working_data, voltage_limit_v, context_label="unspecified"):
    validate_working_data_voltages(working_data, voltage_limit_v)
    voltage_upload_summary(working_data, context_label)
    cu.upload_voltage(mcv, working_data)


def get_left_upper_bar_channel(mzi_id, N):
    cm = build_clements_matrix(N)
    rows, _ = np.where(cm == int(mzi_id))
    if rows.size == 0:
        raise ValueError(f"MZI {mzi_id} not found in Clements matrix.")
    return int(rows[0]) + 1


def set_single_input(input_channel, input_count, working_data, switch_IN):
    for channel in range(1, int(input_count) + 1):
        switch_IN(channel, "OFF", working_data)
    switch_IN(int(input_channel), "ON", working_data)


def build_probe_offsets(half_width_w, step_w):
    if float(half_width_w) <= 0.0 or float(step_w) <= 0.0:
        raise ValueError("probe_half_width_w and probe_step_w must be positive.")
    count = int(round((2.0 * float(half_width_w)) / float(step_w))) + 1
    return np.round(np.linspace(-float(half_width_w), float(half_width_w), count), 9)


def scan_delta_probe_current(
    observed_mzi,
    probe_arm,
    save_dir,
    base_working_data,
    hardware,
    mzi_table,
    args,
):
    import time
    from inter_calibration import switch_IN, write_port_voltage

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    info = get_mzi_arm_info(mzi_table, observed_mzi, probe_arm)
    input_channel = get_left_upper_bar_channel(observed_mzi, getattr(args, "N", 9))
    output_channel = input_channel
    baseline_v = get_port_voltage(base_working_data, info["port"])
    baseline_power = voltage_to_power(baseline_v, info["resistance"])
    rows = []

    for offset in build_probe_offsets(args.probe_half_width_w, args.probe_step_w):
        scan_data = base_working_data.copy(deep=True)
        target_power = float(baseline_power) + float(offset)
        warning = ""
        if target_power < 0.0:
            warning = f"target power clipped at 0 W from {target_power:.9f} W"
            target_power = 0.0
        voltage = write_port_power(scan_data, info["port"], info["resistance"], target_power, write_port_voltage)
        set_single_input(input_channel, int(args.N) - 1, scan_data, switch_IN)
        upload_working_data_checked(
            hardware["cu"],
            hardware["mcv"],
            scan_data,
            args.voltage_limit_v,
            context_label=f"delta current probe: MZI{observed_mzi}{probe_arm}, offset={float(offset):+.6f} W",
        )
        time.sleep(float(args.settle_time))
        optical_power = read_opm_power_uW(hardware["cu"], hardware["opm2"], output_channel)
        rows.append(
            {
                "mzi_id": int(observed_mzi),
                "arm_name": info["arm_name"],
                "arm": info["arm"],
                "port": int(info["port"]),
                "probe_axis_power_w": float(offset),
                "target_power_w": float(target_power),
                "measured_power_w": float(target_power),
                "voltage_v": float(voltage),
                "optical_power_uW": float(optical_power),
                "input_channel": int(input_channel),
                "output_channel": int(output_channel),
                "scan_type": "delta_current_probe",
                "warning": warning,
            }
        )

    out_path = save_dir / f"obs{int(observed_mzi)}_probe.txt"
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.12f")
    print(f"[ColumnPhaseOptimize] saved delta current scan {out_path}")
    return out_path


def fold_power_to_limit(power_w, period_w, power_limit_w):
    power = float(power_w)
    period = float(period_w)
    if period <= 0.0:
        raise ValueError("fold period must be positive.")
    count = 0
    while power > float(power_limit_w):
        power -= period
        count += 1
    return max(0.0, power), count


def scan_sigma_current(
    target,
    save_dir,
    base_working_data,
    hardware,
    mzi_table,
    args,
):
    import time
    import inter_calibration as ic

    from inter_calibration import switch_IN, write_port_voltage

    target = int(target)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    scan_data = base_working_data.copy(deep=True)
    path, input_idx, output_idx, state, bmzi = ic.find_Bmzi_path(target, int(args.N))
    build_bmzi_state_no_upload(path, input_idx, state, bmzi, scan_data, mzi_table, int(args.N), switch_IN, write_port_voltage)

    entry = mzi_table[str(target)]
    ports = [int(value) for value in entry.get("ports", [])[:2]]
    heater_r = [float(value) for value in entry.get("heater_R", [])[:2]]
    ppi = [float(value) for value in entry.get("Ppi", [])[:2]]
    if len(ports) != 2 or len(heater_r) != 2 or len(ppi) != 2:
        raise ValueError(f"MZI {target} requires two ports, heater_R, and Ppi for sigma scan.")

    base_v_upper = get_port_voltage(scan_data, ports[0])
    base_v_lower = get_port_voltage(scan_data, ports[1])
    p_upper_base = float(voltage_to_power(base_v_upper, heater_r[0]))
    p_lower_base = float(voltage_to_power(base_v_lower, heater_r[1]))
    period_upper = 2.0 * ppi[0]
    period_lower = 2.0 * ppi[1]
    phase_points = parse_csv_list(args.sigma_phase_points, float)
    rows = []

    for dp in phase_points:
        p_upper_unfolded = p_upper_base + float(dp) / np.pi * ppi[0]
        p_lower_unfolded = p_lower_base + float(dp) / np.pi * ppi[1]
        p_upper, upper_folds = fold_power_to_limit(p_upper_unfolded, period_upper, args.power_limit_w)
        p_lower, lower_folds = fold_power_to_limit(p_lower_unfolded, period_lower, args.power_limit_w)
        warning = ""
        if upper_folds or lower_folds:
            warning = (
                f"period fold applied: upper {p_upper_unfolded:.9f}->{p_upper:.9f} W "
                f"({upper_folds}), lower {p_lower_unfolded:.9f}->{p_lower:.9f} W ({lower_folds})"
            )
        v_upper = write_port_power(scan_data, ports[0], heater_r[0], p_upper, write_port_voltage)
        v_lower = write_port_power(scan_data, ports[1], heater_r[1], p_lower, write_port_voltage)
        upload_working_data_checked(
            hardware["cu"],
            hardware["mcv"],
            scan_data,
            args.voltage_limit_v,
            context_label=f"sigma current scan: MZI{target}, dp={float(dp):.6f} rad",
        )
        time.sleep(float(args.settle_time))
        optical_power = read_opm_power_uW(hardware["cu"], hardware["opm2"], int(output_idx) + 1)
        rows.append(
            {
                "target": target,
                "observed_mzi": target,
                "dp": float(dp),
                "pow(uW)": float(optical_power),
                "v_primary": float(v_upper),
                "v_secondary": float(v_lower),
                "p_primary": float(p_upper),
                "p_secondary": float(p_lower),
                "p_primary_unfolded": float(p_upper_unfolded),
                "p_secondary_unfolded": float(p_lower_unfolded),
                "upper_fold_count": int(upper_folds),
                "lower_fold_count": int(lower_folds),
                "scan_type": "sigma_current_sync",
                "output_channel": int(output_idx) + 1,
                "input_channel": int(input_idx) + 1,
                "path": json.dumps([int(x) for x in path]),
                "state": json.dumps([str(x) for x in state]),
                "bmzi": int(bmzi),
                "warning": warning,
            }
        )

    out_path = save_dir / f"obs{target}_inter_scan.txt"
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.12f")
    print(f"[ColumnPhaseOptimize] saved sigma current scan {out_path}")
    return out_path


def build_bmzi_state_no_upload(path, input_idx, state, bmzi, working_data, mzi_table, N, switch_IN, write_port_voltage):
    for idx, mzi_value in enumerate(path):
        mzi_id = int(mzi_value)
        entry = mzi_table[str(mzi_id)]
        ports = entry.get("ports", [])
        if not ports:
            raise ValueError(f"MZI {mzi_id} missing ports.")
        if state[idx] == "B":
            write_port_voltage(int(ports[0]), get_mzi_state_voltage(entry, "B"), working_data)
        elif state[idx] == "C":
            write_port_voltage(int(ports[0]), get_mzi_state_voltage(entry, "C"), working_data)
        elif state[idx] == "H":
            half_values = entry.get("half_power", [])
            if not half_values:
                raise ValueError(f"MZI {mzi_id} missing half_power voltage.")
            write_port_voltage(int(ports[0]), float(half_values[0]), working_data)

    if int(bmzi) != 0:
        pairs_path = Path("Scandata") / "inter_cali_pairs.json"
        pair_entry = None
        if pairs_path.exists() and pairs_path.stat().st_size > 0:
            with pairs_path.open("r", encoding="utf-8-sig") as f:
                pair_entry = json.load(f).get(str(int(bmzi)))
        if pair_entry:
            pair_ports = pair_entry.get("ports", [])
            if len(pair_ports) == 2:
                write_port_voltage(int(pair_ports[0]), float(pair_entry.get("upper_arm_voltage", 0.0)), working_data)
                write_port_voltage(int(pair_ports[1]), float(pair_entry.get("lower_arm_voltage", 0.0)), working_data)

    for channel in range(1, int(N)):
        switch_IN(channel, "OFF", working_data)
    switch_IN(int(input_idx) + 1, "ON", working_data)


def acquire_current_theta_scans(base_working_data, hardware, mzi_table, args, current_dir):
    current_dir = Path(current_dir)
    delta_dir = current_dir / "delta_current"
    sigma_dir = current_dir / "sigma_current"
    probe_map = parse_probe_map(getattr(args, "probe_map", ""), parse_csv_list(args.mzi_ids, int))
    base_snapshot = base_working_data.copy(deep=True)
    iter_label = getattr(args, "_current_iter_label", "unknown")
    run_log(f"[ColumnPhaseOptimize] iteration {iter_label}: start automatic current theta scans -> {current_dir}")

    for mzi_id in parse_csv_list(args.mzi_ids, int):
        scan_delta_probe_current(
            mzi_id,
            probe_map[str(int(mzi_id))],
            delta_dir,
            base_snapshot,
            hardware,
            mzi_table,
            args,
        )
    for mzi_id in parse_csv_list(args.mzi_ids, int):
        scan_sigma_current(mzi_id, sigma_dir, base_snapshot, hardware, mzi_table, args)

    base_working_data.iloc[:, 0] = base_snapshot.iloc[:, 0].to_numpy(copy=True)
    upload_working_data_checked(
        hardware["cu"],
        hardware["mcv"],
        base_working_data,
        args.voltage_limit_v,
        context_label=f"iteration {iter_label}: restore optimized second-column state after theta scans",
    )
    run_log(f"[ColumnPhaseOptimize] iteration {iter_label}: finished automatic current theta scans")
    return current_dir


def iterate(args):
    mzi_ids, heater_labels, theta_labels, J_df, heater_info, theta_target = load_inputs(args)
    J = J_df.to_numpy(dtype=float)
    mzi_table = load_mzi_table(args.mzi_table)
    run_dir = Path(args.out_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    P_target = compute_theory_power_matrix(
        theta_target,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )

    current_power_path = Path(args.current_power)
    if not current_power_path.exists() and getattr(args, "auto_init_current_power_from_bar", False):
        mzi_ids = parse_csv_list(args.mzi_ids, int)
        P = save_current_power_from_bar_state(current_power_path, mzi_table, heater_info, mzi_ids)
    else:
        P = load_current_power_csv(current_power_path, heater_labels)
    current_theta_path = Path(args.current_theta) if args.current_theta else None
    log_rows = []
    power_rows = []

    hardware = None
    working_data = None
    if not args.dry_run:
        import utils.communication as cu
        from inter_calibration import write_port_voltage

        hardware = {
            "cu": cu,
            "write_port_voltage": write_port_voltage,
            "mcv": cu.open_ser_connection(args.ser_address),
            "opm2": cu.open_VISA_connection(args.opm2_address),
        }
        if hardware["mcv"] is None:
            raise RuntimeError(f"Failed to open serial port {args.ser_address}.")
        if hardware["opm2"] is None:
            raise RuntimeError(f"Failed to open OPM2 {args.opm2_address}.")
        working_data = cu.generate_working_data()
        apply_power_vector_to_working_data(working_data, heater_info, P, write_port_voltage)

    if not args.dry_run and getattr(args, "auto_measure_current_theta", False):
        ensure_theta_reference(args.reference_dir, args, mzi_ids)
    ref_ready_initial = bool(args.reference_dir) and (Path(args.reference_dir) / "theta_reference.json").exists()
    if not args.dry_run and getattr(args, "auto_measure_current_theta", False) and ref_ready_initial:
        args._current_iter_label = "initial"
        acquire_current_theta_scans(working_data, hardware, mzi_table, args, args.current_dir)
        theta_csv = run_dir / "current_theta_initial_measured.csv"
        measure_current_theta(args.reference_dir, args.current_dir, out_csv=theta_csv)
        theta = load_current_theta_csv(theta_csv, theta_labels)
    else:
        try:
            theta = load_initial_theta(args, theta_labels, run_dir=run_dir)
        except FileNotFoundError:
            if args.dry_run or not getattr(args, "auto_measure_current_theta", False):
                raise
            args._current_iter_label = "initial"
            acquire_current_theta_scans(working_data, hardware, mzi_table, args, args.current_dir)
            theta_csv = run_dir / "current_theta_initial_measured.csv"
            measure_current_theta(args.reference_dir, args.current_dir, out_csv=theta_csv)
            theta = load_current_theta_csv(theta_csv, theta_labels)

    for k in range(args.max_iter):
        run_log(f"[ColumnPhaseOptimize] iteration {k + 1}/{args.max_iter}: evaluate current theta and model output")
        if np.isnan(theta).any() and not args.dry_run:
            raise ValueError("Current theta contains NaN; hardware write is blocked.")
        P_current_model = compute_theory_power_matrix(
            theta,
            N=getattr(args, "N", 9),
            mzi_ids=mzi_ids,
            bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
            other_theta_file=getattr(args, "other_theta_file", None),
        )
        output_metrics = save_output_error_outputs(run_dir, f"iter_{k:03d}", P_current_model, P_target)
        measured_metrics = None
        if getattr(args, "measured_output", None):
            measured_output = load_measured_output_matrix(args.measured_output)
            measured_metrics = compute_output_error_metrics(measured_output, P_target)
            save_matrix_csv(measured_output - P_target, run_dir / f"iter_{k:03d}_measured_output_error_matrix.csv")
        error = compute_phase_error(theta_target, theta)
        theta_converged = bool(np.max(np.abs(error)) < args.theta_tol)
        output_converged = bool(output_metrics["rmse"] < getattr(args, "output_tol", 0.02))
        converged = theta_converged and output_converged
        pd.DataFrame({"theta": theta_labels, "value_rad": theta}).to_csv(run_dir / f"iter_{k:03d}_theta.csv", index=False)
        pd.DataFrame({"heater": heater_labels, "power_w": P}).to_csv(run_dir / f"iter_{k:03d}_power.csv", index=False)
        pd.DataFrame({"theta": theta_labels, "error_rad": error}).to_csv(run_dir / f"iter_{k:03d}_error.csv", index=False)
        power_rows.append({"iter": k, **dict(zip(heater_labels, P))})
        run_log(
            f"[ColumnPhaseOptimize] iteration {k + 1}/{args.max_iter}: "
            f"max_theta_error={float(np.max(np.abs(error))):.6f} rad, "
            f"theta_norm={float(np.linalg.norm(error)):.6f}, "
            f"model_output_rmse={output_metrics['rmse']:.6f}"
        )
        if converged:
            run_log(f"[ColumnPhaseOptimize] iteration {k + 1}/{args.max_iter}: converged")
            log_rows.append(
                {
                    "iter": k,
                    "max_abs_theta_error_rad": float(np.max(np.abs(error))),
                    "norm_theta_error": float(np.linalg.norm(error)),
                    "model_output_rmse": output_metrics["rmse"],
                    "model_output_mae": output_metrics["mae"],
                    "model_output_max_abs": output_metrics["max_abs"],
                    "measured_output_rmse": "" if measured_metrics is None else measured_metrics["rmse"],
                    "measured_output_mae": "" if measured_metrics is None else measured_metrics["mae"],
                    "measured_output_max_abs": "" if measured_metrics is None else measured_metrics["max_abs"],
                    "alpha_used": args.alpha,
                    "lambda_reg": args.lambda_reg,
                    "step_scale": 1.0,
                    "num_clipped": 0,
                    "theta_converged": theta_converged,
                    "output_converged": output_converged,
                    "converged": True,
                    "warning": "",
                }
            )
            break

        plan = plan_update(J, P, theta, theta_target, args, P_target=P_target, mzi_ids=mzi_ids)
        run_log(
            f"[ColumnPhaseOptimize] iteration {k + 1}/{args.max_iter}: "
            f"planned update alpha_used={float(plan.get('alpha_used', args.alpha)):.6f}, "
            f"step_scale={float(plan['step_scale']):.6f}, clipped={int(np.sum(plan['clipped']))}"
        )
        update_df, _, _ = save_plan_outputs(
            run_dir / f"iter_{k:03d}_plan",
            heater_labels,
            theta_labels,
            heater_info,
            P,
            theta,
            plan,
            args,
            P_target=P_target,
            P_current_model=P_current_model,
            output_metrics=output_metrics,
        )
        update_df.to_csv(run_dir / f"iter_{k:03d}_update.csv", index=False)
        if not args.dry_run:
            upload_second_column_voltages(
                working_data,
                update_df,
                hardware["cu"],
                hardware["write_port_voltage"],
                hardware["mcv"],
                args.voltage_limit_v,
                context_label=f"iteration {k + 1}/{args.max_iter}: optimization update",
            )
            import time

            time.sleep(args.settle_time)
            ref_ready = bool(args.reference_dir) and (Path(args.reference_dir) / "theta_reference.json").exists()
            if getattr(args, "auto_measure_current_theta", False):
                if not ref_ready:
                    ensure_theta_reference(args.reference_dir, args, mzi_ids)
                args._current_iter_label = f"{k + 1}/{args.max_iter}"
                acquire_current_theta_scans(working_data, hardware, mzi_table, args, args.current_dir)
                theta_csv = run_dir / f"iter_{k:03d}_measured_theta.csv"
                measure_current_theta(args.reference_dir, args.current_dir, out_csv=theta_csv)
                theta = load_current_theta_csv(theta_csv, theta_labels)
                run_log(f"[ColumnPhaseOptimize] iteration {k + 1}/{args.max_iter}: updated theta from automatic scans")
            elif ref_ready and Path(args.current_dir).exists():
                theta_csv = run_dir / f"iter_{k:03d}_measured_theta.csv"
                measure_current_theta(args.reference_dir, args.current_dir, out_csv=theta_csv)
                theta = load_current_theta_csv(theta_csv, theta_labels)
            elif current_theta_path is not None and current_theta_path.exists():
                theta = load_current_theta_csv(current_theta_path, theta_labels)
            else:
                print(
                    "[ColumnPhaseOptimize] No theta measurement source is available after hardware update; "
                    "stopping after this iteration."
                )
                P = plan["P_next"]
                break
        else:
            theta = theta + J @ (plan["P_next"] - P)
            run_log(f"[ColumnPhaseOptimize] iteration {k + 1}/{args.max_iter}: dry-run theta propagated")
        P = plan["P_next"]
        log_rows.append(
            {
                "iter": k,
                "max_abs_theta_error_rad": float(np.max(np.abs(error))),
                "norm_theta_error": float(np.linalg.norm(error)),
                "model_output_rmse": output_metrics["rmse"],
                "model_output_mae": output_metrics["mae"],
                "model_output_max_abs": output_metrics["max_abs"],
                "measured_output_rmse": "" if measured_metrics is None else measured_metrics["rmse"],
                "measured_output_mae": "" if measured_metrics is None else measured_metrics["mae"],
                "measured_output_max_abs": "" if measured_metrics is None else measured_metrics["max_abs"],
                "alpha_used": float(plan.get("alpha_used", args.alpha)),
                "lambda_reg": args.lambda_reg,
                "step_scale": plan["step_scale"],
                "num_clipped": int(np.sum(plan["clipped"])),
                "theta_converged": theta_converged,
                "output_converged": output_converged,
                "converged": False,
                "warning": "; ".join(plan["warnings"]),
            }
        )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(run_dir / "iteration_log.csv", index=False)
    final = {
        "final_power_w": dict(zip(heater_labels, P)),
        "dry_run": bool(args.dry_run),
        "max_iter": args.max_iter,
    }
    with (run_dir / "final_result.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    if not log_df.empty:
        plot_error_history(log_df, run_dir / "error_history.png")
        plot_output_rmse_history(log_df, run_dir / "output_rmse_history.png")
    if power_rows:
        plot_power_history(pd.DataFrame(power_rows), run_dir / "power_history.png")
    final_model = compute_theory_power_matrix(
        theta,
        N=getattr(args, "N", 9),
        mzi_ids=mzi_ids,
        bw_phases=getattr(args, "bw_phases", DEFAULT_BW_PHASES),
        other_theta_file=getattr(args, "other_theta_file", None),
    )
    plot_theta_target_vs_current(theta_labels, theta, theta_target, run_dir / "final_theta_compare.png")
    plot_output_compare(P_target, final_model, run_dir / "final_output_compare.png")
    if hardware is not None and hardware.get("mcv") is not None:
        for key in ("mcv", "opm2"):
            handle = hardware.get(key)
            close = getattr(handle, "close", None)
            if callable(close):
                close()


def build_parser():
    parser = argparse.ArgumentParser(description="Optimize second-column MZI arm phases using measured J_theta.")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("--j_theta", default="results/J_full/J_theta_rad_per_w.csv")
        p.add_argument("--mzi_table", default="Scandata/MZI_table.json")
        p.add_argument("--mzi_ids", default="5,6,7,8")
        p.add_argument("--out_dir", default="results/ColumnPhaseOptimize")
        p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
        p.add_argument("--lambda_reg", type=float, default=DEFAULT_LAMBDA_REG)
        p.add_argument("--power_limit_w", type=float, default=DEFAULT_POWER_LIMIT_W)
        p.add_argument("--step_limit_w", type=float, default=DEFAULT_STEP_LIMIT_W)
        p.add_argument("--theta_tol", type=float, default=DEFAULT_THETA_TOL)
        p.add_argument("--voltage_limit_v", type=float, default=DEFAULT_VOLTAGE_LIMIT_V)
        p.add_argument("--enable_branch_search", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
        p.add_argument("--branch_candidates", default="0,1")
        p.add_argument("--N", type=int, default=9)
        p.add_argument("--bw_phases", default=DEFAULT_BW_PHASES)
        p.add_argument("--other_theta_file")
        p.add_argument("--output_tol", type=float, default=0.02)
        p.add_argument("--measured_output")
        p.add_argument("--enable_line_search", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
        p.add_argument("--line_search_min_alpha", type=float, default=0.05)
        p.add_argument("--line_search_shrink", type=float, default=0.5)
        p.add_argument("--assume_reference_zero", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)

    p_init = sub.add_parser("init_reference")
    p_init.add_argument("--mzi_ids", default="5,6,7,8")
    p_init.add_argument("--reference_dir", default="Scandata/current_theta_reference")
    p_init.add_argument("--probe_map", default="5:u,6:u,7:u,8:u")
    p_init.add_argument("--sigma_sign", default="Scandata/J_sigma/sign_check/sigma_sign.json")
    p_init.add_argument("--dry_run", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)

    p_measure_theta = sub.add_parser("measure_theta")
    p_measure_theta.add_argument("--reference_dir", default="Scandata/current_theta_reference")
    p_measure_theta.add_argument("--current_dir", default="Scandata/current_theta_measurement")
    p_measure_theta.add_argument("--out_csv", default="current_theta_second_column.csv")

    p_plan = sub.add_parser("plan")
    add_common(p_plan)
    p_plan.add_argument("--current_power", required=True)
    p_plan.add_argument("--current_theta", required=True)
    p_plan.add_argument("--reference_dir", default="Scandata/current_theta_reference")
    p_plan.add_argument("--current_dir", default="Scandata/current_theta_measurement")

    p_sim = sub.add_parser("simulate")
    add_common(p_sim)
    p_sim.add_argument("--current_power", required=True)
    p_sim.add_argument("--current_theta", required=True)
    p_sim.add_argument("--reference_dir", default="Scandata/current_theta_reference")
    p_sim.add_argument("--current_dir", default="Scandata/current_theta_measurement")
    p_sim.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)

    p_iter = sub.add_parser("iterate")
    add_common(p_iter)
    p_iter.add_argument("--current_power", required=True)
    p_iter.add_argument("--current_theta")
    p_iter.add_argument("--reference_dir")
    p_iter.add_argument("--current_dir")
    p_iter.add_argument("--delta_reference_source_dir", default="jacobian_measurements/baseline")
    p_iter.add_argument("--sigma_reference_source_dir", default="Scandata/J_sigma/baseline")
    p_iter.add_argument("--sigma_sign_path", default="Scandata/J_sigma/sign_check/sigma_sign.json")
    p_iter.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)
    p_iter.add_argument("--settle_time", type=float, default=DEFAULT_SETTLE_TIME)
    p_iter.add_argument("--dry_run", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    p_iter.add_argument("--ser_address", default="COM3")
    p_iter.add_argument("--opm2_address", default=DEFAULT_OPM2_ADDRESS)
    p_iter.add_argument("--pause_for_manual_theta_update", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    p_iter.add_argument("--auto_measure_current_theta", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    p_iter.add_argument("--probe_map", default="5:u,6:u,7:u,8:u")
    p_iter.add_argument("--probe_half_width_w", type=float, default=0.001)
    p_iter.add_argument("--probe_step_w", type=float, default=0.00025)
    p_iter.add_argument("--sigma_phase_points", default="0,0.785398,1.570796,2.356194,3.141593,3.926991,4.712389,5.497787,6.283185")
    p_iter.add_argument("--auto_init_current_power_from_bar", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "init_reference":
        mzi_ids = parse_csv_list(args.mzi_ids, int)
        probe_map = parse_probe_map(args.probe_map, mzi_ids)
        init_theta_reference(
            reference_dir=args.reference_dir,
            probe_map=probe_map,
            sigma_sign_path=args.sigma_sign,
            mzi_ids=mzi_ids,
            dry_run=args.dry_run,
        )
    elif args.mode == "measure_theta":
        measure_current_theta_from_files(args.reference_dir, args.current_dir, args.out_csv)
    elif args.mode == "plan":
        run_plan(args)
    elif args.mode == "simulate":
        simulate(args)
    elif args.mode == "iterate":
        iterate(args)
    else:
        parser.error(f"Unsupported mode: {args.mode}")


def direct_main(config=DIRECT_RUN_CONFIG):
    """
    PyCharm/IDE direct-run entry.

    Edit DIRECT_RUN_CONFIG at the top of this file, then click Run.  The default
    path is the real hardware iterate flow, not argparse subcommands.
    """
    args = SimpleNamespace(**config.__dict__)
    run_log(f"[ColumnPhaseOptimize] direct run mode = {args.mode}")
    run_log(f"[ColumnPhaseOptimize] dry_run = {args.dry_run}")

    if args.mode == "iterate":
        if args.dry_run:
            raise RuntimeError(
                "DIRECT_RUN_CONFIG.dry_run is True. Set it to False only when the "
                "hardware connection, current_power, and current_theta/reference scans are ready."
            )
        iterate(args)
    elif args.mode == "plan":
        run_plan(args)
    elif args.mode == "measure_theta":
        measure_current_theta_from_files(args.reference_dir, args.current_dir, args.current_theta)
    elif args.mode == "init_reference":
        mzi_ids = parse_csv_list(args.mzi_ids, int)
        probe_map = parse_probe_map("5:u,6:u,7:u,8:u", mzi_ids)
        init_theta_reference(
            reference_dir=args.reference_dir,
            probe_map=probe_map,
            sigma_sign_path="Scandata/J_sigma/sign_check/sigma_sign.json",
            mzi_ids=mzi_ids,
            dry_run=False,
        )
    else:
        raise ValueError(f"Unsupported DIRECT_RUN_CONFIG.mode: {args.mode}")


if __name__ == "__main__":
    direct_main()
