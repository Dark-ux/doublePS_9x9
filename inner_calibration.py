import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.communication as cu
import utils.AllDecompositionUtils as du
import serial
import json
from colorama import Fore, Style, init
import time
import builtins
import copy

_PROGRAM_START_TIME = time.monotonic()


def _format_elapsed_time() -> str:
    elapsed_seconds = int(time.monotonic() - _PROGRAM_START_TIME)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _elapsed_print(*args, **kwargs):
    builtins.print(f"[{_format_elapsed_time()}]", *args, **kwargs)


print = _elapsed_print

INNER_SCAN_DATA_DIR = os.path.join("Scandata", "inner_cali_powerdata")
INNER_FIT_IMAGE_DIR = os.path.join("Scandata", "inner_cali")
INNER_FIT_PARAMS_PATH = os.path.join("Scandata", "inner_fit_params.json")
INNER_MZI_TABLE_PATH = os.path.join("Scandata", "MZI_table.json")
INNER_MZI_TABLE_BACKUP_PATH = os.path.join("Scandata-backup-04", "MZI_table.json")
INNER_MZI_STATE_TABLE_PATH = os.path.join("Scandata", "mzi_state_table.json")
INNER_MZI_STATE_TABLE_BACKUP_PATH = os.path.join("Scandata-backup-04", "mzi_state_table.json")
INNER_SCAN_START_POWER_W = 0.0
INNER_SCAN_END_POWER_W = 0.06
INNER_SCAN_STEP_POWER_W = 0.001
INNER_FINE_SCAN_WINDOW_POWER_W = 0.001
INNER_FINE_SCAN_STEP_POWER_W = 0.0001


def _inner_arm_suffix(arm_index: int) -> str:
    return "u" if int(arm_index) == 0 else "d"


def _inner_scan_stem(mzi_id=None, arm_index: int = 0, port=None) -> str:
    if mzi_id is not None:
        return f"{int(mzi_id)}-{_inner_arm_suffix(arm_index)}"
    if port is None:
        raise ValueError("port is required when mzi_id is not provided.")
    return str(int(port))


def _inner_scan_data_path(mzi_id=None, arm_index: int = 0, port=None, data_dir: str = INNER_SCAN_DATA_DIR) -> str:
    return os.path.join(data_dir, f"{_inner_scan_stem(mzi_id, arm_index, port)}.txt")


def _infer_mzi_arm_from_port(port: int):
    mzi_id = _find_mzi_id_for_port(int(port))
    if mzi_id is None:
        return None, None
    ports = get_mzi_h_list(int(mzi_id))
    if int(port) not in ports:
        return None, None
    return int(mzi_id), ports.index(int(port))


def _resolve_inner_power_scan_path(port, data_dir: str, mzi_id=None, arm_index=None) -> str:
    candidates = []
    if isinstance(port, str) and (port.endswith(".txt") or os.path.sep in port or "/" in port):
        candidates.append(port)

    if mzi_id is not None:
        resolved_arm_index = 0 if arm_index is None else int(arm_index)
        candidates.append(_inner_scan_data_path(mzi_id, resolved_arm_index, port, data_dir))

    try:
        inferred_mzi_id, inferred_arm_index = _infer_mzi_arm_from_port(int(port))
    except Exception:
        inferred_mzi_id, inferred_arm_index = None, None
    if inferred_mzi_id is not None:
        candidates.append(_inner_scan_data_path(inferred_mzi_id, inferred_arm_index, port, data_dir))

    try:
        candidates.append(os.path.join(data_dir, f"{int(port)}.txt"))
    except (TypeError, ValueError):
        pass

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    if candidates:
        return candidates[0]
    raise ValueError(f"Cannot resolve scan data path for port={port!r}.")


def _inner_power_model(x, A, w, phi, b):
    return A * np.sin(w * x + phi) + b


def _normalize_phase(phi: float) -> float:
    return float((phi + np.pi) % (2 * np.pi) - np.pi)


def _load_inner_power_scan(port, data_dir: str, mzi_id=None, arm_index=None) -> dict:
    filename = _resolve_inner_power_scan_path(port, data_dir, mzi_id=mzi_id, arm_index=arm_index)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find data file: {filename}")

    df = pd.read_csv(filename)
    if df.shape[1] < 3:
        raise ValueError(f"Expect at least three columns in {filename}")

    if {"target_power_w", "voltage_v", "optical_power_uW", "current_mA"}.issubset(df.columns):
        target_power = df["target_power_w"].to_numpy(dtype=float)
        voltage = df["voltage_v"].to_numpy(dtype=float)
        optical_power = df["optical_power_uW"].to_numpy(dtype=float)
        current_ma = df["current_mA"].to_numpy(dtype=float)
        measured_electrical_power = (
            df["measured_power_w"].to_numpy(dtype=float)
            if "measured_power_w" in df.columns
            else voltage * current_ma * 1e-3
        )
        electrical_power = target_power
    else:
        voltage = df.iloc[:, 0].to_numpy(dtype=float)
        optical_power = df.iloc[:, 1].to_numpy(dtype=float)
        current_ma = df.iloc[:, 2].to_numpy(dtype=float)
        current_amp = current_ma * 1e-3
        measured_electrical_power = voltage * current_amp
        electrical_power = measured_electrical_power

    mask = (
        np.isfinite(voltage)
        & np.isfinite(optical_power)
        & np.isfinite(current_ma)
        & np.isfinite(electrical_power)
        & np.isfinite(measured_electrical_power)
    )
    voltage = voltage[mask]
    optical_power = optical_power[mask]
    current_ma = current_ma[mask]
    electrical_power = electrical_power[mask]
    measured_electrical_power = measured_electrical_power[mask]

    if voltage.size < 4:
        raise ValueError(f"Not enough valid samples in {filename}")

    order = np.argsort(electrical_power)
    return {
        "path": filename,
        "voltage": voltage[order],
        "optical_power": optical_power[order],
        "current_ma": current_ma[order],
        "electrical_power": electrical_power[order],
        "measured_electrical_power": measured_electrical_power[order],
    }


def _is_power_step_scan_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except (OSError, pd.errors.ParserError):
        return False
    required_columns = {
        "mzi_id",
        "arm_index",
        "arm_name",
        "port",
        "target_power_w",
        "voltage_v",
        "optical_power_uW",
        "current_mA",
    }
    if not required_columns.issubset(columns) or "scan_stage" not in columns:
        return False
    try:
        stages = pd.read_csv(path, usecols=["scan_stage"])["scan_stage"].astype(str).str.lower()
    except (OSError, pd.errors.ParserError, ValueError):
        return False
    return stages.str.contains("fine", regex=False).any()


def _estimate_inner_power_w(electrical_power: np.ndarray, optical_power: np.ndarray) -> float:
    x = np.asarray(electrical_power, dtype=float)
    y = np.asarray(optical_power, dtype=float)
    if x.size < 4:
        return float(2 * np.pi / max(np.ptp(x), 1e-6))

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]
    if x_unique.size < 4:
        return float(2 * np.pi / max(np.ptp(x), 1e-6))

    x_uniform = np.linspace(x_unique[0], x_unique[-1], max(128, x_unique.size))
    y_uniform = np.interp(x_uniform, x_unique, y_unique)
    spacing = float(x_uniform[1] - x_uniform[0])
    if spacing <= 0:
        return float(2 * np.pi / max(x_unique[-1] - x_unique[0], 1e-6))

    spectrum = np.abs(np.fft.rfft(y_uniform - np.mean(y_uniform)))
    freqs = np.fft.rfftfreq(x_uniform.size, d=spacing)
    if spectrum.size > 1:
        freq_idx = int(np.argmax(spectrum[1:]) + 1)
        if freqs[freq_idx] > 0:
            return float(2 * np.pi * freqs[freq_idx])

    return float(2 * np.pi / max(x_unique[-1] - x_unique[0], 1e-6))


def _fit_inner_power_curve(electrical_power: np.ndarray, optical_power: np.ndarray):
    x = np.asarray(electrical_power, dtype=float)
    y = np.asarray(optical_power, dtype=float)
    if x.size < 4:
        raise ValueError("At least four samples are required for fitting.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    amplitude_guess = max((y_max - y_min) * 0.5, float(np.std(y)) * np.sqrt(2), 1e-9)
    b_guess = float((y_max + y_min) * 0.5)
    w_guess = _estimate_inner_power_w(x, y)
    if not np.isfinite(w_guess) or w_guess <= 0:
        w_guess = float(2 * np.pi / max(np.ptp(x), 1e-6))

    arg0 = np.clip((y[0] - b_guess) / amplitude_guess, -1.0, 1.0)
    base_phase = float(np.arcsin(arg0))
    phase_candidates = [
        base_phase - w_guess * x[0],
        np.pi - base_phase - w_guess * x[0],
        -np.pi / 2,
        0.0,
        np.pi / 2,
    ]

    best_params = None
    best_error = None
    for phi_guess in phase_candidates:
        try:
            popt, _ = curve_fit(
                _inner_power_model,
                x,
                y,
                p0=[amplitude_guess, w_guess, phi_guess, b_guess],
                bounds=([1e-12, 1e-9, -20 * np.pi, -np.inf], [np.inf, np.inf, 20 * np.pi, np.inf]),
                maxfev=12000,
            )
        except (RuntimeError, ValueError):
            continue

        residual = _inner_power_model(x, *popt) - y
        sse = float(np.sum(residual**2))
        if best_error is None or sse < best_error:
            best_error = sse
            best_params = popt

    if best_params is None:
        raise RuntimeError("Sine fit failed for inner calibration data.")

    A, w, phi, b = best_params
    return float(A), float(w), _normalize_phase(float(phi)), float(b)


def _solve_model_points_in_range(base_phase: float, w: float, phi: float, x_min: float, x_max: float):
    if w == 0:
        return []

    k_min = int(np.floor((w * x_min + phi - base_phase) / (2 * np.pi))) - 1
    k_max = int(np.ceil((w * x_max + phi - base_phase) / (2 * np.pi))) + 1
    values = []
    for k in range(k_min, k_max + 1):
        x_val = (base_phase - phi + 2 * np.pi * k) / w
        if x_min <= x_val <= x_max:
            values.append(float(x_val))

    deduped = []
    for value in sorted(values):
        if not deduped or abs(value - deduped[-1]) > 1e-9:
            deduped.append(value)
    return deduped


def _voltage_from_resistance_power(power_w: float, resistance_ohm: float) -> float:
    if power_w < 0:
        raise ValueError("power_w must be non-negative.")
    if resistance_ohm <= 0:
        raise ValueError("resistance_ohm must be positive.")
    return float(np.sqrt(float(power_w) * float(resistance_ohm)))


def _round_voltage(voltage: float) -> float:
    return round(float(voltage), 3)


def _dedupe_sorted(values, tol=1e-9):
    deduped = []
    for value in sorted(float(v) for v in values if np.isfinite(v)):
        if not deduped or abs(value - deduped[-1]) > tol:
            deduped.append(value)
    return deduped


def _get_fine_scan_powers_from_fit(
    A: float,
    w: float,
    phi: float,
    power_min: float,
    power_max: float,
    existing_powers,
):
    del A
    target_specs = [
        ("max", np.pi / 2),
        ("min", -np.pi / 2),
        ("half_rising", 0.0),
    ]
    special_points = []
    intervals = []
    for name, phase in target_specs:
        powers = _solve_model_points_in_range(phase, w, phi, power_min, power_max)
        if not powers:
            raise ValueError(f"Cannot find model {name} point in scan power range.")

        power = max(powers)
        special_points.append({"type": name, "power_w": float(power)})
        left = max(power_min, float(power) - INNER_FINE_SCAN_WINDOW_POWER_W)
        right = min(power_max, float(power) + INNER_FINE_SCAN_WINDOW_POWER_W)
        if right >= left:
            intervals.append((left, right))

    merged_intervals = []
    for left, right in sorted(intervals):
        if not merged_intervals or left > merged_intervals[-1][1] + INNER_FINE_SCAN_STEP_POWER_W * 0.5:
            merged_intervals.append([left, right])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], right)

    existing = set(np.round(np.asarray(existing_powers, dtype=float), 6).tolist())
    fine_points = set()
    for left, right in merged_intervals:
        values = np.arange(left, right + INNER_FINE_SCAN_STEP_POWER_W * 0.5, INNER_FINE_SCAN_STEP_POWER_W)
        fine_points.update(np.round(values, 6).tolist())
    fine_points.difference_update(existing)

    fit_info = {
        "special_points": special_points,
        "fine_windows": [[round(left, 6), round(right, 6)] for left, right in merged_intervals],
        "fine_step_w": float(INNER_FINE_SCAN_STEP_POWER_W),
    }
    return np.array(sorted(fine_points), dtype=float), fit_info


def _get_bar_cross_half_from_fit(
    A: float,
    w: float,
    phi: float,
    resistance_ohm: float,
    power_min: float,
    power_max: float,
):
    del A
    max_powers = _solve_model_points_in_range(np.pi / 2, w, phi, power_min, power_max)
    min_powers = _solve_model_points_in_range(-np.pi / 2, w, phi, power_min, power_max)
    half_powers = _solve_model_points_in_range(0.0, w, phi, power_min, power_max)
    if not max_powers:
        raise ValueError("Cannot find model maximum point in scan power range.")
    if not min_powers:
        raise ValueError("Cannot find model minimum point in scan power range.")
    if not half_powers:
        raise ValueError("Cannot find model increasing half-power point in scan power range.")

    bar_power = max(max_powers)
    cross_power = max(min_powers)
    half_power = max(half_powers)
    return {
        "bar_power_w": float(bar_power),
        "bar_voltage_v": _round_voltage(_voltage_from_resistance_power(bar_power, resistance_ohm)),
        "cross_power_w": float(cross_power),
        "cross_voltage_v": _round_voltage(_voltage_from_resistance_power(cross_power, resistance_ohm)),
        "half_power_w": float(half_power),
        "half_power_voltage_v": _round_voltage(_voltage_from_resistance_power(half_power, resistance_ohm)),
    }


def _power_to_voltage(power_values: np.ndarray, voltage_values: np.ndarray, target_powers):
    power_values = np.asarray(power_values, dtype=float)
    voltage_values = np.asarray(voltage_values, dtype=float)
    order = np.argsort(power_values)
    p_sorted = power_values[order]
    v_sorted = voltage_values[order]
    p_unique, unique_idx = np.unique(np.round(p_sorted, 12), return_index=True)
    v_unique = v_sorted[unique_idx]

    voltages = []
    if p_unique.size == 0:
        return voltages

    p_min = float(np.min(p_unique))
    p_max = float(np.max(p_unique))
    for power in target_powers:
        if not np.isfinite(power) or power < p_min or power > p_max:
            continue
        voltage = float(np.interp(power, p_unique, v_unique))
        voltages.append(voltage)
    return voltages


def load_mzi_table(path: str = INNER_MZI_TABLE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return _normalize_mzi_table_structure(json.load(f))
    if path == INNER_MZI_TABLE_PATH:
        return _normalize_mzi_table_structure(_build_mzi_table_from_list())
    return {}


def fit_p_to_op(index: int, data_dir: str = INNER_SCAN_DATA_DIR, mzi_id=None, arm_index=None):
    """
    Fit OP = A * sin(w * P + phi) + b using the saved scan data.
    Returns (A, w, phi, b, p_elec, op).
    """
    scan = _load_inner_power_scan(index, data_dir, mzi_id=mzi_id, arm_index=arm_index)
    A, w, phi, b = _fit_inner_power_curve(scan["electrical_power"], scan["optical_power"])
    return A, w, phi, b, scan["electrical_power"], scan["optical_power"]


def p_from_op(
    A: float, w: float, phi: float, b: float, op_value: float, port: int, data_dir: str = INNER_SCAN_DATA_DIR
):
    """
    Given OP = A * sin(w * P + phi) + b:
    - return all P in range that match the OP
    - return P where OP is max (sin = 1) in range
    - return P where OP is min (sin = -1) in range
    """
    if A <= 0 or w == 0:
        raise ValueError("A and w must be non-zero.")

    scan = _load_inner_power_scan(port, data_dir)
    p_elec = scan["electrical_power"]
    if p_elec.size == 0:
        raise ValueError(f"No power data for port {port}")

    p_min = max(0.0, float(np.min(p_elec)))
    p_max = float(np.max(p_elec))
    if p_max <= 0:
        raise ValueError(f"Invalid max power for port {port}")

    def dedup(values, tol=1e-9):
        out = []
        for x in sorted(values):
            if not out or abs(x - out[-1]) > tol:
                out.append(x)
        return out

    # Solutions for specified OP
    arg = np.clip((op_value - b) / A, -1.0, 1.0)
    base = np.arcsin(arg)
    p_solutions = []
    p_solutions.extend(_solve_model_points_in_range(base, w, phi, p_min, p_max))
    p_solutions.extend(_solve_model_points_in_range(np.pi - base, w, phi, p_min, p_max))
    p_solutions = dedup(p_solutions)

    max_phase = np.pi / 2
    min_phase = -np.pi / 2

    p_at_max = dedup(_solve_model_points_in_range(max_phase, w, phi, p_min, p_max))
    p_at_min = dedup(_solve_model_points_in_range(min_phase, w, phi, p_min, p_max))

    return p_solutions, p_at_max, p_at_min


def switch_IN(mzi_index: int, state: str, file_data: pd.DataFrame) -> float:
    """
    Lookup voltage for given MZI index and state ("ON"/"OFF") from IN_MZI.txt,
    write it into the intermediate file_data at the corresponding PORT row,
    and return the voltage value.
    """
    table = pd.read_csv("IN_MZI.txt")
    state_norm = state.strip().upper()
    if state_norm not in {"ON", "OFF"}:
        raise ValueError("state must be 'ON' or 'OFF'")

    row = table.loc[table["MZI"] == mzi_index]
    if row.empty:
        raise ValueError(f"MZI index {mzi_index} not found in IN_MZI.txt")

    port = int(row.iloc[0]["PORT"])
    voltage = float(row.iloc[0][state_norm])

    # PORT in file is treated as 1-based; adjust to 0-based index for DataFrame
    port_idx = port - 1
    if port_idx < 0 or port_idx >= len(file_data):
        raise IndexError(f"PORT {port} is out of range for the provided file_data.")

    file_data.iloc[port_idx, 0] = voltage
    return voltage


def write_port_voltage(port: int, voltage: float, file_data: pd.DataFrame) -> None:
    """
    Write voltage into file_data at the row corresponding to PORT (1-based).
    """
    port_idx = port - 1
    if port_idx < 0 or port_idx >= len(file_data):
        raise IndexError(f"PORT {port} is out of range for the provided file_data.")
    file_data.iloc[port_idx, 0] = voltage


def reset_heater_voltage(port: int, ser, file_data: pd.DataFrame) -> None:
    """
    Force one heater voltage to 0 V and upload the table immediately.
    """
    write_port_voltage(int(port), 0.0, file_data)
    cu.upload_voltage(ser, file_data)
    print(f"Port {int(port)} heater voltage reset to 0 V")


def voltage_from_power(port: int, power_w: float, data_dir: str = INNER_SCAN_DATA_DIR) -> float:
    """
    Given a PORT and electrical power (W), interpolate voltage from the saved
    scan data and return the voltage (rounded to 3 decimals).
    """
    if power_w < 0:
        raise ValueError("power_w must be non-negative.")

    scan = _load_inner_power_scan(port, data_dir)
    p_elec = scan["electrical_power"]
    voltage = scan["voltage"]

    sort_idx = np.argsort(p_elec)
    p_sorted = p_elec[sort_idx]
    v_sorted = voltage[sort_idx]
    p_unique, unique_idx = np.unique(np.round(p_sorted, 12), return_index=True)
    v_unique = v_sorted[unique_idx]

    p_min = float(p_unique.min())
    p_max = float(p_unique.max())
    if not (p_min <= power_w <= p_max):
        raise ValueError(f"power_w {power_w} out of range [{p_min}, {p_max}] for port {port}")

    v_at_p = float(np.interp(power_w, p_unique, v_unique))
    return round(v_at_p, 3)


def get_cali_order(N):
    M = du.Clements_matrix(N)
    order1 = np.array([])
    order2 = np.array([])
    for i in range(N):
        if i == 0:
            for item in M[0][M[0] != 0]:
                order1 = np.append(order1, item)
            M = np.delete(M, 0, axis=0)
        else:
            for m in range(M.shape[0]):
                for n in range(M.shape[1]):
                    if m - n == (2 * i - (N + 1) if N % 2 == 0 else 2 * i - N):
                        order2 = np.append(order2, M[m][n])
    return order1, order2


def find_path(target, N):
    M = du.Clements_matrix(N)
    PATH = np.array([])
    idx = np.where(M == target)
    cx, cy = idx[0][0], idx[1][0]
    bx, by = idx[0][0], idx[1][0]
    path = [(cx, cy)]
    while True:
        if cy == 0 and cx != 0:
            input = cx
            break
        if cx == 0 and cy == 0:
            input = 0
            break
        if cx > 0:
            cx -= 1
        if cy > 0:
            cy -= 1
        if M[cx][cy] != 0:
            path.insert(0, (cx, cy))
    while True:
        if by == N - 1 and bx != 0:
            ouput = bx
            break
        if bx == 0 and by == N - 1:
            ouput = 0
            break
        if bx > 0:
            bx -= 1
        if by < N - 1:
            by += 1
        if M[bx][by] != 0:
            path.append((bx, by))
    state = []
    for i in range(len(path)):
        if i == 0:
            if path[i][0] == path[i + 1][0]:
                state.append("B")
            else:
                state.append("C")
        elif i == len(path) - 1:
            if path[i][0] == path[i - 1][0]:
                state.append("B")
            else:
                state.append("C")
        else:
            if path[i - 1][0] == path[i + 1][0]:
                state.append("B")
            else:
                state.append("C")
    for item in path:
        PATH = np.append(PATH, M[item[0]][item[1]])
    return PATH, input, ouput, state


def get_mzi_h_list(mzi_id, file_path="MZI_list.xlsx"):
    df = pd.read_excel(file_path)
    row = df.loc[df["MZI"] == mzi_id]
    if row.empty:
        raise ValueError(f"MZI {mzi_id} not found in {file_path}")

    row = row.iloc[0]
    flag = int(row["flag"])
    h1 = row["H1"]
    if pd.isna(h1):
        raise ValueError(f"MZI {mzi_id} has no H1 value")
    h1 = int(h1)

    if flag == 1:
        return [h1]
    if flag == 2:
        h2 = row["H2"]
        if pd.isna(h2):
            raise ValueError(f"MZI {mzi_id} has flag=2 but no H2 value")
        return [h1, int(h2)]

    raise ValueError(f"MZI {mzi_id} has unsupported flag {flag}")


def _save_inner_fit_plot(
    scan_stem: str,
    electrical_power: np.ndarray,
    optical_power: np.ndarray,
    A: float,
    w: float,
    phi: float,
    b: float,
    title: str = None,
):
    os.makedirs(INNER_FIT_IMAGE_DIR, exist_ok=True)
    x_smooth = np.linspace(float(np.min(electrical_power)), float(np.max(electrical_power)), 800)
    y_smooth = _inner_power_model(x_smooth, A, w, phi, b)

    plt.figure(figsize=(7, 5))
    plt.plot(electrical_power, optical_power, "o", markersize=4, label="samples")
    plt.plot(x_smooth, y_smooth, "-", linewidth=1.8, label="fit")
    plt.xlabel("Electrical Power (W)")
    plt.ylabel("Optical Power (uW)")
    plt.title(title or f"{scan_stem} Power Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    image_path = os.path.join(INNER_FIT_IMAGE_DIR, f"{scan_stem}.png")
    plt.savefig(image_path, dpi=150)
    plt.close()
    return image_path


def _update_inner_fit_params(port: int, A: float, w: float, phi: float, b: float):
    os.makedirs(os.path.dirname(INNER_FIT_PARAMS_PATH), exist_ok=True)
    fit_params = {}
    if os.path.exists(INNER_FIT_PARAMS_PATH):
        with open(INNER_FIT_PARAMS_PATH, "r", encoding="utf-8-sig") as f:
            fit_params = json.load(f)

    ppi = float(np.pi / w)
    fit_params[str(int(port))] = {
        "A": float(A),
        "w": float(w),
        "phi": float(phi),
        "b": float(b),
        "Ppi": ppi,
    }

    with open(INNER_FIT_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(fit_params, f, ensure_ascii=False, indent=2)


def _find_mzi_id_for_port(port: int, file_path: str = "MZI_list.xlsx"):
    df = pd.read_excel(file_path)
    port = int(port)
    for _, row in df.iterrows():
        ports = [row["H1"]]
        if "H2" in row and not pd.isna(row["H2"]):
            ports.append(row["H2"])
        if port in [int(p) for p in ports if not pd.isna(p)]:
            return int(row["MZI"])
    return None


def _is_single_mzi(mzi_id: int) -> bool:
    mzi_id = int(mzi_id)
    return 1 <= mzi_id <= 4 or 33 <= mzi_id <= 36


def _expected_heater_count(mzi_id: int) -> int:
    return 1 if _is_single_mzi(mzi_id) else 2


def _build_mzi_table_from_list(file_path: str = "MZI_list.xlsx") -> dict:
    df = pd.read_excel(file_path)
    table = {}
    for _, row in df.iterrows():
        mzi_id = int(row["MZI"])
        ports = [int(row["H1"])]
        if "H2" in row and not pd.isna(row["H2"]):
            ports.append(int(row["H2"]))
        single = _is_single_mzi(mzi_id)
        ports = ports[: _expected_heater_count(mzi_id)]
        table[str(mzi_id)] = {
            "single": single,
            "ports": ports,
            "heater_R": [None] * len(ports),
            "Ppi": [None] * len(ports),
            "dtheta_Bar": [None] * len(ports),
            "dtheta_Cross": [None] * len(ports),
            "half_power": [None] * len(ports),
        }
    return table


def _load_mzi_table_for_update(path: str = INNER_MZI_TABLE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return _normalize_mzi_table_structure(json.load(f))
    return _normalize_mzi_table_structure(_build_mzi_table_from_list())


def _ensure_list_field(entry: dict, key: str, length: int, fill_value=None) -> list:
    values = entry.get(key)
    if not isinstance(values, list):
        values = []
    while len(values) < length:
        values.append(fill_value)
    if len(values) > length:
        values = values[:length]
    entry[key] = values
    return values


def _normalize_mzi_table_structure(table: dict, file_path: str = "MZI_list.xlsx") -> dict:
    for key, entry in table.items():
        try:
            mzi_id = int(key)
        except (TypeError, ValueError):
            continue

        if not isinstance(entry, dict):
            continue

        single = _is_single_mzi(mzi_id)
        expected_len = _expected_heater_count(mzi_id)
        entry["single"] = single

        try:
            expected_ports = get_mzi_h_list(mzi_id, file_path)[:expected_len]
        except Exception:
            expected_ports = [int(p) for p in entry.get("ports", [])[:expected_len]]

        ports = [int(p) for p in entry.get("ports", []) if p is not None]
        if len(ports) < expected_len and expected_ports:
            ports = expected_ports
        entry["ports"] = ports[:expected_len]

        legacy_dtheta = entry.pop("dtheta", None)
        if "dtheta_Bar" not in entry:
            entry["dtheta_Bar"] = [None] * expected_len
            if isinstance(legacy_dtheta, list) and legacy_dtheta:
                entry["dtheta_Bar"][0] = legacy_dtheta[0]
        if "dtheta_Cross" not in entry:
            entry["dtheta_Cross"] = [None] * expected_len
            if isinstance(legacy_dtheta, list) and len(legacy_dtheta) > 1:
                entry["dtheta_Cross"][0] = legacy_dtheta[1]

        _ensure_list_field(entry, "heater_R", expected_len)
        _ensure_list_field(entry, "Ppi", expected_len)
        _ensure_list_field(entry, "dtheta_Bar", expected_len)
        _ensure_list_field(entry, "dtheta_Cross", expected_len)
        _ensure_list_field(entry, "half_power", expected_len)
        for voltage_key in ("dtheta_Bar", "dtheta_Cross", "half_power"):
            entry[voltage_key] = [
                _round_voltage(value) if _is_filled_number(value) else value for value in entry[voltage_key]
            ]

        for optional_key in ("scan_data_path", "fit_params"):
            if optional_key in entry:
                _ensure_list_field(entry, optional_key, expected_len)

        if isinstance(entry.get("fit_params"), list):
            for fit_item in entry["fit_params"]:
                if isinstance(fit_item, dict):
                    for voltage_key in ("bar_voltage_v", "cross_voltage_v", "half_power_voltage_v"):
                        if _is_filled_number(fit_item.get(voltage_key)):
                            fit_item[voltage_key] = _round_voltage(fit_item[voltage_key])
                    fit_item.pop("bar_power_w", None)
                    fit_item.pop("cross_power_w", None)
                    fit_item.pop("half_power_w", None)
                    fit_item.pop("fine_scan", None)

        for obsolete_key in (
            "dtheta_Bar_power_w",
            "dtheta_Cross_power_w",
            "half_power_w",
            "dtheta_upper",
            "dtheta_upper_port",
            "dtheta_upper_power_w",
            "dtheta_downer",
            "dtheta_downer_port",
            "dtheta_downer_power_w",
            "dtheta_power_w",
            "dtheta_voltage_v",
        ):
            entry.pop(obsolete_key, None)

    return table


def _load_mzi_table_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        table = json.load(f)
    if not isinstance(table, dict):
        return {}
    return _normalize_mzi_table_structure(table)


def _is_filled_number(value) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_missing_reference_value(value) -> bool:
    if isinstance(value, dict):
        return not value
    if isinstance(value, str):
        return value == ""
    return not _is_filled_number(value)


def _fill_reference_list_field(primary_entry: dict, fallback_entry: dict, key: str, length: int) -> None:
    primary_values = _ensure_list_field(primary_entry, key, length)
    fallback_values = fallback_entry.get(key, [])
    if not isinstance(fallback_values, list):
        fallback_values = []

    for index in range(min(length, len(fallback_values))):
        if _is_missing_reference_value(primary_values[index]) and not _is_missing_reference_value(fallback_values[index]):
            primary_values[index] = copy.deepcopy(fallback_values[index])


def _merge_mzi_reference_tables(primary_table: dict, fallback_table: dict) -> dict:
    primary_table = _normalize_mzi_table_structure(copy.deepcopy(primary_table or {}))
    fallback_table = _normalize_mzi_table_structure(copy.deepcopy(fallback_table or {}))
    merged = {}

    for key in sorted(set(fallback_table) | set(primary_table), key=lambda item: int(item)):
        try:
            mzi_id = int(key)
        except (TypeError, ValueError):
            continue

        expected_len = _expected_heater_count(mzi_id)
        primary_entry = copy.deepcopy(primary_table.get(key, {}))
        fallback_entry = copy.deepcopy(fallback_table.get(key, {}))

        if not primary_entry:
            merged[key] = fallback_entry
            continue
        if not fallback_entry:
            merged[key] = primary_entry
            continue

        primary_entry["single"] = _is_single_mzi(mzi_id)
        if not primary_entry.get("ports") and fallback_entry.get("ports"):
            primary_entry["ports"] = copy.deepcopy(fallback_entry["ports"])
        _ensure_list_field(primary_entry, "ports", expected_len)

        for list_key in (
            "heater_R",
            "Ppi",
            "dtheta_Bar",
            "dtheta_Cross",
            "half_power",
            "scan_data_path",
            "fit_params",
        ):
            if list_key in primary_entry or list_key in fallback_entry:
                _fill_reference_list_field(primary_entry, fallback_entry, list_key, expected_len)

        merged[key] = primary_entry

    return _normalize_mzi_table_structure(merged)


def _load_mzi_path_reference_table(primary_table: dict = None) -> dict:
    if primary_table is None:
        primary_table = _load_mzi_table_file(INNER_MZI_TABLE_PATH)
    backup_table = _load_mzi_table_file(INNER_MZI_TABLE_BACKUP_PATH)
    return _merge_mzi_reference_tables(primary_table, backup_table)


def _mzi_table_entry_is_complete(mzi_id: int, table: dict) -> bool:
    key = str(int(mzi_id))
    if key not in table or not isinstance(table[key], dict):
        return False

    entry = _normalize_mzi_table_structure({key: table[key]})[key]
    expected_len = _expected_heater_count(int(mzi_id))
    if bool(entry.get("single")) != _is_single_mzi(int(mzi_id)):
        return False
    if len(entry.get("ports", [])) != expected_len:
        return False

    required_fields = ("heater_R", "Ppi", "dtheta_Bar", "dtheta_Cross", "half_power")
    for field in required_fields:
        values = entry.get(field)
        if not isinstance(values, list) or len(values) != expected_len:
            return False
        if not all(_is_filled_number(value) for value in values):
            return False

    return True


def _save_inner_mzi_scan_results(mzi_id: int, scan_results: list, path: str = INNER_MZI_TABLE_PATH):
    if not scan_results:
        raise ValueError(f"No scan results to save for MZI {mzi_id}.")

    table = _load_mzi_table_for_update(path)
    key = str(int(mzi_id))
    entry = table.setdefault(key, {})
    entry["single"] = _is_single_mzi(int(mzi_id))
    expected_len = _expected_heater_count(int(mzi_id))
    ports = get_mzi_h_list(int(mzi_id))[:expected_len]
    entry["ports"] = ports

    heater_R_values = _ensure_list_field(entry, "heater_R", len(ports))
    ppi_values = _ensure_list_field(entry, "Ppi", len(ports))
    fit_values = _ensure_list_field(entry, "fit_params", len(ports))
    dtheta_bar_values = _ensure_list_field(entry, "dtheta_Bar", len(ports))
    dtheta_cross_values = _ensure_list_field(entry, "dtheta_Cross", len(ports))
    half_power_values = _ensure_list_field(entry, "half_power", len(ports))
    scan_data_path_values = _ensure_list_field(entry, "scan_data_path", len(ports))

    for result in scan_results:
        port = int(result["port"])
        if port not in ports:
            raise ValueError(f"Port {port} does not belong to MZI {mzi_id}: {ports}")
        heater_index = ports.index(port)

        heater_R_values[heater_index] = float(result["heater_R"])
        ppi_values[heater_index] = float(result["Ppi"])
        dtheta_bar_values[heater_index] = _round_voltage(result["bar_voltage_v"])
        dtheta_cross_values[heater_index] = _round_voltage(result["cross_voltage_v"])
        half_power_values[heater_index] = _round_voltage(result["half_power_voltage_v"])
        scan_data_path_values[heater_index] = result["data_path"]
        fit_values[heater_index] = {
            "port": port,
            "A": float(result["A"]),
            "w": float(result["w"]),
            "phi": float(result["phi"]),
            "b": float(result["b"]),
            "model": "OP = A * sin(w * P + phi) + b",
            "bar_voltage_v": _round_voltage(result["bar_voltage_v"]),
            "cross_voltage_v": _round_voltage(result["cross_voltage_v"]),
            "half_power_voltage_v": _round_voltage(result["half_power_voltage_v"]),
            "scan_data_path": result["data_path"],
        }

    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    table = _normalize_mzi_table_structure(table)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    return path


def _get_saved_state_voltages(mzi_id: int, table: dict):
    key = str(int(mzi_id))
    if key not in table:
        raise KeyError(f"MZI {mzi_id} not found in MZI table.")
    entry = _normalize_mzi_table_structure({key: copy.deepcopy(table[key])})[key]
    bar_values = entry.get("dtheta_Bar", [])
    cross_values = entry.get("dtheta_Cross", [])
    if bar_values and cross_values and _is_filled_number(bar_values[0]) and _is_filled_number(cross_values[0]):
        return _round_voltage(bar_values[0]), _round_voltage(cross_values[0])

    fit_params = entry.get("fit_params", [])
    if fit_params and isinstance(fit_params[0], dict):
        bar_voltage = fit_params[0].get("bar_voltage_v")
        cross_voltage = fit_params[0].get("cross_voltage_v")
        if _is_filled_number(bar_voltage) and _is_filled_number(cross_voltage):
            return _round_voltage(bar_voltage), _round_voltage(cross_voltage)

    raise ValueError(f"MZI {mzi_id} has incomplete Bar/Cross voltages.")


def _load_mzi_state_table(path: str = INNER_MZI_STATE_TABLE_PATH) -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8-sig") as f:
        raw_table = json.load(f)

    state_table = {}
    for key, values in raw_table.items():
        if isinstance(values, list) and len(values) >= 2:
            state_table[int(key)] = (_round_voltage(values[0]), _round_voltage(values[1]))
    return state_table


def _save_mzi_state_table(state_table: dict, path: str = INNER_MZI_STATE_TABLE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {
        str(int(key)): [_round_voltage(values[0]), _round_voltage(values[1])]
        for key, values in sorted(state_table.items(), key=lambda item: int(item[0]))
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return path


def _apply_known_path_states(path, state, target, state_table: dict, table: dict, file_data: pd.DataFrame):
    for mzi_value, state_value in zip(path, state):
        mzi_id = int(mzi_value)
        if mzi_id == int(target):
            for port in get_mzi_h_list(mzi_id):
                write_port_voltage(port, 0.0, file_data)
            continue

        if mzi_id in state_table:
            bar_voltage, cross_voltage = state_table[mzi_id]
        else:
            bar_voltage, cross_voltage = _get_saved_state_voltages(mzi_id, table)

        if state_value == "B":
            voltage = bar_voltage
        elif state_value == "C":
            voltage = cross_voltage
        else:
            raise ValueError(f"Unsupported path state {state_value!r} for MZI {mzi_id}.")
        write_port_voltage(get_mzi_h_list(mzi_id)[0], voltage, file_data)
        print(f"MZI {mzi_id} set to {state_value} voltage: {voltage}")


def _apply_saved_bar_biases(table: dict, file_data: pd.DataFrame):
    for key in sorted(table, key=lambda item: int(item)):
        entry = table[key]
        ports = entry.get("ports", [])
        if not ports:
            continue
        try:
            bar_voltage, _ = _get_saved_state_voltages(int(key), table)
        except (KeyError, ValueError):
            continue
        write_port_voltage(int(ports[0]), bar_voltage, file_data)
        print(f"MZI {int(key)} Bar preset: port={int(ports[0])}, voltage={bar_voltage}")


def _prescan_initial_path(
    order1,
    N,
    ser,
    pwm,
    measure_time,
    file_data,
    saved_mzi_table: dict,
    path_reference_table: dict = None,
):
    mzi_state_table = _load_mzi_state_table()
    saved_mzi_table = _normalize_mzi_table_structure(saved_mzi_table)
    path_reference_table = _load_mzi_path_reference_table(saved_mzi_table) if path_reference_table is None else path_reference_table

    for target in order1:
        target_id = int(target)
        if _mzi_table_entry_is_complete(target_id, saved_mzi_table):
            bar_voltage, cross_voltage = _get_saved_state_voltages(target_id, saved_mzi_table)
            mzi_state_table[target_id] = (bar_voltage, cross_voltage)
            _save_mzi_state_table(mzi_state_table)
            print(f"Pre-scan MZI {target_id} already exists in {INNER_MZI_TABLE_PATH}, skip scan.")
            continue

        path, input_idx, output_idx, state = find_path(target_id, N)
        for switch_idx in range(N - 1):
            v_off = switch_IN(switch_idx + 1, "OFF", file_data)
            print(f"MZI {switch_idx + 1} OFF voltage:", v_off)
        v_on = switch_IN(input_idx + 1, "ON", file_data)
        print(f"MZI {input_idx + 1} ON voltage:", v_on)

        _apply_known_path_states(path, state, target_id, mzi_state_table, path_reference_table, file_data)

        scan_results = []
        for port in get_mzi_h_list(target_id):
            result = scan_mzi(
                port,
                INNER_SCAN_START_POWER_W,
                INNER_SCAN_END_POWER_W,
                INNER_SCAN_STEP_POWER_W,
                ser,
                pwm,
                measure_time,
                int(output_idx) + 1,
                file_data,
                mzi_id=target_id,
            )
            scan_results.append(result)

        mzi_table_path = _save_inner_mzi_scan_results(target_id, scan_results, INNER_MZI_TABLE_PATH)
        saved_mzi_table = _load_mzi_table_for_update(INNER_MZI_TABLE_PATH)
        path_reference_table = _load_mzi_path_reference_table(saved_mzi_table)
        bar_voltage, cross_voltage = _get_saved_state_voltages(target_id, saved_mzi_table)
        mzi_state_table[target_id] = (bar_voltage, cross_voltage)
        _save_mzi_state_table(mzi_state_table)
        write_port_voltage(get_mzi_h_list(target_id)[0], bar_voltage, file_data)
        cu.upload_voltage(ser, file_data)
        print(f"Pre-scan saved MZI {target_id} to {mzi_table_path}")

    return mzi_state_table, saved_mzi_table, path_reference_table


def _get_extreme_voltages_from_scan(port: int):
    scan = _load_inner_power_scan(port, INNER_SCAN_DATA_DIR)
    voltage = scan["voltage"]
    optical_power = scan["optical_power"]

    max_power = float(np.nanmax(optical_power))
    min_power = float(np.nanmin(optical_power))

    max_candidates = voltage[np.isclose(optical_power, max_power, rtol=1e-9, atol=1e-12)]
    min_candidates = voltage[np.isclose(optical_power, min_power, rtol=1e-9, atol=1e-12)]
    if max_candidates.size == 0 or min_candidates.size == 0:
        raise ValueError(f"Cannot find extrema from scan data for port {port}.")

    v_at_max = float(np.max(max_candidates))
    v_at_min = float(np.max(min_candidates))
    return v_at_max, v_at_min, max_power, min_power


def _scan_records_to_arrays(data):
    if isinstance(data, list) and data and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        required = {"target_power_w", "voltage_v", "optical_power_uW", "current_mA"}
        if not required.issubset(df.columns):
            raise ValueError(f"Scan records missing columns: {sorted(required - set(df.columns))}")
        electrical_power = df["target_power_w"].to_numpy(dtype=float)
        voltage = df["voltage_v"].to_numpy(dtype=float)
        optical_power = df["optical_power_uW"].to_numpy(dtype=float)
        current_ma = df["current_mA"].to_numpy(dtype=float)
        measured_electrical_power = (
            df["measured_power_w"].to_numpy(dtype=float)
            if "measured_power_w" in df.columns
            else voltage * current_ma * 1e-3
        )
        mask = (
            np.isfinite(voltage)
            & np.isfinite(optical_power)
            & np.isfinite(current_ma)
            & np.isfinite(electrical_power)
            & np.isfinite(measured_electrical_power)
        )
        if np.count_nonzero(mask) < 4:
            raise ValueError("Not enough valid scan samples.")
        order = np.argsort(electrical_power[mask])
        return {
            "voltage": voltage[mask][order],
            "optical_power": optical_power[mask][order],
            "current_ma": current_ma[mask][order],
            "electrical_power": electrical_power[mask][order],
            "measured_electrical_power": measured_electrical_power[mask][order],
        }

    scan_data = np.asarray(data, dtype=float)
    if scan_data.ndim != 2 or scan_data.shape[1] < 3:
        raise ValueError("Scan data must contain voltage, optical power, and current columns.")

    if scan_data.shape[1] >= 5:
        electrical_power = scan_data[:, 0]
        voltage = scan_data[:, 1]
        optical_power = scan_data[:, 2]
        current_ma = scan_data[:, 3]
        measured_electrical_power = scan_data[:, 4]
    else:
        voltage = scan_data[:, 0]
        optical_power = scan_data[:, 1]
        current_ma = scan_data[:, 2]
        measured_electrical_power = voltage * current_ma * 1e-3
        electrical_power = measured_electrical_power
    mask = (
        np.isfinite(voltage)
        & np.isfinite(optical_power)
        & np.isfinite(current_ma)
        & np.isfinite(electrical_power)
        & np.isfinite(measured_electrical_power)
    )

    voltage = voltage[mask]
    optical_power = optical_power[mask]
    current_ma = current_ma[mask]
    electrical_power = electrical_power[mask]
    measured_electrical_power = measured_electrical_power[mask]
    if voltage.size < 4:
        raise ValueError("Not enough valid scan samples.")

    order = np.argsort(electrical_power)
    return {
        "voltage": voltage[order],
        "optical_power": optical_power[order],
        "current_ma": current_ma[order],
        "electrical_power": electrical_power[order],
        "measured_electrical_power": measured_electrical_power[order],
    }


def _get_fine_scan_voltages_from_coarse_data(coarse_data):
    coarse_scan = _scan_records_to_arrays(coarse_data)
    A, w, phi, b = _fit_inner_power_curve(coarse_scan["electrical_power"], coarse_scan["optical_power"])

    power_min = max(0.0, float(np.min(coarse_scan["electrical_power"])))
    power_max = float(np.max(coarse_scan["electrical_power"]))
    special_powers = []
    special_powers.extend(_solve_model_points_in_range(np.pi / 2, w, phi, power_min, power_max))
    special_powers.extend(_solve_model_points_in_range(-np.pi / 2, w, phi, power_min, power_max))
    special_powers.extend(_solve_model_points_in_range(0.0, w, phi, power_min, power_max))
    special_powers.extend(_solve_model_points_in_range(np.pi, w, phi, power_min, power_max))

    special_voltages = _power_to_voltage(
        coarse_scan["electrical_power"],
        coarse_scan["voltage"],
        special_powers,
    )
    voltage_min = float(np.min(coarse_scan["voltage"]))
    voltage_max = float(np.max(coarse_scan["voltage"]))
    fine_window_v = 0.05
    fine_step_v = 0.001
    intervals = []
    for voltage in special_voltages:
        left = max(voltage_min, float(voltage) - fine_window_v)
        right = min(voltage_max, float(voltage) + fine_window_v)
        if right >= left:
            intervals.append((left, right))

    merged_intervals = []
    for left, right in sorted(intervals):
        if not merged_intervals or left > merged_intervals[-1][1] + 1e-9:
            merged_intervals.append([left, right])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], right)

    fine_points = set()
    coarse_points = set(np.round(coarse_scan["voltage"], 3).tolist())
    for left, right in merged_intervals:
        fine_values = np.arange(left, right + fine_step_v * 0.5, fine_step_v)
        fine_points.update(np.round(fine_values, 3).tolist())
    fine_points.difference_update(coarse_points)

    fit_info = {
        "A": float(A),
        "w": float(w),
        "phi": float(phi),
        "b": float(b),
        "special_voltages": [round(float(v), 3) for v in sorted(special_voltages)],
        "fine_windows": [[round(left, 3), round(right, 3)] for left, right in merged_intervals],
    }
    return np.array(sorted(fine_points), dtype=float), fit_info


def _scan_mzi_impl(
    port,
    start_voltage,
    end_voltage,
    step,
    ser,
    pwm,
    measure_time,
    out_num,
    file_path_df,
    mzi_id=None,
):
    del start_voltage, end_voltage, step
    print("=" * 50)
    print(f"Scanning port {port}")

    os.makedirs(INNER_SCAN_DATA_DIR, exist_ok=True)

    data = []
    arm_ports = get_mzi_h_list(int(mzi_id)) if mzi_id is not None else [int(port)]
    arm_index = arm_ports.index(int(port)) if int(port) in arm_ports else 0
    arm_name = "upper" if arm_index == 0 else "lower"
    scan_stem = _inner_scan_stem(mzi_id, arm_index, port)
    data_savepath = _inner_scan_data_path(mzi_id, arm_index, port)
    print(f"Scan data file stem: {scan_stem}")

    if mzi_id is not None:
        for target_port in arm_ports:
            if int(target_port) != int(port):
                write_port_voltage(int(target_port), 0.0, file_path_df)

    R, dI = cu.get_R(ser, port, file_path_df)
    print(f"Port {port} heater R={R} Ohm, dI={dI * 1e3} mA")
    if not np.isfinite(R) or R <= 0:
        raise ValueError(f"Invalid heater resistance for port {port}: {R}")

    target_power_values = np.round(
        np.arange(
            INNER_SCAN_START_POWER_W,
            INNER_SCAN_END_POWER_W + INNER_SCAN_STEP_POWER_W * 0.5,
            INNER_SCAN_STEP_POWER_W,
        ),
        6,
    )
    print(f"Coarse power scan points: {len(target_power_values)}")

    def scan_power_points(power_values, stage_name):
        stage_records = []
        print(f"{stage_name} power scan points: {len(power_values)}")
        for target_power in power_values:
            v = _voltage_from_resistance_power(float(target_power), R)
            v = round(v, 3)
            try:
                file_path_df.at[port - 1, 0] = v
            except Exception as e:
                print(f"Update voltage table failed: {e}")
                continue

            print(
                f"\n{stage_name} power scan, port {port} "
                f"target P: {float(target_power):.6f} W, set voltage: {v:.3f} V"
            )
            c = np.nan
            read_current_limit = 0
            while True:
                cu.upload_voltage(ser, file_path_df)
                c = cu.read_current_port(ser, port) - dI * 1e3
                if v == 0:
                    print(Fore.YELLOW + "Voltage is 0 V, skip current validation")
                    break
                expected_current_ma = v / R * 1000
                if 0.9 * expected_current_ma < c < 1.1 * expected_current_ma:
                    print(Fore.GREEN + f"Current OK, I={c} mA")
                    break
                if read_current_limit >= 5:
                    print(Fore.RED + f"Current abnormal after retries, I={c} mA, keep this point")
                    break

                print(Fore.RED + f"Current abnormal, I={c} mA, retry upload")
                read_current_limit += 1

            time.sleep(measure_time)

            power_str_list = cu.read_pow(pwm)
            try:
                optical_power_value = float(power_str_list[int(out_num) - 1]) * 1e6
            except (ValueError, IndexError) as e:
                print(f"Read optical power failed: {e}")
                optical_power_value = 0

            measured_power = v * c * 1e-3 if np.isfinite(c) else np.nan
            record = {
                "mzi_id": int(mzi_id) if mzi_id is not None else np.nan,
                "arm_index": int(arm_index),
                "arm_name": arm_name,
                "port": int(port),
                "scan_stage": stage_name,
                "target_power_w": float(target_power),
                "voltage_v": float(v),
                "optical_power_uW": float(optical_power_value),
                "current_mA": float(c) if np.isfinite(c) else np.nan,
                "measured_power_w": float(measured_power) if np.isfinite(measured_power) else np.nan,
            }
            stage_records.append(record)
            print(f"Optical power: {optical_power_value} uW")
        return stage_records

    coarse_data = scan_power_points(target_power_values, "coarse")
    data.extend(coarse_data)

    if not coarse_data:
        raise RuntimeError(f"No coarse scan data collected for port {port}")

    coarse_scan = _scan_records_to_arrays(coarse_data)
    coarse_A, coarse_w, coarse_phi, coarse_b = _fit_inner_power_curve(
        coarse_scan["electrical_power"],
        coarse_scan["optical_power"],
    )
    fine_power_values, fine_scan_info = _get_fine_scan_powers_from_fit(
        coarse_A,
        coarse_w,
        coarse_phi,
        INNER_SCAN_START_POWER_W,
        INNER_SCAN_END_POWER_W,
        coarse_scan["electrical_power"],
    )
    print(f"Coarse fit: A={coarse_A:.6f}, w={coarse_w:.6f}, " f"phi={coarse_phi:.6f}, b={coarse_b:.6f}")
    print(f"Fine scan windows: {fine_scan_info['fine_windows']}")
    if fine_power_values.size > 0:
        data.extend(scan_power_points(fine_power_values, "fine"))
    else:
        print(Fore.YELLOW + f"No fine scan points found for port {port}")

    if not data:
        raise RuntimeError(f"No scan data collected for port {port}")

    data = sorted(data, key=lambda row: (row["target_power_w"], row["scan_stage"]))
    scan_df = pd.DataFrame(data)
    scan_df["voltage_v"] = scan_df["voltage_v"].map(lambda value: f"{_round_voltage(value):.3f}")
    scan_df.to_csv(data_savepath, index=False, float_format="%.12f")

    scan = _load_inner_power_scan(data_savepath, INNER_SCAN_DATA_DIR)
    A, w, phi, b = _fit_inner_power_curve(scan["electrical_power"], scan["optical_power"])
    Ppi = float(np.pi / w)
    dtheta = _get_bar_cross_half_from_fit(
        A,
        w,
        phi,
        R,
        INNER_SCAN_START_POWER_W,
        INNER_SCAN_END_POWER_W,
    )
    _save_inner_fit_plot(
        scan_stem,
        scan["electrical_power"],
        scan["optical_power"],
        A,
        w,
        phi,
        b,
        title=f"MZI {int(mzi_id)} {arm_name} Power Fit (Port {int(port)})" if mzi_id is not None else None,
    )
    _update_inner_fit_params(port, A, w, phi, b)
    print(
        f"Post-scan fit: A={A:.6f}, w={w:.6f}, phi={phi:.6f}, "
        f"b={b:.6f}, Ppi={Ppi:.6f} W, "
        f"Bar V={dtheta['bar_voltage_v']:.6f}, Cross V={dtheta['cross_voltage_v']:.6f}, "
        f"Half V={dtheta['half_power_voltage_v']:.6f}"
    )

    return {
        "mzi_id": int(mzi_id) if mzi_id is not None else None,
        "arm_index": int(arm_index),
        "arm_name": arm_name,
        "port": int(port),
        "heater_R": float(R),
        "Ppi": float(Ppi),
        "A": float(A),
        "w": float(w),
        "phi": float(phi),
        "b": float(b),
        "bar_voltage_v": _round_voltage(dtheta["bar_voltage_v"]),
        "bar_power_w": float(dtheta["bar_power_w"]),
        "cross_voltage_v": _round_voltage(dtheta["cross_voltage_v"]),
        "cross_power_w": float(dtheta["cross_power_w"]),
        "half_power_voltage_v": _round_voltage(dtheta["half_power_voltage_v"]),
        "half_power_w": float(dtheta["half_power_w"]),
        "data_path": data_savepath,
        "fine_scan_info": fine_scan_info,
    }


def scan_mzi(
    port,
    start_voltage,
    end_voltage,
    step,
    ser,
    pwm,
    measure_time,
    out_num,
    file_path_df,
    mzi_id=None,
):
    try:
        return _scan_mzi_impl(
            port,
            start_voltage,
            end_voltage,
            step,
            ser,
            pwm,
            measure_time,
            out_num,
            file_path_df,
            mzi_id=mzi_id,
        )
    finally:
        try:
            reset_heater_voltage(port, ser, file_path_df)
        except Exception as e:
            print(Fore.RED + f"Failed to reset port {int(port)} heater voltage to 0 V: {e}")


if __name__ == "__main__":
    N = 9

    net_matrix = du.Clements_matrix(N)
    print(net_matrix)
    order1, order2 = get_cali_order(N)

    mzi_table = load_mzi_table()
    working_data = cu.generate_working_data()
    SER_ADDRESS = "COM3"
    OPM1_ADDRESS = "TCPIP0::192.168.0.5::inst0::INSTR"
    OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
    opm1 = cu.open_VISA_connection(OPM1_ADDRESS)
    opm2 = cu.open_VISA_connection(OPM2_ADDRESS)
    mcv = cu.open_ser_connection(SER_ADDRESS)

    start_v = 0
    end_v = 5
    step_v = 0.1
    measure_time = 1
    output = 1

    for i in range(N - 1):
        v_off = switch_IN(i + 1, "OFF", working_data)
        print(f"MZI {i+1} OFF voltage:", v_off)
    v_on = switch_IN(1, "ON", working_data)
    print(f"MZI 1 ON voltage:", v_on)

    path_reference_table = _load_mzi_path_reference_table(mzi_table)
    _apply_saved_bar_biases(path_reference_table, working_data)
    mzi_state_table, saved_mzi_table, path_reference_table = _prescan_initial_path(
        order1,
        N,
        mcv,
        opm2,
        measure_time,
        working_data,
        mzi_table,
        path_reference_table,
    )

    for target in order2:
        if _mzi_table_entry_is_complete(int(target), saved_mzi_table):
            bar_voltage, cross_voltage = _get_saved_state_voltages(int(target), saved_mzi_table)
            mzi_state_table[int(target)] = (bar_voltage, cross_voltage)
            _save_mzi_state_table(mzi_state_table)
            print(f"MZI {int(target)} already exists in {INNER_MZI_TABLE_PATH}, skip scan.")
            continue

        target_ports = get_mzi_h_list(target)

        path, input, output, state = find_path(target, N)
        for i in range(N - 1):
            v_off = switch_IN(i + 1, "OFF", working_data)
            print(f"MZI {i+1} OFF voltage:", v_off)
        v_on = switch_IN(input + 1, "ON", working_data)
        print(f"MZI {input + 1} ON voltage:", v_on)

        _apply_known_path_states(path, state, target, mzi_state_table, path_reference_table, working_data)

        mzi_scan_results = []
        for port in target_ports:
            heater_idx = target_ports.index(port)
            result = scan_mzi(
                port,
                start_v,
                end_v,
                step_v,
                mcv,
                opm2,
                measure_time,
                output + 1,
                working_data,
                mzi_id=int(target),
            )
            mzi_scan_results.append(result)

            if heater_idx != 0:
                continue

            print(f"Model Bar voltage: {result['bar_voltage_v']} V")
            print(f"Model Cross voltage: {result['cross_voltage_v']} V")
            print(f"Model half-power voltage: {result['half_power_voltage_v']} V")

            mzi_state_table[int(target)] = (result["bar_voltage_v"], result["cross_voltage_v"])

            _save_mzi_state_table(mzi_state_table)

        mzi_table_path = _save_inner_mzi_scan_results(int(target), mzi_scan_results, INNER_MZI_TABLE_PATH)
        saved_mzi_table = _load_mzi_table_for_update(INNER_MZI_TABLE_PATH)
        path_reference_table = _load_mzi_path_reference_table(saved_mzi_table)
        print(f"Saved MZI {int(target)} scan results to {mzi_table_path}")

    print(mzi_state_table)
