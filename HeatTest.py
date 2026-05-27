import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.communication as cu
import utils.AllDecompositionUtils as du
import serial
import time

SCANDATA_DIR = "Scandata"
SCANDATA_BACKUP_DIR = "Scandata-backup-04"
HEAT_TEST_DIR = os.path.join(SCANDATA_DIR, "heat_test")
INNER_SCAN_DATA_DIR = os.path.join(SCANDATA_DIR, "inner_cali_powerdata")
INNER_FIT_PARAMS_PATH = os.path.join(SCANDATA_DIR, "inner_fit_params.json")
MZI_TABLE_PATH = os.path.join(SCANDATA_DIR, "MZI_table.json")
MZI_TABLE_BACKUP_PATH = os.path.join(SCANDATA_BACKUP_DIR, "MZI_table.json")
MZI_STATE_TABLE_PATH = os.path.join(SCANDATA_DIR, "mzi_stata_table.json")
MZI_STATE_TABLE_BACKUP_PATH = os.path.join(SCANDATA_BACKUP_DIR, "mzi_stata_table.json")


def _resolve_existing_file(path: str, *fallback_paths: str) -> str:
    for candidate in (path, *fallback_paths):
        if candidate and os.path.exists(candidate):
            return candidate
    checked = ", ".join([path, *fallback_paths])
    raise FileNotFoundError(f"Cannot find any of: {checked}")


def _load_inner_power_scan(port: int, data_dir: str = INNER_SCAN_DATA_DIR) -> dict:
    filename = os.path.join(data_dir, f"{int(port)}.txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find data file: {filename}")

    df = pd.read_csv(filename)
    if df.shape[1] < 3:
        raise ValueError(f"Expect at least three columns in {filename}")

    voltage = df.iloc[:, 0].to_numpy(dtype=float)
    optical_power = df.iloc[:, 1].to_numpy(dtype=float)
    current_ma = df.iloc[:, 2].to_numpy(dtype=float)
    current_amp = current_ma * 1e-3
    electrical_power = voltage * current_amp

    mask = np.isfinite(voltage) & np.isfinite(optical_power) & np.isfinite(current_ma) & np.isfinite(electrical_power)
    voltage = voltage[mask]
    optical_power = optical_power[mask]
    current_ma = current_ma[mask]
    electrical_power = electrical_power[mask]

    if voltage.size < 4:
        raise ValueError(f"Not enough valid samples in {filename}")

    order = np.argsort(voltage)
    return {
        "path": filename,
        "voltage": voltage[order],
        "optical_power": optical_power[order],
        "current_ma": current_ma[order],
        "electrical_power": electrical_power[order],
    }


def load_mzi_table(path: str = MZI_TABLE_PATH) -> dict:
    fallback = MZI_TABLE_BACKUP_PATH if path == MZI_TABLE_PATH else ""
    table_path = _resolve_existing_file(path, fallback)
    with open(table_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def get_mzi_lower_arm_port(mzi_id, file_path="MZI_list.xlsx") -> int:
    ports = get_mzi_h_list(mzi_id, file_path)
    if len(ports) < 2:
        raise ValueError(f"MZI {mzi_id} does not have a lower-arm heater in {file_path}")
    return int(ports[1])


def get_mzi_upper_arm_port(mzi_id, file_path="MZI_list.xlsx") -> int:
    ports = get_mzi_h_list(mzi_id, file_path)
    return int(ports[0])


def get_mzi_arm_resistance(mzi_id: int, arm_index: int, mzi_table: dict | None = None) -> float:
    if mzi_table is None:
        mzi_table = load_mzi_table()
    entry = mzi_table[str(int(mzi_id))]
    resistance = float(entry["heater_R"][arm_index])
    if resistance <= 0:
        raise ValueError(f"MZI {mzi_id} arm {arm_index} has invalid resistance: {resistance}")
    return resistance


def voltage_for_heater_power(power_w: float, resistance_ohm: float) -> float:
    if power_w < 0:
        raise ValueError("power_w must be non-negative.")
    if resistance_ohm <= 0:
        raise ValueError("resistance_ohm must be positive.")
    return float(np.sqrt(power_w * resistance_ohm))


def set_mzi_arm_power(
    mzi_id: int,
    arm_index: int,
    power_w: float,
    file_data: pd.DataFrame,
    mzi_table: dict | None = None,
) -> dict:
    if arm_index not in {0, 1}:
        raise ValueError("arm_index must be 0 for upper arm or 1 for lower arm.")
    if mzi_table is None:
        mzi_table = load_mzi_table()

    ports = get_mzi_h_list(mzi_id)
    if arm_index >= len(ports):
        raise ValueError(f"MZI {mzi_id} does not have arm index {arm_index}.")

    port = int(ports[arm_index])
    resistance = get_mzi_arm_resistance(mzi_id, arm_index, mzi_table)
    voltage = voltage_for_heater_power(power_w, resistance)
    write_port_voltage(port, voltage, file_data)
    return {
        "mzi": int(mzi_id),
        "arm_index": int(arm_index),
        "arm_name": "upper" if arm_index == 0 else "lower",
        "port": port,
        "power_W": float(power_w),
        "power_mW": float(power_w * 1e3),
        "resistance_ohm": float(resistance),
        "voltage_V": float(voltage),
    }


def find_latest_file(pattern: str, root: str = HEAT_TEST_DIR) -> str:
    matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root!r}")
    return max(matches, key=os.path.getmtime)


def estimate_half_power_upper_voltage_from_scan(scan_path: str, side: str = "left") -> dict:
    df = pd.read_csv(scan_path)
    required_columns = {"upper_voltage_V", "optical_power_uW"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {scan_path}: {sorted(missing)}")

    curve = (
        df[["upper_voltage_V", "optical_power_uW"]]
        .dropna()
        .groupby("upper_voltage_V", as_index=False)["optical_power_uW"]
        .mean()
        .sort_values("upper_voltage_V")
    )
    voltage = curve["upper_voltage_V"].to_numpy(dtype=float)
    power = curve["optical_power_uW"].to_numpy(dtype=float)
    if voltage.size < 2:
        raise ValueError(f"Need at least two voltage points in {scan_path}")

    min_idx = int(np.argmin(power))
    max_idx = int(np.argmax(power))
    min_voltage = float(voltage[min_idx])
    min_power = float(power[min_idx])
    max_power = float(power[max_idx])
    half_power = 0.5 * (min_power + max_power)

    crossings = []
    for i in range(voltage.size - 1):
        v1, v2 = float(voltage[i]), float(voltage[i + 1])
        p1, p2 = float(power[i]), float(power[i + 1])
        if p1 == half_power:
            crossings.append(v1)
        if (p1 - half_power) * (p2 - half_power) < 0:
            crossing = v1 + (half_power - p1) * (v2 - v1) / (p2 - p1)
            crossings.append(float(crossing))
    if power[-1] == half_power:
        crossings.append(float(voltage[-1]))
    if not crossings:
        raise RuntimeError(f"No half-power crossing found in {scan_path}")

    if side == "left":
        candidates = [v for v in crossings if v <= min_voltage]
        selected = max(candidates) if candidates else min(crossings, key=lambda v: abs(v - min_voltage))
    elif side == "right":
        candidates = [v for v in crossings if v >= min_voltage]
        selected = min(candidates) if candidates else min(crossings, key=lambda v: abs(v - min_voltage))
    elif side == "closest":
        selected = min(crossings, key=lambda v: abs(v - min_voltage))
    else:
        raise ValueError("side must be 'left', 'right', or 'closest'")

    return {
        "scan_path": scan_path,
        "side": side,
        "half_power_voltage_V": float(selected),
        "half_power_uW": float(half_power),
        "min_voltage_V": min_voltage,
        "min_power_uW": min_power,
        "max_voltage_V": float(voltage[max_idx]),
        "max_power_uW": max_power,
        "crossings_V": [float(v) for v in crossings],
    }


def write_port_voltage(port: int, voltage: float, file_data: pd.DataFrame) -> None:
    """
    Write voltage into file_data at the row corresponding to PORT (1-based).
    """
    port_idx = port - 1
    if port_idx < 0 or port_idx >= len(file_data):
        raise IndexError(f"PORT {port} is out of range for the provided file_data.")
    file_data.iloc[port_idx, 0] = voltage


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


def fit_p_to_op(index: int, data_dir: str = "powerdata"):
    """
    Fit OP = A*sin(w*P + phi) + b using <data_dir>/<index>.txt.
    Returns (A, w, phi, b, p_elec, op).
    """
    filename = os.path.join(data_dir, f"{index}.txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find data file: {filename}")

    df = pd.read_csv(filename)
    if df.shape[1] < 3:
        raise ValueError(f"Expect at least three columns in {filename}")

    op = df.iloc[:, 1].to_numpy(dtype=float)  # Optical power OP (y-axis)
    current_amp = df.iloc[:, 2].to_numpy(dtype=float) * 1e-3  # current in A
    voltage = df.iloc[:, 0].to_numpy(dtype=float)
    p_elec = voltage * current_amp  # Electrical power P in watts (x-axis)

    def sin_model(x, A, w, phi, b):
        return A * np.sin(w * x + phi) + b

    amplitude_guess = 0.5 * (op.max() - op.min())
    offset_guess = op.mean()
    w_guess = 2 * np.pi / (p_elec.max() - p_elec.min() + 1e-9)
    phi_guess = 0.0

    popt, _ = curve_fit(
        sin_model,
        p_elec,
        op,
        p0=[amplitude_guess, w_guess, phi_guess, offset_guess],
        maxfev=10000,
    )
    return (*popt, p_elec, op)


def p_from_op(A: float, w: float, phi: float, b: float, op_value: float, port: int):
    """
    Given OP = A*sin(w*P + phi) + b:
    - return all P in range that match the OP
    - return P where OP is max (sin = 1) in range
    - return P where OP is min (sin = -1) in range
    """
    if A == 0 or w == 0:
        raise ValueError("A and w must be non-zero.")

    filename = os.path.join("powerdata", f"{port}.txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find data file: {filename}")

    df = pd.read_csv(filename)
    if df.shape[1] < 3:
        raise ValueError(f"Expect at least three columns in {filename}")

    voltage = df.iloc[:, 0].to_numpy(dtype=float)
    current_amp = df.iloc[:, 2].to_numpy(dtype=float) * 1e-3
    p_elec = voltage * current_amp
    if p_elec.size == 0:
        raise ValueError(f"No power data in {filename}")

    p_min = 0.0
    p_max = float(np.max(p_elec))
    if p_max <= 0:
        raise ValueError(f"Invalid max power from {filename}")

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
    for k in range(-5, 6):
        p1 = (base - phi + 2 * np.pi * k) / w
        if p_min <= p1 <= p_max:
            p_solutions.append(p1)
        p2 = (np.pi - base - phi + 2 * np.pi * k) / w
        if p_min <= p2 <= p_max:
            p_solutions.append(p2)
    p_solutions = dedup(p_solutions)

    # Max/min points depend on the sign of A
    max_phase = np.pi / 2
    min_phase = -np.pi / 2
    if A < 0:
        max_phase, min_phase = min_phase, max_phase

    # Max points
    p_at_max = []
    for k in range(-5, 6):
        p = (max_phase - phi + 2 * np.pi * k) / w
        if p_min <= p <= p_max:
            p_at_max.append(p)
    p_at_max = dedup(p_at_max)

    # Min points
    p_at_min = []
    for k in range(-5, 6):
        p = (min_phase - phi + 2 * np.pi * k) / w
        if p_min <= p <= p_max:
            p_at_min.append(p)
    p_at_min = dedup(p_at_min)

    return p_solutions, p_at_max, p_at_min


def voltage_from_power(port: int, power_w: float, data_dir: str = INNER_SCAN_DATA_DIR) -> float:
    """
    Given a PORT and electrical power (W), interpolate voltage from
    <data_dir>/{PORT}.txt and return the voltage (rounded to 3 decimals).
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
            input_idx = cx
            break
        if cx == 0 and cy == 0:
            input_idx = 0
            break
        if cx > 0:
            cx -= 1
        if cy > 0:
            cy -= 1
        if M[cx][cy] != 0:
            path.insert(0, (cx, cy))
    while True:
        if by == N - 1 and bx != 0:
            output_idx = bx
            break
        if bx == 0 and by == N - 1:
            output_idx = 0
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
    return PATH, input_idx, output_idx, state


def load_mzi_state_table(path: str = MZI_STATE_TABLE_PATH) -> dict:
    fallback = MZI_STATE_TABLE_BACKUP_PATH if path == MZI_STATE_TABLE_PATH else ""
    table_path = _resolve_existing_file(path, fallback)
    with open(table_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(target): (values[0], values[1]) for target, values in raw.items()}


def set_all_input_switches(N: int, state: str, file_data: pd.DataFrame) -> dict:
    voltages = {}
    for i in range(N - 1):
        mzi_index = i + 1
        voltages[mzi_index] = switch_IN(mzi_index, state, file_data)
    return voltages


def apply_dtheta_biases(N: int, file_data: pd.DataFrame, mzi_table: dict | None = None) -> dict:
    if mzi_table is None:
        mzi_table = load_mzi_table()

    applied = {}
    for i in range(N * (N - 1) // 2):
        mzi_index = i + 1
        port = int(mzi_table[str(mzi_index)]["ports"][0])
        voltage = float(mzi_table[str(mzi_index)]["dtheta"][0])
        write_port_voltage(port, voltage, file_data)
        applied[mzi_index] = {"port": port, "voltage": voltage}
    return applied


def apply_target_path_voltages(
    target: int,
    N: int,
    file_data: pd.DataFrame,
    mzi_state_table: dict | None = None,
) -> dict:
    if mzi_state_table is None:
        mzi_state_table = load_mzi_state_table()

    path, input_idx, output_idx, state = find_path(target, N)
    set_all_input_switches(N, "OFF", file_data)
    input_voltage = switch_IN(input_idx + 1, "ON", file_data)

    applied_path = {}
    for mzi_value, route_state in zip(path, state):
        mzi_index = int(mzi_value)
        if mzi_index == int(target):
            continue
        if route_state == "B":
            voltage = float(mzi_state_table[mzi_index][0])
        else:
            voltage = float(mzi_state_table[mzi_index][1])
        port = int(get_mzi_h_list(mzi_index)[0])
        write_port_voltage(port, voltage, file_data)
        applied_path[mzi_index] = {"state": route_state, "port": port, "voltage": voltage}

    return {
        "target": int(target),
        "path": [int(x) for x in path],
        "input": int(input_idx),
        "output": int(output_idx),
        "state": list(state),
        "input_voltage": input_voltage,
        "applied_path": applied_path,
    }


def configure_network_for_target(
    target: int,
    N: int,
    file_data: pd.DataFrame,
    mzi_table: dict | None = None,
    mzi_state_table: dict | None = None,
    apply_dtheta: bool = True,
    ser=None,
    upload: bool = False,
) -> dict:
    result = {}
    if apply_dtheta:
        result["dtheta"] = apply_dtheta_biases(N, file_data, mzi_table)
    result["target_path"] = apply_target_path_voltages(target, N, file_data, mzi_state_table)
    if upload:
        if ser is None:
            raise ValueError("ser is required when upload=True")
        cu.upload_voltage(ser, file_data)
        result["uploaded"] = True
    else:
        result["uploaded"] = False
    return result


def read_output_power_uw(pwm, out_num: int) -> float:
    power_str_list = cu.read_pow(pwm)
    try:
        return float(power_str_list[int(out_num) - 1]) * 1e6
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Failed to read optical power from output {out_num}") from exc


def _normalize_phase(phi: float) -> float:
    return float((phi + np.pi) % (2 * np.pi) - np.pi)


def _mzi_power_model(power_w, power_dc, power_amp, k, theta_offset):
    return power_dc + power_amp * np.cos(k * power_w + theta_offset)


def fit_lower_arm_phase_model(scan_df: pd.DataFrame) -> dict:
    required_columns = {"voltage_V", "current_mA", "optical_power_uW"}
    missing = required_columns - set(scan_df.columns)
    if missing:
        raise ValueError(f"Missing columns for phase model fit: {sorted(missing)}")

    voltage = scan_df["voltage_V"].to_numpy(dtype=float)
    current_ma = scan_df["current_mA"].to_numpy(dtype=float)
    optical_power = scan_df["optical_power_uW"].to_numpy(dtype=float)
    electrical_power = voltage * current_ma * 1e-3

    mask = np.isfinite(electrical_power) & np.isfinite(optical_power) & (electrical_power >= 0)
    x = electrical_power[mask]
    y = optical_power[mask]
    if x.size < 5:
        raise ValueError("Need at least five valid scan points to fit the lower-arm phase model.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    power_dc_guess = float(np.mean(y))
    power_amp_guess = max(0.5 * (y_max - y_min), 1e-9)
    x_span = max(float(np.max(x) - np.min(x)), 1e-12)

    best = None
    for k_guess in np.linspace(0.1 * np.pi / x_span, 6.0 * np.pi / x_span, 800):
        design = np.column_stack((np.cos(k_guess * x), np.sin(k_guess * x), np.ones_like(x)))
        try:
            alpha, beta, power_dc = np.linalg.lstsq(design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        fitted = design @ np.array([alpha, beta, power_dc])
        error = float(np.sum((y - fitted) ** 2))
        if best is None or error < best["error"]:
            power_amp = float(np.hypot(alpha, beta))
            theta_offset = _normalize_phase(float(np.arctan2(-beta, alpha)))
            best = {
                "power_dc": float(power_dc),
                "power_amp": power_amp,
                "k": float(k_guess),
                "theta_offset": theta_offset,
                "error": error,
            }

    if best is None:
        raise RuntimeError("Failed to estimate initial parameters for lower-arm phase model.")

    try:
        popt, _ = curve_fit(
            _mzi_power_model,
            x,
            y,
            p0=[best["power_dc"], max(best["power_amp"], power_amp_guess), best["k"], best["theta_offset"]],
            bounds=([0.0, 0.0, 0.0, -4 * np.pi], [np.inf, np.inf, np.inf, 4 * np.pi]),
            maxfev=20000,
        )
        power_dc, power_amp, k, theta_offset = popt
    except (RuntimeError, ValueError):
        power_dc = best["power_dc"]
        power_amp = best["power_amp"]
        k = best["k"]
        theta_offset = best["theta_offset"]

    theta_offset = _normalize_phase(float(theta_offset))
    fitted_power = _mzi_power_model(x, power_dc, power_amp, k, theta_offset)
    residual = y - fitted_power
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "electrical_power_W": x,
        "optical_power_uW": y,
        "power_dc_uW": float(power_dc),
        "power_amp_uW": float(power_amp),
        "k_rad_per_W": float(k),
        "theta1_minus_theta2_rad": theta_offset,
        "r_squared": float(r_squared),
    }


def build_lower_arm_phase_model_from_scan(
    target: int,
    scan_path: str | None = None,
    save_dir: str = HEAT_TEST_DIR,
    run_label: str = "lower_only",
    display_label: str | None = None,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    file_prefix = f"mzi{target}_{run_label}_lower_arm"
    if display_label is None:
        display_label = run_label
    if scan_path is None:
        scan_path = os.path.join(save_dir, f"{file_prefix}_scan.csv")

    scan_df = pd.read_csv(scan_path)
    fit = fit_lower_arm_phase_model(scan_df)
    p2 = fit["electrical_power_W"]
    theta = fit["k_rad_per_W"] * p2 + fit["theta1_minus_theta2_rad"]
    fitted_power = _mzi_power_model(
        p2,
        fit["power_dc_uW"],
        fit["power_amp_uW"],
        fit["k_rad_per_W"],
        fit["theta1_minus_theta2_rad"],
    )

    model_df = pd.DataFrame(
        {
            "P2_W": p2,
            "theta_rad": theta,
            "theta_mod_2pi_rad": (theta + np.pi) % (2 * np.pi) - np.pi,
            "optical_power_uW": fit["optical_power_uW"],
            "fitted_optical_power_uW": fitted_power,
        }
    )
    model_data_path = os.path.join(save_dir, f"{file_prefix}_phase_model.csv")
    model_df.to_csv(model_data_path, index=False, encoding="utf-8")

    model_info = {
        "target": int(target),
        "run_label": run_label,
        "display_label": display_label,
        "model": "dtheta = k * P2 + (theta1 - theta2)",
        "k_rad_per_W": fit["k_rad_per_W"],
        "theta1_minus_theta2_rad": fit["theta1_minus_theta2_rad"],
        "theta1_minus_theta2_pi_units": fit["theta1_minus_theta2_rad"] / np.pi,
        "power_dc_uW": fit["power_dc_uW"],
        "power_amp_uW": fit["power_amp_uW"],
        "r_squared": fit["r_squared"],
        "scan_path": scan_path,
        "model_data_path": model_data_path,
    }
    model_info_path = os.path.join(save_dir, f"{file_prefix}_phase_model.json")

    p_smooth = np.linspace(float(np.min(p2)), float(np.max(p2)), 800)
    theta_smooth = fit["k_rad_per_W"] * p_smooth + fit["theta1_minus_theta2_rad"]
    power_smooth = _mzi_power_model(
        p_smooth,
        fit["power_dc_uW"],
        fit["power_amp_uW"],
        fit["k_rad_per_W"],
        fit["theta1_minus_theta2_rad"],
    )

    image_path = os.path.join(save_dir, f"{file_prefix}_phase_model.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
    ax1.plot(p2, fit["optical_power_uW"], "o", markersize=4, label="Measured")
    ax1.plot(p_smooth, power_smooth, "-", linewidth=1.8, label="Fit")
    ax1.set_ylabel("Optical power (uW)")
    ax1.set_title(f"MZI {target} Lower-Arm Phase Model ({display_label})")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(p_smooth, theta_smooth, "-", linewidth=1.8, color="tab:green")
    ax2.plot(p2, theta, "o", markersize=4, color="tab:green")
    ax2.set_xlabel("P2 electrical power (W)")
    ax2.set_ylabel("dtheta (rad)")
    ax2.grid(True, alpha=0.35)
    formula = (
        f"dtheta = {fit['k_rad_per_W']:.6g} * P2 "
        f"+ ({fit['theta1_minus_theta2_rad']:.6g})"
    )
    ax2.text(0.02, 0.95, formula, transform=ax2.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    model_info["image_path"] = image_path
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    return {
        "model_info_path": model_info_path,
        "model_data_path": model_data_path,
        "image_path": image_path,
        "model_info": model_info,
    }


def plot_combined_lower_arm_phase_models(
    target: int,
    model_results: list[dict],
    save_dir: str = HEAT_TEST_DIR,
    output_label: str = "lower_only_vs_upper_16mW",
) -> dict:
    if len(model_results) < 2:
        raise ValueError("At least two model results are required for a combined phase-model plot.")

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    summary = []

    for result in model_results:
        model_info = result.get("model_info")
        if model_info is None:
            model_info_path = result["model_info_path"]
            with open(model_info_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)

        model_data_path = model_info.get("model_data_path", result.get("model_data_path"))
        model_df = pd.read_csv(model_data_path)
        p2 = model_df["P2_W"].to_numpy(dtype=float)
        p_min = float(np.nanmin(p2))
        p_max = float(np.nanmax(p2))
        p_smooth = np.linspace(p_min, p_max, 800)

        k = float(model_info["k_rad_per_W"])
        theta0 = float(model_info["theta1_minus_theta2_rad"])
        theta_smooth = k * p_smooth + theta0
        label = model_info.get("display_label") or model_info.get("run_label") or model_data_path
        ax.plot(p_smooth, theta_smooth, linewidth=1.8, label=label)

        summary.append(
            {
                "label": label,
                "run_label": model_info.get("run_label"),
                "model_info_path": result.get("model_info_path"),
                "model_data_path": model_data_path,
                "k_rad_per_W": k,
                "theta1_minus_theta2_rad": theta0,
                "r_squared": model_info.get("r_squared"),
                "P2_min_W": p_min,
                "P2_max_W": p_max,
            }
        )

    ax.set_xlabel("P2 electrical power (W)")
    ax.set_ylabel("dtheta (rad)")
    ax.set_title(f"MZI {target} Lower-Arm Phase Models")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    image_path = os.path.join(save_dir, f"mzi{target}_{output_label}_phase_models.png")
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.join(save_dir, f"mzi{target}_{output_label}_phase_models.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"target": int(target), "image_path": image_path, "models": summary}, f, ensure_ascii=False, indent=2)

    return {
        "image_path": image_path,
        "summary_path": summary_path,
        "models": summary,
    }


def scan_lower_arm_heat_test(
    target: int,
    N: int,
    ser,
    pwm,
    file_data: pd.DataFrame,
    start_voltage: float = 0.0,
    end_voltage: float = 5.5,
    step: float = 0.1,
    measure_time: float = 1.0,
    save_dir: str = HEAT_TEST_DIR,
    mzi_table: dict | None = None,
    mzi_state_table: dict | None = None,
    upper_arm_power_w: float | None = None,
    run_label: str = "lower_only",
    display_label: str | None = None,
    apply_global_dtheta_biases: bool = False,
) -> dict:
    if ser is None:
        raise ValueError("Serial connection is required.")
    if pwm is None:
        raise ValueError("Power meter connection is required.")
    if step <= 0:
        raise ValueError("step must be positive.")

    os.makedirs(save_dir, exist_ok=True)
    file_prefix = f"mzi{target}_{run_label}_lower_arm"
    if display_label is None:
        display_label = run_label
    if mzi_table is None:
        mzi_table = load_mzi_table()
    if mzi_state_table is None:
        mzi_state_table = load_mzi_state_table()

    path_setup = configure_network_for_target(
        target,
        N,
        file_data,
        mzi_table=mzi_table,
        mzi_state_table=mzi_state_table,
        apply_dtheta=apply_global_dtheta_biases,
        ser=ser,
        upload=True,
    )
    target_path = path_setup["target_path"]
    upper_arm_port = get_mzi_upper_arm_port(target)
    lower_arm_port = get_mzi_lower_arm_port(target)
    output_port = target_path["output"] + 1
    input_port = target_path["input"] + 1

    upper_arm_bias = None
    if upper_arm_power_w is not None:
        upper_arm_resistance = get_mzi_arm_resistance(target, 0, mzi_table)
        upper_arm_voltage = voltage_for_heater_power(upper_arm_power_w, upper_arm_resistance)
        write_port_voltage(upper_arm_port, upper_arm_voltage, file_data)
        cu.upload_voltage(ser, file_data)
        upper_arm_bias = {
            "upper_arm_port": int(upper_arm_port),
            "upper_arm_resistance_ohm": float(upper_arm_resistance),
            "upper_arm_power_W": float(upper_arm_power_w),
            "upper_arm_power_mW": float(upper_arm_power_w * 1e3),
            "upper_arm_voltage_V": float(upper_arm_voltage),
        }
        print(
            f"MZI {target} upper-arm bias: port {upper_arm_port}, "
            f"P={upper_arm_power_w * 1e3:.3f} mW, R={upper_arm_resistance:.6f} ohm, "
            f"V={upper_arm_voltage:.6f} V"
        )

    metadata = {
        "target": int(target),
        "run_label": run_label,
        "display_label": display_label,
        "upper_arm_port": int(upper_arm_port),
        "lower_arm_port": int(lower_arm_port),
        "upper_arm_bias": upper_arm_bias,
        "input_port": int(input_port),
        "output_port": int(output_port),
        "start_voltage": float(start_voltage),
        "end_voltage": float(end_voltage),
        "step": float(step),
        "apply_global_dtheta_biases": bool(apply_global_dtheta_biases),
        "target_path": target_path,
    }
    metadata_path = os.path.join(save_dir, f"{file_prefix}_path.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    point_count = int(round((end_voltage - start_voltage) / step)) + 1
    v_values = np.round(start_voltage + np.arange(point_count) * step, 3)
    records = []

    print(f"MZI {target} path ({display_label}): {target_path['path']}")
    print(
        f"Input port: {input_port}, output port: {output_port}, "
        f"upper-arm heater port: {upper_arm_port}, lower-arm heater port: {lower_arm_port}"
    )

    for voltage in v_values:
        voltage = float(voltage)
        write_port_voltage(lower_arm_port, voltage, file_data)
        print(f"\nMZI {target} lower-arm scan voltage: {voltage:.3f} V")
        cu.upload_voltage(ser, file_data)
        current_ma = float(cu.read_current_port(ser, lower_arm_port))
        time.sleep(measure_time)
        optical_power_uw = read_output_power_uw(pwm, output_port)
        electrical_power_w = voltage * current_ma * 1e-3
        records.append(
            {
                "voltage_V": voltage,
                "current_mA": current_ma,
                "P2_W": electrical_power_w,
                "upper_arm_voltage_V": upper_arm_bias["upper_arm_voltage_V"] if upper_arm_bias else np.nan,
                "upper_arm_power_W": upper_arm_bias["upper_arm_power_W"] if upper_arm_bias else np.nan,
                "optical_power_uW": optical_power_uw,
            }
        )
        print(f"Current: {current_ma:.6f} mA, P2: {electrical_power_w:.9f} W, optical power: {optical_power_uw:.6f} uW")

    scan_df = pd.DataFrame(records)
    data_path = os.path.join(save_dir, f"{file_prefix}_scan.csv")
    scan_df.to_csv(data_path, index=False, encoding="utf-8")

    image_path = os.path.join(save_dir, f"{file_prefix}_scan.png")
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(scan_df["voltage_V"], scan_df["optical_power_uW"], marker="o", linewidth=1.6, label="Optical power")
    ax1.set_xlabel("Lower-arm voltage (V)")
    ax1.set_ylabel("Optical power (uW)")
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(
        scan_df["voltage_V"],
        scan_df["current_mA"],
        marker="s",
        linewidth=1.2,
        linestyle="--",
        color="tab:orange",
        label="Current",
    )
    ax2.set_ylabel("Current (mA)")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title(f"MZI {target} Lower-Arm Heat Test ({display_label})")
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    return {
        "data_path": data_path,
        "image_path": image_path,
        "metadata_path": metadata_path,
        "records": records,
        "path_setup": path_setup,
    }


def scan_mzi6_upper_with_mzi5_arm_biases(
    target: int,
    scan_mzi: int,
    N: int,
    ser,
    pwm,
    file_data: pd.DataFrame,
    start_voltage: float = 0.0,
    end_voltage: float = 5.5,
    step: float = 0.1,
    measure_time: float = 1.0,
    target_upper_arm_power_w: float = 16e-3,
    target_lower_arm_power_w: float = 8e-3,
    save_dir: str = HEAT_TEST_DIR,
    mzi_table: dict | None = None,
    mzi_state_table: dict | None = None,
    show_plot: bool = True,
) -> dict:
    if ser is None:
        raise ValueError("Serial connection is required.")
    if pwm is None:
        raise ValueError("Power meter connection is required.")
    if step <= 0:
        raise ValueError("step must be positive.")

    os.makedirs(save_dir, exist_ok=True)
    if mzi_table is None:
        mzi_table = load_mzi_table()
    if mzi_state_table is None:
        mzi_state_table = load_mzi_state_table()

    path_setup = configure_network_for_target(
        target,
        N,
        file_data,
        mzi_table=mzi_table,
        mzi_state_table=mzi_state_table,
        apply_dtheta=False,
        ser=ser,
        upload=True,
    )
    target_path = path_setup["target_path"]
    input_port = target_path["input"] + 1
    output_port = target_path["output"] + 1

    target_upper_bias = set_mzi_arm_power(target, 0, target_upper_arm_power_w, file_data, mzi_table)
    target_lower_bias = set_mzi_arm_power(target, 1, target_lower_arm_power_w, file_data, mzi_table)
    cu.upload_voltage(ser, file_data)

    scan_port = get_mzi_upper_arm_port(scan_mzi)
    upper_mw = target_upper_arm_power_w * 1e3
    lower_mw = target_lower_arm_power_w * 1e3
    run_label = f"target{target}_upper_{upper_mw:g}mW_lower_{lower_mw:g}mW_scan_mzi{scan_mzi}_upper"
    metadata = {
        "target": int(target),
        "scan_mzi": int(scan_mzi),
        "scan_arm": "upper",
        "scan_port": int(scan_port),
        "input_port": int(input_port),
        "output_port": int(output_port),
        "target_upper_bias": target_upper_bias,
        "target_lower_bias": target_lower_bias,
        "start_voltage": float(start_voltage),
        "end_voltage": float(end_voltage),
        "step": float(step),
        "target_path": target_path,
    }
    metadata_path = os.path.join(save_dir, f"mzi{scan_mzi}_upper_scan_with_{run_label}_path.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Target MZI {target} path: {target_path['path']}")
    print(f"Input port: {input_port}, monitored output port: {output_port}")
    print(
        f"MZI {target} upper bias: port {target_upper_bias['port']}, "
        f"{target_upper_bias['power_mW']:.3f} mW, V={target_upper_bias['voltage_V']:.6f} V"
    )
    print(
        f"MZI {target} lower bias: port {target_lower_bias['port']}, "
        f"{target_lower_bias['power_mW']:.3f} mW, V={target_lower_bias['voltage_V']:.6f} V"
    )
    print(f"Scanning MZI {scan_mzi} upper arm port {scan_port}")

    point_count = int(round((end_voltage - start_voltage) / step)) + 1
    v_values = np.round(start_voltage + np.arange(point_count) * step, 3)
    records = []

    for voltage in v_values:
        voltage = float(voltage)
        write_port_voltage(scan_port, voltage, file_data)
        print(f"\nMZI {scan_mzi} upper-arm scan voltage: {voltage:.3f} V")
        cu.upload_voltage(ser, file_data)
        current_ma = float(cu.read_current_port(ser, scan_port))
        time.sleep(measure_time)
        optical_power_uw = read_output_power_uw(pwm, output_port)
        scan_power_w = voltage * current_ma * 1e-3
        records.append(
            {
                "voltage_V": voltage,
                "current_mA": current_ma,
                "scan_power_W": scan_power_w,
                "optical_power_uW": optical_power_uw,
                "target_upper_voltage_V": target_upper_bias["voltage_V"],
                "target_lower_voltage_V": target_lower_bias["voltage_V"],
                "target_upper_power_W": float(target_upper_arm_power_w),
                "target_lower_power_W": float(target_lower_arm_power_w),
            }
        )
        print(f"Current: {current_ma:.6f} mA, scan power: {scan_power_w:.9f} W, optical power: {optical_power_uw:.6f} uW")

    scan_df = pd.DataFrame(records)
    data_path = os.path.join(save_dir, f"mzi{scan_mzi}_upper_scan_with_{run_label}.csv")
    scan_df.to_csv(data_path, index=False, encoding="utf-8")

    image_path = os.path.join(save_dir, f"mzi{scan_mzi}_upper_scan_with_{run_label}.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(scan_df["voltage_V"], scan_df["optical_power_uW"], marker="o", linewidth=1.8)
    ax.set_xlabel(f"MZI {scan_mzi} upper-arm voltage (V)")
    ax.set_ylabel("Output optical power (uW)")
    ax.set_title(f"MZI {scan_mzi} Upper-Arm Scan, MZI {target} Upper {upper_mw:g} mW Lower {lower_mw:g} mW")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "data_path": data_path,
        "image_path": image_path,
        "metadata_path": metadata_path,
        "records": records,
        "path_setup": path_setup,
    }


def scan_target_upper_with_lower_bias_min(
    target: int,
    N: int,
    ser,
    pwm,
    file_data: pd.DataFrame,
    lower_arm_power_w: float = 8e-3,
    start_voltage: float = 0.0,
    end_voltage: float = 5.5,
    step: float = 0.1,
    fine_window_v: float = 0.025,
    fine_step: float = 0.001,
    measure_time: float = 1.0,
    save_dir: str = HEAT_TEST_DIR,
    mzi_table: dict | None = None,
    mzi_state_table: dict | None = None,
    show_plot: bool = True,
) -> dict:
    if ser is None:
        raise ValueError("Serial connection is required.")
    if pwm is None:
        raise ValueError("Power meter connection is required.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if fine_window_v <= 0:
        raise ValueError("fine_window_v must be positive.")
    if fine_step <= 0:
        raise ValueError("fine_step must be positive.")

    os.makedirs(save_dir, exist_ok=True)
    if mzi_table is None:
        mzi_table = load_mzi_table()
    if mzi_state_table is None:
        mzi_state_table = load_mzi_state_table()

    path_setup = configure_network_for_target(
        target,
        N,
        file_data,
        mzi_table=mzi_table,
        mzi_state_table=mzi_state_table,
        apply_dtheta=False,
        ser=ser,
        upload=True,
    )
    target_path = path_setup["target_path"]
    input_port = target_path["input"] + 1
    output_port = target_path["output"] + 1

    lower_bias = set_mzi_arm_power(target, 1, lower_arm_power_w, file_data, mzi_table)
    cu.upload_voltage(ser, file_data)

    upper_port = get_mzi_upper_arm_port(target)
    lower_port = get_mzi_lower_arm_port(target)
    lower_mw = lower_arm_power_w * 1e3
    run_label = f"target{target}_lower_{lower_mw:g}mW_scan_upper"

    metadata = {
        "target": int(target),
        "scan_arm": "upper",
        "upper_port": int(upper_port),
        "lower_port": int(lower_port),
        "input_port": int(input_port),
        "output_port": int(output_port),
        "lower_bias": lower_bias,
        "start_voltage": float(start_voltage),
        "end_voltage": float(end_voltage),
        "step": float(step),
        "fine_window_v": float(fine_window_v),
        "fine_step": float(fine_step),
        "target_path": target_path,
    }
    metadata_path = os.path.join(save_dir, f"mzi{target}_lower_{lower_mw:g}mW_scan_upper_path.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Target MZI {target} path: {target_path['path']}")
    print(f"Input port: {input_port}, monitored output port: {output_port}")
    print(
        f"MZI {target} lower bias: port {lower_bias['port']}, "
        f"{lower_bias['power_mW']:.3f} mW, V={lower_bias['voltage_V']:.6f} V"
    )
    print(f"Scanning MZI {target} upper arm port {upper_port}")

    def scan_voltage_points(v_values, stage):
        stage_records = []
        for voltage in v_values:
            voltage = float(voltage)
            write_port_voltage(upper_port, voltage, file_data)
            print(f"\n{stage} scan MZI {target} upper-arm voltage: {voltage:.3f} V")
            cu.upload_voltage(ser, file_data)
            upper_current_ma = float(cu.read_current_port(ser, upper_port))
            time.sleep(measure_time)
            optical_power_uw = read_output_power_uw(pwm, output_port)
            upper_power_w = voltage * upper_current_ma * 1e-3
            stage_records.append(
                {
                    "stage": stage,
                    "upper_voltage_V": voltage,
                    "upper_current_mA": upper_current_ma,
                    "upper_power_W": upper_power_w,
                    "lower_voltage_V": lower_bias["voltage_V"],
                    "lower_power_W": lower_bias["power_W"],
                    "optical_power_uW": optical_power_uw,
                }
            )
            print(
                f"Upper current: {upper_current_ma:.6f} mA, "
                f"upper power: {upper_power_w:.9f} W, optical power: {optical_power_uw:.6f} uW"
            )
        return stage_records

    coarse_count = int(round((end_voltage - start_voltage) / step)) + 1
    coarse_v_values = np.round(start_voltage + np.arange(coarse_count) * step, 3)
    coarse_records = scan_voltage_points(coarse_v_values, "coarse")
    coarse_df = pd.DataFrame(coarse_records)
    coarse_min_idx = int(coarse_df["optical_power_uW"].idxmin())
    coarse_min_voltage = float(coarse_df.loc[coarse_min_idx, "upper_voltage_V"])

    fine_start = max(float(start_voltage), coarse_min_voltage - fine_window_v)
    fine_end = min(float(end_voltage), coarse_min_voltage + fine_window_v)
    fine_v_values = np.round(np.arange(fine_start, fine_end + fine_step * 0.5, fine_step), 3)
    if fine_v_values.size == 0 or fine_v_values[-1] < fine_end - fine_step * 0.5:
        fine_v_values = np.append(fine_v_values, round(fine_end, 3))
    print(
        f"\nCoarse minimum near {coarse_min_voltage:.3f} V; "
        f"fine scan range [{fine_start:.3f}, {fine_end:.3f}] V with step {fine_step:.3f} V"
    )
    fine_records = scan_voltage_points(fine_v_values, "fine")

    records = coarse_records + fine_records
    scan_df = pd.DataFrame(records)
    data_path = os.path.join(save_dir, f"mzi{target}_lower_{lower_mw:g}mW_scan_upper.csv")
    scan_df.to_csv(data_path, index=False, encoding="utf-8")

    fine_df = pd.DataFrame(fine_records)
    min_source_df = fine_df if not fine_df.empty else scan_df
    min_idx = int(min_source_df["optical_power_uW"].idxmin())
    min_row = min_source_df.loc[min_idx]
    min_point = {
        "stage": str(min_row["stage"]),
        "upper_voltage_V": float(min_row["upper_voltage_V"]),
        "lower_voltage_V": float(min_row["lower_voltage_V"]),
        "optical_power_uW": float(min_row["optical_power_uW"]),
        "upper_current_mA": float(min_row["upper_current_mA"]),
        "upper_power_W": float(min_row["upper_power_W"]),
        "lower_power_W": float(min_row["lower_power_W"]),
        "coarse_min_voltage_V": coarse_min_voltage,
        "fine_start_V": float(fine_start),
        "fine_end_V": float(fine_end),
        "fine_step_V": float(fine_step),
    }

    print("=" * 60)
    print(f"MZI {target} minimum output optical power point from fine scan:")
    print(f"Optical power: {min_point['optical_power_uW']:.6f} uW")
    print(f"Upper electrode voltage: {min_point['upper_voltage_V']:.6f} V")
    print(f"Lower electrode voltage: {min_point['lower_voltage_V']:.6f} V")
    print("=" * 60)

    image_path = os.path.join(save_dir, f"mzi{target}_lower_{lower_mw:g}mW_scan_upper.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    coarse_plot_df = scan_df[scan_df["stage"] == "coarse"]
    fine_plot_df = scan_df[scan_df["stage"] == "fine"]
    ax.plot(
        coarse_plot_df["upper_voltage_V"],
        coarse_plot_df["optical_power_uW"],
        marker="o",
        linewidth=1.2,
        label="coarse 0.1 V",
    )
    ax.plot(
        fine_plot_df["upper_voltage_V"],
        fine_plot_df["optical_power_uW"],
        marker=".",
        linewidth=1.6,
        label="fine 0.001 V",
    )
    ax.scatter(
        [min_point["upper_voltage_V"]],
        [min_point["optical_power_uW"]],
        color="tab:red",
        zorder=3,
        label="fine min",
    )
    ax.set_xlabel(f"MZI {target} upper-arm voltage (V)")
    ax.set_ylabel("Output optical power (uW)")
    ax.set_title(f"MZI {target} Upper-Arm Scan, Lower Arm {lower_mw:g} mW")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)

    min_path = os.path.join(save_dir, f"mzi{target}_lower_{lower_mw:g}mW_scan_upper_min.json")
    with open(min_path, "w", encoding="utf-8") as f:
        json.dump({"target": int(target), "min_point": min_point, "data_path": data_path}, f, ensure_ascii=False, indent=2)

    return {
        "data_path": data_path,
        "image_path": image_path,
        "metadata_path": metadata_path,
        "min_path": min_path,
        "min_point": min_point,
        "records": records,
        "path_setup": path_setup,
    }


def scan_lower_arm_with_target_fixed_voltages(
    target: int,
    scan_mzi: int,
    N: int,
    ser,
    pwm,
    file_data: pd.DataFrame,
    target_upper_voltage: float,
    target_lower_voltage: float,
    start_voltage: float = 0.0,
    end_voltage: float = 5.5,
    step: float = 0.1,
    measure_time: float = 1.0,
    save_dir: str = HEAT_TEST_DIR,
    mzi_table: dict | None = None,
    mzi_state_table: dict | None = None,
    show_plot: bool = True,
    scan_arm_index: int = 1,
    zero_mzi_arm: tuple[int, int] | None = None,
    target_operating_point_info: dict | None = None,
) -> dict:
    if ser is None:
        raise ValueError("Serial connection is required.")
    if pwm is None:
        raise ValueError("Power meter connection is required.")
    if step <= 0:
        raise ValueError("step must be positive.")

    os.makedirs(save_dir, exist_ok=True)
    if mzi_table is None:
        mzi_table = load_mzi_table()
    if mzi_state_table is None:
        mzi_state_table = load_mzi_state_table()

    path_setup = configure_network_for_target(
        target,
        N,
        file_data,
        mzi_table=mzi_table,
        mzi_state_table=mzi_state_table,
        apply_dtheta=False,
        ser=ser,
        upload=True,
    )
    target_path = path_setup["target_path"]
    input_port = target_path["input"] + 1
    output_port = target_path["output"] + 1

    target_upper_port = get_mzi_upper_arm_port(target)
    target_lower_port = get_mzi_lower_arm_port(target)
    write_port_voltage(target_upper_port, target_upper_voltage, file_data)
    write_port_voltage(target_lower_port, target_lower_voltage, file_data)

    zero_info = None
    if zero_mzi_arm is not None:
        zero_mzi, zero_arm_index = zero_mzi_arm
        zero_ports = get_mzi_h_list(zero_mzi)
        zero_port = int(zero_ports[zero_arm_index])
        write_port_voltage(zero_port, 0.0, file_data)
        zero_info = {
            "mzi": int(zero_mzi),
            "arm_index": int(zero_arm_index),
            "arm_name": "upper" if zero_arm_index == 0 else "lower",
            "port": zero_port,
            "voltage_V": 0.0,
        }

    cu.upload_voltage(ser, file_data)

    scan_arm_name = "upper" if scan_arm_index == 0 else "lower"
    scan_port = get_mzi_upper_arm_port(scan_mzi) if scan_arm_index == 0 else get_mzi_lower_arm_port(scan_mzi)
    run_label = f"target{target}_fixed_scan_mzi{scan_mzi}_{scan_arm_name}"
    metadata = {
        "target": int(target),
        "scan_mzi": int(scan_mzi),
        "scan_arm": scan_arm_name,
        "scan_port": int(scan_port),
        "zero_info": zero_info,
        "target_upper_port": int(target_upper_port),
        "target_lower_port": int(target_lower_port),
        "target_upper_voltage_V": float(target_upper_voltage),
        "target_lower_voltage_V": float(target_lower_voltage),
        "target_operating_point_info": target_operating_point_info,
        "input_port": int(input_port),
        "output_port": int(output_port),
        "start_voltage": float(start_voltage),
        "end_voltage": float(end_voltage),
        "step": float(step),
        "target_path": target_path,
    }
    metadata_path = os.path.join(save_dir, f"mzi{scan_mzi}_{scan_arm_name}_scan_with_{run_label}_path.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Target MZI {target} path: {target_path['path']}")
    print(f"Input port: {input_port}, monitored output port: {output_port}")
    print(f"MZI {target} upper fixed: port {target_upper_port}, V={target_upper_voltage:.6f} V")
    print(f"MZI {target} lower fixed: port {target_lower_port}, V={target_lower_voltage:.6f} V")
    if zero_info is not None:
        print(f"MZI {zero_info['mzi']} {zero_info['arm_name']} set to 0 V on port {zero_info['port']}")
    print(f"Scanning MZI {scan_mzi} {scan_arm_name} arm port {scan_port}")

    point_count = int(round((end_voltage - start_voltage) / step)) + 1
    v_values = np.round(start_voltage + np.arange(point_count) * step, 3)
    records = []

    for voltage in v_values:
        voltage = float(voltage)
        write_port_voltage(scan_port, voltage, file_data)
        print(f"\nMZI {scan_mzi} {scan_arm_name}-arm scan voltage: {voltage:.3f} V")
        cu.upload_voltage(ser, file_data)
        current_ma = float(cu.read_current_port(ser, scan_port))
        time.sleep(measure_time)
        optical_power_uw = read_output_power_uw(pwm, output_port)
        scan_power_w = voltage * current_ma * 1e-3
        records.append(
            {
                "scan_voltage_V": voltage,
                "scan_current_mA": current_ma,
                "scan_power_W": scan_power_w,
                "optical_power_uW": optical_power_uw,
                "target_upper_voltage_V": float(target_upper_voltage),
                "target_lower_voltage_V": float(target_lower_voltage),
            }
        )
        print(f"Current: {current_ma:.6f} mA, scan power: {scan_power_w:.9f} W, optical power: {optical_power_uw:.6f} uW")

    scan_df = pd.DataFrame(records)
    data_path = os.path.join(save_dir, f"mzi{scan_mzi}_{scan_arm_name}_scan_with_{run_label}.csv")
    scan_df.to_csv(data_path, index=False, encoding="utf-8")

    image_path = os.path.join(save_dir, f"mzi{scan_mzi}_{scan_arm_name}_scan_with_{run_label}.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(scan_df["scan_voltage_V"], scan_df["optical_power_uW"], marker="o", linewidth=1.8)
    ax.set_xlabel(f"MZI {scan_mzi} {scan_arm_name}-arm voltage (V)")
    ax.set_ylabel("Output optical power (uW)")
    ax.set_title(f"MZI {scan_mzi} {scan_arm_name.title()}-Arm Scan, MZI {target} Fixed")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "data_path": data_path,
        "image_path": image_path,
        "metadata_path": metadata_path,
        "records": records,
        "path_setup": path_setup,
    }


def sample_output_power_with_path_only(
    target: int,
    N: int,
    ser,
    pwm,
    file_data: pd.DataFrame,
    target_upper_voltage: float,
    target_lower_voltage: float,
    sample_interval_s: float = 1.0,
    sample_count: int = 100,
    save_dir: str = HEAT_TEST_DIR,
    mzi_table: dict | None = None,
    mzi_state_table: dict | None = None,
    show_plot: bool = True,
    target_operating_point_info: dict | None = None,
) -> dict:
    if ser is None:
        raise ValueError("Serial connection is required.")
    if pwm is None:
        raise ValueError("Power meter connection is required.")
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")

    os.makedirs(save_dir, exist_ok=True)
    if mzi_table is None:
        mzi_table = load_mzi_table()
    if mzi_state_table is None:
        mzi_state_table = load_mzi_state_table()

    path_setup = configure_network_for_target(
        target,
        N,
        file_data,
        mzi_table=mzi_table,
        mzi_state_table=mzi_state_table,
        apply_dtheta=False,
        ser=ser,
        upload=True,
    )
    target_path = path_setup["target_path"]
    input_port = target_path["input"] + 1
    output_port = target_path["output"] + 1

    target_upper_port = get_mzi_upper_arm_port(target)
    target_lower_port = get_mzi_lower_arm_port(target)
    write_port_voltage(target_upper_port, target_upper_voltage, file_data)
    write_port_voltage(target_lower_port, target_lower_voltage, file_data)
    cu.upload_voltage(ser, file_data)

    metadata = {
        "target": int(target),
        "target_upper_port": int(target_upper_port),
        "target_lower_port": int(target_lower_port),
        "target_upper_voltage_V": float(target_upper_voltage),
        "target_lower_voltage_V": float(target_lower_voltage),
        "input_port": int(input_port),
        "output_port": int(output_port),
        "sample_interval_s": float(sample_interval_s),
        "sample_count": int(sample_count),
        "target_path": target_path,
        "target_operating_point_info": target_operating_point_info,
    }
    metadata_path = os.path.join(save_dir, f"mzi{target}_path_only_power_sampling_path.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Target MZI {target} path: {target_path['path']}")
    print(f"Input port: {input_port}, monitored output port: {output_port}")
    print(f"MZI {target} upper fixed: port {target_upper_port}, V={target_upper_voltage:.6f} V")
    print(f"MZI {target} lower fixed: port {target_lower_port}, V={target_lower_voltage:.6f} V")
    print(f"Sampling output power every {sample_interval_s:.3f}s for {sample_count} points")

    records = []
    start_time = time.time()
    for sample_idx in range(sample_count):
        now = time.time()
        elapsed_s = now - start_time
        optical_power_uw = read_output_power_uw(pwm, output_port)
        records.append(
            {
                "sample_index": sample_idx,
                "elapsed_s": elapsed_s,
                "optical_power_uW": optical_power_uw,
            }
        )
        print(f"sample {sample_idx + 1}/{sample_count}: t={elapsed_s:.3f}s, power={optical_power_uw:.6f} uW")
        if sample_idx < sample_count - 1:
            target_time = start_time + (sample_idx + 1) * sample_interval_s
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    sample_df = pd.DataFrame(records)
    data_path = os.path.join(save_dir, f"mzi{target}_path_only_power_sampling.csv")
    sample_df.to_csv(data_path, index=False, encoding="utf-8")

    image_path = os.path.join(save_dir, f"mzi{target}_path_only_power_sampling.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(sample_df["elapsed_s"], sample_df["optical_power_uW"], marker="o", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output optical power (uW)")
    ax.set_title(f"MZI {target} Path-Only Output Power Sampling")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "data_path": data_path,
        "image_path": image_path,
        "metadata_path": metadata_path,
        "records": records,
        "path_setup": path_setup,
    }


if __name__ == "__main__":
    N = 9
    TARGET_MZI = 10
    SCAN_MZI = 28
    START_VOLTAGE = 0.0
    END_VOLTAGE = 5.5
    STEP_VOLTAGE = 0.1
    MEASURE_TIME = 1.0
    TARGET_LOWER_ARM_POWER_W = 8e-3
    HALF_POWER_SIDE = "left"

    SER_ADDRESS = "COM3"
    OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"

    mzi_table = load_mzi_table()
    mzi_state_table = load_mzi_state_table()
    TARGET_LOWER_VOLTAGE = voltage_for_heater_power(
        TARGET_LOWER_ARM_POWER_W,
        get_mzi_arm_resistance(TARGET_MZI, 1, mzi_table),
    )
    half_power_scan_path = find_latest_file(f"mzi{TARGET_MZI}_lower_8mW_scan_upper.csv", HEAT_TEST_DIR)
    half_power_info = estimate_half_power_upper_voltage_from_scan(half_power_scan_path, side=HALF_POWER_SIDE)
    TARGET_UPPER_VOLTAGE = half_power_info["half_power_voltage_V"]
    run_dir = os.path.join(
        HEAT_TEST_DIR,
        f"target{TARGET_MZI}_half_power_scan_mzi{SCAN_MZI}_lower_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    path, input_idx, output_idx, state = find_path(TARGET_MZI, N)
    print(f"MZI {TARGET_MZI} path: {[int(x) for x in path]}")
    print(f"Input port: {input_idx + 1}, output port: {output_idx + 1}, path state: {state}")
    print(f"MZI {TARGET_MZI} half-power scan source: {half_power_scan_path}")
    print(f"MZI {TARGET_MZI} half-power side: {HALF_POWER_SIDE}")
    print(f"MZI {TARGET_MZI} half-power output: {half_power_info['half_power_uW']:.6f} uW")
    print(f"MZI {TARGET_MZI} upper fixed voltage at half power: {TARGET_UPPER_VOLTAGE:.6f} V")
    print(f"MZI {TARGET_MZI} lower fixed voltage: {TARGET_LOWER_VOLTAGE:.6f} V")
    print(f"Scan MZI {SCAN_MZI} lower port: {get_mzi_lower_arm_port(SCAN_MZI)}")
    print("Heat test output directory:", run_dir)

    mcv = cu.open_ser_connection(SER_ADDRESS)
    opm2 = cu.open_VISA_connection(OPM2_ADDRESS)
    if mcv is None:
        raise RuntimeError(f"Failed to open serial connection: {SER_ADDRESS}")
    if opm2 is None:
        raise RuntimeError(f"Failed to open power meter connection: {OPM2_ADDRESS}")

    try:
        working_data = cu.generate_working_data()
        result = scan_lower_arm_with_target_fixed_voltages(
            TARGET_MZI,
            SCAN_MZI,
            N,
            mcv,
            opm2,
            working_data,
            target_upper_voltage=TARGET_UPPER_VOLTAGE,
            target_lower_voltage=TARGET_LOWER_VOLTAGE,
            start_voltage=START_VOLTAGE,
            end_voltage=END_VOLTAGE,
            step=STEP_VOLTAGE,
            measure_time=MEASURE_TIME,
            save_dir=run_dir,
            mzi_table=mzi_table,
            mzi_state_table=mzi_state_table,
            show_plot=True,
            scan_arm_index=1,
            zero_mzi_arm=None,
            target_operating_point_info=half_power_info,
        )
        print("MZI28 lower scan data saved to:", result["data_path"])
        print("MZI28 lower scan curve saved to:", result["image_path"])
        print("Path metadata saved to:", result["metadata_path"])
    finally:
        if opm2 is not None:
            opm2.close()
        if mcv is not None and mcv.is_open:
            mcv.close()
