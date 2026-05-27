import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.communication as cu
import utils.AllDecompositionUtils as du
from inner_calibration import find_path as find_inner_path
from colorama import Fore, Style, init

HALFPI_SCAN_MAX_V = 5.5
HALFPI_COARSE_STEP_V = 0.1


def _estimate_sine_w(dp: np.ndarray, power: np.ndarray) -> float:
    if dp.size < 2:
        return 1.0
    spacing = float(np.mean(np.diff(dp)))
    if spacing <= 0:
        return 1.0
    y = power - np.mean(power)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(dp.size, d=spacing)
    if spectrum.size > 1:
        idx = int(np.argmax(spectrum[1:]) + 1)
        freq = float(freqs[idx])
        if freq > 0:
            return float(2 * np.pi * freq)
    return float(2 * np.pi / (dp.max() - dp.min() + 1e-9))


def _resolve_inter_cali_file(ports, data_dir: str) -> str:
    ports_list = [int(p) for p in ports]
    with_space = os.path.join(data_dir, f"{ports_list}.txt")
    if os.path.exists(with_space):
        return with_space
    no_space = os.path.join(data_dir, f"[{','.join(str(p) for p in ports_list)}].txt")
    if os.path.exists(no_space):
        return no_space
    raise FileNotFoundError(f"No inter_cali_powerdata file for ports {ports_list}")


def load_mzi_table(path: str = os.path.join("Scandata", "MZI_table.json")) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    for entry in table.values():
        if "dtheta" not in entry and "dtheta_Bar" in entry and "dtheta_Cross" in entry:
            bar_values = entry.get("dtheta_Bar", [])
            cross_values = entry.get("dtheta_Cross", [])
            if bar_values and cross_values:
                entry["dtheta"] = [bar_values[0], cross_values[0]]
    return table


def _get_mzi_heater_calibration(target: int, table: dict | None = None, require_two_heaters: bool = True):
    if table is None:
        table = globals().get("MZI_TABLE")
        if table is None:
            table = globals().get("mzi_table")
        if table is None:
            table = load_mzi_table()
            globals()["MZI_TABLE"] = table

    key = str(int(target))
    if key not in table:
        raise ValueError(f"MZI {target} not found in mzi_table.")

    entry = table[key]
    ports = entry.get("ports", [])
    heater_r = entry.get("heater_R", [])
    ppi = entry.get("Ppi", [])
    if require_two_heaters and len(ports) != 2:
        raise ValueError(f"MZI {target} must have exactly two arm ports.")
    if len(heater_r) < len(ports) or len(ppi) < len(ports):
        raise ValueError(f"MZI {target} missing heater_R or Ppi in mzi_table.")

    r_values = [float(heater_r[idx]) for idx in range(len(ports))]
    ppi_values = [float(ppi[idx]) for idx in range(len(ports))]
    if any(value <= 0 or not np.isfinite(value) for value in r_values + ppi_values):
        raise ValueError(f"MZI {target} has invalid heater_R or Ppi in mzi_table.")
    return entry, [int(port) for port in ports], r_values, ppi_values


def _halfpi_sine_model(x, A, w, phi, b):
    return A * np.sin(w * x + phi) + b


def _deduplicate_xy(x, y, tol=1e-12):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    x_out = []
    y_out = []
    bucket_x = []
    bucket_y = []

    for x_value, y_value in zip(x, y):
        if not bucket_x or abs(float(x_value) - bucket_x[-1]) <= tol:
            bucket_x.append(float(x_value))
            bucket_y.append(float(y_value))
            continue

        x_out.append(float(np.mean(bucket_x)))
        y_out.append(float(np.mean(bucket_y)))
        bucket_x = [float(x_value)]
        bucket_y = [float(y_value)]

    if bucket_x:
        x_out.append(float(np.mean(bucket_x)))
        y_out.append(float(np.mean(bucket_y)))

    return np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float)


def _fit_halfpi_power_curve(electrical_power, optical_power):
    x = np.asarray(electrical_power, dtype=float)
    y = np.asarray(optical_power, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        raise ValueError("Need at least four valid samples for halfpi fitting.")

    x, y = _deduplicate_xy(x, y)
    if x.size < 4:
        raise ValueError("Need at least four unique samples for halfpi fitting.")

    power_span = float(y.max() - y.min())
    amplitude_guess = max(0.5 * power_span, 1e-6)
    offset_guess = float(np.mean(y))
    min_idx = int(np.argmin(y))
    max_idx = int(np.argmax(y))

    best_popt = None
    best_error = None
    x_span = max(float(np.ptp(x)), 1e-9)
    w_candidates = [max(_estimate_sine_w(x, y), 1e-6), 2 * np.pi / x_span]
    extrema_distance = abs(float(x[max_idx]) - float(x[min_idx]))
    if extrema_distance > 1e-9:
        w_candidates.append(np.pi / extrema_distance)
    w_seeds = sorted(
        {
            float(w_base * scale)
            for w_base in w_candidates
            for scale in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
            if w_base * scale > 0
        }
    )
    a_seeds = [amplitude_guess, -amplitude_guess, max(0.25 * power_span, 1e-6), -max(0.25 * power_span, 1e-6)]
    b_seeds = [
        offset_guess,
        float(0.5 * (y.max() + y.min())),
        float(y.max()),
        float(y.min()),
    ]

    for w0 in w_seeds:
        phi_seeds = [
            -np.pi / 2 - w0 * float(x[min_idx]),
            np.pi / 2 - w0 * float(x[max_idx]),
            np.pi / 2 - w0 * float(x[min_idx]),
            -np.pi / 2 - w0 * float(x[max_idx]),
            0.0,
            np.pi / 2,
            -np.pi / 2,
        ]
        for A0 in a_seeds:
            for b0 in b_seeds:
                for phi0 in phi_seeds:
                    try:
                        popt, _ = curve_fit(
                            _halfpi_sine_model,
                            x,
                            y,
                            p0=[A0, w0, phi0, b0],
                            bounds=([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                            maxfev=50000,
                        )
                    except (RuntimeError, ValueError):
                        continue

                    fitted = _halfpi_sine_model(x, *popt)
                    error = float(np.sum((y - fitted) ** 2))
                    if best_error is None or error < best_error:
                        best_popt = popt
                        best_error = error

    if best_popt is None:
        raise RuntimeError("Failed to fit halfpi sine model.")

    A, w, phi, b = best_popt
    if w <= 0:
        raise RuntimeError("Invalid non-positive sine frequency from halfpi fit.")
    return float(A), float(w), float(phi), float(b), x, y


def _records_to_halfpi_arrays(records):
    if not records:
        raise ValueError("No scan records.")

    df = pd.DataFrame(records)
    voltage = df["voltage"].to_numpy(dtype=float)
    optical_power = df["optical_power_uW"].to_numpy(dtype=float)
    electrical_power = df["electrical_power_W"].to_numpy(dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(optical_power) & np.isfinite(electrical_power)
    voltage = voltage[mask]
    optical_power = optical_power[mask]
    electrical_power = electrical_power[mask]
    if voltage.size < 4:
        raise ValueError("Not enough valid scan records.")

    order = np.argsort(voltage)
    return {
        "voltage": voltage[order],
        "optical_power": optical_power[order],
        "electrical_power": electrical_power[order],
    }


def _read_output_power_uW(pwm, out_num):
    power_str_list = cu.read_pow(pwm)
    try:
        return float(power_str_list[int(out_num) - 1]) * 1e6
    except (ValueError, IndexError) as e:
        print(f"Read power error: {e}")
        return 0.0


def _get_power_halfpi_context():
    table = globals().get("MZI_TABLE")
    if table is None:
        table = globals().get("mzi_table")
    if table is None:
        table = load_mzi_table()
        globals()["MZI_TABLE"] = table

    file_data = globals().get("working_data")
    if file_data is None:
        file_data = cu.generate_working_data()
        globals()["working_data"] = file_data

    ser = globals().get("mcv")
    if ser is None:
        ser = globals().get("ser")
    pwm = globals().get("opm2")
    if pwm is None:
        pwm = globals().get("pwm")

    missing = []
    if ser is None:
        missing.append("mcv")
    if pwm is None:
        missing.append("opm2")
    if missing:
        raise RuntimeError(
            "Power_halfpi requires initialized hardware globals: "
            + ", ".join(missing)
            + ". Open them the same way as in __main__ before calling this function."
        )

    measure_delay = float(globals().get("measure_time", 1))
    n_value = int(globals().get("N", 9))
    return table, file_data, ser, pwm, measure_delay, n_value


def _prepare_inner_monitor_path(target, table, file_data, n_value):
    target_id = int(target)
    key = str(target_id)
    if key not in table:
        raise ValueError(f"MZI {target} not found in mzi_table.")

    path, input_idx, output_idx, state = find_inner_path(target_id, int(n_value))
    path = [int(item) for item in path]
    state = [str(item) for item in state]

    for switch_idx in range(int(n_value) - 1):
        switch_IN(switch_idx + 1, "OFF", file_data)
    switch_IN(int(input_idx) + 1, "ON", file_data)

    target_ports = [int(port) for port in table[key].get("ports", [])]
    for mzi_id, state_value in zip(path, state):
        entry = table[str(mzi_id)]
        ports = entry.get("ports", [])
        dtheta = entry.get("dtheta", [])
        if mzi_id == target_id:
            for port in target_ports:
                write_port_voltage(port, 0.0, file_data)
            continue

        if not isinstance(ports, list) or not ports:
            raise ValueError(f"MZI {mzi_id} has no ports in mzi_table.")
        if not isinstance(dtheta, list) or len(dtheta) < 2:
            raise ValueError(f"MZI {mzi_id} has invalid dtheta in mzi_table.")

        if state_value == "B":
            voltage = float(dtheta[0])
        elif state_value == "C":
            voltage = float(dtheta[1])
        else:
            raise ValueError(f"Unsupported path state {state_value!r} for MZI {mzi_id}.")
        write_port_voltage(int(ports[0]), voltage, file_data)

    print(f"Target {target_id} monitor path: {path}")
    print(f"Target {target_id} path state: {state}")
    print(f"Input switch: {int(input_idx) + 1}, output channel: {int(output_idx) + 1}")
    return path, state, int(input_idx) + 1, int(output_idx) + 1


def _scan_halfpi_arm(target, arm_name, port, all_target_ports, ser, pwm, measure_time, out_num, file_data):
    print("=" * 50)
    print(f"Scanning target {int(target)} {arm_name} arm, port {int(port)}")

    for target_port in all_target_ports:
        write_port_voltage(int(target_port), 0.0, file_data)

    resistance, current_offset_a = cu.get_R(ser, int(port), file_data)
    resistance = float(resistance)
    print(f"Target {int(target)} {arm_name} port {int(port)} heater R={resistance} Ohm")

    def scan_voltage_points(v_values, stage_name):
        rows = []
        for v_raw in v_values:
            v = round(max(0.0, min(HALFPI_SCAN_MAX_V, float(v_raw))), 3)
            for target_port in all_target_ports:
                if int(target_port) != int(port):
                    write_port_voltage(int(target_port), 0.0, file_data)
            write_port_voltage(int(port), v, file_data)

            print(f"\n{stage_name} scan, target {int(target)} {arm_name} port {int(port)} set {v:.3f} V")
            read_current_limit = 0
            current_ma = np.nan
            while True:
                cu.upload_voltage(ser, file_data)
                current_reading = cu.read_current_port(ser, int(port))
                if current_reading is not None:
                    current_ma = float(current_reading) - current_offset_a * 1e3

                expected_current_ma = v / resistance * 1000.0 if resistance > 0 else 0.0
                if v == 0:
                    print(Fore.YELLOW + "Voltage is 0 V, skip current validation")
                    break
                if np.isfinite(current_ma) and 0.9 * expected_current_ma < current_ma < 1.1 * expected_current_ma:
                    print(Fore.GREEN + f"Current OK, I={current_ma} mA")
                    break
                if read_current_limit >= 5:
                    print(Fore.RED + f"Current abnormal after retries, I={current_ma} mA, keep this point")
                    break

                print(Fore.RED + f"Current abnormal, I={current_ma} mA, retry upload")
                read_current_limit += 1

            time.sleep(measure_time)
            optical_power_uW = _read_output_power_uW(pwm, out_num)
            electrical_power_w = 0.0 if v == 0 else v * current_ma * 1e-3
            print(f"Optical power: {optical_power_uW} uW, electrical power: {electrical_power_w:.9f} W")
            rows.append(
                {
                    "target": int(target),
                    "arm": arm_name,
                    "port": int(port),
                    "stage": stage_name,
                    "voltage": float(v),
                    "optical_power_uW": float(optical_power_uW),
                    "current_mA": float(current_ma) if np.isfinite(current_ma) else np.nan,
                    "electrical_power_W": float(electrical_power_w),
                    "R_ohm": float(resistance),
                }
            )
        return rows

    coarse_v_values = np.round(
        np.arange(0.0, HALFPI_SCAN_MAX_V + HALFPI_COARSE_STEP_V * 0.5, HALFPI_COARSE_STEP_V),
        3,
    )
    coarse_records = scan_voltage_points(coarse_v_values, "coarse")

    scan_arrays = _records_to_halfpi_arrays(coarse_records)
    A, w, phi, b, _, _ = _fit_halfpi_power_curve(
        scan_arrays["electrical_power"],
        scan_arrays["optical_power"],
    )
    ppi = float(np.pi / w)
    print(
        f"Final halfpi fit target {int(target)} {arm_name}: A={A:.6f}, w={w:.6f}, "
        f"phi={phi:.6f}, b={b:.6f}, Ppi={ppi:.9f} W"
    )

    write_port_voltage(int(port), 0.0, file_data)
    cu.upload_voltage(ser, file_data)
    return resistance, ppi


def Power_halfpi(target):
    target_id = int(target)
    table = globals().get("MZI_TABLE")
    if table is None:
        table = globals().get("mzi_table")
    if table is None:
        table = load_mzi_table()
        globals()["MZI_TABLE"] = table

    _, _, r_values, ppi_values = _get_mzi_heater_calibration(target_id, table=table, require_two_heaters=True)
    up_R, down_R = r_values
    up_P, down_P = ppi_values
    print(
        f"Target {target_id} halfpi from MZI_table: "
        f"up_R={up_R:.6f} Ohm, up_P={up_P:.9f} W, "
        f"down_R={down_R:.6f} Ohm, down_P={down_P:.9f} W"
    )
    return up_R, up_P, down_R, down_P


def find_path_BottomWaveguide(N, n):
    cm = du.Clements_matrix(N)
    bottom_row = cm[-1]
    zero_cols = np.flatnonzero(bottom_row == 0)

    if n < 1 or n > len(zero_cols):
        raise ValueError(f"n must be between 1 and {len(zero_cols)} for the bottom-row zeros.")

    if (N % 2 == 1 and n == 1) or n == len(zero_cols):
        return [], [], None, None, None

    zero_col = int(zero_cols[n - 1])
    row = cm.shape[0] - 1
    path_points = [(zero_col, 0)]

    left_row = row
    left_col = zero_col - 1
    if left_col >= 0 and cm[left_row, left_col] != 0:
        path_points.append((left_col, int(cm[left_row, left_col])))
        while left_row > 0 and left_col > 0:
            left_row -= 1
            left_col -= 1
            if cm[left_row, left_col] != 0:
                path_points.append((left_col, int(cm[left_row, left_col])))

    right_row = row
    right_col = zero_col + 1
    if right_col < cm.shape[1] and cm[right_row, right_col] != 0:
        path_points.append((right_col, int(cm[right_row, right_col])))
        while right_row > 0 and right_col < cm.shape[1] - 1:
            right_row -= 1
            right_col += 1
            if cm[right_row, right_col] != 0:
                path_points.append((right_col, int(cm[right_row, right_col])))

    path = [value for _, value in sorted(path_points, key=lambda item: item[0])]
    state = ["C"] * len(path)
    zero_idx = path.index(0)
    state[zero_idx] = "X"
    bmzi = int(cm[row - 1, zero_col])

    if zero_idx - 1 >= 0:
        state[zero_idx - 1] = "H"
    if zero_idx + 1 < len(state):
        state[zero_idx + 1] = "H"

    input = int(np.argwhere(cm == path[0])[0][0]) + 1
    output = int(np.argwhere(cm == path[-1])[0][0]) + 1

    return path, state, input, output, bmzi


def fit_p_to_op(index: int):
    """
    Fit OP = A*sin(w*P + phi) + b using powerdata/<index>.txt.
    Returns (A, w, phi, b, p_elec, op).
    """
    filename = os.path.join("Scandata\\inner_cali_powerdata", f"{index}.txt")
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
    - return slope sign for each P (1 for positive, -1 for negative, 0 near zero)
    - return P where OP is max (sin = 1) in range
    - return P where OP is min (sin = -1) in range
    """
    if A == 0 or w == 0:
        raise ValueError("A and w must be non-zero.")

    filename = os.path.join("Scandata\\inner_cali_powerdata", f"{port}.txt")
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
    # p_max = float(np.max(p_elec))
    p_max = 0.05
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

    def slope_sign(p):
        slope = A * w * np.cos(w * p + phi)
        if abs(slope) < 1e-12:
            return 0
        return 1 if slope > 0 else -1

    p_slope_signs = [slope_sign(p) for p in p_solutions]

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

    return p_solutions, p_slope_signs, p_at_max, p_at_min


def write_port_voltage(port: int, voltage: float, file_data: pd.DataFrame) -> None:
    """
    Write voltage into file_data at the row corresponding to PORT (1-based).
    """
    port_idx = port - 1
    if port_idx < 0 or port_idx >= len(file_data):
        raise IndexError(f"PORT {port} is out of range for the provided file_data.")
    file_data.iloc[port_idx, 0] = voltage


def scan_mzis(target, ser, pwm, measure_time, out_num, file_path_df, up_R, up_P, down_R, down_P):
    mzi_table = load_mzi_table()
    key = str(int(target))
    if key not in mzi_table:
        raise ValueError(f"MZI {target} not found in mzi_table.")

    entry = mzi_table[key]
    ports = entry.get("ports", [])
    if not isinstance(ports, list) or len(ports) != 2:
        raise ValueError(f"MZI {target} must have exactly two ports.")
    ports = [int(p) for p in ports]
    r_values = [float(up_R), float(down_R)]
    halfpi_powers = [float(up_P), float(down_P)]
    if any(value <= 0 or not np.isfinite(value) for value in r_values):
        raise ValueError("up_R and down_R must be positive finite values.")
    if any(value <= 0 or not np.isfinite(value) for value in halfpi_powers):
        raise ValueError("up_P and down_P must be positive finite values.")

    output_dir = os.path.join("Scandata", "BW")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_savepath = os.path.join(output_dir, f"{ports}.txt")
    image_savepath = os.path.join(output_dir, f"{ports}.png")

    def get_current_power(port, r_value):
        port_idx = port - 1
        if port_idx < 0 or port_idx >= len(file_path_df):
            raise IndexError(f"PORT {port} is out of range for the provided file_data.")
        v_current = float(file_path_df.iloc[port_idx, 0])
        return (v_current**2) / r_value

    def clamp_voltage(value):
        return round(max(0.0, min(6.0, float(value))), 3)

    POWER_LIMIT_W = 0.055

    def power_to_voltage(power_w, resistance):
        power_w = max(0.0, min(POWER_LIMIT_W, float(power_w)))
        return float(np.sqrt(power_w * float(resistance)))

    def fold_power_to_limit(power_w, period_w):
        power_w = float(power_w)
        period_w = float(period_w)
        if period_w <= 0:
            raise ValueError("Fold period must be positive.")
        while power_w > POWER_LIMIT_W:
            power_w -= period_w
        return max(0.0, power_w)

    def read_output_power_uW():
        power_str_list = cu.read_pow(pwm)
        try:
            return float(power_str_list[int(out_num) - 1]) * 1e6
        except (ValueError, IndexError) as e:
            print(f"Read power error: {e}")
            return 0

    port_primary = ports[0]
    port_secondary = ports[1]
    r_primary = r_values[0]
    r_secondary = r_values[1]
    ppi_primary = halfpi_powers[0]
    ppi_secondary = halfpi_powers[1]
    period_primary = 2.0 * ppi_primary
    period_secondary = 2.0 * ppi_secondary
    p_primary_base = get_current_power(port_primary, r_primary)
    p_secondary_base = get_current_power(port_secondary, r_secondary)

    def measure_at_voltages(v_primary, v_secondary, label=""):
        v_primary = clamp_voltage(v_primary)
        v_secondary = clamp_voltage(v_secondary)
        write_port_voltage(port_primary, v_primary, file_path_df)
        write_port_voltage(port_secondary, v_secondary, file_path_df)
        cu.upload_voltage(ser, file_path_df)
        time.sleep(measure_time)
        power_value = read_output_power_uW()
        if label:
            print(f"{label} voltages: [{v_primary}, {v_secondary}], optical power {power_value} uW")
        return power_value, v_primary, v_secondary

    def shift_voltage(v_primary, v_secondary, adjust_arm, delta_v):
        if adjust_arm == "up":
            next_v_primary = clamp_voltage(v_primary + delta_v)
            return next_v_primary, v_secondary, next_v_primary != v_primary
        if adjust_arm == "down":
            next_v_secondary = clamp_voltage(v_secondary + delta_v)
            return v_primary, next_v_secondary, next_v_secondary != v_secondary
        raise ValueError("adjust_arm must be 'up' or 'down'.")

    def search_same_power_after_fold(target_power, start_v_primary, start_v_secondary, adjust_arm):
        current_v_primary = clamp_voltage(start_v_primary)
        current_v_secondary = clamp_voltage(start_v_secondary)
        current_power, current_v_primary, current_v_secondary = measure_at_voltages(
            current_v_primary,
            current_v_secondary,
            "Fold search start",
        )
        current_diff = abs(current_power - target_power)
        current_delta = current_power - target_power
        best_v_primary = current_v_primary
        best_v_secondary = current_v_secondary
        best_power = current_power
        best_diff = current_diff
        direction = 1.0

        for step in (0.01, 0.001):
            trial_v_primary, trial_v_secondary, moved = shift_voltage(
                current_v_primary,
                current_v_secondary,
                adjust_arm,
                direction * step,
            )
            if not moved:
                direction *= -1.0
                trial_v_primary, trial_v_secondary, moved = shift_voltage(
                    current_v_primary,
                    current_v_secondary,
                    adjust_arm,
                    direction * step,
                )
            if not moved:
                continue

            trial_power, trial_v_primary, trial_v_secondary = measure_at_voltages(
                trial_v_primary,
                trial_v_secondary,
                f"Fold search step {step:.3f}",
            )
            trial_diff = abs(trial_power - target_power)
            if trial_diff > current_diff:
                direction *= -1.0
                trial_v_primary, trial_v_secondary, moved = shift_voltage(
                    current_v_primary,
                    current_v_secondary,
                    adjust_arm,
                    direction * step,
                )
                if not moved:
                    continue
                trial_power, trial_v_primary, trial_v_secondary = measure_at_voltages(
                    trial_v_primary,
                    trial_v_secondary,
                    f"Fold search step {step:.3f}",
                )
                trial_diff = abs(trial_power - target_power)
                if trial_diff > current_diff:
                    continue

            previous_delta = current_delta
            current_v_primary = trial_v_primary
            current_v_secondary = trial_v_secondary
            current_power = trial_power
            current_diff = trial_diff
            current_delta = current_power - target_power
            if current_diff < best_diff:
                best_v_primary = current_v_primary
                best_v_secondary = current_v_secondary
                best_power = current_power
                best_diff = current_diff

            max_steps = int(6.0 / step) + 2
            for _ in range(max_steps):
                next_v_primary, next_v_secondary, moved = shift_voltage(
                    current_v_primary,
                    current_v_secondary,
                    adjust_arm,
                    direction * step,
                )
                if not moved:
                    break

                next_power, next_v_primary, next_v_secondary = measure_at_voltages(
                    next_v_primary,
                    next_v_secondary,
                    f"Fold search step {step:.3f}",
                )
                next_diff = abs(next_power - target_power)
                next_delta = next_power - target_power
                if next_diff < best_diff:
                    best_v_primary = next_v_primary
                    best_v_secondary = next_v_secondary
                    best_power = next_power
                    best_diff = next_diff

                crossed_target = previous_delta * next_delta <= 0
                if crossed_target or next_diff > current_diff:
                    current_v_primary = best_v_primary
                    current_v_secondary = best_v_secondary
                    current_power = best_power
                    current_diff = best_diff
                    current_delta = current_power - target_power
                    direction *= -1.0
                    break

                previous_delta = next_delta
                current_v_primary = next_v_primary
                current_v_secondary = next_v_secondary
                current_power = next_power
                current_diff = next_diff
                current_delta = next_delta

        print(
            f"Fold matched power: target {target_power} uW, best {best_power} uW "
            f"at [{best_v_primary}, {best_v_secondary}] V"
        )
        return best_v_primary, best_v_secondary

    data = []
    phase_step = min(0.1, np.pi * 0.001 / max(ppi_primary, ppi_secondary))
    phase_step = max(phase_step, 0.001)
    phase_values = np.arange(0.0, 2 * np.pi + phase_step * 0.5, phase_step)
    if phase_values.size == 0 or phase_values[-1] < 2 * np.pi - 1e-9:
        phase_values = np.append(phase_values, 2 * np.pi)

    last_phase = None
    last_power_value = None
    last_v_primary = None
    last_v_secondary = None
    last_p_primary = None
    last_p_secondary = None
    for phase in phase_values:
        primary_increment = float(phase / np.pi * ppi_primary)
        secondary_increment = float(phase / np.pi * ppi_secondary)
        p_primary_target = p_primary_base + primary_increment
        p_secondary_target = p_secondary_base + secondary_increment

        primary_over_limit = p_primary_target > POWER_LIMIT_W
        secondary_over_limit = p_secondary_target > POWER_LIMIT_W
        if primary_over_limit or secondary_over_limit:
            if (
                last_power_value is not None
                and last_phase is not None
                and last_v_primary is not None
                and last_v_secondary is not None
                and last_p_primary is not None
                and last_p_secondary is not None
            ):
                last_primary_increment = float(last_phase / np.pi * ppi_primary)
                last_secondary_increment = float(last_phase / np.pi * ppi_secondary)
                folded_primary_power = last_p_primary
                folded_secondary_power = last_p_secondary
                if primary_over_limit:
                    folded_primary_power = fold_power_to_limit(last_p_primary - period_primary, period_primary)
                if secondary_over_limit:
                    folded_secondary_power = fold_power_to_limit(last_p_secondary - period_secondary, period_secondary)

                start_v_primary = power_to_voltage(folded_primary_power, r_primary)
                start_v_secondary = power_to_voltage(folded_secondary_power, r_secondary)
                adjust_arm = "up" if primary_over_limit else "down"
                fold_v_primary, fold_v_secondary = search_same_power_after_fold(
                    last_power_value,
                    start_v_primary,
                    start_v_secondary,
                    adjust_arm,
                )
                folded_primary_power = (float(fold_v_primary) ** 2) / r_primary
                folded_secondary_power = (float(fold_v_secondary) ** 2) / r_secondary
                p_primary_base = folded_primary_power - last_primary_increment
                p_secondary_base = folded_secondary_power - last_secondary_increment
                p_primary_target = p_primary_base + primary_increment
                p_secondary_target = p_secondary_base + secondary_increment
            else:
                p_primary_target = fold_power_to_limit(p_primary_target, period_primary)
                p_secondary_target = fold_power_to_limit(p_secondary_target, period_secondary)

        p_primary_target = fold_power_to_limit(p_primary_target, period_primary)
        p_secondary_target = fold_power_to_limit(p_secondary_target, period_secondary)

        v_primary = power_to_voltage(p_primary_target, r_primary)
        v_secondary = power_to_voltage(p_secondary_target, r_secondary)
        v_primary = clamp_voltage(v_primary)
        v_secondary = clamp_voltage(v_secondary)

        voltages = [v_primary, v_secondary]
        write_port_voltage(port_primary, v_primary, file_path_df)
        write_port_voltage(port_secondary, v_secondary, file_path_df)

        print(
            f"phase {phase:.6f} rad, powers: [{p_primary_target:.6f}, {p_secondary_target:.6f}] W, "
            f"voltages: {voltages}"
        )
        cu.upload_voltage(ser, file_path_df)
        time.sleep(measure_time)

        power_value = read_output_power_uW()

        print(f"Optical power {power_value} uW")

        p_primary_actual = (float(v_primary) ** 2) / r_primary
        p_secondary_actual = (float(v_secondary) ** 2) / r_secondary
        data.append([phase, power_value, v_primary, v_secondary, p_primary_actual, p_secondary_actual])
        last_phase = float(phase)
        last_power_value = power_value
        last_v_primary = v_primary
        last_v_secondary = v_secondary
        last_p_primary = p_primary_actual
        last_p_secondary = p_secondary_actual

    header = "phase(rad),pow(uW),v_up,v_down,p_up,p_down"
    np.savetxt(data_savepath, data, fmt="%.12f", delimiter=",", header=header, comments="")

    phase_list = [row[0] for row in data]
    pow_list = [row[1] for row in data]
    plt.plot(phase_list, pow_list, marker="o")
    plt.xlabel("Synchronous Phase (rad)")
    plt.ylabel("pow (uW)")
    plt.title(f"Power vs Synchronous Phase for Ports {ports}")
    max_pow = max(pow_list) if pow_list else 0.0
    plt.ylim(bottom=0, top=max_pow + 10.0)
    plt.grid(True)
    plt.savefig(image_savepath)
    plt.close()


def fit_to_half(mzi):
    """
    Return one MZI's saved half-power voltage from MZI_table.
    """
    mzi_key = str(int(mzi))
    table = globals().get("MZI_TABLE")
    if table is None:
        table = globals().get("mzi_table")
    if table is None:
        table = load_mzi_table()
        globals()["MZI_TABLE"] = table
    if mzi_key not in table:
        raise ValueError(f"MZI {mzi_key} not found in mzi_table.")

    entry = table[mzi_key]
    ports = entry.get("ports", [])
    half_power_values = entry.get("half_power", [])
    if not ports or not half_power_values:
        raise ValueError(f"MZI {mzi_key} missing half_power in mzi_table.")

    port = int(ports[0])
    half_voltage = float(half_power_values[0])
    if half_voltage <= 0 or not np.isfinite(half_voltage):
        raise ValueError(f"MZI {mzi_key} has invalid half_power voltage in mzi_table.")

    print(f"MZI {mzi_key} port {port} half-power voltage from MZI_table: {half_voltage:.3f} V")
    return half_voltage


def fit_inter_cali_sine(
    target: int,
    show_plot: bool = True,
):
    """
    Fit P = A*sin(w*dp + phi) + B from inter_cali_powerdata based on target ports.
    Returns (A, w, phi, B).
    """
    data_dir = os.path.join("Scandata", "BW")
    mzi_table = globals().get("mzi_table")
    if mzi_table is None:
        mzi_table = load_mzi_table()
    key = str(int(target))
    if key not in mzi_table:
        raise ValueError(f"MZI {target} not found in mzi_table.")
    ports = mzi_table[key].get("ports", [])
    if not isinstance(ports, list) or not ports:
        raise ValueError(f"ports for MZI {target} is not a non-empty list.")

    file_path = _resolve_inter_cali_file(ports, data_dir)
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expect at least two columns in {file_path}")

    dp = df.iloc[:, 0].to_numpy(dtype=float)
    power = df.iloc[:, 1].to_numpy(dtype=float)
    if dp.size < 3:
        raise ValueError(f"Not enough samples in {file_path}")

    def sin_model(x, A, w, phi, B):
        return A * np.sin(w * x + phi) + B

    A_guess = 0.5 * (power.max() - power.min())
    B_guess = float(np.mean(power))
    w_guess = _estimate_sine_w(dp, power)
    phi_guess = 0.0

    popt, _ = curve_fit(
        sin_model,
        dp,
        power,
        p0=[A_guess, w_guess, phi_guess, B_guess],
        bounds=([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=20000,
    )
    A, w, phi, B = popt

    dp_smooth = np.linspace(dp.min(), dp.max(), 500)
    plt.figure(figsize=(7, 5))
    plt.plot(dp, power, "o", label="samples")
    plt.plot(dp_smooth, sin_model(dp_smooth, A, w, phi, B), "-", label="fit")
    plt.xlabel("dp")
    plt.ylabel("P")
    plt.title(f"Target {target} Ports {ports}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    image_dir = os.path.join("Scandata", "BW")
    os.makedirs(image_dir, exist_ok=True)
    ports_tag = "-".join(str(int(p)) for p in ports)
    image_path = os.path.join(image_dir, f"target_{int(target)}_ports_{ports_tag}.png")
    plt.savefig(image_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()

    return A, w, phi, B


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


if __name__ == "__main__":
    N = 9
    print(du.Clements_matrix(N))
    working_data = cu.generate_working_data()
    MZI_TABLE = load_mzi_table()
    measure_time = 1

    OPM1_ADDRESS = "TCPIP0::192.168.0.5::inst0::INSTR"
    OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
    SER_ADDRESS = "COM3"
    opm1 = cu.open_VISA_connection(OPM1_ADDRESS)
    opm2 = cu.open_VISA_connection(OPM2_ADDRESS)
    mcv = cu.open_ser_connection(SER_ADDRESS)

    for n in range(2, N // 2 + 1):
        path, state, input, output, bmzi = find_path_BottomWaveguide(N, n)
        up_R, up_P, down_R, down_P = Power_halfpi(bmzi)
        print(
            f"{Fore.BLUE}Target {bmzi} halfpi calibration loaded: up_R={up_R:.6f} Ohm, up_P={up_P:.9f} W, down_R={down_R:.6f} Ohm, down_P={down_P:.9f} W"
        )

        print(f"{Fore.GREEN}Path: {path}")
        print(f"{Fore.GREEN}State: {state}")
        print(f"{Fore.GREEN}Input: {input}")
        print(f"{Fore.GREEN}Output: {output}")
        print(f"{Fore.GREEN}BMZI: {bmzi}")

        for i in range(len(path)):
            if state[i] == "C":
                write_port_voltage(
                    MZI_TABLE[str(path[i])]["ports"][0],
                    MZI_TABLE[str(path[i])]["dtheta"][1],
                    working_data,
                )
            elif state[i] == "B":
                write_port_voltage(
                    MZI_TABLE[str(path[i])]["ports"][0],
                    MZI_TABLE[str(path[i])]["dtheta"][0],
                    working_data,
                )
            elif state[i] == "H":
                port = int(MZI_TABLE[str(path[i])]["ports"][0])
                half_voltage = float(fit_to_half(path[i]))
                write_port_voltage(port, half_voltage, working_data)
        for j in range(N - 1):
            switch_IN(j + 1, "OFF", working_data)
        switch_IN(input, "ON", working_data)
        write_port_voltage(MZI_TABLE[str(bmzi)]["ports"][0], MZI_TABLE[str(bmzi)]["dtheta"][0], working_data)
        scan_mzis(bmzi, mcv, opm2, measure_time, output, working_data, up_R, up_P, down_R, down_P)
        write_port_voltage(MZI_TABLE[str(bmzi)]["ports"][1], 0, working_data)
        fit_inter_cali_sine(bmzi, show_plot=True)
