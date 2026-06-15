import os
import json
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.communication as cu
import utils.AllDecompositionUtils as du
from colorama import Fore, Style


class MZI:
    def __init__(self, theta1=None, theta2=None):
        if theta1 is None:
            theta1 = np.random.uniform(0, 2 * np.pi)
        if theta2 is None:
            theta2 = np.random.uniform(0, 2 * np.pi)
        self.theta1 = theta1
        self.theta2 = theta2

    def forward(self):
        T11 = np.sin((self.theta1 - self.theta2) / 2)
        T12 = np.cos((self.theta1 - self.theta2) / 2)
        T21 = np.cos((self.theta1 - self.theta2) / 2)
        T22 = -np.sin((self.theta1 - self.theta2) / 2)
        T = 1j * np.exp(1j * (self.theta1 + self.theta2) / 2) * np.array([[T11, T12], [T21, T22]], dtype=np.complex64)
        return T


def net_T(mzi_net, mzis_param):
    T = np.eye(len(mzi_net) + 1, dtype=complex)
    for j in range(mzi_net.shape[1]):
        C = np.eye(len(mzi_net) + 1, dtype=complex)
        for i in range(mzi_net.shape[0]):
            if mzi_net[i][j] > 0:
                # print(mzi_net[i][j], mzis_param[mzi_net[i][j] - 1])
                mzi = MZI(mzis_param[mzi_net[i][j] - 1][0], mzis_param[mzi_net[i][j] - 1][1])
                A = mzi.forward()
                B = du.embed_2x2(A, i + 1, i + 2, len(mzi_net) + 1)
                C = C @ B
            elif mzi_net[i][j] < 0:
                C[-1, -1] = np.exp(1j * BW[j // 2 - 1])
        T = C @ T
    return T


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


def get_mzi_entry(mzi_table, mzi_id):
    key = str(int(mzi_id))
    entry = mzi_table.get(key)
    if entry is None:
        raise KeyError(f"MZI {int(mzi_id)} not found in MZI_table.json.")
    return entry


def get_mzi_port(mzi_table, mzi_id, arm_index=0):
    entry = get_mzi_entry(mzi_table, mzi_id)
    ports = entry.get("ports", [])
    arm_index = int(arm_index)
    if arm_index < 0 or arm_index >= len(ports):
        raise ValueError(f"MZI {int(mzi_id)} has no port for arm_index {arm_index}.")
    return int(ports[arm_index])


def get_mzi_state_voltage(mzi_table, mzi_id, state, arm_index=0):
    entry = get_mzi_entry(mzi_table, mzi_id)
    arm_index = int(arm_index)
    state_norm = str(state).strip().upper()
    if state_norm in {"B", "BAR"}:
        values = entry.get("dtheta_Bar", entry.get("dtheta", []))
    elif state_norm in {"C", "CROSS"}:
        values = entry.get("dtheta_Cross", entry.get("dtheta", []))
        if "dtheta_Cross" not in entry and len(values) > 1:
            return float(values[1])
    else:
        raise ValueError("state must be 'B'/'BAR' or 'C'/'CROSS'.")

    if arm_index < 0 or arm_index >= len(values):
        raise ValueError(f"MZI {int(mzi_id)} has no {state_norm} voltage for arm_index {arm_index}.")
    return float(values[arm_index])


def _bw_txt_sort_key(txt_path):
    name = os.path.basename(os.fspath(txt_path))
    numbers = [int(num) for num in re.findall(r"-?\d+", name)]
    return numbers, name


def fit_bw_phi_from_txt(txt_path):
    data = np.loadtxt(txt_path, delimiter=",", skiprows=1, usecols=(0, 1))
    x = data[:, 0]
    y = data[:, 1]

    def cos_model(x_value, A, w, phi):
        return A * (1 + np.cos(w * x_value - phi)) / 2

    from scipy.optimize import curve_fit, differential_evolution

    def sse(params):
        A, w, phi = params
        residual = y - cos_model(x, A, w, phi)
        return float(np.sum(residual**2))

    max_power = float(np.max(y))
    bounds = [(0.0, max(10.0, max_power * 2.5)), (0.1, 5.0), (-2 * np.pi, 2 * np.pi)]
    result = differential_evolution(sse, bounds, tol=1e-10, polish=True, seed=1)
    params, _ = curve_fit(cos_model, x, y, p0=result.x, maxfev=100000)
    phi = float(params[2])
    return (phi + np.pi) % (2 * np.pi) - np.pi


def load_bw_phases(bw_dir=os.path.join("Scandata", "BW"), expected_count=None):
    txt_paths = [os.path.join(bw_dir, name) for name in os.listdir(bw_dir) if name.lower().endswith(".txt")]
    txt_paths.sort(key=_bw_txt_sort_key)

    if expected_count is not None and len(txt_paths) != expected_count:
        raise ValueError(f"Expected {expected_count} BW txt files in {bw_dir}, found {len(txt_paths)}.")

    return np.array([fit_bw_phi_from_txt(path) for path in txt_paths], dtype=float)


def upload_v_checked(mcv, working_data, v_min, v_max):
    """
    Upload voltages only if all values in working_data are within [v_min, v_max].
    """
    if working_data is None:
        raise ValueError("working_data must not be None.")
    if v_min > v_max:
        raise ValueError("v_min must be <= v_max.")
    if 0 not in working_data.columns:
        raise ValueError("working_data must contain voltage column 0.")

    voltages = pd.to_numeric(working_data[0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltage values.")

    out_of_range = (voltages < v_min) | (voltages > v_max)
    if out_of_range.any():
        bad_idx = out_of_range[out_of_range].index.tolist()
        raise ValueError(f"Voltage out of range [{v_min}, {v_max}] at rows: {bad_idx}")

    cu.upload_voltage(mcv, working_data)


def switch_IN(mzi_index, state, working_data):
    table = pd.read_csv("IN_MZI.txt")
    state_norm = state.strip().upper()
    if state_norm not in {"ON", "OFF"}:
        raise ValueError("state must be 'ON' or 'OFF'")
    row = table.loc[table["MZI"] == mzi_index]
    if row.empty:
        raise ValueError(f"MZI index {mzi_index} not found in IN_MZI.txt")
    port = int(row.iloc[0]["PORT"])
    voltage = float(row.iloc[0][state_norm])
    cu.write_port_voltage(port, voltage, working_data)
    return voltage


def get_T(N, working_data, sleep_time=0.5, show_figure=True, figure_path=None, title=None, v_min=None, v_max=None):
    print(Fore.CYAN + Style.BRIGHT + f"Measuring T matrix for N={N}..." + Style.RESET_ALL)
    if working_data is None:
        raise ValueError("working_data must not be None.")
    if N < 2:
        raise ValueError("N must be >= 2.")
    for name in ("opm2", "mcv"):
        if name not in globals() or globals()[name] is None:
            raise RuntimeError(f"{name} is not initialized.")

    def read_powers():
        power_str_list = cu.read_pow(opm2)
        powers = []
        for idx, val in enumerate(power_str_list):
            try:
                powers.append(float(val))
            except ValueError as exc:
                raise ValueError(f"Invalid power at channel {idx + 1}: {val}") from exc
        return powers

    input_channels = list(range(1, N))
    cols = []
    output_count = None

    for in_ch in input_channels:
        for ch in input_channels:
            switch_IN(ch, "OFF", working_data)
        switch_IN(in_ch, "ON", working_data)
        if v_min is None or v_max is None:
            cu.upload_voltage(mcv, working_data)
        else:
            upload_v_checked(mcv, working_data, v_min, v_max)
        time.sleep(sleep_time)

        powers = read_powers()
        if output_count is None:
            output_count = len(powers)
            if output_count == 0:
                raise ValueError("No power channels read from opm2.")
        col = np.array(powers[:output_count], dtype=float)
        total = float(np.sum(col))
        if total != 0.0:
            col = col / total
        cols.append(col)

    if not cols:
        raise ValueError("No input channels processed.")

    T = np.column_stack(cols)

    plt.figure(figsize=(6, 5))
    plt.imshow(T, aspect="auto", origin="upper", cmap="viridis")
    plt.yticks(range(T.shape[0]))
    plt.colorbar(label="Normalized power")
    plt.xlabel("Input channel")
    plt.ylabel("Output channel")
    plt.title(title or "Power transfer matrix")
    plt.tight_layout()
    if figure_path is not None:
        figure_dir = os.path.dirname(os.fspath(figure_path))
        if figure_dir:
            os.makedirs(figure_dir, exist_ok=True)
        plt.savefig(figure_path, dpi=200)
        print(f"Saved power matrix figure to {figure_path}")
    if show_figure:
        plt.show()
    plt.close()

    return T


def print_matrix(matrix, decimals=5):
    if matrix is None:
        print("None")
        return
    for row in np.asarray(matrix):
        print(" ".join(f"{val:.{decimals}f}" for val in row))


def normalize_power_vector(values):
    power = np.asarray(values, dtype=float)
    total = float(np.sum(power))
    if total != 0.0:
        return power / total
    return power


def read_normalized_power(opm, output_count):
    power_str_list = cu.read_pow(opm)
    powers = []
    for idx, val in enumerate(power_str_list[:output_count]):
        try:
            powers.append(float(val))
        except ValueError as exc:
            raise ValueError(f"Invalid power at channel {idx + 1}: {val}") from exc
    return normalize_power_vector(powers)


def theoretical_power_matrix(mzi_net, mzis_param, output_count):
    power = np.abs(net_T(mzi_net, mzis_param)[:output_count, :output_count]) ** 2
    col_sum = np.sum(power, axis=0, keepdims=True)
    return np.divide(power, col_sum, out=np.zeros_like(power), where=col_sum != 0.0)


def theta_delta(current, target):
    return float((float(target) - float(current) + np.pi) % (2 * np.pi) - np.pi)


def build_theta_need_change(thetas, thetas_target, arm_index=0, tol=1e-9):
    if thetas.shape != thetas_target.shape:
        raise ValueError("thetas and thetas_target must have the same shape.")
    theta_need_change = []
    for theta_index in range(thetas.shape[0]):
        current = float(thetas[theta_index, arm_index])
        target = float(thetas_target[theta_index, arm_index])
        delta = theta_delta(current, target)
        if abs(delta) <= tol:
            continue
        theta_need_change.append(
            {
                "mzi_id": theta_index + 1,
                "theta_index": theta_index,
                "arm_index": arm_index,
                "theta_current": current,
                "theta_target": target,
                "theta_delta": delta,
            }
        )
    return theta_need_change


def find_changed_input_column(initial_power, changed_power, tol=1e-9):
    initial_power = np.asarray(initial_power, dtype=float)
    changed_power = np.asarray(changed_power, dtype=float)
    if initial_power.shape != changed_power.shape:
        raise ValueError("initial_power and changed_power must have the same shape.")

    column_diff = np.linalg.norm(changed_power - initial_power, axis=0)
    changed_columns = np.where(column_diff > tol)[0]
    if changed_columns.size == 0:
        raise ValueError("No changed input column found from theoretical matrices.")
    selected_col = int(changed_columns[np.argmax(column_diff[changed_columns])])
    return selected_col, changed_columns.tolist(), column_diff


def circular_distance(a, b):
    return abs((float(a) - float(b) + np.pi) % (2 * np.pi) - np.pi)


def is_extreme_phase(phase_value, tol=1e-4):
    return (
        min(
            circular_distance(phase_value, 0.0),
            circular_distance(phase_value, np.pi),
        )
        <= tol
    )


def build_theoretical_slope_info(
    mzi_net,
    target_thetas,
    change,
    input_col,
    output_count,
    phase_delta=1e-3,
    slope_tol=1e-9,
    extreme_phase_tol=1e-4,
):
    theta_index = int(change["theta_index"])
    arm_index = int(change["arm_index"])
    target_thetas = np.asarray(target_thetas, dtype=float)

    theta_minus = target_thetas.copy()
    theta_plus = target_thetas.copy()
    theta_minus[theta_index, arm_index] -= phase_delta
    theta_plus[theta_index, arm_index] += phase_delta

    ratio_minus = theoretical_power_matrix(mzi_net, theta_minus, output_count)[:, input_col]
    ratio_plus = theoretical_power_matrix(mzi_net, theta_plus, output_count)[:, input_col]
    slope = (ratio_plus - ratio_minus) / (2 * phase_delta)
    slope_channel = int(np.argmax(np.abs(slope)))
    slope_value = float(slope[slope_channel])
    phase_value = float(target_thetas[theta_index, 0] - target_thetas[theta_index, 1])
    is_extreme = is_extreme_phase(phase_value, tol=extreme_phase_tol)
    requires_slope = (not is_extreme) and abs(slope_value) > slope_tol

    return {
        "requires_slope": bool(requires_slope),
        "is_extreme": bool(is_extreme),
        "phase_delta": float(phase_delta),
        "phase_value": phase_value,
        "slope_channel": slope_channel,
        "slope_channel_1based": slope_channel + 1,
        "slope_value": slope_value,
        "slope_sign": int(np.sign(slope_value)) if requires_slope else 0,
        "slope_vector": [float(x) for x in slope],
        "ratio_minus": [float(x) for x in ratio_minus],
        "ratio_plus": [float(x) for x in ratio_plus],
    }


def set_single_input(input_port, input_count, working_data):
    for ch in range(1, input_count + 1):
        switch_IN(ch, "OFF", working_data)
    switch_IN(input_port, "ON", working_data)


def restore_working_state(working_data, base_working_data, input_count, mcv, v_min, v_max):
    working_data.iloc[:, 0] = base_working_data.iloc[:, 0].to_numpy(copy=True)
    for ch in range(1, input_count + 1):
        switch_IN(ch, "OFF", working_data)
    upload_v_checked(mcv, working_data, v_min, v_max)


def measure_voltage_error(
    mcv,
    opm,
    working_data,
    adjust_port,
    voltage,
    input_port,
    input_count,
    target_ratio,
    v_min,
    v_max,
    settle_time,
):
    cu.write_port_voltage(adjust_port, voltage, working_data)
    set_single_input(input_port, input_count, working_data)
    upload_v_checked(mcv, working_data, v_min, v_max)
    time.sleep(settle_time)

    measured_ratio = read_normalized_power(opm, input_count)
    error = float(np.linalg.norm(measured_ratio - target_ratio))
    return error, measured_ratio


def build_voltage_grid(v_min, v_max, step):
    start = np.ceil((float(v_min) - 1e-12) / step) * step
    stop = np.floor((float(v_max) + 1e-12) / step) * step
    count = int(round((stop - start) / step)) + 1
    values = [round(start + i * step, 3) for i in range(max(count, 0))]
    return sorted({v for v in values if v_min <= v <= v_max})


def mark_slope_matches(points, slope_info, slope_power_tol=1e-10):
    if not points:
        return

    requires_slope = bool(slope_info.get("requires_slope", False))
    slope_channel = int(slope_info.get("slope_channel", 0))
    theory_sign = int(slope_info.get("slope_sign", 0))

    for idx, point in enumerate(points):
        point["slope_power"] = float(point["ratio"][slope_channel])
        if not requires_slope:
            point["measured_slope"] = 0.0
            point["measured_slope_sign"] = 0
            point["slope_matches"] = True
            continue

        if len(points) < 2:
            measured_slope = 0.0
        elif idx == 0:
            measured_slope = (float(points[1]["ratio"][slope_channel]) - float(points[0]["ratio"][slope_channel])) / (
                points[1]["voltage"] - points[0]["voltage"]
            )
        elif idx == len(points) - 1:
            measured_slope = (float(points[-1]["ratio"][slope_channel]) - float(points[-2]["ratio"][slope_channel])) / (
                points[-1]["voltage"] - points[-2]["voltage"]
            )
        else:
            measured_slope = (
                float(points[idx + 1]["ratio"][slope_channel]) - float(points[idx - 1]["ratio"][slope_channel])
            ) / (points[idx + 1]["voltage"] - points[idx - 1]["voltage"])

        if abs(measured_slope) <= slope_power_tol:
            measured_sign = 0
        else:
            measured_sign = int(np.sign(measured_slope))

        point["measured_slope"] = float(measured_slope)
        point["measured_slope_sign"] = measured_sign
        point["slope_matches"] = measured_sign == theory_sign


def scan_voltage_grid(
    mcv,
    opm,
    working_data,
    adjust_port,
    input_port,
    target_ratio,
    input_count,
    voltages,
    slope_info,
    v_min,
    v_max,
    settle_time,
    scan_step=None,
):
    points = []
    for voltage in sorted(set(voltages)):
        error, measured_ratio = measure_voltage_error(
            mcv,
            opm,
            working_data,
            adjust_port,
            voltage,
            input_port,
            input_count,
            target_ratio,
            v_min,
            v_max,
            settle_time,
        )
        if scan_step is not None:
            print(
                f"Scan step={float(scan_step):g}, port {adjust_port}, "
                f"voltage={float(voltage):.3f} V, error={error:.6g}"
            )
        time.sleep(0.5)
        points.append(
            {
                "voltage": float(voltage),
                "error": float(error),
                "ratio": measured_ratio,
            }
        )
    mark_slope_matches(points, slope_info)
    return points


def tune_port_voltage_by_steps(
    mcv,
    opm,
    working_data,
    adjust_port,
    input_port,
    target_ratio,
    input_count,
    v_min=0.0,
    v_max=5.5,
    step_sizes=(0.1, 0.01, 0.001),
    settle_time=0.2,
    slope_info=None,
):
    target_ratio = normalize_power_vector(target_ratio)
    port_idx = int(adjust_port) - 1
    if port_idx < 0 or port_idx >= len(working_data):
        raise IndexError(f"PORT {adjust_port} is out of range for working_data.")

    if slope_info is None:
        slope_info = {"requires_slope": False, "slope_channel": 0, "slope_sign": 0}

    if slope_info.get("requires_slope", False):
        print(
            f"Slope filter: output {int(slope_info['slope_channel_1based'])}, "
            f"theoretical slope={float(slope_info['slope_value']):.6g}"
        )
    elif slope_info.get("is_extreme", False):
        print("Slope filter skipped: target phase is close to 0/pi extremum.")
    else:
        print("Slope filter skipped: theoretical slope is too small.")

    best_point = None
    prev_step = None
    for step_idx, step in enumerate(step_sizes):
        if step_idx == 0 or best_point is None:
            search_min = float(v_min)
            search_max = float(v_max)
        else:
            radius = float(prev_step)
            search_min = max(float(v_min), best_point["voltage"] - radius)
            search_max = min(float(v_max), best_point["voltage"] + radius)

        voltages = build_voltage_grid(search_min, search_max, step)
        if best_point is not None:
            voltages.append(round(best_point["voltage"], 3))
        if not voltages:
            raise ValueError(f"No voltage candidates for step {step} in [{search_min}, {search_max}].")

        points = scan_voltage_grid(
            mcv,
            opm,
            working_data,
            adjust_port,
            input_port,
            target_ratio,
            input_count,
            voltages,
            slope_info,
            v_min,
            v_max,
            settle_time,
            scan_step=step,
        )
        eligible_points = [point for point in points if point["slope_matches"]]
        if not eligible_points:
            raise ValueError(
                f"No slope-matched voltage candidate for port {adjust_port} at step {step}. "
                f"Theoretical slope sign={slope_info.get('slope_sign')}."
            )

        best_point = min(eligible_points, key=lambda point: point["error"])
        prev_step = step
        print(
            f"  step={step:g}, search=[{search_min:.3f}, {search_max:.3f}], "
            f"best V={best_point['voltage']:.3f}, error={best_point['error']:.6g}, "
            f"slope_sign={best_point['measured_slope_sign']}"
        )

    best_v = round(float(best_point["voltage"]), 3)
    best_error, best_ratio = measure_voltage_error(
        mcv,
        opm,
        working_data,
        adjust_port,
        best_v,
        input_port,
        input_count,
        target_ratio,
        v_min,
        v_max,
        settle_time,
    )
    return best_v, best_error, best_ratio


if __name__ == "__main__":
    N = 9
    v_min = 0
    v_max = 5.5
    input_count = N - 1
    working_data = cu.generate_working_data()
    mzi_table = load_mzi_table()

    # OPM1_ADDRESS = "TCPIP0::192.168.0.5::inst0::INSTR"
    OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
    SER_ADDRESS = "COM3"
    # opm1 = cu.open_VISA_connection(OPM1_ADDRESS)
    opm2 = cu.open_VISA_connection(OPM2_ADDRESS)
    mcv = cu.open_ser_connection(SER_ADDRESS)

    cm = du.Clements_matrix(N)
    print(cm)
    cm[-1, 2] = -1
    cm[-1, 4] = -1
    cm[-1, 6] = -1
    print(cm)
    thetas = np.zeros((N * (N - 1) // 2, 2))
    thetas_target = np.zeros((N * (N - 1) // 2, 2))
    for i in range(N * (N - 1) // 2):
        if (i + 1) not in cm[:, -1] and (i + 1) not in cm[:, 0]:
            thetas[i][0] = np.pi
            thetas[i][1] = 0
            thetas_target[i][0] = np.pi
            thetas_target[i][1] = 0
        else:
            thetas[i][0] = np.pi
            thetas[i][1] = 0
            thetas_target[i][0] = np.pi
            thetas_target[i][1] = 0
    thetas_target[4][0] = np.pi / 3
    thetas_target[5][0] = np.pi / 3
    thetas_target[6][0] = np.pi / 3
    # thetas_target[7][0] = np.pi / 3
    thetas_target[8][0] = np.pi / 3
    thetas_target[9][0] = np.pi / 3
    thetas_target[10][0] = np.pi / 3
    thetas_target[11][0] = np.pi / 3
    thetas_target[12][0] = np.pi / 3
    thetas_target[13][0] = np.pi / 3
    thetas_target[14][0] = np.pi / 3

    print(f"thetas: {thetas}")
    print(f"thetas_target: {thetas_target}")

    BW = load_bw_phases(os.path.join("Scandata", "BW"), expected_count=3)
    # BW = np.array([0, 0, 0])
    print(np.array2string(BW, precision=8, separator=", "))

    T_theory_initial_field = net_T(cm, thetas)[0:input_count, 0:input_count]
    T_theory_initial = theoretical_power_matrix(cm, thetas, input_count)
    print("Initial theoretical field T:")
    print(T_theory_initial_field)
    print("Initial theoretical power T:")
    print_matrix(T_theory_initial, decimals=5)

    for i in range(N * (N - 1) // 2):
        mzi_id = i + 1
        port = get_mzi_port(mzi_table, mzi_id, arm_index=0)
        bar_voltage = get_mzi_state_voltage(mzi_table, mzi_id, "B", arm_index=0)
        print(f"MZI {mzi_id} heater indices:", port, bar_voltage)
        cu.write_port_voltage(port, bar_voltage, working_data)
    for j in range(N - 1):
        switch_IN(j + 1, "OFF", working_data)

    inter_cali_pairs_path = os.path.join("Scandata", "inter_cali_pairs.json")
    if not os.path.exists(inter_cali_pairs_path):
        raise FileNotFoundError(f"Cannot find {inter_cali_pairs_path}")

    with open(inter_cali_pairs_path, "r", encoding="utf-8") as f:
        inter_cali_pairs = json.load(f)

    # for i in range(4, N * (N - 1) // 2 - 4):
    #     mzi_id = str(i + 1)
    #     pair_entry = inter_cali_pairs.get(mzi_id)
    #     if not pair_entry:
    #         print(f"Skip MZI {mzi_id}: no entry in {inter_cali_pairs_path}")
    #         continue

    #     ports = pair_entry.get("ports", [])
    #     if not isinstance(ports, list) or len(ports) != 2:
    #         raise ValueError(f"MZI {mzi_id} in {inter_cali_pairs_path} must contain two ports.")

    #     upper_v = float(pair_entry.get("upper_arm_voltage", 0.0))
    #     lower_v = float(pair_entry.get("lower_arm_voltage", 0.0))
    #     cu.write_port_voltage(int(ports[0]), upper_v, working_data)
    #     cu.write_port_voltage(int(ports[1]), lower_v, working_data)
    #     print(
    #         f"Deploy inter_cali pair MZI {mzi_id}: "
    #         f"port {int(ports[0])} -> {upper_v:.3f} V, "
    #         f"port {int(ports[1])} -> {lower_v:.3f} V"
    #     )

    upload_v_checked(mcv, working_data, v_min, v_max)
    base_working_data = working_data.copy(deep=True)

    T_initial_exp = get_T(N, working_data)
    print("Initial experimental T:")
    print_matrix(T_initial_exp, decimals=5)

    theta_need_change = build_theta_need_change(thetas, thetas_target, arm_index=0)
    print(f"theta_need_change: {theta_need_change}")

    thetas_adjusted_target = thetas.copy()
    for change in theta_need_change:
        thetas_adjusted_target[change["theta_index"], change["arm_index"]] = change["theta_target"]
    T_theory_target = theoretical_power_matrix(cm, thetas_adjusted_target, input_count)
    print("Target theoretical power T:")
    print_matrix(T_theory_target, decimals=5)

    voltage_records = {}
    adjustment_results = []
    restore_working_state(working_data, base_working_data, input_count, mcv, v_min, v_max)

    for change in theta_need_change:
        mzi_id = int(change["mzi_id"])
        adjust_port = get_mzi_port(mzi_table, mzi_id, arm_index=change["arm_index"])
        trial_thetas = thetas.copy()
        trial_thetas[change["theta_index"], change["arm_index"]] = change["theta_target"]
        T_theory_changed = theoretical_power_matrix(cm, trial_thetas, input_count)
        input_col, changed_cols, column_diff = find_changed_input_column(
            T_theory_initial,
            T_theory_changed,
        )
        input_port = input_col + 1
        target_ratio = T_theory_changed[:, input_col]
        slope_info = build_theoretical_slope_info(
            cm,
            trial_thetas,
            change,
            input_col,
            input_count,
        )

        print("=" * 60)
        print(
            f"Adjust MZI {mzi_id}, theta[{change['theta_index']}][{change['arm_index']}]: "
            f"{change['theta_current']:.6g} -> {change['theta_target']:.6g}"
        )
        print(
            f"Port {adjust_port}, changed columns {[col + 1 for col in changed_cols]}, " f"selected input {input_port}"
        )
        print("Target output ratio:")
        print_matrix(target_ratio.reshape(1, -1), decimals=5)

        best_v, best_error, measured_ratio = tune_port_voltage_by_steps(
            mcv,
            opm2,
            working_data,
            adjust_port,
            input_port,
            target_ratio,
            input_count,
            v_min=v_min,
            v_max=v_max,
            slope_info=slope_info,
        )

        voltage_records[str(adjust_port)] = best_v
        adjustment_results.append(
            {
                "mzi_id": mzi_id,
                "theta_index": int(change["theta_index"]),
                "arm_index": int(change["arm_index"]),
                "theta_current": float(change["theta_current"]),
                "theta_target": float(change["theta_target"]),
                "port": adjust_port,
                "voltage": float(best_v),
                "input_port": int(input_port),
                "changed_columns": [int(col + 1) for col in changed_cols],
                "column_diff_norm": [float(x) for x in column_diff],
                "slope_info": slope_info,
                "target_ratio": [float(x) for x in target_ratio],
                "measured_ratio": [float(x) for x in measured_ratio],
                "error": float(best_error),
            }
        )
        print(f"Recorded port {adjust_port} -> {best_v:.3f} V, error={best_error:.6g}")

        restore_working_state(working_data, base_working_data, input_count, mcv, v_min, v_max)

    records_path = os.path.join("Scandata", "theta_voltage_adjustments.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "theta_need_change": theta_need_change,
                "voltage_records": voltage_records,
                "adjustment_results": adjustment_results,
            },
            f,
            indent=2,
        )
    print(f"Saved voltage records to {records_path}")

    working_data.iloc[:, 0] = base_working_data.iloc[:, 0].to_numpy(copy=True)
    for port, voltage in voltage_records.items():
        cu.write_port_voltage(int(port), float(voltage), working_data)
    for j in range(N - 1):
        switch_IN(j + 1, "OFF", working_data)

    upload_v_checked(mcv, working_data, v_min, v_max)
    T_final_exp = get_T(N, working_data)
    print("Final experimental T:")
    print_matrix(T_final_exp, decimals=5)
