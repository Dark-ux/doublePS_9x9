import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.communication as cu
import utils.AllDecompositionUtils as du
from colorama import Fore, Style, init


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
        return json.load(f)


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


def _load_1col_fit_params(path: str = os.path.join("Scandata", "1Col", "fit_params_1col.txt")) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find 1Col fit parameter file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clip_voltage(voltage: float, v_min: float = 0.0, v_max: float = 6.0) -> float:
    return float(min(max(float(voltage), v_min), v_max))


def _round_voltage(voltage: float, digits: int = 3) -> float:
    return float(round(float(voltage), digits))


def _read_opm_channel_uw(opm, channel: int) -> float:
    power_str_list = cu.read_pow(opm)
    channel_idx = int(channel) - 1
    if channel_idx < 0 or channel_idx >= len(power_str_list):
        raise IndexError(f"OPM channel {channel} is out of range.")
    try:
        return float(power_str_list[channel_idx]) * 1e6
    except ValueError as exc:
        raise ValueError(f"Invalid optical power value on channel {channel}: {power_str_list[channel_idx]}") from exc


def _solve_max_voltage_candidates(A: float, w: float, phi: float, R: float, v_min: float = 0.0, v_max: float = 6.0):
    if R <= 0:
        raise ValueError("R must be positive.")
    if w == 0:
        raise ValueError("w must be non-zero.")

    x_min = max(0.0, v_min * v_min / R)
    x_max = max(0.0, v_max * v_max / R)
    max_phase = np.pi / 2 if A >= 0 else -np.pi / 2
    candidates = []

    for k in range(-20, 21):
        x = (max_phase - phi + 2 * np.pi * k) / w
        if x < x_min - 1e-12 or x > x_max + 1e-12 or x < 0:
            continue
        v = float(np.sqrt(max(x * R, 0.0)))
        if v_min - 1e-9 <= v <= v_max + 1e-9:
            v = float(round(v, 6))
            if all(abs(v - existing) > 1e-6 for existing in candidates):
                candidates.append(v)
    candidates.sort()
    return candidates


def _solve_voltage_candidates_for_power(
    target_power: float,
    A: float,
    w: float,
    phi: float,
    B: float,
    R: float,
    v_min: float = 0.0,
    v_max: float = 6.0,
):
    if R <= 0:
        raise ValueError("R must be positive.")
    if A == 0 or w == 0:
        raise ValueError("A and w must be non-zero.")

    x_min = max(0.0, v_min * v_min / R)
    x_max = max(0.0, v_max * v_max / R)
    arg = float(np.clip((target_power - B) / A, -1.0, 1.0))
    base = float(np.arcsin(arg))
    candidates = []

    for k in range(-20, 21):
        for angle in (base, np.pi - base):
            x = (angle - phi + 2 * np.pi * k) / w
            if x < x_min - 1e-12 or x > x_max + 1e-12 or x < 0:
                continue
            v = float(np.sqrt(max(x * R, 0.0)))
            if not (v_min - 1e-9 <= v <= v_max + 1e-9):
                continue
            slope = 0.0
            if v > 0:
                slope = float(A * np.cos(w * x + phi) * w * (2.0 * v / R))
            v = float(round(v, 6))
            if all(abs(v - item[0]) > 1e-6 for item in candidates):
                candidates.append((v, slope))

    candidates.sort(key=lambda item: item[0])
    return candidates


def find_in_p(n, power):
    """
    Find the voltage v for input n such that the measured optical power on opm1 channel n
    is closest to Pmax - power, then write that voltage into working_data and return it.
    """
    n = int(n)
    power = float(power)
    if n < 1:
        raise ValueError("n must be >= 1.")
    if power < 0:
        raise ValueError("power must be non-negative.")

    for name in ("opm1", "mcv", "working_data"):
        if name not in globals() or globals()[name] is None:
            raise RuntimeError(f"{name} is not initialized.")

    sleep_time = float(globals().get("measure_time", 0.2))
    if sleep_time < 0:
        sleep_time = 0.0

    fit_params = _load_1col_fit_params()
    fit_entry = fit_params.get(str(n))
    if fit_entry is None:
        raise KeyError(f"Cannot find fit params for input {n} in Scandata\\1Col\\fit_params_1col.txt")

    port = int(fit_entry["port"])
    R = float(fit_entry["R"])
    A = float(fit_entry["A"])
    w = float(fit_entry["w"])
    phi = float(fit_entry["phi"])
    B = float(fit_entry["B"])

    v_min = 0.0
    v_max = 6.0
    measure_cache = {}

    def apply_and_measure(voltage: float) -> float:
        voltage = _round_voltage(_clip_voltage(voltage, v_min=v_min, v_max=v_max))
        if voltage in measure_cache:
            return measure_cache[voltage]
        write_port_voltage(port, voltage, working_data)
        cu.upload_voltage(mcv, working_data)
        time.sleep(sleep_time)
        optical_power = _read_opm_channel_uw(opm1, n)
        measure_cache[voltage] = optical_power
        return optical_power

    max_candidates = _solve_max_voltage_candidates(A, w, phi, R, v_min=v_min, v_max=v_max)
    if not max_candidates:
        raise ValueError(f"No fitted maximum point found for input {n}, port {port}.")
    v_max_guess = max_candidates[-1]

    max_scan_start = _round_voltage(_clip_voltage(v_max_guess - 0.02, v_min=v_min, v_max=v_max))
    max_scan_end = _round_voltage(_clip_voltage(v_max_guess + 0.02, v_min=v_min, v_max=v_max))
    if max_scan_end < max_scan_start:
        max_scan_end = max_scan_start
    max_scan_voltages = np.round(np.arange(max_scan_start, max_scan_end + 0.0005, 0.001), 3)
    if max_scan_voltages.size == 0:
        max_scan_voltages = np.array([_round_voltage(v_max_guess)])

    max_scan_results = [(float(v), apply_and_measure(float(v))) for v in max_scan_voltages]
    v_pmax, Pmax = max(max_scan_results, key=lambda item: (item[1], item[0]))
    P_target = float(Pmax - power)

    target_candidates = _solve_voltage_candidates_for_power(
        P_target,
        A,
        w,
        phi,
        B,
        R,
        v_min=v_min,
        v_max=v_max,
    )
    if not target_candidates:
        raise ValueError(f"No fitted target point found for input {n}, port {port}.")

    rough_v, _ = max(target_candidates, key=lambda item: item[0])
    rough_v = _round_voltage(_clip_voltage(rough_v, v_min=v_min, v_max=v_max))

    best_v = rough_v
    best_power = apply_and_measure(best_v)
    best_error = abs(best_power - P_target)

    print(
        f"find_in_p input={n}, port={port}, fitted max≈{v_max_guess:.6f} V, "
        f"measured Pmax={Pmax:.6f} uW @ {v_pmax:.3f} V, target={P_target:.6f} uW, rough_v={rough_v:.3f} V"
    )

    for step in (0.01, 0.001):
        current_v = best_v
        current_power = best_power
        current_error = best_error

        neighbor_results = []
        for direction in (1, -1):
            neighbor_v = _round_voltage(_clip_voltage(current_v + direction * step, v_min=v_min, v_max=v_max))
            if abs(neighbor_v - current_v) < 5e-4:
                continue
            neighbor_power = apply_and_measure(neighbor_v)
            neighbor_error = abs(neighbor_power - P_target)
            neighbor_results.append((neighbor_error, -neighbor_v, direction, neighbor_v, neighbor_power))

        if not neighbor_results:
            continue

        _, _, direction, next_v, next_power = min(neighbor_results)
        next_error = abs(next_power - P_target)
        if next_error > current_error + 1e-12:
            continue

        while True:
            current_sign = current_power - P_target
            next_sign = next_power - P_target

            if next_error <= best_error:
                best_v = next_v
                best_power = next_power
                best_error = next_error

            crossed = current_sign == 0 or next_sign == 0 or np.sign(current_sign) != np.sign(next_sign)
            if crossed:
                break
            if next_error > current_error + 1e-12:
                break

            current_v = next_v
            current_power = next_power
            current_error = next_error

            next_v = _round_voltage(_clip_voltage(current_v + direction * step, v_min=v_min, v_max=v_max))
            if abs(next_v - current_v) < 5e-4:
                break
            next_power = apply_and_measure(next_v)
            next_error = abs(next_power - P_target)

        print(
            f"find_in_p refine step={step:.3f}: best_v={best_v:.3f} V, "
            f"best_power={best_power:.6f} uW, target={P_target:.6f} uW"
        )

    write_port_voltage(port, best_v, working_data)
    cu.upload_voltage(mcv, working_data)
    time.sleep(sleep_time)

    print(
        f"find_in_p result: input={n}, port={port}, voltage={best_v:.3f} V, "
        f"measured_power={best_power:.6f} uW, target={P_target:.6f} uW"
    )
    return float(best_v)


def fit_to_half(mzi):
    """
    Set one MZI to the measured half-power point on an increasing interval.
    """
    mzi_key = str(int(mzi))
    port = int(MZI_TABLE[mzi_key]["ports"][0])
    data_path = os.path.join("Scandata", "inner_cali_powerdata", f"{port}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find inner calibration data: {data_path}")

    df = pd.read_csv(data_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expect at least two columns in {data_path}")

    voltage = df.iloc[:, 0].to_numpy(dtype=float)
    optical_power = df.iloc[:, 1].to_numpy(dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(optical_power)
    voltage = voltage[mask]
    optical_power = optical_power[mask]
    if voltage.size < 2:
        raise ValueError(f"Not enough valid samples in {data_path}")

    order = np.argsort(voltage)
    voltage = voltage[order]
    optical_power = optical_power[order]

    half_power = float((np.max(optical_power) + np.min(optical_power)) / 2)
    candidates = []
    for idx in range(voltage.size - 1):
        p0 = float(optical_power[idx])
        p1 = float(optical_power[idx + 1])
        if p1 <= p0:
            continue
        if p0 <= half_power <= p1:
            interval_candidates = []
            for candidate_idx in (idx, idx + 1):
                candidate_v = float(voltage[candidate_idx])
                candidate_p = float(optical_power[candidate_idx])
                interval_candidates.append((abs(candidate_p - half_power), -candidate_v, candidate_v, candidate_p))
            _, _, candidate_v, candidate_p = min(interval_candidates)
            candidates.append((candidate_v, candidate_p))

    if not candidates:
        raise ValueError(f"No increasing half-power interval found for MZI {mzi_key}, port {port}.")

    v, selected_power = max(candidates, key=lambda item: item[0])
    print(
        f"MZI {mzi_key} port {port} half-power voltage: {v:.3f} V, "
        f"power: {selected_power} uW, target half: {half_power} uW"
    )
    return v


def fit_inter_cali_sine(
    target: int,
    show_plot: bool = True,
):
    """
    Fit optical power versus electrical power from 1Col scanned data.
    Returns (A, w, phi, B).
    """
    data_dir = os.path.join("Scandata", "1Col", "power_data")
    direct_file_path = os.path.join(data_dir, f"{int(target)}.txt")
    if not os.path.exists(direct_file_path):
        raise FileNotFoundError(f"Cannot find 1Col scan data: {direct_file_path}")

    file_path = direct_file_path
    df = pd.read_csv(file_path)
    if df.shape[1] < 3:
        raise ValueError(f"Expect at least three columns in {file_path}")

    voltage = df.iloc[:, 0].to_numpy(dtype=float)
    optical_power = df.iloc[:, 1].to_numpy(dtype=float)
    current_amp = df.iloc[:, 2].to_numpy(dtype=float) * 1e-3
    electrical_power = voltage * current_amp

    mask = np.isfinite(electrical_power) & np.isfinite(optical_power)
    electrical_power = electrical_power[mask]
    optical_power = optical_power[mask]
    if electrical_power.size < 3:
        raise ValueError(f"Not enough samples in {file_path}")

    order = np.argsort(electrical_power)
    x = electrical_power[order]
    power = optical_power[order]
    title_label = f"Port {int(target)}"
    image_name = f"{int(target)}_fit.png"

    def sin_model(x, A, w, phi, B):
        return A * np.sin(w * x + phi) + B

    A_guess = 0.5 * (power.max() - power.min())
    B_guess = float(np.mean(power))
    w_guess = _estimate_sine_w(x, power)
    phi_guess = 0.0

    popt, _ = curve_fit(
        sin_model,
        x,
        power,
        p0=[A_guess, w_guess, phi_guess, B_guess],
        bounds=([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=20000,
    )
    A, w, phi, B = popt

    x_smooth = np.linspace(x.min(), x.max(), 500)
    plt.figure(figsize=(7, 5))
    plt.plot(x, power, "o", label="samples")
    plt.plot(x_smooth, sin_model(x_smooth, A, w, phi, B), "-", label="fit")
    plt.xlabel("Electrical Power (W)")
    plt.ylabel("Optical Power (uW)")
    plt.title(title_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    image_dir = os.path.join("Scandata", "1Col", "power_fit_image")
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, image_name)
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


def scan_mzi(port, start_voltage, end_voltage, step, ser, pwm, measure_time, out_num, file_path_df):
    """
    对指定的 MZI 通道进行扫描：
      - 依次更新电压，并调用上传、清零操作，
      - 读取对应功率数据，
      - 保存数据、图片，
      - 返回最大功率和最小功率对应的电压值。
    """
    print("=" * 50)
    print(f"正在扫描 {port} 号 MZI")

    # 定义数据和图片保存路径
    output_root = os.path.join("Scandata", "1Col")
    data_folder = os.path.join(output_root, "power_data")
    image_folder = os.path.join(output_root, "power_image")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    data_savepath = os.path.join(data_folder, f"{port}.txt")
    image_savepath = os.path.join(image_folder, f"{port}.png")

    v_values = np.round(np.arange(start_voltage, end_voltage + step, step), 3)
    data = []

    R, dI = cu.get_R(ser, port, file_path_df)
    print(f"MZI {port} 电阻 R={R} Ohm, dI = {dI*1e3} mA")

    for v in v_values:
        try:
            file_path_df.at[port - 1, 0] = v
        except Exception as e:
            print(f"更新电压 CSV 文件异常: {e}")
            continue

        print(f"\n扫描端口:{port} 设置电压: {v} V")
        read_current_limit = 0
        while True:
            cu.upload_voltage(ser, file_path_df)
            c = cu.read_current_port(ser, port) - dI * 1e3
            if 0.9 * v / R * 1000 < c and c < 1.1 * v / R * 1000:
                print(Fore.GREEN + f"电流正常, I={c}mA")
                break
            elif read_current_limit >= 5:
                print(Fore.RED + f"电流多次异常, I={c}mA,跳过该电压点")
                break
            elif v == 0:
                print(Fore.YELLOW + f"电压为0, 跳过该电压点")
                break
            else:
                print(Fore.RED + f"电流异常, I={c}mA, 重新上传电压")
                read_current_limit += 1
        time.sleep(measure_time)

        power_str_list = cu.read_pow(pwm)
        try:
            power_value = float(power_str_list[int(out_num) - 1]) * 1e6
        except (ValueError, IndexError) as e:
            print(f"读取功率数据错误: {e}")
            power_value = 0
        data.append([v, power_value, c, R])
        print(f"光功率值: {power_value} uW")

    np.savetxt(
        data_savepath,
        data,
        fmt="%.12f",
        delimiter=",",
        header="v,pow(uW),current(mA),R(Ohm)",
        comments="",
    )

    max_item = max(data, key=lambda x: x[1])
    min_item = min(data, key=lambda x: x[1])
    v_for_max_power = max_item[0]
    v_for_min_power = min_item[0]

    v_list = [item[0] for item in data]
    pow_list = [item[1] for item in data]
    plt.plot(v_list, pow_list, marker="o")
    plt.xlabel("v")
    plt.ylabel("pow")
    plt.title(f"V vs Power for Port {port}")
    plt.grid(True)
    plt.savefig(image_savepath)
    # plt.show()
    plt.close()
    return v_for_max_power, v_for_min_power, float(R)


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
    fit_params_1col = {}
    output_dir = os.path.join("Scandata", "1Col")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "power_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "power_image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "power_fit_image"), exist_ok=True)

    in_mzi_table = pd.read_csv("IN_MZI.txt")
    for switchin in range(1, N):
        row = in_mzi_table.loc[in_mzi_table["MZI"] == switchin]
        if row.empty:
            print(Fore.RED + f"switchin {switchin} not found in IN_MZI.txt")
            continue

        port = int(row.iloc[0]["PORT"])
        on_voltage = float(row.iloc[0]["ON"])
        off_voltage = float(row.iloc[0]["OFF"])
        print(f"Current switchin {switchin}: " f"PORT={port}, ON={on_voltage}, OFF={off_voltage}")
        _, _, resistance = scan_mzi(
            port,
            0,
            5.0,
            0.1,
            mcv,
            opm1,
            measure_time,
            out_num=switchin,
            file_path_df=working_data,
        )
        A, w, phi, B = fit_inter_cali_sine(port, show_plot=False)
        fit_params_1col[int(switchin)] = {
            "port": int(port),
            "R": float(resistance),
            "A": float(A),
            "w": float(w),
            "phi": float(phi),
            "B": float(B),
        }
        print(
            f"Fit params for switchin {switchin}: "
            f"port={port}, R={resistance}, A={A}, w={w}, phi={phi}, B={B}"
        )

    fit_params_path = os.path.join(output_dir, "fit_params_1col.txt")
    serializable_fit_params = {str(key): value for key, value in fit_params_1col.items()}
    with open(fit_params_path, "w", encoding="utf-8") as f:
        json.dump(serializable_fit_params, f, ensure_ascii=False, indent=2)
    print(f"Saved fit params to {fit_params_path}")
