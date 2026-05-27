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
from colorama import Fore

HALFPI_SCAN_MAX_V = 5.5
HALFPI_COARSE_STEP_V = 0.25


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
        table = globals().get("mzi_table")
        if table is None:
            table = load_mzi_table()
            globals()["mzi_table"] = table

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


TARGET_SCAN_VOLTAGE_PAIRS_PATH = os.path.join("Scandata", "inter_cali_pairs.json")


def _load_target_scan_voltage_pairs(path: str = TARGET_SCAN_VOLTAGE_PAIRS_PATH) -> dict:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}

    with open(path, "r", encoding="utf-8-sig") as f:
        raw_pairs = json.load(f)
    if not isinstance(raw_pairs, dict):
        raise ValueError(f"Target scan voltage pair file must contain a JSON object: {path}")

    pairs = {}
    for key, value in raw_pairs.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid saved voltage pair for target {key}: expected object.")
        key_str = str(int(key))
        pairs[key_str] = {
            "target": int(value.get("target", int(key))),
            "ports": [int(p) for p in value.get("ports", [])],
            "upper_arm_voltage": float(value.get("upper_arm_voltage", 0.0)),
            "lower_arm_voltage": float(value.get("lower_arm_voltage", 0.0)),
            "min_x_power": float(value.get("min_x_power", value.get("min_dp", 0.0))),
            "min_dp": float(value.get("min_dp", 0.0)),
            "min_power_uW": float(value.get("min_power_uW", 0.0)),
        }
    return pairs


def _get_target_scan_voltage_pairs() -> dict:
    pairs = globals().get("target_scan_voltage_pairs")
    if not isinstance(pairs, dict):
        pairs = _load_target_scan_voltage_pairs()
        globals()["target_scan_voltage_pairs"] = pairs
    return pairs


def _save_target_scan_voltage_pairs(path: str = TARGET_SCAN_VOLTAGE_PAIRS_PATH) -> None:
    pairs = _get_target_scan_voltage_pairs()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {}
    for key, value in pairs.items():
        key_str = str(int(key))
        serializable[key_str] = {
            "target": int(value.get("target", int(key))),
            "ports": [int(p) for p in value.get("ports", [])],
            "upper_arm_voltage": float(value.get("upper_arm_voltage", 0.0)),
            "lower_arm_voltage": float(value.get("lower_arm_voltage", 0.0)),
            "min_x_power": float(value.get("min_x_power", value.get("min_dp", 0.0))),
            "min_dp": float(value.get("min_dp", 0.0)),
            "min_power_uW": float(value.get("min_power_uW", 0.0)),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _apply_saved_target_scan_voltage_pair(target: int, file_data: pd.DataFrame, table: dict | None = None) -> bool:
    key = str(int(target))
    pair_entry = _get_target_scan_voltage_pairs().get(key)
    if pair_entry is None:
        return False

    if table is None:
        table = globals().get("mzi_table")
        if table is None:
            table = load_mzi_table()

    if key not in table:
        raise KeyError(f"MZI {target} not found in mzi_table.")

    ports = table[key].get("ports", [])
    if not isinstance(ports, list) or len(ports) != 2:
        raise ValueError(f"MZI {target} must have exactly two ports.")

    write_port_voltage(int(ports[0]), float(pair_entry["upper_arm_voltage"]), file_data)
    write_port_voltage(int(ports[1]), float(pair_entry["lower_arm_voltage"]), file_data)
    return True


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


def _sine_model(x, A, w, phi, b):
    return A * np.sin(w * x + phi) + b


def _fit_power_sine(x_values, power_values):
    x = np.asarray(x_values, dtype=float)
    power = np.asarray(power_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(power)
    x = x[mask]
    power = power[mask]
    if x.size < 3:
        raise ValueError("Need at least three valid samples for sine fitting.")

    order = np.argsort(x)
    x = x[order]
    power = power[order]

    w_guess = max(_estimate_sine_w(x, power), 1e-6)
    min_idx = int(np.argmin(power))
    max_idx = int(np.argmax(power))
    x_span = max(float(np.ptp(x)), 1e-9)

    w_candidates = set()
    for w_base in (w_guess, 2 * np.pi / x_span):
        if np.isfinite(w_base) and w_base > 0:
            for scale in (0.6, 0.8, 1.0, 1.2, 1.5):
                w_candidates.add(float(w_base * scale))

    extrema_distance = abs(float(x[max_idx]) - float(x[min_idx]))
    if extrema_distance > 1e-9:
        w_candidates.add(float(np.pi / extrema_distance))

    if x_span >= 2 * np.pi * 0.8:
        for w_value in (0.75, 0.9, 1.0, 1.1, 1.25):
            w_candidates.add(float(w_value))

    w_candidates = sorted(w for w in w_candidates if np.isfinite(w) and w > 0)
    if not w_candidates:
        raise RuntimeError("Failed to build sine frequency candidates.")

    best_params = None
    best_error = None
    for w0 in w_candidates:
        design = np.column_stack((np.sin(w0 * x), np.cos(w0 * x), np.ones_like(x)))
        try:
            alpha, beta, b = np.linalg.lstsq(design, power, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        fitted = design @ np.array([alpha, beta, b])
        error = float(np.sum((power - fitted) ** 2))
        if best_error is None or error < best_error:
            amplitude = float(np.hypot(alpha, beta))
            phi = float(np.arctan2(beta, alpha))
            best_params = (amplitude, float(w0), phi, float(b))
            best_error = error

    if best_params is None:
        raise RuntimeError("Failed to fit sine power model.")

    A, w, phi, b = best_params
    return float(A), float(w), float(phi), float(b), x, power


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
        globals()["mzi_table"] = table

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
    table = globals().get("mzi_table")
    if table is None:
        table = load_mzi_table()
        globals()["mzi_table"] = table

    _, _, r_values, ppi_values = _get_mzi_heater_calibration(target_id, table=table, require_two_heaters=True)
    up_R, down_R = r_values
    up_P, down_P = ppi_values
    print(
        f"Target {target_id} halfpi from MZI_table: "
        f"up_R={up_R:.6f} Ohm, up_P={up_P:.9f} W, "
        f"down_R={down_R:.6f} Ohm, down_P={down_P:.9f} W"
    )
    return up_R, up_P, down_R, down_P


def fit_inter_cali_sine(
    target: int,
    show_plot: bool = True,
):
    """
    Fit P = A * sin(w * x + phi) + b from all inter_cali_powerdata samples.
    For synchronous phase scans, x is dp (phase); otherwise x is electrical power.
    Returns (A, w, phi, b).
    """
    data_dir = os.path.join("Scandata", "inter_cali_powerdata")
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

    is_phase_scan = (
        "dp" in df.columns
        and "scan_type" in df.columns
        and df["scan_type"].astype(str).str.lower().str.contains("phase", regex=False).any()
    )
    if is_phase_scan:
        x_values = df["dp"]
        xlabel = "Synchronous Phase (rad)"
    elif "x_power" in df.columns:
        x_values = df["x_power"]
        xlabel = "Electrical Power (W)"
    else:
        x_values = df.iloc[:, 0]
        xlabel = "Electrical Power (W)"
    power_values = df["pow(uW)"] if "pow(uW)" in df.columns else df.iloc[:, 1]

    A, w, phi, b, x, power = _fit_power_sine(x_values, power_values)

    x_smooth = np.linspace(x.min(), x.max(), 500)
    plt.figure(figsize=(7, 5))
    plt.plot(x, power, "o", label="samples")
    plt.plot(x_smooth, _sine_model(x_smooth, A, w, phi, b), "-", label="fit")
    plt.xlabel(xlabel)
    plt.ylabel("Optical Power (uW)")
    plt.title(f"Target {target} Ports {ports}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    image_dir = os.path.join("Scandata", "inter_cali")
    os.makedirs(image_dir, exist_ok=True)
    ports_tag = "-".join(str(int(p)) for p in ports)
    image_path = os.path.join(image_dir, f"target_{int(target)}_ports_{ports_tag}.png")
    plt.savefig(image_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()

    return A, w, phi, b


def scan_mzis(target, ser, pwm, measure_time, out_num, file_path_df, up_R=None, up_P=None, down_R=None, down_P=None):
    mzi_table = load_mzi_table()
    key = str(int(target))
    if key not in mzi_table:
        raise ValueError(f"MZI {target} not found in mzi_table.")

    entry = mzi_table[key]
    ports = entry.get("ports", [])
    if not isinstance(ports, list) or len(ports) != 2:
        raise ValueError(f"MZI {target} must have exactly two ports.")
    ports = [int(p) for p in ports]

    if any(value is None for value in (up_R, up_P, down_R, down_P)):
        up_R, up_P, down_R, down_P = Power_halfpi(target)
        if globals().get("working_data") is file_path_df and "build_Bmzi" in globals():
            build_Bmzi(target, int(globals().get("N", 9)))

    r_values = [float(up_R), float(down_R)]
    halfpi_powers = [float(up_P), float(down_P)]
    if any(value <= 0 or not np.isfinite(value) for value in r_values):
        raise ValueError("up_R and down_R must be positive finite values.")
    if any(value <= 0 or not np.isfinite(value) for value in halfpi_powers):
        raise ValueError("up_P and down_P must be positive finite values.")

    data_folder = os.path.join("Scandata", "inter_cali_powerdata")
    image_folder = os.path.join("Scandata", "inter_cali_powerimage")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    data_savepath = os.path.join(data_folder, f"{ports}.txt")
    image_savepath = os.path.join(image_folder, f"{ports}.png")

    def get_current_power(port, r_value):
        port_idx = port - 1
        if port_idx < 0 or port_idx >= len(file_path_df):
            raise IndexError(f"PORT {port} is out of range for the provided file_data.")
        v_current = float(file_path_df.iloc[port_idx, 0])
        return (v_current**2) / r_value

    POWER_LIMIT_W = 0.055

    def clamp_voltage(value, resistance=None):
        voltage = max(0.0, min(6.0, float(value)))
        if resistance is not None:
            resistance = float(resistance)
            if resistance <= 0:
                raise ValueError("Resistance must be positive.")
            voltage = min(voltage, float(np.sqrt(POWER_LIMIT_W * resistance)))
            voltage = np.floor((voltage + 1e-12) * 1000.0) / 1000.0
        return round(voltage, 3)

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

    def pick_min_point(rows):
        if not rows:
            raise ValueError(f"No scan data generated for target {target}.")
        return min(
            rows,
            key=lambda row: (
                row["pow(uW)"],
                -(float(row["v_primary"]) + float(row["v_secondary"])),
                -max(float(row["v_primary"]), float(row["v_secondary"])),
                -float(row.get("dp", row["x_power"])),
            ),
        )

    port_primary = ports[0]
    port_secondary = ports[1]
    r_primary = r_values[0]
    r_secondary = r_values[1]
    ppi_primary = halfpi_powers[0]
    ppi_secondary = halfpi_powers[1]
    period_primary = 2.0 * ppi_primary
    period_secondary = 2.0 * ppi_secondary
    p_primary_base_initial = get_current_power(port_primary, r_primary)
    p_secondary_base_initial = get_current_power(port_secondary, r_secondary)

    def measure_at_voltages(v_primary, v_secondary, label=""):
        v_primary = clamp_voltage(v_primary, r_primary)
        v_secondary = clamp_voltage(v_secondary, r_secondary)
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
            next_v_primary = clamp_voltage(v_primary + delta_v, r_primary)
            return next_v_primary, v_secondary, next_v_primary != v_primary
        if adjust_arm == "down":
            next_v_secondary = clamp_voltage(v_secondary + delta_v, r_secondary)
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

        for step in (0.1, 0.01, 0.001):
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

    PHASE_SCAN_END = 3.0 * np.pi
    COARSE_PHASE_STEP = 0.25
    FINE_PHASE_STEP = 0.05
    FINE_PHASE_HALF_WIDTH = 0.25

    def make_phase_values(start_phase, end_phase, step, extra_values=None):
        start_phase = max(0.0, float(start_phase))
        end_phase = min(PHASE_SCAN_END, float(end_phase))
        if end_phase < start_phase:
            start_phase, end_phase = end_phase, start_phase

        values = np.arange(start_phase, end_phase + step * 0.5, step, dtype=float)
        if values.size == 0:
            values = np.array([start_phase, end_phase], dtype=float)
        elif values[-1] < end_phase - 1e-9:
            values = np.append(values, end_phase)

        if extra_values is not None:
            for value in extra_values:
                value = float(value)
                if start_phase - 1e-9 <= value <= end_phase + 1e-9:
                    values = np.append(values, value)

        deduped = sorted({round(float(value), 12) for value in values if 0.0 <= value <= PHASE_SCAN_END + 1e-9})
        return np.asarray(deduped, dtype=float)

    def row_voltage_score(row):
        v_primary = float(row["v_primary"])
        v_secondary = float(row["v_secondary"])
        return v_primary + v_secondary, max(v_primary, v_secondary), float(row["dp"])

    def nearest_phase_row(rows, phase):
        return min(rows, key=lambda row: abs(float(row["dp"]) - float(phase)))

    def fitted_min_phase_candidates(rows):
        if len(rows) < 3:
            return [float(pick_min_point(rows)["dp"])], None

        coarse_df = pd.DataFrame(rows)
        try:
            fit_A, fit_w, fit_phi, fit_b, _, _ = _fit_power_sine(coarse_df["dp"], coarse_df["pow(uW)"])
        except (RuntimeError, ValueError) as exc:
            print(f"Coarse phase fit failed for target {target}: {exc}; use measured coarse minimum.")
            return [float(pick_min_point(rows)["dp"])], None

        min_phase = -np.pi / 2 if fit_A >= 0 else np.pi / 2
        candidates = []
        k_min = int(np.floor((0.0 * fit_w + fit_phi - min_phase) / (2 * np.pi))) - 2
        k_max = int(np.ceil((PHASE_SCAN_END * fit_w + fit_phi - min_phase) / (2 * np.pi))) + 2
        for k in range(k_min, k_max + 1):
            candidate = (min_phase - fit_phi + 2 * np.pi * k) / fit_w
            if 0.0 - 1e-9 <= candidate <= PHASE_SCAN_END + 1e-9:
                candidates.append(float(np.clip(candidate, 0.0, PHASE_SCAN_END)))

        if not candidates:
            sample_phase = np.linspace(0.0, PHASE_SCAN_END, 4000)
            sample_power = _sine_model(sample_phase, fit_A, fit_w, fit_phi, fit_b)
            candidates = [float(sample_phase[int(np.argmin(sample_power))])]

        deduped = []
        for candidate in sorted(candidates):
            if not deduped or abs(candidate - deduped[-1]) > 1e-6:
                deduped.append(candidate)
        return deduped, (fit_A, fit_w, fit_phi, fit_b)

    def select_fine_center_phase(candidates, coarse_rows):
        if len(candidates) == 1:
            return float(candidates[0])

        selected = max(
            candidates,
            key=lambda phase: row_voltage_score(nearest_phase_row(coarse_rows, phase)),
        )
        selected_row = nearest_phase_row(coarse_rows, selected)
        print(
            f"Multiple fitted minima for target {target}: "
            f"{[round(value, 6) for value in candidates]}, select {selected:.6f} rad "
            f"with larger voltage pair [{selected_row['v_primary']:.3f}, {selected_row['v_secondary']:.3f}]"
        )
        return float(selected)

    def scan_phase_sequence(phase_values, scan_stage):
        rows = []
        p_primary_base = float(p_primary_base_initial)
        p_secondary_base = float(p_secondary_base_initial)
        last_phase = None
        last_power_value = None
        last_v_primary = None
        last_v_secondary = None
        last_p_primary = None
        last_p_secondary = None

        for phase in phase_values:
            phase = float(phase)
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
                    folded_primary_power = p_primary_target
                    folded_secondary_power = p_secondary_target
                    if primary_over_limit:
                        folded_primary_power = fold_power_to_limit(p_primary_target, period_primary)
                    if secondary_over_limit:
                        folded_secondary_power = fold_power_to_limit(
                            p_secondary_target,
                            period_secondary,
                        )

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
                    p_primary_base = folded_primary_power - primary_increment
                    p_secondary_base = folded_secondary_power - secondary_increment
                    p_primary_target = folded_primary_power
                    p_secondary_target = folded_secondary_power
                else:
                    p_primary_target = fold_power_to_limit(p_primary_target, period_primary)
                    p_secondary_target = fold_power_to_limit(p_secondary_target, period_secondary)

            p_primary_target = fold_power_to_limit(p_primary_target, period_primary)
            p_secondary_target = fold_power_to_limit(p_secondary_target, period_secondary)

            v_primary = power_to_voltage(p_primary_target, r_primary)
            v_secondary = power_to_voltage(p_secondary_target, r_secondary)
            v_primary = clamp_voltage(v_primary, r_primary)
            v_secondary = clamp_voltage(v_secondary, r_secondary)

            write_port_voltage(port_primary, v_primary, file_path_df)
            write_port_voltage(port_secondary, v_secondary, file_path_df)

            print(
                f"{scan_stage} phase {phase:.6f} rad, "
                f"powers: [{p_primary_target:.6f}, {p_secondary_target:.6f}] W, "
                f"voltages: [{v_primary}, {v_secondary}]"
            )
            cu.upload_voltage(ser, file_path_df)
            time.sleep(measure_time)

            power_value = read_output_power_uW()
            print(f"{scan_stage} optical power {power_value} uW")

            p_primary_actual = (float(v_primary) ** 2) / r_primary
            p_secondary_actual = (float(v_secondary) ** 2) / r_secondary
            rows.append(
                {
                    "x_power": float(p_secondary_actual),
                    "dp": float(phase),
                    "pow(uW)": float(power_value),
                    "v_primary": float(v_primary),
                    "v_secondary": float(v_secondary),
                    "p_primary": float(p_primary_actual),
                    "p_secondary": float(p_secondary_actual),
                    "scan_type": f"phase_{scan_stage}",
                }
            )
            last_phase = phase
            last_power_value = power_value
            last_v_primary = v_primary
            last_v_secondary = v_secondary
            last_p_primary = p_primary_actual
            last_p_secondary = p_secondary_actual
            p_primary_base = p_primary_actual - primary_increment
            p_secondary_base = p_secondary_actual - secondary_increment

        return rows

    coarse_phase_values = make_phase_values(0.0, PHASE_SCAN_END, COARSE_PHASE_STEP)
    coarse_rows = scan_phase_sequence(coarse_phase_values, "coarse")
    min_candidates, coarse_fit_params = fitted_min_phase_candidates(coarse_rows)
    fine_center_phase = select_fine_center_phase(min_candidates, coarse_rows)
    fine_start_phase = max(0.0, fine_center_phase - FINE_PHASE_HALF_WIDTH)
    fine_end_phase = min(PHASE_SCAN_END, fine_center_phase + FINE_PHASE_HALF_WIDTH)
    fine_phase_values = make_phase_values(
        fine_start_phase,
        fine_end_phase,
        FINE_PHASE_STEP,
        extra_values=[fine_center_phase],
    )
    print(
        f"Fine scan target {target}: center {fine_center_phase:.6f} rad, "
        f"range [{fine_start_phase:.6f}, {fine_end_phase:.6f}] rad"
    )
    fine_rows = scan_phase_sequence(fine_phase_values, "fine")

    precise_min_row = pick_min_point(fine_rows if fine_rows else coarse_rows)
    all_rows = coarse_rows + fine_rows
    result_df = pd.DataFrame(
        all_rows,
        columns=["x_power", "dp", "pow(uW)", "v_primary", "v_secondary", "p_primary", "p_secondary", "scan_type"],
    )
    result_df.to_csv(data_savepath, index=False)

    plt.figure(figsize=(7, 5))
    coarse_df = pd.DataFrame(coarse_rows)
    plt.plot(coarse_df["dp"], coarse_df["pow(uW)"], "o-", label="coarse")
    if fine_rows:
        fine_df = pd.DataFrame(fine_rows)
        plt.plot(fine_df["dp"], fine_df["pow(uW)"], ".-", label="fine")
    if coarse_fit_params is not None:
        fit_A, fit_w, fit_phi, fit_b = coarse_fit_params
        phase_smooth = np.linspace(0.0, PHASE_SCAN_END, 500)
        plt.plot(
            phase_smooth,
            _sine_model(phase_smooth, fit_A, fit_w, fit_phi, fit_b),
            "-",
            linewidth=1.0,
            label="coarse fit",
        )
    plt.axvline(fine_center_phase, color="gray", linestyle="--", linewidth=1.0, label="fit min")
    plt.axvspan(fine_start_phase, fine_end_phase, color="gray", alpha=0.12, label="fine range")
    plt.plot(
        [precise_min_row["dp"]],
        [precise_min_row["pow(uW)"]],
        "rx",
        markersize=10,
        label="min",
    )
    plt.xlabel("Synchronous Phase (rad)")
    plt.ylabel("Optical Power (uW)")
    plt.title(f"Power vs Synchronous Phase for Ports {ports}")
    max_pow = float(result_df["pow(uW)"].max()) if not result_df.empty else 0.0
    plt.ylim(bottom=0, top=max_pow + 10.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_savepath, dpi=150)
    plt.close()

    target_record = {
        "target": int(target),
        "ports": [int(port_primary), int(port_secondary)],
        "upper_arm_voltage": float(precise_min_row["v_primary"]),
        "lower_arm_voltage": float(precise_min_row["v_secondary"]),
        "min_x_power": float(precise_min_row["x_power"]),
        "min_dp": float(precise_min_row["dp"]),
        "min_power_uW": float(precise_min_row["pow(uW)"]),
    }
    target_scan_voltage_pairs = _get_target_scan_voltage_pairs()
    target_scan_voltage_pairs[key] = target_record
    _save_target_scan_voltage_pairs()
    print(
        f"Target {target} precise minimum: upper={target_record['upper_arm_voltage']:.3f} V, "
        f"lower={target_record['lower_arm_voltage']:.3f} V, "
        f"power={target_record['min_power_uW']:.6f} uW, x={target_record['min_x_power']:.6f} W"
    )
    return target_record


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
    file_data.iloc[port_idx, 0] = round(float(voltage), 3)


def fit_to_half(mzi):
    """
    Return one MZI's saved half-power voltage from MZI_table.
    """
    mzi_key = str(int(mzi))
    table = globals().get("mzi_table")
    if table is None:
        table = load_mzi_table()
        globals()["mzi_table"] = table
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


def build_Bmzi(target, N):
    """
    Build Bmzi by setting H-state MZIs to half and recording their voltages.
    Returns a dict mapping mzi_id to its half-state voltage.
    """
    path, inp, out, state, bmzi = find_Bmzi_path(target, N)
    h_indices = [idx for idx, s in enumerate(state) if s == "H"]
    h_mzis = [int(path[idx]) for idx in h_indices]

    half_voltages = {}
    for mzi_id in h_mzis:
        port = int(mzi_table[str(mzi_id)]["ports"][0])
        half_voltage = float(fit_to_half(mzi_id))
        write_port_voltage(port, half_voltage, working_data)
        half_voltages[mzi_id] = half_voltage

    for idx in range(len(path)):
        mzi = int(path[idx])
        if state[idx] == "B":
            write_port_voltage(
                mzi_table[str(mzi)]["ports"][0],
                mzi_table[str(mzi)]["dtheta"][0],
                working_data,
            )
        elif state[idx] == "C":
            write_port_voltage(
                mzi_table[str(mzi)]["ports"][0],
                mzi_table[str(mzi)]["dtheta"][1],
                working_data,
            )
        elif state[idx] == "H":
            # Already set to half
            pass
        else:
            print(Fore.RED + f"Unexpected state {state[idx]} for MZI {mzi}")

    if int(bmzi) == 0:
        print(f"Target {target}: bmzi is 0, skip saved-voltage application.")
    elif _apply_saved_target_scan_voltage_pair(int(bmzi), working_data, table=mzi_table):
        pair_entry = _get_target_scan_voltage_pairs()[str(int(bmzi))]
        print(
            f"Target {target}: applied saved scan voltages for bmzi {int(bmzi)} "
            f"-> upper={pair_entry['upper_arm_voltage']:.3f} V, "
            f"lower={pair_entry['lower_arm_voltage']:.3f} V"
        )
    else:
        bmzi_entry = mzi_table[str(int(bmzi))]
        bmzi_ports = bmzi_entry.get("ports", [])
        bmzi_dtheta = bmzi_entry.get("dtheta", [])
        if not isinstance(bmzi_ports, list) or len(bmzi_ports) < 2:
            raise ValueError(f"MZI {int(bmzi)} must have two ports for fallback handling.")
        if not isinstance(bmzi_dtheta, list) or len(bmzi_dtheta) < 1:
            raise ValueError(f"MZI {int(bmzi)} missing dtheta for fallback handling.")
        write_port_voltage(int(bmzi_ports[0]), float(bmzi_dtheta[0]), working_data)
        write_port_voltage(int(bmzi_ports[1]), 0.0, working_data)
        print(
            f"Target {target}: bmzi {int(bmzi)} has no saved scan voltages, "
            f"fallback to mzi_table upper={float(bmzi_dtheta[0]):.3f} V, lower=0.000 V"
        )

    for j in range(N - 1):
        switch_IN(j + 1, "OFF", working_data)
    switch_IN(inp + 1, "ON", working_data)
    cu.upload_voltage(mcv, working_data)

    return half_voltages


def find_Bmzi_path(target, N):
    M = du.Clements_matrix(N)
    PATH = np.array([])
    idx = np.where(M == target)
    cx, cy = idx[0][0], idx[1][0]
    bx, by = idx[0][0], idx[1][0]
    if idx[0][0] - 2 >= 0:
        bmzi = M[idx[0][0] - 2][idx[1][0]]
    else:
        bmzi = 0
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
    idx = np.where(PATH == target)
    state[idx[0][0] - 1] = "H"
    state[idx[0][0] + 1] = "H"

    return PATH, input, ouput, state, bmzi


def get_theta2_targets(N):
    M = du.Clements_matrix(N)
    targets = []
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if M[i][j] != 0:
                targets.append(M[i][j])
    return targets


if __name__ == "__main__":
    N = 9
    M = du.Clements_matrix(N)
    # print(du.Clements_matrix(N))
    working_data = cu.generate_working_data()
    mzi_table = load_mzi_table()

    OPM1_ADDRESS = "TCPIP0::192.168.0.5::inst0::INSTR"
    OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
    SER_ADDRESS = "COM3"
    opm1 = cu.open_VISA_connection(OPM1_ADDRESS)
    opm2 = cu.open_VISA_connection(OPM2_ADDRESS)
    mcv = cu.open_ser_connection(SER_ADDRESS)

    targets = get_theta2_targets(N)
    measure_time = 1
    start_v = 0
    end_v = 5
    step_v = 0.1
    measure_time = 2
    target_scan_voltage_pairs = _load_target_scan_voltage_pairs()
    print(
        f"Loaded {len(target_scan_voltage_pairs)} saved target scan voltage pairs from {TARGET_SCAN_VOLTAGE_PAIRS_PATH}"
    )

    for target in targets:
        target_key = str(int(target))
        if target_key in target_scan_voltage_pairs:
            pair_entry = target_scan_voltage_pairs[target_key]
            print(
                f"{Fore.YELLOW}Skip target {int(target)}: saved voltage pair exists "
                f"upper={pair_entry['upper_arm_voltage']:.3f} V, "
                f"lower={pair_entry['lower_arm_voltage']:.3f} V"
            )
            continue

        print(50 * "-")
        path, inp, out, state, bmzi = find_Bmzi_path(target, N)
        print(f"Target MZI: {target}")
        print(f"path: {path}")
        print(f"input: {inp}")
        print(f"output: {out}")
        print(f"state: {state}")
        print(f"bmzi: {bmzi}")

        up_R, up_P, down_R, down_P = Power_halfpi(target)
        print(
            f"{Fore.BLUE}Target {target} halfpi calibration loaded: "
            f"up_R={up_R:.6f} Ohm, up_P={up_P:.9f} W, "
            f"down_R={down_R:.6f} Ohm, down_P={down_P:.9f} W"
        )

        for i in range(N * (N - 1) // 2):
            print(f"MZI {i+1} heater indices:", mzi_table[str(i + 1)]["ports"][0], mzi_table[str(i + 1)]["dtheta"][0])
            write_port_voltage(mzi_table[str(i + 1)]["ports"][0], mzi_table[str(i + 1)]["dtheta"][0], working_data)
        build_Bmzi(target, N)
        scan_mzis(target, mcv, opm2, measure_time, out + 1, working_data, up_R, up_P, down_R, down_P)
        write_port_voltage(mzi_table[str(target)]["ports"][1], 0, working_data)
        fit_inter_cali_sine(target, show_plot=False)

    _save_target_scan_voltage_pairs()
    print(f"Saved target scan voltage pairs to {TARGET_SCAN_VOLTAGE_PAIRS_PATH}")
