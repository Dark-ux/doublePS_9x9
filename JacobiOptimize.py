import os
import json
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style

import utils.communication as cu
import utils.AllDecompositionUtils as du
import compute_j_delta as jd


DEFAULT_N = 9
DEFAULT_V_MIN = 0.0
DEFAULT_V_MAX = 5.5
DEFAULT_SETTLE_TIME = 0.5

DEFAULT_OPM1_ADDRESS = "TCPIP0::192.168.0.5::inst0::INSTR"
DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
DEFAULT_SER_ADDRESS = "COM3"

MZI_TABLE_PATH = os.path.join("Scandata", "MZI_table.json")
INTER_CALI_PAIRS_PATH = os.path.join("Scandata", "inter_cali_pairs.json")
BW_DIR = os.path.join("Scandata", "BW")
JACOBIAN_MEASUREMENT_DIR = "jacobian_measurements"
J_DELTA_RESULT_DIR = "results"
SECOND_COLUMN_MZIS = (5, 6, 7, 8)
SECOND_COLUMN_HEATERS = ("5u", "5d", "6u", "6d", "7u", "7d", "8u", "8d")
RUN_START_TIME = time.perf_counter()


def elapsed_time_text() -> str:
    elapsed = time.perf_counter() - RUN_START_TIME
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((elapsed - int(elapsed)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def log(message: str, color: str = "", bright: bool = False) -> None:
    prefix = f"[elapsed {elapsed_time_text()}] "
    style = Style.BRIGHT if bright else ""
    reset = Style.RESET_ALL if color or bright else ""
    print(f"{color}{style}{prefix}{message}{reset}")


@dataclass
class DeviceConfig:
    opm1_address: str = DEFAULT_OPM1_ADDRESS
    opm2_address: str = DEFAULT_OPM2_ADDRESS
    ser_address: str = DEFAULT_SER_ADDRESS
    enable_opm1: bool = False
    enable_opm2: bool = True
    enable_mcv: bool = True


@dataclass
class OptimizeConfig:
    n: int = DEFAULT_N
    v_min: float = DEFAULT_V_MIN
    v_max: float = DEFAULT_V_MAX
    settle_time: float = DEFAULT_SETTLE_TIME
    mzi_table_path: str = MZI_TABLE_PATH
    inter_cali_pairs_path: str = INTER_CALI_PAIRS_PATH
    bw_dir: str = BW_DIR
    measure_current_t: bool = True
    run_j_delta_measurement: bool = True


@dataclass
class JDeltaMeasurementConfig:
    observed_mzis: tuple[int, ...] = SECOND_COLUMN_MZIS
    perturbed_heaters: tuple[str, ...] = SECOND_COLUMN_HEATERS
    jacobian_dir: str = JACOBIAN_MEASUREMENT_DIR
    out_dir: str = J_DELTA_RESULT_DIR
    probe_map: str = "5:u,6:u,7:u,8:u"
    fix_w: bool = True
    measurement_mode: str = "single"  # "full" for all scans, "single" for one obs/perturb pair.
    single_observed_mzi: int = 7
    single_perturbed_heater: str = "8d"
    delta_power_w: float = 0.001
    probe_half_width_w: float = 0.001
    probe_step_w: float = 0.00025


@dataclass
class HardwareHandles:
    opm1: Any | None = None
    opm2: Any | None = None
    mcv: Any | None = None


def load_mzi_table(path: str = MZI_TABLE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    for entry in table.values():
        if "dtheta" not in entry and "dtheta_Bar" in entry and "dtheta_Cross" in entry:
            bar_values = entry.get("dtheta_Bar", [])
            cross_values = entry.get("dtheta_Cross", [])
            if bar_values and cross_values:
                entry["dtheta"] = [bar_values[0], cross_values[0]]
    return table


def load_inter_cali_pairs(path: str = INTER_CALI_PAIRS_PATH) -> dict:
    if not os.path.exists(path):
        log(f"Inter calibration pair file not found: {path}", Fore.YELLOW)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if not isinstance(pairs, dict):
        raise ValueError(f"Inter calibration pair file must contain a JSON object: {path}")
    return pairs


def validate_voltage_range(working_data: pd.DataFrame, v_min: float, v_max: float) -> None:
    if v_min > v_max:
        raise ValueError("v_min must be <= v_max.")
    if working_data is None:
        raise ValueError("working_data must not be None.")
    if 0 not in working_data.columns:
        raise ValueError("working_data must contain voltage column 0.")

    voltages = pd.to_numeric(working_data[0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltage values.")

    out_of_range = (voltages < v_min) | (voltages > v_max)
    if out_of_range.any():
        bad_idx = out_of_range[out_of_range].index.tolist()
        raise ValueError(f"Voltage out of range [{v_min}, {v_max}] at rows: {bad_idx}")


def upload_v_checked(mcv, working_data: pd.DataFrame, v_min: float, v_max: float) -> None:
    validate_voltage_range(working_data, v_min, v_max)
    cu.upload_voltage(mcv, working_data)


def write_port_voltage(port: int, voltage: float, working_data: pd.DataFrame) -> None:
    voltage = round(float(voltage), 3)
    if voltage < DEFAULT_V_MIN or voltage > DEFAULT_V_MAX:
        raise ValueError(f"Refuse to set PORT {port} to {voltage:.3f} V; allowed range is 0-5.5 V.")

    port_idx = int(port) - 1
    if port_idx < 0 or port_idx >= len(working_data):
        raise IndexError(f"PORT {port} is out of range for working_data.")
    working_data.iloc[port_idx, 0] = voltage


def switch_in(mzi_index: int, state: str, working_data: pd.DataFrame) -> float:
    table = pd.read_csv("IN_MZI.txt")
    state_norm = state.strip().upper()
    if state_norm not in {"ON", "OFF"}:
        raise ValueError("state must be 'ON' or 'OFF'")

    row = table.loc[table["MZI"] == int(mzi_index)]
    if row.empty:
        raise ValueError(f"MZI index {mzi_index} not found in IN_MZI.txt")

    port = int(row.iloc[0]["PORT"])
    voltage = float(row.iloc[0][state_norm])
    write_port_voltage(port, voltage, working_data)
    return voltage


def open_hardware(config: DeviceConfig) -> HardwareHandles:
    handles = HardwareHandles()

    if config.enable_opm1:
        log(f"Opening OPM1: {config.opm1_address}", Fore.CYAN)
        handles.opm1 = cu.open_VISA_connection(config.opm1_address)
    if config.enable_opm2:
        log(f"Opening OPM2: {config.opm2_address}", Fore.CYAN)
        handles.opm2 = cu.open_VISA_connection(config.opm2_address)
    if config.enable_mcv:
        log(f"Opening MCV: {config.ser_address}", Fore.CYAN)
        handles.mcv = cu.open_ser_connection(config.ser_address)

    return handles


def close_hardware(handles: HardwareHandles) -> None:
    for name in ("opm1", "opm2", "mcv"):
        handle = getattr(handles, name)
        if handle is None:
            continue
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
                log(f"Closed {name}", Fore.CYAN)
            except Exception as exc:
                log(f"Failed to close {name}: {exc}", Fore.YELLOW)


def initialize_working_data(config: OptimizeConfig, mzi_table: dict) -> pd.DataFrame:
    working_data = cu.generate_working_data()
    mzi_count = config.n * (config.n - 1) // 2

    for mzi_id in range(1, mzi_count + 1):
        entry = mzi_table.get(str(mzi_id))
        if entry is None:
            raise KeyError(f"MZI {mzi_id} not found in MZI table.")

        ports = entry.get("ports", [])
        bar_values = entry.get("dtheta_Bar", entry.get("dtheta", []))
        if not ports or not bar_values:
            raise ValueError(f"MZI {mzi_id} missing ports or Bar voltage.")

        write_port_voltage(int(ports[0]), float(bar_values[0]), working_data)

    for input_idx in range(1, config.n):
        switch_in(input_idx, "OFF", working_data)

    return working_data


def apply_inter_cali_pairs(
    working_data: pd.DataFrame,
    inter_cali_pairs: dict,
    n: int,
    inter_cali_pairs_path: str = INTER_CALI_PAIRS_PATH,
) -> None:
    for i in range(4, int(n) * (int(n) - 1) // 2 - 4):
        mzi_id = str(i + 1)
        pair_entry = inter_cali_pairs.get(mzi_id)
        if not pair_entry:
            log(f"Skip MZI {mzi_id}: no entry in {inter_cali_pairs_path}")
            continue

        ports = pair_entry.get("ports", [])
        if not isinstance(ports, list) or len(ports) != 2:
            raise ValueError(f"MZI {mzi_id} in {inter_cali_pairs_path} must contain two ports.")

        upper_v = float(pair_entry.get("upper_arm_voltage", 0.0))
        lower_v = float(pair_entry.get("lower_arm_voltage", 0.0))
        write_port_voltage(int(ports[0]), upper_v, working_data)
        write_port_voltage(int(ports[1]), lower_v, working_data)
        log(
            f"Deploy inter_cali pair MZI {mzi_id}: "
            f"port {int(ports[0])} -> {upper_v:.3f} V, "
            f"port {int(ports[1])} -> {lower_v:.3f} V"
        )


def build_initial_context(optimize_config: OptimizeConfig) -> dict:
    mzi_table = load_mzi_table(optimize_config.mzi_table_path)
    inter_cali_pairs = load_inter_cali_pairs(optimize_config.inter_cali_pairs_path)
    clements_matrix = du.Clements_matrix(optimize_config.n)
    working_data = initialize_working_data(optimize_config, mzi_table)
    apply_inter_cali_pairs(
        working_data,
        inter_cali_pairs,
        optimize_config.n,
        optimize_config.inter_cali_pairs_path,
    )

    return {
        "mzi_table": mzi_table,
        "inter_cali_pairs": inter_cali_pairs,
        "clements_matrix": clements_matrix,
        "working_data": working_data,
    }


def normalize_power_vector(values) -> np.ndarray:
    power = np.asarray(values, dtype=float)
    total = float(np.sum(power))
    if total != 0.0:
        return power / total
    return power


def print_matrix(matrix, decimals: int = 5) -> None:
    if matrix is None:
        print("None")
        return
    for row in np.asarray(matrix):
        print(" ".join(f"{val:.{decimals}f}" for val in row))


def get_T(
    n: int,
    mcv,
    opm,
    working_data: pd.DataFrame,
    v_min: float = DEFAULT_V_MIN,
    v_max: float = DEFAULT_V_MAX,
    sleep_time: float = DEFAULT_SETTLE_TIME,
    show_plot: bool = True,
) -> np.ndarray:
    log(f"Measuring T matrix for N={n}...", Fore.CYAN, bright=True)
    if working_data is None:
        raise ValueError("working_data must not be None.")
    if mcv is None:
        raise RuntimeError("mcv is not initialized.")
    if opm is None:
        raise RuntimeError("opm is not initialized.")
    if n < 2:
        raise ValueError("n must be >= 2.")

    input_count = int(n) - 1
    cols = []

    for input_port in range(1, input_count + 1):
        for ch in range(1, input_count + 1):
            switch_in(ch, "OFF", working_data)
        switch_in(input_port, "ON", working_data)
        upload_v_checked(mcv, working_data, v_min, v_max)
        time.sleep(sleep_time)

        power_str_list = cu.read_pow(opm)
        powers = []
        for idx, value in enumerate(power_str_list[:input_count]):
            try:
                powers.append(float(value))
            except ValueError as exc:
                raise ValueError(f"Invalid power at channel {idx + 1}: {value}") from exc

        if len(powers) < input_count:
            raise ValueError(f"Expected at least {input_count} power channels, got {len(powers)}.")

        col = normalize_power_vector(powers)
        cols.append(col)
        log(f"Input {input_port}: raw={powers}, normalized={np.array2string(col, precision=5)}")

    transfer_matrix = np.column_stack(cols)

    if show_plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(transfer_matrix, aspect="auto", origin="upper", cmap="viridis")
        plt.yticks(range(transfer_matrix.shape[0]))
        plt.colorbar(label="Normalized power")
        plt.xlabel("Input channel")
        plt.ylabel("Output channel")
        plt.title("Current power transfer matrix")
        plt.tight_layout()
        plt.show()
        plt.close()

    return transfer_matrix


def arm_to_index(arm: str) -> int:
    arm_norm = str(arm).strip().lower()
    if arm_norm in {"u", "upper", "up", "0"}:
        return 0
    if arm_norm in {"d", "lower", "down", "1"}:
        return 1
    raise ValueError(f"Unsupported arm {arm!r}; expected u/upper or d/lower.")


def arm_to_name(arm: str) -> str:
    return "upper" if arm_to_index(arm) == 0 else "lower"


def parse_heater_label(heater_label: str) -> tuple[int, str]:
    text = str(heater_label).strip().lower()
    if len(text) < 2:
        raise ValueError(f"Invalid heater label {heater_label!r}; expected like 5u.")
    return int(text[:-1]), text[-1]


def get_mzi_arm_info(mzi_table: dict, mzi_id: int, arm: str) -> dict:
    entry = mzi_table.get(str(int(mzi_id)))
    if entry is None:
        raise KeyError(f"MZI {mzi_id} not found in MZI table.")

    arm_index = arm_to_index(arm)
    ports = entry.get("ports", [])
    heater_r = entry.get("heater_R", [])
    if arm_index >= len(ports):
        raise ValueError(f"MZI {mzi_id} has no port for arm {arm}.")
    if arm_index >= len(heater_r):
        raise ValueError(f"MZI {mzi_id} has no heater_R for arm {arm}.")

    resistance = float(heater_r[arm_index])
    if not np.isfinite(resistance) or resistance <= 0.0:
        raise ValueError(f"MZI {mzi_id} arm {arm} has invalid heater_R={resistance}.")

    return {
        "mzi_id": int(mzi_id),
        "arm": "u" if arm_index == 0 else "d",
        "arm_index": arm_index,
        "arm_name": "upper" if arm_index == 0 else "lower",
        "port": int(ports[arm_index]),
        "resistance": resistance,
    }


def voltage_to_power_w(voltage: float, resistance: float) -> float:
    return float(voltage) ** 2 / float(resistance)


def power_to_voltage(power_w: float, resistance: float) -> float:
    power_w = float(power_w)
    if power_w < 0.0:
        raise ValueError(f"power_w must be non-negative, got {power_w}.")
    voltage = float(np.sqrt(power_w * float(resistance)))
    if voltage < DEFAULT_V_MIN or voltage > DEFAULT_V_MAX:
        raise ValueError(
            f"Required voltage {voltage:.3f} V for {power_w:.9f} W is outside "
            f"[{DEFAULT_V_MIN}, {DEFAULT_V_MAX}] V."
        )
    return round(voltage, 3)


def get_port_voltage(working_data: pd.DataFrame, port: int) -> float:
    port_idx = int(port) - 1
    if port_idx < 0 or port_idx >= len(working_data):
        raise IndexError(f"PORT {port} is out of range for working_data.")
    return float(working_data.iloc[port_idx, 0])


def set_port_power(
    working_data: pd.DataFrame,
    port: int,
    resistance: float,
    power_w: float,
) -> float:
    voltage = power_to_voltage(power_w, resistance)
    write_port_voltage(port, voltage, working_data)
    return voltage


def get_left_upper_bar_channel(clements_matrix: np.ndarray, mzi_id: int) -> int:
    rows, _ = np.where(clements_matrix == int(mzi_id))
    if rows.size == 0:
        raise ValueError(f"MZI {mzi_id} not found in Clements matrix.")
    # Row r couples waveguides r+1 and r+2. Left-upper input and right-upper
    # bar output are therefore channel r+1 in 1-based hardware numbering.
    return int(rows[0]) + 1


def set_single_input(input_port: int, input_count: int, working_data: pd.DataFrame) -> None:
    for ch in range(1, int(input_count) + 1):
        switch_in(ch, "OFF", working_data)
    switch_in(int(input_port), "ON", working_data)


def read_output_power_uW(opm, output_channel: int) -> float:
    power_str_list = cu.read_pow(opm)
    idx = int(output_channel) - 1
    if idx < 0 or idx >= len(power_str_list):
        raise IndexError(f"Output channel {output_channel} is not available from OPM readout.")
    return float(power_str_list[idx]) * 1e6


def build_probe_offsets(half_width_w: float, step_w: float) -> np.ndarray:
    if half_width_w <= 0.0:
        raise ValueError("probe_half_width_w must be positive.")
    if step_w <= 0.0:
        raise ValueError("probe_step_w must be positive.")
    count = int(round((2.0 * half_width_w) / step_w)) + 1
    offsets = np.linspace(-half_width_w, half_width_w, count)
    return np.round(offsets, 9)


def apply_perturbation(
    working_data: pd.DataFrame,
    mzi_table: dict,
    heater_label: str | None,
    delta_power_w: float,
    base_working_data: pd.DataFrame,
) -> dict | None:
    if heater_label is None:
        return None

    mzi_id, arm = parse_heater_label(heater_label)
    info = get_mzi_arm_info(mzi_table, mzi_id, arm)
    baseline_voltage = get_port_voltage(base_working_data, info["port"])
    baseline_power_w = voltage_to_power_w(baseline_voltage, info["resistance"])
    perturbed_power_w = baseline_power_w + float(delta_power_w)
    perturbed_voltage = set_port_power(working_data, info["port"], info["resistance"], perturbed_power_w)
    return {
        "perturbed_heater": heater_label,
        "mzi_id": mzi_id,
        "arm": info["arm"],
        "port": info["port"],
        "heater_R_ohm": info["resistance"],
        "delta_power_w": float(delta_power_w),
        "baseline_power_w": baseline_power_w,
        "perturbed_power_w": perturbed_power_w,
        "baseline_voltage_v": baseline_voltage,
        "perturbed_voltage_v": perturbed_voltage,
    }


def measure_probe_scan(
    handles: HardwareHandles,
    optimize_config: OptimizeConfig,
    context: dict,
    base_working_data: pd.DataFrame,
    observed_mzi: int,
    probe_arm: str,
    save_path: str,
    perturb_heater: str | None = None,
    delta_power_w: float = 0.0,
    probe_half_width_w: float = 0.001,
    probe_step_w: float = 0.00025,
) -> pd.DataFrame:
    if handles.mcv is None or handles.opm2 is None:
        raise RuntimeError("J_delta measurement requires initialized mcv and opm2.")

    mzi_table = context["mzi_table"]
    clements_matrix = context["clements_matrix"]
    probe_info = get_mzi_arm_info(mzi_table, observed_mzi, probe_arm)
    input_channel = get_left_upper_bar_channel(clements_matrix, observed_mzi)
    output_channel = input_channel
    input_count = optimize_config.n - 1

    baseline_probe_voltage = get_port_voltage(base_working_data, probe_info["port"])
    baseline_probe_power_w = voltage_to_power_w(baseline_probe_voltage, probe_info["resistance"])
    offsets = build_probe_offsets(probe_half_width_w, probe_step_w)
    rows = []

    log(
        f"Probe scan obs MZI {observed_mzi} arm {probe_info['arm']} "
        f"input/output {input_channel}, perturb={perturb_heater or 'baseline'}",
        Fore.CYAN,
    )

    for offset_w in offsets:
        working_data = base_working_data.copy(deep=True)
        perturb_info = apply_perturbation(
            working_data,
            mzi_table,
            perturb_heater,
            delta_power_w,
            base_working_data,
        )

        probe_center_power_w = baseline_probe_power_w
        if perturb_info is not None and int(perturb_info["port"]) == int(probe_info["port"]):
            probe_center_power_w = float(perturb_info["perturbed_power_w"])

        target_probe_power_w = probe_center_power_w + float(offset_w)
        voltage_v = set_port_power(
            working_data,
            probe_info["port"],
            probe_info["resistance"],
            target_probe_power_w,
        )
        set_single_input(input_channel, input_count, working_data)
        upload_v_checked(handles.mcv, working_data, optimize_config.v_min, optimize_config.v_max)
        time.sleep(optimize_config.settle_time)
        optical_power_uW = read_output_power_uW(handles.opm2, output_channel)

        rows.append(
            {
                "mzi_id": int(observed_mzi),
                "arm_index": int(probe_info["arm_index"]),
                "arm_name": probe_info["arm_name"],
                "port": int(probe_info["port"]),
                "scan_stage": "baseline" if perturb_heater is None else f"perturb_{perturb_heater}",
                "perturbed_heater": "" if perturb_heater is None else str(perturb_heater),
                "input_channel": int(input_channel),
                "output_channel": int(output_channel),
                "probe_axis_power_w": float(offset_w),
                "target_power_w": float(target_probe_power_w),
                "voltage_v": float(voltage_v),
                "optical_power_uW": float(optical_power_uW),
                "measured_power_w": voltage_to_power_w(voltage_v, probe_info["resistance"]),
                "baseline_probe_power_w": float(baseline_probe_power_w),
            }
        )
        log(
            f"  offset={float(offset_w):+.6f} W, target={target_probe_power_w:.6f} W, "
            f"V={voltage_v:.3f}, OP={optical_power_uW:.6f} uW"
        )

    scan_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scan_df.to_csv(save_path, index=False, float_format="%.12f")
    return scan_df


def collect_j_delta_measurements(
    handles: HardwareHandles,
    optimize_config: OptimizeConfig,
    context: dict,
    j_config: JDeltaMeasurementConfig,
) -> None:
    probe_map = jd.parse_probe_map(j_config.probe_map)
    jacobian_dir = os.path.abspath(j_config.jacobian_dir)
    baseline_dir = os.path.join(jacobian_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    base_working_data = context["working_data"].copy(deep=True)
    upload_v_checked(handles.mcv, base_working_data, optimize_config.v_min, optimize_config.v_max)
    time.sleep(optimize_config.settle_time)

    for observed_mzi in j_config.observed_mzis:
        probe_arm = probe_map.get(int(observed_mzi), "u")
        measure_probe_scan(
            handles,
            optimize_config,
            context,
            base_working_data,
            int(observed_mzi),
            probe_arm,
            os.path.join(baseline_dir, f"obs{int(observed_mzi)}_probe.txt"),
            perturb_heater=None,
            delta_power_w=0.0,
            probe_half_width_w=j_config.probe_half_width_w,
            probe_step_w=j_config.probe_step_w,
        )

    for heater_label in j_config.perturbed_heaters:
        perturb_dir = os.path.join(jacobian_dir, f"perturb_{heater_label}")
        os.makedirs(perturb_dir, exist_ok=True)
        metadata_working_data = base_working_data.copy(deep=True)
        metadata = apply_perturbation(
            metadata_working_data,
            context["mzi_table"],
            heater_label,
            j_config.delta_power_w,
            base_working_data,
        )
        if metadata is None:
            raise RuntimeError(f"Failed to build metadata for perturbation {heater_label}.")
        with open(os.path.join(perturb_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        for observed_mzi in j_config.observed_mzis:
            probe_arm = probe_map.get(int(observed_mzi), "u")
            measure_probe_scan(
                handles,
                optimize_config,
                context,
                base_working_data,
                int(observed_mzi),
                probe_arm,
                os.path.join(perturb_dir, f"obs{int(observed_mzi)}_probe.txt"),
                perturb_heater=heater_label,
                delta_power_w=j_config.delta_power_w,
                probe_half_width_w=j_config.probe_half_width_w,
                probe_step_w=j_config.probe_step_w,
            )

    restored = base_working_data.copy(deep=True)
    for ch in range(1, optimize_config.n):
        switch_in(ch, "OFF", restored)
    upload_v_checked(handles.mcv, restored, optimize_config.v_min, optimize_config.v_max)


def remeasure_j_delta_pair(
    handles: HardwareHandles,
    optimize_config: OptimizeConfig,
    context: dict,
    j_config: JDeltaMeasurementConfig,
    observed_mzi: int,
    perturbed_heater: str,
) -> None:
    probe_map = jd.parse_probe_map(j_config.probe_map)
    jacobian_dir = os.path.abspath(j_config.jacobian_dir)
    perturb_dir = os.path.join(jacobian_dir, f"perturb_{perturbed_heater}")
    baseline_file = os.path.join(jacobian_dir, "baseline", f"obs{int(observed_mzi)}_probe.txt")
    os.makedirs(perturb_dir, exist_ok=True)

    if not os.path.exists(baseline_file):
        raise FileNotFoundError(
            f"Cannot single-remeasure obs{int(observed_mzi)} perturb {perturbed_heater}: "
            f"baseline file is missing: {baseline_file}"
        )

    base_working_data = context["working_data"].copy(deep=True)
    upload_v_checked(handles.mcv, base_working_data, optimize_config.v_min, optimize_config.v_max)
    time.sleep(optimize_config.settle_time)

    metadata_working_data = base_working_data.copy(deep=True)
    metadata = apply_perturbation(
        metadata_working_data,
        context["mzi_table"],
        perturbed_heater,
        j_config.delta_power_w,
        base_working_data,
    )
    if metadata is None:
        raise RuntimeError(f"Failed to build metadata for perturbation {perturbed_heater}.")
    with open(os.path.join(perturb_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    probe_arm = probe_map.get(int(observed_mzi), "u")
    save_path = os.path.join(perturb_dir, f"obs{int(observed_mzi)}_probe.txt")
    log(
        f"Single remeasure: obs{int(observed_mzi)} perturb {perturbed_heater}; "
        f"writing {save_path}",
        Fore.GREEN,
    )
    try:
        measure_probe_scan(
            handles,
            optimize_config,
            context,
            base_working_data,
            int(observed_mzi),
            probe_arm,
            save_path,
            perturb_heater=perturbed_heater,
            delta_power_w=j_config.delta_power_w,
            probe_half_width_w=j_config.probe_half_width_w,
            probe_step_w=j_config.probe_step_w,
        )
    finally:
        restored = base_working_data.copy(deep=True)
        for ch in range(1, optimize_config.n):
            switch_in(ch, "OFF", restored)
        upload_v_checked(handles.mcv, restored, optimize_config.v_min, optimize_config.v_max)
        log("Restored base network state after single remeasure.", Fore.CYAN)


def run_jacobi_optimization(handles: HardwareHandles, config: OptimizeConfig, context: dict) -> None:
    j_config = JDeltaMeasurementConfig()
    if not config.run_j_delta_measurement:
        log("J_delta measurement is disabled.", Fore.YELLOW)
        return

    if j_config.measurement_mode == "single":
        remeasure_j_delta_pair(
            handles,
            config,
            context,
            j_config,
            j_config.single_observed_mzi,
            j_config.single_perturbed_heater,
        )
    elif j_config.measurement_mode == "full":
        log("Collecting full J_delta perturbation measurements...", Fore.GREEN)
        collect_j_delta_measurements(handles, config, context, j_config)
    else:
        raise ValueError("JDeltaMeasurementConfig.measurement_mode must be 'single' or 'full'.")

    probe_map = jd.parse_probe_map(j_config.probe_map)
    log("Computing J_delta from collected probe scans...", Fore.GREEN)
    jd.compute_j_delta(
        mzi_table_path=config.mzi_table_path,
        jacobian_dir=j_config.jacobian_dir,
        out_dir=j_config.out_dir,
        probe_map=probe_map,
        fix_w=j_config.fix_w,
    )


def main() -> None:
    optimize_config = OptimizeConfig()
    device_config = DeviceConfig()
    handles = HardwareHandles()

    try:
        context = build_initial_context(optimize_config)
        handles = open_hardware(device_config)

        if handles.mcv is not None:
            upload_v_checked(
                handles.mcv,
                context["working_data"],
                optimize_config.v_min,
                optimize_config.v_max,
            )
            time.sleep(optimize_config.settle_time)

        log("JacobiOptimize framework initialized.", Fore.GREEN)
        log(f"N={optimize_config.n}, MZI count={optimize_config.n * (optimize_config.n - 1) // 2}")
        log(f"Clements matrix shape: {context['clements_matrix'].shape}")

        if optimize_config.measure_current_t and handles.mcv is not None and handles.opm2 is not None:
            current_T = get_T(
                optimize_config.n,
                handles.mcv,
                handles.opm2,
                context["working_data"],
                optimize_config.v_min,
                optimize_config.v_max,
                optimize_config.settle_time,
            )
            log("Current experimental T:")
            print_matrix(current_T, decimals=5)

        run_jacobi_optimization(handles, optimize_config, context)
    finally:
        close_hardware(handles)


if __name__ == "__main__":
    main()
