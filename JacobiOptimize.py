import os
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style
from scipy.optimize import curve_fit

import utils.communication as cu
import utils.AllDecompositionUtils as du
import compute_j_delta as jd
import inter_calibration as ic


DEFAULT_N = 9
DEFAULT_V_MIN = 0.0
DEFAULT_V_MAX = 5.6
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
    run_j_delta_measurement: bool = False


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
class DirectRunConfig:
    mode: str = "measure_sigma"

    # Common second-column selection.
    mzi_ids: str = "5,6,7,8"
    heaters: str = "5u,5d,6u,6d,7u,7d,8u,8d"

    # J_sigma measurement controls.
    sigma_dir: str = "Scandata/J_sigma"
    sigma_result_dir: str = "results/J_sigma"
    full_result_dir: str = "results/J_full"
    scan_scope: str = "all"  # "all", "baseline", or "perturb"
    skip_existing: bool = False
    dry_run: bool = False
    measure_time: float = 2.0
    phase_points: str = (
        "0,0.785398,1.570796,2.356194,3.141593,"
        "3.926991,4.712389,5.497787,6.283185"
    )
    delta_power_w: float = 0.001
    common_delta_power_w: float = 0.001
    power_limit_w: float = 0.055
    fix_w: bool = True

    # Hardware and mesh config.
    n: int = DEFAULT_N
    mzi_table_path: str = MZI_TABLE_PATH
    opm2_address: str = DEFAULT_OPM2_ADDRESS
    ser_address: str = DEFAULT_SER_ADDRESS

    # Combine paths.
    j_delta_path: str = "results/J_delta/J_delta_rad_per_w.csv"
    j_sigma_path: str = "results/J_sigma/J_sigma_rad_per_w.csv"


DIRECT_RUN_CONFIG = DirectRunConfig()


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


def get_voltage_out_of_range_rows(working_data: pd.DataFrame, v_min: float, v_max: float) -> pd.Series:
    if working_data is None:
        raise ValueError("working_data must not be None.")
    if 0 not in working_data.columns:
        raise ValueError("working_data must contain voltage column 0.")
    voltages = pd.to_numeric(working_data[0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltage values.")
    return voltages[(voltages < v_min) | (voltages > v_max)]


def find_port_owner(mzi_table: dict, port: int) -> str:
    for mzi_id, entry in mzi_table.items():
        ports = entry.get("ports", [])
        if int(port) in [int(p) for p in ports]:
            arm_index = [int(p) for p in ports].index(int(port))
            arm_name = "upper" if arm_index == 0 else "lower"
            return f"MZI {int(mzi_id)} {arm_name}"
    return "unknown owner"


def confirm_or_zero_overlimit_voltages(
    working_data: pd.DataFrame,
    mcv,
    v_min: float,
    v_max: float,
    mzi_table: dict | None = None,
    context_label: str = "",
) -> None:
    out_of_range = get_voltage_out_of_range_rows(working_data, v_min, v_max)
    if out_of_range.empty:
        return

    label = f" during {context_label}" if context_label else ""
    log(f"Voltage limit exceeded{label}. Nothing has been uploaded yet.", Fore.YELLOW, bright=True)
    for row_idx, old_v_raw in out_of_range.items():
        old_v = float(old_v_raw)
        new_v = min(max(old_v, v_min), v_max)
        port = int(row_idx) + 1
        owner = find_port_owner(mzi_table, port) if mzi_table is not None else "unknown owner"
        log(
            f"  row {row_idx}, port {port} ({owner}): {old_v:.3f} V -> limit {new_v:.3f} V",
            Fore.YELLOW,
        )

    try:
        answer = input(f"Clamp all over-limit voltages to [{v_min}, {v_max}] and continue? Yes/No: ")
    except EOFError:
        answer = "No"
    answer = str(answer).strip().lower()

    if answer in {"yes", "y"}:
        for row_idx, old_v_raw in out_of_range.items():
            old_v = float(old_v_raw)
            new_v = min(max(old_v, v_min), v_max)
            port = int(row_idx) + 1
            owner = find_port_owner(mzi_table, port) if mzi_table is not None else "unknown owner"
            log(
                f"Clamp confirmed: row {row_idx}, port {port} ({owner}) "
                f"{old_v:.3f} V -> {new_v:.3f} V",
                Fore.YELLOW,
            )
            working_data.iloc[int(row_idx), 0] = round(new_v, 3)
        return

    log("Clamp rejected. Setting all voltage outputs to 0 V and uploading zero table.", Fore.RED, bright=True)
    working_data.iloc[:, 0] = 0.0
    cu.upload_voltage(mcv, working_data)
    raise RuntimeError("Voltage limit exceeded and user rejected clamping; all outputs were set to 0 V.")


def clamp_working_data_voltages(
    working_data: pd.DataFrame,
    v_min: float,
    v_max: float,
    mzi_table: dict | None = None,
    context_label: str = "",
    mcv=None,
) -> None:
    if mcv is None:
        raise RuntimeError("mcv is required for over-limit voltage confirmation and zero-output fallback.")
    confirm_or_zero_overlimit_voltages(working_data, mcv, v_min, v_max, mzi_table, context_label)


def upload_v_checked(
    mcv,
    working_data: pd.DataFrame,
    v_min: float,
    v_max: float,
    mzi_table: dict | None = None,
    context_label: str = "",
) -> None:
    confirm_or_zero_overlimit_voltages(working_data, mcv, v_min, v_max, mzi_table, context_label)
    validate_voltage_range(working_data, v_min, v_max)
    cu.upload_voltage(mcv, working_data)


def write_port_voltage(port: int, voltage: float, working_data: pd.DataFrame) -> None:
    voltage = round(float(voltage), 3)
    if voltage < DEFAULT_V_MIN or voltage > DEFAULT_V_MAX:
        log(
            f"PORT {port} is being set to {voltage:.3f} V outside "
            f"{DEFAULT_V_MIN}-{DEFAULT_V_MAX} V; confirmation will be required before upload.",
            Fore.YELLOW,
        )

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
        log(
            f"Power target {power_w:.9f} W requires {voltage:.3f} V outside "
            f"[{DEFAULT_V_MIN}, {DEFAULT_V_MAX}] V; confirmation will be required before upload.",
            Fore.YELLOW,
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


def wrap_to_pi(angle):
    return float((float(angle) + np.pi) % (2 * np.pi) - np.pi)


def parse_csv_list(text, item_type=str):
    if text is None or str(text).strip() == "":
        return []
    return [item_type(item.strip()) for item in str(text).split(",") if item.strip()]


def load_scan_file(path):
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    if "dp" not in df.columns:
        raise ValueError(f"{path} must contain column dp.")
    if "pow(uW)" in df.columns:
        power_col = "pow(uW)"
    elif "optical_power_uW" in df.columns:
        power_col = "optical_power_uW"
    else:
        raise ValueError(f"{path} must contain pow(uW) or optical_power_uW.")

    x = pd.to_numeric(df["dp"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        raise ValueError(f"{path} has fewer than four valid points.")
    order = np.argsort(x[mask])
    return {
        "path": path,
        "df": df,
        "x": x[mask][order],
        "y": y[mask][order],
        "power_col": power_col,
    }


def sine_model(x, A, w, beta, b):
    return A * np.sin(w * x + beta) + b


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_sine_curve(x, y, fix_w=None, init_params=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4:
        raise ValueError("Need at least four points for sine fit.")

    y_span = float(np.nanmax(y) - np.nanmin(y))
    A0 = max(0.5 * y_span, 1e-9)
    b0 = float(np.nanmean(y))
    w0 = 1.0
    beta0 = 0.0
    if init_params is not None:
        A0 = max(abs(float(init_params.get("A", A0))), 1e-9)
        w0 = max(abs(float(init_params.get("w", w0))), 1e-9)
        beta0 = float(init_params.get("beta", init_params.get("phi", beta0)))
        b0 = float(init_params.get("b", b0))

    if fix_w is None:
        popt, _ = curve_fit(
            sine_model,
            x,
            y,
            p0=[A0, w0, beta0, b0],
            bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        A, w, beta, b = popt
    else:
        w = float(fix_w)
        popt, _ = curve_fit(
            lambda x_value, A, beta, b: sine_model(x_value, A, w, beta, b),
            x,
            y,
            p0=[A0, beta0, b0],
            bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        A, beta, b = popt

    fitted = sine_model(x, A, w, beta, b)
    return {
        "A": float(A),
        "w": float(w),
        "beta": wrap_to_pi(beta),
        "b": float(b),
        "rmse_uW": rmse(y, fitted),
        "x": x,
        "y": y,
        "fitted": fitted,
    }


def load_sigma_sign(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fit_sigma_phase(scan_file, fix_w=None, init_params=None):
    scan = load_scan_file(scan_file)
    fit = fit_sine_curve(scan["x"], scan["y"], fix_w=fix_w, init_params=init_params)
    fit["scan"] = scan
    return fit


def plot_heatmap(matrix, row_labels, col_labels, out_path, title, colorbar_label="rad/mW"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

    plt.figure(figsize=(9, 4.8))
    im = plt.imshow(values, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xlabel("Perturbed heater")
    plt.ylabel("Observed phase")
    plt.title(title)
    plt.colorbar(im, label=colorbar_label)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            label = "NaN" if not np.isfinite(value) else f"{value:.3g}"
            plt.text(c, r, label, ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_sigma_fit_comparison(baseline_fit, perturbed_fit, observed_mzi, perturbed_heater, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_min = min(np.min(baseline_fit["x"]), np.min(perturbed_fit["x"]))
    x_max = max(np.max(baseline_fit["x"]), np.max(perturbed_fit["x"]))
    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(7, 5))
    plt.plot(baseline_fit["x"], baseline_fit["y"], "o", markersize=4, label="baseline data")
    plt.plot(
        x_grid,
        sine_model(x_grid, baseline_fit["A"], baseline_fit["w"], baseline_fit["beta"], baseline_fit["b"]),
        "-",
        linewidth=1.5,
        label="baseline fit",
    )
    plt.plot(perturbed_fit["x"], perturbed_fit["y"], "s", markersize=4, label="perturbed data")
    plt.plot(
        x_grid,
        sine_model(x_grid, perturbed_fit["A"], perturbed_fit["w"], perturbed_fit["beta"], perturbed_fit["b"]),
        "-",
        linewidth=1.5,
        label="perturbed fit",
    )
    plt.xlabel("Synchronous phase dp (rad)")
    plt.ylabel("Power (uW)")
    plt.title(f"Sigma obs{observed_mzi}, perturb {perturbed_heater}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _missing_sigma_detail(observed_mzi, perturbed_heater, delta_power_w, warning):
    return {
        "observed_mzi": int(observed_mzi),
        "perturbed_heater": str(perturbed_heater),
        "delta_power_w": delta_power_w,
        "baseline_A": np.nan,
        "baseline_w": np.nan,
        "baseline_beta": np.nan,
        "baseline_b": np.nan,
        "perturbed_A": np.nan,
        "perturbed_beta": np.nan,
        "perturbed_b": np.nan,
        "delta_beta_rad": np.nan,
        "sign_s": np.nan,
        "coeff_c": np.nan,
        "delta_sigma_rad": np.nan,
        "J_rad_per_w": np.nan,
        "J_rad_per_mw": np.nan,
        "baseline_rmse_uW": np.nan,
        "perturbed_rmse_uW": np.nan,
        "visibility_ratio": np.nan,
        "warning": warning,
    }


def compute_j_sigma(sigma_dir, mzi_ids, heaters, out_dir, fix_w=True):
    sigma_dir = Path(sigma_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = out_dir / "fit_figures"
    fit_dir.mkdir(parents=True, exist_ok=True)

    row_labels = [f"Sigma{mzi_id}" for mzi_id in mzi_ids]
    col_labels = [f"P{heater}" for heater in heaters]
    j_sigma = np.full((len(mzi_ids), len(heaters)), np.nan, dtype=float)
    details = []
    baseline_fits = {}

    sigma_sign = load_sigma_sign(sigma_dir / "sign_check" / "sigma_sign.json")
    sign_missing_warning = ""
    if not sigma_sign:
        sign_missing_warning = "sigma_sign.json missing; default coeff_c=+2.0"
        log(sign_missing_warning, Fore.YELLOW)

    for observed_mzi in mzi_ids:
        baseline_file = sigma_dir / "baseline" / f"obs{observed_mzi}_inter_scan.txt"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Missing required baseline sigma scan: {baseline_file}")
        baseline_fits[int(observed_mzi)] = fit_sigma_phase(baseline_file)

    for col_idx, heater in enumerate(heaters):
        perturb_dir = sigma_dir / f"perturb_{heater}"
        metadata_file = perturb_dir / "metadata.json"
        delta_power_w = np.nan
        column_warning = ""
        if not perturb_dir.exists():
            column_warning = f"missing perturb directory: {perturb_dir}"
            log(column_warning, Fore.YELLOW)
        elif not metadata_file.exists():
            column_warning = f"missing metadata.json: {metadata_file}"
            log(column_warning, Fore.YELLOW)
        else:
            try:
                with metadata_file.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                delta_power_w = float(metadata["delta_power_w"])
                if not np.isfinite(delta_power_w) or delta_power_w == 0.0:
                    raise ValueError(f"invalid delta_power_w={delta_power_w}")
            except Exception as exc:
                column_warning = f"metadata error: {exc}"
                log(column_warning, Fore.YELLOW)

        for row_idx, observed_mzi in enumerate(mzi_ids):
            baseline_fit = baseline_fits[int(observed_mzi)]
            if column_warning:
                details.append(_missing_sigma_detail(observed_mzi, heater, delta_power_w, column_warning))
                continue

            perturb_file = perturb_dir / f"obs{observed_mzi}_inter_scan.txt"
            if not perturb_file.exists():
                warning = f"missing perturb scan: {perturb_file}"
                log(warning, Fore.YELLOW)
                details.append(_missing_sigma_detail(observed_mzi, heater, delta_power_w, warning))
                continue

            try:
                fix_w_value = baseline_fit["w"] if fix_w else None
                perturbed_fit = fit_sigma_phase(
                    perturb_file,
                    fix_w=fix_w_value,
                    init_params=baseline_fit,
                )
                delta_beta = wrap_to_pi(perturbed_fit["beta"] - baseline_fit["beta"])
                sign_entry = sigma_sign.get(str(int(observed_mzi)), {})
                sign_s = int(sign_entry.get("s", 1))
                coeff_c = float(sign_entry.get("c", 2.0 * sign_s))
                delta_sigma = coeff_c * delta_beta
                j_value = delta_sigma / delta_power_w
                visibility_ratio = (
                    float(perturbed_fit["A"]) / float(baseline_fit["A"])
                    if float(baseline_fit["A"]) != 0.0
                    else np.nan
                )

                warnings = []
                if sign_missing_warning:
                    warnings.append(sign_missing_warning)
                if abs(delta_beta) > np.pi / 2:
                    warnings.append("possible branch jump or perturbation too large")
                if np.isfinite(visibility_ratio) and visibility_ratio < 0.3:
                    warnings.append("low visibility, phase extraction unreliable")

                details.append(
                    {
                        "observed_mzi": int(observed_mzi),
                        "perturbed_heater": str(heater),
                        "delta_power_w": delta_power_w,
                        "baseline_A": baseline_fit["A"],
                        "baseline_w": baseline_fit["w"],
                        "baseline_beta": baseline_fit["beta"],
                        "baseline_b": baseline_fit["b"],
                        "perturbed_A": perturbed_fit["A"],
                        "perturbed_beta": perturbed_fit["beta"],
                        "perturbed_b": perturbed_fit["b"],
                        "delta_beta_rad": delta_beta,
                        "sign_s": sign_s,
                        "coeff_c": coeff_c,
                        "delta_sigma_rad": delta_sigma,
                        "J_rad_per_w": j_value,
                        "J_rad_per_mw": j_value / 1000.0,
                        "baseline_rmse_uW": baseline_fit["rmse_uW"],
                        "perturbed_rmse_uW": perturbed_fit["rmse_uW"],
                        "visibility_ratio": visibility_ratio,
                        "warning": "; ".join(warnings),
                    }
                )
                j_sigma[row_idx, col_idx] = j_value
                plot_sigma_fit_comparison(
                    baseline_fit,
                    perturbed_fit,
                    observed_mzi,
                    heater,
                    fit_dir / f"obs{observed_mzi}_perturb_{heater}.png",
                )
            except Exception as exc:
                warning = f"fit failed: {exc}"
                log(f"obs{observed_mzi}, perturb {heater}: {warning}", Fore.YELLOW)
                details.append(_missing_sigma_detail(observed_mzi, heater, delta_power_w, warning))

    j_w_df = pd.DataFrame(j_sigma, index=row_labels, columns=col_labels)
    j_mw_df = j_w_df / 1000.0
    j_w_df.to_csv(out_dir / "J_sigma_rad_per_w.csv")
    j_mw_df.to_csv(out_dir / "J_sigma_rad_per_mw.csv")
    pd.DataFrame(details).to_csv(out_dir / "sigma_phase_shift_details.csv", index=False)
    plot_heatmap(
        j_mw_df.to_numpy(dtype=float),
        row_labels,
        col_labels,
        out_dir / "J_sigma_heatmap_rad_per_mw.png",
        "J_sigma (rad/mW)",
    )
    log(f"Saved J_sigma results to {out_dir}", Fore.GREEN)
    return j_w_df


def _phase_row_id(index_value):
    digits = "".join(ch for ch in str(index_value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse MZI id from row label {index_value!r}")
    return int(digits)


def combine_j_delta_sigma(j_delta_path, j_sigma_path, out_dir):
    j_delta_path = Path(j_delta_path)
    j_sigma_path = Path(j_sigma_path)
    if not j_delta_path.exists():
        fallback = Path("results") / "J_delta_rad_per_w.csv"
        if str(j_delta_path).replace("\\", "/") == "results/J_delta/J_delta_rad_per_w.csv" and fallback.exists():
            j_delta_path = fallback
    j_delta = pd.read_csv(j_delta_path, index_col=0)
    j_sigma = pd.read_csv(j_sigma_path, index_col=0)

    delta_by_id = {_phase_row_id(idx): j_delta.loc[idx] for idx in j_delta.index}
    sigma_by_id = {_phase_row_id(idx): j_sigma.loc[idx] for idx in j_sigma.index}
    common_ids = sorted(set(delta_by_id) & set(sigma_by_id))
    if not common_ids:
        raise ValueError("No common MZI rows found between J_delta and J_sigma.")

    common_cols = [col for col in j_sigma.columns if col in j_delta.columns]
    if not common_cols:
        raise ValueError("No common heater columns found between J_delta and J_sigma.")

    delta_values = pd.DataFrame([delta_by_id[mzi_id][common_cols] for mzi_id in common_ids])
    sigma_values = pd.DataFrame([sigma_by_id[mzi_id][common_cols] for mzi_id in common_ids])
    delta_values.index = [f"phi_u{mzi_id}" for mzi_id in common_ids]
    sigma_values.index = delta_values.index
    delta_values.columns = common_cols
    sigma_values.columns = common_cols

    j_upper = (sigma_values + delta_values) / 2.0
    j_lower = (sigma_values - delta_values) / 2.0
    j_lower.index = [f"phi_d{mzi_id}" for mzi_id in common_ids]
    theta_rows = []
    theta_index = []
    for mzi_id in common_ids:
        theta_rows.append(j_upper.loc[f"phi_u{mzi_id}"])
        theta_index.append(f"phi_u{mzi_id}")
        theta_rows.append(j_lower.loc[f"phi_d{mzi_id}"])
        theta_index.append(f"phi_d{mzi_id}")
    j_theta = pd.DataFrame(theta_rows, index=theta_index, columns=common_cols)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j_upper.to_csv(out_dir / "J_upper_rad_per_w.csv")
    j_lower.to_csv(out_dir / "J_lower_rad_per_w.csv")
    j_theta.to_csv(out_dir / "J_theta_rad_per_w.csv")
    j_upper_mw = j_upper / 1000.0
    j_lower_mw = j_lower / 1000.0
    j_theta_mw = j_theta / 1000.0
    j_upper_mw.to_csv(out_dir / "J_upper_rad_per_mw.csv")
    j_lower_mw.to_csv(out_dir / "J_lower_rad_per_mw.csv")
    j_theta_mw.to_csv(out_dir / "J_theta_rad_per_mw.csv")
    plot_heatmap(
        j_upper_mw.to_numpy(dtype=float),
        list(j_upper_mw.index),
        list(j_upper_mw.columns),
        out_dir / "J_upper_heatmap_rad_per_mw.png",
        "J_upper (rad/mW)",
    )
    plot_heatmap(
        j_lower_mw.to_numpy(dtype=float),
        list(j_lower_mw.index),
        list(j_lower_mw.columns),
        out_dir / "J_lower_heatmap_rad_per_mw.png",
        "J_lower (rad/mW)",
    )
    plot_heatmap(
        j_theta_mw.to_numpy(dtype=float),
        list(j_theta_mw.index),
        list(j_theta_mw.columns),
        out_dir / "J_theta_heatmap_rad_per_mw.png",
        "J_theta (rad/mW)",
    )
    log(f"Saved combined phase response matrices to {out_dir}", Fore.GREEN)
    return j_upper, j_lower, j_theta


def set_heater_power(file_data, port, resistance, power_w):
    voltage = power_to_voltage(power_w, resistance)
    write_port_voltage(int(port), voltage, file_data)
    return voltage


def restore_baseline_powers(ser, file_data, baseline_data, v_min=DEFAULT_V_MIN, v_max=DEFAULT_V_MAX):
    file_data.iloc[:, 0] = baseline_data.iloc[:, 0].to_numpy(copy=True)
    upload_v_checked(ser, file_data, v_min, v_max)


def setup_hardware(opm2_address=DEFAULT_OPM2_ADDRESS, ser_address=DEFAULT_SER_ADDRESS, enable_opm2=True):
    handles = HardwareHandles()
    if enable_opm2:
        log(f"Opening OPM2: {opm2_address}", Fore.CYAN)
        handles.opm2 = cu.open_VISA_connection(opm2_address)
    log(f"Opening MCV: {ser_address}", Fore.CYAN)
    handles.mcv = cu.open_ser_connection(ser_address)
    return handles


def _get_target_arm_calibration(mzi_table, target):
    entry = mzi_table[str(int(target))]
    ports = [int(port) for port in entry.get("ports", [])]
    heater_r = [float(value) for value in entry.get("heater_R", [])]
    ppi = [float(value) for value in entry.get("Ppi", [])]
    if len(ports) < 2 or len(heater_r) < 2 or len(ppi) < 2:
        raise ValueError(f"MZI {target} requires two ports, heater_R, and Ppi for sigma scan.")
    return ports, heater_r, ppi


def fold_power_to_limit(power_w: float, period_w: float, power_limit_w: float) -> tuple[float, int]:
    power = float(power_w)
    period = float(period_w)
    if period <= 0.0:
        raise ValueError("fold period must be positive.")
    fold_count = 0
    while power > float(power_limit_w):
        power -= period
        fold_count += 1
    return max(0.0, power), fold_count


def _read_opm_uW(pwm, output_channel):
    values = cu.read_pow(pwm)
    idx = int(output_channel) - 1
    if idx < 0 or idx >= len(values):
        raise IndexError(f"Output channel {output_channel} not available.")
    return float(values[idx]) * 1e6


def scan_mzi_sigma_interference(
    target: int,
    perturb_label: str,
    save_dir: Path,
    ser,
    pwm,
    measure_time: float,
    file_data,
    N: int = 9,
    phase_points=None,
    perturbed_heater: str | None = None,
    mzi_table=None,
    power_limit_w: float = 0.055,
    perturb_delta_power_w: float = 0.0,
    common_delta_power_w: float = 0.0,
):
    target = int(target)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if phase_points is None:
        phase_points = np.linspace(0.0, 2 * np.pi, 9)
    phase_points = [float(value) for value in phase_points]
    if mzi_table is None:
        mzi_table = load_mzi_table()

    ic.mzi_table = mzi_table
    ic.working_data = file_data
    ic.mcv = ser
    path, input_idx, output_idx, state, bmzi = ic.find_Bmzi_path(target, N)
    ic.build_Bmzi(target, N)
    clamp_working_data_voltages(
        file_data,
        DEFAULT_V_MIN,
        DEFAULT_V_MAX,
        mzi_table=mzi_table,
        context_label=f"build_Bmzi target {target}",
        mcv=ser,
    )
    if perturbed_heater and str(perturbed_heater)[-1].lower() in {"u", "d"} and float(perturb_delta_power_w) != 0.0:
        _apply_sigma_perturbation(file_data, mzi_table, perturbed_heater, perturb_delta_power_w)
        clamp_working_data_voltages(
            file_data,
            DEFAULT_V_MIN,
            DEFAULT_V_MAX,
            mzi_table=mzi_table,
            context_label=f"perturb {perturbed_heater}",
            mcv=ser,
        )

    ports, heater_r, ppi = _get_target_arm_calibration(mzi_table, target)
    if float(common_delta_power_w) != 0.0:
        for port, resistance in zip(ports[:2], heater_r[:2]):
            base_v = get_port_voltage(file_data, port)
            base_p = voltage_to_power_w(base_v, resistance)
            set_heater_power(file_data, port, resistance, base_p + float(common_delta_power_w))

    base_v_upper = get_port_voltage(file_data, ports[0])
    base_v_lower = get_port_voltage(file_data, ports[1])
    p_upper_base = voltage_to_power_w(base_v_upper, heater_r[0])
    p_lower_base = voltage_to_power_w(base_v_lower, heater_r[1])
    period_upper = 2.0 * float(ppi[0])
    period_lower = 2.0 * float(ppi[1])

    rows = []
    for dp in phase_points:
        p_upper_unfolded = p_upper_base + dp / np.pi * ppi[0]
        p_lower_unfolded = p_lower_base + dp / np.pi * ppi[1]
        warning = ""
        p_upper, upper_fold_count = fold_power_to_limit(p_upper_unfolded, period_upper, power_limit_w)
        p_lower, lower_fold_count = fold_power_to_limit(p_lower_unfolded, period_lower, power_limit_w)
        if upper_fold_count or lower_fold_count:
            warning = (
                f"period fold applied: upper {p_upper_unfolded:.9f}->{p_upper:.9f} W "
                f"({upper_fold_count} folds), lower {p_lower_unfolded:.9f}->{p_lower:.9f} W "
                f"({lower_fold_count} folds)"
            )
            log(f"Sigma scan target {target}, dp={dp:.6f}: {warning}", Fore.YELLOW)

        v_upper = set_heater_power(file_data, ports[0], heater_r[0], p_upper)
        v_lower = set_heater_power(file_data, ports[1], heater_r[1], p_lower)
        upload_v_checked(
            ser,
            file_data,
            DEFAULT_V_MIN,
            DEFAULT_V_MAX,
            mzi_table=mzi_table,
            context_label=f"sigma scan target {target}, dp={dp:.6f}",
        )
        time.sleep(float(measure_time))
        power_uW = _read_opm_uW(pwm, int(output_idx) + 1)
        rows.append(
            {
                "target": target,
                "observed_mzi": target,
                "dp": dp,
                "pow(uW)": power_uW,
                "v_primary": v_upper,
                "v_secondary": v_lower,
                "p_primary": p_upper,
                "p_secondary": p_lower,
                "p_primary_unfolded": p_upper_unfolded,
                "p_secondary_unfolded": p_lower_unfolded,
                "upper_fold_count": upper_fold_count,
                "lower_fold_count": lower_fold_count,
                "period_primary": period_upper,
                "period_secondary": period_lower,
                "scan_type": "sigma_sync",
                "output_channel": int(output_idx) + 1,
                "input_channel": int(input_idx) + 1,
                "path": json.dumps([int(x) for x in path]),
                "state": json.dumps([str(x) for x in state]),
                "bmzi": int(bmzi),
                "perturb_label": perturb_label,
                "perturbed_heater": "" if perturbed_heater is None else str(perturbed_heater),
                "warning": warning,
            }
        )
        log(f"Sigma scan target {target}, dp={dp:.6f}, OP={power_uW:.6f} uW")

    out_path = save_dir / f"obs{target}_inter_scan.txt"
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.12f")
    log(f"Saved sigma scan to {out_path}", Fore.GREEN)
    return out_path


def _apply_sigma_perturbation(file_data, mzi_table, heater_label, delta_power_w):
    mzi_id, arm = parse_heater_label(heater_label)
    info = get_mzi_arm_info(mzi_table, mzi_id, arm)
    baseline_voltage = get_port_voltage(file_data, info["port"])
    baseline_power = voltage_to_power_w(baseline_voltage, info["resistance"])
    perturbed_power = baseline_power + float(delta_power_w)
    perturbed_voltage = set_heater_power(file_data, info["port"], info["resistance"], perturbed_power)
    return {
        "perturbed_heater": heater_label,
        "delta_power_w": float(delta_power_w),
        "baseline_power_w": baseline_power,
        "perturbed_power_w": perturbed_power,
        "baseline_voltage_v": baseline_voltage,
        "perturbed_voltage_v": perturbed_voltage,
        "port": info["port"],
        "heater_R_ohm": info["resistance"],
    }


def measure_sigma(args):
    mzi_ids = parse_csv_list(args.mzi_ids, int)
    heaters = parse_csv_list(args.heaters, str)
    phase_points = parse_csv_list(args.phase_points, float)
    out_dir = Path(args.out_dir)
    scan_scope = str(args.scan_scope).strip().lower()
    if scan_scope not in {"all", "baseline", "perturb"}:
        raise ValueError("--scan_scope must be one of: all, baseline, perturb")

    if args.dry_run:
        log("Dry-run measure_sigma plan:", Fore.YELLOW)
        if scan_scope in {"all", "baseline"}:
            for target in mzi_ids:
                log(f"baseline target={target} -> {out_dir / 'baseline' / f'obs{target}_inter_scan.txt'}")
        if scan_scope in {"all", "perturb"}:
            for heater in heaters:
                for target in mzi_ids:
                    log(f"perturb={heater}, target={target} -> {out_dir / f'perturb_{heater}' / f'obs{target}_inter_scan.txt'}")
        log(f"phase_points={phase_points}")
        return

    mzi_table = load_mzi_table(args.mzi_table)
    file_data = cu.generate_working_data()
    context_config = OptimizeConfig(n=args.N, mzi_table_path=args.mzi_table)
    file_data = initialize_working_data(context_config, mzi_table)
    handles = setup_hardware(args.opm2_address, args.ser_address, enable_opm2=True)
    try:
        base_data = file_data.copy(deep=True)
        if scan_scope in {"all", "baseline"}:
            for target in mzi_ids:
                save_path = out_dir / "baseline" / f"obs{target}_inter_scan.txt"
                if args.skip_existing and save_path.exists():
                    log(f"Skip existing baseline scan: {save_path}", Fore.YELLOW)
                    continue
                scan_mzi_sigma_interference(
                    target,
                    "baseline",
                    out_dir / "baseline",
                    handles.mcv,
                    handles.opm2,
                    args.measure_time,
                    file_data,
                    N=args.N,
                    phase_points=phase_points,
                    perturbed_heater=None,
                    mzi_table=mzi_table,
                    power_limit_w=args.power_limit_w,
                )
                file_data.iloc[:, 0] = base_data.iloc[:, 0].to_numpy(copy=True)

        if scan_scope in {"all", "perturb"}:
            for heater in heaters:
                perturb_dir = out_dir / f"perturb_{heater}"
                perturb_dir.mkdir(parents=True, exist_ok=True)
                file_data.iloc[:, 0] = base_data.iloc[:, 0].to_numpy(copy=True)
                metadata = _apply_sigma_perturbation(file_data, mzi_table, heater, args.delta_power_w)
                with (perturb_dir / "metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                perturbed_base = file_data.copy(deep=True)
                for target in mzi_ids:
                    save_path = perturb_dir / f"obs{target}_inter_scan.txt"
                    if args.skip_existing and save_path.exists():
                        log(f"Skip existing perturb scan: {save_path}", Fore.YELLOW)
                        continue
                    file_data.iloc[:, 0] = perturbed_base.iloc[:, 0].to_numpy(copy=True)
                    scan_mzi_sigma_interference(
                        target,
                        f"perturb_{heater}",
                        perturb_dir,
                        handles.mcv,
                        handles.opm2,
                        args.measure_time,
                        file_data,
                        N=args.N,
                        phase_points=phase_points,
                        perturbed_heater=heater,
                        mzi_table=mzi_table,
                        power_limit_w=args.power_limit_w,
                        perturb_delta_power_w=args.delta_power_w,
                    )
                file_data.iloc[:, 0] = base_data.iloc[:, 0].to_numpy(copy=True)
                upload_v_checked(handles.mcv, file_data, DEFAULT_V_MIN, DEFAULT_V_MAX)
    finally:
        close_hardware(handles)


def sign_check(args):
    mzi_ids = parse_csv_list(args.mzi_ids, int)
    out_dir = Path(args.out_dir)
    sign_dir = out_dir / "sign_check"
    sign_dir.mkdir(parents=True, exist_ok=True)
    phase_points = parse_csv_list(args.phase_points, float) if args.phase_points else list(np.linspace(0.0, 2 * np.pi, 9))
    if args.dry_run:
        for target in mzi_ids:
            log(f"sign_check target={target} -> {sign_dir / f'obs{target}_common_plus.txt'}")
        return

    mzi_table = load_mzi_table(args.mzi_table)
    file_data = initialize_working_data(OptimizeConfig(n=args.N, mzi_table_path=args.mzi_table), mzi_table)
    handles = setup_hardware(args.opm2_address, args.ser_address, enable_opm2=True)
    sigma_sign = {}
    try:
        base_data = file_data.copy(deep=True)
        for target in mzi_ids:
            baseline_path = scan_mzi_sigma_interference(
                target,
                "sign_baseline",
                sign_dir,
                handles.mcv,
                handles.opm2,
                args.measure_time,
                file_data,
                N=args.N,
                phase_points=phase_points,
                mzi_table=mzi_table,
                power_limit_w=args.power_limit_w,
            )
            baseline_fit = fit_sigma_phase(baseline_path)
            file_data.iloc[:, 0] = base_data.iloc[:, 0].to_numpy(copy=True)
            plus_path = scan_mzi_sigma_interference(
                target,
                "sign_common_plus",
                sign_dir,
                handles.mcv,
                handles.opm2,
                args.measure_time,
                file_data,
                N=args.N,
                phase_points=phase_points,
                perturbed_heater=f"{target}common",
                mzi_table=mzi_table,
                power_limit_w=args.power_limit_w,
                common_delta_power_w=args.common_delta_power_w,
            )
            plus_target = sign_dir / f"obs{target}_common_plus.txt"
            Path(plus_path).replace(plus_target)
            plus_fit = fit_sigma_phase(plus_target, fix_w=baseline_fit["w"], init_params=baseline_fit)
            delta_beta = wrap_to_pi(plus_fit["beta"] - baseline_fit["beta"])
            sign_s = 1 if delta_beta >= 0 else -1
            coeff_c = 2.0 * sign_s
            sigma_sign[str(target)] = {
                "s": sign_s,
                "c": coeff_c,
                "delta_beta_rad": delta_beta,
                "note": "c uses default +/-2 scale from common-plus sign check",
            }
            file_data.iloc[:, 0] = base_data.iloc[:, 0].to_numpy(copy=True)
        with (sign_dir / "sigma_sign.json").open("w", encoding="utf-8") as f:
            json.dump(sigma_sign, f, indent=2)
        log(f"Saved sigma sign check to {sign_dir / 'sigma_sign.json'}", Fore.GREEN)
    finally:
        close_hardware(handles)


def direct_main(config: DirectRunConfig = DIRECT_RUN_CONFIG):
    mode = str(config.mode).strip().lower()
    log(f"Direct run mode: {mode}", Fore.CYAN, bright=True)

    if mode == "measure_sigma":
        args = SimpleNamespace(
            mzi_ids=config.mzi_ids,
            heaters=config.heaters,
            out_dir=config.sigma_dir,
            mzi_table=config.mzi_table_path,
            measure_time=config.measure_time,
            phase_points=config.phase_points,
            delta_power_w=config.delta_power_w,
            power_limit_w=config.power_limit_w,
            scan_scope=config.scan_scope,
            skip_existing=config.skip_existing,
            opm2_address=config.opm2_address,
            ser_address=config.ser_address,
            N=config.n,
            dry_run=config.dry_run,
        )
        measure_sigma(args)
        return

    if mode == "compute_sigma":
        compute_j_sigma(
            sigma_dir=config.sigma_dir,
            mzi_ids=parse_csv_list(config.mzi_ids, int),
            heaters=parse_csv_list(config.heaters, str),
            out_dir=config.sigma_result_dir,
            fix_w=config.fix_w,
        )
        return

    if mode == "combine_delta_sigma":
        combine_j_delta_sigma(config.j_delta_path, config.j_sigma_path, config.full_result_dir)
        return

    if mode == "sign_check":
        args = SimpleNamespace(
            mzi_ids=config.mzi_ids,
            out_dir=config.sigma_dir,
            mzi_table=config.mzi_table_path,
            measure_time=config.measure_time,
            phase_points=config.phase_points,
            common_delta_power_w=config.common_delta_power_w,
            power_limit_w=config.power_limit_w,
            opm2_address=config.opm2_address,
            ser_address=config.ser_address,
            N=config.n,
            dry_run=config.dry_run,
        )
        sign_check(args)
        return

    raise ValueError(
        "DirectRunConfig.mode must be one of: "
        "measure_sigma, compute_sigma, combine_delta_sigma, sign_check"
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
    direct_main()
