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
        print(Fore.YELLOW + f"Inter calibration pair file not found: {path}" + Style.RESET_ALL)
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
    port_idx = int(port) - 1
    if port_idx < 0 or port_idx >= len(working_data):
        raise IndexError(f"PORT {port} is out of range for working_data.")
    working_data.iloc[port_idx, 0] = round(float(voltage), 3)


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
        print(Fore.CYAN + f"Opening OPM1: {config.opm1_address}" + Style.RESET_ALL)
        handles.opm1 = cu.open_VISA_connection(config.opm1_address)
    if config.enable_opm2:
        print(Fore.CYAN + f"Opening OPM2: {config.opm2_address}" + Style.RESET_ALL)
        handles.opm2 = cu.open_VISA_connection(config.opm2_address)
    if config.enable_mcv:
        print(Fore.CYAN + f"Opening MCV: {config.ser_address}" + Style.RESET_ALL)
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
                print(Fore.CYAN + f"Closed {name}" + Style.RESET_ALL)
            except Exception as exc:
                print(Fore.YELLOW + f"Failed to close {name}: {exc}" + Style.RESET_ALL)


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
            print(f"Skip MZI {mzi_id}: no entry in {inter_cali_pairs_path}")
            continue

        ports = pair_entry.get("ports", [])
        if not isinstance(ports, list) or len(ports) != 2:
            raise ValueError(f"MZI {mzi_id} in {inter_cali_pairs_path} must contain two ports.")

        upper_v = float(pair_entry.get("upper_arm_voltage", 0.0))
        lower_v = float(pair_entry.get("lower_arm_voltage", 0.0))
        write_port_voltage(int(ports[0]), upper_v, working_data)
        write_port_voltage(int(ports[1]), lower_v, working_data)
        print(
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
    print(Fore.CYAN + Style.BRIGHT + f"Measuring T matrix for N={n}..." + Style.RESET_ALL)
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
        print(f"Input {input_port}: raw={powers}, normalized={np.array2string(col, precision=5)}")

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


def run_jacobi_optimization(handles: HardwareHandles, config: OptimizeConfig, context: dict) -> None:
    """
    Jacobi optimization workflow placeholder.

    Expected next steps:
    1. Define target unitary / target power matrix.
    2. Select Jacobi rotation order on Clements mesh.
    3. Measure current transfer matrix from OPM channels.
    4. Iteratively tune selected MZI heater voltages.
    5. Save optimized voltages and verification data.
    """
    _ = handles
    _ = config
    _ = context
    print(Fore.YELLOW + "Jacobi optimization body is not implemented yet." + Style.RESET_ALL)


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

        print(Fore.GREEN + "JacobiOptimize framework initialized." + Style.RESET_ALL)
        print(f"N={optimize_config.n}, MZI count={optimize_config.n * (optimize_config.n - 1) // 2}")
        print(f"Clements matrix shape: {context['clements_matrix'].shape}")

        if handles.mcv is not None and handles.opm2 is not None:
            current_T = get_T(
                optimize_config.n,
                handles.mcv,
                handles.opm2,
                context["working_data"],
                optimize_config.v_min,
                optimize_config.v_max,
                optimize_config.settle_time,
            )
            print("Current experimental T:")
            print_matrix(current_T, decimals=5)

        run_jacobi_optimization(handles, optimize_config, context)
    finally:
        close_hardware(handles)


if __name__ == "__main__":
    main()
