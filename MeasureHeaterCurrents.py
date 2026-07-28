import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import upload_matrix as um
import utils.communication as cu


FOCUS_NETWORK_MZIS = {2, 5, 6, 10, 13, 14, 18, 21, 22, 26, 29, 30, 34}
FOCUS_SWITCH_MZIS = {3, 4}


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a voltage state and diagnose all heater currents.")
    parser.add_argument("--voltage-file", required=True)
    parser.add_argument("--mzi-table", default="Scandata/MZI_table.json")
    parser.add_argument("--switch-table", default="IN_MZI.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ser-address", default="COM3")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-delay", type=float, default=0.2)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=5.5)
    return parser.parse_args()


def load_voltage_state(path):
    state = pd.read_csv(path)
    if "voltage" not in state.columns:
        raise ValueError("Voltage file must contain a 'voltage' column")
    if "port" in state.columns:
        state = state.sort_values("port")
    values = state["voltage"].to_numpy(dtype=float)
    working_data = cu.generate_working_data()
    if len(values) > len(working_data):
        raise ValueError("Voltage file has more rows than the hardware voltage table")
    working_data.iloc[: len(values), 0] = values
    return working_data


def build_heater_metadata(mzi_table_path, switch_table_path):
    with open(mzi_table_path, "r", encoding="utf-8") as handle:
        mzi_table = json.load(handle)
    rows = []
    for mzi_key, entry in mzi_table.items():
        mzi_id = int(mzi_key)
        ports = entry.get("ports", [])
        resistances = entry.get("heater_R", [])
        for arm_index, port in enumerate(ports):
            resistance = float(resistances[arm_index]) if arm_index < len(resistances) else np.nan
            rows.append(
                {
                    "heater_type": "network",
                    "mzi_id": mzi_id,
                    "arm_index": arm_index,
                    "port": int(port),
                    "configured_resistance_ohm": resistance,
                    "focus_3_4_path": mzi_id in FOCUS_NETWORK_MZIS,
                }
            )
    switches = pd.read_csv(switch_table_path)
    for _, entry in switches.iterrows():
        mzi_id = int(entry["MZI"])
        rows.append(
            {
                "heater_type": "input_switch",
                "mzi_id": mzi_id,
                "arm_index": 0,
                "port": int(entry["PORT"]),
                "configured_resistance_ohm": float(entry["HEATER_R_OHM"]),
                "focus_3_4_path": mzi_id in FOCUS_SWITCH_MZIS,
            }
        )
    return pd.DataFrame(rows).sort_values(["heater_type", "mzi_id", "arm_index"])


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    working_data = load_voltage_state(args.voltage_file)
    metadata = build_heater_metadata(args.mzi_table, args.switch_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serial_connection = cu.open_ser_connection(args.ser_address)
    if serial_connection is None:
        raise RuntimeError(f"Could not open serial port {args.ser_address}")
    try:
        um.upload_v_checked(serial_connection, working_data, args.v_min, args.v_max)
        readings = []
        for _ in range(args.samples):
            readings.append(cu.read_current(serial_connection))
            time.sleep(args.sample_delay)
    finally:
        serial_connection.close()

    current_array = np.array(
        [[np.nan if value is None else float(value) for value in sample] for sample in readings],
        dtype=float,
    )
    median_current = np.nanmedian(current_array, axis=0)
    std_current = np.nanstd(current_array, axis=0)
    all_ports = pd.DataFrame(
        {
            "port": np.arange(1, len(working_data) + 1),
            "voltage_v": working_data.iloc[:, 0].to_numpy(dtype=float),
            "measured_current_mA": median_current,
            "current_std_mA": std_current,
        }
    )
    raw_samples = pd.DataFrame(
        current_array.T,
        columns=[f"current_sample_{index + 1}_mA" for index in range(args.samples)],
    )
    raw_samples.insert(0, "port", np.arange(1, len(working_data) + 1))
    diagnostics = metadata.merge(all_ports, on="port", how="left")
    diagnostics["expected_current_mA"] = (
        diagnostics["voltage_v"] / diagnostics["configured_resistance_ohm"] * 1000.0
    )
    diagnostics["measured_resistance_ohm"] = np.where(
        diagnostics["measured_current_mA"].abs() > 1e-6,
        diagnostics["voltage_v"] / (diagnostics["measured_current_mA"] * 1e-3),
        np.nan,
    )
    diagnostics["relative_current_error"] = np.where(
        diagnostics["expected_current_mA"].abs() > 1e-6,
        (diagnostics["measured_current_mA"] - diagnostics["expected_current_mA"])
        / diagnostics["expected_current_mA"],
        np.nan,
    )
    diagnostics["relative_current_std"] = np.where(
        diagnostics["expected_current_mA"].abs() > 1e-6,
        diagnostics["current_std_mA"] / diagnostics["expected_current_mA"].abs(),
        np.nan,
    )
    diagnostics["status"] = np.where(
        diagnostics["voltage_v"].abs() < 0.5,
        "LOW_VOLTAGE_NOT_ASSESSED",
        np.where(
            diagnostics["measured_current_mA"].isna(),
            "NO_READING",
            np.where(
                diagnostics["relative_current_std"] > args.relative_tolerance,
                "UNSTABLE",
                np.where(
                diagnostics["relative_current_error"].abs() > args.relative_tolerance,
                "ANOMALY",
                "OK",
                ),
            ),
        ),
    )

    all_ports.to_csv(out_dir / "all_128_port_currents.csv", index=False)
    raw_samples.to_csv(out_dir / "all_128_port_current_samples.csv", index=False)
    diagnostics.to_csv(out_dir / "all_heater_current_diagnostics.csv", index=False)
    focus = diagnostics.loc[diagnostics["focus_3_4_path"]].copy()
    focus.to_csv(out_dir / "channel_3_4_path_heater_diagnostics.csv", index=False)
    print(focus.to_string(index=False))
    print(f"All assessed heater anomalies: {diagnostics['status'].isin(['ANOMALY', 'UNSTABLE']).sum()}")
    print(f"Focus-path anomalies: {focus['status'].isin(['ANOMALY', 'UNSTABLE']).sum()}")
    print(f"Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
