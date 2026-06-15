import argparse
import builtins
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import upload_matrix as um
import utils.AllDecompositionUtils as du
import utils.communication as cu


DEFAULT_SER_ADDRESS = "COM3"
DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"


def format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def install_elapsed_print():
    if getattr(builtins.print, "_layerwise_elapsed_print", False):
        return
    start_time = time.perf_counter()
    original_print = builtins.print

    def elapsed_print(*args, **kwargs):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = format_elapsed(time.perf_counter() - start_time)
        original_print(f"[{now} | elapsed {elapsed}]", *args, **kwargs)

    elapsed_print._layerwise_elapsed_print = True
    builtins.print = elapsed_print


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def load_target_thetas(path, expected_count):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"target theta file not found: {path}")
    if path.suffix.lower() == ".npy":
        thetas = np.load(path)
    elif path.suffix.lower() == ".csv":
        thetas = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("--target-thetas must be a .npy or .csv file")

    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim == 1:
        if thetas.size == expected_count:
            two_arm = np.zeros((expected_count, 2), dtype=float)
            two_arm[:, 0] = thetas
            return two_arm
        if thetas.size == expected_count * 2:
            return thetas.reshape(expected_count, 2)
    if thetas.shape == (expected_count, 2):
        return thetas
    raise ValueError(
        f"target theta shape must be ({expected_count}, 2), ({expected_count},), "
        f"or flat length {expected_count * 2}; got {thetas.shape}"
    )


def load_inter_cali_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_mzi_table_phase_constraints(target_thetas, mzi_table):
    constrained = np.asarray(target_thetas, dtype=float).copy()
    for mzi_id in range(1, constrained.shape[0] + 1):
        entry = um.get_mzi_entry(mzi_table, mzi_id)
        ports = entry.get("ports", [])
        if len(ports) < 2:
            constrained[mzi_id - 1, 1] = 0.0
    return constrained


def get_layer_mzi_ids(cm, layer_index):
    """Return positive MZI ids in the selected 0-based Clements column."""
    if layer_index < 0 or layer_index >= cm.shape[1]:
        raise ValueError(f"layer_index must be in [0, {cm.shape[1] - 1}], got {layer_index}")
    return [int(x) for x in cm[:, layer_index] if int(x) > 0]


def get_active_heater_ports(mzi_table, mzi_ids):
    active = []
    for mzi_id in mzi_ids:
        entry = um.get_mzi_entry(mzi_table, mzi_id)
        ports = entry.get("ports", [])
        if not ports:
            raise ValueError(f"MZI {mzi_id} has no ports entry in MZI table.")
        for arm_index, port in enumerate(ports):
            active.append(
                {
                    "mzi_id": int(mzi_id),
                    "arm_index": int(arm_index),
                    "port": int(port),
                    "label": f"MZI{int(mzi_id)}_arm{int(arm_index)}",
                }
            )
    return active


def write_checked_port_voltage(port, voltage, working_data, v_min, v_max):
    voltage = float(np.clip(float(voltage), float(v_min), float(v_max)))
    cu.write_port_voltage(int(port), voltage, working_data)
    return voltage


def set_layers_after_to_bar(working_data, cm, layer_index, mzi_table, v_min=0.0, v_max=5.5):
    for col in range(layer_index + 1, cm.shape[1]):
        for mzi_id in get_layer_mzi_ids(cm, col):
            entry = um.get_mzi_entry(mzi_table, mzi_id)
            ports = entry.get("ports", [])
            if not ports:
                continue
            # MZI_table.dtheta_Bar stores alternative Bar settings per arm.
            # This optimizer uses the canonical Bar convention:
            # upper arm = dtheta_Bar[0], lower arm = 0 V.
            bar_voltage = um.get_mzi_state_voltage(mzi_table, mzi_id, "BAR", arm_index=0)
            write_checked_port_voltage(ports[0], bar_voltage, working_data, v_min, v_max)
            if len(ports) > 1:
                write_checked_port_voltage(ports[1], 0.0, working_data, v_min, v_max)


def set_layer_to_bar_voltage_pair(working_data, cm, layer_index, mzi_table, v_min=0.0, v_max=5.5):
    for mzi_id in get_layer_mzi_ids(cm, layer_index):
        entry = um.get_mzi_entry(mzi_table, mzi_id)
        ports = entry.get("ports", [])
        bar_values = entry.get("dtheta_Bar", entry.get("dtheta", []))
        if not ports:
            continue
        for arm_index, port in enumerate(ports):
            if arm_index >= len(bar_values):
                raise ValueError(f"MZI {mzi_id} has port {port} but no dtheta_Bar[{arm_index}].")
            write_checked_port_voltage(port, float(bar_values[arm_index]), working_data, v_min, v_max)


def voltage_from_phase_offset(base_voltage, phase_offset, ppi, heater_r, v_min, v_max):
    base_voltage = float(base_voltage)
    phase_offset = float(phase_offset)
    ppi = float(ppi)
    heater_r = float(heater_r)
    base_power = base_voltage * base_voltage / heater_r
    target_power = base_power + phase_offset / np.pi * ppi
    target_power = max(0.0, target_power)
    voltage = np.sqrt(target_power * heater_r)
    return float(np.clip(voltage, float(v_min), float(v_max)))


def set_layer_to_target_prebiased_pair(
    working_data,
    cm,
    layer_index,
    mzi_table,
    inter_cali_pairs,
    target_thetas,
    v_min=0.0,
    v_max=5.5,
):
    records = []
    for mzi_id in get_layer_mzi_ids(cm, layer_index):
        entry = um.get_mzi_entry(mzi_table, mzi_id)
        ports = entry.get("ports", [])
        ppi_values = entry.get("Ppi", [])
        heater_r_values = entry.get("heater_R", [])
        if not ports:
            continue

        if len(ports) < 2:
            bar_values = entry.get("dtheta_Bar", entry.get("dtheta", []))
            bar_voltage = float(bar_values[0])
            actual_voltage = write_checked_port_voltage(ports[0], bar_voltage, working_data, v_min, v_max)
            records.append(
                {
                    "mzi_id": int(mzi_id),
                    "arm_index": 0,
                    "port": int(ports[0]),
                    "base_voltage": bar_voltage,
                    "target_phase": float(target_thetas[mzi_id - 1, 0]),
                    "initial_voltage": actual_voltage,
                    "mode": "single_heater_bar",
                }
            )
            continue

        pair_entry = inter_cali_pairs.get(str(int(mzi_id)))
        if not pair_entry:
            raise KeyError(f"MZI {mzi_id} not found in inter_cali_pairs.")
        pair_ports = [int(p) for p in pair_entry.get("ports", [])]
        if pair_ports != [int(p) for p in ports]:
            raise ValueError(
                f"MZI {mzi_id} port mismatch: MZI_table ports={ports}, inter_cali_pairs ports={pair_ports}."
            )
        pair_base_values = [
            float(pair_entry["upper_arm_voltage"]),
            float(pair_entry["lower_arm_voltage"]),
        ]

        for arm_index, port in enumerate(ports):
            if arm_index >= len(ppi_values) or arm_index >= len(heater_r_values):
                raise ValueError(f"MZI {mzi_id} missing Ppi/heater_R for arm_index {arm_index}.")
            base_voltage = float(pair_base_values[arm_index])
            target_phase = float(target_thetas[mzi_id - 1, arm_index])
            initial_voltage = voltage_from_phase_offset(
                base_voltage,
                target_phase,
                ppi_values[arm_index],
                heater_r_values[arm_index],
                v_min,
                v_max,
            )
            actual_voltage = write_checked_port_voltage(port, initial_voltage, working_data, v_min, v_max)
            records.append(
                {
                    "mzi_id": int(mzi_id),
                    "arm_index": int(arm_index),
                    "port": int(port),
                    "base_voltage": base_voltage,
                    "target_phase": target_phase,
                    "initial_voltage": actual_voltage,
                    "mode": "inter_cali_pair_target_prebiased",
                }
            )
    return records


def build_target_thetas_for_layer(cm, target_thetas, layer_index):
    thetas = np.zeros_like(target_thetas, dtype=float)
    thetas[:, 0] = np.pi
    thetas[:, 1] = 0.0
    for col in range(0, layer_index + 1):
        for mzi_id in get_layer_mzi_ids(cm, col):
            thetas[mzi_id - 1, :] = target_thetas[mzi_id - 1, :]
    return thetas


def build_target_power_matrix(cm, target_thetas, output_count, layer_index):
    target_thetas_for_loss = build_target_thetas_for_layer(cm, target_thetas, layer_index)
    # Theoretical Bar approximation used by this project: theta1=pi, theta2=0.
    return um.theoretical_power_matrix(cm, target_thetas_for_loss, output_count)


def voltage_to_dry_run_theta(mzi_table, mzi_id, arm_index, voltage):
    entry = um.get_mzi_entry(mzi_table, mzi_id)
    ports = entry.get("ports", [])
    bar_values = entry.get("dtheta_Bar", entry.get("dtheta", []))
    cross_values = entry.get("dtheta_Cross", entry.get("dtheta", []))
    ppi_values = entry.get("Ppi", [])
    heater_r_values = entry.get("heater_R", [])
    if arm_index >= len(bar_values) or arm_index >= len(cross_values):
        return np.pi if arm_index == 0 else 0.0

    if len(ports) >= 2 and arm_index < len(ppi_values) and arm_index < len(heater_r_values):
        pair_entry = getattr(voltage_to_dry_run_theta, "inter_cali_pairs", {}).get(str(int(mzi_id)))
        if pair_entry:
            pair_base_values = [float(pair_entry["upper_arm_voltage"]), float(pair_entry["lower_arm_voltage"])]
            base_voltage = pair_base_values[arm_index]
        else:
            base_voltage = float(bar_values[arm_index])
        base_power = base_voltage**2 / float(heater_r_values[arm_index])
        power = float(voltage) ** 2 / float(heater_r_values[arm_index])
        return float((power - base_power) / float(ppi_values[arm_index]) * np.pi)

    bar_v = float(bar_values[arm_index])
    cross_v = float(cross_values[arm_index])
    if abs(cross_v - bar_v) < 1e-12:
        ratio = 0.0
    else:
        ratio = (float(voltage) - bar_v) / (cross_v - bar_v)
    return float((1.0 - ratio) * np.pi) if arm_index == 0 else float(ratio * np.pi)


def build_dry_run_thetas(args, working_data):
    thetas = build_target_thetas_for_layer(args.cm, args.target_thetas_array, args.layer_index).copy()
    voltage_to_dry_run_theta.inter_cali_pairs = getattr(args, "inter_cali_pairs_data", {})
    for heater in args.active_heaters:
        port_idx = int(heater["port"]) - 1
        voltage = float(working_data.iloc[port_idx, 0])
        thetas[int(heater["mzi_id"]) - 1, int(heater["arm_index"])] = voltage_to_dry_run_theta(
            args.mzi_table_data,
            int(heater["mzi_id"]),
            int(heater["arm_index"]),
            voltage,
        )
    return thetas


def measure_power_matrix(args, working_data):
    if args.dry_run:
        simulated_thetas = build_dry_run_thetas(args, working_data)
        return um.theoretical_power_matrix(args.cm, simulated_thetas, args.output_count)

    if not args.confirm_hardware:
        raise RuntimeError("Refusing hardware access: set --confirm-hardware true with --dry-run false.")
    validate_all_voltages(working_data, args.v_min, args.v_max)
    measure_idx = int(getattr(args, "measure_count", 0))
    measure_context = str(getattr(args, "measure_context", "measurement"))
    figure_path = None
    if getattr(args, "run_dir", None) is not None:
        figure_dir = Path(args.run_dir) / "power_matrix_figures"
        figure_path = figure_dir / f"measure_{measure_idx:04d}_{measure_context}.png"
    print(f"[measure {measure_idx:04d}] {measure_context}: uploading voltages and reading normalized power matrix")
    um.upload_v_checked(args.hardware["mcv"], working_data, args.v_min, args.v_max)
    time.sleep(args.settle_time)
    P_current = um.get_T(
        args.N,
        working_data,
        sleep_time=args.settle_time,
        show_figure=args.show_figure,
        figure_path=figure_path,
        title=f"{measure_context} measure {measure_idx:04d}",
        v_min=args.v_min,
        v_max=args.v_max,
    )
    args.measure_count = measure_idx + 1
    return P_current


def power_matrix_loss(P_current, P_target, eps=1e-12):
    diff = np.asarray(P_current, dtype=float) - np.asarray(P_target, dtype=float)
    return float(np.sum(diff**2) / (np.sum(np.asarray(P_target, dtype=float) ** 2) + eps))


def get_heater_voltages(working_data, active_heaters):
    return np.array([float(working_data.iloc[int(h["port"]) - 1, 0]) for h in active_heaters], dtype=float)


def set_heater_voltages(working_data, active_heaters, voltages, v_min, v_max):
    for heater, voltage in zip(active_heaters, voltages):
        write_checked_port_voltage(heater["port"], voltage, working_data, v_min, v_max)


def finite_difference_gradient(args, working_data, active_heaters, base_voltages, P_target):
    base_voltages = np.asarray(base_voltages, dtype=float)
    grad = np.zeros_like(base_voltages, dtype=float)
    records = []
    context_prefix = str(getattr(args, "measure_context_prefix", "finite_diff"))

    for idx, heater in enumerate(active_heaters):
        v0 = float(base_voltages[idx])
        plus_v = min(args.v_max, v0 + args.delta_v)
        minus_v = max(args.v_min, v0 - args.delta_v)

        plus_loss = None
        minus_loss = None

        if plus_v != v0:
            trial = base_voltages.copy()
            trial[idx] = plus_v
            set_heater_voltages(working_data, active_heaters, trial, args.v_min, args.v_max)
            args.measure_context = f"{context_prefix}_{heater['label']}_plus"
            plus_loss = power_matrix_loss(measure_power_matrix(args, working_data), P_target)

        if minus_v != v0:
            trial = base_voltages.copy()
            trial[idx] = minus_v
            set_heater_voltages(working_data, active_heaters, trial, args.v_min, args.v_max)
            args.measure_context = f"{context_prefix}_{heater['label']}_minus"
            minus_loss = power_matrix_loss(measure_power_matrix(args, working_data), P_target)

        if plus_loss is not None and minus_loss is not None and plus_v != minus_v:
            grad[idx] = (plus_loss - minus_loss) / (plus_v - minus_v)
            mode = "central"
        elif plus_loss is not None:
            args.measure_context = f"{context_prefix}_{heater['label']}_base"
            base_loss = power_matrix_loss(measure_power_matrix(args, working_data), P_target)
            grad[idx] = (plus_loss - base_loss) / (plus_v - v0)
            mode = "forward"
        elif minus_loss is not None:
            args.measure_context = f"{context_prefix}_{heater['label']}_base"
            base_loss = power_matrix_loss(measure_power_matrix(args, working_data), P_target)
            grad[idx] = (base_loss - minus_loss) / (v0 - minus_v)
            mode = "backward"
        else:
            grad[idx] = 0.0
            mode = "fixed"

        set_heater_voltages(working_data, active_heaters, base_voltages, args.v_min, args.v_max)
        records.append(
            {
                "label": heater["label"],
                "mzi_id": int(heater["mzi_id"]),
                "arm_index": int(heater["arm_index"]),
                "port": int(heater["port"]),
                "base_v": v0,
                "plus_v": plus_v,
                "minus_v": minus_v,
                "plus_loss": plus_loss,
                "minus_loss": minus_loss,
                "grad": float(grad[idx]),
                "mode": mode,
            }
        )

    return grad, records


def update_voltages(base_voltages, grad, lr, max_step_v, v_min, v_max):
    step = -float(lr) * np.asarray(grad, dtype=float)
    step = np.clip(step, -float(max_step_v), float(max_step_v))
    return np.clip(np.asarray(base_voltages, dtype=float) + step, float(v_min), float(v_max))


def validate_all_voltages(working_data, v_min, v_max):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltages")
    bad = (voltages < float(v_min)) | (voltages > float(v_max))
    if bad.any():
        raise ValueError(f"Voltage out of range [{v_min}, {v_max}] at rows {bad[bad].index.tolist()}")


def save_voltage_state(path, working_data):
    pd.DataFrame({"port": np.arange(1, len(working_data) + 1), "voltage": working_data.iloc[:, 0]}).to_csv(
        path, index=False
    )


def write_rows(path, rows, fieldnames):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_iteration_logs(run_dir, iter_record, voltage_rows, gradient_rows):
    write_rows(
        run_dir / "iter_log.csv",
        [iter_record],
        [
            "iter",
            "loss",
            "accepted_loss",
            "lr",
            "grad_norm",
            "max_abs_grad",
            "active_heater_labels",
            "active_heater_ports",
            "accepted",
            "rejected",
        ],
    )
    write_rows(
        run_dir / "voltage_log.csv",
        voltage_rows,
        ["iter", "label", "mzi_id", "arm_index", "port", "voltage_before", "voltage_after", "accepted"],
    )
    write_rows(
        run_dir / "gradient_log.csv",
        gradient_rows,
        ["iter", "label", "mzi_id", "arm_index", "port", "base_v", "plus_v", "minus_v", "plus_loss", "minus_loss", "grad", "mode"],
    )


def create_run_dir(out_dir):
    run_dir = Path(out_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def initialize_working_data(args, mzi_table, cm):
    working_data = cu.generate_working_data()
    init_records = []
    if args.init_mode == "bar":
        for mzi_id in [int(x) for x in cm.ravel() if int(x) > 0]:
            entry = um.get_mzi_entry(mzi_table, mzi_id)
            ports = entry.get("ports", [])
            if not ports:
                continue
            # Same canonical Bar convention as set_layers_after_to_bar:
            # use upper-arm Bar voltage only, keep the lower arm at 0 V.
            write_checked_port_voltage(ports[0], um.get_mzi_state_voltage(mzi_table, mzi_id, "BAR", 0), working_data, args.v_min, args.v_max)
            if len(ports) > 1:
                write_checked_port_voltage(ports[1], 0.0, working_data, args.v_min, args.v_max)
        # The active layer starts from its measured Bar voltage pair when two
        # heaters exist. Single-heater edge layers naturally use one voltage.
        init_records = set_layer_to_target_prebiased_pair(
            working_data,
            cm,
            args.layer_index,
            mzi_table,
            args.inter_cali_pairs_data,
            args.target_thetas_array,
            args.v_min,
            args.v_max,
        )
    elif args.init_mode == "prior-file":
        if not args.init_voltage_file:
            raise ValueError("--init-voltage-file is required when --init-mode prior-file")
        prior = pd.read_csv(args.init_voltage_file)
        if "voltage" in prior.columns:
            values = prior["voltage"].to_numpy(dtype=float)
        else:
            values = prior.iloc[:, 0].to_numpy(dtype=float)
        if len(values) > len(working_data):
            raise ValueError("init voltage file has more rows than working_data")
        working_data.iloc[: len(values), 0] = values
    elif args.init_mode != "current":
        raise ValueError(f"Unsupported init mode: {args.init_mode}")

    set_layers_after_to_bar(working_data, cm, args.layer_index, mzi_table, args.v_min, args.v_max)
    validate_all_voltages(working_data, args.v_min, args.v_max)
    args.initial_voltage_records = init_records
    return working_data


def maybe_initialize_hardware(args):
    if args.dry_run:
        return None
    if not args.confirm_hardware:
        raise RuntimeError("Refusing hardware access: set --confirm-hardware true with --dry-run false.")
    hardware = {
        "mcv": cu.open_ser_connection(args.ser_address),
        "opm2": cu.open_VISA_connection(args.opm2_address),
    }
    if hardware["mcv"] is None or hardware["opm2"] is None:
        raise RuntimeError("Failed to initialize mcv or opm2.")
    um.mcv = hardware["mcv"]
    um.opm2 = hardware["opm2"]
    return hardware


def run_optimization(args):
    cm = du.Clements_matrix(args.N)
    mzi_count = args.N * (args.N - 1) // 2
    mzi_table = um.load_mzi_table(args.mzi_table)
    inter_cali_pairs = load_inter_cali_pairs(args.inter_cali_pairs)
    target_thetas_raw = load_target_thetas(args.target_thetas, mzi_count)
    target_thetas = apply_mzi_table_phase_constraints(target_thetas_raw, mzi_table)

    args.layer_index = int(args.layer) - 1
    if args.layer_index < 0 or args.layer_index >= cm.shape[1]:
        raise ValueError(f"--layer is 1-based and must be in [1, {cm.shape[1]}], got {args.layer}")
    args.output_count = args.N - 1
    args.cm = cm
    args.target_thetas_array = target_thetas
    args.mzi_table_data = mzi_table
    args.inter_cali_pairs_data = inter_cali_pairs

    um.BW = um.load_bw_phases(args.bw_dir, expected_count=(args.N - 3) // 2)

    active_mzi_ids = get_layer_mzi_ids(cm, args.layer_index)
    active_heaters = get_active_heater_ports(mzi_table, active_mzi_ids)
    args.active_heaters = active_heaters
    args.hardware = maybe_initialize_hardware(args)

    working_data = initialize_working_data(args, mzi_table, cm)
    P_target = build_target_power_matrix(cm, target_thetas, args.output_count, args.layer_index)
    run_dir = create_run_dir(args.out_dir)
    args.run_dir = run_dir
    args.measure_count = 0

    config = vars(args).copy()
    for key in (
        "cm",
        "target_thetas_array",
        "mzi_table_data",
        "inter_cali_pairs_data",
        "active_heaters",
        "hardware",
        "run_dir",
        "measure_count",
        "measure_context",
        "measure_context_prefix",
        "initial_voltage_records",
    ):
        config.pop(key, None)
    config["layer_index_0based"] = args.layer_index
    config["active_mzi_ids"] = active_mzi_ids
    config["active_heaters"] = active_heaters
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    np.savetxt(run_dir / "target_thetas_used.csv", target_thetas, delimiter=",", header="theta1,theta2", comments="")
    np.savetxt(run_dir / "target_power_matrix.csv", P_target, delimiter=",")
    if getattr(args, "initial_voltage_records", None):
        pd.DataFrame(args.initial_voltage_records).to_csv(run_dir / "initial_voltage_plan.csv", index=False)

    print("Layerwise loss optimization")
    print(f"Layer: {args.layer} (0-based column {args.layer_index})")
    print(f"MZI ids: {active_mzi_ids}")
    print(f"Active heaters: {active_heaters}")
    print(f"Voltage range: [{args.v_min}, {args.v_max}], delta_v={args.delta_v}, lr={args.lr}")
    print(f"dry_run={args.dry_run}, confirm_hardware={args.confirm_hardware}")
    print(f"Run dir: {run_dir}")
    if getattr(args, "initial_voltage_records", None):
        print("Initial active-layer voltages:")
        for rec in args.initial_voltage_records:
            print(
                f"  MZI{rec['mzi_id']}_arm{rec['arm_index']} port {rec['port']}: "
                f"{rec['base_voltage']:.3f} V -> {rec['initial_voltage']:.3f} V "
                f"(target_phase={rec['target_phase']:.6g}, {rec['mode']})"
            )

    prev_loss = None
    final_loss = None
    final_power = None

    for iter_idx in range(args.max_iter):
        args.measure_context = f"iter{iter_idx:03d}_baseline"
        P_current = measure_power_matrix(args, working_data)
        loss = power_matrix_loss(P_current, P_target)
        np.savetxt(run_dir / f"P_current_iter{iter_idx:03d}.csv", P_current, delimiter=",")
        save_voltage_state(run_dir / f"voltage_state_iter{iter_idx:03d}.csv", working_data)

        current_loss = loss
        grad_values = []
        grad_records = []
        voltage_rows = []
        accepted_any = False
        rejected_any = False
        accepted_lrs = []

        for heater in active_heaters:
            heater_before = get_heater_voltages(working_data, [heater])
            args.measure_context_prefix = f"iter{iter_idx:03d}"
            grad, single_grad_records = finite_difference_gradient(
                args,
                working_data,
                [heater],
                heater_before,
                P_target,
            )
            grad_value = float(grad[0]) if grad.size else 0.0
            grad_values.append(grad_value)
            grad_records.extend(single_grad_records)

            accepted = True
            rejected = False
            accepted_lr = float(args.lr)
            heater_after = update_voltages(
                heater_before,
                grad,
                args.lr,
                args.max_step_v,
                args.v_min,
                args.v_max,
            )

            if args.line_search:
                accepted = False
                trial_lr = float(args.lr)
                while trial_lr >= float(args.line_search_min_lr):
                    trial_after = update_voltages(
                        heater_before,
                        grad,
                        trial_lr,
                        args.max_step_v,
                        args.v_min,
                        args.v_max,
                    )
                    set_heater_voltages(working_data, [heater], trial_after, args.v_min, args.v_max)
                    args.measure_context = f"iter{iter_idx:03d}_{heater['label']}_line_search"
                    trial_loss = power_matrix_loss(measure_power_matrix(args, working_data), P_target)
                    if trial_loss < current_loss:
                        accepted = True
                        accepted_lr = trial_lr
                        heater_after = trial_after
                        current_loss = trial_loss
                        break
                    trial_lr *= float(args.line_search_shrink)
                if not accepted:
                    rejected = True
                    heater_after = heater_before.copy()
                    set_heater_voltages(working_data, [heater], heater_before, args.v_min, args.v_max)
            else:
                set_heater_voltages(working_data, [heater], heater_after, args.v_min, args.v_max)
                args.measure_context = f"iter{iter_idx:03d}_{heater['label']}_accepted"
                current_loss = power_matrix_loss(measure_power_matrix(args, working_data), P_target)

            accepted_any = accepted_any or accepted
            rejected_any = rejected_any or rejected
            accepted_lrs.append(float(accepted_lr))
            voltage_rows.append(
                {
                    "iter": iter_idx,
                    "label": heater["label"],
                    "mzi_id": heater["mzi_id"],
                    "arm_index": heater["arm_index"],
                    "port": heater["port"],
                    "voltage_before": float(heater_before[0]),
                    "voltage_after": float(heater_after[0]),
                    "accepted": bool(accepted),
                }
            )

        args.measure_context = f"iter{iter_idx:03d}_final"
        final_power = measure_power_matrix(args, working_data)
        final_loss = power_matrix_loss(final_power, P_target)
        grad_array = np.asarray(grad_values, dtype=float)
        grad_norm = float(np.linalg.norm(grad_array))
        max_abs_grad = float(np.max(np.abs(grad_array))) if grad_array.size else 0.0
        for rec in grad_records:
            rec["iter"] = iter_idx

        append_iteration_logs(
            run_dir,
            {
                "iter": iter_idx,
                "loss": loss,
                "accepted_loss": final_loss,
                "lr": min(accepted_lrs) if accepted_lrs else float(args.lr),
                "grad_norm": grad_norm,
                "max_abs_grad": max_abs_grad,
                "active_heater_labels": ";".join(h["label"] for h in active_heaters),
                "active_heater_ports": ";".join(str(h["port"]) for h in active_heaters),
                "accepted": bool(accepted_any),
                "rejected": bool(rejected_any),
            },
            voltage_rows,
            grad_records,
        )

        print(
            f"iter={iter_idx:03d}, loss={loss:.6g}, final_loss={final_loss:.6g}, "
            f"grad_norm={grad_norm:.6g}, accepted_any={accepted_any}"
        )

        if prev_loss is not None and abs(prev_loss - final_loss) < args.loss_tol:
            break
        prev_loss = final_loss

    if final_power is None:
        args.measure_context = "final"
        final_power = measure_power_matrix(args, working_data)
        final_loss = power_matrix_loss(final_power, P_target)

    np.savetxt(run_dir / "final_power_matrix.csv", final_power, delimiter=",")
    save_voltage_state(run_dir / "final_voltage.csv", working_data)
    print(f"Final loss: {final_loss:.6g}")
    print("Final active voltages:")
    for heater, voltage in zip(active_heaters, get_heater_voltages(working_data, active_heaters)):
        print(f"  {heater['label']} port {heater['port']}: {voltage:.3f} V")
    return run_dir


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Layerwise finite-difference loss optimization for normalized power matrices. "
            "--layer is a 1-based Clements column/layer index; internally it is converted to 0-based."
        )
    )
    parser.add_argument("--N", type=int, default=9)
    parser.add_argument("--layer", type=int, required=True, help="1-based Clements layer/column index to optimize.")
    parser.add_argument("--target-thetas", required=True, help="Target theta file path, .csv or .npy.")
    parser.add_argument("--mzi-table", default=os.path.join("Scandata", "MZI_table.json"))
    parser.add_argument("--inter-cali-pairs", default=os.path.join("Scandata", "inter_cali_pairs.json"))
    parser.add_argument("--bw-dir", default=os.path.join("Scandata", "BW"))
    parser.add_argument("--out-dir", default=os.path.join("results", "LayerwiseLossOptimize"))
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=5.5)
    parser.add_argument("--delta-v", type=positive_float, default=0.02)
    parser.add_argument("--lr", type=positive_float, default=0.1)
    parser.add_argument("--max-step-v", type=positive_float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--loss-tol", type=positive_float, default=1e-4)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--dry-run", type=parse_bool, default=True)
    parser.add_argument("--confirm-hardware", type=parse_bool, default=False)
    parser.add_argument("--show-figure", type=parse_bool, default=False)
    parser.add_argument("--init-mode", choices=["current", "bar", "prior-file"], default="bar")
    parser.add_argument("--init-voltage-file")
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--line-search-shrink", type=positive_float, default=0.5)
    parser.add_argument("--line-search-min-lr", type=positive_float, default=1e-3)
    parser.add_argument("--ser-address", default=DEFAULT_SER_ADDRESS)
    parser.add_argument("--opm2-address", default=DEFAULT_OPM2_ADDRESS)
    return parser


def main():
    install_elapsed_print()
    args = build_arg_parser().parse_args()
    if args.N < 2:
        raise ValueError("--N must be >= 2")
    if args.v_min > args.v_max:
        raise ValueError("--v-min must be <= --v-max")
    if not args.dry_run and not args.confirm_hardware:
        raise RuntimeError("Refusing hardware access: set --confirm-hardware true with --dry-run false.")
    run_optimization(args)


if __name__ == "__main__":
    main()
