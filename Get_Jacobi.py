import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import utils.communication as cu
import utils.AllDecompositionUtils as du
from inter_calibration import find_Bmzi_path, switch_IN, write_port_voltage


DEFAULT_OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"
DEFAULT_SER_ADDRESS = "COM3"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv_list(text, item_type=str):
    if text is None or str(text).strip() == "":
        return []
    return [item_type(item.strip()) for item in str(text).split(",") if item.strip()]


def load_mzi_table(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_voltage_range(working_data, v_min, v_max):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltages.")
    bad = (voltages < float(v_min)) | (voltages > float(v_max))
    if bad.any():
        raise ValueError(f"Voltage out of range [{v_min}, {v_max}] at rows: {bad[bad].index.tolist()}")


def voltage_range_summary(working_data):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltages.")
    min_idx = int(voltages.idxmin())
    max_idx = int(voltages.idxmax())
    return {
        "min_v": float(voltages.iloc[min_idx]),
        "max_v": float(voltages.iloc[max_idx]),
        "min_row": min_idx,
        "max_row": max_idx,
    }


def upload_voltage_checked(mcv, working_data, args, label):
    validate_voltage_range(working_data, 0.0, float(args.voltage_limit_v))
    summary = voltage_range_summary(working_data)
    print(
        "[Get_Jacobi] "
        f"{label}: voltage protection passed, "
        f"min={summary['min_v']:.6f} V(row {summary['min_row']}), "
        f"max={summary['max_v']:.6f} V(row {summary['max_row']}), "
        f"limit=[0.000000, {float(args.voltage_limit_v):.6f}] V"
    )
    cu.upload_voltage(mcv, working_data)


def get_mzi_arm_info(mzi_table, mzi_id, arm):
    mzi_id = int(mzi_id)
    arm = str(arm).lower()
    idx = 0 if arm == "u" else 1
    entry = mzi_table[str(mzi_id)]
    ports = entry.get("ports", [])
    heater_r = entry.get("heater_R", [])
    ppi = entry.get("Ppi", [])
    if len(ports) <= idx or len(heater_r) <= idx:
        raise ValueError(f"MZI {mzi_id} missing port/heater_R for arm {arm}.")
    return {
        "mzi_id": mzi_id,
        "arm": arm,
        "arm_index": idx,
        "arm_name": "upper" if arm == "u" else "lower",
        "port": int(ports[idx]),
        "resistance": float(heater_r[idx]),
        "ppi": float(ppi[idx]) if len(ppi) > idx else np.nan,
    }


def power_to_voltage(power_w, resistance):
    return float(np.sqrt(max(0.0, float(power_w)) * float(resistance)))


def voltage_to_power_w(voltage, resistance):
    return float(float(voltage) ** 2 / float(resistance))


def get_port_voltage(working_data, port):
    return float(working_data.iloc[int(port) - 1, 0])


def get_left_upper_bar_channel(mzi_id, n_value):
    matrix = du.Clements_matrix(int(n_value))
    rows, _ = np.where(matrix == int(mzi_id))
    if rows.size == 0:
        raise ValueError(f"MZI {mzi_id} not found in Clements matrix.")
    return int(rows[0]) + 1


def read_output_power_uW(opm, output_channel):
    values = cu.read_pow(opm)
    idx = int(output_channel) - 1
    if idx < 0 or idx >= len(values):
        raise IndexError(f"Output channel {output_channel} is not available from OPM readout.")
    return float(values[idx]) * 1e6


def fold_power_to_limit(power_w, period_w, power_limit_w):
    power = float(power_w)
    period = float(period_w)
    if period <= 0.0:
        raise ValueError("fold period must be positive.")
    folds = 0
    while power > float(power_limit_w):
        power -= period
        folds += 1
    return max(0.0, power), folds


def parse_probe_map(text, mzi_ids):
    result = {int(mzi_id): "u" for mzi_id in mzi_ids}
    if text is None or str(text).strip() == "":
        return result
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid probe_map item {item!r}; expected like 5:u.")
        mzi_text, arm_text = item.split(":", 1)
        arm = arm_text.strip().lower()
        if arm not in {"u", "d"}:
            raise ValueError(f"Unsupported probe arm {arm!r}; expected u or d.")
        result[int(mzi_text)] = arm
    missing = [int(mzi_id) for mzi_id in mzi_ids if int(mzi_id) not in result]
    if missing:
        raise ValueError(f"probe_map missing MZI ids: {missing}")
    return result


def parse_sigma_bmzi_map(text, mzi_ids):
    if text is None or str(text).strip() == "":
        text = "5:0,6:5,7:6,8:7"
    parsed = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid sigma_bmzi_map item {item!r}; expected like 6:5.")
        mzi_text, bmzi_text = item.split(":", 1)
        parsed[str(int(mzi_text.strip()))] = int(bmzi_text.strip())
    missing = [str(int(mzi_id)) for mzi_id in mzi_ids if str(int(mzi_id)) not in parsed]
    if missing:
        raise ValueError(f"sigma_bmzi_map missing MZI ids: {missing}")
    return {str(int(mzi_id)): int(parsed[str(int(mzi_id))]) for mzi_id in mzi_ids}


def read_power_file(path, heater_order):
    df = pd.read_csv(path)
    if not {"heater", "power_w"}.issubset(df.columns):
        raise ValueError(f"{path} must contain heater,power_w columns.")
    values = {str(row.heater).strip().lower(): float(row.power_w) for row in df.itertuples()}
    return {heater: values[heater.lower()] for heater in heater_order}


def read_opm_repeated(opm, output_channel, args):
    values = []
    reads = max(1, int(args.opm_reads_per_point))
    for idx in range(reads):
        values.append(read_output_power_uW(opm, int(output_channel)))
        if idx + 1 < reads:
            time.sleep(float(args.opm_read_interval_s))
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    median = float(np.median(arr))
    return {
        "opm_raw_uW": json.dumps([float(v) for v in arr]),
        "opm_mean_uW": mean,
        "opm_std_uW": std,
        "opm_median_uW": median,
        "opm_relative_std": float(std / max(abs(mean), 1e-9)),
        "opm_read_count": int(arr.size),
    }


def read_stable_opm(opm, output_channel, args):
    result = None
    warning = ""
    for _ in range(int(args.opm_max_retry_per_point) + 1):
        result = read_opm_repeated(opm, output_channel, args)
        if result["opm_relative_std"] <= float(args.opm_relative_std_threshold):
            warning = ""
            break
        warning = "unstable OPM reading"
    result["warning"] = warning
    return result


def write_port_power(working_data, port, resistance, power_w):
    voltage = float(power_to_voltage(float(max(0.0, power_w)), float(resistance)))
    write_port_voltage(int(port), voltage, working_data)
    return voltage


def get_mzi_state_voltage(entry, state):
    state = str(state).upper()
    values = entry.get("dtheta_Bar", entry.get("dtheta", [])) if state == "B" else entry.get("dtheta_Cross", entry.get("dtheta", []))
    if not values:
        raise ValueError(f"MZI entry missing {state} voltage.")
    return float(values[0 if state == "B" else min(1, len(values) - 1)])


def build_bmzi_state_no_upload(path, input_idx, state, bmzi, working_data, mzi_table, n_value):
    for idx, mzi_value in enumerate(path):
        mzi_id = int(mzi_value)
        entry = mzi_table[str(mzi_id)]
        ports = entry.get("ports", [])
        if not ports:
            raise ValueError(f"MZI {mzi_id} missing ports.")
        if state[idx] == "B":
            write_port_voltage(int(ports[0]), get_mzi_state_voltage(entry, "B"), working_data)
        elif state[idx] == "C":
            write_port_voltage(int(ports[0]), get_mzi_state_voltage(entry, "C"), working_data)
        elif state[idx] == "H":
            half_values = entry.get("half_power", [])
            if not half_values:
                raise ValueError(f"MZI {mzi_id} missing half_power voltage.")
            write_port_voltage(int(ports[0]), float(half_values[0]), working_data)
    for channel in range(1, int(n_value)):
        switch_IN(channel, "OFF", working_data)
    switch_IN(int(input_idx) + 1, "ON", working_data)


def apply_second_column_powers(working_data, mzi_table, mzi_ids, heater_order, powers):
    for heater in heater_order:
        mzi_id = int(heater[:-1])
        arm = heater[-1]
        info = get_mzi_arm_info(mzi_table, mzi_id, arm)
        write_port_power(working_data, info["port"], info["resistance"], powers[heater])


def scan_delta_probe(observed_mzi, probe_arm, save_dir, base_working_data, hardware, mzi_table, args, progress_label=""):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    info = get_mzi_arm_info(mzi_table, observed_mzi, probe_arm)
    input_channel = get_left_upper_bar_channel(int(observed_mzi), int(args.N))
    output_channel = input_channel
    baseline_v = get_port_voltage(base_working_data, info["port"])
    baseline_power = voltage_to_power_w(baseline_v, info["resistance"])
    offsets = np.asarray(parse_csv_list(args.delta_probe_points, float), dtype=float)
    rows = []
    total_points = int(len(offsets))
    for point_idx, offset in enumerate(offsets, start=1):
        scan_data = base_working_data.copy(deep=True)
        target_power = max(0.0, float(baseline_power) + float(offset))
        voltage = write_port_power(scan_data, info["port"], info["resistance"], target_power)
        for channel in range(1, int(args.N)):
            switch_IN(channel, "OFF", scan_data)
        switch_IN(input_channel, "ON", scan_data)
        label = (
            f"{progress_label} delta obs{int(observed_mzi)} "
            f"point {point_idx}/{total_points}, offset={float(offset):.9f} W"
        ).strip()
        upload_voltage_checked(hardware["mcv"], scan_data, args, label)
        time.sleep(float(args.settle_time))
        opm = read_stable_opm(hardware["opm2"], output_channel, args)
        rows.append(
            {
                "target": int(observed_mzi),
                "observed_mzi": int(observed_mzi),
                "probe_arm": probe_arm,
                "arm_name": info["arm_name"],
                "arm_index": info["arm_index"],
                "port": int(info["port"]),
                "probe_axis_power_w": float(offset),
                "target_power_w": float(target_power),
                "measured_power_w": float(target_power),
                "voltage_v": float(voltage),
                "optical_power_uW": float(opm["opm_median_uW"]),
                **opm,
                "scan_type": "delta_probe",
                "output_channel": int(output_channel),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    out_path = save_dir / f"obs{int(observed_mzi)}_probe.txt"
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.12f")
    return out_path


def scan_sigma_inter(observed_mzi, save_dir, base_working_data, hardware, mzi_table, args, sigma_bmzi_map, progress_label=""):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    target = int(observed_mzi)
    scan_data = base_working_data.copy(deep=True)
    path, input_idx, output_idx, state, bmzi = find_Bmzi_path(target, int(args.N))
    build_bmzi_state_no_upload(path, input_idx, state, bmzi, scan_data, mzi_table, int(args.N))
    entry = mzi_table[str(target)]
    ports = [int(v) for v in entry.get("ports", [])[:2]]
    heater_r = [float(v) for v in entry.get("heater_R", [])[:2]]
    ppi = [float(v) for v in entry.get("Ppi", [])[:2]]
    if len(ports) != 2 or len(heater_r) != 2 or len(ppi) != 2:
        raise ValueError(f"MZI {target} requires two ports, heater_R, and Ppi for sigma scan.")
    p_upper_base = voltage_to_power_w(get_port_voltage(scan_data, ports[0]), heater_r[0])
    p_lower_base = voltage_to_power_w(get_port_voltage(scan_data, ports[1]), heater_r[1])
    phase_points = parse_csv_list(args.sigma_phase_points, float)
    rows = []
    total_points = int(len(phase_points))
    for point_idx, dp in enumerate(phase_points, start=1):
        p_upper_unfolded = p_upper_base + float(dp) / np.pi * ppi[0]
        p_lower_unfolded = p_lower_base + float(dp) / np.pi * ppi[1]
        p_upper, upper_folds = fold_power_to_limit(p_upper_unfolded, 2.0 * ppi[0], args.power_limit_w)
        p_lower, lower_folds = fold_power_to_limit(p_lower_unfolded, 2.0 * ppi[1], args.power_limit_w)
        v_upper = write_port_power(scan_data, ports[0], heater_r[0], p_upper)
        v_lower = write_port_power(scan_data, ports[1], heater_r[1], p_lower)
        label = (
            f"{progress_label} sigma obs{target} "
            f"point {point_idx}/{total_points}, dp={float(dp):.9f} rad"
        ).strip()
        upload_voltage_checked(hardware["mcv"], scan_data, args, label)
        time.sleep(float(args.settle_time))
        opm = read_stable_opm(hardware["opm2"], int(output_idx) + 1, args)
        rows.append(
            {
                "target": target,
                "observed_mzi": target,
                "bmzi": int(sigma_bmzi_map.get(str(target), bmzi)),
                "input_channel": int(input_idx) + 1,
                "output_channel": int(output_idx) + 1,
                "path": json.dumps([int(v) for v in path]),
                "state": json.dumps([str(v) for v in state]),
                "dp": float(dp),
                "v_primary": float(v_upper),
                "v_secondary": float(v_lower),
                "p_primary": float(p_upper),
                "p_secondary": float(p_lower),
                "p_primary_unfolded": float(p_upper_unfolded),
                "p_secondary_unfolded": float(p_lower_unfolded),
                "upper_fold_count": int(upper_folds),
                "lower_fold_count": int(lower_folds),
                "optical_power_uW": float(opm["opm_median_uW"]),
                "pow(uW)": float(opm["opm_median_uW"]),
                **opm,
                "scan_type": "sigma_inter",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    out_path = save_dir / f"obs{target}_inter_scan.txt"
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.12f")
    return out_path


def collect_all_scans(save_root, base_working_data, hardware, mzi_table, args, probe_map, sigma_bmzi_map, progress_label=""):
    delta_dir = Path(save_root) / "delta"
    sigma_dir = Path(save_root) / "sigma"
    total_mzis = int(len(args._mzi_ids))
    for mzi_idx, mzi_id in enumerate(args._mzi_ids, start=1):
        print(f"[Get_Jacobi] {progress_label} delta scan {mzi_idx}/{total_mzis}: obs{int(mzi_id)}")
        scan_delta_probe(
            mzi_id,
            probe_map[int(mzi_id)],
            delta_dir,
            base_working_data,
            hardware,
            mzi_table,
            args,
            progress_label=progress_label,
        )
    for mzi_idx, mzi_id in enumerate(args._mzi_ids, start=1):
        print(f"[Get_Jacobi] {progress_label} sigma scan {mzi_idx}/{total_mzis}: obs{int(mzi_id)}")
        scan_sigma_inter(
            mzi_id,
            sigma_dir,
            base_working_data,
            hardware,
            mzi_table,
            args,
            sigma_bmzi_map,
            progress_label=progress_label,
        )


def write_dry_run_placeholders(run_dir, args):
    for group in ["baseline", *[f"perturb_{h}" for h in args._heaters]]:
        for sub in ("delta", "sigma"):
            (run_dir / group / sub).mkdir(parents=True, exist_ok=True)
    for heater in args._heaters:
        metadata_path = run_dir / f"perturb_{heater}" / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "perturbed_heater": heater,
                    "delta_power_w": float(args.delta_power_w),
                    "baseline_power_w": None,
                    "perturbed_power_w": None,
                    "heater_order": args._heaters,
                    "mzi_ids": args._mzi_ids,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dry_run": True,
                },
                f,
                indent=2,
            )


def measure(args):
    args._mzi_ids = parse_csv_list(args.mzi_ids, int)
    args._heaters = parse_csv_list(args.heaters, str)
    probe_map = parse_probe_map(args.probe_map, args._mzi_ids)
    sigma_bmzi_map = parse_sigma_bmzi_map(args.sigma_bmzi_map, args._mzi_ids)
    run_dir = Path(args.out_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mzi_ids": args._mzi_ids,
        "heater_order": args._heaters,
        "probe_map": {str(k): v for k, v in probe_map.items()},
        "sigma_bmzi_map": sigma_bmzi_map,
        "sigma_reference_mode": args.sigma_reference_mode,
        "delta_probe_points": parse_csv_list(args.delta_probe_points, float),
        "sigma_phase_points": parse_csv_list(args.sigma_phase_points, float),
        "opm_reads_per_point": int(args.opm_reads_per_point),
        "power_limit_w": float(args.power_limit_w),
        "voltage_limit_v": float(args.voltage_limit_v),
        "initial_state": args.initial_state,
        "initial_power_file": args.initial_power_file,
        "dry_run": bool(args.dry_run),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))

    if args.dry_run:
        write_dry_run_placeholders(run_dir, args)
        print(f"[Get_Jacobi] dry run: created directory skeleton at {run_dir}")
        return run_dir
    if not parse_bool(args.confirm_hardware):
        raise RuntimeError("Refusing hardware write: set --confirm_hardware true when --dry_run false.")

    mzi_table = load_mzi_table(args.mzi_table)
    baseline_powers = read_power_file(args.initial_power_file, args._heaters)
    mcv = cu.open_ser_connection(args.ser_address)
    opm2 = cu.open_VISA_connection(args.opm2_address)
    if mcv is None:
        raise RuntimeError(f"Failed to open serial port {args.ser_address}.")
    if opm2 is None:
        raise RuntimeError(f"Failed to open OPM2 {args.opm2_address}.")
    hardware = {"mcv": mcv, "opm2": opm2}
    try:
        total_groups = 1 + int(len(args._heaters))
        working_data = cu.generate_working_data()
        apply_second_column_powers(working_data, mzi_table, args._mzi_ids, args._heaters, baseline_powers)
        upload_voltage_checked(mcv, working_data, args, f"group 1/{total_groups} baseline initial state")
        time.sleep(float(args.settle_time))
        collect_all_scans(
            run_dir / "baseline",
            working_data,
            hardware,
            mzi_table,
            args,
            probe_map,
            sigma_bmzi_map,
            progress_label=f"group 1/{total_groups} baseline",
        )
        for group_idx, heater in enumerate(args._heaters, start=2):
            pert_dir = run_dir / f"perturb_{heater}"
            pert_dir.mkdir(parents=True, exist_ok=True)
            pert_powers = dict(baseline_powers)
            baseline_power = float(pert_powers[heater])
            pert_powers[heater] = baseline_power + float(args.delta_power_w)
            pert_working = working_data.copy(deep=True)
            apply_second_column_powers(pert_working, mzi_table, args._mzi_ids, args._heaters, pert_powers)
            validate_voltage_range(pert_working, 0.0, float(args.voltage_limit_v))
            with (pert_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "perturbed_heater": heater,
                        "delta_power_w": float(args.delta_power_w),
                        "baseline_power_w": baseline_power,
                        "perturbed_power_w": float(pert_powers[heater]),
                        "heater_order": args._heaters,
                        "mzi_ids": args._mzi_ids,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                    f,
                    indent=2,
                )
            collect_all_scans(
                pert_dir,
                pert_working,
                hardware,
                mzi_table,
                args,
                probe_map,
                sigma_bmzi_map,
                progress_label=f"group {group_idx}/{total_groups} perturb {heater}",
            )
            upload_voltage_checked(mcv, working_data, args, f"group {group_idx}/{total_groups} restore baseline after {heater}")
            time.sleep(float(args.settle_time))
    finally:
        for handle in (mcv, opm2):
            close = getattr(handle, "close", None)
            if callable(close):
                close()
    print(f"[Get_Jacobi] saved raw measurements to {run_dir}")
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Collect raw scans for second-column Jacobian measurement.")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("measure")
    p.add_argument("--mzi_ids", default="5,6,7,8")
    p.add_argument("--heaters", default="5u,5d,6u,6d,7u,7d,8u,8d")
    p.add_argument("--out_root", default="jacobian_measurements_new")
    p.add_argument("--mzi_table", default="Scandata/MZI_table.json")
    p.add_argument("--initial_power_file", default="current_power_second_column.csv")
    p.add_argument("--probe_map", default="5:u,6:u,7:u,8:u")
    p.add_argument("--sigma_bmzi_map", default="5:0,6:5,7:6,8:7")
    p.add_argument("--sigma_reference_mode", default="chained_bmzi", choices=["chained_bmzi", "direct_reference"])
    p.add_argument("--delta_power_w", type=float, default=0.001)
    p.add_argument("--delta_probe_points", default="-0.001,-0.0005,0,0.0005,0.001")
    p.add_argument("--sigma_phase_points", default="0,1.570796327,3.141592654,4.71238898,6.283185307")
    p.add_argument("--opm_reads_per_point", type=int, default=3)
    p.add_argument("--opm_read_interval_s", type=float, default=0.1)
    p.add_argument("--opm_relative_std_threshold", type=float, default=0.05)
    p.add_argument("--opm_max_retry_per_point", type=int, default=2)
    p.add_argument("--power_limit_w", type=float, default=0.055)
    p.add_argument("--voltage_limit_v", type=float, default=6.0)
    p.add_argument("--settle_time", type=float, default=2.0)
    p.add_argument("--initial_state", default="voltage_pair")
    p.add_argument("--N", type=int, default=9)
    p.add_argument("--dry_run", type=parse_bool, default=True)
    p.add_argument("--confirm_hardware", type=parse_bool, default=False)
    p.add_argument("--ser_address", default=DEFAULT_SER_ADDRESS)
    p.add_argument("--opm2_address", default=DEFAULT_OPM2_ADDRESS)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "measure":
        measure(args)


if __name__ == "__main__":
    main()
