import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.communication as cu


DEFAULT_SER_ADDRESS = "COM3"
DEFAULT_MZI_TABLE = "Scandata/MZI_table.json"
DEFAULT_OUT_ROOT = "results/HeaterResistanceTest"
DEFAULT_MZI_IDS = [5, 6, 7, 8]
DEFAULT_TARGET_HEATER = "6d"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_heater_label(label):
    text = str(label).strip().lower()
    if len(text) < 2 or text[-1] not in {"u", "d"}:
        raise ValueError(f"Invalid heater label {label!r}; expected like 6d.")
    return int(text[:-1]), text[-1]


def load_second_column_heaters(mzi_table_path, mzi_ids=DEFAULT_MZI_IDS):
    with open(mzi_table_path, "r", encoding="utf-8") as f:
        table = json.load(f)
    heaters = {}
    for mzi_id in mzi_ids:
        entry = table[str(int(mzi_id))]
        ports = entry["ports"]
        resistances = entry.get("heater_R", [None] * len(ports))
        for arm_idx, arm in enumerate(("u", "d")):
            label = f"{int(mzi_id)}{arm}"
            heaters[label] = {
                "mzi_id": int(mzi_id),
                "arm": arm,
                "port": int(ports[arm_idx]),
                "resistance_ohm": None if resistances[arm_idx] is None else float(resistances[arm_idx]),
            }
    return heaters


def validate_voltage_range(working_data, voltage_limit_v):
    voltages = pd.to_numeric(working_data.iloc[:, 0], errors="coerce")
    if voltages.isna().any():
        raise ValueError("working_data contains non-numeric voltage values.")
    bad = (voltages < 0.0) | (voltages > float(voltage_limit_v))
    if bad.any():
        bad_ports = (bad[bad].index + 1).tolist()
        raise ValueError(f"Voltage out of range [0, {voltage_limit_v}] at ports: {bad_ports}")


def upload_checked(mcv, working_data, voltage_limit_v, label):
    validate_voltage_range(working_data, voltage_limit_v)
    nonzero = int(np.count_nonzero(np.asarray(working_data.iloc[:, 0], dtype=float)))
    print(f"[heater_resistance_test] upload {label}: nonzero_ports={nonzero}")
    cu.upload_voltage(mcv, working_data)


def zero_all(working_data):
    working_data.iloc[:, 0] = 0.0


def set_port_voltage(working_data, port, voltage):
    cu.write_port_voltage(int(port), float(voltage), working_data)


def read_current_port_retry(mcv, port, retries=10, delay_s=0.2):
    attempts = max(1, int(retries))
    last_none_count = None
    for attempt in range(1, attempts + 1):
        try:
            currents = cu.read_current(mcv)
        except Exception as exc:
            if attempt == attempts:
                return None, attempt, f"read_current_failed: {exc}"
            time.sleep(float(delay_s))
            continue
        last_none_count = sum(value is None for value in currents)
        idx = int(port) - 1
        if 0 <= idx < len(currents) and currents[idx] is not None:
            return float(currents[idx]), attempt, ""
        time.sleep(float(delay_s))
    return None, attempts, f"target_current_none; all_channel_none_count={last_none_count}"


def build_voltage_points(start_v, stop_v, step_v):
    count = int(round((float(stop_v) - float(start_v)) / float(step_v))) + 1
    return np.round(np.linspace(float(start_v), float(stop_v), count), 10)


def fit_line(voltage_v, current_a):
    x = np.asarray(voltage_v, dtype=float)
    y = np.asarray(current_a, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return {"slope_a_per_v": np.nan, "intercept_a": np.nan, "r2": np.nan, "point_count": int(np.count_nonzero(valid))}
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    pred = slope * x[valid] + intercept
    ss_res = float(np.sum((y[valid] - pred) ** 2))
    ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    resistance = 1.0 / slope if abs(slope) > 1e-15 else np.nan
    return {
        "slope_a_per_v": float(slope),
        "intercept_a": float(intercept),
        "r2": float(r2),
        "resistance_ohm_from_slope": float(resistance),
        "point_count": int(np.count_nonzero(valid)),
    }


def scan_target_heater(
    *,
    mcv,
    target_label,
    target_port,
    other_heaters,
    other_voltage_v,
    voltage_points,
    settle_s,
    current_retries,
    current_retry_delay_s,
    voltage_limit_v,
    scenario_name,
    out_dir,
    dry_run,
):
    working_data = cu.generate_working_data()
    zero_all(working_data)
    for info in other_heaters:
        set_port_voltage(working_data, info["port"], other_voltage_v)

    rows = []
    for idx, voltage_v in enumerate(voltage_points):
        set_port_voltage(working_data, target_port, voltage_v)
        if dry_run:
            current_a = np.nan
            attempts = 0
            warning = "dry_run"
        else:
            upload_checked(mcv, working_data, voltage_limit_v, f"{scenario_name} point {idx}: {target_label}={voltage_v:.3f} V")
            time.sleep(float(settle_s))
            current_a, attempts, warning = read_current_port_retry(
                mcv,
                target_port,
                retries=current_retries,
                delay_s=current_retry_delay_s,
            )
        rows.append(
            {
                "scenario": scenario_name,
                "point_index": int(idx),
                "target_heater": target_label,
                "target_port": int(target_port),
                "target_voltage_v": float(voltage_v),
                "target_current_a": "" if current_a is None else current_a,
                "other_second_column_voltage_v": float(other_voltage_v),
                "current_read_attempts": int(attempts),
                "warning": warning,
                "timestamp": datetime.now().isoformat(timespec="microseconds"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{scenario_name}.csv", index=False)
    return df


def plot_comparison(df_zero, df_bias, fit_zero, fit_bias, target_label, out_path):
    x_zero = pd.to_numeric(df_zero["target_voltage_v"], errors="coerce").to_numpy(dtype=float)
    y_zero = pd.to_numeric(df_zero["target_current_a"], errors="coerce").to_numpy(dtype=float)
    x_bias = pd.to_numeric(df_bias["target_voltage_v"], errors="coerce").to_numpy(dtype=float)
    y_bias = pd.to_numeric(df_bias["target_current_a"], errors="coerce").to_numpy(dtype=float)

    slope_delta = fit_bias["slope_a_per_v"] - fit_zero["slope_a_per_v"]
    intercept_delta = fit_bias["intercept_a"] - fit_zero["intercept_a"]
    slope_rel = slope_delta / fit_zero["slope_a_per_v"] if abs(fit_zero["slope_a_per_v"]) > 1e-15 else np.nan
    intercept_rel = intercept_delta / fit_zero["intercept_a"] if abs(fit_zero["intercept_a"]) > 1e-15 else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    label_zero = (
        "others 0 V: "
        f"slope={fit_zero['slope_a_per_v']:.6g} A/V, "
        f"b={fit_zero['intercept_a']:.6g} A"
    )
    label_bias = (
        "others 3 V: "
        f"slope={fit_bias['slope_a_per_v']:.6g} A/V, "
        f"b={fit_bias['intercept_a']:.6g} A"
    )
    ax.plot(x_zero, y_zero, "o-", label=label_zero)
    ax.plot(x_bias, y_bias, "s-", label=label_bias)
    text = (
        "bias - zero\n"
        f"Delta slope = {slope_delta:.6g} A/V ({slope_rel:.3%})\n"
        f"Delta intercept = {intercept_delta:.6g} A ({intercept_rel:.3%})"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.85})
    ax.set_title(f"{target_label} voltage-current curve")
    ax.set_xlabel("6d voltage (V)")
    ax.set_ylabel("6d current reading (A)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Second-column heater resistance comparison scan.")
    parser.add_argument("--target_heater", default=DEFAULT_TARGET_HEATER)
    parser.add_argument("--mzi_table", default=DEFAULT_MZI_TABLE)
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--ser_address", default=DEFAULT_SER_ADDRESS)
    parser.add_argument("--start_v", type=float, default=0.0)
    parser.add_argument("--stop_v", type=float, default=5.5)
    parser.add_argument("--step_v", type=float, default=0.1)
    parser.add_argument("--other_bias_v", type=float, default=3.0)
    parser.add_argument("--settle_s", type=float, default=0.2)
    parser.add_argument("--current_retries", type=int, default=10)
    parser.add_argument("--current_retry_delay_s", type=float, default=0.2)
    parser.add_argument("--voltage_limit_v", type=float, default=5.5)
    parser.add_argument("--dry_run", type=parse_bool, default=True)
    parser.add_argument("--confirm_hardware", type=parse_bool, default=False)
    parser.add_argument("--restore_zero", type=parse_bool, default=True)
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_hardware:
        raise RuntimeError("Refusing hardware write: set --confirm_hardware true with --dry_run false.")

    heaters = load_second_column_heaters(args.mzi_table)
    target_label = str(args.target_heater).strip().lower()
    if target_label not in heaters:
        raise ValueError(f"{target_label} not found in second-column heaters: {sorted(heaters)}")
    target_info = heaters[target_label]
    other_heaters = [info for label, info in heaters.items() if label != target_label]
    voltage_points = build_voltage_points(args.start_v, args.stop_v, args.step_v)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"run_{timestamp}_{target_label}"
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "target_port": target_info["port"],
        "second_column_heaters": heaters,
        "voltage_points": [float(v) for v in voltage_points],
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    mcv = None
    try:
        if not args.dry_run:
            mcv = cu.open_ser_connection(args.ser_address)
            if mcv is None:
                raise RuntimeError(f"Failed to open serial port {args.ser_address}.")

        df_zero = scan_target_heater(
            mcv=mcv,
            target_label=target_label,
            target_port=target_info["port"],
            other_heaters=other_heaters,
            other_voltage_v=0.0,
            voltage_points=voltage_points,
            settle_s=args.settle_s,
            current_retries=args.current_retries,
            current_retry_delay_s=args.current_retry_delay_s,
            voltage_limit_v=args.voltage_limit_v,
            scenario_name="others_0v",
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
        df_bias = scan_target_heater(
            mcv=mcv,
            target_label=target_label,
            target_port=target_info["port"],
            other_heaters=other_heaters,
            other_voltage_v=args.other_bias_v,
            voltage_points=voltage_points,
            settle_s=args.settle_s,
            current_retries=args.current_retries,
            current_retry_delay_s=args.current_retry_delay_s,
            voltage_limit_v=args.voltage_limit_v,
            scenario_name="others_3v",
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
    finally:
        if mcv is not None and args.restore_zero:
            try:
                working_data = cu.generate_working_data()
                zero_all(working_data)
                upload_checked(mcv, working_data, args.voltage_limit_v, "restore all zero")
            except Exception as exc:
                print(f"[heater_resistance_test] restore zero failed: {exc}")
        if mcv is not None:
            close = getattr(mcv, "close", None)
            if callable(close):
                close()

    fit_zero = fit_line(df_zero["target_voltage_v"], pd.to_numeric(df_zero["target_current_a"], errors="coerce"))
    fit_bias = fit_line(df_bias["target_voltage_v"], pd.to_numeric(df_bias["target_current_a"], errors="coerce"))
    fit_summary = {
        "target_heater": target_label,
        "target_port": target_info["port"],
        "target_reference_resistance_ohm": target_info.get("resistance_ohm"),
        "others_0v": fit_zero,
        "others_3v": fit_bias,
        "bias_minus_zero": {
            "slope_delta_a_per_v": fit_bias["slope_a_per_v"] - fit_zero["slope_a_per_v"],
            "intercept_delta_a": fit_bias["intercept_a"] - fit_zero["intercept_a"],
        },
    }
    with (out_dir / "fit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fit_summary, f, indent=2)
    plot_comparison(df_zero, df_bias, fit_zero, fit_bias, target_label, out_dir / "voltage_current_comparison.png")
    print(f"[heater_resistance_test] saved results to {out_dir}")


if __name__ == "__main__":
    main()
