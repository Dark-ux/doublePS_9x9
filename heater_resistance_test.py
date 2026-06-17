import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.communication as cu


DEFAULT_IN_MZI = "IN_MZI.txt"
DEFAULT_OUT_ROOT = "results/HeaterResistanceTest"
DEFAULT_SER_ADDRESS = "COM3"
RESISTANCE_COLUMN = "HEATER_R_OHM"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_switch_mzis(path):
    table = pd.read_csv(path)
    required = {"MZI", "PORT", "ON", "OFF"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    table["MZI"] = table["MZI"].astype(int)
    table["PORT"] = table["PORT"].astype(int)
    if len(table) != 8:
        print(f"[heater_resistance_test] warning: expected 8 switch MZIs, found {len(table)} rows.")
    return table


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
            # communication.read_current returns mA.
            return float(currents[idx]), attempt, ""
        time.sleep(float(delay_s))
    return None, attempts, f"target_current_none; all_channel_none_count={last_none_count}"


def build_voltage_points(start_v, stop_v, step_v):
    if step_v <= 0:
        raise ValueError("--step-v must be positive.")
    if stop_v < start_v:
        raise ValueError("--stop-v must be >= --start-v.")
    count = int(np.floor((float(stop_v) - float(start_v)) / float(step_v) + 1e-12)) + 1
    values = [round(float(start_v) + idx * float(step_v), 6) for idx in range(count)]
    if values[-1] < float(stop_v) - 1e-12:
        values.append(round(float(stop_v), 6))
    return np.asarray(values, dtype=float)


def fit_resistance(voltage_v, current_mA):
    voltage = np.asarray(voltage_v, dtype=float)
    current_a = np.asarray(current_mA, dtype=float) * 1e-3
    valid = np.isfinite(voltage) & np.isfinite(current_a)
    if np.count_nonzero(valid) < 2:
        return {
            "resistance_ohm": np.nan,
            "intercept_v": np.nan,
            "r2": np.nan,
            "point_count": int(np.count_nonzero(valid)),
        }

    current_valid = current_a[valid]
    voltage_valid = voltage[valid]
    resistance, intercept = np.polyfit(current_valid, voltage_valid, 1)
    pred = resistance * current_valid + intercept
    ss_res = float(np.sum((voltage_valid - pred) ** 2))
    ss_tot = float(np.sum((voltage_valid - np.mean(voltage_valid)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "resistance_ohm": float(resistance),
        "intercept_v": float(intercept),
        "r2": float(r2),
        "point_count": int(np.count_nonzero(valid)),
    }


def scan_switch_mzi(
    *,
    mcv,
    mzi_id,
    port,
    voltage_points,
    settle_s,
    current_retries,
    current_retry_delay_s,
    voltage_limit_v,
    dry_run,
):
    working_data = cu.generate_working_data()
    zero_all(working_data)
    rows = []

    for point_index, voltage_v in enumerate(voltage_points):
        zero_all(working_data)
        set_port_voltage(working_data, port, voltage_v)
        if dry_run:
            current_mA = np.nan
            attempts = 0
            warning = "dry_run"
        else:
            upload_checked(mcv, working_data, voltage_limit_v, f"MZI {mzi_id} port {port} point {point_index}")
            time.sleep(float(settle_s))
            current_mA, attempts, warning = read_current_port_retry(
                mcv,
                port,
                retries=current_retries,
                delay_s=current_retry_delay_s,
            )

        rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "MZI": int(mzi_id),
                "PORT": int(port),
                "point_index": int(point_index),
                "voltage_v": float(voltage_v),
                "current_mA": "" if current_mA is None else current_mA,
                "current_read_attempts": int(attempts),
                "warning": warning,
            }
        )

    return pd.DataFrame(rows)


def plot_fit(raw_df, fit_row, out_path):
    voltage = pd.to_numeric(raw_df["voltage_v"], errors="coerce").to_numpy(dtype=float)
    current_mA = pd.to_numeric(raw_df["current_mA"], errors="coerce").to_numpy(dtype=float)
    current_a = current_mA * 1e-3
    valid = np.isfinite(voltage) & np.isfinite(current_a)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(current_a[valid], voltage[valid], "o", label="measured")
    if np.count_nonzero(valid) >= 2 and np.isfinite(fit_row["resistance_ohm"]):
        x = np.linspace(float(np.min(current_a[valid])), float(np.max(current_a[valid])), 100)
        y = fit_row["resistance_ohm"] * x + fit_row["intercept_v"]
        ax.plot(x, y, "-", label=f"fit R={fit_row['resistance_ohm']:.3f} ohm")
    ax.set_title(f"Switch MZI {int(fit_row['MZI'])} port {int(fit_row['PORT'])}")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def update_in_mzi_resistance_column(in_mzi_path, resistance_by_mzi):
    table = load_switch_mzis(in_mzi_path)
    if RESISTANCE_COLUMN in table.columns:
        existing = pd.to_numeric(table[RESISTANCE_COLUMN], errors="coerce")
    else:
        existing = pd.Series([np.nan] * len(table), index=table.index)
    table[RESISTANCE_COLUMN] = [
        resistance_by_mzi.get(int(mzi), existing.iloc[idx])
        for idx, mzi in enumerate(table["MZI"])
    ]
    table.to_csv(in_mzi_path, index=False)


def parse_mzi_filter(value):
    if value is None or str(value).strip() == "":
        return None
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def main():
    parser = argparse.ArgumentParser(description="Measure heater resistance for switch MZIs listed in IN_MZI.txt.")
    parser.add_argument("--in-mzi", default=DEFAULT_IN_MZI)
    parser.add_argument("--mzi-filter", default="", help="Comma-separated MZI ids to measure, e.g. 1 or 1,3,8.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--ser-address", default=DEFAULT_SER_ADDRESS)
    parser.add_argument("--start-v", type=float, default=0.5)
    parser.add_argument("--stop-v", type=float, default=2.5)
    parser.add_argument("--step-v", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=0.2)
    parser.add_argument("--current-retries", type=int, default=10)
    parser.add_argument("--current-retry-delay-s", type=float, default=0.2)
    parser.add_argument("--voltage-limit-v", type=float, default=5.5)
    parser.add_argument("--dry-run", type=parse_bool, default=True)
    parser.add_argument("--confirm-hardware", type=parse_bool, default=False)
    parser.add_argument("--update-in-mzi", type=parse_bool, default=True)
    parser.add_argument("--restore-zero", type=parse_bool, default=True)
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_hardware:
        raise RuntimeError("Refusing hardware write: set --confirm-hardware true with --dry-run false.")

    switch_table = load_switch_mzis(args.in_mzi)
    mzi_filter = parse_mzi_filter(args.mzi_filter)
    if mzi_filter is not None:
        known = set(int(x) for x in switch_table["MZI"])
        unknown = sorted(mzi_filter.difference(known))
        if unknown:
            raise ValueError(f"--mzi-filter contains MZI ids not in {args.in_mzi}: {unknown}")
        switch_table = switch_table[switch_table["MZI"].isin(mzi_filter)].copy()
        if switch_table.empty:
            raise ValueError("--mzi-filter selected no rows.")
    voltage_points = build_voltage_points(args.start_v, args.stop_v, args.step_v)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"switch_mzi_run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    config = {
        **vars(args),
        "selected_mzis": sorted(mzi_filter) if mzi_filter is not None else "all",
        "voltage_points": [float(v) for v in voltage_points],
        "switch_mzis": switch_table.to_dict(orient="records"),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    mcv = None
    raw_frames = []
    summary_rows = []
    try:
        if not args.dry_run:
            mcv = cu.open_ser_connection(args.ser_address)
            if mcv is None:
                raise RuntimeError(f"Failed to open serial port {args.ser_address}.")

        for _, row in switch_table.iterrows():
            mzi_id = int(row["MZI"])
            port = int(row["PORT"])
            print(f"[heater_resistance_test] scan switch MZI {mzi_id}, port {port}")
            raw_df = scan_switch_mzi(
                mcv=mcv,
                mzi_id=mzi_id,
                port=port,
                voltage_points=voltage_points,
                settle_s=args.settle_s,
                current_retries=args.current_retries,
                current_retry_delay_s=args.current_retry_delay_s,
                voltage_limit_v=args.voltage_limit_v,
                dry_run=args.dry_run,
            )
            raw_frames.append(raw_df)
            raw_df.to_csv(out_dir / f"switch_mzi_{mzi_id}_port_{port}.csv", index=False)

            fit = fit_resistance(raw_df["voltage_v"], pd.to_numeric(raw_df["current_mA"], errors="coerce"))
            summary_row = {
                "MZI": mzi_id,
                "PORT": port,
                "ON": float(row["ON"]),
                "OFF": float(row["OFF"]),
                **fit,
            }
            summary_rows.append(summary_row)
            plot_fit(raw_df, summary_row, out_dir / f"switch_mzi_{mzi_id}_fit.png")
            print(
                f"[heater_resistance_test] MZI {mzi_id}, port {port}, "
                f"R={summary_row['resistance_ohm']:.6g} ohm, r2={summary_row['r2']:.6g}"
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

    all_raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    all_raw.to_csv(out_dir / "all_measurements.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "switch_mzi_resistance_summary.csv", index=False)

    resistance_by_mzi = {
        int(row["MZI"]): float(row["resistance_ohm"])
        for row in summary_rows
        if np.isfinite(float(row["resistance_ohm"]))
    }
    if args.update_in_mzi and not args.dry_run:
        update_in_mzi_resistance_column(args.in_mzi, resistance_by_mzi)
        print(f"[heater_resistance_test] updated {args.in_mzi} column {RESISTANCE_COLUMN}")
    elif args.update_in_mzi:
        print("[heater_resistance_test] dry-run: IN_MZI.txt not updated.")

    print(f"[heater_resistance_test] saved results to {out_dir}")


if __name__ == "__main__":
    main()
