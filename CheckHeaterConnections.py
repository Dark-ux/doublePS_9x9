"""Test recorded heaters at 1 V, with at most ten voltage uploads."""

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import utils.communication as cu
from MeasureHeaterCurrents import build_heater_metadata


TEST_VOLTAGE_V = 1.0
MAX_UPLOADS = 10
MIN_CURRENT_MA = 0.01


def load_heaters(mzi_table, switch_table):
    rows = build_heater_metadata(mzi_table, switch_table).to_dict(orient="records")
    seen = set()
    for row in rows:
        port = row["port"]
        resistance = row["configured_resistance_ohm"]
        if port in seen or not 1 <= port <= cu.CHANNEL_NUM:
            raise ValueError(f"Duplicate or invalid heater port: {port}")
        if not math.isfinite(resistance) or resistance <= 0:
            raise ValueError(f"Missing or invalid reference resistance for port {port}")
        seen.add(port)
        row.pop("focus_3_4_path", None)
    if not rows:
        raise ValueError("No heaters configured")
    return rows


def upload_once(connection, voltages, command_delay=0.0):
    """One physical upload; do not nest communication.upload_voltage retries."""
    if len(voltages) != cu.CHANNEL_NUM:
        raise ValueError("Expected exactly 128 voltage settings")
    if any(not math.isfinite(v) or not 0 <= v <= TEST_VOLTAGE_V for v in voltages):
        raise ValueError("This test only permits voltages between 0 and 1 V")
    commands = cu.generate_voltage_commands([(v / 24.0 + 0.5) * 65535 for v in voltages])
    for command in commands:
        payload = bytes.fromhex(command)
        if connection.write(payload) != len(payload):
            raise IOError("Incomplete serial voltage write")
        if command_delay:
            time.sleep(command_delay)
    connection.flush()


def read_current_verified(connection):
    """Use the existing protocol, rejecting truncated or wrong-bank replies."""
    currents = [None] * cu.CHANNEL_NUM
    for bank in range(4):
        command = bytes.fromhex(f"EB90B{bank}5000000000")
        # Discard stale replies before requesting this bank.
        connection.reset_input_buffer()
        if connection.write(command) != len(command):
            raise IOError("Incomplete serial current request")
        time.sleep(0.1)
        block = connection.read(84)
        if len(block) != 84 or block[:3] != command[:3]:
            continue
        data_hex = block.hex()
        for index in range(32):
            sign = int(data_hex[8 + 5 * index], 16)
            raw = int(data_hex[9 + 5 * index:13 + 5 * index], 16)
            voltage_part = raw / 65535.0 * 20.48 - 10.24
            resistance_factor = 5.025 if sign == 0 else 2005.0
            currents[bank * 32 + index] = voltage_part / 19.0 / resistance_factor * 1000.0
    return currents


def read_samples(connection, sample_count, sample_delay):
    samples = []
    for sample_index in range(sample_count):
        connection.reset_input_buffer()
        sample = read_current_verified(connection)
        if len(sample) != cu.CHANNEL_NUM:
            raise IOError("Current response does not contain 128 ports")
        samples.append(sample)
        if sample_index + 1 < sample_count:
            time.sleep(sample_delay)
    return samples


def diagnose(heaters, samples, attempt, tolerance):
    results = []
    for heater in heaters:
        raw_values = [sample[heater["port"] - 1] for sample in samples]
        values = [float(v) for v in raw_values if v is not None and math.isfinite(float(v))]
        current = statistics.median(values) if values else None
        reference_r = heater["configured_resistance_ohm"]
        expected = TEST_VOLTAGE_V / reference_r * 1000.0
        measured_r = TEST_VOLTAGE_V / (current * 1e-3) if current is not None and current > MIN_CURRENT_MA else None
        current_error = (current / expected - 1.0) if current is not None else None
        resistance_error = (measured_r / reference_r - 1.0) if measured_r is not None else None
        spread = max(abs(v - current) for v in values) / expected if values else None
        if len(values) != len(samples):
            status = "NO_READING"
        elif min(values) <= MIN_CURRENT_MA:
            status = "NO_CURRENT"
        elif spread > tolerance:
            status = "UNSTABLE"
        elif abs(current_error) > tolerance or abs(resistance_error) > tolerance:
            status = "MISMATCH"
        else:
            status = "OK"
        results.append({
            **heater,
            "voltage_setpoint_v": TEST_VOLTAGE_V,
            "expected_current_mA": expected,
            "measured_current_mA": current,
            "measured_resistance_ohm": measured_r,
            "relative_current_error": current_error,
            "relative_resistance_error": resistance_error,
            "relative_current_spread": spread,
            "valid_samples": len(values),
            "sample_count": len(samples),
            "upload_attempt": attempt,
            "status": status,
        })
    return results


def save_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_zero(connection, out_dir, settle_time=0.5):
    """Verify cleanup separately from the ten permitted 1 V test uploads."""
    raw_rows = []
    state = {"zero_voltage_command_sent": False, "zero_current_check_passed": False}
    try:
        for attempt in range(1, MAX_UPLOADS + 1):
            # Allow the controller to process each channel command during cleanup.
            upload_once(connection, [0.0] * cu.CHANNEL_NUM, command_delay=0.003)
            state["zero_voltage_command_sent"] = True
            state["zero_upload_attempts"] = attempt
            time.sleep(settle_time)
            reading = read_samples(connection, 1, 0)[0]
            rows = [{"port": i + 1, "voltage_setpoint_v": 0.0, "current_mA": v} for i, v in enumerate(reading)]
            raw_rows.extend({"zero_upload_attempt": attempt, **row} for row in rows)
            save_rows(out_dir / "zero_current_check.csv", rows)
            save_rows(out_dir / "zero_current_attempts.csv", raw_rows)
            flagged = [i + 1 for i, v in enumerate(reading) if v is None or not math.isfinite(float(v)) or abs(v) > MIN_CURRENT_MA]
            state["zero_check_flagged_ports"] = flagged
            state["zero_current_check_passed"] = not flagged
            print(f"Zero cleanup {attempt}/{MAX_UPLOADS}: flagged ports={flagged}", flush=True)
            if not flagged:
                break
    except Exception as exc:
        state["zero_error"] = f"{type(exc).__name__}: {exc}"
    return state


def run_test(connection, heaters, out_dir, samples=3, sample_delay=0.2, settle_time=0.5, tolerance=0.10):
    voltages = [0.0] * cu.CHANNEL_NUM
    for heater in heaters:
        voltages[heater["port"] - 1] = TEST_VOLTAGE_V
    save_rows(out_dir / "test_voltage.csv", [
        {"port": i + 1, "voltage": v} for i, v in enumerate(voltages)
    ])
    history, raw_rows, results = [], [], []
    summary = {"completed": False, "test_uploads": 0, "zero_voltage_command_sent": False}
    try:
        for attempt in range(1, MAX_UPLOADS + 1):
            summary["test_uploads"] = attempt
            upload_once(connection, voltages)
            time.sleep(settle_time)
            readings = read_samples(connection, samples, sample_delay)
            results = diagnose(heaters, readings, attempt, tolerance)
            for sample_index, reading in enumerate(readings, start=1):
                raw_rows.extend({
                    "upload_attempt": attempt, "sample": sample_index, "port": port,
                    "voltage_setpoint_v": voltages[port - 1], "current_mA": value,
                } for port, value in enumerate(reading, start=1))
            history.extend(results)
            save_rows(out_dir / "current_samples.csv", raw_rows)
            save_rows(out_dir / "attempt_diagnostics.csv", history)
            save_rows(out_dir / "final_heater_diagnostics.csv", results)
            missing = [r["port"] for r in results if r["status"] in {"NO_CURRENT", "NO_READING"}]
            print(f"Upload {attempt}/{MAX_UPLOADS}: OK={sum(r['status'] == 'OK' for r in results)}/{len(results)}; no current/reading ports={missing}", flush=True)
            if not missing:
                break
        summary.update({
            "completed": True,
            "heater_count": len(heaters),
            "normal_count": sum(r["status"] == "OK" for r in results),
            "status_counts": {status: sum(r["status"] == status for r in results) for status in sorted({r["status"] for r in results})},
            "abnormal_ports": [r["port"] for r in results if r["status"] != "OK"],
            "recovered_ports": sorted({r["port"] for r in history if r["status"] in {"NO_CURRENT", "NO_READING"}} & {r["port"] for r in results if r["status"] == "OK"}),
            "voltage_setpoint_v": TEST_VOLTAGE_V,
            "voltage_measurement": "Commanded voltage only; device interface returns current, not measured voltage",
            "relative_tolerance": tolerance,
        })
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary.update(restore_zero(connection, out_dir, settle_time))
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ser-address", default="COM3")
    parser.add_argument("--mzi-table", default="Scandata/MZI_table.json")
    parser.add_argument("--switch-table", default="IN_MZI.txt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--confirm-hardware", action="store_true")
    args = parser.parse_args()
    if not args.confirm_hardware:
        parser.error("Use --confirm-hardware to perform the 1 V hardware test")
    if args.samples < 1 or not 0 < args.relative_tolerance < 1:
        parser.error("samples must be positive and relative-tolerance must be between 0 and 1")
    heaters = load_heaters(args.mzi_table, args.switch_table)
    out_dir = Path(args.out_dir) if args.out_dir else Path("results/HeaterConnectionTest") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "started_at": datetime.now().isoformat(timespec="seconds"), "test_voltage_v": TEST_VOLTAGE_V, "max_test_uploads": MAX_UPLOADS, "minimum_current_mA": MIN_CURRENT_MA, "heaters": heaters}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    connection = cu.open_ser_connection(args.ser_address)
    if connection is None:
        raise RuntimeError(f"Could not open {args.ser_address}; no test performed")
    try:
        summary = run_test(connection, heaters, out_dir, samples=args.samples, tolerance=args.relative_tolerance)
    finally:
        connection.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Results: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
