"""Bounded 0/1 V communication check; saves raw replies and ends at zero."""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import serial
import CheckHeaterConnections as check

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--command-delay", type=float, default=0.02)
parser.add_argument("--cycles", type=int, default=3)
options = parser.parse_args()
if not 0.02 <= options.command_delay <= 0.1 or not 1 <= options.cycles <= 3:
    parser.error("Require command delay 0.02..0.1 s and cycles 1..3")
DELAY = options.command_delay
OUT = Path(__file__).resolve().parent / f"paced_{DELAY * 1000:g}ms_check"
OUT.mkdir(exist_ok=False)
frames = []
raw_currents = []
diagnostics = []
phases = []


def read_three(connection, phase):
    samples = []
    for sample_index in range(3):
        currents = [None] * 128
        for bank in range(4):
            command = bytes.fromhex(f"EB90B{bank}5000000000")
            stale = connection.read(connection.in_waiting)
            started = time.perf_counter()
            if connection.write(command) != len(command):
                raise IOError("Incomplete current request")
            connection.flush()
            time.sleep(0.1)
            block = connection.read(84)
            valid = len(block) == 84 and block[:3] == command[:3]
            frames.append({
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "phase": phase, "sample": sample_index + 1, "bank": bank,
                "request_hex": command.hex(), "reply_hex": block.hex(),
                "reply_length": len(block), "valid": valid,
                "discarded_stale_hex": stale.hex(),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            })
            if valid:
                data_hex = block.hex()
                for index in range(32):
                    flag = int(data_hex[8 + 5 * index], 16)
                    raw = int(data_hex[9 + 5 * index:13 + 5 * index], 16)
                    adc_v = raw / 65535.0 * 20.48 - 10.24
                    currents[bank * 32 + index] = adc_v / 19.0 / (5.025 if flag == 0 else 2005.0) * 1000
        samples.append(currents)
        raw_currents.extend({"phase": phase, "sample": sample_index + 1,
                             "port": i + 1, "current_mA": v}
                            for i, v in enumerate(currents))
        time.sleep(0.15)
    return samples


def zero_phase(connection, label, delay=DELAY):
    check.upload_once(connection, [0.0] * 128, command_delay=delay)
    time.sleep(0.8)
    samples = read_three(connection, label)
    flagged = sorted({i + 1 for sample in samples for i, v in enumerate(sample)
                      if v is None or not math.isfinite(v) or abs(v) > 0.01})
    valid = [abs(v) for sample in samples for v in sample
             if v is not None and math.isfinite(v)]
    result = {"phase": label, "command_delay_s": delay,
              "all_zero": not flagged, "flagged_ports": flagged,
              "max_abs_current_mA": max(valid, default=None)}
    phases.append(result)
    print(json.dumps(result), flush=True)
    return not flagged


heaters = check.load_heaters(ROOT / "Scandata/MZI_table.json", ROOT / "IN_MZI.txt")
voltages = [0.0] * 128
for heater in heaters:
    voltages[heater["port"] - 1] = 1.0
report = {"started_at": datetime.now().isoformat(timespec="seconds"),
          "test_voltage_v": 1.0, "command_delay_s": DELAY,
          "samples_per_state": 3, "cycles_requested": options.cycles,
          "voltage_measurement": "Setpoint only; measured feedback is current"}
connection = None
try:
    connection = serial.Serial("COM3", 115200, timeout=0.8, write_timeout=3)
    time.sleep(0.2)
    connection.reset_input_buffer()
    for cycle in range(1, options.cycles + 1):
        check.upload_once(connection, voltages, command_delay=DELAY)
        time.sleep(0.8)
        readings = read_three(connection, f"cycle{cycle}_1v")
        rows = check.diagnose(heaters, readings, cycle, 0.10)
        diagnostics.extend({"cycle": cycle, **row} for row in rows)
        result = {"phase": f"cycle{cycle}_1v", "normal_count": sum(row["status"] == "OK" for row in rows),
                  "heater_count": len(rows),
                  "abnormal": [{"port": row["port"], "status": row["status"],
                                "current_mA": row["measured_current_mA"]}
                               for row in rows if row["status"] != "OK"]}
        phases.append(result)
        print(json.dumps(result), flush=True)
        zero_ok = zero_phase(connection, f"cycle{cycle}_zero")
        if result["normal_count"] != len(rows) or not zero_ok:
            break
except BaseException as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    print(report["error"], flush=True)
finally:
    if connection is not None:
        try:
            report["final_zero_verified"] = False
            for attempt, delay in enumerate((DELAY, max(DELAY, 0.05), 0.1), start=1):
                if zero_phase(connection, f"final_zero{attempt}", delay):
                    report["final_zero_verified"] = True
                    break
        except BaseException as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            connection.close()
    report["phases"] = phases
    report["frame_count"] = len(frames)
    report["invalid_frame_count"] = sum(not frame["valid"] for frame in frames)
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    check.save_rows(OUT / "raw_serial_frames.csv", frames)
    check.save_rows(OUT / "current_samples.csv", raw_currents)
    check.save_rows(OUT / "heater_diagnostics.csv", diagnostics)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "phases"}), flush=True)
