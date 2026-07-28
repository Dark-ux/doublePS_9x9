import json
import shutil
from datetime import datetime
from pathlib import Path

import inner_calibration as inner
import inter_calibration as inter
import utils.communication as cu


N = 9
INNER_TARGET = 10
INTER_TARGETS = [9, 10, 11, 12]
SER_ADDRESS = "COM3"
OPM2_ADDRESS = "TCPIP0::192.168.0.7::inst0::INSTR"


def backup_calibration_files():
    backup_dir = Path("results") / "CalibrationBackups" / f"run_{datetime.now():%Y%m%d_%H%M%S}_mzi10_column"
    backup_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        Path("Scandata/MZI_table.json"),
        Path("Scandata/mzi_state_table.json"),
        Path("Scandata/inter_cali_pairs.json"),
        Path("Scandata/inner_fit_params.json"),
        Path("Scandata/inner_cali_powerdata/10-u.txt"),
        Path("Scandata/inner_cali_powerdata/10-d.txt"),
    ]
    for target in INTER_TARGETS:
        table = inter.load_mzi_table()
        ports = table[str(target)]["ports"]
        paths.extend(
            [
                Path("Scandata/inter_cali_powerdata") / f"{ports}.txt",
                Path("Scandata/inter_cali_powerimage") / f"{ports}.png",
                Path("Scandata/inter_cali") / f"target_{target}_ports_{'-'.join(map(str, ports))}.png",
            ]
        )
    for path in paths:
        if path.exists():
            destination = backup_dir / path.relative_to("Scandata") if "Scandata" in path.parts else backup_dir / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return backup_dir


def run_inner_calibration(mcv, opm2):
    print(f"Starting targeted inner calibration for MZI {INNER_TARGET}")
    inner.mzi_table = inner.load_mzi_table()
    inner.working_data = cu.generate_working_data()
    inner.mcv = mcv
    inner.opm2 = opm2

    order1, _ = inner.get_cali_order(N)
    for index in range(N - 1):
        inner.switch_IN(index + 1, "OFF", inner.working_data)
    path_reference_table = inner._load_mzi_path_reference_table(inner.mzi_table)
    inner._apply_saved_bar_biases(path_reference_table, inner.working_data)
    mzi_state_table, saved_mzi_table, path_reference_table = inner._prescan_initial_path(
        order1,
        N,
        mcv,
        opm2,
        1,
        inner.working_data,
        inner.mzi_table,
        path_reference_table,
    )

    target = INNER_TARGET
    target_ports = inner.get_mzi_h_list(target)
    path, input_index, output_index, state = inner.find_path(target, N)
    for index in range(N - 1):
        inner.switch_IN(index + 1, "OFF", inner.working_data)
    inner.switch_IN(input_index + 1, "ON", inner.working_data)
    inner._apply_known_path_states(
        path,
        state,
        target,
        mzi_state_table,
        path_reference_table,
        inner.working_data,
    )

    scan_results = []
    for arm_index, port in enumerate(target_ports):
        result = inner.scan_mzi(
            port,
            0,
            inner.INNER_SCAN_MAX_VOLTAGE_V,
            0.1,
            mcv,
            opm2,
            1,
            output_index + 1,
            inner.working_data,
            mzi_id=target,
        )
        scan_results.append(result)
        if arm_index == 0:
            mzi_state_table[target] = (result["bar_voltage_v"], result["cross_voltage_v"])
            inner._save_mzi_state_table(mzi_state_table)

    table_path = inner._save_inner_mzi_scan_results(target, scan_results, inner.INNER_MZI_TABLE_PATH)
    print(f"Updated MZI {target} inner calibration in {table_path}")
    return scan_results


def run_inter_calibration(mcv, opm2):
    print(f"Starting inter calibration for column targets {INTER_TARGETS}")
    inter.mzi_table = inter.load_mzi_table()
    inter.working_data = cu.generate_working_data()
    inter.mcv = mcv
    inter.opm2 = opm2
    inter.N = N
    inter.target_scan_voltage_pairs = inter._load_target_scan_voltage_pairs()

    records = []
    for target in INTER_TARGETS:
        print("-" * 60)
        print(f"Force recalibrating target MZI {target}")
        path, input_index, output_index, state, bmzi = inter.find_Bmzi_path(target, N)
        print(f"path={path}, input={input_index}, output={output_index}, state={state}, bmzi={bmzi}")
        up_r, up_p, down_r, down_p = inter.Power_halfpi(target)
        for mzi_id in range(1, N * (N - 1) // 2 + 1):
            entry = inter.mzi_table[str(mzi_id)]
            inter.write_port_voltage(entry["ports"][0], entry["dtheta"][0], inter.working_data)
            if len(entry["ports"]) > 1:
                inter.write_port_voltage(entry["ports"][1], 0.0, inter.working_data)
        inter.build_Bmzi(target, N)
        record = inter.scan_mzis(
            target,
            mcv,
            opm2,
            2,
            output_index + 1,
            inter.working_data,
            up_r,
            up_p,
            down_r,
            down_p,
        )
        inter.write_port_voltage(inter.mzi_table[str(target)]["ports"][1], 0.0, inter.working_data)
        inter.fit_inter_cali_sine(target, show_plot=False)
        inter._save_target_scan_voltage_pairs()
        records.append(record)
    return records


def main():
    backup_dir = backup_calibration_files()
    print(f"Calibration backup saved to {backup_dir}")
    mcv = cu.open_ser_connection(SER_ADDRESS)
    if mcv is None:
        raise RuntimeError(f"Could not open {SER_ADDRESS}")
    opm2 = cu.open_VISA_connection(OPM2_ADDRESS)
    if opm2 is None:
        mcv.close()
        raise RuntimeError(f"Could not open {OPM2_ADDRESS}")
    try:
        inner_results = run_inner_calibration(mcv, opm2)
        cu.clearallvoltage(mcv)
        inter_results = run_inter_calibration(mcv, opm2)
        summary = {
            "inner_target": INNER_TARGET,
            "inter_targets": INTER_TARGETS,
            "inner_results": inner_results,
            "inter_results": inter_results,
            "backup_dir": str(backup_dir),
        }
        (backup_dir / "new_calibration_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    finally:
        try:
            cu.clearallvoltage(mcv)
        finally:
            mcv.close()
            opm2.close()
    print("MZI10 column recalibration completed; all voltages cleared.")


if __name__ == "__main__":
    main()
