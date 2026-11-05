"""Run the original layerwise optimizer from random voltages near Vpi.

Vpi = sqrt(Ppi * heater_R), with Ppi in watts and resistance in ohms.
It is different from the optical half-power voltage in MZI_table.json.
Each optimized heater in a newly entered layer is sampled independently
on the controller's 1 mV grid. The default interval is Vpi +/- 0.1 V.
The seed and layer index determine the initial state reproducibly.

Only set_layer_to_target_prebiased_pair is replaced during the run. The
original optimizer, loss, focus selection, previous-layer compensation,
line search, convergence, measurement, and dry-run model are reused intact.
The first/last layers and future layers retain the original Bar rules.
Focus-excluded MZIs retain the original direct target mapping, since the
original optimizer never iterates those heaters.

Preview without hardware access or output files:
    python -B LayerwiseLossOptimizeRandomInit.py --layer 7 \
        --target-thetas target_thetas.npy --preview-init

A full run saves the same artifacts as LayerwiseLossOptimize.py and asks
before creating them. Use --save-records only when saving is intended.
Hardware access still requires --dry-run false --confirm-hardware true.
"""

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

import LayerwiseLossOptimize as base


INITIALIZATION_FIELDS = (
    "mzi_id", "layer", "arm_index", "port", "base_voltage", "base_phase",
    "target_phase", "folded_target_phase", "phase_wrap_cycles", "voltage_wrap_cycles",
    "total_wrap_cycles", "canonical_phase_offset", "phase_offset", "canonical_voltage",
    "initial_voltage", "mode", "vpi_voltage", "random_lower_v", "random_upper_v",
    "random_radius_v", "random_seed", "voltage_offset_v",
)


def finite_positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(out_dir=os.path.join("results", "LayerwiseLossOptimizeRandomInit"))
    parser.add_argument("--random-radius-v", type=finite_positive, default=0.1,
                        help="Half-width of the random interval around Vpi, in volts (default: 0.1).")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="Nonnegative seed; same seed reproduces each layer independently (default: 0).")
    parser.add_argument("--preview-init", action="store_true",
                        help="Print layer initialization voltages without hardware access or saving files.")
    parser.add_argument("--save-records", action="store_true",
                        help="Explicitly allow the original optimizer's experiment files; otherwise ask first.")
    return parser


def validate_args(args):
    if args.N < 2 or not 1 <= args.layer <= args.N:
        raise ValueError("Require N >= 2 and 1 <= layer <= N.")
    if not all(math.isfinite(v) for v in (args.v_min, args.v_max)):
        raise ValueError("Voltage bounds must be finite.")
    if not 0 <= args.v_min < args.v_max <= 5.5:
        raise ValueError("Voltage bounds must satisfy 0 <= v-min < v-max <= 5.5 V.")
    if args.random_seed < 0:
        raise ValueError("--random-seed must be nonnegative.")
    if not math.isfinite(args.random_radius_v) or args.random_radius_v <= 0:
        raise ValueError("--random-radius-v must be finite and positive.")
    if args.previous_layer_regularization < 0:
        raise ValueError("--previous-layer-regularization must be >= 0.")
    if args.convergence_patience < 1 or args.max_iter < 1:
        raise ValueError("--convergence-patience and --max-iter must be >= 1.")
    if args.init_mode != "bar" and not (args.resume_run_dir or args.resume_voltage_file):
        raise ValueError("Every new run must start from the original all-Bar background.")
    if not 0 < args.line_search_shrink < 1:
        raise ValueError("--line-search-shrink must be in (0, 1).")
    if not args.preview_init and not args.dry_run and not args.confirm_hardware:
        raise ValueError("Hardware access requires --dry-run false --confirm-hardware true.")


def layer_voltage_plan(cm, layer_index, mzi_table, seed, radius_v, v_min, v_max):
    """Build a complete layer plan without changing state or consuming a shared RNG."""
    if seed < 0 or not math.isfinite(radius_v) or radius_v <= 0:
        raise ValueError("Invalid random seed or radius.")
    if not all(math.isfinite(v) for v in (v_min, v_max)) or not 0 <= v_min < v_max <= 5.5:
        raise ValueError("Invalid voltage bounds.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(layer_index)]))
    records, seen_ports = [], set()
    for mzi_id in base.get_layer_mzi_ids(cm, layer_index):
        entry = base.um.get_mzi_entry(mzi_table, mzi_id)
        ports, ppi, resistance = entry.get("ports", []), entry.get("Ppi", []), entry.get("heater_R", [])
        if not ports or len(ppi) != len(ports) or len(resistance) != len(ports):
            raise ValueError(f"MZI {mzi_id}: one Ppi and heater_R are required for each heater.")
        for arm_index, raw_port in enumerate(ports):
            port = int(raw_port)
            if port != raw_port or not 1 <= port <= base.cu.CHANNEL_NUM or port in seen_ports:
                raise ValueError(f"MZI {mzi_id}: invalid or duplicate port {raw_port}.")
            seen_ports.add(port)
            power, ohms = float(ppi[arm_index]), float(resistance[arm_index])
            if not all(math.isfinite(v) and v > 0 for v in (power, ohms)):
                raise ValueError(f"MZI {mzi_id} arm {arm_index}: Ppi and heater_R must be finite and positive.")
            center = math.sqrt(power * ohms)
            if not math.isfinite(center):
                raise ValueError(f"MZI {mzi_id} arm {arm_index}: Vpi is not finite.")
            lower = max(float(v_min), center - radius_v)
            upper = min(float(v_max), center + radius_v)
            first_mv = math.ceil(lower * 1000 - 1e-9)
            last_mv = math.floor(upper * 1000 + 1e-9)
            if first_mv > last_mv:
                raise ValueError(f"MZI {mzi_id} arm {arm_index}: no safe 1 mV point near Vpi={center:.6g} V.")
            initial = int(rng.integers(first_mv, last_mv + 1)) / 1000.0
            records.append({
                "mzi_id": int(mzi_id), "layer": int(layer_index + 1),
                "arm_index": int(arm_index), "port": port,
                "base_voltage": center, "vpi_voltage": center,
                "random_lower_v": first_mv / 1000.0, "random_upper_v": last_mv / 1000.0,
                "random_radius_v": float(radius_v), "random_seed": int(seed),
                "initial_voltage": initial, "voltage_offset_v": initial - center,
                "mode": "random_near_vpi",
            })
    return records


def make_initializer(seed, radius_v, focus_size=None, focus_row_start=1, focus_col_start=1,
                     original_initializer=None):
    """Match the original initialization callback's signature exactly."""
    if original_initializer is None:
        original_initializer = base.set_layer_to_target_prebiased_pair

    def initialize(working_data, cm, layer_index, mzi_table, inter_cali_pairs,
                   target_thetas, v_min=0.0, v_max=5.5):
        active_ids, direct_ids = base.get_focus_influential_mzi_ids(
            cm, target_thetas, cm.shape[0], layer_index,
            focus_size, focus_row_start, focus_col_start,
        )
        records = [record for record in layer_voltage_plan(
            cm, layer_index, mzi_table, seed, radius_v, v_min, v_max
        ) if record["mzi_id"] in active_ids]
        candidate_data = working_data.copy(deep=True)
        direct_records = []
        if direct_ids:
            direct_cm = cm.copy()
            direct_cm[:, layer_index] = np.where(
                np.isin(cm[:, layer_index], direct_ids), cm[:, layer_index], 0
            )
            direct_records = original_initializer(
                candidate_data, direct_cm, layer_index, mzi_table, inter_cali_pairs,
                target_thetas, v_min, v_max,
            )
        # Use the unchanged simulator only to label phases in the existing log.
        # Neither the target phase nor the calibration pair selects the voltage.
        mapper = base.voltage_to_dry_run_theta
        had_pairs = hasattr(mapper, "inter_cali_pairs")
        old_pairs = getattr(mapper, "inter_cali_pairs", None)
        try:
            mapper.inter_cali_pairs = inter_cali_pairs
            for record in records:
                port = record["port"]
                if port > len(working_data):
                    raise ValueError(f"Port {port} exceeds the working voltage table.")
                mzi_id, arm_index = record["mzi_id"], record["arm_index"]
                record["base_phase"] = float(mapper(mzi_table, mzi_id, arm_index, record["base_voltage"]))
                record["target_phase"] = float(target_thetas[mzi_id - 1, arm_index])
                # The shared console output formats this field as a number.
                # No target-phase folding is needed to choose a random voltage.
                record["folded_target_phase"] = record["target_phase"]
                initial_phase = float(mapper(mzi_table, mzi_id, arm_index, record["initial_voltage"]))
                record["phase_offset"] = initial_phase - record["base_phase"]
        finally:
            if had_pairs:
                mapper.inter_cali_pairs = old_pairs
            else:
                del mapper.inter_cali_pairs
        for record in records:
            base.write_checked_port_voltage(record["port"], record["initial_voltage"], candidate_data, v_min, v_max)
            assert float(candidate_data.iloc[record["port"] - 1, 0]) == record["initial_voltage"]
        base.validate_all_voltages(candidate_data, v_min, v_max)
        working_data.iloc[:, 0] = candidate_data.iloc[:, 0]
        records = sorted(records + direct_records, key=lambda record: (record["mzi_id"], record["arm_index"]))
        # Every layer appends exactly the same CSV columns, including layers
        # that contain both random starts and preserved direct mappings.
        return [{key: record.get(key, "") for key in INITIALIZATION_FIELDS} for record in records]
    return initialize


@contextmanager
def random_initialization(seed, radius_v, focus_size=None, focus_row_start=1, focus_col_start=1):
    """Replace only the initializer and restore it even when a run fails."""
    original = base.set_layer_to_target_prebiased_pair
    base.set_layer_to_target_prebiased_pair = make_initializer(
        seed, radius_v, focus_size, focus_row_start, focus_col_start, original
    )
    try:
        yield
    finally:
        base.set_layer_to_target_prebiased_pair = original


def preview_initialization(args):
    """Validate every candidate layer before any device is opened or file created."""
    cm = base.du.Clements_matrix(args.N)
    table = base.um.load_mzi_table(args.mzi_table)
    target = base.load_target_thetas(args.target_thetas, args.N * (args.N - 1) // 2)
    if not np.isfinite(target).all():
        raise ValueError("Target theta values must be finite.")
    target = base.apply_mzi_table_phase_constraints(target, table)
    pairs = base.load_inter_cali_pairs(args.inter_cali_pairs)
    initialize = make_initializer(args.random_seed, args.random_radius_v, args.focus_size,
                                  args.focus_row_start, args.focus_col_start)
    working_data = base.pd.DataFrame(np.full((base.cu.CHANNEL_NUM, 1), args.v_min, dtype=float))
    records = []
    for layer_index in range(args.layer):
        if base.is_first_or_last_layer(layer_index, cm.shape[1]):
            continue
        records.extend(initialize(working_data, cm, layer_index, table, pairs,
                                  target, args.v_min, args.v_max))
    if len({record["port"] for record in records}) != len(records):
        raise ValueError("A heater port is assigned to more than one optimization layer.")
    return records


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    records = preview_initialization(args)
    print(f"Random layer initialization: Vpi +/- {args.random_radius_v:g} V; seed={args.random_seed}.")
    if args.preview_init:
        for record in records:
            center = f"Vpi={record['vpi_voltage']:.6f} V" if record["mode"] == "random_near_vpi" else "preserved direct target"
            print(f"Layer {record['layer']} MZI{record['mzi_id']}_arm{record['arm_index']} "
                  f"port {record['port']}: {center}, initial={record['initial_voltage']:.3f} V")
        print(f"Previewed {len(records)} heater initializations. No hardware access or files written.")
        return None
    if not args.save_records:
        try:
            answer = input(f"Save experiment records under {Path(args.out_dir)}? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes", "是", "保存"}:
            print("Run cancelled before hardware access or file creation. Use --preview-init for a file-free preview.")
            return None
        args.save_records = True
    base.install_elapsed_print()
    print("The original optimization algorithm and dry-run model are unchanged.")
    with random_initialization(args.random_seed, args.random_radius_v, args.focus_size,
                               args.focus_row_start, args.focus_col_start):
        return base.run_optimization(args)


if __name__ == "__main__":
    main()
