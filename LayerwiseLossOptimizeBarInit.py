"""Layerwise loss optimization with upper-Bar/lower-zero layer initialization.

This variant keeps the optimization, measurement, resume, and logging behavior
from LayerwiseLossOptimize.py.  The active layer starts from the upper-arm Bar
voltage in Scandata/MZI_table.json and 0 V on the lower arm instead of using an
inter-calibration voltage pair and target-phase prebias.
"""

import os

import numpy as np

import LayerwiseLossOptimize as base


HARDWARE_V_MIN = 0.0
HARDWARE_V_MAX = 5.5
VOLTAGE_EPS = 1e-12


def set_layer_to_upper_bar_lower_zero(
    working_data,
    cm,
    layer_index,
    mzi_table,
    target_thetas,
    v_min=HARDWARE_V_MIN,
    v_max=HARDWARE_V_MAX,
):
    """Initialize one active layer from MZI-table upper Bar and lower 0 V."""
    records = []
    for mzi_id in base.get_layer_mzi_ids(cm, layer_index):
        entry = base.um.get_mzi_entry(mzi_table, mzi_id)
        ports = entry.get("ports", [])
        if not ports:
            continue

        upper_bar_voltage = float(
            base.um.get_mzi_state_voltage(mzi_table, mzi_id, "BAR", arm_index=0)
        )
        for arm_index, port in enumerate(ports):
            requested_voltage = upper_bar_voltage if arm_index == 0 else 0.0
            initial_voltage = base.write_checked_port_voltage(
                port,
                requested_voltage,
                working_data,
                v_min,
                v_max,
            )
            base_phase = np.pi if arm_index == 0 else 0.0
            target_phase = float(target_thetas[mzi_id - 1, arm_index])
            records.append(
                {
                    "mzi_id": int(mzi_id),
                    "layer": int(layer_index + 1),
                    "arm_index": int(arm_index),
                    "port": int(port),
                    "base_voltage": float(initial_voltage),
                    "base_phase": float(base_phase),
                    "target_phase": target_phase,
                    "phase_offset": float(target_phase - base_phase),
                    "initial_voltage": float(initial_voltage),
                    "mode": "mzi_table_upper_bar_lower_zero",
                }
            )
    base.validate_all_voltages(working_data, v_min, v_max)
    return records


def voltage_to_bar_init_dry_run_theta(mzi_table, mzi_id, arm_index, voltage):
    """Map dry-run voltage to phase using upper-Bar/lower-zero references."""
    entry = base.um.get_mzi_entry(mzi_table, mzi_id)
    ports = entry.get("ports", [])
    bar_values = entry.get("dtheta_Bar", entry.get("dtheta", []))
    cross_values = entry.get("dtheta_Cross", entry.get("dtheta", []))
    ppi_values = entry.get("Ppi", [])
    heater_r_values = entry.get("heater_R", [])

    if arm_index >= len(bar_values) or arm_index >= len(cross_values):
        return float(np.pi if arm_index == 0 else 0.0)

    if len(ports) >= 2 and arm_index < len(ppi_values) and arm_index < len(heater_r_values):
        base_voltage = float(bar_values[0]) if arm_index == 0 else 0.0
        base_phase = np.pi if arm_index == 0 else 0.0
        base_power = base_voltage**2 / float(heater_r_values[arm_index])
        power = float(voltage) ** 2 / float(heater_r_values[arm_index])
        return float(base_phase + (power - base_power) / float(ppi_values[arm_index]) * np.pi)

    bar_v = float(bar_values[arm_index])
    cross_v = float(cross_values[arm_index])
    ratio = 0.0 if abs(cross_v - bar_v) < VOLTAGE_EPS else (float(voltage) - bar_v) / (cross_v - bar_v)
    return float((1.0 - ratio) * np.pi) if arm_index == 0 else float(ratio * np.pi)


def build_coarse_search_voltages(args, base_voltage):
    """Return a safe 0.5 V-style positive grid for a lower-arm coarse search."""
    upper_limit = min(float(args.lower_arm_coarse_max_v), float(args.v_max))
    if upper_limit <= float(base_voltage) + VOLTAGE_EPS:
        return []

    step_v = float(args.lower_arm_coarse_step_v)
    first = float(base_voltage) + step_v
    voltages = list(np.arange(first, upper_limit + VOLTAGE_EPS, step_v, dtype=float))
    if not voltages or voltages[-1] < upper_limit - VOLTAGE_EPS:
        voltages.append(upper_limit)
    return [float(np.clip(v, args.v_min, args.v_max)) for v in voltages]


def try_lower_arm_coarse_search(
    args,
    working_data,
    heater,
    heater_before,
    current_loss,
    p_target,
    layer_index,
    iter_idx,
):
    """Try larger positive lower-arm steps after normal line search rejects."""
    base_voltage = float(heater_before[0])
    is_lower_arm = int(heater["arm_index"]) == 1
    at_lower_bound = base_voltage <= float(args.v_min) + VOLTAGE_EPS
    if not is_lower_arm or not at_lower_bound:
        return {
            "accepted": False,
            "voltage": heater_before.copy(),
            "loss": float(current_loss),
            "trials": 0,
            "step_v": 0.0,
        }

    trials = 0
    best_loss = float(current_loss)
    best_voltage = base_voltage
    for trial_voltage in build_coarse_search_voltages(args, base_voltage):
        if trial_voltage <= base_voltage + VOLTAGE_EPS:
            continue
        trials += 1
        trial_after = np.array([trial_voltage], dtype=float)
        base.set_heater_voltages(working_data, [heater], trial_after, args.v_min, args.v_max)
        args.measure_context = (
            f"layer{layer_index + 1:02d}_iter{iter_idx:03d}_"
            f"{heater['label']}_lower_coarse_search_{trials:02d}"
        )
        trial_loss = base.power_matrix_loss(base.measure_power_matrix(args, working_data), p_target)
        if trial_loss < best_loss:
            best_loss = float(trial_loss)
            best_voltage = float(trial_voltage)

    if best_voltage > base_voltage + VOLTAGE_EPS:
        best_after = np.array([best_voltage], dtype=float)
        base.set_heater_voltages(working_data, [heater], best_after, args.v_min, args.v_max)
        return {
            "accepted": True,
            "voltage": best_after,
            "loss": float(best_loss),
            "trials": int(trials),
            "step_v": float(best_voltage - base_voltage),
        }

    base.set_heater_voltages(working_data, [heater], heater_before, args.v_min, args.v_max)
    return {
        "accepted": False,
        "voltage": heater_before.copy(),
        "loss": float(current_loss),
        "trials": int(trials),
        "step_v": 0.0,
    }


def append_heater_update_log(run_dir, rows):
    base.write_rows(
        run_dir / "heater_update_log.csv",
        rows,
        [
            "global_target_layer",
            "current_layer",
            "layer_status",
            "skip_reason",
            "layer_iter",
            "loss_before_iter",
            "loss_after_iter",
            "heater_label",
            "heater_port",
            "mzi_id",
            "arm_index",
            "voltage_before",
            "voltage_after",
            "gradient",
            "accepted",
            "line_search_trials",
            "lr_start",
            "lr_used",
            "lr_next",
            "update_mode",
            "coarse_search_trials",
            "coarse_search_step_v",
        ],
    )


def optimize_one_layer(
    args,
    working_data,
    cm,
    layer_index,
    mzi_table,
    target_thetas,
    run_dir,
    start_iter=0,
    initialize_layer=True,
):
    args.current_layer_index = int(layer_index)
    if initialize_layer:
        base.set_layers_after_to_bar(working_data, cm, layer_index, mzi_table, args.v_min, args.v_max)
        init_records = set_layer_to_upper_bar_lower_zero(
            working_data,
            cm,
            layer_index,
            mzi_table,
            target_thetas,
            args.v_min,
            args.v_max,
        )
    else:
        base.validate_all_voltages(working_data, args.v_min, args.v_max)
        init_records = []
    base.save_initial_voltage_records(run_dir, init_records)

    active_mzi_ids = base.get_layer_mzi_ids(cm, layer_index)
    active_heaters = base.get_active_heater_ports(mzi_table, active_mzi_ids)
    args.active_heaters = active_heaters
    p_target = base.build_target_power_matrix(cm, target_thetas, args.output_count, layer_index)
    np.savetxt(run_dir / f"target_power_matrix_layer{layer_index + 1:02d}.csv", p_target, delimiter=",")

    print(f"Optimizing layer {layer_index + 1} from upper-Bar/lower-zero: MZI ids {active_mzi_ids}")
    print(f"Active heaters: {active_heaters}")
    if initialize_layer:
        print("Initial active-layer voltages:")
        for rec in init_records:
            print(
                f"  MZI{rec['mzi_id']}_arm{rec['arm_index']} port {rec['port']}: "
                f"{rec['initial_voltage']:.3f} V ({rec['mode']})"
            )
    else:
        print(f"Resuming layer {layer_index + 1} from existing voltage state at iter {int(start_iter)}.")

    prev_loss = None
    final_loss = None
    final_power = None
    iterations_completed = 0
    layer_start_loss = None
    previous_grad_by_label = base.load_previous_layer_gradients(
        run_dir,
        layer_index + 1,
        int(start_iter) - 1,
    )
    adaptive_lr_by_label = {heater["label"]: float(args.lr) for heater in active_heaters}

    for iter_idx in range(int(start_iter), args.max_iter):
        args.measure_context = f"layer{layer_index + 1:02d}_iter{iter_idx:03d}_baseline"
        p_current = base.measure_power_matrix(args, working_data)
        loss = base.power_matrix_loss(p_current, p_target)
        cosine_before_iter = base.power_matrix_cosine_similarity(p_current, p_target)
        if layer_start_loss is None:
            layer_start_loss = float(loss)
        np.savetxt(run_dir / f"P_current_layer{layer_index + 1:02d}_iter{iter_idx:03d}.csv", p_current, delimiter=",")
        base.save_voltage_state(
            run_dir / f"voltage_state_layer{layer_index + 1:02d}_iter{iter_idx:03d}.csv",
            working_data,
        )

        current_loss = loss
        grad_values = []
        heater_rows = []
        accepted_any = False
        rejected_any = False
        lr_values = []
        iteration_heaters, heater_order_mode = base.order_heaters_for_iteration(
            active_heaters,
            previous_grad_by_label,
            args.heater_order_strategy,
        )
        print(
            f"Layer {layer_index + 1} iter {iter_idx:03d} heater order "
            f"({heater_order_mode}): {', '.join(h['label'] for h in iteration_heaters)}"
        )
        current_grad_by_label = {}

        for heater in iteration_heaters:
            heater_loss_before = current_loss
            heater_before = base.get_heater_voltages(working_data, [heater])
            args.measure_context_prefix = f"layer{layer_index + 1:02d}_iter{iter_idx:03d}"
            grad, _ = base.finite_difference_gradient(args, working_data, [heater], heater_before, p_target)
            grad_value = float(grad[0]) if grad.size else 0.0
            grad_values.append(grad_value)
            current_grad_by_label[heater["label"]] = grad_value

            lr_start = float(adaptive_lr_by_label.get(heater["label"], args.lr))
            accepted = True
            rejected = False
            accepted_lr = lr_start
            line_search_trials = 0
            coarse_search_trials = 0
            coarse_search_step_v = 0.0
            update_mode = "direct"
            heater_after = base.update_voltages(
                heater_before,
                grad,
                lr_start,
                args.max_step_v,
                args.v_min,
                args.v_max,
            )

            if args.line_search:
                accepted = False
                update_mode = "rejected"
                trial_lr = lr_start
                while trial_lr >= float(args.line_search_min_lr):
                    trial_after = base.update_voltages(
                        heater_before,
                        grad,
                        trial_lr,
                        args.max_step_v,
                        args.v_min,
                        args.v_max,
                    )
                    # At a voltage boundary, a gradient step can clip back to the
                    # same voltage.  Do not spend hardware measurements on no-op trials.
                    if np.allclose(trial_after, heater_before, atol=VOLTAGE_EPS, rtol=0.0):
                        trial_lr *= float(args.line_search_shrink)
                        continue
                    line_search_trials += 1
                    base.set_heater_voltages(working_data, [heater], trial_after, args.v_min, args.v_max)
                    args.measure_context = (
                        f"layer{layer_index + 1:02d}_iter{iter_idx:03d}_"
                        f"{heater['label']}_line_search"
                    )
                    trial_loss = base.power_matrix_loss(base.measure_power_matrix(args, working_data), p_target)
                    if trial_loss < current_loss:
                        accepted = True
                        accepted_lr = trial_lr
                        heater_after = trial_after
                        current_loss = trial_loss
                        update_mode = "line_search"
                        break
                    trial_lr *= float(args.line_search_shrink)

                if not accepted:
                    base.set_heater_voltages(
                        working_data,
                        [heater],
                        heater_before,
                        args.v_min,
                        args.v_max,
                    )
                    escape = try_lower_arm_coarse_search(
                        args,
                        working_data,
                        heater,
                        heater_before,
                        current_loss,
                        p_target,
                        layer_index,
                        iter_idx,
                    )
                    coarse_search_trials = int(escape["trials"])
                    coarse_search_step_v = float(escape["step_v"])
                    if escape["accepted"]:
                        accepted = True
                        heater_after = escape["voltage"]
                        current_loss = float(escape["loss"])
                        update_mode = "lower_boundary_coarse_search"
                    else:
                        rejected = True
                        heater_after = heater_before.copy()
                        update_mode = "rejected"
                        base.set_heater_voltages(
                            working_data,
                            [heater],
                            heater_before,
                            args.v_min,
                            args.v_max,
                        )
            else:
                base.set_heater_voltages(working_data, [heater], heater_after, args.v_min, args.v_max)
                args.measure_context = (
                    f"layer{layer_index + 1:02d}_iter{iter_idx:03d}_{heater['label']}_accepted"
                )
                current_loss = base.power_matrix_loss(base.measure_power_matrix(args, working_data), p_target)

            if args.adaptive_lr and args.line_search:
                if accepted:
                    lr_next = min(float(args.lr_max), accepted_lr * float(args.lr_growth))
                else:
                    lr_next = max(
                        float(args.line_search_min_lr),
                        lr_start * float(args.line_search_shrink),
                    )
            else:
                lr_next = lr_start
            adaptive_lr_by_label[heater["label"]] = float(lr_next)

            base.validate_all_voltages(working_data, args.v_min, args.v_max)
            accepted_any = accepted_any or accepted
            rejected_any = rejected_any or rejected
            lr_values.append(float(accepted_lr))
            heater_rows.append(
                {
                    "global_target_layer": int(args.layer),
                    "current_layer": int(layer_index + 1),
                    "layer_status": "optimized",
                    "skip_reason": "",
                    "layer_iter": int(iter_idx),
                    "loss_before_iter": float(heater_loss_before),
                    "loss_after_iter": float(current_loss),
                    "heater_label": heater["label"],
                    "heater_port": int(heater["port"]),
                    "mzi_id": int(heater["mzi_id"]),
                    "arm_index": int(heater["arm_index"]),
                    "voltage_before": float(heater_before[0]),
                    "voltage_after": float(heater_after[0]),
                    "gradient": grad_value,
                    "accepted": bool(accepted),
                    "line_search_trials": int(line_search_trials),
                    "lr_start": float(lr_start),
                    "lr_used": float(accepted_lr),
                    "lr_next": float(lr_next),
                    "update_mode": update_mode,
                    "coarse_search_trials": int(coarse_search_trials),
                    "coarse_search_step_v": float(coarse_search_step_v),
                }
            )

        args.measure_context = f"layer{layer_index + 1:02d}_iter{iter_idx:03d}_final"
        final_power = base.measure_power_matrix(args, working_data)
        final_loss = base.power_matrix_loss(final_power, p_target)
        cosine_after_iter = base.power_matrix_cosine_similarity(final_power, p_target)
        grad_array = np.asarray(grad_values, dtype=float)
        grad_norm = float(np.linalg.norm(grad_array))
        max_abs_grad = float(np.max(np.abs(grad_array))) if grad_array.size else 0.0

        base.append_multilayer_iter_log(
            run_dir,
            {
                "global_target_layer": int(args.layer),
                "current_layer": int(layer_index + 1),
                "layer_status": "optimized",
                "skip_reason": "",
                "layer_iter": int(iter_idx),
                "loss_before_iter": float(loss),
                "loss_after_iter": float(final_loss),
                "cosine_before_iter": float(cosine_before_iter),
                "cosine_after_iter": float(cosine_after_iter),
                "grad_norm": grad_norm,
                "max_abs_grad": max_abs_grad,
                "accepted": bool(accepted_any),
                "rejected": bool(rejected_any),
                "lr_used": min(lr_values) if lr_values else float(args.lr),
            },
        )
        append_heater_update_log(run_dir, heater_rows)
        iterations_completed += 1
        previous_grad_by_label = current_grad_by_label

        print(
            f"layer={layer_index + 1}, iter={iter_idx:03d}, loss={loss:.6g}, "
            f"final_loss={final_loss:.6g}, grad_norm={grad_norm:.6g}, "
            f"accepted_any={accepted_any}"
        )
        if prev_loss is not None and abs(prev_loss - final_loss) < args.loss_tol:
            break
        prev_loss = final_loss

    if final_power is None:
        args.measure_context = f"layer{layer_index + 1:02d}_final"
        final_power = base.measure_power_matrix(args, working_data)
        final_loss = base.power_matrix_loss(final_power, p_target)

    base.append_layer_summary(
        run_dir,
        {
            "global_target_layer": int(args.layer),
            "current_layer": int(layer_index + 1),
            "layer_status": "optimized",
            "skip_reason": "",
            "start_loss": "" if layer_start_loss is None else float(layer_start_loss),
            "final_loss": float(final_loss),
            "iterations_completed": int(iterations_completed),
            "active_heater_labels": ";".join(h["label"] for h in active_heaters),
            "active_heater_ports": ";".join(str(h["port"]) for h in active_heaters),
        },
    )
    return final_power, final_loss


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = (
        "Layerwise finite-difference loss optimization initialized from the "
        "MZI-table upper-arm Bar voltage and lower-arm 0 V."
    )
    parser.set_defaults(out_dir=os.path.join("results", "LayerwiseLossOptimizeBarInit"))
    parser.add_argument(
        "--lower-arm-coarse-step-v",
        type=base.positive_float,
        default=0.5,
        help="Voltage-grid spacing for lower-arm coarse search after normal line search rejects.",
    )
    parser.add_argument(
        "--lower-arm-coarse-max-v",
        type=base.positive_float,
        default=5.5,
        help="Maximum absolute lower-arm voltage considered by the coarse search.",
    )
    parser.add_argument(
        "--adaptive-lr",
        type=base.parse_bool,
        default=True,
        help="Grow learning rate after acceptance and shrink it after rejection.",
    )
    parser.add_argument(
        "--lr-growth",
        type=base.positive_float,
        default=1.5,
        help="Adaptive learning-rate growth factor after an accepted update; must be > 1.",
    )
    parser.add_argument(
        "--lr-max",
        type=base.positive_float,
        default=1.0,
        help="Maximum adaptive learning rate.",
    )
    return parser


def validate_args(args):
    if args.N < 2:
        raise ValueError("--N must be >= 2")
    if abs(float(args.v_min) - HARDWARE_V_MIN) > VOLTAGE_EPS:
        raise ValueError("Bar-init mode requires --v-min 0 so lower arms start at 0 V.")
    if args.v_max < args.v_min or args.v_max > HARDWARE_V_MAX:
        raise ValueError("Voltage limits must satisfy 0 <= v-min <= v-max <= 5.5 V.")
    if args.lower_arm_coarse_step_v > args.v_max - args.v_min:
        raise ValueError("--lower-arm-coarse-step-v exceeds the configured voltage range")
    if not args.v_min < args.lower_arm_coarse_max_v <= args.v_max:
        raise ValueError("--lower-arm-coarse-max-v must be within (v-min, v-max]")
    if args.lr_growth <= 1.0:
        raise ValueError("--lr-growth must be > 1")
    if not 0.0 < args.line_search_shrink < 1.0:
        raise ValueError("--line-search-shrink must be within (0, 1)")
    if args.lr < args.line_search_min_lr:
        raise ValueError("--lr must be >= --line-search-min-lr")
    if args.lr_max < args.line_search_min_lr:
        raise ValueError("--lr-max must be >= --line-search-min-lr")
    if args.adaptive_lr and args.lr > args.lr_max:
        raise ValueError("Adaptive learning rate requires --lr <= --lr-max")
    if args.init_mode != "bar" and not (args.resume_run_dir or args.resume_voltage_file):
        raise RuntimeError("Resume/current initialization is disabled; every run must start from all-Bar.")
    if not args.dry_run and not args.confirm_hardware:
        raise RuntimeError("Refusing hardware access: set --confirm-hardware true with --dry-run false.")


def main():
    base.install_elapsed_print()
    args = build_arg_parser().parse_args()
    validate_args(args)

    # Patch only the two extension points needed by the shared optimization
    # workflow.  Everything else continues to use LayerwiseLossOptimize.py.
    base.optimize_one_layer = optimize_one_layer
    base.voltage_to_dry_run_theta = voltage_to_bar_init_dry_run_theta

    print("Layerwise Bar-init loss optimization")
    print("Layer initialization: MZI-table upper Bar voltage, lower arm 0 V")
    print(
        "Lower-arm coarse search: "
        f"step={args.lower_arm_coarse_step_v} V, "
        f"max_voltage={args.lower_arm_coarse_max_v} V"
    )
    print(
        "Adaptive learning rate: "
        f"enabled={args.adaptive_lr}, growth={args.lr_growth}, max={args.lr_max}"
    )
    base.run_optimization(args)


if __name__ == "__main__":
    main()
