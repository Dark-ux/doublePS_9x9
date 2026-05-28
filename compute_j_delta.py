import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


OBSERVED_MZIS = [5, 6, 7, 8]
PERTURBED_HEATERS = ["5u", "5d", "6u", "6d", "7u", "7d", "8u", "8d"]
ARM_ALIASES = {
    "u": "u",
    "upper": "u",
    "up": "u",
    "0": "u",
    "d": "d",
    "lower": "d",
    "down": "d",
    "1": "d",
}


def load_mzi_table(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_arm(arm):
    key = str(arm).strip().lower()
    if key not in ARM_ALIASES:
        raise ValueError(f"Unsupported arm {arm!r}; expected u/upper or d/lower.")
    return ARM_ALIASES[key]


def get_fit_params(mzi_table, mzi_id, arm):
    arm = normalize_arm(arm)
    key = str(int(mzi_id))
    if key not in mzi_table:
        raise KeyError(f"MZI {mzi_id} not found in MZI table.")

    entry = mzi_table[key]
    fit_params = entry.get("fit_params", [])
    if not isinstance(fit_params, list) or not fit_params:
        raise ValueError(f"MZI {mzi_id} has no fit_params.")

    for idx, params in enumerate(fit_params):
        scan_path = str(params.get("scan_data_path", "")).replace("\\", "/").lower()
        name = Path(scan_path).name
        if name.endswith(f"-{arm}.txt") or name.endswith(f"_{arm}.txt"):
            return _build_fit_param_record(entry, params, idx, mzi_id, arm)

    fallback_idx = 0 if arm == "u" else 1
    if fallback_idx < len(fit_params):
        return _build_fit_param_record(entry, fit_params[fallback_idx], fallback_idx, mzi_id, arm)

    raise ValueError(f"MZI {mzi_id} has no fit_params for arm {arm}.")


def _build_fit_param_record(entry, params, idx, mzi_id, arm):
    required = ("A", "w", "phi", "b")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"MZI {mzi_id} arm {arm} fit_params missing: {', '.join(missing)}")

    ppi_values = entry.get("Ppi", [])
    ports = entry.get("ports", [])
    ppi = float(ppi_values[idx]) if idx < len(ppi_values) else np.nan
    port = int(params.get("port", ports[idx] if idx < len(ports) else -1))
    return {
        "mzi_id": int(mzi_id),
        "arm": arm,
        "port": port,
        "A": float(params["A"]),
        "w": float(params["w"]),
        "phi": float(params["phi"]),
        "b": float(params["b"]),
        "Ppi_w": ppi,
        "Ppi_mw": ppi * 1000.0 if np.isfinite(ppi) else np.nan,
        "scan_data_path": params.get("scan_data_path", ""),
    }


def load_scan_file(path):
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    if "probe_axis_power_w" in df.columns:
        power_col = "probe_axis_power_w"
    elif "measured_power_w" in df.columns:
        power_col = "measured_power_w"
    elif "target_power_w" in df.columns:
        power_col = "target_power_w"
    else:
        raise ValueError(f"{path} must contain measured_power_w or target_power_w.")

    if "optical_power_uW" not in df.columns:
        raise ValueError(f"{path} must contain optical_power_uW.")

    power_w = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=float)
    optical_power_uW = pd.to_numeric(df["optical_power_uW"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(power_w) & np.isfinite(optical_power_uW)
    if np.count_nonzero(valid) < 3:
        raise ValueError(f"{path} has fewer than three valid scan points.")

    metadata = {}
    for col in ("mzi_id", "arm_name", "arm_index", "port", "scan_stage"):
        if col in df.columns and not df[col].dropna().empty:
            metadata[col] = df[col].dropna().iloc[0]

    return {
        "dataframe": df,
        "power_w": power_w[valid],
        "optical_power_uW": optical_power_uW[valid],
        "power_column": power_col,
        "metadata": metadata,
    }


def sin_model(P, A, w, phi, b):
    return A * np.sin(w * P + phi) + b


def _sin_model_fixed_w(P, A, phi, b, w_fixed):
    return sin_model(P, A, w_fixed, phi, b)


def fit_probe_curve(power_w, optical_power_uW, init_params, fix_w=True):
    power_w = np.asarray(power_w, dtype=float)
    optical_power_uW = np.asarray(optical_power_uW, dtype=float)
    order = np.argsort(power_w)
    power_w = power_w[order]
    optical_power_uW = optical_power_uW[order]

    A0 = max(abs(float(init_params["A"])), 1e-9)
    w0 = max(abs(float(init_params["w"])), 1e-9)
    phi0 = float(init_params["phi"])
    b0 = float(init_params["b"])

    if fix_w:
        popt, _ = curve_fit(
            lambda P, A, phi, b: _sin_model_fixed_w(P, A, phi, b, w0),
            power_w,
            optical_power_uW,
            p0=[A0, phi0, b0],
            bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        A, phi, b = popt
        w = w0
    else:
        popt, _ = curve_fit(
            sin_model,
            power_w,
            optical_power_uW,
            p0=[A0, w0, phi0, b0],
            bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        A, w, phi, b = popt

    fitted = sin_model(power_w, A, w, phi, b)
    rmse = float(np.sqrt(np.mean((optical_power_uW - fitted) ** 2)))
    return {
        "A": float(A),
        "w": float(w),
        "phi": float(wrap_to_pi(phi)),
        "b": float(b),
        "rmse_uW": rmse,
        "power_w": power_w,
        "optical_power_uW": optical_power_uW,
        "fitted_uW": fitted,
    }


def wrap_to_pi(angle):
    return float((float(angle) + np.pi) % (2 * np.pi) - np.pi)


def parse_probe_map(probe_map_string):
    probe_map = {mzi_id: "u" for mzi_id in OBSERVED_MZIS}
    if probe_map_string is None or str(probe_map_string).strip() == "":
        return probe_map

    for item in str(probe_map_string).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid probe_map item {item!r}; expected mzi:arm.")
        mzi_text, arm_text = item.split(":", 1)
        mzi_id = int(mzi_text)
        if mzi_id not in OBSERVED_MZIS:
            raise ValueError(f"probe_map only supports MZI {OBSERVED_MZIS}, got {mzi_id}.")
        probe_map[mzi_id] = normalize_arm(arm_text)
    return probe_map


def compute_phase_shift_for_pair(baseline_file, perturbed_file, init_params, probe_arm, fix_w=True):
    baseline_scan = load_scan_file(baseline_file)
    perturbed_scan = load_scan_file(perturbed_file)

    baseline_fit = fit_probe_curve(
        baseline_scan["power_w"],
        baseline_scan["optical_power_uW"],
        init_params,
        fix_w=fix_w,
    )
    perturbed_fit = fit_probe_curve(
        perturbed_scan["power_w"],
        perturbed_scan["optical_power_uW"],
        init_params,
        fix_w=fix_w,
    )

    delta_eta = wrap_to_pi(perturbed_fit["phi"] - baseline_fit["phi"])
    probe_arm = normalize_arm(probe_arm)
    delta_delta = delta_eta if probe_arm == "u" else -delta_eta
    warning = ""
    if abs(delta_eta) > np.pi / 2:
        warning = "abs(delta_eta_rad) > pi/2; possible phase-branch jump or too-large perturbation"

    return {
        "baseline_scan": baseline_scan,
        "perturbed_scan": perturbed_scan,
        "baseline_fit": baseline_fit,
        "perturbed_fit": perturbed_fit,
        "delta_eta_rad": delta_eta,
        "delta_delta_rad": delta_delta,
        "warning": warning,
    }


def plot_fit_comparison(
    baseline_file,
    perturbed_file,
    baseline_fit,
    perturbed_fit,
    observed_mzi,
    probe_arm,
    perturbed_heater,
    out_path,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p_min = min(np.min(baseline_fit["power_w"]), np.min(perturbed_fit["power_w"]))
    p_max = max(np.max(baseline_fit["power_w"]), np.max(perturbed_fit["power_w"]))
    p_grid = np.linspace(p_min, p_max, 400)

    plt.figure(figsize=(7, 5))
    plt.plot(
        baseline_fit["power_w"] * 1000.0,
        baseline_fit["optical_power_uW"],
        "o",
        markersize=4,
        label="baseline data",
    )
    plt.plot(
        p_grid * 1000.0,
        sin_model(p_grid, baseline_fit["A"], baseline_fit["w"], baseline_fit["phi"], baseline_fit["b"]),
        "-",
        linewidth=1.5,
        label="baseline fit",
    )
    plt.plot(
        perturbed_fit["power_w"] * 1000.0,
        perturbed_fit["optical_power_uW"],
        "s",
        markersize=4,
        label="perturbed data",
    )
    plt.plot(
        p_grid * 1000.0,
        sin_model(p_grid, perturbed_fit["A"], perturbed_fit["w"], perturbed_fit["phi"], perturbed_fit["b"]),
        "-",
        linewidth=1.5,
        label="perturbed fit",
    )
    plt.xlabel("Probe heater power (mW)")
    plt.ylabel("Bar optical power (uW)")
    plt.title(f"obs{observed_mzi} probe {probe_arm}, perturb {perturbed_heater}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_heatmap(j_delta_rad_per_mw, row_labels, col_labels, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(j_delta_rad_per_mw, dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

    plt.figure(figsize=(9, 4.8))
    im = plt.imshow(values, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xlabel("Perturbed heater")
    plt.ylabel("Observed differential phase")
    plt.title("J_delta (rad/mW)")
    plt.colorbar(im, label="rad/mW")

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            text = "NaN" if not np.isfinite(value) else f"{value:.3g}"
            plt.text(c, r, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _read_delta_power(metadata_path):
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "delta_power_w" not in metadata:
        raise ValueError(f"{metadata_path} missing delta_power_w.")
    delta_power_w = float(metadata["delta_power_w"])
    if not np.isfinite(delta_power_w) or delta_power_w == 0.0:
        raise ValueError(f"{metadata_path} has invalid delta_power_w={delta_power_w}.")
    return delta_power_w, metadata


def _missing_detail_row(observed_mzi, probe_arm, perturbed_heater, delta_power_w, warning):
    return {
        "observed_mzi": observed_mzi,
        "probe_arm": probe_arm,
        "perturbed_heater": perturbed_heater,
        "delta_power_w": delta_power_w,
        "baseline_phi": np.nan,
        "perturbed_phi": np.nan,
        "delta_eta_rad": np.nan,
        "delta_delta_rad": np.nan,
        "J_rad_per_w": np.nan,
        "J_rad_per_mw": np.nan,
        "baseline_rmse_uW": np.nan,
        "perturbed_rmse_uW": np.nan,
        "baseline_A": np.nan,
        "baseline_b": np.nan,
        "perturbed_A": np.nan,
        "perturbed_b": np.nan,
        "fixed_w": np.nan,
        "warning": warning,
        "baseline_scan_mzi_id": "",
        "perturbed_scan_mzi_id": "",
        "baseline_scan_arm_name": "",
        "perturbed_scan_arm_name": "",
        "baseline_scan_port": "",
        "perturbed_scan_port": "",
    }


def _detail_row_from_result(observed_mzi, probe_arm, perturbed_heater, delta_power_w, init_params, result):
    baseline_fit = result["baseline_fit"]
    perturbed_fit = result["perturbed_fit"]
    j_rad_per_w = result["delta_delta_rad"] / delta_power_w
    baseline_meta = result["baseline_scan"]["metadata"]
    perturbed_meta = result["perturbed_scan"]["metadata"]
    return {
        "observed_mzi": observed_mzi,
        "probe_arm": probe_arm,
        "perturbed_heater": perturbed_heater,
        "delta_power_w": delta_power_w,
        "baseline_phi": baseline_fit["phi"],
        "perturbed_phi": perturbed_fit["phi"],
        "delta_eta_rad": result["delta_eta_rad"],
        "delta_delta_rad": result["delta_delta_rad"],
        "J_rad_per_w": j_rad_per_w,
        "J_rad_per_mw": j_rad_per_w / 1000.0,
        "baseline_rmse_uW": baseline_fit["rmse_uW"],
        "perturbed_rmse_uW": perturbed_fit["rmse_uW"],
        "baseline_A": baseline_fit["A"],
        "baseline_b": baseline_fit["b"],
        "perturbed_A": perturbed_fit["A"],
        "perturbed_b": perturbed_fit["b"],
        "fixed_w": init_params["w"],
        "warning": result["warning"],
        "baseline_scan_mzi_id": baseline_meta.get("mzi_id", ""),
        "perturbed_scan_mzi_id": perturbed_meta.get("mzi_id", ""),
        "baseline_scan_arm_name": baseline_meta.get("arm_name", ""),
        "perturbed_scan_arm_name": perturbed_meta.get("arm_name", ""),
        "baseline_scan_port": baseline_meta.get("port", ""),
        "perturbed_scan_port": perturbed_meta.get("port", ""),
    }


def write_calibration_summary(mzi_table, out_dir, probe_map):
    rows = []
    for mzi_id in OBSERVED_MZIS:
        for arm in ("u", "d"):
            params = get_fit_params(mzi_table, mzi_id, arm)
            rows.append(params)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_summary_second_column.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("当前只有已有单臂扫描标定参数，不能直接计算完整 J_delta。")
    print("要计算 J_delta，需要 baseline 和逐个 heater 扰动后的 probe scan 曲线。")
    print("正式计算应使用曲线相位平移法，而不是全 Bar 点的单点功率反推。")
    print(f"Calibration summary saved to {out_path}")
    print(f"Probe map for future J_delta: {probe_map}")


def compute_j_delta(mzi_table_path, jacobian_dir, out_dir, probe_map, fix_w=True):
    mzi_table = load_mzi_table(mzi_table_path)
    jacobian_dir = Path(jacobian_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jacobian_dir.exists():
        write_calibration_summary(mzi_table, out_dir, probe_map)
        return None

    row_labels = [f"Delta{mzi_id}" for mzi_id in OBSERVED_MZIS]
    col_labels = [f"P{heater}" for heater in PERTURBED_HEATERS]
    j_delta = np.full((len(OBSERVED_MZIS), len(PERTURBED_HEATERS)), np.nan, dtype=float)
    detail_rows = []
    fit_fig_dir = out_dir / "fit_figures"

    for col_idx, perturbed_heater in enumerate(PERTURBED_HEATERS):
        perturb_dir = jacobian_dir / f"perturb_{perturbed_heater}"
        metadata_path = perturb_dir / "metadata.json"
        delta_power_w = np.nan
        perturb_warning = ""

        if not perturb_dir.exists():
            perturb_warning = f"missing perturb directory: {perturb_dir}"
            print(f"WARNING: {perturb_warning}")
        elif not metadata_path.exists():
            perturb_warning = f"missing metadata.json: {metadata_path}"
            print(f"WARNING: {perturb_warning}")
        else:
            try:
                delta_power_w, _ = _read_delta_power(metadata_path)
            except Exception as exc:
                perturb_warning = str(exc)
                print(f"WARNING: {perturb_warning}")

        for row_idx, observed_mzi in enumerate(OBSERVED_MZIS):
            probe_arm = probe_map.get(observed_mzi, "u")
            baseline_file = jacobian_dir / "baseline" / f"obs{observed_mzi}_probe.txt"
            perturbed_file = perturb_dir / f"obs{observed_mzi}_probe.txt"

            if perturb_warning:
                detail_rows.append(
                    _missing_detail_row(observed_mzi, probe_arm, perturbed_heater, delta_power_w, perturb_warning)
                )
                continue
            if not baseline_file.exists():
                warning = f"missing baseline file: {baseline_file}"
                print(f"WARNING: {warning}")
                detail_rows.append(
                    _missing_detail_row(observed_mzi, probe_arm, perturbed_heater, delta_power_w, warning)
                )
                continue
            if not perturbed_file.exists():
                warning = f"missing perturbed file: {perturbed_file}"
                print(f"WARNING: {warning}")
                detail_rows.append(
                    _missing_detail_row(observed_mzi, probe_arm, perturbed_heater, delta_power_w, warning)
                )
                continue

            try:
                init_params = get_fit_params(mzi_table, observed_mzi, probe_arm)
                result = compute_phase_shift_for_pair(
                    baseline_file,
                    perturbed_file,
                    init_params,
                    probe_arm,
                    fix_w=fix_w,
                )
                detail = _detail_row_from_result(
                    observed_mzi,
                    probe_arm,
                    perturbed_heater,
                    delta_power_w,
                    init_params,
                    result,
                )
                detail_rows.append(detail)
                j_delta[row_idx, col_idx] = detail["J_rad_per_w"]

                if detail["warning"]:
                    print(f"WARNING: obs{observed_mzi}, perturb {perturbed_heater}: {detail['warning']}")

                plot_fit_comparison(
                    baseline_file,
                    perturbed_file,
                    result["baseline_fit"],
                    result["perturbed_fit"],
                    observed_mzi,
                    probe_arm,
                    perturbed_heater,
                    fit_fig_dir / f"obs{observed_mzi}_perturb_{perturbed_heater}.png",
                )
            except Exception as exc:
                warning = f"fit failed: {exc}"
                print(f"WARNING: obs{observed_mzi}, perturb {perturbed_heater}: {warning}")
                detail_rows.append(
                    _missing_detail_row(observed_mzi, probe_arm, perturbed_heater, delta_power_w, warning)
                )

    j_w_df = pd.DataFrame(j_delta, index=row_labels, columns=col_labels)
    j_mw_df = j_w_df / 1000.0
    j_w_df.to_csv(out_dir / "J_delta_rad_per_w.csv")
    j_mw_df.to_csv(out_dir / "J_delta_rad_per_mw.csv")
    pd.DataFrame(detail_rows).to_csv(out_dir / "phase_shift_details.csv", index=False)
    plot_heatmap(j_mw_df.to_numpy(dtype=float), row_labels, col_labels, out_dir / "J_delta_heatmap_rad_per_mw.png")

    print(f"Saved J_delta results to {out_dir}")
    return j_w_df


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def resolve_default_mzi_table(path_text):
    path = Path(path_text)
    if path.exists():
        return path
    scandata_path = Path("Scandata") / path_text
    if scandata_path.exists():
        return scandata_path
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Compute the 4x8 differential-phase Jacobian J_delta for MZI 5-8."
    )
    parser.add_argument("--mzi_table", default="MZI_table.json", help="Path to MZI_table.json.")
    parser.add_argument("--jacobian_dir", default="jacobian_measurements", help="Jacobian measurement directory.")
    parser.add_argument("--out_dir", default="results", help="Output directory.")
    parser.add_argument("--probe_map", default="", help="Example: 5:u,6:u,7:u,8:u")
    parser.add_argument("--fix_w", type=str_to_bool, default=True, help="true: fix w from MZI_table; false: fit w.")
    args = parser.parse_args()

    probe_map = parse_probe_map(args.probe_map)
    mzi_table_path = resolve_default_mzi_table(args.mzi_table)
    compute_j_delta(
        mzi_table_path=mzi_table_path,
        jacobian_dir=args.jacobian_dir,
        out_dir=args.out_dir,
        probe_map=probe_map,
        fix_w=args.fix_w,
    )


if __name__ == "__main__":
    main()
