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


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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

    baseline_x_all = np.concatenate([baseline_fit["power_w"], np.asarray(baseline_fit.get("excluded_x", []), dtype=float)])
    perturbed_x_all = np.concatenate([perturbed_fit["power_w"], np.asarray(perturbed_fit.get("excluded_x", []), dtype=float)])
    p_min = min(np.min(baseline_x_all), np.min(perturbed_x_all))
    p_max = max(np.max(baseline_x_all), np.max(perturbed_x_all))
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
    if len(baseline_fit.get("excluded_x", [])):
        plt.plot(
            np.asarray(baseline_fit["excluded_x"]) * 1000.0,
            baseline_fit["excluded_y"],
            "x",
            markersize=8,
            label="baseline excluded",
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
    if len(perturbed_fit.get("excluded_x", [])):
        plt.plot(
            np.asarray(perturbed_fit["excluded_x"]) * 1000.0,
            perturbed_fit["excluded_y"],
            "x",
            markersize=8,
            label="perturbed excluded",
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


def plot_matrix_heatmap(matrix, row_labels, col_labels, out_path, title):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    plt.figure(figsize=(9, 4.8))
    plt.imshow(values, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xlabel("Perturbed heater")
    plt.ylabel("Observed phase")
    plt.title(title)
    plt.colorbar(label="rad/mW")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            plt.text(c, r, "NaN" if not np.isfinite(value) else f"{value:.3g}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_csv_list(text, item_type=str):
    if text is None or str(text).strip() == "":
        return []
    return [item_type(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_probe_map_for_ids(text, mzi_ids):
    result = {int(mzi_id): "u" for mzi_id in mzi_ids}
    if text is None or str(text).strip() == "":
        return result
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        mzi_text, arm_text = item.split(":", 1)
        result[int(mzi_text)] = normalize_arm(arm_text)
    return result


def parse_sigma_bmzi_map(text, mzi_ids):
    if text is None or str(text).strip() == "":
        text = "5:0,6:5,7:6,8:7"
    parsed = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        mzi_text, bmzi_text = item.split(":", 1)
        parsed[str(int(mzi_text))] = int(bmzi_text)
    missing = [str(int(mzi_id)) for mzi_id in mzi_ids if str(int(mzi_id)) not in parsed]
    if missing:
        raise ValueError(f"sigma_bmzi_map missing MZI ids: {missing}")
    return {str(int(mzi_id)): int(parsed[str(int(mzi_id))]) for mzi_id in mzi_ids}


def compute_global_sigma_from_relative_links(relative_links, sigma_bmzi_map, mzi_ids, mode):
    mode = str(mode or "chained_bmzi").strip().lower()
    if mode not in {"chained_bmzi", "direct_reference"}:
        raise ValueError("--sigma_reference_mode must be chained_bmzi or direct_reference")
    keys = [str(int(mzi_id)) for mzi_id in mzi_ids]
    if mode == "direct_reference":
        return {key: float(relative_links.get(key, np.nan)) for key in keys}, [int(key) for key in keys], True
    resolved = {"0": 0.0}
    visiting = set()
    order = []
    valid = True

    def resolve(key):
        nonlocal valid
        key = str(int(key))
        if key in resolved:
            return resolved[key]
        if key in visiting or key not in relative_links:
            valid = False
            return np.nan
        visiting.add(key)
        bmzi = int(sigma_bmzi_map[key])
        parent = 0.0 if bmzi == 0 else resolve(str(bmzi))
        visiting.remove(key)
        value = float(relative_links[key]) + float(parent) if np.isfinite(relative_links[key]) and np.isfinite(parent) else np.nan
        resolved[key] = value
        order.append(int(key))
        if not np.isfinite(value):
            valid = False
        return value

    return {key: resolve(key) for key in keys}, order, bool(valid)


def compute_global_sigma_matrix(j_sigma_link, sigma_bmzi_map, mzi_ids, mode):
    global_df = j_sigma_link.copy()
    chain_order = []
    chain_valid = True
    for col in j_sigma_link.columns:
        relative = {str(int(mzi_id)): float(j_sigma_link.loc[f"Sigma{int(mzi_id)}", col]) for mzi_id in mzi_ids}
        global_links, order, valid = compute_global_sigma_from_relative_links(relative, sigma_bmzi_map, mzi_ids, mode)
        chain_order = order
        chain_valid = chain_valid and valid
        for mzi_id in mzi_ids:
            global_df.loc[f"Sigma{int(mzi_id)}", col] = global_links[str(int(mzi_id))]
    return global_df, chain_order, bool(chain_valid)


def load_generic_scan(path, x_candidates, y_candidates=("optical_power_uW", "pow(uW)")):
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    x_col = next((col for col in x_candidates if col in df.columns), None)
    y_col = next((col for col in y_candidates if col in df.columns), None)
    if x_col is None or y_col is None:
        raise ValueError(f"{path} missing scan columns.")
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 4:
        raise ValueError(f"{path} has fewer than four valid points.")
    order = np.argsort(x[valid])
    return {"path": path, "df": df, "x": x[valid][order], "y": y[valid][order], "x_col": x_col, "y_col": y_col}


def fit_sine_free_or_fixed(x, y, fix_w=None, init_params=None, remove_one_outlier=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def fit_core(x_fit, y_fit):
        y_span = float(np.nanmax(y_fit) - np.nanmin(y_fit))
        A0 = max(0.5 * y_span, 1e-9)
        b0 = float(np.nanmean(y_fit))
        w0 = 1.0 if float(np.ptp(x_fit)) == 0.0 else float(2.0 * np.pi / max(float(np.ptp(x_fit)), 1e-9))
        phi0 = 0.0
        if init_params:
            A0 = max(abs(float(init_params.get("A", A0))), 1e-9)
            w0 = max(abs(float(init_params.get("w", w0))), 1e-9)
            phi0 = float(init_params.get("phi", init_params.get("beta", phi0)))
            b0 = float(init_params.get("b", b0))
        if fix_w is None:
            popt, _ = curve_fit(sin_model, x_fit, y_fit, p0=[A0, w0, phi0, b0], bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]), maxfev=50000)
            A, w, phi, b = popt
        else:
            w = float(fix_w)
            popt, _ = curve_fit(lambda xx, A, phi, b: sin_model(xx, A, w, phi, b), x_fit, y_fit, p0=[A0, phi0, b0], bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=50000)
            A, phi, b = popt
        fitted = sin_model(x_fit, A, w, phi, b)
        return {
            "A": float(A),
            "w": float(w),
            "phi": wrap_to_pi(phi),
            "b": float(b),
            "rmse_uW": rmse(y_fit, fitted),
            "x": x_fit,
            "y": y_fit,
            "fitted": fitted,
            "excluded_index": "",
            "excluded_x": np.asarray([], dtype=float),
            "excluded_y": np.asarray([], dtype=float),
            "outlier_removed": False,
        }

    full_fit = fit_core(x, y)
    if not remove_one_outlier or len(x) < 7:
        return full_fit

    best_fit = None
    best_drop_idx = None
    for drop_idx in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[drop_idx] = False
        try:
            candidate = fit_core(x[mask], y[mask])
        except Exception:
            continue
        if best_fit is None or candidate["rmse_uW"] < best_fit["rmse_uW"]:
            best_fit = candidate
            best_drop_idx = drop_idx

    if best_fit is None:
        return full_fit
    y_span = float(np.nanmax(y) - np.nanmin(y))
    min_improvement_uW = max(0.05, 0.02 * y_span)
    improves_enough = (
        best_fit["rmse_uW"] < 0.5 * full_fit["rmse_uW"]
        and (full_fit["rmse_uW"] - best_fit["rmse_uW"]) > min_improvement_uW
    )
    if not improves_enough:
        return full_fit

    best_fit["excluded_index"] = int(best_drop_idx)
    best_fit["excluded_x"] = np.asarray([x[best_drop_idx]], dtype=float)
    best_fit["excluded_y"] = np.asarray([y[best_drop_idx]], dtype=float)
    best_fit["outlier_removed"] = True
    best_fit["full_rmse_before_outlier_removal_uW"] = float(full_fit["rmse_uW"])
    return best_fit


def _read_delta_power(metadata_path):
    with Path(metadata_path).open("r", encoding="utf-8-sig") as f:
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


def compute_all(jacobian_dir, out_dir, mzi_ids, heaters, probe_map, sigma_bmzi_map, sigma_reference_mode, fix_w=True, mzi_table_path="Scandata/MZI_table.json"):
    jacobian_dir = Path(jacobian_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / "heatmaps"
    fit_dir = out_dir / "fit_figures"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)
    row_delta = [f"Delta{int(mzi_id)}" for mzi_id in mzi_ids]
    row_sigma = [f"Sigma{int(mzi_id)}" for mzi_id in mzi_ids]
    cols = [f"P{heater}" for heater in heaters]
    j_delta = pd.DataFrame(np.nan, index=row_delta, columns=cols)
    j_sigma_link = pd.DataFrame(np.nan, index=row_sigma, columns=cols)
    delta_details = []
    sigma_details = []
    warnings = []
    baseline_delta_fits = {}
    baseline_sigma_fits = {}
    mzi_table = load_mzi_table(resolve_default_mzi_table(mzi_table_path))

    for mzi_id in mzi_ids:
        probe_arm = probe_map.get(int(mzi_id), "u")
        delta_base = jacobian_dir / "baseline" / "delta" / f"obs{int(mzi_id)}_probe.txt"
        sigma_base = jacobian_dir / "baseline" / "sigma" / f"obs{int(mzi_id)}_inter_scan.txt"
        delta_scan = load_generic_scan(delta_base, ("probe_axis_power_w", "measured_power_w", "target_power_w"))
        sigma_scan = load_generic_scan(sigma_base, ("dp",))
        delta_init = get_fit_params(mzi_table, int(mzi_id), probe_arm)
        baseline_delta_fits[int(mzi_id)] = fit_sine_free_or_fixed(delta_scan["x"], delta_scan["y"], fix_w=delta_init["w"] if fix_w else None, init_params=delta_init, remove_one_outlier=True)
        baseline_sigma_fits[int(mzi_id)] = fit_sine_free_or_fixed(sigma_scan["x"], sigma_scan["y"], fix_w=None, remove_one_outlier=True)

    for heater in heaters:
        perturb_dir = jacobian_dir / f"perturb_{heater}"
        metadata_path = perturb_dir / "metadata.json"
        delta_power_w = np.nan
        try:
            delta_power_w, metadata = _read_delta_power(metadata_path)
        except Exception as exc:
            warning = f"{metadata_path}: {exc}"
            warnings.append(warning)
            metadata = {}
        for mzi_id in mzi_ids:
            probe_arm = probe_map.get(int(mzi_id), "u")
            col = f"P{heater}"
            try:
                pert_delta_scan = load_generic_scan(perturb_dir / "delta" / f"obs{int(mzi_id)}_probe.txt", ("probe_axis_power_w", "measured_power_w", "target_power_w"))
                base_fit = baseline_delta_fits[int(mzi_id)]
                pert_fit = fit_sine_free_or_fixed(pert_delta_scan["x"], pert_delta_scan["y"], fix_w=base_fit["w"] if fix_w else None, init_params=base_fit, remove_one_outlier=True)
                delta_eta = wrap_to_pi(pert_fit["phi"] - base_fit["phi"])
                delta_delta = delta_eta if probe_arm == "u" else -delta_eta
                j_value = delta_delta / delta_power_w
                j_delta.loc[f"Delta{int(mzi_id)}", col] = j_value
                delta_details.append(
                    {
                        "observed_mzi": int(mzi_id),
                        "probe_arm": probe_arm,
                        "perturbed_heater": heater,
                        "delta_power_w": delta_power_w,
                        "eta_base": base_fit["phi"],
                        "eta_pert": pert_fit["phi"],
                        "delta_eta_rad": delta_eta,
                        "delta_delta_rad": delta_delta,
                        "J_delta_rad_per_w": j_value,
                        "baseline_rmse_uW": base_fit["rmse_uW"],
                        "perturbed_rmse_uW": pert_fit["rmse_uW"],
                        "baseline_outlier_removed": bool(base_fit.get("outlier_removed", False)),
                        "baseline_excluded_index": base_fit.get("excluded_index", ""),
                        "baseline_full_rmse_before_outlier_removal_uW": base_fit.get("full_rmse_before_outlier_removal_uW", ""),
                        "perturbed_outlier_removed": bool(pert_fit.get("outlier_removed", False)),
                        "perturbed_excluded_index": pert_fit.get("excluded_index", ""),
                        "perturbed_full_rmse_before_outlier_removal_uW": pert_fit.get("full_rmse_before_outlier_removal_uW", ""),
                        "warning": "abs(delta_eta)>pi/2" if abs(delta_eta) > np.pi / 2 else "",
                    }
                )
                plot_fit_comparison(
                    jacobian_dir / "baseline" / "delta" / f"obs{int(mzi_id)}_probe.txt",
                    perturb_dir / "delta" / f"obs{int(mzi_id)}_probe.txt",
                    {"power_w": base_fit["x"], "optical_power_uW": base_fit["y"], "A": base_fit["A"], "w": base_fit["w"], "phi": base_fit["phi"], "b": base_fit["b"], "excluded_x": base_fit.get("excluded_x", []), "excluded_y": base_fit.get("excluded_y", [])},
                    {"power_w": pert_fit["x"], "optical_power_uW": pert_fit["y"], "A": pert_fit["A"], "w": pert_fit["w"], "phi": pert_fit["phi"], "b": pert_fit["b"], "excluded_x": pert_fit.get("excluded_x", []), "excluded_y": pert_fit.get("excluded_y", [])},
                    int(mzi_id),
                    probe_arm,
                    heater,
                    fit_dir / f"delta_obs{int(mzi_id)}_perturb_{heater}.png",
                )
            except Exception as exc:
                warning = f"delta obs{mzi_id} perturb {heater}: {exc}"
                warnings.append(warning)
                delta_details.append({"observed_mzi": int(mzi_id), "probe_arm": probe_arm, "perturbed_heater": heater, "delta_power_w": delta_power_w, "warning": warning})
            try:
                pert_sigma_scan = load_generic_scan(perturb_dir / "sigma" / f"obs{int(mzi_id)}_inter_scan.txt", ("dp",))
                base_fit = baseline_sigma_fits[int(mzi_id)]
                pert_fit = fit_sine_free_or_fixed(pert_sigma_scan["x"], pert_sigma_scan["y"], fix_w=base_fit["w"] if fix_w else None, init_params=base_fit, remove_one_outlier=True)
                delta_beta = wrap_to_pi(pert_fit["phi"] - base_fit["phi"])
                sigma_coeff = 2.0
                relative_link = sigma_coeff * delta_beta
                j_link_value = relative_link / delta_power_w
                j_sigma_link.loc[f"Sigma{int(mzi_id)}", col] = j_link_value
                sigma_details.append(
                    {
                        "observed_mzi": int(mzi_id),
                        "bmzi": int(sigma_bmzi_map.get(str(int(mzi_id)), 0)),
                        "perturbed_heater": heater,
                        "delta_power_w": delta_power_w,
                        "beta_base": base_fit["phi"],
                        "beta_pert": pert_fit["phi"],
                        "delta_beta": delta_beta,
                        "sigma_coeff": sigma_coeff,
                        "relative_sigma_link_rad": relative_link,
                        "J_sigma_link_rad_per_w": j_link_value,
                        "baseline_rmse_uW": base_fit["rmse_uW"],
                        "perturbed_rmse_uW": pert_fit["rmse_uW"],
                        "baseline_outlier_removed": bool(base_fit.get("outlier_removed", False)),
                        "baseline_excluded_index": base_fit.get("excluded_index", ""),
                        "baseline_full_rmse_before_outlier_removal_uW": base_fit.get("full_rmse_before_outlier_removal_uW", ""),
                        "perturbed_outlier_removed": bool(pert_fit.get("outlier_removed", False)),
                        "perturbed_excluded_index": pert_fit.get("excluded_index", ""),
                        "perturbed_full_rmse_before_outlier_removal_uW": pert_fit.get("full_rmse_before_outlier_removal_uW", ""),
                        "warning": "sigma_coeff default +2",
                    }
                )
                plot_fit_comparison(
                    jacobian_dir / "baseline" / "sigma" / f"obs{int(mzi_id)}_inter_scan.txt",
                    perturb_dir / "sigma" / f"obs{int(mzi_id)}_inter_scan.txt",
                    {"power_w": base_fit["x"], "optical_power_uW": base_fit["y"], "A": base_fit["A"], "w": base_fit["w"], "phi": base_fit["phi"], "b": base_fit["b"], "excluded_x": base_fit.get("excluded_x", []), "excluded_y": base_fit.get("excluded_y", [])},
                    {"power_w": pert_fit["x"], "optical_power_uW": pert_fit["y"], "A": pert_fit["A"], "w": pert_fit["w"], "phi": pert_fit["phi"], "b": pert_fit["b"], "excluded_x": pert_fit.get("excluded_x", []), "excluded_y": pert_fit.get("excluded_y", [])},
                    int(mzi_id),
                    "sigma",
                    heater,
                    fit_dir / f"sigma_obs{int(mzi_id)}_perturb_{heater}.png",
                )
            except Exception as exc:
                warning = f"sigma obs{mzi_id} perturb {heater}: {exc}"
                warnings.append(warning)
                sigma_details.append({"observed_mzi": int(mzi_id), "bmzi": int(sigma_bmzi_map.get(str(int(mzi_id)), 0)), "perturbed_heater": heater, "delta_power_w": delta_power_w, "warning": warning})

    j_sigma_global, chain_order, chain_valid = compute_global_sigma_matrix(j_sigma_link, sigma_bmzi_map, mzi_ids, sigma_reference_mode)
    sigma_details_df = pd.DataFrame(sigma_details)
    if not sigma_details_df.empty:
        for heater in heaters:
            col = f"P{heater}"
            matches = sigma_details_df["perturbed_heater"].astype(str) == str(heater)
            delta_power_values = pd.to_numeric(sigma_details_df.loc[matches, "delta_power_w"], errors="coerce")
            delta_power_w = float(delta_power_values[np.isfinite(delta_power_values)].iloc[0]) if np.isfinite(delta_power_values).any() else np.nan
            for mzi_id in mzi_ids:
                mask = matches & (sigma_details_df["observed_mzi"].astype(int) == int(mzi_id))
                if not mask.any():
                    continue
                bmzi = int(sigma_bmzi_map.get(str(int(mzi_id)), 0))
                bmzi_global = 0.0 if bmzi == 0 else float(j_sigma_global.loc[f"Sigma{bmzi}", col])
                global_value = float(j_sigma_global.loc[f"Sigma{int(mzi_id)}", col])
                sigma_details_df.loc[mask, "bmzi_global_delta_sigma_rad"] = bmzi_global * delta_power_w if np.isfinite(delta_power_w) else np.nan
                sigma_details_df.loc[mask, "global_delta_sigma_rad"] = global_value * delta_power_w if np.isfinite(delta_power_w) else np.nan
                sigma_details_df.loc[mask, "J_sigma_global_rad_per_w"] = global_value

    j_upper = pd.DataFrame(index=[f"theta{mzi_id}u" for mzi_id in mzi_ids], columns=cols, dtype=float)
    j_lower = pd.DataFrame(index=[f"theta{mzi_id}d" for mzi_id in mzi_ids], columns=cols, dtype=float)
    theta_rows = []
    theta_index = []
    for mzi_id in mzi_ids:
        sigma_row = j_sigma_global.loc[f"Sigma{int(mzi_id)}"].astype(float)
        delta_row = j_delta.loc[f"Delta{int(mzi_id)}"].astype(float)
        j_upper.loc[f"theta{int(mzi_id)}u"] = (sigma_row + delta_row) / 2.0
        j_lower.loc[f"theta{int(mzi_id)}d"] = (sigma_row - delta_row) / 2.0
        theta_rows.extend([j_upper.loc[f"theta{int(mzi_id)}u"], j_lower.loc[f"theta{int(mzi_id)}d"]])
        theta_index.extend([f"theta{int(mzi_id)}u", f"theta{int(mzi_id)}d"])
    j_theta = pd.DataFrame(theta_rows, index=theta_index, columns=cols)

    j_delta.to_csv(out_dir / "J_delta_rad_per_w.csv")
    (j_delta / 1000.0).to_csv(out_dir / "J_delta_rad_per_mw.csv")
    j_sigma_link.to_csv(out_dir / "J_sigma_link_rad_per_w.csv")
    (j_sigma_link / 1000.0).to_csv(out_dir / "J_sigma_link_rad_per_mw.csv")
    j_sigma_global.to_csv(out_dir / "J_sigma_global_rad_per_w.csv")
    (j_sigma_global / 1000.0).to_csv(out_dir / "J_sigma_global_rad_per_mw.csv")
    j_sigma_global.to_csv(out_dir / "J_sigma_rad_per_w.csv")
    (j_sigma_global / 1000.0).to_csv(out_dir / "J_sigma_rad_per_mw.csv")
    j_upper.to_csv(out_dir / "J_upper_rad_per_w.csv")
    j_lower.to_csv(out_dir / "J_lower_rad_per_w.csv")
    (j_upper / 1000.0).to_csv(out_dir / "J_upper_rad_per_mw.csv")
    (j_lower / 1000.0).to_csv(out_dir / "J_lower_rad_per_mw.csv")
    j_theta.to_csv(out_dir / "J_theta_rad_per_w.csv")
    (j_theta / 1000.0).to_csv(out_dir / "J_theta_rad_per_mw.csv")
    pd.DataFrame(delta_details).to_csv(out_dir / "delta_phase_shift_details.csv", index=False)
    sigma_details_df.to_csv(out_dir / "sigma_link_phase_shift_details.csv", index=False)
    plot_matrix_heatmap((j_delta / 1000.0).to_numpy(dtype=float), list(j_delta.index), cols, heatmap_dir / "J_delta_heatmap.png", "J_delta (rad/mW)")
    plot_matrix_heatmap((j_sigma_link / 1000.0).to_numpy(dtype=float), list(j_sigma_link.index), cols, heatmap_dir / "J_sigma_link_heatmap.png", "J_sigma_link (rad/mW)")
    plot_matrix_heatmap((j_sigma_global / 1000.0).to_numpy(dtype=float), list(j_sigma_global.index), cols, heatmap_dir / "J_sigma_global_heatmap.png", "J_sigma_global (rad/mW)")
    plot_matrix_heatmap((j_theta / 1000.0).to_numpy(dtype=float), list(j_theta.index), cols, heatmap_dir / "J_theta_heatmap.png", "J_theta (rad/mW)")
    summary = {
        "mzi_ids": [int(v) for v in mzi_ids],
        "heater_order": list(heaters),
        "probe_map": {str(k): v for k, v in probe_map.items()},
        "sigma_reference_mode": sigma_reference_mode,
        "sigma_bmzi_map": sigma_bmzi_map,
        "mzi_table_path": str(resolve_default_mzi_table(mzi_table_path)),
        "sigma_chain_order": chain_order,
        "sigma_chain_valid": bool(chain_valid),
        "J_delta_shape": list(j_delta.shape),
        "J_sigma_link_shape": list(j_sigma_link.shape),
        "J_sigma_global_shape": list(j_sigma_global.shape),
        "J_theta_shape": list(j_theta.shape),
        "J_sigma_definition": "global J_sigma relative to top straight waveguide, reconstructed from relative sigma links measured against bmzi",
        "relative_sigma_link_definition": "sigma inter scan measures delta_Sigma_i - delta_Sigma_bmzi",
        "J_theta_definition": "J_u=(J_sigma_global+J_delta)/2, J_d=(J_sigma_global-J_delta)/2",
        "J_delta_fit_definition": "delta probe fits use fixed heater phase frequency from MZI_table when fix_w=true",
        "warnings": warnings,
    }
    with (out_dir / "J_compute_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full J results to {out_dir}")
    return j_delta, j_sigma_link, j_sigma_global, j_theta


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
        description="Compute second-column Jacobian matrices from raw scans."
    )
    sub = parser.add_subparsers(dest="mode")

    p_delta = sub.add_parser("compute_delta")
    p_delta.add_argument("--mzi_table", default="MZI_table.json", help="Path to MZI_table.json.")
    p_delta.add_argument("--jacobian_dir", default="jacobian_measurements", help="Jacobian measurement directory.")
    p_delta.add_argument("--out_dir", default="results", help="Output directory.")
    p_delta.add_argument("--probe_map", default="", help="Example: 5:u,6:u,7:u,8:u")
    p_delta.add_argument("--fix_w", type=str_to_bool, default=True, help="true: fix w from MZI_table; false: fit w.")

    p_all = sub.add_parser("compute_all")
    p_all.add_argument("--jacobian_dir", required=True)
    p_all.add_argument("--out_dir", default="results/J_new")
    p_all.add_argument("--mzi_ids", default="5,6,7,8")
    p_all.add_argument("--heaters", default="5u,5d,6u,6d,7u,7d,8u,8d")
    p_all.add_argument("--probe_map", default="5:u,6:u,7:u,8:u")
    p_all.add_argument("--sigma_bmzi_map", default="5:0,6:5,7:6,8:7")
    p_all.add_argument("--sigma_reference_mode", default="chained_bmzi", choices=["chained_bmzi", "direct_reference"])
    p_all.add_argument("--fix_w", type=str_to_bool, default=True)
    p_all.add_argument("--mzi_table", default="Scandata/MZI_table.json")

    parser.add_argument("--mzi_table", default="MZI_table.json", help=argparse.SUPPRESS)
    parser.add_argument("--jacobian_dir", default="jacobian_measurements", help=argparse.SUPPRESS)
    parser.add_argument("--out_dir", default="results", help=argparse.SUPPRESS)
    parser.add_argument("--probe_map", default="", help=argparse.SUPPRESS)
    parser.add_argument("--fix_w", type=str_to_bool, default=True, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mode == "compute_all":
        mzi_ids = parse_csv_list(args.mzi_ids, int)
        heaters = parse_csv_list(args.heaters, str)
        compute_all(
            jacobian_dir=args.jacobian_dir,
            out_dir=args.out_dir,
            mzi_ids=mzi_ids,
            heaters=heaters,
            probe_map=parse_probe_map_for_ids(args.probe_map, mzi_ids),
            sigma_bmzi_map=parse_sigma_bmzi_map(args.sigma_bmzi_map, mzi_ids),
            sigma_reference_mode=args.sigma_reference_mode,
            fix_w=args.fix_w,
            mzi_table_path=args.mzi_table,
        )
    else:
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
