from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class FitParams:
    port: int
    A: float
    w: float
    phi: float
    C: float
    R: Optional[float]


def _load_params(csv_path: str | Path, port: int) -> FitParams:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"{p}")

    df = pd.read_csv(p, encoding="utf-8")
    if "port" not in df.columns:
        raise ValueError("missing 'port'")

    row = df.loc[df["port"] == port]
    if row.empty:
        raise ValueError(f"port={port} not found")

    row = row.iloc[0]
    for col in ["A", "w", "phi", "C"]:
        if col not in df.columns:
            raise ValueError(f"missing '{col}'")
    A = float(row["A"])
    w = float(row["w"])
    phi = float(row["phi"])
    C = float(row["C"])
    R = None
    if "R" in df.columns and not pd.isna(row.get("R", pd.NA)):
        R = float(row["R"])

    return FitParams(port=port, A=A, w=w, phi=phi, C=C, R=R)


def _find_first_increasing_solution_p_over_0_50(A: float, w: float, phi: float, C: float, rate: float) -> float:
    if not (-1.0 <= rate <= 1.0):
        raise ValueError("rate out of range")
    if w == 0 or A == 0:
        raise ValueError("A,w must be non-zero")

    two_pi = 2.0 * np.pi
    k_start = int(np.floor((phi - (-np.pi / 2)) / two_pi)) - 2
    found_interval = None
    for k in range(k_start, k_start + 10000):
        u_low = -np.pi / 2 + 2 * k * np.pi
        u_high = np.pi / 2 + 2 * k * np.pi
        p_low = (u_low - phi) / w
        p_high = (u_high - phi) / w
        p_min, p_max = (min(p_low, p_high), max(p_low, p_high))
        seg_min = max(p_min, 5.0)
        seg_max = min(p_max, 50.0)
        if seg_max > seg_min:
            found_interval = (seg_min, seg_max, k)
            break
    if found_interval is None:
        raise ValueError("no increasing interval")
    seg_min, seg_max, k_inc = found_interval

    u0 = np.arcsin((rate * (A + C) - C) / A)
    best_p = None
    for m in range(k_inc - 5, k_inc + 6):
        u = u0 + 2.0 * m * np.pi
        p_candidate = (u - phi) / w
        if seg_min < p_candidate < seg_max:
            if 5 < p_candidate < 50:
                if best_p is None or p_candidate < best_p:
                    best_p = p_candidate

    if best_p is None:
        for m in range(k_inc - 5, k_inc + 6):
            u = np.pi - u0 + 2.0 * m * np.pi
            p_candidate = (u - phi) / w
            if seg_min < p_candidate < seg_max and 0 < p_candidate < 50:
                if best_p is None or p_candidate < best_p:
                    best_p = p_candidate

    if best_p is None:
        raise ValueError("no solution in first increasing interval")

    return float(best_p)


def findrateV(port: int, rate: float, csv_path: str | Path = "iris/fit_curve_data.csv") -> float:
    params = _load_params(csv_path, port)
    if params.R is None or np.isnan(params.R):
        raise ValueError("R missing")

    p = _find_first_increasing_solution_p_over_0_50(params.A, params.w, params.phi, params.C, rate)
    V = np.sqrt(p / 1000 * params.R)
    return float(np.round(V, 3))


if __name__ == "__main__":
    v = findrateV(43, 0.15, "iris/fit_curve_data.csv")
    print(v)
