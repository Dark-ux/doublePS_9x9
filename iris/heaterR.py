import csv
import math
from pathlib import Path

import pandas as pd


def compute_resistance_from_vi(file_path: str) -> float:
    """
    Compute average resistance R from a V-I dataset.
    powerdata/41.txt format (based on sample):
      header: v,pow(W),current(mA)
      rows: V (volts), P (watts), I_mA (milliamps)
    We will compute per-point R = V / I (I in amps), filter invalid values, and take the mean.

    Args:
        file_path: path to the txt/CSV file

    Returns:
        float: average resistance in ohms
    """
    volts = []
    currents_a = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Detect delimiter: powerdata files appear comma-separated
        # Skip header
        header = next(reader, None)

        for row in reader:
            if not row:
                continue
            # be robust to extra whitespace
            row = [c.strip() for c in row]
            # Expect 3 columns: v, pow(W), current(mA)
            try:
                v = float(row[0])
                # row[1] is power, not needed for R
                i_mA = float(row[2])
            except (ValueError, IndexError):
                continue

            i_a = i_mA / 1000.0  # convert mA to A
            volts.append(v)
            currents_a.append(i_a)

    # Compute R per point; avoid division by near-zero
    Rs = []
    for v, i in zip(volts, currents_a):
        if abs(i) < 1e-9:
            continue
        Rs.append(v / i)

    if not Rs:
        raise ValueError("No valid resistance points computed (all currents ~0).")

    # Simple robust filtering: remove obvious outliers via IQR
    Rs_sorted = sorted(Rs)
    n = len(Rs_sorted)
    q1_idx = max(0, int(0.25 * (n - 1)))
    q3_idx = min(n - 1, int(0.75 * (n - 1)))
    q1 = Rs_sorted[q1_idx]
    q3 = Rs_sorted[q3_idx]
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    Rs_filtered = [r for r in Rs_sorted if low <= r <= high] or Rs

    return float(sum(Rs_filtered) / len(Rs_filtered))


def update_fit_curve_csv(csv_path: str, port_value: int, R_value: float):
    """
    Update iris/fit_curve_data.csv by adding/replacing a column 'R'.
    - If a row with port == port_value exists, set its R to R_value.
    - Otherwise, append a new row with port=port_value and R=R_value, leaving other columns NaN.

    Args:
        csv_path: path to CSV file
        port_value: the port index (e.g., 41)
        R_value: computed resistance
    """
    p = Path(csv_path)
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p, encoding="utf-8")
    else:
        df = pd.DataFrame(columns=["port"])

    # ensure 'port' column exists
    if "port" not in df.columns:
        df.insert(0, "port", pd.NA)

    # ensure 'R' column exists
    if "R" not in df.columns:
        df["R"] = pd.NA

    # find or create row for the given port
    mask = df["port"] == port_value
    if mask.any():
        df.loc[mask, "R"] = R_value
    else:
        # append a new row with only port and R set; other columns will be NaN
        new_row = {col: pd.NA for col in df.columns}
        new_row["port"] = port_value
        new_row["R"] = R_value
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save CSV
    df.to_csv(p, index=False, encoding="utf-8")


def main():
    data_file = Path("powerdata/49.txt")
    csv_out = Path("iris/fit_curve_data.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    R = compute_resistance_from_vi(str(data_file))
    # Update CSV at port 41
    update_fit_curve_csv(str(csv_out), port_value=49, R_value=R)


if __name__ == "__main__":
    main()
