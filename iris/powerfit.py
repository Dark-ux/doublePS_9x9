import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path
import re


def estimate_initial_frequency(x, y):
    """
    估计角频率 w 的初值和上界：
    - 利用峰值/谷值数量估计周期数
    - w0 取 2π / 估计周期
    - w_max 控制不超过一个合理倍数，避免过拟合的高频
    """
    # 按 x 排序
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    span = float(xs.max() - xs.min())
    if span <= 0:
        return 10.0, np.inf  # 退化情况

    # 归一化去趋势，提升峰谷检测稳定性
    y_norm = ys - np.median(ys)

    # 找峰和谷（对 -y 找峰即谷）
    prom = 0.05 * (y_norm.max() - y_norm.min())
    peaks_pos, _ = find_peaks(y_norm, prominence=prom if prom > 0 else None)
    peaks_neg, _ = find_peaks(-y_norm, prominence=prom if prom > 0 else None)

    # 使用峰/谷的相邻距离估计半周期，再推周期
    periods = []
    for idxs in [peaks_pos, peaks_neg]:
        if len(idxs) >= 2:
            dx = np.diff(xs[idxs])
            # 半周期 ~ 相邻峰距或相邻谷距
            halfT = np.median(dx)
            if halfT > 0:
                periods.append(2.0 * halfT)

    if len(periods) > 0:
        T_est = float(np.median(periods))
    else:
        # 回退：假设约 1.5 个周期覆盖 span，则 T ~ span / 1.5
        T_est = span / 1.5

    # 初值与上界
    w0 = 2.0 * np.pi / T_est if T_est > 0 else 10.0
    # 上界：允许最多 ~ 3 个周期跨越全区间（比 1.5 更宽容，但防止过大）
    w_max = 2.0 * np.pi * 3.0 / span
    return w0, w_max


def sin_model(x, A, w, phi, C):
    return A * np.sin(w * x + phi) + C


def extract_port_number(file_path: Path) -> int:
    m = re.search(r"(\d+)", file_path.stem)
    return int(m.group(1)) if m else -1


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    # 读取数据
    path = Path("powerdata") / "49.txt"
    df = pd.read_csv(path)

    V = df["v"].to_numpy(float)
    P_W = df["pow(W)"].to_numpy(float)
    I_mA = df["current(mA)"].to_numpy(float)

    # 坐标变换
    x = V * I_mA  # V·mA
    y = P_W * 1e3  # mW

    # 排序便于可视化与频率估计
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    # 初值：振幅/偏置
    A0 = 0.5 * (ys.max() - ys.min())
    C0 = ys.mean()

    # 估计频率初值和上界
    w0, w_max = estimate_initial_frequency(xs, ys)

    # 初值相位：用最小二乘线性化粗估
    # y ≈ A*cos(w0*x) + B*sin(w0*x) + C
    Xlin = np.column_stack([np.cos(w0 * xs), np.sin(w0 * xs), np.ones_like(xs)])
    coeffs, _, _, _ = np.linalg.lstsq(Xlin, ys, rcond=None)
    Ac, Bs, C_lin = coeffs
    A_lin = np.hypot(Ac, Bs)
    phi_lin = np.arctan2(Bs, Ac)  # cos(w0x)*Ac + sin(w0x)*Bs = A*cos(w0x - phi'); 相位等价
    # 将线性估计作为初值，但保证 A 非负
    A0 = max(A0, A_lin)
    C0 = C_lin
    phi0 = -phi_lin  # 与上式等价的相位换元

    # 边界：A>=0, 0<=w<=w_max, phi∈[-pi, pi], C 任意
    bounds = ([0.0, 0.0, -np.pi, -np.inf], [np.inf, w_max, np.pi, np.inf])

    # 拟合
    p0 = [A0, w0, phi0, C0]
    popt, pcov = curve_fit(sin_model, x, y, p0=p0, bounds=bounds, maxfev=300000)
    A_fit, w_fit, phi_fit, C_fit = [float(v) for v in popt]

    # 生成平滑曲线
    x_line = np.linspace(xs.min(), xs.max(), 1000)
    y_fit = sin_model(x_line, *popt)

    # 保存图像到 iris/fit_curve_figure/fit_curve_41.png
    figure_dir = Path("iris") / "fit_curve_figure"
    ensure_parent_dir(figure_dir / "dummy.txt")
    port_num = extract_port_number(path)
    fig_path = figure_dir / f"fit_curve_{port_num}.png"

    plt.figure(figsize=(7.5, 5.0), dpi=120)
    plt.scatter(x, y, s=18, c="#1f77b4", label="Measured (P vs V·mA)")
    plt.plot(xs, ys, color="#1f77b4", alpha=0.35, linewidth=1, label="Measured (line)")
    plt.plot(
        x_line, y_fit, color="#d62728", linewidth=2.0, label=f"Sine fit: A={A_fit:.3e} mW, w={w_fit:.3e} per (V·mA)"
    )
    plt.xlabel("V·mA")
    plt.ylabel("Power (mW)")
    plt.title("Power vs V·mA with Sine Fit (~1.5 periods)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Figure saved to: {fig_path.resolve()}")

    # 保存参数到 iris/fit_curve_data.csv，列名 port,A,w,phi,C
    csv_path = Path("iris") / "fit_curve_data.csv"
    ensure_parent_dir(csv_path)
    header = ["port", "A", "w", "phi", "C"]

    new_row = pd.DataFrame([{"port": port_num, "A": A_fit, "w": w_fit, "phi": phi_fit, "C": C_fit}], columns=header)

    if not csv_path.exists():
        new_row.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"CSV created and first row written: {csv_path.resolve()}")
    else:
        try:
            df_exist = pd.read_csv(csv_path)
            # 确保列齐全且顺序一致
            for col in header:
                if col not in df_exist.columns:
                    df_exist[col] = np.nan
            df_exist = df_exist[header]
            # 同一 port 去重
            df_exist = df_exist[df_exist["port"] != port_num]
            df_exist = pd.concat([df_exist, new_row], ignore_index=True)
            df_exist.to_csv(csv_path, index=False, encoding="utf-8-sig")
        except Exception:
            # 若读取失败，回退为直接追加
            new_row.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        print(f"Parameters appended to: {csv_path.resolve()}")

    # 控制台输出拟合参数（ASCII 名称）
    print("Fitted parameters on P (mW) vs x = V·mA:")
    print(f"A   = {A_fit:.6e} mW")
    print(f"w   = {w_fit:.6e} per (V·mA)")
    print(f"phi = {phi_fit:.6f} rad")
    print(f"C   = {C_fit:.6e} mW")


if __name__ == "__main__":
    main()
