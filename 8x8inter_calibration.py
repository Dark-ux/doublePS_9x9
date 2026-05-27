import json
import utils.AllDecompositionUtils as du
import numpy as np
from pathlib import Path
import lumapi
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

try:
    # Prefer scipy for robust non-linear fitting if available
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


def load_mzi_table(path: str = "Scandata/mzi_stata_table.json") -> dict:
    table_path = Path(path)
    with table_path.open("r", encoding="utf-8") as f:
        raw_table = json.load(f)
    flattened = {str(k): [num for group in v for num in group] for k, v in raw_table.items()}
    return {k: flattened[k] for k in sorted(flattened.keys(), key=lambda x: float(x))}


def find_mzi_ER(
    index: int, er: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    path = Path(f"Scandata/MZI_{index}_scan_data.txt")
    data = np.loadtxt(path, skiprows=1)  # first row is header
    voltages = data[:, 0]
    power = data[:, 1]

    def sine_model(v, a, w, phi, b):
        return a * np.sin(w * v + phi) + b

    # Initial guesses
    amp_guess = 0.5 * (power.max() - power.min())
    offset_guess = float(power.mean())
    dv = np.median(np.diff(voltages)) if voltages.size > 1 else 1.0
    # Estimate dominant angular frequency using FFT (cycles per volt -> rad per volt)
    freq_guess = 1.0
    if voltages.size > 3:
        yf = np.fft.rfft(power - power.mean())
        xf = np.fft.rfftfreq(voltages.size, d=dv)
        if xf.size > 1:
            idx = np.argmax(np.abs(yf[1:])) + 1  # skip DC component
            freq_guess = float(xf[idx]) if idx < xf.size else freq_guess
    w_guess = 2 * np.pi * freq_guess
    phi_guess = 0.0
    p0 = [amp_guess if amp_guess != 0 else 1.0, w_guess, phi_guess, offset_guess]

    fitted_power = None
    if curve_fit is not None:
        try:
            params, _ = curve_fit(sine_model, voltages, power, p0=p0, maxfev=10000)
            fitted_power = sine_model(voltages, *params)
        except Exception:
            fitted_power = None

    # Fallback: linear least squares with fixed w_guess
    if fitted_power is None:
        w = w_guess
        sin_v = np.sin(w * voltages)
        cos_v = np.cos(w * voltages)
        ones = np.ones_like(voltages)
        A = np.column_stack((sin_v, cos_v, ones))
        coeffs, _, _, _ = np.linalg.lstsq(A, power, rcond=None)
        B, C, b_fit = coeffs
        a_fit = float(np.hypot(B, C))
        phi_fit = float(np.arctan2(C, B))
        fitted_power = sine_model(voltages, a_fit, w, phi_fit, b_fit)

    er_voltages = None
    if er is not None and fitted_power.size > 0:
        target = float(er) * float(fitted_power.max())
        diff = fitted_power - target
        crossings = []
        for i in range(diff.size - 1):
            if diff[i] == 0:
                crossings.append(voltages[i])
            if diff[i] * diff[i + 1] < 0:
                v0, v1 = voltages[i], voltages[i + 1]
                d0, d1 = diff[i], diff[i + 1]
                v_cross = v0 - d0 * (v1 - v0) / (d1 - d0)
                crossings.append(v_cross)
        if diff.size > 0 and diff[-1] == 0:
            crossings.append(voltages[-1])
        er_voltages = np.array(crossings)

    return voltages, power, fitted_power, er_voltages


def sweep_MZI_2arm(inc, num):
    inc.switchtodesign()
    MZI_name = f"MZI_{num}"
    inc.setnamed(MZI_name, f"path1", "sweep_signal.txt")
    inc.setnamed(MZI_name, f"path2", "sweep_signal.txt")
    inc.setnamed(MZI_name, f"amp1", 1)
    inc.setnamed(MZI_name, f"amp2", 1)


def set_MZI_V(inc, MZI_name, arm, V):
    inc.switchtodesign()
    inc.setnamed(MZI_name, "path" + str(arm), "1.txt")
    inc.setnamed(MZI_name, "amp" + str(arm), 0)
    inc.setnamed(MZI_name, "bios" + str(arm), V)
    return


def In_switch(inc, no, state):
    inc.switchtodesign()
    data = np.loadtxt("In_list.txt", delimiter="\t", skiprows=1)
    off_theta = data[no - 1, 1]
    on_theta = data[no - 1, 2]
    if state == "ON":
        inc.setnamed(f"In_{no}", "path1", "1.txt")
        inc.setnamed(f"In_{no}", "amp1", on_theta)
    elif state == "OFF":
        inc.setnamed(f"In_{no}", "path1", "1.txt")
        inc.setnamed(f"In_{no}", "amp1", off_theta)
    else:
        print("输入参数有误！")
    return


def find_Bmzi_path(target, N):
    M = du.Clements_matrix(N)
    PATH = np.array([])
    idx = np.where(M == target)
    cx, cy = idx[0][0], idx[1][0]
    bx, by = idx[0][0], idx[1][0]
    if idx[0][0] - 2 >= 0:
        bmzi = M[idx[0][0] - 2][idx[1][0]]
    else:
        bmzi = 0
    path = [(cx, cy)]
    while True:
        if cy == 0 and cx != 0:
            input = cx
            break
        if cx == 0 and cy == 0:
            input = 0
            break
        if cx > 0:
            cx -= 1
        if cy > 0:
            cy -= 1
        if M[cx][cy] != 0:
            path.insert(0, (cx, cy))
    while True:
        if by == N - 1 and bx != 0:
            ouput = bx
            break
        if bx == 0 and by == N - 1:
            ouput = 0
            break
        if bx > 0:
            bx -= 1
        if by < N - 1:
            by += 1
        if M[bx][by] != 0:
            path.append((bx, by))
    state = []
    for i in range(len(path)):
        if i == 0:
            if path[i][0] == path[i + 1][0]:
                state.append("B")
            else:
                state.append("C")
        elif i == len(path) - 1:
            if path[i][0] == path[i - 1][0]:
                state.append("B")
            else:
                state.append("C")
        else:
            if path[i - 1][0] == path[i + 1][0]:
                state.append("B")
            else:
                state.append("C")
    for item in path:
        PATH = np.append(PATH, M[item[0]][item[1]])
    state[idx[1][0] - 1] = "H"
    state[idx[1][0] + 1] = "H"

    return PATH, input, ouput, state, bmzi


def read_OOSC_data(inc, OOSC_name):
    r = inc.getresult(OOSC_name, "mode 1/signal")
    t = r["time"]
    p = r["TE amplitude at 193.1e+012 Hz"]
    t = t.reshape(
        -1,
    )
    p = p.reshape(
        -1,
    )
    t = np.delete(t, range(0, 32))
    p = np.delete(p, range(0, 32))
    power = np.average(np.abs(p) ** 2)
    return t, p, power


def get_dtheta(inc, out):
    name = "OOSC_" + str(out + 1)
    _, p, _ = read_OOSC_data(inc, name)
    a = np.abs(p[-1]) ** 2 / np.max(np.abs(p) ** 2)
    dtheta = np.arccos(1 - 2 * a)
    return dtheta


if __name__ == "__main__":
    N = 8
    target = 6
    Photonics_SoC_path = "lumSim\hk8x8.icp"
    inc = lumapi.INTERCONNECT(hide=False)
    inc.load(Photonics_SoC_path)
    inc.switchtodesign()

    mzi_table = load_mzi_table()

    for i in range(N):
        In_switch(inc, i + 1, "OFF")

    for i in range((N - 1) * N // 2):
        set_MZI_V(inc, f"MZI_{i+1}", 1, 0)
        set_MZI_V(inc, f"MZI_{i+1}", 2, 0)

    path, inp, out, state, bmzi = find_Bmzi_path(target, N)
    print(path)
    print(inp, out)
    print(state)
    print(bmzi)

    for i in range(len(path)):
        if state[i] == "B":
            set_MZI_V(inc, "MZI_" + str(int(path[i])), 1, mzi_table[str(int(path[i]))][0])
        elif state[i] == "C":
            set_MZI_V(inc, "MZI_" + str(int(path[i])), 1, mzi_table[str(int(path[i]))][1])
        elif state[i] == "H":
            _, _, _, v_er = find_mzi_ER(int(path[i]), er=0.5)
            set_MZI_V(inc, "MZI_" + str(int(path[i])), 1, v_er[0])

    set_MZI_V(inc, "MZI_" + str(int(bmzi)), 1, mzi_table[str(int(bmzi))][0])
    In_switch(inc, inp + 1, "ON")

    set_MZI_V(inc, "MZI_" + str(int(target)), 1, mzi_table[str(int(path[i]))][0] + 1)
    set_MZI_V(inc, "MZI_" + str(int(target)), 2, 1)
    sweep_MZI_2arm(inc, target)

    inc.run()
    dtheta = get_dtheta(inc, out)
    print("dtheta:", dtheta)
    os.system("pause")

    # # Quick visual check of the sine fit
    # idx = 1
    # v, p_raw, p_fit, v_er = find_mzi_ER(idx, er=1)
    # plt.figure()
    # plt.plot(v, p_raw, label="Measured", linewidth=1)
    # plt.plot(v, p_fit, label="Fit", linewidth=1)
    # if v_er is not None and v_er.size > 0:
    #     plt.scatter(v_er, np.full_like(v_er, 0.5 * p_fit.max()), color="red", s=20, label="P=0.5*Pmax")
    # plt.xlabel("V")
    # plt.ylabel("Power")
    # plt.title(f"MZI {idx} V-Power Fit")
    # plt.legend()
    # plt.show()
