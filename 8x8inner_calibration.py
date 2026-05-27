import os
import numpy as np
import lumapi
import matplotlib.pyplot as plt
from utils import AllDecompositionUtils as du
import json

np.random.seed(0)


def sweep_MZI(inc, MZI_name, mode):
    inc.switchtodesign()
    if mode == 1:
        inc.setnamed(MZI_name, "path1", "sweep_signal.txt")
        inc.setnamed(MZI_name, "amp1", 1)
        inc.setnamed(MZI_name, "bios1", 0)
    elif mode == 2:
        inc.setnamed(MZI_name, "path2", "sweep_signal.txt")
        inc.setnamed(MZI_name, "amp2", 1)
        inc.setnamed(MZI_name, "bios2", 0)
    return


def set_MZI_V(inc, MZI_name, arm, V):
    inc.switchtodesign()
    inc.setnamed(MZI_name, "path" + str(arm), "1.txt")
    inc.setnamed(MZI_name, "amp" + str(arm), 0)
    inc.setnamed(MZI_name, "bios" + str(arm), V)
    return


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


def get_OOSC_max_min_power(inc, OOSC_name):
    _, p, _ = read_OOSC_data(inc, OOSC_name)
    idx_max = np.where(np.abs(p) ** 2 == np.max(np.abs(p) ** 2))[0]
    max_theta = (idx_max + 32) / (len(p) + 32) * 2 * np.pi
    idx_min = np.where(np.abs(p) ** 2 == np.min(np.abs(p) ** 2))[0]
    min_theta = (idx_min + 32) / (len(p) + 32) * 2 * np.pi
    return max_theta, min_theta


def save_scan_data(inc, MZI_name, OOSC_name):
    t, p, _ = read_OOSC_data(inc, OOSC_name)
    max, min = get_OOSC_max_min_power(inc, OOSC_name)
    t = t / 1.024e-8 * 2 * np.pi
    plt.figure()
    plt.plot(t, np.abs(p) ** 2)
    plt.savefig(f"Scandata/{MZI_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    data = np.column_stack((t, np.abs(p) ** 2))
    np.savetxt(f"Scandata/{MZI_name}_scan_data.txt", data, delimiter="\t", header="V\tPower", comments="")
    return max, min


def get_cali_order(N):
    M = du.Clements_matrix(N)
    order1 = np.array([])
    order2 = np.array([])
    for i in range(N):
        if i == 0:
            for item in M[0][M[0] != 0]:
                order1 = np.append(order1, item)
            M = np.delete(M, 0, axis=0)
        else:
            for m in range(M.shape[0]):
                for n in range(M.shape[1]):
                    if m - n == (2 * i - (N + 1) if N % 2 == 0 else 2 * i - N):
                        order2 = np.append(order2, M[m][n])
    return order1, order2


def find_path(target, N):
    M = du.Clements_matrix(N)
    PATH = np.array([])
    idx = np.where(M == target)
    cx, cy = idx[0][0], idx[1][0]
    bx, by = idx[0][0], idx[1][0]
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
    return PATH, input, ouput, state


if __name__ == "__main__":
    Photonics_SoC_path = "lumSim\hk8x8.icp"
    inc = lumapi.INTERCONNECT(hide=False)
    inc.load(Photonics_SoC_path)
    inc.switchtodesign()

    mu = 1.255406805
    sigma_squared = 5.924169417e-02
    c = 299792458
    frequency = 193.1e12
    n = 2.6

    N = 8
    sum_power = 0

    cm = du.Clements_matrix(N)

    order1, order2 = get_cali_order(N)
    for i in range(N * (N - 1) // 2):
        # theta1 = np.random.normal(mu, np.sqrt(sigma_squared))
        # theta2 = np.random.normal(mu, np.sqrt(sigma_squared))
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        dl1 = theta1 * c / (2 * np.pi * frequency * n)
        dl2 = theta2 * c / (2 * np.pi * frequency * n)
        mzi_name = f"MZI_{i+1}"
        inc.setnamed(mzi_name, "arm1", 0.0002 + dl1)
        inc.setnamed(mzi_name, "arm2", 0.0002 + dl2)
        set_MZI_V(inc, mzi_name, 1, 0)
        set_MZI_V(inc, mzi_name, 2, 0)
        # inc.setnamed(mzi_name, "path2", "1.txt")
        # inc.setnamed(mzi_name, "amp2", 0)

    # for i in range(N):
    #     inc.setnamed(f"In_{i+1}", "path1", "sweep_signal.txt")
    #     inc.setnamed(f"In_{i+1}", "amp1", 1)
    # inc.run()
    # for i in range(N):
    #     max, min = get_OOSC_max_min_power(inc, f"OOSC_{i+9}")
    #     print(f"IN_{i+1} max_theta: {max}, min_theta: {min}")

    for i in range(N):
        In_switch(inc, i + 1, "OFF")
    In_switch(inc, 1, "ON")

    if N % 2 == 0:
        while 1 - sum_power > 0.01:
            for i in range(int(N / 2)):
                mzi_name = f"MZI_{7*i+1}"
                sweep_MZI(inc, mzi_name, 1)
                inc.run()
                max_theta, min_theta = get_OOSC_max_min_power(inc, "OOSC_1")
                set_MZI_V(inc, mzi_name, 1, max_theta)
            inc.run()
            t, p, sum_power = read_OOSC_data(inc, "OOSC_1")
            print(f"sum_power: {sum_power}")
            # plt.plot(t, np.abs(p) ** 2)
            # plt.show()
            for i in cm[1]:
                if i != 0:
                    mzi_name = f"MZI_{int(i)}"
                    set_MZI_V(inc, mzi_name, 1, np.random.uniform(-np.pi, np.pi))

    mzi_stata_table = {}

    for target in order1:
        mzi_name = f"MZI_{int(target)}"
        sweep_MZI(inc, mzi_name, 1)
        inc.run()
        OOSC_name = f"OOSC_1"
        max, min = save_scan_data(inc, mzi_name, OOSC_name)
        mzi_stata_table[int(target)] = (max, min)
        set_MZI_V(inc, mzi_name, 1, max)

    for target in order2:
        path, input, output, state = find_path(target, N)
        print("Target MZI:", target)
        print("Path:", path)
        print("Input:", input)
        print("Output:", output)
        print("State:", state, end="\n")
        for i in range(N):
            In_switch(inc, i + 1, "OFF")
        In_switch(inc, input + 1, "ON")
        for i in range(len(path)):
            mzi_name = f"MZI_{int(path[i])}"
            if path[i] != target:
                if state[i] == "B":
                    set_MZI_V(inc, mzi_name, 1, mzi_stata_table[path[i]][0])
                elif state[i] == "C":
                    set_MZI_V(inc, mzi_name, 1, mzi_stata_table[path[i]][1])
            else:
                sweep_MZI(inc, mzi_name, 1)
        inc.run()
        max, min = save_scan_data(inc, f"MZI_{int(target)}", f"OOSC_{output + 1}")
        mzi_stata_table[int(target)] = (max, min)
        set_MZI_V(inc, f"MZI_{int(target)}", 1, max)

    serializable_dict = {}
    for k, (v1, v2) in mzi_stata_table.items():
        serializable_dict[str(k)] = [v1.tolist(), v2.tolist()]

    with open("Scandata/mzi_stata_table.json", "w", encoding="utf-8") as f:
        json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
    print(mzi_stata_table)
    os.system("pause")
    inc.save()
