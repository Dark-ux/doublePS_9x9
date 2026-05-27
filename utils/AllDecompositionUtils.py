import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.linalg import eigh
from mpmath import mp, matrix


def embed_2x2(small_matrix, i, j, size=4):
    # 创建单位矩阵
    result = np.eye(size, dtype=complex)

    # 嵌入2x2矩阵的元素
    result[i - 1, i - 1] = small_matrix[0, 0]  # T11
    result[i - 1, j - 1] = small_matrix[0, 1]  # T12
    result[j - 1, i - 1] = small_matrix[1, 0]  # T21
    result[j - 1, j - 1] = small_matrix[1, 1]  # T22

    return result


def reck_transfer_matrix(N, phi_list=None, theta_list=None):
    num_mzi = N * (N - 1) // 2

    # 独立处理phi_list和theta_list
    if phi_list is None:
        phi_list = np.random.uniform(0, 2 * np.pi, num_mzi)
    if theta_list is None:
        theta_list = np.random.uniform(0, np.pi, num_mzi)

    # 初始化单位矩阵
    U = np.eye(N, dtype=complex)

    # 构建Reck结构
    idx = 0
    for layer in range(N - 1):
        for mzi in range(layer // 2 + 1):
            i = N - layer + 2 * mzi - 2
            j = i + 1
            T = np.eye(N, dtype=complex)
            phi = phi_list[idx]
            theta = theta_list[idx]
            T[i, i] = np.exp(1j * phi) * (np.exp(1j * theta) - 1) / 2
            T[i, j] = 1j * (np.exp(1j * theta) + 1) / 2
            T[j, i] = np.exp(1j * phi) * 1j * (np.exp(1j * theta) + 1) / 2
            T[j, j] = -(np.exp(1j * theta) - 1) / 2
            U = T @ U
            idx += 1

    for layer in range(N - 2):
        for mzi in range((N - 1 - layer) // 2):
            i = layer + 1 + 2 * mzi
            j = i + 1
            T = np.eye(N, dtype=complex)
            phi = phi_list[idx]
            theta = theta_list[idx]
            T[i, i] = np.exp(1j * phi) * (np.exp(1j * theta) - 1) / 2
            T[i, j] = 1j * (np.exp(1j * theta) + 1) / 2
            T[j, i] = np.exp(1j * phi) * 1j * (np.exp(1j * theta) + 1) / 2
            T[j, j] = -(np.exp(1j * theta) - 1) / 2
            U = T @ U
            idx += 1

    return U, phi_list, theta_list


def reck_decompose(U):
    N = U.shape[0]
    u = U.copy()
    params = []
    for row in range(N - 1):
        for col in range(N - 1, row, -1):
            a = u[row][col - 1]
            b = u[row][col]
            if np.abs(b) < 1e-7:
                phi = np.pi
                theta = np.pi
            else:
                z = a / b
                phi = np.angle(z)
                theta = np.arctan(np.abs(z)) * 2
            params.append([phi, theta])
            T11 = np.exp(-1j * phi) * np.sin(theta / 2)
            T12 = np.exp(-1j * phi) * np.cos(theta / 2)
            T21 = np.cos(theta / 2)
            T22 = -np.sin(theta / 2)
            T = np.block([[T11, T12], [T21, T22]])
            T = -1j * np.exp(-1j * theta / 2) * T
            T = embed_2x2(T, col, col + 1, N)
            u = u @ T

    return params, u


def plot_complex_vectors(complex_numbers, figsize=(6, 6), color="b"):
    complex_numbers = np.array(complex_numbers)

    re = complex_numbers.real
    im = complex_numbers.imag

    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.quiver(
        np.zeros_like(re),
        np.zeros_like(im),
        re,
        im,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=0.005,
    )

    ax.scatter(re, im, color=color, s=50, zorder=5)

    theta = np.linspace(0, 2 * np.pi, 300)
    circle_re = np.cos(theta)
    circle_im = np.sin(theta)
    ax.plot(circle_re, circle_im, "k--", lw=1.5, label="Unit Circle")

    all_vals = np.concatenate((re, im, circle_re, circle_im))
    max_val = np.max(np.abs(all_vals)) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Complex Vectors")

    ax.legend()

    ax.set_aspect("equal", adjustable="box")
    plt.show()


def generate_order(N):
    result = []
    direction = -1  # 1表示从左下到右上，-1表示从右上到左下
    for k in range(N, 1, -1):
        points = []
        max_j = N - k + 1
        if direction == -1:
            i_start = N
            i_end = k
            j_start = max_j
            j_end = 1
            i_list = [index for index in range(i_start, i_end - 1, -1)]
            j_list = [index for index in range(j_start, j_end - 1, -1)]
            for i, j in zip(i_list, j_list):
                points.append((i, j, direction))
        elif direction == 1:
            i_start = k
            i_end = N
            j_start = 1
            j_end = max_j
            i_list = [index for index in range(i_start, i_end + 1)]
            j_list = [index for index in range(j_start, j_end + 1)]
            for i, j in zip(i_list, j_list):
                points.append((i, j, direction))
        direction *= -1
        result.extend(points)
    result.extend([(0, 0, 0)])
    return result


def clements_decomposition(U=np.array):
    if U.shape[0] != U.shape[1]:
        raise ValueError("输入矩阵U不是方阵")
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-6):
        raise ValueError("输入矩阵U不是酉矩阵")

    N = U.shape[0]
    u = U.copy()
    params = []
    order = generate_order(N)

    for i, j, direction in order:
        if direction == -1:
            a = u[i - 1, j - 1]
            b = u[i - 1, j]
            if np.abs(a) < 1e-7:
                params.append([np.pi, np.pi, 1])
                # print("\n", u)
            else:
                z = b / a
                phi = np.pi - np.angle(z)
                theta = np.arctan(np.abs(z)) * 2
                params.append([phi, theta, 0])
                T11 = np.exp(-1j * phi) * np.sin(theta / 2)
                T12 = np.exp(-1j * phi) * np.cos(theta / 2)
                T21 = np.cos(theta / 2)
                T22 = -np.sin(theta / 2)
                T = np.array([[T11, T12], [T21, T22]], dtype=complex)
                T = -1j * np.exp(-1j * theta / 2) * T
                T = embed_2x2(T, j, j + 1, size=N)
                u = u @ T
                # print("\n", u)
        elif direction == 1:
            a = u[i - 2, j - 1]
            b = u[i - 1, j - 1]
            if np.abs(b) < 1e-7:
                params.append([np.pi, np.pi, 1])
                # print("\n", u)
            else:
                z = a / b
                phi = np.mod(-np.angle(z), 2 * np.pi)
                theta = np.arctan(np.abs(z)) * 2
                params.append([phi, theta, 0])
                T11 = np.exp(1j * phi) * np.sin(theta / 2)
                T12 = np.cos(theta / 2)
                T21 = np.exp(1j * phi) * np.cos(theta / 2)
                T22 = -np.sin(theta / 2)
                T = np.array([[T11, T12], [T21, T22]], dtype=complex)
                T = 1j * np.exp(1j * theta / 2) * T
                T = embed_2x2(T, i - 1, i, size=N)
                u = T @ u
                # print("\n", u)

    return params, u


def Phi_correction(params, D):
    N = D.shape[0]
    order = generate_order(N)
    for i, j, direction in order[::-1]:
        index = order.index((i, j, direction))
        if direction == 1 and params[index][2] == 0:
            phi = params[index][0]
            theta = params[index][1]
            D1 = D[i - 2, i - 2]
            D2 = D[i - 1, i - 1]
            phi_prime = np.angle(D1 / D2)
            params[index][0] = phi_prime
            D1_prime = D2 * np.exp(-1j * phi) * -1 * np.exp(-1j * theta)
            D2_prime = D2 * -1 * np.exp(-1j * theta)
            D[i - 2, i - 2] = D1_prime
            D[i - 1, i - 1] = D2_prime
            # print(D)
    return params, D


def U_recovery(params, D):
    N = D.shape[0]
    order = generate_order(N)
    U = np.eye(N)
    for i, j, direction in order:
        if direction == 1:
            index = order.index((i, j, direction))
            phi = params[index][0]
            theta = params[index][1]
            T11 = np.exp(1j * phi) * np.sin(theta / 2)
            T12 = np.cos(theta / 2)
            T21 = np.exp(1j * phi) * np.cos(theta / 2)
            T22 = -np.sin(theta / 2)
            T = np.array([[T11, T12], [T21, T22]], dtype=complex)
            T = 1j * np.exp(1j * theta / 2) * T
            T = embed_2x2(T, i - 1, i, size=N)
            U = U @ T
    for i, j, direction in order[::-1]:
        if direction == -1:
            index = order.index((i, j, direction))
            phi = params[index][0]
            theta = params[index][1]
            T11 = np.exp(1j * phi) * np.sin(theta / 2)
            T12 = np.cos(theta / 2)
            T21 = np.exp(1j * phi) * np.cos(theta / 2)
            T22 = -np.sin(theta / 2)
            T = np.array([[T11, T12], [T21, T22]], dtype=complex)
            T = 1j * np.exp(1j * theta / 2) * T
            T = embed_2x2(T, j, j + 1, size=N)
            U = U @ T
    U = D @ U
    return U


class MZI:
    def __init__(self, phi=None, theta=None):
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi)
        if phi is None:
            phi = np.random.uniform(0, 2 * np.pi)
        self.theta = theta
        self.phi = phi

    def forward(self):
        T11 = np.exp(1j * self.phi) * (np.exp(1j * self.theta) - 1) / 2
        T12 = 1j * (np.exp(1j * self.theta) + 1) / 2
        T21 = T12 * np.exp(1j * self.phi)
        T22 = (1 - np.exp(1j * self.theta)) / 2
        T = np.array([[T11, T12], [T21, T22]], dtype=np.complex64)
        return T


def net_T(mzi_net, mzis_param):
    T = np.eye(len(mzi_net) + 1, dtype=complex)
    for j in range(mzi_net.shape[1]):
        C = np.eye(len(mzi_net) + 1, dtype=complex)
        for i in range(mzi_net.shape[0]):
            if mzi_net[i][j] != 0:
                # print(mzi_net[i][j], mzis_param[mzi_net[i][j] - 1])
                mzi = MZI(mzis_param[mzi_net[i][j] - 1][0], mzis_param[mzi_net[i][j] - 1][1])
                A = mzi.forward()
                B = embed_2x2(A, i + 1, i + 2, len(mzi_net) + 1)
                C = C @ B
        T = C @ T
    return T


def Clements_matrix(N):
    if N <= 0:
        raise ValueError("输入的 N 必须是一个大于 0 的正整数。")
    rows, cols = N - 1, N
    matrix = np.zeros((rows, cols), dtype=int)
    current_number = 1
    for col in range(cols):
        if col % 2 == 0:
            for row in range(rows):
                if row % 2 == 0:  # 奇数列从数字开始
                    matrix[row, col] = current_number
                    current_number += 1
        else:
            for row in range(rows):
                if row % 2 == 1:  # 偶数列从 0 开始
                    matrix[row, col] = current_number
                    current_number += 1

    return matrix


# 只适用于Clements结构，对分解完成的list重排列
def C_Sequence_calibration(input_list):
    length = len(input_list)
    N = int((1 + (1 + 8 * length) ** 0.5) / 2)
    if N * (N - 1) // 2 != length:
        raise ValueError("输入列表的长度不符合 N*(N-1)/2 的规则。")

    rows, cols = N - 1, N
    matrix = np.zeros((rows, cols), dtype=object)

    # 生成数字顺序矩阵
    current_number = 1
    indices = []
    for i in range(N - 1):
        if i % 2 == 0:  # 从左下到右上
            x = i
            y = 0
            for j in range(i + 1):
                indices.append((x, y))  # 记录数字对应的位置
                current_number += 1
                y += 1
                x -= 1
        else:  # 从右上到左下
            x = N - i - 2
            y = N - 1
            for j in range(i + 1):
                indices.append((x, y))  # 记录数字对应的位置
                current_number += 1
                y -= 1
                x += 1

    # 将列表中的值按顺序嵌入矩阵
    for idx, value in zip(indices, input_list):
        matrix[idx] = value

    result = []
    for col in range(cols):  # 遍历每一列
        if col % 2 == 0:
            for item in matrix[0::2, col]:
                result.append(item)
        else:
            for item in matrix[1::2, col]:
                result.append(item)
    return np.array(result)


def generate_complex_matrix(N, real_range=(0, 1), imag_range=(0, 1)):
    real_part = np.random.uniform(real_range[0], real_range[1], (N, N))
    imag_part = np.random.uniform(imag_range[0], imag_range[1], (N, N))
    complex_matrix = real_part + 1j * imag_part

    singular_values = np.linalg.svd(complex_matrix, compute_uv=False)

    if np.all(singular_values <= 1):
        return complex_matrix
    else:
        complex_matrix = complex_matrix / max(singular_values)
        return complex_matrix


def C_extension(T):
    m = T.shape[0]
    n = T.shape[1]
    T_CT = T.conj().T
    B = np.eye(n) - T_CT @ T
    C = np.eye(m) - T @ T_CT
    eigvals_B, eigvecs_B = eigh(B)
    eigvals_C, eigvecs_C = eigh(C)
    # print(eigvals_B, eigvals_C)
    sqrt_Lambda_B = np.diag(np.sqrt(np.abs(eigvals_B)))
    sqrt_Lambda_C = np.diag(np.sqrt(np.abs(eigvals_C)))
    sqrt_B = eigvecs_B @ sqrt_Lambda_B @ eigvecs_B.conj().T
    sqrt_C = eigvecs_C @ sqrt_Lambda_C @ eigvecs_C.conj().T
    U = np.block([[T, sqrt_C], [sqrt_B, -T_CT]])
    return U


def embed_2x2_mp(small_matrix, i, j, size=4):
    result = mp.eye(size)  # 高精度单位矩阵

    # 嵌入 small_matrix 的元素
    result[i - 1, i - 1] = small_matrix[0, 0]  # T11
    result[i - 1, j - 1] = small_matrix[0, 1]  # T12
    result[j - 1, i - 1] = small_matrix[1, 0]  # T21
    result[j - 1, j - 1] = small_matrix[1, 1]  # T22

    return result


def R_recovery(params, D):
    N = D.shape[0]
    U = np.eye(N)
    index = len(params) - 1
    position = N - 1
    for i in range(N - 1):
        for j in range(i + 1):
            phi = params[index][0]
            theta = params[index][1]
            T11 = np.exp(1j * phi) * (np.exp(1j * theta) - 1) / 2
            T12 = 1j * (np.exp(1j * theta) + 1) / 2
            T21 = np.exp(1j * phi) * 1j * (np.exp(1j * theta) + 1) / 2
            T22 = -(np.exp(1j * theta) - 1) / 2
            T = np.block([[T11, T12], [T21, T22]])
            T = embed_2x2(T, position + j, position + j + 1, N)
            U = U @ T
            index -= 1
        position -= 1

    U = D @ U
    return U


def Diamond_Decomposition(U):
    N = U.shape[0] / 2
    N = int(N)
    UU = np.copy(U)
    theta = np.zeros((N, N))
    phi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            theta[i, j] = 2 * np.arctan(np.abs(UU[i, N - j + i - 1]) / np.abs(UU[i, N - j + i]))
            phi[i, j] = np.angle(UU[i, N - j + i - 1]) - np.angle(UU[i, N - j + i])
            transform_matrix = np.complex128(np.eye(2 * N))
            transform_matrix[N - j + i - 1, N - j + i - 1] = np.exp(-1j * phi[i, j]) * (np.exp(-1j * theta[i, j]) - 1) / 2
            transform_matrix[N - j + i - 1, N - j + i] = np.exp(-1j * phi[i, j]) * -1j * (np.exp(-1j * theta[i, j]) + 1) / 2
            transform_matrix[N - j + i, N - j + i - 1] = -1j * (np.exp(-1j * theta[i, j]) + 1) / 2
            transform_matrix[N - j + i, N - j + i] = -(np.exp(-1j * theta[i, j]) - 1) / 2
            UU = UU @ transform_matrix
    return UU, phi, theta


def make_positive_definite(matrix, epsilon=1e-10):
    # 获取特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # 将特征值中小于 epsilon 的部分置为 epsilon
    eigenvalues[eigenvalues < epsilon] = epsilon
    # 重构矩阵
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T


def get_A_B_C(T):
    N = T.shape[0]
    A_prime = np.eye(N) - T @ T.conj().T
    B_prime = np.eye(N) - T.conj().T @ T
    A_prime = make_positive_definite(A_prime)
    B_prime = make_positive_definite(B_prime)
    A = cholesky(A_prime, lower=True)
    B = cholesky(B_prime, lower=False)
    C = -1 * B @ T.conj().T @ np.linalg.inv(A.conj().T)
    return A, B, C
