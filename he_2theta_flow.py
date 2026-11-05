"""HE extension and 2-theta MZI network verification flow.

Flow:
1. Generate a random m x n nonunitary contraction matrix T.
2. Extend T to a unitary matrix with the HE/C_extension method.
3. Decompose the unitary, then map each internal MZI to theta1/theta2 arms.
4. Recover the network transfer matrix from the theta1/theta2 network.
5. Compare the recovered top-left block with the original nonunitary T.

The 2-theta mapping here follows the notebook implementation in ERAM-2yxy.ipynb.
MZI positions are read from Dutils.Clements_matrix, so the same logic supports
both even- and odd-dimensional Clements meshes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import AllDecompositionUtils as Dutils


DEFAULT_VERIFY_CASES = ((2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4))


@dataclass
class TwoThetaNetwork:
    size: int
    input_phases: np.ndarray
    theta_pairs: np.ndarray
    mzi_locations: list[tuple[int, int, int, int]]
    top_straight_phases: list[tuple[int, float]]
    output_phases: np.ndarray


@dataclass
class FlowResult:
    rows: int
    cols: int
    original: np.ndarray
    extended: np.ndarray
    recovered: np.ndarray
    network: TwoThetaNetwork


def generate_random_nonunitary_matrix(
    rows: int,
    cols: int,
    seed: int = 1,
    margin: float = 0.2,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0, 1, (rows, cols)) + 1j * rng.uniform(0, 1, (rows, cols))
    max_singular = np.linalg.svd(matrix, compute_uv=False)[0]
    return matrix / (max_singular + margin)


def embed_2x2_zero_based(small_matrix: np.ndarray, index: int, size: int) -> np.ndarray:
    result = np.eye(size, dtype=complex)
    result[index, index] = small_matrix[0, 0]
    result[index, index + 1] = small_matrix[0, 1]
    result[index + 1, index] = small_matrix[1, 0]
    result[index + 1, index + 1] = small_matrix[1, 1]
    return result


def mzi_2theta(theta_1: float, theta_2: float) -> np.ndarray:
    delta = theta_1 - theta_2
    matrix = np.array(
        [
            [np.sin(delta / 2), np.cos(delta / 2)],
            [np.cos(delta / 2), -np.sin(delta / 2)],
        ],
        dtype=complex,
    )
    return 1j * np.exp(1j * (theta_1 + theta_2) / 2) * matrix


@lru_cache(maxsize=None)
def clements_numbering(size: int) -> np.ndarray:
    return Dutils.Clements_matrix(size)


def column_mzi_count(col: int, size: int) -> int:
    return size // 2 if col % 2 == 0 else (size - 1) // 2


def column_mode_start(row: int, col: int) -> int:
    return 2 * row if col % 2 == 0 else 2 * row + 1


def mzi_index(row: int, col: int, size: int) -> int:
    if not 0 <= col < size:
        raise ValueError(f"Column {col} is outside the Clements mesh size {size}.")
    if not 0 <= row < column_mzi_count(col, size):
        raise ValueError(f"Row {row} is outside column {col} for mesh size {size}.")

    mode = column_mode_start(row, col)
    param_number = int(clements_numbering(size)[mode, col])
    if param_number <= 0:
        raise ValueError(
            f"No MZI parameter at row {row}, column {col}, mode {mode}, size {size}."
        )
    return param_number - 1


def theta2_from_phi_after(row: int, col: int, params_relist: np.ndarray, size: int) -> float:
    phis = params_relist[:, 0]
    if col >= size - 1:
        return 0.0

    next_col = col + 1
    start_row = row if col % 2 == 0 else row + 1
    return float(
        sum(
            phis[mzi_index(next_row, next_col, size)]
            for next_row in range(start_row, column_mzi_count(next_col, size))
        )
    )


def theta2_from_phi_same(row: int, col: int, params_relist: np.ndarray, size: int) -> float:
    phis = params_relist[:, 0]
    if col == 0:
        return 0.0
    return float(
        -sum(
            phis[mzi_index(same_row, col, size)]
            for same_row in range(row + 1, column_mzi_count(col, size))
        )
    )


def top_straight_phase(col: int, params_relist: np.ndarray, size: int) -> float:
    phis = params_relist[:, 0]
    phase = -sum(
        phis[mzi_index(row, col, size)]
        for row in range(column_mzi_count(col, size))
    )
    if col < size - 1:
        phase += sum(
            phis[mzi_index(row, col + 1, size)]
            for row in range(column_mzi_count(col + 1, size))
        )
    return float(phase)


def build_2theta_network(D: np.ndarray, params_relist: np.ndarray, size: int) -> TwoThetaNetwork:
    phis = params_relist[:, 0]
    splitting_thetas = params_relist[:, 1]

    input_phases = np.array(
        [phis[mzi_index(row, 0, size)] for row in range(column_mzi_count(0, size))],
        dtype=float,
    )
    theta_pairs: list[list[float]] = []
    locations: list[tuple[int, int, int, int]] = []
    straight_phases: list[tuple[int, float]] = []

    for col in range(size):
        for row in range(column_mzi_count(col, size)):
            theta_2 = theta2_from_phi_after(row, col, params_relist, size)
            theta_2 += theta2_from_phi_same(row, col, params_relist, size)
            theta_1 = theta_2 + splitting_thetas[mzi_index(row, col, size)]
            mode_a = column_mode_start(row, col)
            theta_pairs.append([theta_1, theta_2])
            locations.append((col, row, mode_a, mode_a + 1))
        if col % 2 == 1:
            straight_phases.append((col, top_straight_phase(col, params_relist, size)))

    return TwoThetaNetwork(
        size=size,
        input_phases=input_phases,
        theta_pairs=np.array(theta_pairs, dtype=float),
        mzi_locations=locations,
        top_straight_phases=straight_phases,
        output_phases=np.angle(np.diag(D)),
    )


def recover_2theta_network(network: TwoThetaNetwork) -> np.ndarray:
    size = network.size
    recovered = np.eye(size, dtype=complex)

    for row, phase in enumerate(network.input_phases):
        phase_layer = np.array([[np.exp(1j * phase), 0], [0, 1]], dtype=complex)
        recovered = (
            embed_2x2_zero_based(phase_layer, column_mode_start(row, 0), size)
            @ recovered
        )

    pair_index = 0
    straight_index = 0
    for col in range(size):
        for row in range(column_mzi_count(col, size)):
            theta_1, theta_2 = network.theta_pairs[pair_index]
            pair_index += 1
            recovered = embed_2x2_zero_based(
                mzi_2theta(theta_1, theta_2), column_mode_start(row, col), size
            ) @ recovered
        if col % 2 == 1:
            _, phase = network.top_straight_phases[straight_index]
            straight_index += 1
            recovered[0, :] *= np.exp(1j * phase)

    output_phase_layer = np.diag(np.exp(1j * network.output_phases))
    return output_phase_layer @ recovered


def print_theta_pairs(network: TwoThetaNetwork, max_rows: int) -> None:
    print("\n2theta MZI arm parameters:")
    print("idx,col,row,mode_a,mode_b,theta1,theta2")
    row_count = min(max_rows, len(network.theta_pairs))
    for idx in range(row_count):
        col, row, mode_a, mode_b = network.mzi_locations[idx]
        theta_1, theta_2 = network.theta_pairs[idx]
        print(
            f"{idx},{col},{row},{mode_a},{mode_b},"
            f"{theta_1:.16g},{theta_2:.16g}"
        )
    if row_count < len(network.theta_pairs):
        remaining = len(network.theta_pairs) - row_count
        print(f"... {remaining} more MZI rows omitted")


def compute_flow(rows: int, cols: int, seed: int) -> FlowResult:
    if rows + cols <= 1:
        raise ValueError("rows + cols must be greater than 1.")

    original = generate_random_nonunitary_matrix(rows, cols, seed=seed)
    extended = Dutils.C_extension(original)
    size = extended.shape[0]

    params, D = Dutils.clements_decomposition(extended)
    params, D = Dutils.Phi_correction(params, D)
    params_relist = Dutils.C_Sequence_calibration(params)

    network = build_2theta_network(D, params_relist, size)
    recovered = recover_2theta_network(network)

    return FlowResult(
        rows=rows,
        cols=cols,
        original=original,
        extended=extended,
        recovered=recovered,
        network=network,
    )


def print_flow_summary(result: FlowResult, max_print: int) -> None:
    rows = result.rows
    cols = result.cols
    original = result.original
    extended = result.extended
    recovered = result.recovered
    network = result.network
    recovered_block = recovered[:rows, :cols]
    size = extended.shape[0]

    print(f"original nonunitary shape: {original.shape}")
    print(f"HE unitary shape: {extended.shape}")
    print(f"max singular value of original: {np.linalg.svd(original, compute_uv=False)[0]:.16g}")
    print(f"HE unitarity Frobenius error: {np.linalg.norm(extended @ extended.conj().T - np.eye(size)):.3e}")
    print(f"2theta MZI count: {len(network.theta_pairs)}")

    print_theta_pairs(network, max_print)

    print("\ncomparison:")
    print(f"full HE matrix Frobenius error: {np.linalg.norm(recovered - extended):.3e}")
    print(f"top-left block Frobenius error: {np.linalg.norm(recovered_block - original):.3e}")
    print(f"top-left block max abs error: {np.max(np.abs(recovered_block - original)):.3e}")
    print(f"top-left block allclose: {np.allclose(recovered_block, original, atol=1e-10)}")


def run_flow(rows: int, cols: int, seed: int, max_print: int) -> None:
    print_flow_summary(compute_flow(rows, cols, seed), max_print)


def print_verification_case(result: FlowResult) -> None:
    rows = result.rows
    cols = result.cols
    original = result.original
    extended = result.extended
    recovered = result.recovered
    recovered_block = recovered[:rows, :cols]

    full_error = recovered - extended
    block_error = recovered_block - original
    full_power_error = np.abs(recovered) ** 2 - np.abs(extended) ** 2
    block_power_error = np.abs(recovered_block) ** 2 - np.abs(original) ** 2

    print("\n" + "=" * 88)
    print(f"verification case: {rows}x{cols} nonunitary -> {extended.shape[0]}x{extended.shape[1]} HE unitary")
    print(f"2theta MZI count: {len(result.network.theta_pairs)}")
    print("\noriginal nonunitary matrix:")
    print(original)
    print("\ntheoretical HE unitary matrix:")
    print(extended)
    print("\n2theta recovered unitary matrix:")
    print(recovered)
    print("\nerror metrics against theory:")
    print(f"full complex Frobenius error: {np.linalg.norm(full_error):.3e}")
    print(f"full complex max abs error: {np.max(np.abs(full_error)):.3e}")
    print(f"full power Frobenius error: {np.linalg.norm(full_power_error):.3e}")
    print(f"top-left complex Frobenius error: {np.linalg.norm(block_error):.3e}")
    print(f"top-left complex max abs error: {np.max(np.abs(block_error)):.3e}")
    print(f"top-left power Frobenius error: {np.linalg.norm(block_power_error):.3e}")
    print(f"top-left allclose: {np.allclose(recovered_block, original, atol=1e-10)}")


def run_verification_cases(seed: int) -> None:
    with np.printoptions(linewidth=220, precision=6, suppress=True):
        print(f"batch verification seed: {seed}")
        print(f"cases: {DEFAULT_VERIFY_CASES}")
        for rows, cols in DEFAULT_VERIFY_CASES:
            print_verification_case(compute_flow(rows, cols, seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3, help="Rows of the random nonunitary matrix.")
    parser.add_argument("--cols", type=int, default=5, help="Columns of the random nonunitary matrix.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--verify-cases",
        action="store_true",
        help="Run batch verification for 2x2, 2x3, 2x4, 2x5, 3x3, and 3x4.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=40,
        help="Maximum number of theta1/theta2 rows to print.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_cases:
        run_verification_cases(args.seed)
    else:
        run_flow(args.rows, args.cols, args.seed, args.max_print)
