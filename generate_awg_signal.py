import numpy as np
from pathlib import Path
import json
import math

# =========================================================
# M8194A RZ / NRZ PAM-256 Ideal 8bit CSV Generator
# =========================================================

FS = 120e9  # AWG sample rate: 120 GSa/s
GRANULARITY = 128  # M8194A waveform granularity

baud_rates = [5e9, 10e9, 16e9, 20e9, 24e9, 30e9, 32e9, 40e9, 48e9, 64e9]

# =========================================================
# Symbol sequence:
# 50 zero header + 0→255→0
# =========================================================

header_len = 50

payload_levels = np.concatenate([np.arange(0, 256, dtype=int), np.arange(255, -1, -1, dtype=int)])

header_levels = np.zeros(header_len, dtype=int)

symbol_levels = np.concatenate([header_levels, payload_levels])

total_symbols = len(symbol_levels)

# 输出文件夹
output_dir = Path("M8194A_Ideal_RZ_NRZ_PAM256_8bit_120GSa")
output_dir.mkdir(exist_ok=True)

# 保存原始码元序列
np.savetxt(
    output_dir / "symbol_levels_562_50zero_then_0to255to0.csv",
    np.column_stack([np.arange(total_symbols), symbol_levels]),
    delimiter=",",
    fmt="%d",
    header="symbol_index,level_0_to_255",
    comments="",
)

np.savetxt(output_dir / "payload_levels_512_0to255to0.csv", payload_levels, delimiter=",", fmt="%d")

# =========================================================
# Utility functions
# =========================================================


def pad_to_granularity(x, granularity=128):
    """
    补零到 128 点整数倍。
    """

    n = len(x)
    n_total = int(math.ceil(n / granularity) * granularity)
    pad_len = n_total - n

    if pad_len > 0:
        x = np.concatenate([x, np.zeros(pad_len, dtype=int)])

    return x, pad_len


def generate_rz_8bit(symbol_levels, baud, fs, duty=0.5):
    """
    生成 ideal RZ PAM-256 8bit 波形。

    每个码元前 duty 部分输出当前 level；
    后 1-duty 部分回到 0。

    输出范围：0~255 整数。
    """

    total_time = len(symbol_levels) / baud
    n_useful = int(np.ceil(total_time * fs))

    n = np.arange(n_useful)
    t = n / fs

    u = t * baud

    symbol_index = np.floor(u).astype(int)
    symbol_index = np.clip(symbol_index, 0, len(symbol_levels) - 1)

    phase = u - symbol_index

    waveform = np.where(phase < duty, symbol_levels[symbol_index], 0).astype(int)

    return waveform, n_useful


def generate_nrz_8bit(symbol_levels, baud, fs):
    """
    生成 ideal NRZ PAM-256 8bit 波形。

    每个码元整个周期都输出当前 level。

    输出范围：0~255 整数。
    """

    total_time = len(symbol_levels) / baud
    n_useful = int(np.ceil(total_time * fs))

    n = np.arange(n_useful)
    t = n / fs

    u = t * baud

    symbol_index = np.floor(u).astype(int)
    symbol_index = np.clip(symbol_index, 0, len(symbol_levels) - 1)

    waveform = symbol_levels[symbol_index].astype(int)

    return waveform, n_useful


# =========================================================
# Main generation loop
# =========================================================

metadata = {}

for baud in baud_rates:
    baud_G = baud / 1e9
    sps = FS / baud

    # -------------------------
    # RZ, fixed 50% duty
    # -------------------------
    rz_waveform, rz_useful_samples = generate_rz_8bit(symbol_levels=symbol_levels, baud=baud, fs=FS, duty=0.5)

    rz_waveform_padded, rz_pad_samples = pad_to_granularity(rz_waveform, GRANULARITY)

    rz_filename = output_dir / f"RZ_PAM256tri_{baud_G:g}Gbaud_ideal_duty0.50_8bit_120GSa.csv"

    np.savetxt(rz_filename, rz_waveform_padded, delimiter=",", fmt="%d")

    # -------------------------
    # NRZ
    # -------------------------
    nrz_waveform, nrz_useful_samples = generate_nrz_8bit(symbol_levels=symbol_levels, baud=baud, fs=FS)

    nrz_waveform_padded, nrz_pad_samples = pad_to_granularity(nrz_waveform, GRANULARITY)

    nrz_filename = output_dir / f"NRZ_PAM256tri_{baud_G:g}Gbaud_ideal_8bit_120GSa.csv"

    np.savetxt(nrz_filename, nrz_waveform_padded, delimiter=",", fmt="%d")

    metadata[f"{baud_G:g}Gbaud"] = {
        "baud_Gbaud": baud_G,
        "sample_rate_GSa": FS / 1e9,
        "SPS": sps,
        "header_symbols": header_len,
        "payload_symbols": len(payload_levels),
        "total_symbols": total_symbols,
        "duration_ns_without_padding": total_symbols / baud * 1e9,
        "RZ_duty": 0.5,
        "RZ_useful_samples": int(rz_useful_samples),
        "RZ_pad_samples": int(rz_pad_samples),
        "RZ_total_samples": int(len(rz_waveform_padded)),
        "RZ_file": rz_filename.name,
        "NRZ_useful_samples": int(nrz_useful_samples),
        "NRZ_pad_samples": int(nrz_pad_samples),
        "NRZ_total_samples": int(len(nrz_waveform_padded)),
        "NRZ_file": nrz_filename.name,
    }

# 保存 metadata
with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# =========================================================
# Print summary
# =========================================================

print("CSV 文件生成完成")
print("输出目录：", output_dir)
print()
print("码元结构：50 zero header + 0→255→0")
print("payload symbols:", len(payload_levels))
print("total symbols:", total_symbols)
print("RZ duty: 50%")
print("Output format: 8bit integer, 0~255")
print()

print("baud_Gbaud | SPS    | useful_samples | pad | total_samples | duration_ns")
for key, item in metadata.items():
    print(
        f"{item['baud_Gbaud']:>10g} | "
        f"{item['SPS']:>6.3f} | "
        f"{item['RZ_useful_samples']:>14d} | "
        f"{item['RZ_pad_samples']:>3d} | "
        f"{item['RZ_total_samples']:>13d} | "
        f"{item['duration_ns_without_padding']:>10.3f}"
    )
