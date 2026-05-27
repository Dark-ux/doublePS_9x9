import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pyvisa
import os
from colorama import Fore, Style, init
import serial
from pathlib import Path

init(autoreset=True)

BASE_DIR = Path(__file__).resolve().parent
VOLTAGE_FILE = BASE_DIR / "VOLTAGE.csv"
CHANNEL_NUM = 128


def crc8(data):
    """
    根据多项式 0x31 计算 CRC8 校验值
    """
    polynomial = 0x31
    crc = 0x00
    data_byte = bytes.fromhex(data)
    for byte in data_byte:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
    return crc & 0xFF


def generate_channel_addresses():
    """
    生成 128 个通道对应的地址
    """
    addresses = []
    for i in range(CHANNEL_NUM):
        # ch_num 范围 1~128, i 为 0~127
        if 0 <= i <= 31:
            addr = "EB90" + "A0" + hex(i + 1 + 15)[2:].zfill(2)
        elif 32 <= i <= 63:
            addr = "EB90" + "A1" + hex(i + 1 - 17)[2:].zfill(2)
        elif 64 <= i <= 95:
            addr = "EB90" + "A2" + hex(i + 1 - 49)[2:].zfill(2)
        elif 96 <= i <= 127:
            addr = "EB90" + "A3" + hex(i + 1 - 81)[2:].zfill(2)
        else:
            addr = "00000000"
        addresses.append(addr)
    return addresses


def generate_voltage_commands(voltage_values):
    """
    根据电压值列表生成对应的命令列表
    参数:
        voltage_values -- 长度为 CHANNEL_NUM 的电压值列表（已经计算好待发送数值）
    返回:
        每个通道的完整命令列表（包含地址、电压值以及 CRC 校验）
    """
    voltage_int = np.round(voltage_values).astype(int)
    voltage_str_list = [hex(num)[2:].zfill(6) for num in voltage_int]
    addresses = generate_channel_addresses()
    commands = []
    for addr, volt_str in zip(addresses, voltage_str_list):
        cmd_body = addr + volt_str
        cmd_crc = hex(crc8(cmd_body))[2:].zfill(2)
        full_cmd = cmd_body + cmd_crc
        commands.append(full_cmd)
    return commands


def open_ser_connection(port, baudrate=115200, timeout=0.5):
    """
    打开串口连接
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        if ser.is_open:
            print(Fore.GREEN + f"串口 {port} 已成功打开")
        else:
            ser.open()
            print(Fore.GREEN + f"串口 {port} 已成功打开")
        return ser
    except serial.SerialException as e:
        print(Fore.RED + f"无法打开串口 {port}: {e}")
        return None


def upload_voltage(ser, voltage_df):
    """
    上传电压值到设备。使用传入的 voltage_df DataFrame 计算每个通道的电压，
    然后生成命令上传至设备。尝试上传至多 10 次，
    当检测到所有需要的通道电流均大于 0.01 mA 时即退出。

    参数:
        ser         -- 打开的串口对象
        voltage_df  -- 已经读取好的 Pandas DataFrame 对象，包含电压数据，
                       该 DataFrame 的第 0 列存储了各通道的电压值
    """
    # 使用传入的 DataFrame 对象，不再进行文件 I/O 操作
    # 找出存在非零电压设置的通道索引
    check_channel_index = voltage_df.index[voltage_df[0] != 0].tolist()

    # 计算公式： (v / 24 + 0.5) * 65535
    # CHANNEL_NUM 为 128（确保该变量在模块内有定义或引入）
    voltages = [(float(voltage_df.at[i, 0]) / 24 + 0.5) * 65535 for i in range(CHANNEL_NUM)]
    commands = generate_voltage_commands(voltages)

    for attempt in range(10):
        for cmd in commands:
            ser.write(bytes.fromhex(cmd))
        time.sleep(0.5)
        current_list = read_current(ser)
        check_currents = [current_list[idx] for idx in check_channel_index]
        if all(val is not None for val in check_currents) and all(val > 0.01 for val in check_currents):
            # print("所有数值均大于0.01且在精度范围内，即将跳出循环！")
            # print(check_currents)
            break

    time.sleep(0.5)
    if attempt < 9:
        print(Fore.GREEN + "成功上传电压数据!")
    else:
        print(Fore.RED + "上传电压数据失败，超出上传尝试次数!")


def clearallvoltage(ser):
    """
    将所有通道电压清零。清零后待电流全都接近 0 时则认为成功。
    """
    # 清零公式： 0.5 * 65535
    voltages = [0.5 * 65535] * CHANNEL_NUM
    commands = generate_voltage_commands(voltages)

    count = 0
    while count < 10:
        for cmd in commands:
            ser.write(bytes.fromhex(cmd))
        current_list = read_current(ser)
        if all(val is not None for val in current_list) and all(-0.01 < val < 0.01 for val in current_list):
            # print(current_list)
            print(Fore.GREEN + "电压清零成功!")
            break
        count += 1
    return


def read_pow(pwm):
    """
    读取 pwm 设备的功率数据，返回包含 8 个通道的功率值（字符串列表）
    """
    power_list = []
    for i in range(8):
        power_list.append(pwm.query(f"read{i+1}:pow?").rstrip("\n"))
    return power_list


def get_R(ser, channel, voltage_df):
    while True:
        current = []
        voltages = []
        last_current = None
        for i in range(5):
            clearallvoltage(ser)
            v = (i + 1) * 0.5
            voltage_df.at[channel - 1, 0] = v
            upload_voltage(ser, voltage_df)
            c = read_current_port(ser, channel)
            retry = 0
            while last_current is not None and c <= last_current and retry < 5:
                c = read_current_port(ser, channel)
                retry += 1
            current.append(c)
            voltages.append(v)
            print(f"测试端口{channel} 测试电压{v}V, 电流{c}mA \n")
            last_current = c

        voltage_df.at[channel - 1, 0] = 0

        valid_pairs = [(v, c) for v, c in zip(voltages, current) if c is not None]
        if len(valid_pairs) != 5:
            continue

        valid_voltages = np.array([v for v, _ in valid_pairs], dtype=float)
        valid_currents = np.array([c for _, c in valid_pairs], dtype=float)

        currents_a = valid_currents * 1e-3
        try:
            a, b = np.polyfit(currents_a, valid_voltages, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
            continue

        residuals = valid_voltages - (a * currents_a + b)

        # X = [I, 1]  (n x 2)
        X = np.column_stack([currents_a, np.ones_like(currents_a)])
        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        h = np.diag(H)

        RSS = float(np.sum(residuals**2))
        n = len(residuals)
        p = 2  # slope + intercept
        df = n - p - 1

        if df <= 0 or RSS <= 0:
            mask = np.ones_like(residuals, dtype=bool)
        else:
            denom = RSS * (1.0 - h) - residuals**2
            # 防止数值问题
            denom = np.maximum(denom, 1e-18)
            t_ext = residuals * np.sqrt(df / denom)

            threshold = 4.0  # 推荐从 4.0 开始
            mask = np.abs(t_ext) <= threshold

        filtered_voltages = valid_voltages[mask]
        filtered_currents = valid_currents[mask]
        if len(filtered_currents) <= 3:
            continue

        currents_a = filtered_currents * 1e-3
        try:
            a, b = np.polyfit(currents_a, filtered_voltages, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
            continue

        offset_current = -b / a
        if not np.isfinite(offset_current):
            continue

        return a, offset_current


def scan_mzi(port, start_voltage, end_voltage, step, ser, pwm, measure_time, out_num, file_path_df):
    """
    对指定的 MZI 通道进行扫描：
      - 依次更新电压，并调用上传、清零操作，
      - 读取对应功率数据，
      - 保存数据、图片，
      - 返回最大功率和最小功率对应的电压值。
    """
    print("=" * 50)
    print(f"正在扫描 {port} 号 MZI")

    # 定义数据和图片保存路径
    data_folder = "./powerdata"
    image_folder = "./powerimage"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    data_savepath = os.path.join(data_folder, f"{port}.txt")
    image_savepath = os.path.join(image_folder, f"{port}.png")

    v_values = np.round(np.arange(start_voltage, end_voltage + step, step), 3)
    data = []

    R, dI = get_R(ser, port, file_path_df)
    print(f"MZI {port} 电阻 R={R} Ohm, dI = {dI*1e3} mA")

    for v in v_values:
        try:
            file_path_df.at[port - 1, 0] = v
        except Exception as e:
            print(f"更新电压 CSV 文件异常: {e}")
            continue

        print(f"\n扫描端口:{port} 设置电压: {v} V")
        read_current_limit = 0
        while True:
            upload_voltage(ser, file_path_df)
            c = read_current_port(ser, port) - dI * 1e3
            if 0.9 * v / R * 1000 < c and c < 1.1 * v / R * 1000:
                print(Fore.GREEN + f"电流正常, I={c}mA")
                break
            elif read_current_limit >= 5:
                print(Fore.RED + f"电流多次异常, I={c}mA,跳过该电压点")
                break
            elif v == 0:
                print(Fore.YELLOW + f"电压为0, 跳过该电压点")
                break
            else:
                print(Fore.RED + f"电流异常, I={c}mA, 重新上传电压")
                read_current_limit += 1
        time.sleep(measure_time)

        power_str_list = read_pow(pwm)
        try:
            power_value = float(power_str_list[int(out_num) - 1]) * 1e6
        except (ValueError, IndexError) as e:
            print(f"读取功率数据错误: {e}")
            power_value = 0
        data.append([v, power_value, c])
        print(f"光功率值: {power_value} uW")

    np.savetxt(data_savepath, data, fmt="%.12f", delimiter=",", header="v,pow(uW),current(mA)", comments="")

    max_item = max(data, key=lambda x: x[1])
    min_item = min(data, key=lambda x: x[1])
    v_for_max_power = max_item[0]
    v_for_min_power = min_item[0]

    v_list = [item[0] for item in data]
    pow_list = [item[1] for item in data]
    plt.plot(v_list, pow_list, marker="o")
    plt.xlabel("v")
    plt.ylabel("pow")
    plt.title(f"V vs Power for Port {port}")
    plt.grid(True)
    plt.savefig(image_savepath)
    # plt.show()
    plt.close()
    return v_for_max_power, v_for_min_power


def read_current(ser):
    """
    读取 128 通道的电流（单位：mA），通过解析设备返回的 32 通道数据块
    """
    currents_mA = [None] * CHANNEL_NUM

    def parse_32_channels(data_hex, start_channel_index):
        if len(data_hex) < 168:
            return
        header = data_hex[0:8]
        if not (
            header.startswith("eb90b0")
            or header.startswith("eb90b1")
            or header.startswith("eb90b2")
            or header.startswith("eb90b3")
        ):
            return
        for i in range(32):
            sign_char = data_hex[8 + 5 * i]
            data_str = data_hex[9 + 5 * i : 9 + 5 * i + 4]
            if len(data_str) < 4:
                currents_mA[start_channel_index + i] = None
                continue
            try:
                sign_val = int(sign_char, 16)
                raw_val = int(data_str, 16)
            except ValueError:
                currents_mA[start_channel_index + i] = None
                continue
            voltage_part = (raw_val / 65535.0 * 20.48) - 10.24
            if sign_val == 0:
                current_mA = voltage_part / 19.0 / 5.025 * 1000.0
            else:
                current_mA = voltage_part / 19.0 / 2005.0 * 1000.0
            currents_mA[start_channel_index + i] = current_mA

    # 依次发送命令读取 4 个 32 通道数据块
    commands = ["EB90B05000000000", "EB90B15000000000", "EB90B25000000000", "EB90B35000000000"]
    for idx, cmd in enumerate(commands):
        ser.write(bytes.fromhex(cmd.replace(" ", "")))
        time.sleep(0.1)
        data_block = ser.read(84)
        data_block_hex = data_block.hex().lower().zfill(168)
        parse_32_channels(data_block_hex, idx * 32)

    return currents_mA


def read_current_port(ser, port):
    while True:
        currents = read_current(ser)
        if currents[port - 1] is not None:
            break
    return currents[port - 1]


def get_binary_waveform_data(flexdca):
    """
    获取二进制波形数据，将 X, Y 数据解析为 numpy 数组
    """
    flexdca.read_termination = ""
    flexdca.write_termination = ""
    dataX = flexdca.query(":WAVeform:XYFormat:ASCii:XDATa?")
    dataY = flexdca.query(":WAVeform:XYFormat:ASCii:YDATa?")
    dataX = np.fromstring(dataX, sep=",")
    dataY = np.fromstring(dataY, sep=",")
    data = np.array([dataX, dataY]).T
    return data


def open_VISA_connection(address):
    """
    打开 VISA 连接到仪器
    """
    IOTIMEOUT = 20000
    print("Connecting ...")
    try:
        rm = pyvisa.ResourceManager()
        connection = rm.open_resource(address)
        connection.timeout = IOTIMEOUT  # 设置 20 秒超时
        connection.read_termination = "\n"
        connection.write_termination = "\n"
        inst_id = connection.query("*IDN?")
        print("\nConnection established to:\n" + inst_id, flush=True)
    except (pyvisa.VisaIOError, pyvisa.InvalidSession) as e:
        print("\nVISA ERROR: 无法打开仪器地址.\n", flush=True)
        return None
    except Exception as other:
        print("\nVISA ERROR: 无法连接至仪器:", other, flush=True)
        return None
    return connection


def laser_channel_on(Laser, OUTP, CHAN):
    Laser.write(f"OUTP{OUTP}:CHAN{CHAN}:STATE ON")
    print(f"OUTP{OUTP}:CHAN{CHAN} ON")
    return


def laser_channel_off(Laser, OUTP, CHAN):
    Laser.write(f"OUTP{OUTP}:CHAN{CHAN}:STATE OFF")
    print(f"OUTP{OUTP}:CHAN{CHAN} OFF")
    return


def generate_working_data(FILE_PATH=VOLTAGE_FILE):
    """
    生成工作电压 DataFrame 对象
    """
    df = pd.read_csv(FILE_PATH, header=None)
    working_data = df.astype(float)
    return working_data


def write_port_voltage(port: int, voltage: float, file_data: pd.DataFrame) -> None:
    """
    Write voltage into file_data at the row corresponding to PORT (1-based).
    """
    port_idx = port - 1
    if port_idx < 0 or port_idx >= len(file_data):
        raise IndexError(f"PORT {port} is out of range for the provided file_data.")
    file_data.iloc[port_idx, 0] = round(float(voltage), 3)
