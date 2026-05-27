from utils import communication as cu
import serial
import pandas as pd

if __name__ == "__main__":
    file_path = "utils/VOLTAGE.csv"
    file_data = pd.read_csv(file_path, header=None)
    OPM1_ADDRESS = "TCPIP0::192.168.0.5::inst0::INSTR"
    opm1 = cu.open_VISA_connection(OPM1_ADDRESS)
    SER_ADDRESS = "COM3"
    mcv = serial.Serial(SER_ADDRESS, 115200, timeout=0.5)

    start_v = 0
    end_v = 5
    step_v = 0.1
    measure_time = 5

    mcv_port = 43
    opm_port = 1

    # print(cu.read_pow(opm1)[0])
    cu.scan_mzi(mcv_port, start_v, end_v, step_v, mcv, opm1, measure_time, opm_port, file_data)
