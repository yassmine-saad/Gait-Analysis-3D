import numpy as np
from pyModbusTCP.client import ModbusClient
import ctypes
import struct

# =====================================================
# MODBUS CONFIG  (CHANGE HOST IF PLC IS REMOTE)
# =====================================================
MODBUS_HOST = "127.0.0.1"
MODBUS_PORT = 1502

REG_ENABLE = 10
REG_MODE = 11
REG_LINEAR = 100
REG_ANGULAR = 101

# =====================================================
# CONTROL PARAMETERS (LOCKED)
# =====================================================
DESIRED_DISTANCE = 3.0     # meters (MANDATORY)
KD = 0.6                   # distance gain
KX = 0.003                 # centering gain

MAX_LINEAR = 0.4           # m/s
MAX_ANGULAR = 0.8          # rad/s

DEAD_ZONE = 0.10           # ±10 cm dead zone

SCALE = 100
class FollowController:
    """
    Vision-based follow controller.
    Camera mounted on the BACK of the robot.
    Robot can move FORWARD and BACKWARD.
    """

    def __init__(self):
        self.client = ModbusClient(
            host=MODBUS_HOST,
            port=MODBUS_PORT,
            auto_open=True
        )

        print("[ROBOT] Connecting to Modbus...")
        connected = self.client.open()
        print("[ROBOT] Connected:", connected)

        if not connected:
            print("Modbus connection failed")

        # Enable robot (VERY IMPORTANT)
        try:
            self.client.write_single_register(REG_ENABLE, 1)
            self.client.write_single_register(REG_MODE, 1)
            print("[ROBOT] Enable + Auto mode sent")
        except Exception as e:
            print("[ROBOT] Enable failed:", e)

        self.stop()

    # -------------------------------------------------
    def update(self, distance, offset_x):

        # ---------- Distance control ----------
        error_d = DESIRED_DISTANCE - distance

        if abs(error_d) < DEAD_ZONE:
            linear = 0.0
        else:
            linear = KD * error_d

        linear = np.clip(linear, -MAX_LINEAR, MAX_LINEAR)

        # ---------- Angular control ----------
        angular = -KX * offset_x
        angular = np.clip(angular, -MAX_ANGULAR, MAX_ANGULAR)
        # linear = 0.1
        # angular = 0.0
        # Write the linear and angular speeds to Modbus
        Vx_scaled = int(linear * SCALE)
        Wz_scaled = int(angular * SCALE)
        # Convert to int16
        Vx_16bit = ctypes.c_int16(Vx_scaled).value
        Wz_16bit = ctypes.c_int16(Wz_scaled).value

        # Pack the int16 values as binary data
        packed_Vx = struct.pack('>h', Vx_16bit)  # '>h' for big-endian 16-bit signed integer
        packed_Wz = struct.pack('>h', Wz_16bit)  # '>h' for big-endian 16-bit signed integer

        # Unpack the binary data as uint16 values
        Vx_uint16 = struct.unpack('>H', packed_Vx)[0]  # '>H' for big-endian 16-bit unsigned integer
        Wz_uint16 = struct.unpack('>H', packed_Wz)[0]  # '>H' for big-endian 16-bit unsigned integer

        print(
            f"[ROBOT] dist={distance:.2f} | "
            f"lin={linear:.2f} | ang={angular:.2f} | "
            f"reg_lin={Vx_uint16} reg_ang={Wz_uint16}"
        )

        self.client.write_multiple_registers(REG_LINEAR, [Vx_uint16, Wz_uint16])

    # -------------------------------------------------
    def stop(self):
        self.client.write_single_register(REG_LINEAR, 0)
        self.client.write_single_register(REG_ANGULAR, 0)
