from pymavlink import mavutil
import time

def connect_drone(connection_string='tcp:10.42.0.1:5760'):
    master = mavutil.mavlink_connection(connection_string)
    master.wait_heartbeat()
    print("Conectado al dron")
    return master


def test_motor(master, motor=1, throttle=10, duration=3):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        motor,
        0,
        throttle,
        duration,
        0, 0, 0
    )
    print(f"Probando motor {motor}")
