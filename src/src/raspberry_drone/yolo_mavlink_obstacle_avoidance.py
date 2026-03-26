# Description:
# Autonomous obstacle avoidance using YOLOv8 and MAVLink on a Raspberry Pi.
# The system processes camera input, detects obstacles, and sends velocity
# commands to the flight controller (Matek) using MAVLink.
# Implements real-time decision-making based on object position and size.

# Author: Luis Gonzalez

import cv2
import time
from ultralytics import YOLO
from pymavlink import mavutil

# ======================
# CONFIG
# ======================
PI_IP = "192.168.4.1"
PORT = 5760

CONF_THRESHOLD = 0.5
AREA_THRESHOLD = 15000

FORWARD_SPEED = 0.5
TURN_SPEED = 0.5

# ======================
# MAVLINK CONNECTION
# ======================
print("Conectando a dron...")
master = mavutil.mavlink_connection(f"tcp:{PI_IP}:{PORT}")
master.wait_heartbeat()
print("Conectado")

# ======================
# YOLO
# ======================
model = YOLO("yolov8n.pt")

# ======================
# CAMARA (USB o IP)
# ======================
cap = cv2.VideoCapture(0)  # cambia si usas cámara IP

def send_velocity(vx, vy, vz, yaw):
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0,0,0,
        vx, vy, vz,
        0,0,0,
        0, yaw
    )

# ======================
# LOOP
# ======================
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    obstacle = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) // 2

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            if area > AREA_THRESHOLD:
                obstacle = (cx, area)

    # ======================
    # DECISION
    # ======================
    if obstacle is None:
        print("Avanzando")
        send_velocity(FORWARD_SPEED, 0, 0, 0)

    else:
        cx, area = obstacle

        if 200 < cx < 440:
            print("Obstáculo enfrente → girar")
            send_velocity(0, 0, 0, TURN_SPEED)

        elif cx <= 200:
            print("Obstáculo izquierda → girar derecha")
            send_velocity(0, 0, 0, TURN_SPEED)

        else:
            print("Obstáculo derecha → girar izquierda")
            send_velocity(0, 0, 0, -TURN_SPEED)

    cv2.imshow("Drone AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

send_velocity(0,0,0,0)
cap.release()
cv2.destroyAllWindows()
