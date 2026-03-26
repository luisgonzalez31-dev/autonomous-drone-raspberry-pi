# Description:
# Autonomous obstacle avoidance using YOLOv8 with DJI Tello.
# The system detects objects in real-time and makes movement decisions
# based on object size (proximity), enabling basic autonomous navigation.

# Author: Luis Gonzalez

import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO

# ======================
# CONFIG
# ======================
CONF_THRESHOLD = 0.5
AREA_THRESHOLD = 15000   # Ajusta esto (sensibilidad)
SPEED = 30

# ======================
# YOLO
# ======================
model = YOLO("yolov8n.pt")  # modelo ligero

# ======================
# TELLO
# ======================
tello = Tello()
tello.connect()
print("Batería:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()
time.sleep(2)

# ======================
# LOOP PRINCIPAL
# ======================
while True:
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))

    # YOLO detección
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    obstacle_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Detectar obstáculo cercano
            if area > AREA_THRESHOLD:
                obstacle_detected = True

                # Mostrar alerta
                cv2.putText(frame, "OBSTACULO!", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ======================
    # DECISION DE MOVIMIENTO
    # ======================
    if obstacle_detected:
        print("Obstáculo detectado → girando")
        tello.send_rc_control(0, 0, 0, 40)  # girar derecha
        time.sleep(0.5)
    else:
        print("Libre → avanzando")
        tello.send_rc_control(0, SPEED, 0, 0)  # avanzar

    # ======================
    # DISPLAY
    # ======================
    cv2.imshow("Tello YOLO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ======================
# FINAL
# ======================
tello.send_rc_control(0,0,0,0)
tello.land()
tello.streamoff()
cv2.destroyAllWindows()
