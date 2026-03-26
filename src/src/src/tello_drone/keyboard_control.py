# Description:
# Control DJI Tello drone using keyboard input and real-time video streaming.
# Includes takeoff, landing, movement (WASD), altitude control, and yaw rotation.
# Uses threading for video feed and continuous RC command updates.

# Author: Luis Gonzalez

import time
import cv2
import keyboard
import threading
from djitellopy import Tello

# CONFIG
SPEED = 50
RC_INTERVAL = 0.02


# INIT
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")

tello.streamon()
frame_reader = tello.get_frame_read()

flying = False
running = True

print("T = despegar | L = aterrizar | Q = salir")
print("WASD = movimiento | Flechas = altura/giro")

# VIDEO THREAD
def video_loop():
    while running:
        frame = frame_reader.frame

        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Tello Camera", frame)

        # IMPORTANTE: necesario para actualizar ventana
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# START VIDEO THREAD
video_thread = threading.Thread(target=video_loop)
video_thread.daemon = True
video_thread.start()


# CONTROL LOOP
last_rc_time = time.time()

while True:
    lr = fb = ud = yaw = 0

    # SALIR
    if keyboard.is_pressed('q'):
        running = False
        break

    # DESPEGAR
    if keyboard.is_pressed('t') and not flying:
        tello.takeoff()
        flying = True
        time.sleep(0.5)

    # ATERRIZAR
    if keyboard.is_pressed('l') and flying:
        tello.land()
        flying = False
        time.sleep(0.5)

    if flying:
        # MOVIMIENTO
        if keyboard.is_pressed('a'):
            lr = -SPEED
        elif keyboard.is_pressed('d'):
            lr = SPEED

        if keyboard.is_pressed('w'):
            fb = SPEED
        elif keyboard.is_pressed('s'):
            fb = -SPEED

        if keyboard.is_pressed('z'):
            ud = SPEED
        elif keyboard.is_pressed('c'):
            ud = -SPEED

        # 🔥 ROTACIÓN (YAW)
        if keyboard.is_pressed('left'):
            yaw = -SPEED
        elif keyboard.is_pressed('right'):
            yaw = SPEED

        # ENVÍO CONSTANTE
        now = time.time()
        if now - last_rc_time > RC_INTERVAL:
            tello.send_rc_control(lr, fb, ud, yaw)
            last_rc_time = now


# CLEAN EXIT
running = False
video_thread.join()

try:
    tello.send_rc_control(0, 0, 0, 0)
    if flying:
        tello.land()
except:
    pass

tello.streamoff()
tello.end()
cv2.destroyAllWindows()

print("Programa terminado")
