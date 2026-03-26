# Description:
# Real-time person detection and tracking using YOLOv8 with DJI Tello.
# The system identifies the largest detected person, calculates position error
# relative to screen center, and visualizes tracking metrics.
# GPU acceleration supported via PyTorch (CUDA).

# Author: Luis Gonzalez

import cv2
from djitellopy import Tello
from ultralytics import YOLO
import torch

# ======================
# GPU
# ======================
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = 0 if torch.cuda.is_available() else "cpu"

# ======================
# MODELO
# ======================
model = YOLO("yolov8n.pt")  # Este debe estar en la misma carpeta de VS

# ======================
# TELLO
# ======================
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

print("Presiona Q para salir")

# ======================
# LOOP
# ======================
while True:
    frame = frame_read.frame

    if frame is None:
        continue

    height, width, _ = frame.shape
    center_screen_x = width // 2
    center_screen_y = height // 2

    results = model(frame, device=device, classes=[0])

    annotated_frame = frame.copy()

    cv2.circle(annotated_frame, (center_screen_x, center_screen_y), 5, (0, 0, 255), -1)

    largest_area = 0
    best_box = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        error_x = person_center_x - center_screen_x
        error_y = person_center_y - center_screen_y

        norm_x = person_center_x / width
        norm_y = person_center_y / height

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(annotated_frame, (person_center_x, person_center_y), 6, (255, 0, 0), -1)

        cv2.line(
            annotated_frame,
            (center_screen_x, center_screen_y),
            (person_center_x, person_center_y),
            (255, 255, 0),
            2,
        )

        cv2.putText(annotated_frame,
                    f"Pixel: ({person_center_x}, {person_center_y})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.putText(annotated_frame,
                    f"Error: ({error_x}, {error_y})",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.putText(annotated_frame,
                    f"Norm: ({norm_x:.2f}, {norm_y:.2f})",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.putText(annotated_frame,
                    f"Area: {largest_area}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

    cv2.imshow("Tello - Coordenadas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# CIERRE
# ======================
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
print("Programa terminado")

