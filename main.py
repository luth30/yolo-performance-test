from ultralytics import YOLO
import cv2
import time
import statistics
import torch
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"

from config import *
from utils.monitor import get_system_usage
from utils.counter import count_vehicles
from utils.logger import init_csv, write_log


def main():
    print("Starting YOLO performance test...")

    # load model
    model = YOLO(MODEL_PATH)

    # load video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # cek video
    if not cap.isOpened():
        print("ERROR: Video tidak bisa dibuka")
        return

    seen_ids = set()

    frame_total = 0
    vehicle_total = 0

    inference_list = []
    cpu_list = []
    ram_list = []

    start_time = time.time()

    # init csv
    init_csv(CSV_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_total += 1

        # ===== DEBUG frame pertama =====
        if frame_total == 1:
            print("Frame shape:", frame.shape)

        # ===== INFERENCE + TRACKING =====
        t1 = time.time()
        results = model.track(frame, conf=0.05, persist=True, imgsz=480)
        t2 = time.time()

        infer_time = t2 - t1
        inference_list.append(infer_time)

        # ===== DEBUG DETEKSI =====
        if frame_total == 1:
            print("DEBUG DETECTION:", results[0].boxes)

        # ===== MONITORING =====
        cpu, ram = get_system_usage()
        cpu_list.append(cpu)
        ram_list.append(ram)

        # ===== COUNTING =====
        count = count_vehicles(results, model, LINE_Y, VEHICLE_LABELS, seen_ids)
        vehicle_total += count

        # ===== LOGGING =====
        write_log(CSV_PATH, [frame_total, infer_time, cpu, ram])

    cap.release()

    end_time = time.time()
    duration = end_time - start_time

    fps = frame_total / duration if duration > 0 else 0

    print("\n=== RESULT ===")
    print(f"Frames processed : {frame_total}")
    print(f"Total time       : {duration:.2f}s")
    print(f"FPS              : {fps:.2f}")

    print("\nInference:")
    print(f"avg  : {statistics.mean(inference_list):.4f}")
    print(f"max  : {max(inference_list):.4f}")
    print(f"min  : {min(inference_list):.4f}")

    print("\nSystem:")
    print(f"CPU avg : {statistics.mean(cpu_list):.2f}%")
    print(f"RAM avg : {statistics.mean(ram_list):.2f}%")

    print("\nVehicles counted:", vehicle_total)


if __name__ == "__main__":
    main()