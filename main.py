from ultralytics import YOLO
import cv2
import time
import statistics

from config import *
from utils.monitor import get_system_usage
from utils.counter import count_vehicles
from utils.logger import init_csv, write_log


def main():
    print("Starting YOLO performance test...")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Video tidak bisa dibuka")
        return

    seen_ids = set()

    frame_total = 0
    vehicle_total = 0

    inference_list = []
    cpu_list = []
    ram_list = []
    confidence_list = []

    # ===== METRICS =====
    TP = 0
    FP = 0
    FN = 0

    start_time = time.time()
    init_csv(CSV_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_total += 1

        # ===== INFERENCE =====
        t1 = time.time()
        results = model.track(frame, conf=0.05, persist=True, imgsz=480)
        t2 = time.time()

        inference_time = t2 - t1
        inference_list.append(inference_time)

        # ===== SYSTEM =====
        cpu, ram = get_system_usage()
        cpu_list.append(cpu)
        ram_list.append(ram)

        # ===== DETECTION =====
        detected_this_frame = 0

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                label = model.names[int(box.cls[0])]

                if label in VEHICLE_LABELS:
                    confidence_list.append(conf)
                    detected_this_frame += 1

                    # heuristic TP/FP
                    if conf >= 0.5:
                        TP += 1
                    else:
                        FP += 1
        else:
            FN += 1  # miss detection

        # ===== COUNT =====
        count = count_vehicles(results, model, LINE_Y, VEHICLE_LABELS, seen_ids)
        vehicle_total += count

        # ===== LOG =====
        write_log(CSV_PATH, [frame_total, inference_time, cpu, ram])

    cap.release()

    # ===== CALCULATION =====
    duration = time.time() - start_time
    fps = frame_total / duration if duration > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    # ===== OUTPUT =====
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

    print("\n=== EVALUATION ===")

    if confidence_list:
        print("\nConfidence:")
        print(f"avg : {statistics.mean(confidence_list):.4f}")
        print(f"max : {max(confidence_list):.4f}")
        print(f"min : {min(confidence_list):.4f}")
    else:
        print("Confidence: tidak ada data")

    print("\nPrecision : {:.4f}".format(precision))
    print("Recall    : {:.4f}".format(recall))
    print("F1 Score  : {:.4f}".format(f1_score))


if __name__ == "__main__":
    main()