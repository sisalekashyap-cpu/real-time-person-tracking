#!/usr/bin/env python3
"""AI-Powered Person Detection & Tracking System V2.

MobileNet-SSD person detection with a lightweight, smoother multi-object tracker.
Designed for webcam, video-file, and network-stream input.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Keep OpenCV/Qt reliable on Linux Wayland/X11.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from centroidtracker import CentroidTracker


CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]


def load_model(proto_path: str, model_path: str):
    """Load MobileNet-SSD and fail with a useful error."""
    if not os.path.isfile(proto_path):
        raise FileNotFoundError(f"Prototxt file not found: {proto_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print("[INFO] Loading MobileNet-SSD model...")
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)


def configure_backend(net, device: str):
    """Select CUDA only when the installed OpenCV build actually supports it."""
    device = device.lower()

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")

    cuda_available = False
    try:
        cuda_available = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except cv2.error:
        cuda_available = False

    if device == "cuda" and not cuda_available:
        print("[WARNING] CUDA was requested but this OpenCV build has no usable CUDA backend.")
        print("[INFO] Falling back to CPU.")
        device = "cpu"

    if device == "auto":
        device = "cuda" if cuda_available else "cpu"

    if device == "cuda":
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return "CUDA"
        except cv2.error as exc:
            print(f"[WARNING] CUDA backend setup failed: {exc}")
            print("[INFO] Falling back to CPU.")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return "CPU"


def open_video_source(source_arg: str):
    """Open camera index, network stream, or local video file."""
    source_lower = source_arg.lower()
    network = source_lower.startswith(
        ("http://", "https://", "rtsp://", "rtmp://")
    )

    try:
        index = int(source_arg)
        print(f"[INFO] Opening live camera {index}...")
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, True, "LIVE CAMERA"
    except ValueError:
        pass

    if network:
        print(f"[INFO] Opening network stream: {source_arg}")
        cap = cv2.VideoCapture(source_arg)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, True, "NETWORK STREAM"

    print(f"[INFO] Opening video file: {source_arg}")
    return cv2.VideoCapture(source_arg), False, "VIDEO FILE"


def parse_arguments():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Smooth real-time MobileNet-SSD person detection and tracking V2."
    )
    parser.add_argument("-s", "--source", default="0",
                        help="Camera index, video path, or network stream URL.")
    parser.add_argument("-c", "--confidence", type=float, default=0.50,
                        help="Minimum person confidence (0-1).")
    parser.add_argument("-n", "--nms-thresh", type=float, default=0.30,
                        help="NMS overlap threshold (0-1).")
    parser.add_argument("-d", "--max-disappeared", type=int, default=12,
                        help="Frames a lost track can survive before removal.")
    parser.add_argument("-m", "--max-distance", type=float, default=110,
                        help="Maximum centroid distance for association.")
    parser.add_argument("--width", type=int, default=640,
                        help="Processing/display width. Height keeps aspect ratio.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop local video files at EOF.")
    parser.add_argument("--smoothing", type=float, default=0.55,
                        help="Bounding-box smoothing factor, 0-1. Higher = more responsive.")
    parser.add_argument("--min-box-area", type=float, default=500,
                        help="Reject detections smaller than this pixel area.")
    parser.add_argument("--min-box-width", type=int, default=12,
                        help="Reject detections narrower than this many pixels.")
    parser.add_argument("--min-box-height", type=int, default=20,
                        help="Reject detections shorter than this many pixels.")
    parser.add_argument("--trail", action="store_true",
                        help="Draw a short motion trail for each tracked person.")
    parser.add_argument("--trail-length", type=int, default=20,
                        help="Maximum points kept in each trajectory trail.")
    parser.add_argument("--no-hud", action="store_true",
                        help="Disable the statistics HUD.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="DNN device selection. CUDA requires CUDA-enabled OpenCV.")
    parser.add_argument("--prototxt",
                        default=str(script_dir / "MobileNetSSD_deploy.prototxt"),
                        help="MobileNet-SSD prototxt path.")
    parser.add_argument("--weights",
                        default=str(script_dir / "MobileNetSSD_deploy.caffemodel"),
                        help="MobileNet-SSD weights path.")

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.nms_thresh <= 1.0:
        parser.error("--nms-thresh must be between 0 and 1")
    if not 0.0 <= args.smoothing <= 1.0:
        parser.error("--smoothing must be between 0 and 1")
    if args.width <= 0 or args.max_disappeared < 1 or args.max_distance <= 0:
        parser.error("width, max-disappeared and max-distance must be positive")
    if args.min_box_area < 0 or args.min_box_width < 1 or args.min_box_height < 1:
        parser.error("box filtering values are invalid")
    if args.trail_length < 1:
        parser.error("--trail-length must be positive")

    return args


def clamp_rect(x1, y1, x2, y2, width, height):
    """Clip a rectangle to the frame and return integer coordinates."""
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    return x1, y1, x2, y2


def detect_people(net, frame, confidence_threshold, nms_threshold,
                  min_area, min_width, min_height):
    """Run MobileNet-SSD and return person rectangles."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5,
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    scores = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        if class_id != CLASSES.index("person"):
            continue

        x1, y1, x2, y2 = detections[0, 0, i, 3:7] * np.array(
            [width, height, width, height]
        )
        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, width, height)

        box_width = x2 - x1
        box_height = y2 - y1
        area = box_width * box_height

        if box_width < min_width or box_height < min_height or area < min_area:
            continue

        boxes.append([x1, y1, box_width, box_height])
        scores.append(confidence)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, confidence_threshold, nms_threshold
    )
    if len(indices) == 0:
        return []

    rects = []
    for idx in np.asarray(indices).reshape(-1):
        x, y, w, h = boxes[int(idx)]
        rects.append((x, y, x + w, y + h))

    return rects


def draw_hud(frame, fps, active_count, unique_count, mode, device):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 132), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    lines = [
        f"MODE: {mode}",
        f"FPS: {fps:.1f}",
        f"ACTIVE: {active_count}",
        f"UNIQUE: {unique_count}",
        f"DEVICE: {device}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (10, 23 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (235, 235, 235), 1, cv2.LINE_AA
        )


def draw_track(frame, object_id, bbox, color=(0, 165, 255)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    label = f"ID #{object_id}"
    label_y = y1 - 8 if y1 > 24 else y1 + 18
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )

    cv2.rectangle(
        frame,
        (x1, max(0, label_y - th - 5)),
        (min(frame.shape[1] - 1, x1 + tw + 6), label_y + 3),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        frame, label, (x1 + 3, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 120), 2, cv2.LINE_AA
    )


def main():
    args = parse_arguments()

    try:
        detector = load_model(args.prototxt, args.weights)
        device_name = configure_backend(detector, args.device)
    except (FileNotFoundError, cv2.error, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    tracker = CentroidTracker(
        maxDisappeared=args.max_disappeared,
        maxDistance=args.max_distance,
        smoothing=args.smoothing,
        trail_length=args.trail_length,
    )

    cap, is_live, mode = open_video_source(args.source)

    # Camera/network fallback to the bundled sample video.
    fallback_video = Path(__file__).resolve().parent / "test_video.mp4"
    if not cap.isOpened():
        cap.release()
        if is_live and fallback_video.is_file():
            print("[WARNING] Live source unavailable; using bundled test video.")
            cap, is_live, mode = open_video_source(str(fallback_video))
        if not cap.isOpened():
            print("[ERROR] Could not open the requested source.")
            return 1

    print(f"[INFO] Device: {device_name}")
    print("[INFO] Processing started. Press q or ESC to exit.")

    window_name = "AI Person Tracking V2"
    fps_time = time.perf_counter()
    fps_counter = 0
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                if not is_live and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker.reset()
                    fps_counter = 0
                    continue
                print("[INFO] Input stream ended.")
                break

            frame = cv2.resize(
                frame,
                (args.width, int(frame.shape[0] * args.width / frame.shape[1])),
                interpolation=cv2.INTER_AREA,
            )

            rects = detect_people(
                detector,
                frame,
                args.confidence,
                args.nms_thresh,
                args.min_box_area,
                args.min_box_width,
                args.min_box_height,
            )

            objects = tracker.update(rects)

            if args.trail:
                for object_id, points in tracker.trails.items():
                    if len(points) > 1:
                        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(frame, [pts], False, (255, 120, 0), 2)

            for object_id, bbox in objects.items():
                draw_track(frame, object_id, bbox)

            fps_counter += 1
            now = time.perf_counter()
            elapsed = now - fps_time
            if elapsed >= 0.5:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = now

            if not args.no_hud:
                draw_hud(
                    frame,
                    display_fps,
                    tracker.visible_count,
                    tracker.nextObjectID,
                    mode,
                    device_name,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
