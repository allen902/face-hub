"""
FPS 体感测试预览窗口

快速打开一个 OpenCV 窗口，实时显示摄像头画面 + 人脸检测框 + 识别结果 + FPS。
用于在接入正式 PyQt6 UI 前，先“肉眼感受”当前推理帧率是否跟手。

Usage:
    python fps_preview.py
    python fps_preview.py --device cpu --det_size 320
    python fps_preview.py --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np

from face_vision import (
    CameraThread,
    FaceDatabase,
    FaceDetector,
    FaceRecognizer,
    FaceTracker,
    FaceVisionPipeline,
)


def draw_overlay(
    frame: np.ndarray,
    pipeline_result,
    ui_fps: float,
    cam_fps: float,
    proc_fps: float,
    latency_ms: float,
) -> np.ndarray:
    """在画面上绘制检测框、姓名、置信度和 FPS 信息。"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 人脸框与标签
    for face in pipeline_result.tracked_faces:
        x1, y1, x2, y2 = face.bbox.to_tuple()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = (0, 255, 0) if face.is_known else (0, 165, 255)
        name = face.name if face.is_known else "Unknown"
        label = f"{name} {face.confidence:.0%}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    # 左上角信息面板（半透明黑底）
    lines = [
        f"UI FPS:    {ui_fps:5.1f}",
        f"Camera:    {cam_fps:5.1f} fps",
        f"Pipeline:  {proc_fps:5.1f} fps",
        f"Latency:   {latency_ms:5.1f} ms",
        f"Faces:     {pipeline_result.total_faces} ({pipeline_result.unknown_count} unknown)",
    ]
    line_h = 26
    panel_h = line_h * len(lines) + 16
    cv2.rectangle(overlay, (0, 0), (360, panel_h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        y = 24 + i * line_h
        cv2.putText(
            overlay, text, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    return overlay


def run_preview(args: argparse.Namespace) -> None:
    """主循环：采集 → 推理 → 显示。"""
    print("Initializing FaceVision FPS preview...")
    print(f"  resolution: {args.width}x{args.height}")
    print(f"  device:     {args.device}")
    print(f"  det_size:   {args.det_size}")
    print("Press 'q' or ESC to quit.\n")

    camera = CameraThread(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.cam_fps,
    )
    detector = FaceDetector(
        device=args.device,
        det_size=args.det_size,
        confidence=args.confidence,
        quality_filter=args.quality_filter,
        min_face_size=args.min_face_size,
    )
    recognizer = FaceRecognizer(tolerance=args.tolerance)
    tracker = FaceTracker(smooth_frames=args.track_smooth)
    db = FaceDatabase(
        db_path=args.db_path,
        encoding_path=args.encoding_path,
    )

    pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
    pipeline.start()

    # 体感平滑：最近 30 帧的 UI 刷新间隔
    ui_intervals: deque[float] = deque(maxlen=30)
    latency_window: deque[float] = deque(maxlen=30)
    last_ui_time = time.perf_counter()
    last_log_time = time.perf_counter()
    ui_fps = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            result = pipeline.process_frame()
            t1 = time.perf_counter()

            if result is None:
                time.sleep(0.001)
                continue

            latency_ms = (t1 - t0) * 1000.0
            latency_window.append(latency_ms)

            now = time.perf_counter()
            ui_intervals.append(now - last_ui_time)
            last_ui_time = now
            if len(ui_intervals) == ui_intervals.maxlen:
                avg_interval = sum(ui_intervals) / len(ui_intervals)
                ui_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0

            overlay = draw_overlay(
                result.frame,
                result,
                ui_fps=ui_fps,
                cam_fps=camera.actual_fps,
                proc_fps=result.fps,
                latency_ms=sum(latency_window) / len(latency_window),
            )

            cv2.imshow("FaceVision FPS Preview (q/ESC to quit)", overlay)

            # 每秒在终端输出一次，方便后台观察或截图对比
            if now - last_log_time >= 1.0:
                last_log_time = now
                print(
                    f"UI={ui_fps:5.1f}fps | "
                    f"Cam={camera.actual_fps:5.1f}fps | "
                    f"Pipe={result.fps:5.1f}fps | "
                    f"Latency={sum(latency_window)/len(latency_window):5.1f}ms | "
                    f"Faces={result.total_faces}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nPreview stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceVision FPS perceptual preview")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640, help="frame width")
    parser.add_argument("--height", type=int, default=360, help="frame height")
    parser.add_argument("--cam_fps", type=int, default=30, help="camera target fps")
    parser.add_argument("--device", type=str, default="cuda", help="cpu / cuda")
    parser.add_argument("--det_size", type=int, default=640, help="320/480/640")
    parser.add_argument("--confidence", type=float, default=0.50, help="detection confidence")
    parser.add_argument("--tolerance", type=float, default=0.45, help="recognition tolerance")
    parser.add_argument("--track_smooth", type=int, default=5, help="tracker smooth frames")
    parser.add_argument("--min_face_size", type=int, default=80, help="min face size")
    parser.add_argument("--quality_filter", action="store_true", default=True, help="enable quality filter")
    parser.add_argument("--db_path", type=str, default="face_db.json", help="face database json")
    parser.add_argument("--encoding_path", type=str, default="encodings.pkl", help="encodings pickle")
    args = parser.parse_args()

    run_preview(args)


if __name__ == "__main__":
    main()
