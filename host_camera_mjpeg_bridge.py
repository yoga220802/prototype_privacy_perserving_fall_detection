import argparse
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native Windows Camera MJPEG Bridge for Streamlit + Docker GPU API"
    )
    parser.add_argument("--api", default="http://localhost:8000/predict")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7001)

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--process-every", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--target-size", type=int, default=640)

    parser.add_argument("--raw-quality", type=int, default=95)
    parser.add_argument("--send-quality", type=int, default=95)
    parser.add_argument("--result-quality", type=int, default=92)
    parser.add_argument(
        "--view",
        choices=["annotated", "skeleton", "triple"],
        default="triple",
        help="triple = RGB + skeleton + prediksi seperti tampilan Streamlit lama",
    )

    return parser.parse_args()


args = parse_args()
app = FastAPI(title="Native Camera MJPEG Bridge", version="1.0.0")

lock = threading.Lock()
running = True

raw_jpeg: Optional[bytes] = None
result_jpeg: Optional[bytes] = None
last_error = ""
last_api_latency_ms = 0.0
camera_fps_smooth = 0.0
frame_counter = 0
camera_opened = False
camera_info = {}


def put_overlay(frame_bgr: np.ndarray, text: str, y: int = 36):
    cv2.putText(
        frame_bgr,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def encode_jpeg(frame_bgr: np.ndarray, quality: int) -> Optional[bytes]:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return None
    return encoded.tobytes()


def inference_worker_loop():
    global result_jpeg, last_error, last_api_latency_ms

    session = requests.Session()
    local_last_frame = None
    local_last_frame_id = -1

    while running:
        with lock:
            current_raw = raw_jpeg
            current_frame_id = frame_counter

        if current_raw is None or current_frame_id == local_last_frame_id:
            time.sleep(0.01)
            continue

        # Hanya proses frame sesuai interval.
        if current_frame_id % max(args.process_every, 1) != 0:
            time.sleep(0.005)
            continue

        local_last_frame_id = current_frame_id
        local_last_frame = current_raw

        params = {
            "conf": args.conf,
            "target_size": args.target_size,
            "quality": args.result_quality,
            "view": args.view,
        }
        files = {
            "file": ("frame.jpg", local_last_frame, "image/jpeg"),
        }

        start = time.perf_counter()
        try:
            response = session.post(
                args.api,
                params=params,
                files=files,
                timeout=15,
            )
            response.raise_for_status()

            latency_ms = (time.perf_counter() - start) * 1000.0

            with lock:
                result_jpeg = response.content
                last_api_latency_ms = latency_ms
                last_error = ""

        except Exception as exc:
            with lock:
                last_error = str(exc)

            time.sleep(0.1)


def camera_loop():
    global raw_jpeg, last_error, camera_fps_smooth, frame_counter, camera_opened, camera_info

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        with lock:
            camera_opened = False
            last_error = f"Tidak dapat membuka kamera ID {args.camera}"
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with lock:
        camera_opened = True
        camera_info = {
            "camera_id": args.camera,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "backend": "CAP_DSHOW",
        }

    last_time = time.perf_counter()

    try:
        while running:
            ret, frame_bgr = cap.read()
            if not ret:
                with lock:
                    last_error = "Gagal membaca frame kamera."
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if dt > 0:
                camera_fps_smooth = 0.9 * camera_fps_smooth + 0.1 * (1.0 / dt)

            frame_counter += 1

            preview = frame_bgr.copy()
            put_overlay(
                preview,
                f"Native Camera FPS: {camera_fps_smooth:.1f} | API: {last_api_latency_ms:.0f} ms | send 1/{args.process_every}",
                36,
            )

            if last_error:
                cv2.putText(
                    preview,
                    f"Error: {last_error[:80]}",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            encoded = encode_jpeg(preview, args.raw_quality)
            if encoded is not None:
                with lock:
                    raw_jpeg = encoded

            # Jangan membakar CPU.
            time.sleep(0.001)

    finally:
        cap.release()


def mjpeg_generator(kind: str):
    boundary = b"--frame"

    while True:
        with lock:
            if kind == "raw":
                frame = raw_jpeg
            else:
                frame = result_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + b"Content-Length: "
            + str(len(frame)).encode()
            + b"\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.001)


@app.on_event("startup")
def startup():
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=inference_worker_loop, daemon=True).start()


@app.get("/")
def root():
    return {
        "message": "Native Camera MJPEG Bridge is running",
        "raw_stream": "/raw.mjpg",
        "result_stream": "/result.mjpg",
        "status": "/status",
    }


@app.get("/status")
def status():
    with lock:
        return {
            "running": running,
            "camera_opened": camera_opened,
            "camera_info": camera_info,
            "frame_counter": frame_counter,
            "camera_fps": round(camera_fps_smooth, 2),
            "api_url": args.api,
            "api_latency_ms": round(last_api_latency_ms, 2),
            "process_every": args.process_every,
            "target_size": args.target_size,
            "view": args.view,
            "last_error": last_error,
            "raw_ready": raw_jpeg is not None,
            "result_ready": result_jpeg is not None,
        }


@app.get("/raw.jpg")
def raw_jpg():
    with lock:
        frame = raw_jpeg
    if frame is None:
        return Response(status_code=204)
    return Response(content=frame, media_type="image/jpeg")


@app.get("/result.jpg")
def result_jpg():
    with lock:
        frame = result_jpeg
    if frame is None:
        return Response(status_code=204)
    return Response(content=frame, media_type="image/jpeg")


@app.get("/raw.mjpg")
def raw_mjpg():
    return StreamingResponse(
        mjpeg_generator("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/result.mjpg")
def result_mjpg():
    return StreamingResponse(
        mjpeg_generator("result"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("[INFO] Starting Native Camera MJPEG Bridge")
    print(f"[INFO] Camera ID     : {args.camera}")
    print(f"[INFO] Resolution    : {args.width}x{args.height}@{args.fps}")
    print(f"[INFO] Docker API    : {args.api}")
    print(f"[INFO] Stream URL    : http://localhost:{args.port}/raw.mjpg")
    print(f"[INFO] Result URL    : http://localhost:{args.port}/result.mjpg")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
