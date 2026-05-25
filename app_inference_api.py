import argparse
import io
import sys
import time

import cv2
import numpy as np
import supervision as sv
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pytorch_lightning.callbacks import Callback
from rfdetr import RFDETRNano
from ultralytics import YOLO


# ==========================================================
# FIX CHECKPOINT LAMA:
# Checkpoint RF-DETR Anda menyimpan referensi:
# __main__.EpochTimerCallback
#
# Saat jalan via Streamlit, __main__ = app.py sehingga aman.
# Saat jalan via Uvicorn/FastAPI, __main__ = /opt/conda/bin/uvicorn.
# Karena itu class harus didaftarkan manual ke sys.modules["__main__"].
# ==========================================================
class EpochTimerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pass


EpochTimerCallback.__module__ = "__main__"
setattr(sys.modules["__main__"], "EpochTimerCallback", EpochTimerCallback)
setattr(sys.modules["__main__"], "Namespace", argparse.Namespace)

torch.serialization.add_safe_globals([argparse.Namespace, EpochTimerCallback])


# --- KONFIGURASI MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

CLASSES = ["Fall", "No-Fall"]

SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

JOINT_COLOR = (0, 255, 0)
BONE_COLOR = (255, 255, 255)
CONF_THRESHOLD_POSE = 0.25
PAD_COLOR = (114, 114, 114)

app = FastAPI(title="Fall Detection Inference API", version="1.0.1-fixed")

pose_model = None
rfdetr_model = None


def letterbox_image(img_array: np.ndarray, target_size: int) -> np.ndarray:
    h_ori, w_ori = img_array.shape[:2]
    scale = min(target_size / w_ori, target_size / h_ori)

    new_w = int(round(w_ori * scale))
    new_h = int(round(h_ori * scale))

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0

    img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    return cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOR,
    )


def extract_skeleton_canvas(img_square_rgb: np.ndarray) -> np.ndarray:
    black_canvas = np.zeros(img_square_rgb.shape, dtype=np.uint8)

    with torch.inference_mode():
        results = pose_model(
            img_square_rgb,
            verbose=False,
            device=DEVICE.type,
            imgsz=img_square_rgb.shape[0],
            half=USE_HALF,
        )

    if len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        for person_kpts in keypoints_data:
            if np.all(person_kpts[:, 2] == 0):
                continue

            for edge in SKELETON_EDGES:
                pt1 = person_kpts[edge[0]]
                pt2 = person_kpts[edge[1]]

                x1, y1, conf1 = int(pt1[0]), int(pt1[1]), pt1[2]
                x2, y2, conf2 = int(pt2[0]), int(pt2[1]), pt2[2]

                if (conf1 > CONF_THRESHOLD_POSE and conf2 > CONF_THRESHOLD_POSE) and (
                    x1 != 0 and x2 != 0
                ):
                    cv2.line(black_canvas, (x1, y1), (x2, y2), BONE_COLOR, 5)

            for pt in person_kpts:
                x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                if conf > CONF_THRESHOLD_POSE and x != 0:
                    cv2.circle(black_canvas, (x, y), 6, JOINT_COLOR, -1)

    return black_canvas


def annotate_detection(skeleton_canvas: np.ndarray, conf_detection: float) -> np.ndarray:
    with torch.inference_mode():
        raw_dets = rfdetr_model.predict(skeleton_canvas, threshold=conf_detection)

    if (
        hasattr(raw_dets, "xyxy")
        and raw_dets.xyxy is not None
        and len(raw_dets.xyxy) > 0
    ):
        dets = sv.Detections(
            xyxy=np.asarray(raw_dets.xyxy),
            class_id=np.asarray(raw_dets.class_id).astype(int),
            confidence=np.asarray(raw_dets.confidence).astype(float),
        )

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)

        labels = [
            f"{CLASSES[int(cls_id)]} {conf_val:.2f}"
            for cls_id, conf_val in zip(dets.class_id, dets.confidence)
        ]

        annotated_output = box_annotator.annotate(
            scene=skeleton_canvas.copy(),
            detections=dets,
        )
        annotated_output = label_annotator.annotate(
            scene=annotated_output,
            detections=dets,
            labels=labels,
        )
        return annotated_output

    return skeleton_canvas.copy()


def make_side_by_side(
    img_square_rgb: np.ndarray,
    skeleton_canvas: np.ndarray,
    annotated_output: np.ndarray,
) -> np.ndarray:
    gap = np.zeros((img_square_rgb.shape[0], 12, 3), dtype=np.uint8)
    gap[:] = 30
    return np.concatenate([img_square_rgb, gap, skeleton_canvas, gap, annotated_output], axis=1)


@app.on_event("startup")
def load_models():
    global pose_model, rfdetr_model

    print(f"[INFO] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    pose_model = YOLO("models/yolo11l-pose.pt")
    pose_model.to(DEVICE.type)

    rfdetr_model = RFDETRNano(
        num_classes=2,
        pretrain_weights="models/rfdetr_nano_skeleton.pth",
    )
    rfdetr_model.optimize_for_inference()

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        _ = pose_model(dummy, verbose=False, device=DEVICE.type, imgsz=640, half=USE_HALF)
        _ = rfdetr_model.predict(dummy, threshold=0.25)
    except Exception as exc:
        print(f"[WARNING] Warm-up dilewati: {exc}")

    print("[INFO] Models loaded successfully.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": DEVICE.type,
        "models_loaded": pose_model is not None and rfdetr_model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    target_size: int = Query(640, ge=320, le=960),
    quality: int = Query(92, ge=50, le=100),
    view: str = Query("annotated", pattern="^(annotated|skeleton|triple)$"),
):
    if pose_model is None or rfdetr_model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    start = time.perf_counter()

    try:
        image_bytes = await file.read()
        img_rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal membaca gambar: {exc}")

    img_square_rgb = letterbox_image(img_rgb, target_size)
    skeleton_canvas = extract_skeleton_canvas(img_square_rgb)
    annotated_output = annotate_detection(skeleton_canvas, conf)

    if view == "skeleton":
        out_rgb = skeleton_canvas
    elif view == "triple":
        out_rgb = make_side_by_side(img_square_rgb, skeleton_canvas, annotated_output)
    else:
        out_rgb = annotated_output

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(
        ".jpg",
        out_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Gagal encode hasil prediksi.")

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    headers = {
        "X-Inference-Time-Ms": f"{elapsed_ms:.2f}",
        "X-Device": DEVICE.type,
    }

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers=headers,
    )
