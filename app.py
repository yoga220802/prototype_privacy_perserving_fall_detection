import argparse
import base64
import csv
import io
import os
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback
from rfdetr import RFDETRNano
from ultralytics import YOLO
import supervision as sv

# Optional dependency, already present in requirements_gpu_rtx4060.txt
try:
    import imageio_ffmpeg

    IMAGEIO_FFMPEG_AVAILABLE = True
except Exception:
    imageio_ffmpeg = None
    IMAGEIO_FFMPEG_AVAILABLE = False

# Optional video reader. Supports MoviePy v1 and v2 import style.
try:
    from moviepy.editor import VideoFileClip

    MOVIEPY_AVAILABLE = True
except Exception:
    try:
        from moviepy import VideoFileClip

        MOVIEPY_AVAILABLE = True
    except Exception:
        VideoFileClip = None
        MOVIEPY_AVAILABLE = False

try:
    import GPUtil
except Exception:
    GPUtil = None


# =========================================================
# PyTorch safe loading helper for RF-DETR checkpoints
# =========================================================
class EpochTimerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pass


torch.serialization.add_safe_globals([argparse.Namespace, EpochTimerCallback])


# =========================================================
# Global configuration
# =========================================================
st.set_page_config(page_title="Privacy-Preserving Fall Detection", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 640
PAD_COLOR = (114, 114, 114)
CLASSES = ["Fall", "No-Fall"]
DEFAULT_DROIDCAM_URL = "http://192.168.0.111:4747/video"
MODEL_DIR = Path("models")


_cached_alarm_html = None

def get_alarm_audio_html() -> str:
    global _cached_alarm_html
    if _cached_alarm_html is not None:
        return _cached_alarm_html

    sound_dir = Path("assets/sound")
    if not sound_dir.exists():
        _cached_alarm_html = ""
        return ""

    alarm_files = list(sound_dir.glob("alarm.*"))
    if not alarm_files:
        _cached_alarm_html = ""
        return ""

    file_path = alarm_files[0]
    try:
        data = file_path.read_bytes()
        b64 = base64.b64encode(data).decode()
        ext = file_path.suffix.lower().replace(".", "")
        mime = f"audio/{ext}"
        if ext == "mp3":
            mime = "audio/mpeg"

        _cached_alarm_html = f"""
        <audio autoplay loop style="display:none;">
            <source src="data:{mime};base64,{b64}" type="{mime}">
        </audio>
        """
    except Exception as e:
        print(f"Error loading alarm audio: {e}")
        _cached_alarm_html = ""
    return _cached_alarm_html

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


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    model_family: str  # "YOLO" or "RF-DETR"
    input_mode: str  # "RGB" or "Skeleton"
    weight_candidates: Tuple[str, ...]

    @property
    def short_label(self) -> str:
        return f"{self.model_family} {self.input_mode}"


MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        key="rfdetr_rgb",
        display_name="RF-DETR Nano - RGB",
        model_family="RF-DETR",
        input_mode="RGB",
        weight_candidates=("rfdetr_nano_rgb.pth",),
    ),
    ModelConfig(
        key="rfdetr_skeleton",
        display_name="RF-DETR Nano - Skeleton",
        model_family="RF-DETR",
        input_mode="Skeleton",
        weight_candidates=("rfdetr_nano_skeleton.pth",),
    ),
    ModelConfig(
        key="yolo_rgb",
        display_name="YOLO Nano - RGB",
        model_family="YOLO",
        input_mode="RGB",
        weight_candidates=(
            "Salinan best.pt",
            "best.pt",
            "yolo_nano_rgb.pt",
            "yolo11n_rgb.pt",
        ),
    ),
    ModelConfig(
        key="yolo_skeleton",
        display_name="YOLO Nano - Skeleton",
        model_family="YOLO",
        input_mode="Skeleton",
        weight_candidates=(
            "yolo_nano_skeleton.pt",
            "yolov11_nano_skeleton.pt",
            "yolo11n_skeleton.pt",
        ),
    ),
]
MODEL_BY_KEY: Dict[str, ModelConfig] = {cfg.key: cfg for cfg in MODEL_CONFIGS}


# =========================================================
# File/path helpers
# =========================================================
def resolve_model_path(cfg: ModelConfig) -> Optional[Path]:
    for candidate in cfg.weight_candidates:
        p = MODEL_DIR / candidate
        if p.exists():
            return p
    return None


def readable_model_paths() -> Dict[str, str]:
    rows = {}
    for cfg in MODEL_CONFIGS:
        path = resolve_model_path(cfg)
        rows[cfg.display_name] = str(path) if path else "BELUM DITEMUKAN"
    return rows


def safe_filename(text: str) -> str:
    clean = text.lower().replace(" ", "_").replace("-", "_")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    return "".join(ch for ch in clean if ch in allowed)


def read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def make_zip_from_files(
    file_map: Dict[str, Path], extra_text_files: Optional[Dict[str, str]] = None
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, file_path in file_map.items():
            if file_path and Path(file_path).exists():
                zf.write(file_path, arcname=arcname)
        if extra_text_files:
            for arcname, content in extra_text_files.items():
                zf.writestr(arcname, content)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# Image preprocessing and skeleton abstraction
# =========================================================
def letterbox_image(img_array: np.ndarray) -> np.ndarray:
    h_ori, w_ori = img_array.shape[:2]
    scale = min(TARGET_SIZE / w_ori, TARGET_SIZE / h_ori)
    new_w, new_h = int(round(w_ori * scale)), int(round(h_ori * scale))
    pad_w, pad_h = (TARGET_SIZE - new_w) / 2.0, (TARGET_SIZE - new_h) / 2.0
    img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOR,
    )


def extract_skeleton_canvas(img_640_rgb: np.ndarray, pose_model: YOLO) -> np.ndarray:
    canvas = np.zeros(img_640_rgb.shape, dtype=np.uint8)
    results = pose_model(img_640_rgb, verbose=False, device=DEVICE.type)
    if len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        for person_kpts in keypoints_data:
            if np.all(person_kpts[:, 2] == 0):
                continue

            for edge in SKELETON_EDGES:
                pt1, pt2 = person_kpts[edge[0]], person_kpts[edge[1]]
                x1, y1, conf1 = int(pt1[0]), int(pt1[1]), pt1[2]
                x2, y2, conf2 = int(pt2[0]), int(pt2[1]), pt2[2]
                if (
                    conf1 > CONF_THRESHOLD_POSE
                    and conf2 > CONF_THRESHOLD_POSE
                    and x1 != 0
                    and x2 != 0
                ):
                    cv2.line(canvas, (x1, y1), (x2, y2), BONE_COLOR, 5)

            for pt in person_kpts:
                x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                if conf > CONF_THRESHOLD_POSE and x != 0:
                    cv2.circle(canvas, (x, y), 6, JOINT_COLOR, -1)
    return canvas


def prepare_model_input(
    frame_rgb: np.ndarray, cfg: ModelConfig, pose_model: Optional[YOLO]
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    padded_rgb = letterbox_image(frame_rgb)
    if cfg.input_mode == "Skeleton":
        if pose_model is None:
            raise RuntimeError(
                "Pose model belum dimuat, padahal mode Skeleton membutuhkan pose extractor."
            )
        skeleton = extract_skeleton_canvas(padded_rgb, pose_model)
        return padded_rgb, skeleton, skeleton
    return padded_rgb, None, padded_rgb


# =========================================================
# Model loading
# =========================================================
@st.cache_resource(show_spinner=False)
def load_pose_model_cached() -> YOLO:
    pose_path = MODEL_DIR / "yolo11l-pose.pt"
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose model tidak ditemukan: {pose_path}")
    model = YOLO(str(pose_path))
    model.to(DEVICE.type)
    return model


@st.cache_resource(show_spinner=False)
def load_detector_cached(model_key: str, weight_path: str):
    cfg = MODEL_BY_KEY[model_key]
    if cfg.model_family == "YOLO":
        model = YOLO(weight_path)
        model.to(DEVICE.type)
        return model
    if cfg.model_family == "RF-DETR":
        model = RFDETRNano(num_classes=len(CLASSES), pretrain_weights=weight_path)
        model.optimize_for_inference()
        return model
    raise ValueError(f"Model family tidak dikenali: {cfg.model_family}")


def load_detector_uncached(cfg: ModelConfig):
    weight_path = resolve_model_path(cfg)
    if weight_path is None:
        raise FileNotFoundError(
            f"Weight untuk {cfg.display_name} tidak ditemukan. Kandidat: {cfg.weight_candidates}"
        )
    if cfg.model_family == "YOLO":
        model = YOLO(str(weight_path))
        model.to(DEVICE.type)
        return model
    if cfg.model_family == "RF-DETR":
        model = RFDETRNano(num_classes=len(CLASSES), pretrain_weights=str(weight_path))
        model.optimize_for_inference()
        return model
    raise ValueError(f"Model family tidak dikenali: {cfg.model_family}")


def unload_model(model):
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# Unified detection result helpers
# =========================================================
def normalize_yolo_detections(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=float),
        )
    xyxy = boxes.xyxy.detach().cpu().numpy()
    class_id = boxes.cls.detach().cpu().numpy().astype(int)
    confidence = boxes.conf.detach().cpu().numpy().astype(float)
    return xyxy, class_id, confidence


def normalize_rfdetr_detections(raw_dets) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        not hasattr(raw_dets, "xyxy")
        or raw_dets.xyxy is None
        or len(raw_dets.xyxy) == 0
    ):
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=float),
        )
    xyxy = np.asarray(raw_dets.xyxy, dtype=float)
    class_id = np.asarray(raw_dets.class_id, dtype=int)
    confidence = np.asarray(raw_dets.confidence, dtype=float)
    return xyxy, class_id, confidence


def run_detector(
    input_img_rgb: np.ndarray, detector, cfg: ModelConfig, threshold: float
) -> Dict:
    if cfg.model_family == "YOLO":
        results = detector.predict(
            input_img_rgb,
            conf=threshold,
            imgsz=TARGET_SIZE,
            device=DEVICE.type,
            verbose=False,
        )
        xyxy, class_id, confidence = normalize_yolo_detections(results[0])
    elif cfg.model_family == "RF-DETR":
        raw = detector.predict(input_img_rgb, threshold=threshold)
        xyxy, class_id, confidence = normalize_rfdetr_detections(raw)
    else:
        raise ValueError(f"Model family tidak dikenali: {cfg.model_family}")

    return {
        "xyxy": xyxy,
        "class_id": class_id,
        "confidence": confidence,
        "count": int(len(xyxy)),
    }


def class_name(class_id: int) -> str:
    if 0 <= int(class_id) < len(CLASSES):
        return CLASSES[int(class_id)]
    return f"class_{int(class_id)}"


def format_detection_result(pred: Dict) -> str:
    if pred["count"] <= 0:
        return "None"
    parts = []
    for cls_id, conf in zip(pred["class_id"], pred["confidence"]):
        parts.append(f"{class_name(int(cls_id))}:{float(conf):.2f}")
    return " | ".join(parts)


def detections_to_sv(pred: Dict) -> Optional[sv.Detections]:
    if pred["count"] <= 0:
        return None
    return sv.Detections(
        xyxy=np.asarray(pred["xyxy"], dtype=float),
        class_id=np.asarray(pred["class_id"], dtype=int),
        confidence=np.asarray(pred["confidence"], dtype=float),
    )


def annotate_prediction(scene_rgb: np.ndarray, pred: Dict) -> np.ndarray:
    dets = detections_to_sv(pred)
    if dets is None:
        return scene_rgb.copy()

    labels = [
        f"{class_name(int(c))} {float(conf):.2f}"
        for c, conf in zip(dets.class_id, dets.confidence)
    ]
    box_ann = sv.BoxAnnotator(thickness=3)
    label_ann = sv.LabelAnnotator(text_thickness=2, text_scale=0.7)
    annotated = box_ann.annotate(scene=scene_rgb.copy(), detections=dets)
    annotated = label_ann.annotate(scene=annotated, detections=dets, labels=labels)
    return annotated


def encode_png_bytes(img_rgb: np.ndarray) -> bytes:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# =========================================================
# Metrics helpers
# =========================================================
def get_system_metrics(
    start_time: Optional[float] = None, end_time: Optional[float] = None
) -> Dict:
    metrics = {}
    if start_time is not None and end_time is not None:
        elapsed = end_time - start_time
        metrics["inference_ms"] = elapsed * 1000.0
        metrics["fps"] = (1.0 / elapsed) if elapsed > 0 else 0.0
    else:
        metrics["inference_ms"] = None
        metrics["fps"] = None

    metrics["cpu_usage"] = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    metrics["ram_used_gb"] = ram.used / (1024**3)
    metrics["ram_total_gb"] = ram.total / (1024**3)

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        used_mem = total_mem - free_mem
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "total_gb": total_mem / (1024**3),
            "used_gb": used_mem / (1024**3),
            "util_percent": None,
        }
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info["util_percent"] = gpus[0].load * 100
        metrics["gpu"] = gpu_info
    else:
        metrics["gpu"] = None
    return metrics


def metric_record_base(
    cfg: ModelConfig, pred: Dict, metrics: Dict, extra: Optional[Dict] = None
) -> Dict:
    row = {
        "model_key": cfg.key,
        "model_name": cfg.display_name,
        "model_family": cfg.model_family,
        "input_mode": cfg.input_mode,
        "detections": format_detection_result(pred),
        "detection_count": pred["count"],
        "fps": f"{metrics['fps']:.2f}" if metrics.get("fps") is not None else "N/A",
        "inference_ms": f"{metrics['inference_ms']:.2f}"
        if metrics.get("inference_ms") is not None
        else "N/A",
        "cpu_percent": f"{metrics['cpu_usage']:.1f}",
        "ram_used_gb": f"{metrics['ram_used_gb']:.2f}",
        "ram_total_gb": f"{metrics['ram_total_gb']:.2f}",
        "gpu_util_percent": (
            f"{metrics['gpu']['util_percent']:.1f}"
            if metrics.get("gpu") and metrics["gpu"]["util_percent"] is not None
            else "N/A"
        ),
        "gpu_used_gb": f"{metrics['gpu']['used_gb']:.2f}"
        if metrics.get("gpu")
        else "N/A",
        "gpu_total_gb": f"{metrics['gpu']['total_gb']:.2f}"
        if metrics.get("gpu")
        else "N/A",
    }
    if extra:
        row.update(extra)
    return row


def records_to_csv_bytes(records: List[Dict]) -> bytes:
    if not records:
        return b""
    all_fields = []
    for rec in records:
        for key in rec.keys():
            if key not in all_fields:
                all_fields.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8")


def show_metrics_row(metrics: Dict):
    cols = st.columns([1, 1, 1, 1.5, 2])
    cols[0].metric(
        "FPS", f"{metrics['fps']:.2f}" if metrics.get("fps") is not None else "N/A"
    )
    cols[1].metric(
        "Inferensi (ms)",
        f"{metrics['inference_ms']:.2f}"
        if metrics.get("inference_ms") is not None
        else "N/A",
    )
    cols[2].metric("CPU (%)", f"{metrics['cpu_usage']:.1f}")
    cols[3].metric(
        "RAM (GB)", f"{metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}"
    )
    if metrics.get("gpu"):
        gpu = metrics["gpu"]
        util = gpu["util_percent"]
        gpu_text = (
            f"{util:.1f}% | {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB"
            if util is not None
            else f"{gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB"
        )
        cols[4].metric("GPU Util & VRAM", gpu_text)
    else:
        cols[4].metric("GPU Util & VRAM", "N/A")


# =========================================================
# Camera helpers: Windows/DroidCam friendly
# =========================================================
def _camera_backend_candidates():
    """Media Foundation first because this setup detected DroidCam/webcam with MSMF."""
    candidates = []
    if hasattr(cv2, "CAP_MSMF"):
        candidates.append((cv2.CAP_MSMF, "Media Foundation"))
    if hasattr(cv2, "CAP_DSHOW"):
        candidates.append((cv2.CAP_DSHOW, "DirectShow"))
    candidates.append((cv2.CAP_ANY, "Default"))
    return candidates


def _configure_capture(cap):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass


def open_camera_capture(source_config: Dict):
    mode = source_config.get("mode")
    value = source_config.get("value")

    if mode == "index":
        index = int(value)
        last_cap = None
        for backend, backend_name in _camera_backend_candidates():
            cap = cv2.VideoCapture(index, backend)
            _configure_capture(cap)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, f"Kamera ID {index} ({backend_name})"
            cap.release()
            last_cap = cap
        return last_cap, f"Kamera ID {index}"

    if mode == "url":
        url = str(value).strip()
        cap = cv2.VideoCapture(url)
        _configure_capture(cap)
        return cap, url

    if mode == "custom":
        raw_value = str(value).strip()
        if raw_value.isdigit():
            return open_camera_capture({"mode": "index", "value": int(raw_value)})
        cap = cv2.VideoCapture(raw_value)
        _configure_capture(cap)
        return cap, raw_value

    cap = cv2.VideoCapture(0)
    _configure_capture(cap)
    return cap, "Kamera default"


def scan_camera_indices(max_index: int = 10) -> List[int]:
    valid_ids = []
    for cam_idx in range(max_index + 1):
        for backend, _ in _camera_backend_candidates():
            cap = cv2.VideoCapture(cam_idx, backend)
            _configure_capture(cap)
            ok = False
            if cap.isOpened():
                ret, frame = cap.read()
                ok = bool(ret and frame is not None)
            cap.release()
            if ok:
                valid_ids.append(cam_idx)
                break
    return valid_ids


def render_camera_source_selector(key_prefix: str) -> Dict:
    source_mode = st.sidebar.radio(
        "Sumber kamera:",
        ["DroidCam via URL", "ID Kamera / Virtual Webcam", "Custom OpenCV Source"],
        key=f"{key_prefix}_camera_source_mode",
        help="DroidCam paling stabil lewat URL HTTP, misalnya http://192.168.0.111:4747/video.",
    )

    if source_mode == "ID Kamera / Virtual Webcam":
        max_scan = st.sidebar.number_input(
            "Maksimal ID yang discan",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key=f"{key_prefix}_max_scan",
        )
        if st.sidebar.button("🔎 Scan kamera", key=f"{key_prefix}_scan_camera"):
            with st.spinner("Mencari kamera yang bisa dibaca OpenCV..."):
                st.session_state[f"{key_prefix}_valid_camera_ids"] = (
                    scan_camera_indices(int(max_scan))
                )

        valid_ids = st.session_state.get(f"{key_prefix}_valid_camera_ids", [])
        options = valid_ids if valid_ids else list(range(0, int(max_scan) + 1))
        selected_id = st.sidebar.selectbox(
            "Pilih ID Kamera:",
            options,
            key=f"{key_prefix}_camera_id",
            format_func=lambda x: (
                f"Kamera ID {x}" + (" ✅ terdeteksi" if x in valid_ids else "")
            ),
        )
        if valid_ids:
            st.sidebar.success(f"ID valid ditemukan: {valid_ids}")
        else:
            st.sidebar.caption(
                "Belum ada hasil scan. Jika DroidCam tidak muncul di ID mana pun, gunakan mode URL."
            )
        return {"mode": "index", "value": int(selected_id)}

    if source_mode == "DroidCam via URL":
        url = st.sidebar.text_input(
            "URL DroidCam",
            value=DEFAULT_DROIDCAM_URL,
            key=f"{key_prefix}_droidcam_url",
            help="Alternatif umum: /mjpegfeed atau /mjpegfeed?640x480.",
        )
        st.sidebar.caption(
            "Contoh alternatif: http://192.168.0.111:4747/mjpegfeed?640x480"
        )
        return {"mode": "url", "value": url}

    custom_source = st.sidebar.text_input(
        "Custom source",
        value="0",
        key=f"{key_prefix}_custom_source",
        help="Bisa diisi ID kamera, URL HTTP/RTSP, atau source lain yang dikenali OpenCV.",
    )
    return {"mode": "custom", "value": custom_source}


def show_camera_troubleshooting(source_desc: str):
    st.info(
        "Jika DroidCam tidak terbaca lewat Camera ID, buka DroidCam di HP dan gunakan mode **DroidCam via URL**. "
        "Masukkan URL seperti `http://192.168.0.111:4747/video` atau `http://192.168.0.111:4747/mjpegfeed?640x480`."
    )
    st.caption(f"Source yang dicoba: {source_desc}")


# =========================================================
# Video writer and video processing helpers
# =========================================================
class VideoWriterBase:
    def write(self, frame_rgb: np.ndarray):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class FFmpegRGBWriter(VideoWriterBase):
    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        fps_str = f"{float(fps):.03f}"
        cmd = [
            exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            fps_str,
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_rgb: np.ndarray):
        if self.process.stdin is None:
            return
        self.process.stdin.write(
            np.ascontiguousarray(frame_rgb).astype(np.uint8).tobytes()
        )

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()


class OpenCVAVIWriter(VideoWriterBase):
    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def write(self, frame_rgb: np.ndarray):
        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()


def create_video_writer(
    output_stem: str, fps: float, width: int = TARGET_SIZE, height: int = TARGET_SIZE
) -> Tuple[VideoWriterBase, Path, str]:
    temp_dir = Path(tempfile.gettempdir())
    if IMAGEIO_FFMPEG_AVAILABLE:
        path = temp_dir / f"{output_stem}_{int(time.time() * 1000)}.mp4"
        return FFmpegRGBWriter(path, fps, width, height), path, "video/mp4"

    path = temp_dir / f"{output_stem}_{int(time.time() * 1000)}.avi"
    return OpenCVAVIWriter(path, fps, width, height), path, "video/avi"


def process_single_frame(
    frame_rgb: np.ndarray,
    cfg: ModelConfig,
    detector,
    pose_model: Optional[YOLO],
    threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict, Dict]:
    padded, skeleton, model_input = prepare_model_input(frame_rgb, cfg, pose_model)
    start_t = time.time()
    pred = run_detector(model_input, detector, cfg, threshold)
    end_t = time.time()
    annotated = annotate_prediction(model_input, pred)
    metrics = get_system_metrics(start_t, end_t)
    return padded, skeleton, annotated, pred, metrics


def process_image_all_models(
    image_rgb: np.ndarray, threshold: float, progress_container
) -> Tuple[Dict[str, Dict], List[Dict]]:
    results = {}
    records = []
    pose_model = load_pose_model_cached()
    total = len(MODEL_CONFIGS)

    for idx, cfg in enumerate(MODEL_CONFIGS):
        progress_container.info(f"Memproses {cfg.display_name} ({idx + 1}/{total})...")
        weight_path = resolve_model_path(cfg)
        if weight_path is None:
            results[cfg.key] = {
                "error": f"Weight tidak ditemukan. Kandidat: {cfg.weight_candidates}"
            }
            continue

        detector = load_detector_uncached(cfg)
        try:
            padded, skeleton, annotated, pred, metrics = process_single_frame(
                image_rgb,
                cfg,
                detector,
                pose_model if cfg.input_mode == "Skeleton" else None,
                threshold,
            )
            png_bytes = encode_png_bytes(annotated)
            result = {
                "cfg": cfg,
                "padded": padded,
                "skeleton": skeleton,
                "annotated": annotated,
                "pred": pred,
                "metrics": metrics,
                "png_bytes": png_bytes,
            }
            results[cfg.key] = result
            records.append(
                metric_record_base(cfg, pred, metrics, extra={"source_type": "image"})
            )
        finally:
            unload_model(detector)

    progress_container.success("Semua model selesai memproses gambar.")
    return results, records


def process_video_for_model(
    input_video_path: Path, cfg: ModelConfig, threshold: float, progress, status
) -> Tuple[Optional[Path], Optional[str], List[Dict], Optional[str]]:
    weight_path = resolve_model_path(cfg)
    if weight_path is None:
        return (
            None,
            None,
            [],
            f"Weight untuk {cfg.display_name} tidak ditemukan. Kandidat: {cfg.weight_candidates}",
        )

    pose_model = load_pose_model_cached() if cfg.input_mode == "Skeleton" else None
    detector = load_detector_uncached(cfg)
    records: List[Dict] = []
    writer = None
    output_path = None
    mime = None

    try:
        if MOVIEPY_AVAILABLE:
            clip = VideoFileClip(str(input_video_path))
            fps = float(clip.fps or 25)
            total_frames = max(1, int((clip.duration or 0) * fps))
            writer, output_path, mime = create_video_writer(
                f"{safe_filename(cfg.display_name)}", fps
            )

            for idx, frame_rgb in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
                padded, skeleton, annotated, pred, metrics = process_single_frame(
                    frame_rgb, cfg, detector, pose_model, threshold
                )
                writer.write(annotated)
                timestamp = idx / fps
                records.append(
                    metric_record_base(
                        cfg,
                        pred,
                        metrics,
                        extra={
                            "source_type": "video_upload",
                            "frame_index": idx,
                            "timestamp_sec": f"{timestamp:.2f}",
                        },
                    )
                )
                progress.progress(min((idx + 1) / total_frames, 1.0))
            clip.close()
        else:
            cap = cv2.VideoCapture(str(input_video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            writer, output_path, mime = create_video_writer(
                f"{safe_filename(cfg.display_name)}", fps
            )

            idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                padded, skeleton, annotated, pred, metrics = process_single_frame(
                    frame_rgb, cfg, detector, pose_model, threshold
                )
                writer.write(annotated)
                timestamp = idx / fps
                records.append(
                    metric_record_base(
                        cfg,
                        pred,
                        metrics,
                        extra={
                            "source_type": "video_upload",
                            "frame_index": idx,
                            "timestamp_sec": f"{timestamp:.2f}",
                        },
                    )
                )
                idx += 1
                progress.progress(min(idx / max(total_frames, 1), 1.0))
            cap.release()
    finally:
        if writer is not None:
            writer.close()
        unload_model(detector)

    return output_path, mime, records, None


def process_recorded_frames_for_model(
    frames_rgb: List[np.ndarray],
    cfg: ModelConfig,
    threshold: float,
    capture_fps: float,
    progress,
) -> Tuple[Optional[Path], Optional[str], List[Dict], Optional[str]]:
    weight_path = resolve_model_path(cfg)
    if weight_path is None:
        return (
            None,
            None,
            [],
            f"Weight untuk {cfg.display_name} tidak ditemukan. Kandidat: {cfg.weight_candidates}",
        )

    pose_model = load_pose_model_cached() if cfg.input_mode == "Skeleton" else None
    detector = load_detector_uncached(cfg)
    records: List[Dict] = []
    writer = None
    output_path = None
    mime = None

    try:
        writer, output_path, mime = create_video_writer(
            f"testing_{safe_filename(cfg.display_name)}", capture_fps
        )
        total = max(len(frames_rgb), 1)
        for idx, frame_rgb in enumerate(frames_rgb):
            padded, skeleton, annotated, pred, metrics = process_single_frame(
                frame_rgb, cfg, detector, pose_model, threshold
            )
            writer.write(annotated)
            timestamp = idx / capture_fps
            records.append(
                metric_record_base(
                    cfg,
                    pred,
                    metrics,
                    extra={
                        "source_type": "camera_testing",
                        "frame_index": idx,
                        "timestamp_sec": f"{timestamp:.2f}",
                    },
                )
            )
            progress.progress((idx + 1) / total)
    finally:
        if writer is not None:
            writer.close()
        unload_model(detector)

    return output_path, mime, records, None


# =========================================================
# UI helpers
# =========================================================
def render_model_status():
    with st.sidebar.expander("Status file model", expanded=False):
        for name, path in readable_model_paths().items():
            if path == "BELUM DITEMUKAN":
                st.error(f"{name}: {path}")
            else:
                st.success(f"{name}: {path}")
        pose_path = MODEL_DIR / "yolo11l-pose.pt"
        if pose_path.exists():
            st.success(f"Pose extractor: {pose_path}")
        else:
            st.error(f"Pose extractor: {pose_path} BELUM DITEMUKAN")


def select_realtime_model_config() -> ModelConfig:
    st.sidebar.markdown("### Pilihan Pipeline Realtime")
    col_mode, col_model = st.sidebar.columns(2)
    with col_mode:
        input_mode = st.radio("Mode", ["RGB", "Skeleton"], key="rt_input_mode")
    with col_model:
        model_family = st.radio("Model", ["YOLO", "RF-DETR"], key="rt_model_family")

    for cfg in MODEL_CONFIGS:
        if cfg.input_mode == input_mode and cfg.model_family == model_family:
            return cfg
    raise RuntimeError("Kombinasi model tidak ditemukan.")


def display_image_results(results: Dict[str, Dict], records: List[Dict]):
    st.subheader("Hasil Inferensi Semua Model")
    cols = st.columns(2)
    image_file_map: Dict[str, Path] = {}

    for idx, cfg in enumerate(MODEL_CONFIGS):
        result = results.get(cfg.key, {})
        with cols[idx % 2]:
            st.markdown(f"#### {cfg.display_name}")
            if "error" in result:
                st.error(result["error"])
                continue
            st.image(
                result["annotated"],
                caption=f"Output {cfg.display_name}",
                use_container_width=True,
            )
            st.write(f"**Deteksi:** {format_detection_result(result['pred'])}")
            show_metrics_row(result["metrics"])
            filename = f"{safe_filename(cfg.display_name)}.png"
            st.download_button(
                f"Download PNG - {cfg.short_label}",
                result["png_bytes"],
                file_name=filename,
                mime="image/png",
                key=f"dl_img_{cfg.key}",
            )
            tmp_img = (
                Path(tempfile.gettempdir())
                / f"{safe_filename(cfg.display_name)}_{int(time.time() * 1000)}.png"
            )
            tmp_img.write_bytes(result["png_bytes"])
            image_file_map[f"images/{filename}"] = tmp_img

    csv_bytes = records_to_csv_bytes(records)
    if csv_bytes:
        st.download_button(
            "Download CSV Ringkasan Gambar",
            csv_bytes,
            file_name="image_all_models_metrics.csv",
            mime="text/csv",
        )
        zip_bytes = make_zip_from_files(
            image_file_map,
            {"metrics/image_all_models_metrics.csv": csv_bytes.decode("utf-8")},
        )
        st.download_button(
            "Download ZIP Semua Output Gambar",
            zip_bytes,
            file_name="image_all_models_outputs.zip",
            mime="application/zip",
        )


def display_video_downloads(
    video_outputs: Dict[str, Dict], all_records: List[Dict], prefix: str
):
    st.subheader("Output Video per Model")
    file_map: Dict[str, Path] = {}
    cols = st.columns(2)
    for idx, cfg in enumerate(MODEL_CONFIGS):
        output = video_outputs.get(cfg.key, {})
        with cols[idx % 2]:
            st.markdown(f"#### {cfg.display_name}")
            if output.get("error"):
                st.error(output["error"])
                continue
            path = output.get("path")
            mime = output.get("mime") or "video/mp4"
            if path and Path(path).exists():
                data = read_file_bytes(Path(path))
                try:
                    st.video(data)
                except Exception:
                    st.caption(
                        "Preview video tidak tersedia, tetapi file tetap bisa diunduh."
                    )
                ext = ".mp4" if mime == "video/mp4" else ".avi"
                filename = f"{prefix}_{safe_filename(cfg.display_name)}{ext}"
                st.download_button(
                    f"Download Video - {cfg.short_label}",
                    data,
                    file_name=filename,
                    mime=mime,
                    key=f"dl_video_{prefix}_{cfg.key}",
                )
                file_map[f"videos/{filename}"] = Path(path)

    csv_bytes = records_to_csv_bytes(all_records)
    if csv_bytes:
        st.download_button(
            "Download CSV Semua Model",
            csv_bytes,
            file_name=f"{prefix}_all_models_metrics.csv",
            mime="text/csv",
            key=f"dl_csv_{prefix}",
        )
        zip_bytes = make_zip_from_files(
            file_map,
            {f"metrics/{prefix}_all_models_metrics.csv": csv_bytes.decode("utf-8")},
        )
        st.download_button(
            "Download ZIP Semua Output",
            zip_bytes,
            file_name=f"{prefix}_all_models_outputs.zip",
            mime="application/zip",
            key=f"dl_zip_{prefix}",
        )


def capture_camera_frames(
    source_config: Dict, duration_sec: float, target_fps: float
) -> Tuple[List[np.ndarray], float, Optional[str]]:
    cap, source_desc = open_camera_capture(source_config)
    if cap is None or not cap.isOpened():
        return [], target_fps, f"Tidak dapat membuka kamera: {source_desc}"

    frames: List[np.ndarray] = []
    preview = st.empty()
    progress = st.progress(0)
    start = time.time()
    next_capture = start
    interval = 1.0 / max(target_fps, 1.0)

    while True:
        now = time.time()
        if now - start >= duration_sec:
            break
        ret, frame_bgr = cap.read()
        if not ret:
            cap.release()
            return frames, target_fps, "Frame kamera gagal dibaca."
        if now >= next_capture:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            preview.image(
                frame_rgb,
                caption=f"Capture frame ke-{len(frames)}",
                use_container_width=True,
            )
            next_capture = now + interval
        progress.progress(min((now - start) / duration_sec, 1.0))

    cap.release()
    progress.progress(1.0)
    actual_fps = len(frames) / max(duration_sec, 0.001)
    return frames, actual_fps, None


# =========================================================
# App layout
# =========================================================
st.title("🤸‍♂️ Privacy-Preserving Fall Detection")
st.caption(
    "Multi-model inference: RF-DETR Nano dan YOLO Nano pada mode RGB dan Skeleton."
)

st.sidebar.header("Sistem Akselerasi")
if DEVICE.type == "cuda":
    st.sidebar.success(f"✅ CUDA Aktif: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ Menggunakan Komputasi CPU.")

render_model_status()

st.sidebar.header("Pilihan Sumber Data")
source_option = st.sidebar.radio(
    "Pilih Mode:",
    ["Unggah Gambar", "Inferensi Realtime", "Unggah Video", "Mode Testing"],
)
conf_detection = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
alarm_delay = st.sidebar.slider(
    "Toleransi Waktu Alarm (detik)",
    1.0,
    30.0,
    5.0,
    0.5,
    help="Berapa lama waktu deteksi jatuh terus-menerus sebelum alarm berbunyi.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Catatan: mode gambar, video, dan testing memproses model satu per satu agar beban GPU/CPU tidak melonjak bersamaan."
)


# =========================================================
# MODE 1: upload image, run all models sequentially
# =========================================================
if source_option == "Unggah Gambar":
    st.header("🖼️ Unggah Gambar - Inferensi Semua Model")
    st.write(
        "Gambar akan diproses oleh empat pipeline secara bergantian: RF-DETR RGB, RF-DETR Skeleton, YOLO RGB, dan YOLO Skeleton."
    )

    uploaded_img = st.sidebar.file_uploader(
        "Pilih file gambar...", type=["jpg", "jpeg", "png"]
    )
    if uploaded_img:
        image_rgb = np.array(Image.open(uploaded_img).convert("RGB"))
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

        if st.button("Mulai Inferensi Semua Model", type="primary"):
            progress_container = st.empty()
            results, records = process_image_all_models(
                image_rgb, conf_detection, progress_container
            )
            st.session_state["image_results"] = results
            st.session_state["image_records"] = records

    if st.session_state.get("image_results"):
        display_image_results(
            st.session_state["image_results"], st.session_state.get("image_records", [])
        )


# =========================================================
# MODE 2: realtime, user chooses one model and one input mode
# =========================================================
elif source_option == "Inferensi Realtime":
    st.header("📷 Inferensi Realtime")
    st.write(
        "Pada mode realtime, pilih satu pipeline agar stream tetap ringan. Pilihan dibuat dari dua kolom: mode input dan model deteksi."
    )

    selected_cfg = select_realtime_model_config()
    st.sidebar.markdown("---")
    camera_source = render_camera_source_selector("rt")
    run_cam = st.sidebar.checkbox("🔴 Mulai Streaming Kamera")

    st.info(f"Pipeline aktif: **{selected_cfg.display_name}**")
    weight_path = resolve_model_path(selected_cfg)
    if weight_path is None:
        st.error(
            f"Weight untuk {selected_cfg.display_name} tidak ditemukan. Kandidat: {selected_cfg.weight_candidates}"
        )
    elif run_cam:
        pose_model = (
            load_pose_model_cached() if selected_cfg.input_mode == "Skeleton" else None
        )
        detector = load_detector_cached(selected_cfg.key, str(weight_path))
        cap, source_desc = open_camera_capture(camera_source)

        if cap is None or not cap.isOpened():
            st.error("❌ Tidak dapat membuka sumber kamera.")
            show_camera_troubleshooting(source_desc)
        else:
            st.success(f"✅ Kamera terhubung: {source_desc}. Stream berjalan...")
            metric_cols = st.columns([1, 1, 1, 1.5, 2])
            fps_ph, inf_ph, cpu_ph, ram_ph, gpu_ph = [
                col.empty() for col in metric_cols
            ]

            if selected_cfg.input_mode == "Skeleton":
                frame_cols = st.columns(3)
                titles = [
                    "1. RGB Letterbox",
                    "2. Skeleton Pose",
                    f"3. Prediksi {selected_cfg.model_family}",
                ]
                for col, title in zip(frame_cols, titles):
                    col.subheader(title)
                fp1, fp2, fp3 = [col.empty() for col in frame_cols]
            else:
                frame_cols = st.columns(2)
                titles = [
                    "1. RGB Letterbox",
                    f"2. Prediksi {selected_cfg.model_family}",
                ]
                for col, title in zip(frame_cols, titles):
                    col.subheader(title)
                fp1, fp3 = [col.empty() for col in frame_cols]
                fp2 = None

            det_text = st.empty()
            alarm_placeholder = st.empty()
            prev_metrics = None
            fall_start_time = None
            alarm_active = False

            while run_cam:
                ret, frame_bgr = cap.read()
                if not ret:
                    st.error("Sinyal kamera terputus.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                padded, skeleton, annotated, pred, metrics = process_single_frame(
                    frame_rgb,
                    selected_cfg,
                    detector,
                    pose_model,
                    conf_detection,
                )

                fp1.image(padded, channels="RGB")
                if fp2 is not None and skeleton is not None:
                    fp2.image(skeleton, channels="RGB")
                fp3.image(annotated, channels="RGB")
                det_text.write(f"**Deteksi:** {format_detection_result(pred)}")

                # Cek deteksi jatuh secara terus menerus selama waktu toleransi yang ditentukan
                is_fall_detected = False
                if pred and pred.get("class_id") is not None:
                    is_fall_detected = any(int(cid) == 0 for cid in pred["class_id"])

                if is_fall_detected:
                    if fall_start_time is None:
                        fall_start_time = time.time()
                    else:
                        elapsed = time.time() - fall_start_time
                        if elapsed >= alarm_delay:
                            if not alarm_active:
                                alarm_html = get_alarm_audio_html()
                                if alarm_html:
                                    alarm_placeholder.markdown(alarm_html, unsafe_allow_html=True)
                                    st.toast("🚨 DETEKSI JATUH TERUS MENERUS! ALARM AKTIF!", icon="🚨")
                                else:
                                    alarm_placeholder.error(f"🚨 ALARM: Jatuh terdeteksi terus menerus selama {alarm_delay} detik (file suara tidak ditemukan di assets/sound)!")
                                alarm_active = True
                else:
                    fall_start_time = None
                    if alarm_active:
                        alarm_placeholder.empty()
                        alarm_active = False

                mstr = {
                    "fps": f"{metrics['fps']:.2f}",
                    "inf": f"{metrics['inference_ms']:.2f}",
                    "cpu": f"{metrics['cpu_usage']:.1f}",
                    "ram": f"{metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}",
                    "gpu": (
                        f"{metrics['gpu']['util_percent']:.1f}% | {metrics['gpu']['used_gb']:.1f}/{metrics['gpu']['total_gb']:.1f} GB"
                        if metrics.get("gpu")
                        and metrics["gpu"]["util_percent"] is not None
                        else (
                            f"{metrics['gpu']['used_gb']:.1f}/{metrics['gpu']['total_gb']:.1f} GB"
                            if metrics.get("gpu")
                            else "N/A"
                        )
                    ),
                }
                if mstr != prev_metrics:
                    fps_ph.metric("FPS", mstr["fps"])
                    inf_ph.metric("Inferensi (ms)", mstr["inf"])
                    cpu_ph.metric("CPU (%)", mstr["cpu"])
                    ram_ph.metric("RAM (GB)", mstr["ram"])
                    gpu_ph.metric("GPU Util & VRAM", mstr["gpu"])
                    prev_metrics = mstr

            cap.release()
            alarm_placeholder.empty()


# =========================================================
# MODE 3: upload video, run all models sequentially
# =========================================================
elif source_option == "Unggah Video":
    st.header("🎞️ Unggah Video - Inferensi Semua Model")
    st.write(
        "Video diproses oleh semua model secara bergantian. Setiap model menghasilkan video output dan CSV metrik."
    )

    uploaded_video = st.sidebar.file_uploader(
        "Pilih file video...", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_video:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        temp_orig = (
            Path(tempfile.gettempdir()) / f"uploaded_{int(time.time() * 1000)}{suffix}"
        )
        temp_orig.write_bytes(uploaded_video.read())
        st.video(read_file_bytes(temp_orig))

        if st.button("Mulai Proses Semua Model", type="primary"):
            video_outputs = {}
            all_records = []
            total = len(MODEL_CONFIGS)
            overall = st.progress(0)
            status = st.empty()

            for idx, cfg in enumerate(MODEL_CONFIGS):
                status.info(f"Memproses {cfg.display_name} ({idx + 1}/{total})...")
                per_model_progress = st.progress(0)
                output_path, mime, records, error = process_video_for_model(
                    temp_orig, cfg, conf_detection, per_model_progress, status
                )
                per_model_progress.empty()

                if error:
                    video_outputs[cfg.key] = {"error": error}
                else:
                    video_outputs[cfg.key] = {"path": output_path, "mime": mime}
                    all_records.extend(records)
                overall.progress((idx + 1) / total)

            status.success("Semua model selesai memproses video.")
            st.session_state["video_outputs"] = video_outputs
            st.session_state["video_records"] = all_records

    if st.session_state.get("video_outputs"):
        display_video_downloads(
            st.session_state["video_outputs"],
            st.session_state.get("video_records", []),
            prefix="uploaded_video",
        )


# =========================================================
# MODE 4: testing, capture frames once and process all models sequentially
# =========================================================
elif source_option == "Mode Testing":
    st.header("🧪 Mode Testing Kamera - Semua Model")
    st.write(
        "Mode ini merekam frame kamera terlebih dahulu, lalu memproses hasil rekaman tersebut oleh semua model secara bergantian. "
        "Dengan begitu, semua model diuji pada frame yang sama tanpa menjalankan empat model sekaligus."
    )

    camera_source_test = render_camera_source_selector("test")
    st.sidebar.markdown("### Pengaturan Testing")
    duration_sec = st.sidebar.number_input(
        "Durasi capture (detik)", min_value=1.0, max_value=120.0, value=10.0, step=1.0
    )
    target_capture_fps = st.sidebar.number_input(
        "Target FPS capture", min_value=1.0, max_value=30.0, value=10.0, step=1.0
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        capture_btn = st.button("1) Rekam Frame Testing", type="primary")
    with col_b:
        process_btn = st.button("2) Proses Semua Model")
    with col_c:
        clear_btn = st.button("Reset Testing")

    if clear_btn:
        for key in [
            "test_frames",
            "test_capture_fps",
            "testing_outputs",
            "testing_records",
        ]:
            st.session_state.pop(key, None)
        st.success("Data testing direset.")

    if capture_btn:
        st.info("Merekam frame dari kamera...")
        frames, actual_fps, error = capture_camera_frames(
            camera_source_test, float(duration_sec), float(target_capture_fps)
        )
        if error:
            st.error(error)
        else:
            st.session_state["test_frames"] = frames
            st.session_state["test_capture_fps"] = actual_fps
            st.success(
                f"Capture selesai: {len(frames)} frame, estimasi FPS capture {actual_fps:.2f}."
            )

    if st.session_state.get("test_frames"):
        frames = st.session_state["test_frames"]
        actual_fps = float(st.session_state.get("test_capture_fps", target_capture_fps))
        st.info(
            f"Frame testing tersedia: {len(frames)} frame | FPS capture: {actual_fps:.2f}"
        )
        preview_cols = st.columns(3)
        if len(frames) > 0:
            preview_cols[0].image(
                frames[0], caption="Frame awal", use_container_width=True
            )
            preview_cols[1].image(
                frames[len(frames) // 2],
                caption="Frame tengah",
                use_container_width=True,
            )
            preview_cols[2].image(
                frames[-1], caption="Frame akhir", use_container_width=True
            )

    if process_btn:
        frames = st.session_state.get("test_frames", [])
        if not frames:
            st.warning("Rekam frame testing terlebih dahulu.")
        else:
            capture_fps = float(
                st.session_state.get("test_capture_fps", target_capture_fps)
            )
            testing_outputs = {}
            all_records = []
            total = len(MODEL_CONFIGS)
            overall = st.progress(0)
            status = st.empty()

            for idx, cfg in enumerate(MODEL_CONFIGS):
                status.info(
                    f"Memproses testing dengan {cfg.display_name} ({idx + 1}/{total})..."
                )
                per_model_progress = st.progress(0)
                output_path, mime, records, error = process_recorded_frames_for_model(
                    frames, cfg, conf_detection, capture_fps, per_model_progress
                )
                per_model_progress.empty()
                if error:
                    testing_outputs[cfg.key] = {"error": error}
                else:
                    testing_outputs[cfg.key] = {"path": output_path, "mime": mime}
                    all_records.extend(records)
                overall.progress((idx + 1) / total)

            status.success("Testing semua model selesai.")
            st.session_state["testing_outputs"] = testing_outputs
            st.session_state["testing_records"] = all_records

    if st.session_state.get("testing_outputs"):
        display_video_downloads(
            st.session_state["testing_outputs"],
            st.session_state.get("testing_records", []),
            prefix="camera_testing",
        )
