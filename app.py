import streamlit as st
import numpy as np
import cv2
import torch
import supervision as sv
import argparse
import time
from PIL import Image
from ultralytics import YOLO
from rfdetr import RFDETRNano
from pytorch_lightning.callbacks import Callback
import psutil
import io
import csv
import tempfile
import uuid
import base64

# Coba import moviepy (opsional untuk rotasi & encoding)
try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import GPUtil
except ImportError:
    GPUtil = None


# --- FIX PYTORCH UNPICKLING ---
class EpochTimerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pass


torch.serialization.add_safe_globals([argparse.Namespace, EpochTimerCallback])

# --- CONFIG ---
st.set_page_config(page_title="Privacy-Preserving Fall Detection", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Fall", "No-Fall"]
TARGET_SIZE = 640
PAD_COLOR = (114, 114, 114)

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


# Helper: letterbox
def letterbox_image(img_array: np.ndarray) -> np.ndarray:
    h_ori, w_ori = img_array.shape[:2]
    scale = min(TARGET_SIZE / w_ori, TARGET_SIZE / h_ori)
    new_w, new_h = int(round(w_ori * scale)), int(round(h_ori * scale))
    pad_w, pad_h = (TARGET_SIZE - new_w) / 2.0, (TARGET_SIZE - new_h) / 2.0
    img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR
    )


# Helper: skeleton extraction
def extract_skeleton_canvas(img_640: np.ndarray, pose_model: YOLO) -> np.ndarray:
    canvas = np.zeros(img_640.shape, dtype=np.uint8)
    results = pose_model(img_640, verbose=False, device=DEVICE.type)
    if len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        for person_kpts in keypoints_data:
            if np.all(person_kpts[:, 2] == 0):
                continue
            # Draw bones
            for edge in SKELETON_EDGES:
                pt1, pt2 = person_kpts[edge[0]], person_kpts[edge[1]]
                x1, y1, conf1 = int(pt1[0]), int(pt1[1]), pt1[2]
                x2, y2, conf2 = int(pt2[0]), int(pt2[1]), pt2[2]
                if (conf1 > CONF_THRESHOLD_POSE and conf2 > CONF_THRESHOLD_POSE) and (
                    x1 != 0 and x2 != 0
                ):
                    cv2.line(canvas, (x1, y1), (x2, y2), BONE_COLOR, 5)
            # Draw joints
            for pt in person_kpts:
                x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                if conf > CONF_THRESHOLD_POSE and x != 0:
                    cv2.circle(canvas, (x, y), 6, JOINT_COLOR, -1)
    return canvas


# Load models (cached)
@st.cache_resource
def load_models():
    pose_model = YOLO("models/yolo11l-pose.pt")
    pose_model.to(DEVICE.type)
    rfdetr_model = RFDETRNano(
        num_classes=2, pretrain_weights="models/rfdetr_nano_skeleton.pth"
    )
    rfdetr_model.optimize_for_inference()
    return pose_model, rfdetr_model


# Get system metrics
def get_system_metrics(start_time=None, end_time=None):
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


# Format detections string
def format_detections(raw_dets):
    if (
        hasattr(raw_dets, "xyxy")
        and raw_dets.xyxy is not None
        and len(raw_dets.xyxy) > 0
    ):
        parts = []
        for cls_id, conf in zip(raw_dets.class_id, raw_dets.confidence):
            parts.append(f"{CLASSES[int(cls_id)]}:{conf:.2f}")
        return " | ".join(parts)
    return "None"


# UI header
st.title("🤸‍♂️ Privacy-Preserving Fall Detection (Real-Time)")
st.sidebar.header("Sistem Akselerasi")
if DEVICE.type == "cuda":
    st.sidebar.success(f"✅ CUDA Aktif: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ Menggunakan Komputasi CPU.")
with st.spinner("Memuat model..."):
    pose_model, rfdetr_model = load_models()

st.sidebar.header("Pilihan Sumber Data")
source_option = st.sidebar.radio(
    "Pilih Metode Input:",
    ["Unggah Gambar", "Inferensi Realtime", "Unggah Video", "Mode Testing"],
)
conf_detection = st.sidebar.slider("RF-DETR Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# =========================================================
# MODE 1: UPLOAD IMAGE
if source_option == "Unggah Gambar":
    uploaded_img = st.sidebar.file_uploader(
        "Pilih file gambar...", type=["jpg", "jpeg", "png"]
    )
    if uploaded_img:
        image = np.array(Image.open(uploaded_img).convert("RGB"))
        # Inference pipeline
        start_t = time.time()
        padded = letterbox_image(image)
        skeleton = extract_skeleton_canvas(padded, pose_model)
        raw_dets = rfdetr_model.predict(skeleton, threshold=conf_detection)
        end_t = time.time()
        metrics = get_system_metrics(start_t, end_t)
        det_str = format_detections(raw_dets)
        # Annotate skeleton for detection boxes
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
            box_annot = sv.BoxAnnotator(thickness=4)
            label_annot = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
            labels = [
                f"{CLASSES[int(c)]} {conf:.2f}"
                for c, conf in zip(dets.class_id, dets.confidence)
            ]
            annotated = box_annot.annotate(scene=skeleton.copy(), detections=dets)
            annotated = label_annot.annotate(
                scene=annotated, detections=dets, labels=labels
            )
        else:
            annotated = skeleton.copy()
        # Metrics display
        top_cols = st.columns([1, 1, 1, 1.5, 2])
        fps_str = f"{metrics['fps']:.2f}" if metrics["fps"] is not None else "N/A"
        inf_str = (
            f"{metrics['inference_ms']:.2f}"
            if metrics["inference_ms"] is not None
            else "N/A"
        )
        top_cols[0].metric("FPS", fps_str)
        top_cols[1].metric("Inferensi (ms)", inf_str)
        top_cols[2].metric("CPU (%)", f"{metrics['cpu_usage']:.1f}")
        top_cols[3].metric(
            "RAM (GB)", f"{metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}"
        )
        if metrics["gpu"]:
            gpu = metrics["gpu"]
            util = gpu["util_percent"]
            gpu_str = (
                f"{util:.1f}% | {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB"
                if util is not None
                else "N/A"
            )
            top_cols[4].metric("GPU Util & VRAM", gpu_str)
        else:
            top_cols[4].metric("GPU Util & VRAM", "N/A")
        # Images
        col1, col2, col3 = st.columns(3)
        col1.subheader("1. RGB Masukan")
        col1.image(padded, use_container_width=True)
        col2.subheader("2. Skeleton Pose")
        col2.image(skeleton, use_container_width=True)
        col3.subheader("3. Prediksi RF-DETR")
        col3.image(annotated, use_container_width=True)
        st.write(f"**Deteksi:** {det_str}")

# =========================================================
# MODE 2: REALTIME INFERENCE (CAMERA)
elif source_option == "Inferensi Realtime":
    st.sidebar.markdown("---")
    cam_id = st.sidebar.selectbox(
        "Pilih ID Kamera:",
        [0, 1, 2, 3, 4],
        format_func=lambda x: f"Kamera ID {x} {'(Default)' if x == 0 else ''}",
    )
    run_cam = st.sidebar.checkbox("🔴 Mulai Streaming Kamera")
    if run_cam:
        cap = (
            cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if cv2.CAP_DSHOW
            else cv2.VideoCapture(cam_id)
        )
        if not cap.isOpened():
            st.error(f"❌ Tidak dapat membuka Kamera ID {cam_id}.")
        else:
            st.success(f"✅ Kamera ID {cam_id} terhubung! Stream berjalan...")
            top_cols = st.columns([1, 1, 1, 1.5, 2])
            fps_ph, inf_ph, cpu_ph, ram_ph, gpu_ph = [col.empty() for col in top_cols]
            frame_cols = st.columns(3)
            for col, title in zip(
                frame_cols,
                ["1. RGB Masukan", "2. Skeleton Pose", "3. Prediksi RF-DETR"],
            ):
                col.subheader(title)
            fp1, fp2, fp3 = [col.empty() for col in frame_cols]
            prev_metrics = None
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Sinyal kamera terputus.")
                    break
                start_t = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                padded = letterbox_image(rgb)
                skeleton = extract_skeleton_canvas(padded, pose_model)
                raw_dets = rfdetr_model.predict(skeleton, threshold=conf_detection)
                end_t = time.time()
                # Annotate
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
                    box_ann = sv.BoxAnnotator(thickness=4)
                    label_ann = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
                    labels = [
                        f"{CLASSES[int(c)]} {conf:.2f}"
                        for c, conf in zip(dets.class_id, dets.confidence)
                    ]
                    annotated = box_ann.annotate(scene=skeleton.copy(), detections=dets)
                    annotated = label_ann.annotate(
                        scene=annotated, detections=dets, labels=labels
                    )
                else:
                    annotated = skeleton.copy()
                fp1.image(padded, channels="RGB")
                fp2.image(skeleton, channels="RGB")
                fp3.image(annotated, channels="RGB")
                metrics = get_system_metrics(start_t, end_t)
                # metrics display
                mstr = {
                    "fps": f"{metrics['fps']:.2f}",
                    "inf": f"{metrics['inference_ms']:.2f}",
                    "cpu": f"{metrics['cpu_usage']:.1f}",
                    "ram": f"{metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}",
                    "gpu": (
                        f"{metrics['gpu']['util_percent']:.1f}% | "
                        f"{metrics['gpu']['used_gb']:.1f}/{metrics['gpu']['total_gb']:.1f}"
                        if metrics["gpu"] and metrics["gpu"]["util_percent"] is not None
                        else "N/A"
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

# =========================================================
# MODE 3: UPLOAD VIDEO
elif source_option == "Unggah Video":
    uploaded_video = st.sidebar.file_uploader(
        "Pilih file video...", type=["mp4", "avi", "mov"]
    )
    if uploaded_video:
        if st.button("Mulai Proses Video"):
            st.info("Video sedang di-resize (letterboxing) dan di-inferensi...")
            # Save original video to disk
            temp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_orig.write(uploaded_video.read())
            temp_orig.flush()
            records = []
            annotated_frames = []
            fps = 25
            # If MoviePy is available, use it to handle rotation and extraction
            if MOVIEPY_AVAILABLE:
                clip = VideoFileClip(temp_orig.name)
                fps = clip.fps or 25
                total_frames = int(clip.reader.nframes)
                progress = st.progress(0)
                for idx, frame in enumerate(clip.iter_frames()):
                    start_t = time.time()
                    # frame already oriented correctly (RGB)
                    padded = letterbox_image(frame)
                    skeleton = extract_skeleton_canvas(padded, pose_model)
                    raw_dets = rfdetr_model.predict(skeleton, threshold=conf_detection)
                    end_t = time.time()
                    # annotate skeleton
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
                        box_ann = sv.BoxAnnotator(thickness=2)
                        label_ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.6)
                        labels = [
                            f"{CLASSES[int(c)]} {conf:.2f}"
                            for c, conf in zip(dets.class_id, dets.confidence)
                        ]
                        annotated = box_ann.annotate(
                            scene=skeleton.copy(), detections=dets
                        )
                        annotated = label_ann.annotate(
                            scene=annotated, detections=dets, labels=labels
                        )
                    else:
                        annotated = skeleton.copy()
                    annotated_frames.append(annotated)
                    # metrics & records
                    metrics = get_system_metrics(start_t, end_t)
                    timestamp = idx / fps
                    records.append(
                        {
                            "frame_index": idx,
                            "timestamp_sec": f"{timestamp:.2f}",
                            "fps": f"{metrics['fps']:.2f}"
                            if metrics["fps"] is not None
                            else "N/A",
                            "inference_ms": f"{metrics['inference_ms']:.2f}"
                            if metrics["inference_ms"] is not None
                            else "N/A",
                            "cpu_percent": f"{metrics['cpu_usage']:.1f}",
                            "ram_used_gb": f"{metrics['ram_used_gb']:.1f}",
                            "ram_total_gb": f"{metrics['ram_total_gb']:.1f}",
                            "gpu_util_percent": (
                                f"{metrics['gpu']['util_percent']:.1f}"
                                if metrics["gpu"]
                                and metrics["gpu"]["util_percent"] is not None
                                else "N/A"
                            ),
                            "gpu_used_gb": (
                                f"{metrics['gpu']['used_gb']:.1f}"
                                if metrics["gpu"]
                                else "N/A"
                            ),
                            "gpu_total_gb": (
                                f"{metrics['gpu']['total_gb']:.1f}"
                                if metrics["gpu"]
                                else "N/A"
                            ),
                            "detections": format_detections(raw_dets),
                        }
                    )
                    progress.progress((idx + 1) / total_frames)
                clip.close()
                # encode annotated video with MoviePy
                st.info("Menyimpan video inferensi (H264)...")
                temp_annot = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip_out = ImageSequenceClip(
                    [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in annotated_frames],
                    fps=fps,
                )
                clip_out.write_videofile(
                    temp_annot.name,
                    codec="libx264",
                    audio=False,
                    verbose=False,
                    logger=None,
                )
                clip_out.close()
            else:
                # Fallback using OpenCV if MoviePy not available (orientation may be wrong)
                cap = cv2.VideoCapture(temp_orig.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                # Use MJPG codec to ensure cross-platform support; output as AVI
                temp_annot = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out_writer = cv2.VideoWriter(
                    temp_annot.name, fourcc, fps, (TARGET_SIZE, TARGET_SIZE)
                )
                progress = st.progress(0)
                for idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    start_t = time.time()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    padded = letterbox_image(rgb)
                    skeleton = extract_skeleton_canvas(padded, pose_model)
                    raw_dets = rfdetr_model.predict(skeleton, threshold=conf_detection)
                    end_t = time.time()
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
                        box_ann = sv.BoxAnnotator(thickness=2)
                        label_ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.6)
                        labels = [
                            f"{CLASSES[int(c)]} {conf:.2f}"
                            for c, conf in zip(dets.class_id, dets.confidence)
                        ]
                        annotated = box_ann.annotate(
                            scene=skeleton.copy(), detections=dets
                        )
                        annotated = label_ann.annotate(
                            scene=annotated, detections=dets, labels=labels
                        )
                    else:
                        annotated = skeleton.copy()
                    annotated_frames.append(annotated)
                    out_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    metrics = get_system_metrics(start_t, end_t)
                    timestamp = idx / fps
                    records.append(
                        {
                            "frame_index": idx,
                            "timestamp_sec": f"{timestamp:.2f}",
                            "fps": f"{metrics['fps']:.2f}"
                            if metrics["fps"] is not None
                            else "N/A",
                            "inference_ms": f"{metrics['inference_ms']:.2f}"
                            if metrics["inference_ms"] is not None
                            else "N/A",
                            "cpu_percent": f"{metrics['cpu_usage']:.1f}",
                            "ram_used_gb": f"{metrics['ram_used_gb']:.1f}",
                            "ram_total_gb": f"{metrics['ram_total_gb']:.1f}",
                            "gpu_util_percent": (
                                f"{metrics['gpu']['util_percent']:.1f}"
                                if metrics["gpu"]
                                and metrics["gpu"]["util_percent"] is not None
                                else "N/A"
                            ),
                            "gpu_used_gb": (
                                f"{metrics['gpu']['used_gb']:.1f}"
                                if metrics["gpu"]
                                else "N/A"
                            ),
                            "gpu_total_gb": (
                                f"{metrics['gpu']['total_gb']:.1f}"
                                if metrics["gpu"]
                                else "N/A"
                            ),
                            "detections": format_detections(raw_dets),
                        }
                    )
                    progress.progress((idx + 1) / total_frames)
                cap.release()
                out_writer.release()
                st.info(
                    "Proses video selesai. Video inferensi disimpan sebagai AVI (codec MJPG)."
                )
            st.success("Proses video selesai!")

            # Video playback and download
            # Read bytes
            orig_bytes = open(temp_orig.name, "rb").read()
            annot_bytes = open(temp_annot.name, "rb").read()
            orig_b64 = base64.b64encode(orig_bytes).decode()
            annot_b64 = base64.b64encode(annot_bytes).decode()
            # Build HTML for playback
            html_player = f"""
                <button onclick="document.getElementById('origVid').play();document.getElementById('annotVid').play();" style="margin-bottom:10px;">Play Both</button>
                <div style='display:flex; gap:20px;'>
                    <video id='origVid' width='320' controls>
                        <source src='data:video/mp4;base64,{orig_b64}' type='video/mp4'>
                    </video>
                    <video id='annotVid' width='320' controls>
                        <source src='data:video/mp4;base64,{annot_b64}' type='video/mp4'>
                    </video>
                </div>
            """
            st.iframe(srcdoc=html_player, height=380)

            # CSV download
            csv_io = io.StringIO()
            writer = csv.DictWriter(csv_io, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
            csv_data = csv_io.getvalue().encode("utf-8")
            st.download_button(
                "Download CSV Metrik",
                csv_data,
                file_name="video_metrics.csv",
                mime="text/csv",
            )
            # Video download
            if MOVIEPY_AVAILABLE:
                st.download_button(
                    "Download Video Inferensi",
                    annot_bytes,
                    file_name="annotated_video.mp4",
                    mime="video/mp4",
                )
            else:
                st.download_button(
                    "Download Video Inferensi (AVI)",
                    annot_bytes,
                    file_name="annotated_video.avi",
                    mime="video/avi",
                )

# =========================================================
# MODE 4: MODE TESTING
elif source_option == "Mode Testing":
    st.sidebar.markdown("---")
    cam_id_test = st.sidebar.selectbox(
        "Pilih ID Kamera:",
        [0, 1, 2, 3, 4],
        format_func=lambda x: f"Kamera ID {x} {'(Default)' if x == 0 else ''}",
    )

    # Session state initialization
    if "test_active" not in st.session_state:
        st.session_state.test_active = False
    if "test_records" not in st.session_state:
        st.session_state.test_records = []
    if "test_start_time" not in st.session_state:
        st.session_state.test_start_time = None
    if "test_frames" not in st.session_state:
        st.session_state.test_frames = []

    # Buttons
    if not st.session_state.test_active:
        if st.button("Mulai Test"):
            st.session_state.test_active = True
            st.session_state.test_records = []
            st.session_state.test_frames = []
            st.session_state.test_start_time = time.time()
    else:
        if st.button("Stop Test"):
            st.session_state.test_active = False

    # During test
    if st.session_state.test_active:
        cap = (
            cv2.VideoCapture(cam_id_test, cv2.CAP_DSHOW)
            if cv2.CAP_DSHOW
            else cv2.VideoCapture(cam_id_test)
        )
        if not cap.isOpened():
            st.error("❌ Tidak dapat membuka kamera untuk testing.")
        else:
            top_cols = st.columns([1, 1, 1, 1.5, 2])
            fps_ph, inf_ph, cpu_ph, ram_ph, gpu_ph = [col.empty() for col in top_cols]
            frame_cols = st.columns(3)
            for col, title in zip(
                frame_cols,
                ["1. RGB Masukan", "2. Skeleton Pose", "3. Prediksi RF-DETR"],
            ):
                col.subheader(title)
            fp1, fp2, fp3 = [col.empty() for col in frame_cols]
            prev_metrics = None
            while st.session_state.test_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Sinyal kamera terputus saat testing.")
                    break
                start_t = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                padded = letterbox_image(rgb)
                skeleton = extract_skeleton_canvas(padded, pose_model)
                raw_dets = rfdetr_model.predict(skeleton, threshold=conf_detection)
                end_t = time.time()

                # Annotate
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
                    box_ann = sv.BoxAnnotator(thickness=4)
                    label_ann = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
                    labels = [
                        f"{CLASSES[int(c)]} {conf:.2f}"
                        for c, conf in zip(dets.class_id, dets.confidence)
                    ]
                    annotated = box_ann.annotate(scene=skeleton.copy(), detections=dets)
                    annotated = label_ann.annotate(
                        scene=annotated, detections=dets, labels=labels
                    )
                else:
                    annotated = skeleton.copy()

                # Save frame for final video
                st.session_state.test_frames.append(annotated)

                fp1.image(padded, channels="RGB")
                fp2.image(skeleton, channels="RGB")
                fp3.image(annotated, channels="RGB")

                metrics = get_system_metrics(start_t, end_t)
                detections_str = format_detections(raw_dets)
                elapsed_video = time.time() - st.session_state.test_start_time
                st.session_state.test_records.append(
                    {
                        "timestamp_sec": f"{elapsed_video:.2f}",
                        "fps": f"{metrics['fps']:.2f}"
                        if metrics["fps"] is not None
                        else "N/A",
                        "inference_ms": f"{metrics['inference_ms']:.2f}"
                        if metrics["inference_ms"] is not None
                        else "N/A",
                        "cpu_percent": f"{metrics['cpu_usage']:.1f}",
                        "ram_used_gb": f"{metrics['ram_used_gb']:.1f}",
                        "ram_total_gb": f"{metrics['ram_total_gb']:.1f}",
                        "gpu_util_percent": (
                            f"{metrics['gpu']['util_percent']:.1f}"
                            if metrics["gpu"]
                            and metrics["gpu"]["util_percent"] is not None
                            else "N/A"
                        ),
                        "gpu_used_gb": (
                            f"{metrics['gpu']['used_gb']:.1f}"
                            if metrics["gpu"]
                            else "N/A"
                        ),
                        "gpu_total_gb": (
                            f"{metrics['gpu']['total_gb']:.1f}"
                            if metrics["gpu"]
                            else "N/A"
                        ),
                        "detections": detections_str,
                    }
                )
                # Show metrics if changed
                current_metrics = {
                    "fps": f"{metrics['fps']:.2f}",
                    "inf": f"{metrics['inference_ms']:.2f}",
                    "cpu": f"{metrics['cpu_usage']:.1f}",
                    "ram": f"{metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}",
                    "gpu": (
                        f"{metrics['gpu']['util_percent']:.1f}% | "
                        f"{metrics['gpu']['used_gb']:.1f}/{metrics['gpu']['total_gb']:.1f}"
                        if metrics["gpu"] and metrics["gpu"]["util_percent"] is not None
                        else "N/A"
                    ),
                }
                if current_metrics != prev_metrics:
                    fps_ph.metric("FPS", current_metrics["fps"])
                    inf_ph.metric("Inferensi (ms)", current_metrics["inf"])
                    cpu_ph.metric("CPU (%)", current_metrics["cpu"])
                    ram_ph.metric("RAM (GB)", current_metrics["ram"])
                    gpu_ph.metric("GPU Util & VRAM", current_metrics["gpu"])
                    prev_metrics = current_metrics
            cap.release()

    # After test finishes, provide downloads
    if (
        not st.session_state.test_active
        and st.session_state.test_records
        and st.session_state.test_frames
    ):
        st.success("Testing selesai! Menulis video dan menyiapkan unduhan...")
        # Save video
        fps_est = 25
        tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        if MOVIEPY_AVAILABLE:
            clip = ImageSequenceClip(
                [
                    cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    for f in st.session_state.test_frames
                ],
                fps=fps_est,
            )
            clip.write_videofile(
                tmp_vid.name, codec="libx264", audio=False, verbose=False, logger=None
            )
            clip.close()
        else:
            # Fallback: use MJPG AVI
            tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(
                tmp_vid.name, fourcc, fps_est, (TARGET_SIZE, TARGET_SIZE)
            )
            for f in st.session_state.test_frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
        # Save CSV
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer, fieldnames=list(st.session_state.test_records[0].keys())
        )
        writer.writeheader()
        writer.writerows(st.session_state.test_records)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        # Download buttons
        with open(tmp_vid.name, "rb") as f_vid:
            video_bytes = f_vid.read()
        st.download_button(
            "Download CSV Metrik",
            csv_bytes,
            file_name="testing_metrics.csv",
            mime="text/csv",
        )
        if MOVIEPY_AVAILABLE:
            st.download_button(
                "Download Video Testing",
                video_bytes,
                file_name="testing_output.mp4",
                mime="video/mp4",
            )
        else:
            st.download_button(
                "Download Video Testing (AVI)",
                video_bytes,
                file_name="testing_output.avi",
                mime="video/avi",
            )
