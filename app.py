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


# --- SOLUSI PYTORCH 2.6 UNPICKLING ERROR ---
class EpochTimerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pass


torch.serialization.add_safe_globals([argparse.Namespace, EpochTimerCallback])

# --- KONFIGURASI HALAMAN ---
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


# --- HELPER FUNCTIONS ---
def letterbox_image(img_array):
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


def extract_skeleton_canvas(img_640, pose_model):
    black_canvas = np.zeros(img_640.shape, dtype=np.uint8)
    results = pose_model(img_640, verbose=False, device=DEVICE.type)

    if len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        for person_kpts in keypoints_data:
            if np.all(person_kpts[:, 2] == 0):
                continue
            for edge in SKELETON_EDGES:
                pt1, pt2 = person_kpts[edge[0]], person_kpts[edge[1]]
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


@st.cache_resource
def load_models():
    pose_model = YOLO("models/yolo11l-pose.pt")
    pose_model.to(DEVICE.type)

    rfdetr_model = RFDETRNano(
        num_classes=2, pretrain_weights="models/rfdetr_nano_skeleton.pth"
    )
    rfdetr_model.optimize_for_inference()

    return pose_model, rfdetr_model


# --- UI STREAMLIT ---
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
    "Pilih Metode Input:", ["Unggah Gambar", "Kamera Real-Time (Hardware)"]
)
conf_detection = st.sidebar.slider("RF-DETR Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# ==========================================
# MODE 1: UPLOAD GAMBAR STATIS
# ==========================================
if source_option == "Unggah Gambar":
    uploaded_file = st.sidebar.file_uploader(
        "Pilih file gambar...", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        img_array = np.array(Image.open(uploaded_file).convert("RGB"))

        col1, col2, col3 = st.columns(3)
        img_640_padded = letterbox_image(img_array)
        skeleton_canvas = extract_skeleton_canvas(img_640_padded, pose_model)

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
                scene=skeleton_canvas.copy(), detections=dets
            )
            annotated_output = label_annotator.annotate(
                scene=annotated_output, detections=dets, labels=labels
            )
        else:
            annotated_output = skeleton_canvas.copy()

        col1.subheader("1. RGB Masukan")
        col1.image(img_640_padded, use_container_width=True)
        col2.subheader("2. Skeleton Pose")
        col2.image(skeleton_canvas, use_container_width=True)
        col3.subheader("3. Prediksi RF-DETR")
        col3.image(annotated_output, use_container_width=True)

# ==========================================
# MODE 2: KAMERA MULTI-DEVICE (REAL-TIME)
# ==========================================
elif source_option == "Kamera Real-Time (Hardware)":
    st.sidebar.markdown("---")
    # Dropdown untuk memilih index perangkat kamera
    cam_id = st.sidebar.selectbox(
        "Pilih ID Kamera:",
        [0, 1, 2, 3, 4],
        format_func=lambda x: f"Kamera ID {x} {'(Default)' if x == 0 else ''}",
    )
    run_camera = st.sidebar.checkbox("🔴 Mulai Streaming Kamera")

    if run_camera:
        # Inisialisasi tangkapan video menggunakan OpenCV
        # cv2.CAP_DSHOW sering membantu kamera eksternal (USB) berjalan lebih cepat di Windows
        cap = (
            cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if cv2.CAP_DSHOW
            else cv2.VideoCapture(cam_id)
        )

        if not cap.isOpened():
            st.error(
                f"❌ Tidak dapat membuka Kamera ID {cam_id}. Coba pilih ID lain di sidebar."
            )
        else:
            st.success(f"✅ Kamera ID {cam_id} terhubung! Stream berjalan...")

            # Buat placeholder kosong yang akan terus ditimpa frame baru
            col1, col2, col3 = st.columns(3)
            col1.subheader("1. RGB Masukan")
            col2.subheader("2. Skeleton Pose")
            col3.subheader("3. Prediksi RF-DETR")

            placeholder_1 = col1.empty()
            placeholder_2 = col2.empty()
            placeholder_3 = col3.empty()

            # Loop streaming real-time
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Sinyal kamera terputus.")
                    break

                # Ubah format BGR (OpenCV) ke RGB (Streamlit)
                img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # PIPELINE PROSES (Sama persis seperti statis)
                img_640_padded = letterbox_image(img_array)
                skeleton_canvas = extract_skeleton_canvas(img_640_padded, pose_model)
                raw_dets = rfdetr_model.predict(
                    skeleton_canvas, threshold=conf_detection
                )

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
                    label_annotator = sv.LabelAnnotator(
                        text_thickness=2, text_scale=0.8
                    )
                    labels = [
                        f"{CLASSES[int(cls_id)]} {conf_val:.2f}"
                        for cls_id, conf_val in zip(dets.class_id, dets.confidence)
                    ]
                    annotated_output = box_annotator.annotate(
                        scene=skeleton_canvas.copy(), detections=dets
                    )
                    annotated_output = label_annotator.annotate(
                        scene=annotated_output, detections=dets, labels=labels
                    )
                else:
                    annotated_output = skeleton_canvas.copy()

                # Update tampilan web dengan frame terbaru
                placeholder_1.image(img_640_padded, channels="RGB")
                placeholder_2.image(skeleton_canvas, channels="RGB")
                placeholder_3.image(annotated_output, channels="RGB")

            # Bersihkan resource saat kamera dimatikan
            cap.release()
