import cv2
import numpy as np

# Konfigurasi persis seperti di script preprocessing
SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Kepala / Wajah
    (5, 6),  # Bahu
    (5, 7),
    (7, 9),  # Lengan Kiri
    (6, 8),
    (8, 10),  # Lengan Kanan
    (5, 11),
    (6, 12),
    (11, 12),  # Torso (Badan)
    (11, 13),
    (13, 15),  # Kaki Kiri
    (12, 14),
    (14, 16),  # Kaki Kanan
]
JOINT_COLOR = (0, 255, 0)  # Hijau
BONE_COLOR = (255, 255, 255)  # Putih
CONF_THRESHOLD = 0.25
TARGET_SIZE = 640
PAD_COLOR = (114, 114, 114)  # Warna padding abu-abu


def letterbox_image(img_array):
    """Me-resize gambar menjadi 640x640 dengan metode padding (mempertahankan aspect ratio)"""
    h_ori, w_ori = img_array.shape[:2]
    scale = min(TARGET_SIZE / w_ori, TARGET_SIZE / h_ori)
    new_w, new_h = int(round(w_ori * scale)), int(round(h_ori * scale))

    pad_w = (TARGET_SIZE - new_w) / 2.0
    pad_h = (TARGET_SIZE - new_h) / 2.0

    img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR
    )
    return img_padded


def process_pipeline_step_1(rgb_image_array, pose_model, device):
    """
    Menjalankan: RGB -> Letterbox 640x640 -> YOLO Pose -> Kanvas Hitam 640x640
    Mengembalikan: Gambar RGB Padded, dan Gambar Kanvas Skeleton (keduanya 640x640)
    """
    # 1. Terapkan Letterboxing
    img_640 = letterbox_image(rgb_image_array)

    # 2. Siapkan kanvas hitam murni berukuran 640x640
    black_canvas = np.zeros(img_640.shape, dtype=np.uint8)

    # 3. Ekstrak keypoints dari gambar yang sudah di-padding
    results = pose_model(img_640, verbose=False, device=device)

    if len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        for person_kpts in keypoints_data:
            # Lewati jika orang ini tidak punya keypoints (kosong)
            if np.all(person_kpts[:, 2] == 0):
                continue

            # Gambar Tulang
            for edge in SKELETON_EDGES:
                pt1, pt2 = person_kpts[edge[0]], person_kpts[edge[1]]
                x1, y1, conf1 = int(pt1[0]), int(pt1[1]), pt1[2]
                x2, y2, conf2 = int(pt2[0]), int(pt2[1]), pt2[2]

                if (conf1 > CONF_THRESHOLD and conf2 > CONF_THRESHOLD) and (
                    x1 != 0 and x2 != 0
                ):
                    cv2.line(black_canvas, (x1, y1), (x2, y2), BONE_COLOR, 5)

            # Gambar Engsel
            for pt in person_kpts:
                x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                if conf > CONF_THRESHOLD and x != 0:
                    cv2.circle(black_canvas, (x, y), 6, JOINT_COLOR, -1)

    return img_640, black_canvas
