# Privacy-Preserving Fall Detection (Real-Time) 🤸‍♂️

Proyek ini adalah sistem deteksi jatuh (fall detection) yang mengutamakan privasi dengan menggunakan ekstraksi skeleton (kerangka tubuh). Sistem ini bekerja dengan mengubah gambar input menjadi representasi kerangka tubuh manusia sebelum melakukan deteksi, sehingga identitas visual pengguna tetap terjaga.

## 🌟 Fitur Utama
- **Privacy-Preserving:** Hanya menggunakan data skeleton untuk deteksi jatuh, membuang data visual sensitif.
- **Deteksi Real-Time:** Mendukung input dari kamera (webcam) secara langsung maupun unggahan gambar statis.
- **Multi-Model Pipeline:**
  - **YOLOv11-Pose:** Digunakan untuk ekstraksi keypoints (skeleton) yang akurat.
  - **RF-DETR Nano:** Model transformer yang ringan untuk klasifikasi "Fall" (Jatuh) atau "No-Fall" (Tidak Jatuh) dari data skeleton.
- **Antarmuka Streamlit:** UI yang intuitif dan mudah digunakan.
- **Akselerasi Hardware:** Mendukung penggunaan GPU (CUDA) untuk performa yang lebih cepat.

## 📁 Struktur Folder
```text
.
├── app.py                   # Entry point aplikasi (Streamlit)
├── requirements.txt         # Daftar dependensi Python
├── models/                  # Folder penyimpanan model (.pt / .pth)
│   ├── model_download.py    # Skrip untuk mengunduh model YOLO
│   └── ...
├── utils/                   # Fungsi pembantu (skeleton extractor, dll)
│   └── skeleton_extractor.py
└── error_log/               # Log kesalahan (jika ada)
```

## 🚀 Langkah Instalasi

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek di lingkungan lokal Anda:

### 1. Persiapan Lingkungan (Virtual Environment)
Disarankan untuk menggunakan virtual environment agar tidak mengganggu pustaka sistem lainnya.
```bash
python -m venv venv
# Aktifkan venv (Windows)
.\venv\Scripts\activate
# Aktifkan venv (Linux/Mac)
source venv/bin/activate
```

### 2. Instalasi Dependensi
Pastikan Anda memiliki koneksi internet untuk mengunduh library yang dibutuhkan. Proyek ini menggunakan PyTorch dengan dukungan CUDA 12.1 secara default.
```bash
pip install -r requirements.txt
```

### 3. Unduh Model
Anda perlu memastikan file model berada di folder `models/`. 

1. **Model YOLOv11-Pose:** Jalankan skrip berikut untuk mengunduh model YOLO:
   ```bash
   python models/model_download.py
   mv yolo11l-pose.pt models/
   ```
2. **Model RF-DETR:** Unduh secara manual file `rfdetr_nano_skeleton.pth` melalui tautan berikut dan letakkan di dalam folder `models/`:
   - [Download RF-DETR Model (Google Drive)](https://drive.google.com/file/d/1PRTq1m9J3B8jIxHCFaU54t5Ugyscqo6V/view?usp=drive_link)

### 4. Menjalankan Aplikasi
Setelah semua siap, jalankan aplikasi menggunakan Streamlit:
```bash
streamlit run app.py
```

## 🛠 Cara Penggunaan
1. Buka browser pada alamat yang tertera (biasanya `http://localhost:8501`).
2. Pilih metode input di Sidebar:
   - **Unggah Gambar:** Untuk menguji gambar yang sudah ada.
   - **Kamera Real-Time:** Untuk menggunakan kamera perangkat secara langsung.
3. Atur **RF-DETR Confidence Threshold** jika diperlukan untuk menyesuaikan sensitivitas deteksi.
4. Sistem akan menampilkan tiga kolom:
   - **RGB Masukan:** Gambar asli (sudah di-resize).
   - **Skeleton Pose:** Hasil ekstraksi kerangka tubuh pada kanvas hitam.
   - **Prediksi RF-DETR:** Hasil deteksi jatuh pada data skeleton.

## 💻 Teknologi yang Digunakan
- **Bahasa:** Python 3.x
- **UI Framework:** [Streamlit](https://streamlit.io/)
- **Deep Learning:** [PyTorch](https://pytorch.org/), [Ultralytics (YOLOv11)](https://ultralytics.com/)
- **Computer Vision:** OpenCV, Supervision
- **Model Arsitektur:** RF-DETR (Transformer-based)

---
*Dibuat untuk keperluan prototipe sistem keamanan berbasis privasi.*
