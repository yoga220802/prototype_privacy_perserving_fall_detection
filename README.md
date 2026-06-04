# Privacy-Preserving Fall Detection (Real-Time) 🤸‍♂️

Proyek ini adalah sistem deteksi jatuh (fall detection) yang mengutamakan privasi dengan menggunakan ekstraksi skeleton (kerangka tubuh). Sistem ini bekerja dengan mengubah gambar input menjadi representasi kerangka tubuh manusia sebelum melakukan deteksi, sehingga identitas visual pengguna tetap terjaga.

## 🌟 Fitur Utama
- **Privacy-Preserving:** Hanya menggunakan data skeleton untuk deteksi jatuh, membuang data visual sensitif.
- **Deteksi Real-Time:** Mendukung input dari kamera (webcam) secara langsung maupun unggahan gambar statis.
- **Multi-Model Pipeline:**
  - **YOLOv11-Pose:** Digunakan untuk ekstraksi keypoints (skeleton) yang akurat.
  - **RF-DETR Nano:** Model transformer yang ringan untuk klasifikasi "Fall" (Jatuh) atau "No-Fall" (Tidak Jatuh) dari data skeleton.
- **Alarm Suara Deteksi Jatuh:** Menyalakan alarm suara jika kondisi jatuh terdeteksi terus-menerus selama 5 detik tanpa perubahan kondisi.
- **Antarmuka Streamlit:** UI yang intuitif dan mudah digunakan.
- **Akselerasi Hardware:** Mendukung penggunaan GPU (CUDA) untuk performa yang lebih cepat.

## 📁 Struktur Folder
```text
.
├── app.py                   # Entry point aplikasi (Streamlit)
├── requirements.txt         # Daftar dependensi Python
├── assets/                  # Folder aset statis
│   └── sound/               # Penyimpanan file alarm (alarm.mp3/alarm.wav)
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
2. **Model RF-DETR & YOLO:** Unduh secara manual file model model yang akan digunakan melalui tautan berikut dan letakkan di dalam folder `models/`:
   - [Download Models Model (Google Drive)](https://drive.google.com/drive/folders/173npx31AImzFgmK3MBJ9XMj52PfUk7IB?usp=sharing)

### 4. Menjalankan Aplikasi
Setelah semua siap, jalankan aplikasi menggunakan Streamlit:
```bash
streamlit run app.py
```

## 🐳 Versi Docker
Proyek ini juga mendukung dijalankan dalam container Docker dengan GPU menggunakan nama container `fall-detection-gpu`.

1. Bangun image Docker:
```bash
docker compose build
```
2. Jalankan container:
```bash
docker compose up -d
```
3. Buka aplikasi di browser:
```text
http://localhost:8501
```

Lihat `DOCKER.md` untuk detail lengkap.

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

## 🔔 Fitur Alarm Suara
Prototipe ini dilengkapi dengan alarm suara yang akan berbunyi secara otomatis jika terdeteksi kondisi **Fall (Jatuh)** secara terus-menerus selama durasi toleransi waktu yang ditentukan (default **5 detik**).

### Mengonfigurasi Waktu Toleransi Alarm:
Anda dapat menyesuaikan berapa lama kondisi jatuh ditoleransi sebelum alarm dipicu:
* **Mode Streamlit Lokal (`app.py`):** Gunakan slider **"Toleransi Waktu Alarm (detik)"** yang terletak di Sidebar untuk memilih durasi antara 1.0 hingga 30.0 detik.
* **Mode Hybrid (`host_camera_mjpeg_bridge.py`):** Jalankan skrip bridge native di Windows dengan menyertakan argumen `--alarm-delay`. Contoh pemanggilan untuk delay 10 detik:
  ```powershell
  python host_camera_mjpeg_bridge.py --alarm-delay 10.0
  ```

### Cara Mengaktifkan:
1. Buat folder `assets/sound/` di direktori root proyek (telah dibuat secara default).
2. Simpan file suara alarm Anda di folder tersebut dengan nama `alarm.mp3`, `alarm.wav`, atau `alarm.ogg`.
3. Jalankan aplikasi seperti biasa. Sistem akan mendeteksi file suara tersebut secara otomatis saat alarm aktif.

### Perilaku Suara:
* **Mode Streamlit Lokal (`app.py`):** Suara alarm dimainkan secara internal di web browser melalui tag audio HTML5.
* **Mode Hybrid (`host_camera_mjpeg_bridge.py`):** Suara alarm dimainkan secara native melalui API Windows Media (MCI) pada speaker komputer host.
* **Penghentian Otomatis:** Ketika kondisi berubah (jatuh tidak lagi terdeteksi atau kamera dinonaktifkan), alarm suara akan berhenti secara instan.

### Fallback (Jika File Suara Tidak Ada):
Jika folder `assets/sound/` kosong atau file audio alarm belum disediakan, sistem memiliki mekanisme fallback agar aplikasi tetap berjalan aman tanpa crash:
* **Mode Streamlit Lokal (`app.py`):** Halaman web Streamlit menampilkan pesan peringatan visual berwarna merah: `"🚨 ALARM: Jatuh terdeteksi terus menerus (file suara tidak ditemukan di assets/sound)!"`
* **Mode Hybrid (`host_camera_mjpeg_bridge.py`):** Konsol terminal akan mencetak pesan peringatan `"🚨 [WARNING] ALARM AKTIF..."` disertai dengan bunyi beep bel bawaan sistem komputer (`\a`).

## 🖥️ Mode Hybrid (Streamlit + Windows Native Camera + Docker GPU)
Bagi pengguna yang ingin menjalankan pemrosesan model di dalam container Docker GPU tetapi tetap menginginkan kualitas input kamera yang optimal dekat dengan native Windows, proyek ini menyediakan Mode Hybrid.

Panduan lengkap mengenai arsitektur, instalasi, dan cara penggunaan Mode Hybrid dapat dibaca pada file dokumentasi terpisah: [HYBRID_STREAMLIT_NATIVE_CAMERA_README.md](file:///D:/prototype_privacy_perserving_fall_detection/HYBRID_STREAMLIT_NATIVE_CAMERA_README.md).

## 💻 Teknologi yang Digunakan
- **Bahasa:** Python 3.x
- **UI Framework:** [Streamlit](https://streamlit.io/)
- **Deep Learning:** [PyTorch](https://pytorch.org/), [Ultralytics (YOLOv11)](https://ultralytics.com/), [RF-DETR](https://rfdetr.roboflow.com/latest/)
- **Computer Vision:** OpenCV, Supervision
- **Model Arsitektur:** RF-DETR (Transformer-based)

---
*Dibuat untuk keperluan prototipe sistem keamanan berbasis privasi.*
