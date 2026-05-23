# Hybrid Streamlit + Native Windows Camera + Docker GPU

Solusi ini menampilkan hasil di Streamlit, tetapi kualitas kamera tetap mendekati native Windows.

## Arsitektur

```text
Windows webcam
   ↓
host_camera_mjpeg_bridge.py  ← berjalan native Windows
   ↓ raw.mjpg dan result.mjpg
Browser Streamlit Viewer     ← http://localhost:8501
   ↑
Docker GPU API               ← YOLO11l-pose + RF-DETR di CUDA
```

## Kenapa ini lebih baik dari WebRTC Streamlit?

Versi WebRTC membuat kamera masuk lewat browser dan callback Python Streamlit.
Kualitas bisa turun karena kompresi WebRTC dan proses render Streamlit.

Versi hybrid ini:
- kamera dibuka native Windows dengan `cv2.CAP_DSHOW`;
- Docker hanya menjalankan model/inference;
- Streamlit hanya menampilkan MJPEG stream, bukan memproses frame;
- raw camera dan result bisa tampil di halaman Streamlit.

## File yang perlu ditaruh di root project

```text
app_inference_api.py                  # dari fix sebelumnya
Dockerfile.gpu.api
docker-compose.hybrid.streamlit.yml
app_streamlit_hybrid_viewer.py
Dockerfile.streamlit.viewer
requirements_streamlit_viewer.txt
host_camera_mjpeg_bridge.py
requirements_host_mjpeg.txt
```

Pastikan model tetap ada:

```text
models/
├─ yolo11l-pose.pt
└─ rfdetr_nano_skeleton.pth
```

## 1. Jalankan Docker API + Streamlit Viewer

```powershell
docker compose -f docker-compose.hybrid.streamlit.yml up --build
```

Tunggu sampai API menampilkan:

```text
Models loaded successfully.
Uvicorn running on http://0.0.0.0:8000
```

## 2. Jalankan Native Camera Bridge di terminal Windows baru

```powershell
cd "D:\bang yoga\prototype_privacy_perserving_fall_detection"

py -3.10 -m venv .venv-camera
.\.venv-camera\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_host_mjpeg.txt
```

Jalankan bridge:

```powershell
python host_camera_mjpeg_bridge.py --camera 0 --width 1280 --height 720 --fps 30 --process-every 3 --target-size 640 --view triple
```

## 3. Buka Streamlit

```text
http://localhost:8501
```

Di halaman Streamlit akan tampil:
- Native Windows Camera - RAW
- Docker GPU Result

## Setting yang disarankan

Kualitas tinggi, dekat native:

```powershell
python host_camera_mjpeg_bridge.py --camera 0 --width 1280 --height 720 --fps 30 --process-every 3 --target-size 640 --view triple
```

Lebih ringan:

```powershell
python host_camera_mjpeg_bridge.py --camera 0 --width 1280 --height 720 --fps 30 --process-every 5 --target-size 512 --view triple
```

Prediksi saja, tidak triple view:

```powershell
python host_camera_mjpeg_bridge.py --camera 0 --width 1280 --height 720 --fps 30 --process-every 3 --target-size 640 --view annotated
```

## Endpoint debug

Native camera bridge:

```text
http://localhost:7001/status
http://localhost:7001/raw.mjpg
http://localhost:7001/result.mjpg
```

Docker API:

```text
http://localhost:8000/health
```

## Catatan

Solusi ini masih membutuhkan satu proses Python native di Windows untuk membuka kamera.
Itu memang kompromi paling stabil di Windows Docker: kamera native tetap berkualitas, model tetap berjalan di Docker GPU, dan tampilan akhir tetap di Streamlit.
