# Instalasi GPU RTX 4060 via venv

Panduan ini dibuat untuk menjalankan aplikasi Streamlit + YOLO Pose + RF-DETR dengan GPU NVIDIA RTX 4060.

## Versi yang disarankan

- Python: 3.10.x 64-bit, paling aman Python 3.10.11 atau 3.10.13
- GPU: NVIDIA RTX 4060
- PyTorch: 2.3.1 + CUDA 12.1 wheel
- Driver NVIDIA: gunakan driver terbaru dari NVIDIA App / NVIDIA Driver. Untuk CUDA 12.1, driver lama sering membuat `torch.cuda.is_available()` menjadi `False`.

## Struktur folder minimal

Pastikan struktur project seperti ini:

```text
project/
├─ app.py
├─ skeleton_extractor.py
├─ requirements_gpu_rtx4060.txt
└─ models/
   ├─ yolo11l-pose.pt
   └─ rfdetr_nano_skeleton.pth
```

Nama file model harus sesuai dengan yang dipanggil di `app.py`.

## Instalasi di Windows CMD / PowerShell

Masuk ke folder project:

```bat
cd path\ke\project
```

Buat virtual environment:

```bat
py -3.10 -m venv .venv
```

Aktifkan venv:

```bat
.venv\Scripts\activate
```

Upgrade pip:

```bat
python -m pip install --upgrade pip setuptools wheel
```

Bersihkan instalasi lama yang berpotensi bentrok:

```bat
pip uninstall -y torch torchvision torchaudio opencv-python opencv-python-headless
pip cache purge
```

Install semua dependency:

```bat
pip install -r requirements_gpu_rtx4060.txt
```

## Instalasi alternatif yang lebih ketat

Jika `pip install -r requirements_gpu_rtx4060.txt` tetap memasang Torch CPU, pakai urutan ini:

```bat
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy<2.0 Pillow opencv-python streamlit ultralytics rfdetr supervision pytorch-lightning transformers peft "pydantic<3" tqdm requests pydeprecate
```

## Cek GPU

Jalankan:

```bat
python check_cuda.py
```

Hasil yang benar kira-kira seperti ini:

```text
torch version: 2.3.1+cu121
torch cuda version: 12.1
cuda available: True
gpu name: NVIDIA GeForce RTX 4060
```

Jika `cuda available: False`, biasanya penyebabnya:
1. Driver NVIDIA belum terpasang/terlalu lama.
2. Torch yang terpasang adalah versi CPU, bukan `+cu121`.
3. Venv belum aktif.
4. Ada konflik dari instalasi lama.

## Menjalankan aplikasi

```bat
streamlit run app.py
```

Jika nama file Anda masih `app(1).py`, jalankan:

```bat
streamlit run "app(1).py"
```

## Catatan penting

- Jangan campur `opencv-python` dan `opencv-python-headless`. Untuk kamera real-time di Windows, pakai `opencv-python`.
- Jangan install `torch` dari perintah `pip install torch` biasa, karena itu bisa mengambil versi CPU atau versi CUDA yang tidak cocok.
- Pastikan folder `models/` berisi `yolo11l-pose.pt` dan `rfdetr_nano_skeleton.pth`.
