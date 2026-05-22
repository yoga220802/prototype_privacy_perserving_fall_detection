from ultralytics import YOLO

# Baris ini akan otomatis mencari file di lokal.
# Jika tidak ada, ia akan men-download-nya dari server Ultralytics.
model = YOLO("yolo11l-pose.pt")

print("Model yolo11l-pose.pt berhasil diunduh dan siap digunakan!")
