import torch

print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3).cuda()
    print("cuda tensor test:", x.device, x.shape)
else:
    print("CUDA tidak terbaca oleh PyTorch.")
    print("Cek driver NVIDIA, versi torch, dan pastikan venv yang benar sedang aktif.")
