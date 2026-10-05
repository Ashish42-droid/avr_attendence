import torch
import cv2

print("=" * 50)
print("GPU & ENVIRONMENT VERIFICATION")
print("=" * 50)
cuda_ok = torch.cuda.is_available()
print(f"CUDA Available: {cuda_ok}")
if cuda_ok:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    # Quick tensor test on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print("GPU Tensor multiplication test: SUCCESS!")
else:
    print("WARNING: CUDA is not available to PyTorch!")

print(f"OpenCV Version: {cv2.__version__}")
print("=" * 50)
