
import torch
from wtpsplit import SaT
import os
import onnxruntime as ort

def segment_text(text: str):
    # Print version info for debugging
    cuda_paths = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
    print("CUDA Library Check:")
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"\nFiles in {path}:")
            print([f for f in os.listdir(path) if "cuda" in f or "cudnn" in f])

    print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
    print(f"\nONNX Runtime version: {ort.__version__}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

    # Initialize model with CUDA support
    model = SaT(
        "sat-3l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Perform segmentation
    segmented = model.split(text, do_paragraph_segmentation=True, verbose=True)

    return {"segmented_text": segmented}



print(segment_text("BLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))