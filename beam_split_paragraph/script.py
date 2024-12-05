from beam import endpoint, Image
import torch
from wtpsplit import SaT

image = Image(
    base_image="docker.io/nvidia/cuda:12.4.0-runtime-ubuntu20.04",
    python_version="python3.10",
).add_python_packages(
    [
        "wtpsplit",
        "onnxruntime-gpu==1.16.3",
        "torch",
    ]
)


@endpoint(image=image, gpu=["T4", "A10G", "A100-40"])
def segment_text(text: str):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

    model = SaT(
        "sat-3l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    segmented = model.split(text, do_paragraph_segmentation=True, verbose=True)
    return {"segmented_text": segmented}
