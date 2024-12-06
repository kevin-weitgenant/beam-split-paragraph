from beam import endpoint, Image

image = (
    Image(
        base_image="docker.io/nvidia/cuda:12.4.0-runtime-ubuntu20.04",
        python_version="python3.10",
    )
    .add_commands(
        ["apt-get update -y", "apt-get install -y cuda-cudart-12-4 libcudnn8"]
    )
    .add_python_packages(
        [
            "wtpsplit",
            "onnxruntime-gpu==1.19.2",  # Matching your local version
            "torch",
        ]
    )
)


@endpoint(image=image, gpu=["T4", "A10G", "A100-40"])
def segment_text(text: str):
    import onnxruntime as ort
    import torch
    from wtpsplit import SaT
    import os

    # Print CUDA library information
    cuda_paths = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
    print("CUDA Library Check:")
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"\nFiles in {path}:")
            print([f for f in os.listdir(path) if "cuda" in f or "cudnn" in f])

    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

    model = SaT(
        "sat-3l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    segmented = model.split(text, do_paragraph_segmentation=True, verbose=True)
    return {"segmented_text": segmented}
