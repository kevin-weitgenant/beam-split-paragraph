from beam import endpoint, Image

image = Image(
    base_image="nvcr.io/nvidia/pytorch:23.12-py3",  # Using a slightly older but stable version
    python_packages="requirements.txt",
)


@endpoint(image=image, gpu="T4", memory="16Gi", cpu=2, name="segment-paragraphs")
def segment_text(text: str):
    from wtpsplit import SaT
    import torch

    # Print version info for debugging
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
