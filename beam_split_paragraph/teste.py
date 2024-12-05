
import torch
from wtpsplit import SaT

def segment_text(text: str):
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



print(segment_text("BLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))