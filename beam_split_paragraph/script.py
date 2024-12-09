from beam import endpoint, Image, Volume
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_PATH = "./cached_weights"

# Set the correct environment variables according to documentation
os.environ["HF_HOME"] = os.path.abspath(CACHE_PATH)
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.abspath(CACHE_PATH), "hub")
os.environ["HF_ASSETS_CACHE"] = os.path.join(os.path.abspath(CACHE_PATH), "assets")

image = Image(
    base_image="docker.io/nvidia/cuda:12.4.0-runtime-ubuntu20.04",
    python_version="python3.10",
).add_python_packages(["wtpsplit", "torch", "transformers"])


def download_models():
    # Log the environment variables to verify they're set correctly
    logger.info(f"HF_HOME is set to: {os.environ.get('HF_HOME')}")
    logger.info(f"HF_HUB_CACHE is set to: {os.environ.get('HF_HUB_CACHE')}")
    logger.info(f"HF_ASSETS_CACHE is set to: {os.environ.get('HF_ASSETS_CACHE')}")

    # Create cache directories if they don't exist
    for path in [
        os.environ["HF_HOME"],
        os.environ["HF_HUB_CACHE"],
        os.environ["HF_ASSETS_CACHE"],
    ]:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")

    # Create test file
    test_file_path = os.path.join(CACHE_PATH, "test.txt")
    if not os.path.exists(test_file_path):
        with open(test_file_path, "w") as f:
            f.write("Test file to verify volume persistence")
        logger.info(f"Created test file at {test_file_path}")
    else:
        logger.info(f"Test file already exists at {test_file_path}")

    # Now proceed with model download
    from wtpsplit import SaT

    logger.info("=== Cache Contents Before Model Download ===")
    cache_path = Path(CACHE_PATH)
    for path in cache_path.iterdir():
        if path.is_dir():
            logger.info(path)

    model = SaT("sat-3l-sm")
    model.half().to("cuda")

   

    return model


@endpoint(
    image=image,
    gpu=["T4", "A10G", "A100-40"],
    volumes=[Volume(name="huggingface_cache", mount_path=CACHE_PATH)],
    on_start=download_models,
    keep_warm_seconds=5,
)
def segment_text(text: str, context):
    import time

    overall_start = time.time()
    model = context.on_start_value
    segmented = model.split(text, do_paragraph_segmentation=True, verbose=True)
    overall_end = time.time()
    logger.info(f"\nTotal execution time: {overall_end - overall_start:.2f} seconds")
    return {"segmented_text": segmented}
