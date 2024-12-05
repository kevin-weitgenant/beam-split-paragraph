FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    nvidia-cuda-toolkit

# Set working directory
WORKDIR /app

# Environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python packages
RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch \
    wtpsplit \
    onnxruntime-gpu==1.19.2

# Copy the script
COPY script.py .

CMD ["python3", "script.py", "--text", "Test text."]



