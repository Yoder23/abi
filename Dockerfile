FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Sam Yoder <https://github.com/Yoder23/abi>"
LABEL description="ABI: Frozen-Module Domain Transfer Across LLM Architectures"

WORKDIR /workspace/abi

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# Default: run the standalone verifier (no GPU required, no model download)
CMD ["python", "verify_result.py"]
