FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/Yoder23/abi"
LABEL org.opencontainers.image.description="ABI capability acquisition research toolkit"

WORKDIR /opt/abi
COPY . .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -e . --no-deps

CMD ["python", "-m", "abi", "self-check"]
