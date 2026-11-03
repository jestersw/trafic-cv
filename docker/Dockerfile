ARG BASE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
FROM ${BASE}

ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON=3.11 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/app/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra train --extra track --extra export --extra data \
    && if [ "$TARGETARCH" = "amd64" ]; then \
         uv pip install onnxruntime-gpu tensorrt-cu12; \
       fi

COPY run.py ./
COPY configs ./configs

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
