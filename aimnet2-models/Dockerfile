# ── Stage 1: install + pre-download model ────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (prevents CUDA wheel being pulled as a dep)
# RUN pip install --no-cache-dir \
#     torch --index-url https://download.pytorch.org/whl/cpu

# CUDA-enabled torch
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu126

# Also add the GPU-accelerated neighbor list ops
RUN pip install --no-cache-dir 'nvalchemi-toolkit-ops[torch]'

# Install aimnet with HuggingFace support
RUN pip install --no-cache-dir "aimnet[hf,ase]"

# Pre-download the NSE model into HF cache
# Set HF_HOME so we know exactly where it lands
ENV HF_HOME=/hf_cache
RUN python -c "from aimnet.calculators import AIMNet2Calculator; calc = AIMNet2Calculator('isayevlab/aimnet2-nse'); print('Model cached successfully')"

# ── Stage 2: slim runtime ─────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ARG USER=appuser
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${USER} \
    && useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USER}

# Add gcc so Triton can JIT-compile CUDA kernels
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=${UID}:${GID} /hf_cache /hf_cache

ENV HF_HOME=/hf_cache \
    HOME=/home/${USER} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/${USER}
USER ${USER}

# Quick sanity check
RUN python -c "from aimnet.calculators import AIMNet2Calculator; print('OK')"

CMD ["python"]

# docker build -f Dockerfile -t aimnet2:nse .