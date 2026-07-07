# syntax=docker/dockerfile:1
#
# Adapted from LLaMA-Factory's upstream docker/docker-rocm/Dockerfile:
#   https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/docker/docker-rocm/Dockerfile
#
# Upstream base : rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1
# This image    : rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.7.1
#   -> only the ROCm patch level is bumped 7.2 -> 7.2.3; OS (ubuntu24.04),
#      Python (3.12) and PyTorch (release 2.7.1) are unchanged. RDNA4 / gfx1201
#      (Radeon RX 9070 XT) requires ROCm >= 7.2.3. Tag verified to exist on
#      https://hub.docker.com/r/rocm/pytorch/tags .
#
# Keep the base image's PyTorch/ROCm build; do NOT reinstall torch and do NOT
# `pip install bitsandbytes` a CUDA wheel over ROCm torch — bitsandbytes comes
# ONLY from LLaMA-Factory's own requirements/bitsandbytes.txt (the ROCm path).
ARG BASE_IMAGE=rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.7.1
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""
# LLaMA-Factory source. Upstream does `COPY . /app` because its build context is
# the LLaMA-Factory repo; ours is tg2llm, so we clone instead. Override the ref
# to pin a tag/branch (build-arg only — nothing is installed on the host).
ARG LLAMAFACTORY_REPO=https://github.com/hiyouga/LLaMA-Factory.git
ARG LLAMAFACTORY_REF=main

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"

# Fetch LLaMA-Factory into /app (replaces upstream's `COPY . /app`).
RUN if ! command -v git >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; \
    fi && \
    git clone --depth 1 --branch "${LLAMAFACTORY_REF}" "${LLAMAFACTORY_REPO}" /app

# Install LLaMA Factory (use base image's PyTorch/ROCm; do not reinstall).
# bitsandbytes is pulled by requirements/bitsandbytes.txt — LLaMA-Factory's own
# dependency path, kept intact (never a hand-picked CUDA wheel).
RUN pip install --no-cache-dir -e . --pre && \
    pip install --no-cache-dir -r requirements/deepspeed.txt -r requirements/liger-kernel.txt -r requirements/bitsandbytes.txt

# Rebuild flash attention (OFF by default — RDNA4 uses AOTriton SDPA, not FA2).
RUN if [ "${INSTALL_FLASHATTN}" == "true" ]; then \
        pip uninstall -y ninja && \
        pip install --no-cache-dir ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# Expose port 7860 for LLaMA Board
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Expose port 8000 for API service
ENV API_PORT=8000
EXPOSE 8000

# unset proxy
ENV http_proxy=
ENV https_proxy=

# Set no_proxy environment variable
ENV no_proxy="localhost, 127.0.0.1, ::1"

# fix pydantic version
RUN pip install --no-cache-dir pydantic==2.10.6

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
