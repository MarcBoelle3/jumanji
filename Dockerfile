ARG PYTHON_VERSION=3.10.6
ARG UV_VERSION=0.6.2

###########################################################################################
# Stage: 'uv'
# Define the uv Docker image
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_base

############################################################################################
# Stage: python_base
# Contains python and uv with correct env variables
FROM python:${PYTHON_VERSION} AS python_base
COPY --from=uv_base /uv /bin/
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_PYTHON=python${PYTHON_VERSION} \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda/nvvm /usr/local/cuda/nvvm

#added for profiling
ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb

RUN apt-get update && apt install -y wget libglib2.0-0
RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

WORKDIR /app
COPY pyproject.toml ./
COPY jumanji ./jumanji
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group reqs --group train --extra gpu --no-install-project

COPY . .
