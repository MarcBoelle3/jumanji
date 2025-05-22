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
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

############################################################################################
# Stage: 'lint_image'
# For running pre-commit in CI
FROM python_base AS lint_image
RUN apt update && apt install -y --no-install-recommends git \
    && apt clean autoclean \
    && apt autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
WORKDIR /
COPY pyproject.toml ./
RUN uv sync --no-cache --only-group lint
COPY .pre-commit-config.yaml ./
RUN git init && pre-commit install-hooks -c .pre-commit-config.yaml

############################################################################################
# Stage: 'base_image'
# Installs core Python dependencies globally
FROM python_base AS base_image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake && \
    rm -rf /var/lib/{apt,dpkg,cache,log}
WORKDIR /app
COPY pyproject.toml ./
COPY jumanji ./jumanji
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

#############################################################################################
# Stage: 'base_image_with_extras'
# Installs all optional groups globally
FROM base_image AS base_image_with_extras
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --all-groups

#############################################################################################
# Stage: 'small_python_image'
# Runtime image with minimal dependencies and Python globally
FROM python:${PYTHON_VERSION}-slim AS small_python_image
COPY --from=uv_base /uv /bin/
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Europe/Paris \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

################################################################################################
# Stage: 'main_image'
# Runtime image with project installed globally
FROM small_python_image AS main_image
WORKDIR /app
COPY pyproject.toml ./
COPY jumanji ./jumanji
RUN uv sync --no-install-project
COPY . .

#################################################################################################
# Stage: 'test_image'
# Runtime test image with all dependencies and test extras
FROM small_python_image AS test_image
WORKDIR /app
COPY --from=base_image_with_extras /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY pyproject.toml ./
COPY jumanji ./jumanji
RUN uv sync --no-install-project --all-groups
COPY . .


#################################################################################################
# Stage: 'gpu_image'
# GPU-enabled runtime image
FROM base_image_with_extras AS gpu_image
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda/nvvm /usr/local/cuda/nvvm

#added for profiling
# FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update -y && \
#     apt-get install -y --no-install-recommends \
#         apt-transport-https \
#         ca-certificates \
#         dbus \
#         fontconfig \
#         gnupg \
#         libasound2 \
#         libfreetype6 \
#         libglib2.0-0 \
#         libnss3 \
#         libsqlite3-0 \
#         libx11-xcb1 \
#         libxcb-glx0 \
#         libxcb-xkb1 \
#         libxcomposite1 \
#         libxcursor1 \
#         libxdamage1 \
#         libxi6 \
#         libxml2 \
#         libxrandr2 \
#         libxrender1 \
#         libxtst6 \
#         libgl1-mesa-glx \
#         libxkbfile-dev \
#         openssh-client \
#         wget \
#         xcb \
#         xkb-data && \
#     apt-get clean

# RUN cd /tmp && \
#     wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
#     apt-get install -y ./nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
#     rm -rf /tmp/*
#FROM ubuntu:22.04

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb

RUN apt-get update && apt install -y wget libglib2.0-0
RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

WORKDIR /app
COPY . .
RUN uv sync
