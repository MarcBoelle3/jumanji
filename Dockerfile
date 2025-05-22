ARG PYTHON_VERSION=3.10.6
# replace with call from makefile ?
ARG UV_VERSION=0.6.2
###########################################################################################
# Stage: 'uv'
# It is used to define the uv Docker image
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_base
############################################################################################
# Stage: python_base
# Contains python and uv with correct env variables
FROM python:${PYTHON_VERSION} AS python_base
COPY --from=uv_base /uv /bin/
# Set the environment variables
#   - Don't use the cache to reduce the image size
#   - Use copy mode to keep all the packages in the .venv
#   - Byte-compile the Python files for faster application startup
#   - Assert that the uv.lock will remain unchanged
ENV UV_PYTHON=python${PYTHON_VERSION} \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    #UV_LOCKED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
############################################################################################
# Stage: 'lint_image'
# It is to run the pre-commit in the CI
FROM python_base AS lint_image
# Install git
RUN apt update && apt install -y --no-install-recommends git \
    && apt clean autoclean \
    && apt autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
WORKDIR /
#COPY uv.lock pyproject.toml ./
COPY pyproject.toml ./

# Install lint-specific dependencies only
# RUN uv sync --locked --no-cache --only-group lint
RUN uv sync --no-cache --only-group lint

ENV PATH="/.venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT="/.venv"
# Install pre-commit hooks
COPY .pre-commit-config.yaml ./
RUN git init && pre-commit install-hooks -c .pre-commit-config.yaml
############################################################################################
# Stage: 'base_image'
# It is used to define python version and install all the Python dependencies
FROM python_base AS base_image
# Needed librairies to compile linearfold
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake && \
    rm -rf /var/lib/{apt,dpkg,cache,log}
WORKDIR /app
# Install main dependencies (not the project to optimise Docker caching)
# This uses netrc to authenticate with the private packages
# Setting the HOME variable is needed for uv to find the `.netrc` file
#COPY uv.lock pyproject.toml ./
COPY pyproject.toml ./
COPY jumanji ./jumanji
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project
#############################################################################################
# Stage: 'base_image_with_extras'
# Same as base-image but with all groups, used for tests
FROM base_image AS base_image_with_extras
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --all-groups
#############################################################################################
# Stage: 'small_python_image'
# It is used to define the image for the runtime with:
#   - build the runtime image without the build dependencies, reducing the image size
#   - non root user
#   - set the timezone
#   - env variables
#   - linux packages
#   - set the working directory
#   - copy the project files
FROM python:${PYTHON_VERSION}-slim AS small_python_image
COPY --from=uv_base /uv /bin/
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Europe/Paris \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_LOCKED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT="/app/.venv"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}
################################################################################################
# Stage: 'main_image'
# It is used to define the main runtime image with the project installed, inheriting from the small_python_image
FROM small_python_image AS main_image
COPY --from=base_image /app/.venv /app/.venv
WORKDIR /app
COPY . .
RUN uv sync
#################################################################################################
# Stage: 'test_image'
# Runtime test image with all the extras
FROM small_python_image AS test_image
COPY --from=base_image_with_extras /app/.venv /app/.venv
#################################################################################################
# Stage: 'gpu_image'
# It is used to define the gpu runtime image with the project installed and the gpu depeendencies, inheriting from the small_python_image
#FROM small_python_image AS gpu_image
FROM base_image_with_extras AS gpu_image
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda/nvvm /usr/local/cuda/nvvm
COPY --from=base_image /app/.venv /app/.venv
WORKDIR /app
COPY . .
RUN uv sync
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT="/app/.venv"
