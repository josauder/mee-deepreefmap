FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set commit hash for gpmfstream repo as there are no versions
ARG GPMFSTREAM_GIT_HASH=f1a9742
ARG DEBIAN_FRONTEND=noninteractive

# # Get the gpmfstream library
RUN apt-get update && \
    apt-get install -y git ffmpeg pipx python3-pip \
    && pip3 install uv

# RUN pip3 install poetry
WORKDIR /app

# Copy the model checkpoints and environment
COPY segmentation_net.pth /app/
COPY sfm_net.pth /app/
COPY uv.lock pyproject.toml /app/

# Change to non-root user
RUN groupadd mygroup --gid 1000 && \
    useradd -m -U -s /bin/bash -G mygroup -u 1000 myuser && \
    chown -R 1000:1000 /app && \
    chmod -R 755 /app && \
    mkdir /output /input && \
    chown -R 1000:1000 /output /input /tmp && \
    chmod -R o+w /input /output


# Build gpmfstream
RUN git clone https://github.com/hovren/gpmfstream.git
WORKDIR /app/gpmfstream
RUN pip3 install pybind11 setuptools
RUN git checkout ${GPMFSTREAM_GIT_HASH} \
    && git submodule update --init
RUN python3 setup.py bdist_wheel

# Install dependencies
WORKDIR /app
RUN uv sync --no-dev --no-cache
RUN uv pip install "./gpmfstream/dist/gpmfstream-0.5-cp310-cp310-linux_x86_64.whl"

COPY src /app/src
COPY example_inputs /app/example_inputs

WORKDIR /app/src

RUN chown -R 1000:1000 /app

USER 1000

ENTRYPOINT ["uv", "run", "python3", "reconstruct.py"]
