FROM python:3.10-slim

# Set commit hash for gpmfstream repo as there are no versions
ARG GPMFSTREAM_GIT_HASH=f1a9742
RUN pip install poetry pybind11

WORKDIR /app

# # Get the gpmfstream library
RUN apt-get update && \
    apt-get install -y git wget unzip build-essential \
    curl ffmpeg libsm6 libxext6
RUN git clone https://github.com/hovren/gpmfstream.git
WORKDIR /app/gpmfstream
RUN git checkout ${GPMFSTREAM_GIT_HASH}
RUN git submodule update --init
RUN ls -la
RUN python setup.py install
# RUN poetry shell && poetry run setup.py install

# Copy the pyproject.toml and poetry.lock files to the container
COPY poetry.lock pyproject.toml /app/

WORKDIR /app
# Install the dependencies
RUN poetry install --no-root --no-dev
RUN apt-get install -y

# Download the example data and checkpoints
RUN curl -f -L -o example_data.zip https://zenodo.org/records/10624794/files/example_data.zip?download=1 && unzip example_data.zip example_data/checkpoints/* && rm -rf example_data.zip
#RUN unzip example_data.zip

# Delete everything except checkpoints
#RUN rm -rf example_data.zip example_data/input_videos

COPY src /app/src
COPY example_inputs /app/example_inputs

WORKDIR /app/src

ENTRYPOINT ["poetry", "run", "python3", "reconstruct.py"]
