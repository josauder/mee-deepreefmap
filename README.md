# Scalable 3D Semantic Mapping of Coral Reefs using Deep Learning

This repository contains the source code for the paper [Scalable 3D Semantic Mapping of Coral Reefs using Deep Learning](https://arxiv.org/abs/2309.12804).

The [project page](https://josauder.github.io/deepreefmap/) contains more information about DeepReefMap.

## Installation

First install [Pixi](https://pixi.sh/). 

Then clone this repository:
```bash
git clone https://github.com/Tobias-Fischer/mee-deepreefmap.git
cd mee-deepreefmap
pixi install
```

The Pixi environment installs the Python dependencies, the editable local package, `gpmfstream`, FFmpeg, Git-LFS, and the compiler tooling needed for the native extension.

To fetch the Git-LFS model checkpoints used by reconstruction, run:

```bash
pixi run _install
```

## Example Data

The example videos are hosted on [Zenodo](https://zenodo.org/records/10624794). They are large, so the download is a separate cached Pixi task:

```bash
pixi run download-example-data
```

This extracts the archive to `example_data/`, including:

```text
example_data/
    input_videos/
        GX_SINGLE_VIDEO.MP4
        GX_VIDEO_1_OF_2.MP4
        GX_VIDEO_2_OF_2.MP4
```

## Running Reconstructions

Run a single GoPro Hero 10 video:

```bash
pixi run reconstruct \
  example_data/input_videos/GX_SINGLE_VIDEO.MP4 \
  0-367 \
  out_single_video
```

Run a transect split across two GoPro Hero 10 videos:

```bash
pixi run reconstruct \
  example_data/input_videos/GX_VIDEO_1_OF_2.MP4,example_data/input_videos/GX_VIDEO_2_OF_2.MP4 \
  310-end,begin-100 \
  out_two_videos
```

The `reconstruct` task depends on `_install`, so model checkpoints are pulled before the pipeline runs. Pixi also tracks the task inputs and outputs, so unchanged runs can be skipped.

## Other Cameras

DeepReefMap currently targets GoPro Hero 10 videos. Other cameras can be used if you provide camera intrinsics in the same simplified EUCM JSON format as `example_inputs/intrinsics_eucm.json`.

The Pixi task exposes the common GoPro workflow. For custom reconstruction options such as `--intrinsics_file`, run the script inside the Pixi environment:

```bash
pixi run python src/reconstruct.py \
  --input_video=example_data/input_videos/OTHER_CAMERA_VID.MP4 \
  --timestamp=10-120 \
  --out_dir=out_other_camera \
  --intrinsics_file=example_inputs/intrinsics_eucm.json
```

## Training

Train the SfM network:

```bash
pixi run train-sfm <PATH_TO_DATA> <WANDB_EXPERIMENT_NAME>
```

The SfM dataset should follow the KITTI VO-style layout:

```text
train.txt
val.txt
sequence1/
    000001.jpg
    000002.jpg
sequence2/
    000001.jpg
    000002.jpg
```

Train the semantic segmentation network:

```bash
pixi run train-segmentation <PATH_TO_DATA> <WANDB_EXPERIMENT_NAME>
```

The segmentation dataset should contain class metadata and scene folders:

```text
data/
    classes.json
    colors.json
    counts.json
    scene_1/
        image_0.png
        image_0_seg.npy
        image_0_poly.npy
```

## Pixi Tasks

```bash
pixi task list
```

Available tasks:

```text
_install               Fetch Git-LFS model checkpoints.
download-example-data  Download and extract the Zenodo example videos.
reconstruct            Run the reconstruction pipeline.
train-sfm              Train the SfM model.
train-segmentation     Train the segmentation model.
```
