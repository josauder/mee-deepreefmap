# Scalable 3D Semantic Mapping of Coral Reefs using Deep Learning

This repository contais the source code for the Paper [Scalable 3D Semantic Mapping of Coral Reefs using Deep Learning](https://arxiv.org/abs/2309.12804).

[The project page](https://josauder.github.io/deepreefmap/) contains updated information on the state of the project.


## Installation

Installing the required dependencies via pip:

```pip install -r requirements.txt```

## Running 3D Reconstructions of GoPro Hero 10 Videos

Simple usage: the input is one MP4 video taken with a GoPro Hero 10 camera, as well as the timestamps on when the transect begins and ends in the video (TODO: discuss format).

```python3 reconstruct.py \
    --input_video=<PATH_TO_VIDEO.MP4> \
    --timestamp=<START_TIMESTAMP>-<END_TIMESTAMP> \
    --out_dir=<OUTPUT_DIRECTORY>
```

Advanced usage: the GoPro Hero 10 camera cuts videos into 4GB chuns. If the transect is spread over two or more videos, the following command can be used to reconstruct the transect.

```python3 reconstruct.py \
    --input_video=<PATH_TO_VIDEO_1.MP4>,<PATH_TO_VIDEO_2.MP4> \
    --timestamp=<START_TIMESTAMP_1>-end,begin-<END_TIMESTAMP_2> \
    --out_dir=<OUTPUT_DIRECTORY>
```

## Running 3D Reconstructions of Videos from other Cameras

This repository, for now, supports the GoPro Hero 10 Camera. If you want to use a different camera, be sure to provide the correct camera intrinsics as `intrinsics.json`, which are passed as a command line argument to any other scripts. For now, the intrinsics follow a simplified UCM format, with the focal lengths `fx, fy` in pixels, the focal point `cx, cy` in pixels, and the `alpha` value to account for distortion, which is set to zero in the default case to assume linear camera intrinsics.

## Download Example Data and Pre-Trained Models:

Pre-trained model checkpints and example input videos can be downloaded from the [Zenodo archive](https://zenodo.org/record/TODO).

## Training the 3D Reconstruction Network on Your Own Data

To train the 3D reconstruction data on your own data, use 

```sfm/train_sfm.py 
    --data <PATH_TO_DATA> \
    --checkpoint <PATH_TO_PRETRAINED_CHECKPOINT> \
    --name <NAME_OF_WANDB_EXPERIMENT>
````

Where your data should be a directory of the following structure (same as KITTI VO Dataset):

```train.txt
val.txt
sequence1/
    000001.jpg
    000002.jpg
    ...
sequence1/
    000001.jpg
    000002.jpg
    ...    
sequence3/
    000001.jpg
    000002.jpg
    ...    
```

With `train.txt` containing, for example

```sequence1
sequence2
```
And `val.txt` containing, for example

```sequence3
```

## Training the Semantic Segmentation Network on Your Own Data

For training the segmentation model, use

```python3 train_segmentation.py \
    --data <PATH_TO_DATA> \
    --checkpoint <PATH_TO_PRETRAINED_CHECKPOINT> \
    --test_splits <TEST_SCENE1>,<TEST_SCENE2> \
    --name <NAME_OF_WANDB_EXPERIMENT>
```

Where your data should be a directory with the following structure:
```data/
    classes.json
    colors.json
    counts.json
    scene_1/
        image_0.png
        image_0_seg.npy
        image_0_poly.npy
        image_1.png
        image_1_seg.npy
        image_1_poly.npy
        ...
    scene_2/
        image_0.png
        image_0_seg.npy
        image_0_poly.npy
        ...
    ...
```

Following the example dataset from the [Zenodo archive](https://zenodo.org/record/TODO).