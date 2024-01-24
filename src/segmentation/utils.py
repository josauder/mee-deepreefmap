import numpy as np
import os
import json
import torch.nn as nn
from collections import defaultdict

def load_files(base_dir, test_splits, ignore_splits):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    test_polygons = []
    counts = defaultdict(int)

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file = root + "/" + file
            fileending = "." + file.split('.')[-1]
            if fileending in ['.jpg', '.png', '.jpeg']:
                split = file.split("/")[-2]
                if split not in ignore_splits and split not in test_splits:
                    train_images.append(file)
                    train_labels.append(file.replace(fileending, '_seg.npy'))
                elif split not in ignore_splits and split in test_splits:
                    test_images.append(file)
                    test_labels.append(file.replace(fileending, '_seg.npy'))
                    test_polygons.append(file.replace(fileending, '_poly.npy'))
    all_counts = json.loads(open(base_dir+"/counts.json").read())
    for split in all_counts.keys():
        if split not in ignore_splits and split not in test_splits:
            for class_name, count in all_counts[split].items():
                counts[int(class_name)] += int(count)
    return train_images, test_images, train_labels, test_labels, test_polygons, counts

    
def color_rgb_image(x, classes, colors):
    classes_inverse = {v: k for k, v in classes.items()}
    """Takes a semantic segmentation image and returns a colored RGB image."""
    semseg = np.ones((x.shape[0], x.shape[1], 3))
    for val in np.unique(x):
        if val > 0:
            name = classes_inverse[val]
            semseg[x==val] = np.array(colors[name])/255.
    return semseg

def color_by_correctness(prediction, label):
    semseg = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    semseg[prediction==label] = np.array([0, 0, 1])
    semseg[prediction!=label] = np.array([1, 0, 0])
    semseg[label==0] = 1
    return semseg

def rotatedRectWithMaxArea(h, w, angle):
    """
    From https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    angle = np.deg2rad(angle)
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return int(hr),int(wr)


def change_bn_momentum(model, new_value):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = new_value

