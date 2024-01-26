import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import mode
from PIL import Image
import matplotlib.pyplot as plt

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_closest_to_centroid(g):
    if len(g) == 1:
        return g[0]
    return g[np.argmin(np.linalg.norm(g[:,:3] - g[:,:3].mean(axis=0), axis=1))]


def get_closest_to_centroid_with_attributes_of_closest_to_cam(g):
    if len(g) == 1:
        return g
    xyz = g[np.argmin(np.linalg.norm(g[:,:3] - g[:,:3].mean(axis=0), axis=1)),:3]
    attributes = g[np.argmin(g[:,3], axis=0)][3:]
    return np.concatenate([xyz, attributes]).reshape(1, -1)

def remove_outliers(g):
    if len(g) == 1:
        return []
    return g

def map_3d(x, fn, size=0.03):
    # Makes Floats into Int-Bins
    to_bin = np.floor(x[:, :3] / size).astype(np.int32)
    # Lexsort Bins
    inds = np.lexsort(np.transpose(to_bin)[::-1])
    to_bin = to_bin[inds]
    x = x[inds]
    del inds
    splits = np.split(x, np.cumsum(np.unique(to_bin, return_counts=True, axis=0)[1])[:-1])
    del to_bin
    del x
    results = np.concatenate([x for x in np.vectorize(fn, otypes=[np.ndarray])(splits) if len(x)>0], axis=0)
    return results


def get_matching_indices(arr_a, arr_b):
    tree = cKDTree(arr_b)
    dist, index = tree.query(arr_a, workers=64)
    return index

def get_rotation_matrix_to_align_pose_with_gravity(pose, g):
    """Used to find rotation that rotates pose matrix to align with gravity vector g"""
    xx = np.array([0,0,1]) # Vector to which gravity is aligned
    return rotation_matrix_from_vectors(pose[:3,:3] @ (g / np.linalg.norm(g)), xx)


def get_edgeness(x):
    edgeness_x = torch.abs(x[:-1] - x[1:]) # has shape (height, width-1)
    edgeness_y = torch.abs(x[:,:-1] - x[:,1:]) # has shape (height-1, width)
    edgeness = torch.zeros_like(x)
    edgeness[:,:-1] += edgeness_y
    edgeness[:,1:] += edgeness_y
    edgeness[:-1,:] += edgeness_x
    edgeness[1:,:] += edgeness_x
    return edgeness


def aggregate_2d_grid(inp, size):
    """ Builds a 2D Grid along the two principal components of the point cloud 
    in each grid element, the points in the point cloud are aggregated to give a final 
    semantic class, height, and color"""
    to_bin = np.floor(inp[:, 0:2] / size).astype(np.int32)
    inds = np.lexsort(np.transpose(to_bin)[::-1])
    to_bin = to_bin[inds]
    inp = inp[inds]
    
    def aggregate_2d_grid_cell(group):
        # If only one point in grid element, return this one point, set the counter of points in grid element to 1
        if len(group) == 1:
            return np.concatenate([group, np.array([[1]])], axis=1)
        # If two points in grid element, return the one with higher z value, set counter of points in grid element to 2
        if len(group) == 2:
            return np.concatenate([group[np.argmax(group[:,2])], np.array([2])]).reshape(1, -1)
        # If more than two points in grid element, discard points below the mean height
        z = group[:,2]
        mean_height = z.mean()

        #TODO doublecheck orientation of z axis
        group_ = group[z >= mean_height]
        if len(group_) == 0:
            return np.concatenate([group[:1], np.array([[1]])], axis=1)
        x, y, z, r, g, b, distance_to_cam, class_, class_r, class_g, class_b, frame_index, depth_unc = group_.T
        
        most_common_class = mode(class_, keepdims=False)[0]

        class_r = class_r[class_==most_common_class][0]
        class_g = class_g[class_==most_common_class][0]
        class_b = class_b[class_==most_common_class][0]

        return np.array([[
            x[0], # Is the same for all points in group
            y[0], # Is the same for all points in group
            np.mean(z), # Height is calculated as mean
            np.mean(r), # Color is calculated as mean
            np.mean(g), # Color is calculated as mean
            np.mean(b), # Color is calculated as mean
            np.mean(distance_to_cam), # Distance to camera is calculated as mean
            most_common_class, # Class is the most common class
            class_r, # Class color is the color of most common class 
            class_g, # Class color is the color of most common class
            class_b, # Class color is the color of most common class
            np.mean(frame_index), # Frame index is calculated as mean
            np.mean(depth_unc), # Depth uncertainty is calculated as mean
            len(group) # Number of points in 2D Grid element
        ]])

    
    del inds
    splits = np.split(inp, np.cumsum(np.unique(to_bin, return_counts=True, axis=0)[1])[:-1])
    del to_bin
    del inp
    results = np.concatenate([inp for inp in np.vectorize(aggregate_2d_grid_cell, otypes=[np.ndarray])(splits) if len(inp)>0], axis=0)
    results[:,0] /= size
    results[:,1] /= size
    return results

def get_legend(class_to_colors, tmp_dir):
    
    class_to_colors = dict(sorted(class_to_colors.items(), key=lambda item: len(item[0])))

    labels = list(class_to_colors.keys())
    fig,ax = plt.subplots()

    def f(m, c,l):
        return plt.plot([],[],marker=m, color=c, ls="none", label=l)[0]

    [f("s", class_color/255., class_name) for class_name, class_color in class_to_colors.items()]

    ax.axis('off')
    legend = plt.legend(ncol=5)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(tmp_dir + "/legend.png", dpi="figure", bbox_inches=bbox)
    return np.array(Image.open(tmp_dir + "/legend.png"))[:,:,:3].transpose(1, 0, 2)/255.