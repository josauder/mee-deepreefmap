import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torch import nn
import torch


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225
    return array


def change_bn_momentum(model, new_value):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = new_value


def get_depths_and_poses(encoder, segmentation_head, decoder, pose_decoder, images, features_, reduction, squeeze_unsqueeze):
    b, l, c, h, w = images.shape

    ref_features = [x[:b] for x in features_]

    features = []
    lf = len(features_)
    _, c_feat, h_feat, w_feat = features_[-1].shape

    for i in range(lf):
        if i == lf - 1:
            features.append(
                features_[-1][b:] + squeeze_unsqueeze(torch.cat([
                            features_[-1][b:],
                            ref_features[-1].reshape(b,1,c_feat,h_feat,w_feat).expand(b, l, c_feat, h_feat, w_feat).reshape(b*l,c_feat, h_feat, w_feat)
                ], dim=1)))
        else:
            features.append(features_[i][b:])


    depths = segmentation_head(decoder(*features)).reshape(b, l, 1, h, w)

    _, c_feat, h_feat, w_feat = features[-1].shape
    last_feat = features[-1]

    last_feat = reduction(last_feat.reshape(b* l, c_feat, h_feat, w_feat))
    c_feat = last_feat.shape[1]
    last_feat = last_feat.reshape(b, l, c_feat, h_feat, w_feat)


    last_feat = last_feat.unsqueeze(2).expand(b, l, l, c_feat, h_feat, w_feat).contiguous()
    features_sq = torch.cat([last_feat, last_feat.transpose(1, 2)], dim=3).reshape(b*l*l, c_feat*2, h_feat, w_feat)


    poses = pose_decoder(features_sq)
    poses = poses.mean(dim=(2,3)).reshape(b,l,l,6)

    return depths, poses * 0.01
