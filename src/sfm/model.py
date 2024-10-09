import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .utils import get_depths_and_poses


class SfMModel(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()

        self.depth_net = smp.DeepLabV3Plus(in_channels=in_channels, encoder_name="resnext50_32x4d", encoder_weights='swsl', activation=None)
        self.pose_reduction = nn.Sequential(
            nn.Conv2d(2048, 512, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(512),
        )
        self.squeeze_unsqueeze = nn.Sequential(
            nn.Conv2d(4096, 512, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(512),
            nn.Conv2d(512, 2048, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(2048),
        )
        self.pose_decoder = nn.Sequential(
            nn.Conv2d(1024, 256, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.BatchNorm2d(256), 
            nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  nn.BatchNorm2d(256),
            nn.Conv2d(256, 6, (3, 3), bias=False),
        )  

    def extract_features(self, x):
        return self.depth_net.encoder(x)
        
    def get_depth_and_poses_from_features(self, images, features, intrinsics):
        depth, pose = get_depths_and_poses(
                self.depth_net.encoder, 
                self.depth_net.segmentation_head, 
                self.depth_net.decoder, 
                self.pose_decoder,
                torch.stack(images).transpose(1,0),
                [torch.stack([features[0][flen]] + [feature[flen] for feature in features]).squeeze() for flen in range(len(features[0]))],
                self.pose_reduction,
                self.squeeze_unsqueeze,
            )
        depth = (1 / (25 * torch.sigmoid(depth) + 0.1))
        return depth.squeeze(), pose.squeeze(), intrinsics.repeat(len(images),1)
