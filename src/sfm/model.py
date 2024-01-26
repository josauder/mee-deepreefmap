import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SfMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.depth_net = smp.DeepLabV3Plus(encoder_name="resnext50_32x4d", encoder_weights='swsl', activation=None)
        self.pose_reduction = nn.Sequential(
            nn.Conv2d(2048, 1024, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(1024),
        )
        self.pose_decoder = nn.Sequential(
            nn.Conv2d(2048, 256, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.BatchNorm2d(256), 
            nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  nn.BatchNorm2d(256),
            nn.Conv2d(256, 6, (3, 3)),
        )  

    def forward(self, images, intrinsics, neighbor_range=1):

        depth_features = []


        for image_i in images:
            depth_features.append(self.depth_net.encoder(image_i))
        
        return self.get_depth_and_poses_from_features(images, depth_features, intrinsics, neighbor_range=neighbor_range)
    
    def extract_features(self, image):
        return self.depth_net.encoder(image)
    
    def get_depth_and_poses_from_features(self, images, features, intrinsics, neighbor_range=1):
        depths = []
        poses = [[[] for __ in range(len(images))] for _ in range(len(images))]

        for i, image_i in enumerate(images):
            depth = self.depth_net.segmentation_head(self.depth_net.decoder(*features[i]))
            depths.append(1 / (10 * torch.sigmoid(depth) + 0.01))

            for j, image_j in enumerate(images):
                if i != j and abs(i - j) <= neighbor_range:
                    pose_out = self.pose_decoder(
                        torch.cat([
                            self.pose_reduction(features[i][-1]), 
                            self.pose_reduction(features[j][-1])
                        ], dim=1))
                    poses[i][j] = pose_out.mean(dim=2).mean(dim=2) * 0.005
        return depths, poses, [intrinsics for _ in range(len(images))]