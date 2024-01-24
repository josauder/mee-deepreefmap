
from torch.utils.data import Dataset
import numpy as np
import random
from path import Path
from PIL import Image
import custom_transforms
import torchvision

class SequenceDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, individual_transform=None, seed=0, long_sequence_length=7, subsampled_sequence_length=5, with_replacement=True):
        """The data_path argument points to a directory structured in the KITTI format:
        which means that it contains a subdirectory for each sequence, and each sequence
        <data_path>/<path/to/sequence_1>/0000001.jpg
        <data_path>/<path/to/sequence_1>/0000002.jpg
        ..
        <data_path>/<path/to/sequence_2>/0000001.jpg
        <data_path>/<path/to/sequence_2>/0000002.jpg

        One sample from the dataset is chosen as follows:
        1. A random sequence is chosen
        2. A sequence of <long_sequence_length> subsequent frames is randomly chosen from the sequence
        3. A sequence of <subsampled_sequence_length> random frames is subsampled from the sequence, with or without replacement
        """
        np.random.seed(seed)
        random.seed(seed)

        self.root = Path(data_path)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.individual_transform = individual_transform
        self.long_sequence_length = long_sequence_length
        self.subsampled_sequence_length = subsampled_sequence_length

        self.backward = {}
        with open( self.root/"forward.txt", "r") as f:
            for line in f.readlines():
                self.backward[line.strip()] = False
        with open( self.root/"backward.txt", "r") as f:
            for line in f.readlines():
                self.backward[line.strip()] = True


        self.imgs = {(scene): sorted(list(scene.files('*.jpg'))+list(scene.files('*.jpeg'))) for scene in self.scenes}

        self.num_samples = 0
        self.index_to_sequence = []
        self.index_to_index_in_sequence = []
        for scene in self.scenes:
            self.num_samples += len(self.imgs[scene]) - long_sequence_length + 1
            self.index_to_sequence += [scene] * (len(self.imgs[scene]) - long_sequence_length + 1)
            self.index_to_index_in_sequence += list(range(len(self.imgs[scene]) - long_sequence_length + 1))
        self.index_to_sequence = np.array(self.index_to_sequence)
        self.with_replacement = with_replacement

        normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
        self.totensor = torchvision.transforms.ToTensor()
        self.out_transform = normalize


    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # First, get right scene via self.index_to_sequence
        scene = self.index_to_sequence[idx]
        # Then, get corresponding sequence of images within scene
        index = self.index_to_index_in_sequence[idx]
        backward = False
        long_sequence = self.imgs[scene][index:index+self.long_sequence_length]

        # Then, subsample from sequence, but keep the order of the sequence intact
        if self.with_replacement:
            indices = sorted(np.random.choice(len(long_sequence), self.subsampled_sequence_length, replace=True), reverse=backward)
        else:
            indices = sorted(np.random.choice(len(long_sequence), self.subsampled_sequence_length, replace=False), reverse=backward)
        subsampled_sequence = [long_sequence[i] for i in indices]

        # TODO Fix
        intrinsics = np.eye(3)

        imgs = [Image.open(img) for img in subsampled_sequence]
        if self.transform:
            # TODO remove intrinsics
            imgs, intrinsics = self.transform(imgs, intrinsics)
            imgs_out, intrinsics = self.out_transform([self.totensor(img) for img in imgs], intrinsics)
        else:
            imgs_out, intrinsics = self.out_transform([self.totensor(img) for img in imgs], intrinsics)

        if self.individual_transform:
            # TODO remove intrinsics
            imgs_individually_transformed = [self.individual_transform(img) for img in imgs]
            imgs_individually_transformed, intrinsics = self.out_transform([self.totensor(img) for img in imgs_individually_transformed], intrinsics)
        else:
            imgs_individually_transformed = imgs_out
        
        return imgs_out, imgs_individually_transformed, 0
    