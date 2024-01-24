import segmentation_models_pytorch as smp
import torch.nn as nn 
import torch
import numpy as np
import torchvision
from PIL import Image
import utils
import wandb
from time import time
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentationModel(nn.Module):

    def __init__(self, num_classes):
        self.model = smp.DeepLabV3Plus(encoder_name="resnext50_32x4d", classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, label_files, output_size, imagenet_normalization=False):
        self.output_size = output_size
        self.image_files = image_files
        self.label_files = label_files
        
        self.totensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
        self.unnormalize = torchvision.transforms.Normalize(
            mean=[-0.45/0.225, -0.45/0.225, -0.45/0.225],
            std=[1/0.225, 1/0.225, 1/0.255]
        )
        if imagenet_normalization:
            self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            self.unnormalize = torchvision.transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
        self.colorjitter = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.resize_img = torchvision.transforms.Resize((self.output_size[0], self.output_size[1]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.resize_label = torchvision.transforms.Resize((self.output_size[0], self.output_size[1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        
        self.randomresizedcrop = torchvision.transforms.RandomResizedCrop(size=(352*2, 608*2))
        self.scale = (0.2, 1)
        self.ratio = (3/4, 4/3)
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        label = np.load(self.label_files[index])
        
        
        # randomcrop on images and labels
        i, j, h, w = self.randomresizedcrop.get_params(image, self.scale, self.ratio)
        
        # convert image to tensor, normalize with imagenet normalization
        image = self.totensor(image)   
        label = self.totensor(label.astype(np.int32)).int()
        
        # Start with transforms that alter the image size, like crops and (non-90-degree) rotations
        # first random resized crops
        image = torchvision.transforms.functional.resized_crop(image, i, j, h, w, size=(352, 608*2), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        label = torchvision.transforms.functional.resized_crop(label, i, j, h, w, size=(352*2, 608*2), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        # Then rotations
        deg = np.random.uniform(-15, 15)
        image = F.rotate(image, deg, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=True)
        label = F.rotate(label, deg, interpolation=torchvision.transforms.InterpolationMode.NEAREST, expand=True)
        new_dimensions = utils.rotatedRectWithMaxArea(352*2, 608*2, deg)
        image = F.center_crop(image, new_dimensions)
        label = F.center_crop(label, new_dimensions)

        # Resize to output size
        image = self.resize_img(image)
        label = self.resize_label(label)

        # Now, augmentations that don't alter the image size, like colorjitter and flips
        image = self.colorjitter(image)
        
        if np.random.uniform() < 0.5:
            image = torchvision.transforms.functional.hflip(image)
            label = torchvision.transforms.functional.hflip(label)

        image = self.normalize(image)

        return image, label 
    

class BaselineExperiment:

    def __init__(self, classes, train_images, train_labels, counts, colors, output_size=(352, 608), batch_size=6):
        self.model = SegmentationModel(num_classes=len(classes)+1).to(device)
        utils.change_bn_momentum(self.model, 0.01)
        self.classes = classes 
        self.colors = colors 
        weights = np.zeros(len(classes)+1)
        for index, count in counts.items():
            weights[index] = 1 / np.sqrt(count + 2000000)
        weight = torch.tensor(weights).float()
        weight /= weight.mean()

        self.train_dataset = SegmentationDataset(train_images, train_labels, output_size, imagenet_normalization=False)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight.to(device), ignore_index=0,label_smoothing=0.05)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00003)

        self.epochs = 900

        # Attributes used for prediction
        self.totensor = torchvision.transforms.ToTensor()
        self.in_resize = torchvision.transforms.Resize(output_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.out_resize = torchvision.transforms.Resize((1080, 1920), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.model.encoder.requires_grad_(True)


    def train_epoch(self, epoch, logger):
        t = time()
        self.model.train()
        mean_train_acc = []
        mean_train_loss = []

        for (X, y) in self.train_dataloader:
            X = X.to(device)
            y = y.to(device).squeeze(1).long()
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            accuracy = (y_pred.argmax(dim=1) == y)[y!=0].float().mean()
            mean_train_acc.append(accuracy.item())
            mean_train_loss.append(loss.item())

        image = np.clip(self.train_dataset.unnormalize(X[0].cpu()).permute(1,2,0).numpy(), 0, 1)
        label = utils.color_rgb_image(y[0].cpu().numpy(), self.classes, self.colors)
        prediction = utils.color_rgb_image(y_pred[0].argmax(dim=0).cpu().numpy(), self.classes, self.colors)

        logger.log({
            "train/loss": np.mean(mean_train_loss), 
            "train/accuracy": np.mean(mean_train_acc),
            "train/image": wandb.Image(image, caption="Image"),
            "train/label": wandb.Image(label, caption="Label"),
            "train/prediction": wandb.Image(prediction, caption="Prediction"),
            "train/time_taken": time() - t
        }, step=epoch)

    
    def predict(self, image):
        self.model.eval()
        # Gets PIL image, returns numpy array
        image = self.train_dataset.normalize(self.totensor(self.in_resize(image)).unsqueeze(0).to(device))
        prediction = self.model(image).argmax(dim=1).cpu()
        prediction = self.out_resize(prediction).squeeze()
        return prediction.numpy()

