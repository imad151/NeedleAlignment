import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from glob import glob
from tqdm import tqdm


DATASET_PATH = r"/home/imad/NeedleAlignment/data/needleonly"


# -------------------
#    Data Loader
# -------------------
class LoadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.ImagePaths = []
        self.MaskPaths = []
        self.transform = transform

        for subdir in sorted(os.listdir(root_dir), key=lambda x: int(x)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                images = sorted(glob(os.path.join(subdir_path, "*.png")))
                for img_path in images:
                    if "rgb" in img_path.lower():
                        self.ImagePaths.append(img_path)
                    else:
                        self.MaskPaths.append(img_path)
        print(f"Loaded {len(self.ImagePaths)} Images and {len(self.MaskPaths)} Masks")
        if not len(self.ImagePaths) == len(self.MaskPaths):
            raise FileNotFoundError("Number of RGB and Mask images dont match")
        

    def __len__(self):
        return len(self.ImagePaths)
    
    def __getitem__(self, index):
        image = cv2.imread(self.ImagePaths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        
        mask = cv2.imread(self.MaskPaths[index], cv2.IMREAD_UNCHANGED)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        mask = mask[:3, :, :]
        return image, mask


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):  # Adjust out_channels based on number of instances
        super(UNet, self).__init__()

        # Encoder (Downsampling Path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Upsampling Path)
        self.up4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Outputs instance segmentation map

    def conv_block(self, in_channels, out_channels):
        """Two Convolutional Layers with ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        """Upsampling + Conv Layer"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2, stride=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2, stride=2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2, stride=2))

        # Decoding path with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Final output
        output = self.final_conv(d1)
        return output


# ---------------
# Optimizers
# ---------------
data_dir = r"/home/imad/NeedleAlignment/data/needleonly"
dataset = LoadDataset(data_dir, None)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
device = torch.device("cpu")
model = UNet(in_channels=3, out_channels=dataset.__getitem__(0)[1].max().item()+1).to(device)
criteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criteration(outputs, masks)
        loss.backwards()

        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), os.path.join(r"/home/imad/NeedleAlignment/output/Models", "FirstBasicModel", f"unet_epoch{epoch+1}.pth"))

