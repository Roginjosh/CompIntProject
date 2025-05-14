# Taken from Pytorch Tutorial https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_to_index = {
            label: idx for idx, label in enumerate(self.img_labels["label"].unique())
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.normpath(self.img_labels.at[idx, "filepaths"])
        image = read_image(img_path).float() / 255.0
        label_str = self.img_labels.at[idx, "label"]
        label = self.label_to_index[label_str]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
