import os
from torchvision import transforms
import torch
from PIL import Image
import numpy
from torch.utils.data import Dataset, DataLoader
import pickle


class TCAV_dataset(Dataset):
    def __init__(self, data_path, transform=True):
        with open(data_path, 'rb') as f:
            self.data_info = pickle.load(f)

        self.lbm_scores = self.data_info['lbm_score']
        self.data_paths = self.data_info['paths']
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        image = Image.open(img_path)
        label = self.lbm_scores[idx]
        if self.transform:
            image = transforms.Compose(
                [transforms.ToTensor()])(image)
        return (image, label)
