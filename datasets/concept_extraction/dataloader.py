import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tif


class labeled_data(Dataset):
    def __init__(self, image_indices, transform=None):
        self.transform = transform
        self.img_labels = image_indices

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # load image to tensor
        img_path = self.img_labels[idx]

        img = tif.imread(img_path)

        # use given tansformations
        if self.transform == 'RGB':
            img = img[:, :, :3]

        image_item = torch.tensor(img)
        return (self.img_labels[idx], image_item)
