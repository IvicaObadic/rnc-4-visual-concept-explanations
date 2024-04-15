import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import skimage

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets

from torchvision.io import read_image

EPSILON = 1e-10


def discretize_labels(income_values):
    # Define Income Percentiles
    min = np.percentile(income_values, q=0)
    q20 = np.percentile(income_values, q=20)
    q40 = np.percentile(income_values, q=40)
    q60 = np.percentile(income_values, q=60)
    q80 = np.percentile(income_values, q=80)
    max = np.percentile(income_values, q=100)

    # added EPSILON to avoid getting 0 value for the minimum values because with right=True, the left interval is open
    income_classes = np.digitize(income_values, bins=[min - EPSILON, q20, q40, q60, q80, max], right=True) - 1

    print(income_classes.min(), income_classes.max())

    return income_classes

class HouseholdIncomeDataset(Dataset):

    def __init__(self, split_file, transform, discrete_labels=False, target_transform=None):
        self.images_with_labels = pd.read_csv(split_file, index_col=0)
        self.transform = transform
        self.discrete_labels = discrete_labels
        self.target_transform = target_transform

        print("{} images in file {}".format(len(self.images_with_labels.index), split_file))

    def __len__(self):
        return len(self.images_with_labels)


    def __getitem__(self, index):
        image_info = self.images_with_labels.iloc[index]
        image_id = image_info["idINSPIRE"]
        image_full_path = image_info["image_path"]
        image = read_image(image_full_path)
        if self.transform:
            image = self.transform(image)

        if self.discrete_labels:
            income_label = torch.tensor(image_info["income_class"], dtype=torch.long)
        else:
            income_label = float(image_info["income"])

        return image_id, image, income_label

def check_non_void_image(image_path, null_thresh = 1):
    image = skimage.io.imread(image_path)
    if (100 * (image == 0).sum() / image.size) > null_thresh:
        return False
    return True

def collect_non_void_images(images_root_dir):
    print("Selecting non-void images")
    non_void_images = []
    for inter_sat_dir in os.listdir(images_root_dir):
        inter_sat_dir_full_path = os.path.join(images_root_dir, inter_sat_dir)
        if os.path.isdir(inter_sat_dir_full_path):
            for im_file in os.listdir(inter_sat_dir_full_path):
                if im_file.endswith(".png"):
                    image_inspire_id = im_file.split(".")[0].split("_")[-1]
                    image_path = os.path.join(inter_sat_dir_full_path, im_file)
                    if check_non_void_image(image_path):
                        non_void_images.append((image_inspire_id, image_path))
        elif os.path.isfile(inter_sat_dir_full_path) and inter_sat_dir_full_path.endswith(".png"):
            image_insipire_id = os.path.split(inter_sat_dir_full_path)[1].split("_")[-1].split(".")[0]
            image_path = inter_sat_dir_full_path
            if check_non_void_image(image_path):
                non_void_images.append((image_insipire_id, image_path))
    print("Total non-void images: {}".format(len(non_void_images)))

    non_void_images = pd.DataFrame(data=non_void_images, columns=["idINSPIRE", "image_path"])
    non_void_images.set_index("idINSPIRE", inplace=True)

    return non_void_images


def stratified_data_split(images_root_dir, labels_file):
    non_void_images = collect_non_void_images(images_root_dir)

    print("Discretizing labels")
    # merge with labels
    labels = pd.read_csv(labels_file)
    labels.rename({"IdINSPIRE": "idINSPIRE"}, axis=1, inplace=True)
    labels.dropna(subset=["income"], inplace=True)
    labels.set_index("idINSPIRE", inplace=True)

    img_with_income_label = pd.merge(non_void_images, labels, left_index=True, right_index=True)
    img_with_income_label["income_class"] = discretize_labels(img_with_income_label[["income"]])
    img_with_income_label.reset_index(inplace=True)

    images_train, images_test = train_test_split(img_with_income_label,
                                                 train_size=0.8,
                                                 test_size=0.2,
                                                 stratify=img_with_income_label["income_class"])
    images_train, images_val = train_test_split(images_train, test_size=0.2,
                                                stratify=images_train["income_class"])

    print("Proportions of income classes in training: ")
    print(np.unique(images_train["income_class"], return_counts=True))

    print("Proportions of income classes in validation: ")
    print(np.unique(images_val["income_class"], return_counts=True))

    print("Proportions of income classes in test: ")
    print(np.unique(images_test["income_class"], return_counts=True))

    images_train.to_csv(os.path.join(images_root_dir, "train.csv"))
    images_val.to_csv(os.path.join(images_root_dir, "val.csv"))
    images_test.to_csv(os.path.join(images_root_dir, "test.csv"))


if __name__ == '__main__':
    root_dir = "/home/ConceptDiscovery/SESEfficientCAM-master/data"
    #root_dir = "./data/"

    images_root_dir = os.path.join(root_dir, "imagery_out")
    labels_file = os.path.join(root_dir, "census_data", "squares_to_ses_2019.csv")

    stratified_data_split(images_root_dir, labels_file)
