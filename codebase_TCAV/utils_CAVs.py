from PIL import Image
import os
from torch.utils.data import DataLoader, IterableDataset, Dataset
from utils.augmentation import *
from captum.concept import Concept
from random import sample
import math

class ConceptDataset(Dataset):
    def __init__(self, data_paths, device, num_concept_examples=None):
        self.path_list = []
        self.device = device
        for data_path in data_paths:
            concept_examples = []
            for concept_example_filename in os.listdir(data_path):
                concept_examples.append(os.path.join(data_path, concept_example_filename))
            if num_concept_examples is not None:
                num_examples_per_concept = math.ceil(num_concept_examples / len(data_paths))
                concept_examples = sample(concept_examples, k=min(len(concept_examples), num_examples_per_concept))
            self.path_list.extend(concept_examples)


    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path)
        image = test_transforms(image)
        return image.to(self.device)


def assemble_concept(name, id, concept_path, device):
    concept_data_path = os.path.join(concept_path, name) + "/"
    conceptDataLoader = DataLoader(ConceptDataset(data_paths=[concept_data_path], device=device))
    print("Number of examples {} for concept {}".format(len(conceptDataLoader), name))
    return Concept(id=id, name=name, data_iter=conceptDataLoader)


def assemble_random_concept(names, id, concept_path, device, num_examples_to_take):
    data_paths = [(os.path.join(concept_path, name) + "/") for name in names]
    concept_data_loader = DataLoader(ConceptDataset(data_paths=data_paths, device=device, num_concept_examples=num_examples_to_take))
    print("Number of examples for the random concept {}".format(len(concept_data_loader)))
    return Concept(id=id, name="random", data_iter=concept_data_loader)
