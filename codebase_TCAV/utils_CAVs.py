from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader, IterableDataset, Dataset
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
import glob
from captum.concept import Concept

class NormalizeImage(object):
    def __call__(self, img):
        EPSILON = 1e-10
        min, max = img.min(), img.max()
        return (img - min) / (EPSILON + max - min)

class ConceptDataset(Dataset):
    def __init__(self, data_paths, device):
        self.path_list = []
        self.device = device
        for data_path in data_paths:
            for concept_example_filename in os.listdir(data_path):
                self.path_list.append(os.path.join(data_path, concept_example_filename))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path)
        image = transforms.Compose([transforms.ToTensor(), transforms.Resize((500, 500)), NormalizeImage()])(image)
        return image.to(self.device)


def assemble_concept(name, id, concepts_path, device):
    concept_path = os.path.join(concepts_path, name) + "/"
    ConceptDataLoader = DataLoader(ConceptDataset(data_paths=[concept_path], device=device))

    print("Number of examples {} for concept {}".format(len(ConceptDataLoader), name))

    return Concept(id=id, name=name, data_iter=ConceptDataLoader)

def assemble_random_concept(names, id, concepts_path, device):
    data_paths = [(os.path.join(concepts_path, name) + "/") for name in names]
    ConceptDataLoader = DataLoader(ConceptDataset(data_paths=data_paths, device=device))

    print("Number of examples random_concepts for concept {}".format(len(ConceptDataLoader), "random"))

    return Concept(id=id, name="random", data_iter=ConceptDataLoader)
