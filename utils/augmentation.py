import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

resize_size = 500
image_size = 500
resize_transform = v2.Resize((resize_size, resize_size), antialias=True)
center_crop_transform = v2.CenterCrop(image_size)

class NormalizeImage(object):
    def __call__(self, img):
        EPSILON = 1e-10
        min, max = img.min(), img.max()
        return (img - min) / (EPSILON + max - min)


train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    resize_transform,
    v2.RandAugment(),
    v2.RandomErasing(),
    v2.ToDtype(torch.float32, scale=True),
    NormalizeImage()])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    resize_transform,
    v2.ToDtype(torch.float32, scale=True),
    NormalizeImage()
])

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_transforms(training_objective, transforms):
    if training_objective == "contrastive":
        return TwoCropTransform(transforms)
    else:
        return transforms

