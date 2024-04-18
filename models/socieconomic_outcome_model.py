import torch
import torch.nn as nn
from torchvision import models
import math
import torch
from torch import nn
from scipy.special import binom


class Encoder(torch.nn.Module):
    def __init__(self, pretrained=True, encoder_name="efficientNet"):
        super(Encoder, self).__init__()
        self.encoder_name = encoder_name
        if self.encoder_name == "efficientNet":
            encoder_fn = models.efficientnet_b0
        else:
            encoder_fn = models.resnet50

        self.pretrained = pretrained
        if self.pretrained:
            self.encoder = encoder_fn(weights="DEFAULT")
        else:
            self.encoder = encoder_fn(weights=None)

        if self.encoder_name == "efficientNet":
            self.encoder.classifier = nn.Identity()
        else:
            self.encoder.fc = nn.Identity()

    def forward(self, x):
        features = self.encoder(x)
        return features

    def model_name(self):
        model_name = "encoder_{}".format(self.encoder_name)
        if self.pretrained:
            model_name = "{}_pretrained".format(model_name)
        return model_name


class SocioeconomicOutcomePredictor(torch.nn.Module):

    def __init__(self, encoder, num_classes=1):
        super(SocioeconomicOutcomePredictor, self).__init__()
        self.encoder = encoder
        encoder_output_dim = 1280 if self.encoder.encoder_name == "efficientNet" else 2048
        self.prediction_layer = nn.Linear(in_features=encoder_output_dim, out_features=num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.prediction_layer(features)
        return logits

    def model_name(self):
        model_name = self.encoder.model_name()
        return model_name


def init_model(objective="contrastive", encoder_name="efficientNet", pretrained=False, encoder_weights_path=None, num_classes=1):
    encoder = Encoder(pretrained=pretrained, encoder_name=encoder_name)
    if encoder_weights_path is not None:
        print("Loading model checkpoint from {}".format(encoder_weights_path), flush=True)
        encoder_state_dict = torch.load(encoder_weights_path)["state_dict"]
        new_state_dict = {}
        for k, v in encoder_state_dict.items():
            k = k.replace("model.", "")
            new_state_dict[k] = v
        encoder_state_dict = new_state_dict
        encoder.load_state_dict(encoder_state_dict)
        print("Probing mode, disabling gradients for the pretrained model", flush=True)
        for param in encoder.parameters():
            param.requires_grad = False

    if objective == "contrastive":
        return encoder
    else:
        prediction_model = SocioeconomicOutcomePredictor(encoder, num_classes=num_classes)
        return prediction_model

