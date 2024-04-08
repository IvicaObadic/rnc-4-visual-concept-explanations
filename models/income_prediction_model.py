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


class SocioeconomicInferencePredictor(torch.nn.Module):

    def __init__(self, encoder, num_classes=1, use_l_softmax_loss=False, margin=2):
        super(SocioeconomicInferencePredictor, self).__init__()

        self.encoder = encoder
        self.use_l_softmax_loss = use_l_softmax_loss
        self.margin = margin
        encoder_output_dim = 1280 if self.encoder.encoder_name == "efficientNet" else 2048
        #self.dropout_layer = nn.Dropout(p=0.2, inplace=True)

        if use_l_softmax_loss:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.prediction_layer = LSoftmaxLinear(input_features=encoder_output_dim, output_features=num_classes, margin=self.margin, device=device)
            self.prediction_layer.reset_parameters()
        else:
            self.prediction_layer = nn.Linear(in_features=encoder_output_dim, out_features=num_classes)

    def forward(self, x, target=None):
        features = self.encoder(x)
        if self.use_l_softmax_loss:
            logits = self.prediction_layer(features, target)
        else:
            logits = self.prediction_layer(features)
        return logits

    def model_name(self):
        model_name = self.encoder.model_name()
        if self.use_l_softmax_loss:
            model_name = "{}_lsoftmax_{}".format(model_name, self.margin)

        return model_name


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


def init_model(objective="contrastive", encoder_name="efficientNet", pretrained=True, encoder_weights_path=None, num_classes=1):
    encoder = Encoder(pretrained=pretrained, encoder_name=encoder_name)
    if encoder_weights_path is not None:
        encoder_state_dict = torch.load(encoder_weights_path)["state_dict"]
        print(encoder_state_dict)
        print(encoder)
        new_state_dict = {}
        for k, v in encoder_state_dict.items():
            k = k.replace("model.", "")
            new_state_dict[k] = v
        encoder_state_dict = new_state_dict
        encoder.load_state_dict(encoder_state_dict)
        print("Probing mode, disabling gradients for the pretrained model")
        for param in encoder.parameters():
            param.requires_grad = False

        # encoder.load_state_dict(encoder_state_dict, strict=False)
    if objective == "contrastive":
        return encoder
    else:
        prediction_model = SocioeconomicInferencePredictor(encoder, num_classes=num_classes)
        return prediction_model

