import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model import common


class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1, texture_loss=False):
        super(VGG, self).__init__()

        self.texture_loss = texture_loss

        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]

        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])
        elif texture_loss:
            self.layers = [8, 17, 26, 35]
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def perceptual_forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

    def texture_forward(self, sr, hr):

        def _forward(x):
            x = self.sub_mean(x)
            features = []
            for name, layer in enumerate(self.vgg):
                x = layer(x)
                if name in self.layers:
                    features.append(x)
                    if len(features) == len(self.layers):
                        break
            return features

        with torch.no_grad():
            vgg_hr = _forward(hr)
            vgg_sr = _forward(sr)

        # gram_hr = [common.gram_matrix_v2(y) for y in vgg_hr]
        # gram_sr = [common.gram_matrix_v2(y) for y in vgg_sr]

        text_loss = 0
        for m in range(len(vgg_sr)):
            text_loss += F.mse_loss(common.gram_matrix(vgg_sr[m]), common.gram_matrix(vgg_hr[m]))

        return text_loss

    def forward(self, sr, hr):

        if self.texture_loss:
            loss = self.texture_forward(sr, hr)
        else:
            loss = self.perceptual_forward(sr, hr)

        return loss
