import torch.nn as nn
import torch.nn.functional as F


class SameSizeOutputModelWrapper(nn.Module):

    def __init__(self, model):
        super(SameSizeOutputModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        size = x.shape[-2:]
        return F.interpolate(output, size=size, mode='bilinear', align_corners=False)
