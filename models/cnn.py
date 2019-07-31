import torch
import torch.nn as nn

from typing import Tuple

class ConvNet(nn.Module):
    """ Examples to use ConvNet:
        
        from commons.torch_utils import flatten
        
        input_size = (224,224)
        conv_layers = (
            nn.Conv2d(3, 64, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2))
        )
        fc_layers = (
            nn.Linear(flatten(conv_layers, input_size), 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096)
        )
        
        net = ConvNet(conv_layers, fc_layers)
    """

    def __init__(self, conv, fc):
        super(ConvNet, self).__init__()

        self._conv = self._conv = nn.Sequential(
            *conv
        )
        self._fc = self._fc = nn.Sequential(
            *fc
        )

    def forward(self, x):
        x = self._conv(x)
        x = x.view((x.size(0), -1))
        x = self._fc(x)

        return x
