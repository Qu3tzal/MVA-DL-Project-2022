import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        pass
