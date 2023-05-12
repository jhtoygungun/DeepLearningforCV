import torch
import torch.nn as nn
import torch.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv = nn.Conv2d()
        