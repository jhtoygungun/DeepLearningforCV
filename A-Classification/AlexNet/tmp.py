import torch
from torch import nn
m = nn.Conv1d(16, 33, 3, stride=2)
m=m.to('cuda:2')
input = torch.randn(20, 16, 50)
input=input.to('cuda:2')
output = m(input)
print(output)