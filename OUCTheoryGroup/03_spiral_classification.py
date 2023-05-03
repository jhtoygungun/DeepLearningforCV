import wget
# wget.download("https://raw.githubusercontent.com/Atcold/pytorch-Deep-Learning/master/res/plot_lib.py")

import random
import torch
from torch import nn, optim
import math
from plot_lib import plot_data, plot_model, set_default
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

seed = 12345
random.seed(seed)
torch.manual_seed(seed)

N = 1000 # the number of sample per class
D = 2 # the dimension of the feature
C = 3 # the classes of the samples
H = 100 # the number of hidden layer


X = torch.zeros(N * C, D).to(device)
Y = torch.zeros(N * C, dtype=torch.int64).to(device)

for c in range(C):
    index = 0
    t = torch.linspace(0, 1, N) # 在[0，1]间均匀的取10000个数，赋给t
    # 下面的代码不用理解太多，总之是根据公式计算出三类样本（可以构成螺旋形）
    # torch.randn(N) 是得到 N 个均值为0，方差为 1 的一组随机数，注意要和 rand 区分开
    inner_var = torch.linspace( (2*math.pi/C)*c, (2*math.pi/C)*(2+c), N) + torch.randn(N) * 0.2
    
    # 每个样本的(x,y)坐标都保存在 X 里
    # Y 里存储的是样本的类别，分别为 [0, 1, 2]
    for ix in range(N * c, N * (c + 1)):
        X[ix] = t[index] * torch.FloatTensor((math.sin(inner_var[index]), math.cos(inner_var[index])))
        Y[ix] = c
        index += 1

print("Shapes:")
print("X:", X.size())
print("Y:", Y.size())


plot_data(X, Y)
# plt.show()

# create a linear classification mode

learning_rate = 1e-3
lambda_l2 = 1e-5

model = nn.Sequential(
    nn.Linear(D, H),
    nn.Linear(H, C)
)

model.to(device)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

epoches = 1000
for epoch in range(epoches):
    y_pred = model(X)
    ls = loss(y_pred, Y)
    score, predicted = torch.max(y_pred, 1)
    acc = (Y == predicted).sum().float() / len(Y)

    print('[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f' % (epoch, ls.item(), acc))

    optimizer.zero_grad()
    ls.backward()
    optimizer.step()

print(y_pred.shape)
print(y_pred[10, :])
print(score[10])
print(predicted[10])

# Plot trained model
print(model)
plot_model(X, Y, model)
plt.show()


# create a two layers network

learning_rate = 1e-3
lambda_l2 = 1e-5
model = nn.Sequential(
    nn.Linear(D, H),
    nn.ReLU(),
    nn.Linear(H, C)
)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

for t in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    score, predicted = torch.max(y_pred, 1)
    acc = ((Y == predicted).sum().float() / len(Y))
    print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
    
    # zero the gradients before running the backward pass.
    optimizer.zero_grad()
    # Backward pass to compute the gradient
    loss.backward()
    # Update params
    optimizer.step()

# Plot trained model
print(model)
plot_model(X, Y, model)
plt.show()