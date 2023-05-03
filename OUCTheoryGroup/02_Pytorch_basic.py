import torch

# a number
x = torch.tensor(666)
print(x)

# 1d array(vector)
x = torch.tensor([1,2,3,4])
print(x)

# 2d array(matrix)
x = torch.ones(2,3)
print(x)

# nd array(tensor)
x = torch.ones(2,3,4)
print(x)

# empty tensor
x = torch.empty(5,3)
print(x)

# a random initialization tensor
x = torch.rand(5, 3)
print(x)

# zeros tensor
x = torch.zeros(5,3, dtype=torch.int64)
print(x)

# create a tensor based on exits
y = x.new_ones(5,3)  # x.new_* methonds: using the exits x's dtype, device...
print(y)
print(y.dtype)

z = torch.rand_like(x, dtype=torch.float)
print(z)

# operation
m = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
print(m.size(), m.size(0), m.size(1), sep=' -- ')

# return the number of elements of a tensor
print(m.numel())

# retrun any elements
print(m[0][1])

# return one row
print(m[1, :])

# return one column
print(m[:, 1])

# arange
n = torch.arange(4)
print(n)

# scalar product
print(m @ n)

print(m[[0]] @ n)

# transpose
print(m.t())
print(m.transpose(0, 1)) # transpose!!!

# linspace
a = torch.linspace(3, 8, 20)
print(a)


# normalization
from matplotlib import pyplot as plt
plt.hist(torch.randn(10**6).numpy(), 100)
# plt.show()

# cat
a = torch.randn(5).reshape(1, -1)
b = torch.randn(5).reshape(1, -1)

print(torch.cat((a, b), 0).shape)
print(torch.cat((a, b), 1).shape)
