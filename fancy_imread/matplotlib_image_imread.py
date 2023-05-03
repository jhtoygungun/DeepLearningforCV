# Name: matplotlib
# Version: 3.3.0

import matplotlib.image as matimage

img_path = '../datasets/lenna/lena_std.jpg'
img = matimage.imread(img_path)

print(type(img)) # <class 'numpy.ndarray'>
print(img.shape) # (512, 512, 3)

print(img)
# [[[225 137 125]