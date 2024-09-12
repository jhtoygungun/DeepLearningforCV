# Name: scikit-image
# Version: 0.20.0

import skimage.io as io

img_path = '../datasets/lenna/lena_std.jpg'
img = io.imread(img_path)

print(type(img)) # <class 'numpy.ndarray'>
print(img.shape) # (512, 512, 3)

print(img)
# [[[225 137 125]