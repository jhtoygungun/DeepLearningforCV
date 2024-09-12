# Name: imageio
# Version: 2.27.0

import imageio.v2 as imageio

img_path = '../datasets/lenna/lena_std.jpg'
img = imageio.imread(img_path)

print(type(img)) # <class 'imageio.core.util.Array'>
print(img.shape) # (512, 512, 3)

print(img)
# [[[225 137 125]