# Name: scipy
# Version: 1.9.1

import scipy.ndimage as spyimage

img_path = '../datasets/lenna/lena_std.jpg'
img = spyimage.imread(img_path)

# AttributeError: module 'scipy.ndimage' has no attribute 'imread'