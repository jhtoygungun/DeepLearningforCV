# Name: scipy
# Version: 1.9.1

import scipy.misc as misc

img_path = '../datasets/lenna/lena_std.jpg'
img = misc.imread(img_path)

# AttributeError: module 'scipy.misc' has no attribute 'imread'