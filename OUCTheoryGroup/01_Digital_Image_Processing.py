import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
from OUCTheoryGroup.utils import get_dataset

img = get_dataset("lenna_jpg")
print(type(img))
print(img.shape)
print(img)
cv2.imshow('lenna', img)
cv2.waitKey(0)

# img = matplotlib.image.imread("datasets\lenna\lena_std.jpg")
# print(type(img))
# print(img.shape)
# print(img)

plt.imshow(img)
plt.show()

# img = np.transpose(img, (0, 1, 2))
# img = np.transpose(img, (0, 2, 1))
# img = np.transpose(img, (1, 0, 2))
# img = np.transpose(img, (1, 2, 0))
# img = np.transpose(img, (1, 0, 2))
# img = np.transpose(img, (2, 1, 0))
img = np.transpose(img, (2, 0, 1))

plt.imshow(img)
plt.show()
