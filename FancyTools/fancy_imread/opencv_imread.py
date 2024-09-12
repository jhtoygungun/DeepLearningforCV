# Name: opencv-python
# Version: 4.7.0.72

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = '../datasets/lenna/lena_std.jpg'
img = cv2.imread(img_path)

print(type(img)) # <class 'numpy.ndarray'>
print(img.shape) # (512, 512, 3)

cv2.imshow("lenna", img)
cv2.waitKey(0)

print(img)
# [[[125 137 225]
# BGR模式

# b,g,r = cv2.split(img)
# img2 = cv2.merge([r,g,b])

img2 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

plt.subplot(121);plt.imshow(img) # expects distorted color
plt.subplot(122);plt.imshow(img2) # expect true color
plt.show()