# Name: Pillow
# Version: 9.5.0

from PIL import Image
import numpy as np

img_path = '../datasets/lenna/lena_std.jpg'
img = Image.open(img_path)

print(type(img)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>

img = np.array(img)
print(type(img)) # <class 'numpy.ndarray'>
print(img.shape) # (512, 512, 3)

print(img)
# [[[225 137 125]