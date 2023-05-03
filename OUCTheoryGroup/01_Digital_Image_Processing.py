import skimage.data as data
import skimage
import matplotlib.pyplot as plt
import numpy as np


from utils import get_dataset

# basic information
img = get_dataset("lenna_jpg")
print(type(img))
print(img.shape)

# img show
# plt.subplot(121)
plt.imshow(img)
plt.title("img 3 channels")
plt.axis("off")

# plt.subplot(122)
plt.imshow(img[:,:,2])
plt.title("img 1 channels")
plt.axis("off")
# plt.show()

# quickly handdle pics
camera  = data.camera() # get a pic called 'camera' from skimage lib
print(type(camera))
print(camera.shape)
camera[100:200, 100:200] = 0
plt.imshow(camera)
# plt.show()

camera  = data.camera()
mask = camera < 80
camera[mask] = 255
plt.imshow(camera, 'gray')
# plt.show()

# change color for real images
cat = data.chelsea()
plt.imshow(cat)
# plt.show()
red_cat = cat.copy()
reddish = cat[:, :, 0] > 160
red_cat[reddish] = [255, 0, 0]
plt.imshow(red_cat)
# plt.show()

# change RGB color to BGR for OpenCV
BGR_cat = cat[:, :, ::-1]
plt.imshow(BGR_cat)
# plt.show()
plt.close()

# data format conversion

# - img_as_float Convert to 64-bit floating point.
# - img_as_ubyte Convert to 8-bit uint.
# - img_as_uint Convert to 16-bit uint.
# - img_as_int Convert to 16-bit int.

from skimage import img_as_float, img_as_ubyte
float_cat = img_as_float(cat)
print(float_cat.dtype)

ubyte_cat = img_as_ubyte(cat)
print(ubyte_cat.dtype)

# histogram
img = data.camera()
plt.hist(img.ravel(), bins=256, histtype='step', color='black')
# plt.show()
plt.close()

# img segement
img = get_dataset("lenna_jpg")
img = skimage.color.rgb2gray(img)
print(img.shape)
plt.hist(img.ravel(), bins=256, histtype='step', color='black')
# plt.show()
plt.close()

plt.imshow(img>0.5)
# plt.show()
plt.close()

# cannny operator for edge detection
# https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
from skimage.feature import canny
from scipy import ndimage as ndi

img_edge = canny(img)
img_filled = ndi.binary_fill_holes(img_edge)

plt.subplot(121)
plt.imshow(img_edge, 'gray')
plt.subplot(122)
plt.imshow(img_filled, 'gray')
# plt.show()
plt.close()

# change the contrast
img = get_dataset("lenna_jpg")
# contrast stretching
p2, p98 = np.percentile(img, (2, 98))
from skimage import exposure
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
plt.imshow(img_rescale)
# plt.show()
plt.close()

# equalization
img = get_dataset("lenna_jpg")
img_eq = exposure.equalize_hist(img)
plt.hist(img_eq.ravel(), bins=256, histtype='step', color='black')
plt.imshow(img_eq)
# plt.show()
plt.close()

# adaptive equalization
img = get_dataset("lenna_jpg")
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
plt.imshow(img_adapteq)
# plt.show()
plt.close()



# Display results
def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf



fig = plt.figure(figsize=(16, 8))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

fig.tight_layout()
plt.show()
