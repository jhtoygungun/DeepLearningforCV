import cv2

from utils import get_dataset

img = get_dataset("lenna_jpg")
cv2.imshow('lenna', img)
cv2.waitKey(0)

