import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

image = np.array(Image.open(r"C:\leftImg8bit\test\berlin\berlin_000000_000019_leftImg8bit.png"))


img = cv2.imread(r"C:\leftImg8bit\test\berlin\berlin_000000_000019_leftImg8bit.png")

# convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

# slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

plt.figure()
plt.clf()
h = plt.subplot(111)
plt.imshow(image)
plt.figure()
plt.clf()
plt.subplot(111, sharex=h, sharey=h)
plt.imshow(mask, cmap='gray')
plt.figure()
plt.clf()
plt.subplot(111, sharex=h, sharey=h)
plt.imshow(green, cmap='gray')
plt.show()
