import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

image = np.array(Image.open(r"C:\Users\leele\Desktop\Bootcamp\MobileyeProject\mobileye-project-mobileye-group-5\mobileye-project\test\berlin_000455_000019_leftImg8bit.png"))


# read the image
img = cv2.imread(r"C:\Users\leele\Desktop\Bootcamp\MobileyeProject\mobileye-project-mobileye-group-5\mobileye-project\test\berlin_000455_000019_leftImg8bit.png")

# convert the BGR image to HSV colour space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_green = np.array([112, 167, 128])
upper_green = np.array([180, 252, 190])
mask = cv2.inRange(hsv, lower_green, upper_green)

# perform bitwise and on the original image arrays using the mask
res = cv2.bitwise_and(img, img, mask=mask)
image_max = ndi.maximum_filter(res, size=20, mode='constant')

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
plt.imshow(image_max, cmap='gray')

x=0
