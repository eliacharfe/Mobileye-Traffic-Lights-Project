from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def imshow_components():
    comp_image = np.array(Image.open(r'C:/leftImg8bit/train/aachen/aachen_000004_000019_gtFine_color.png').convert('L'))

    all_labels = measure.label(comp_image)
    blobs_labels = measure.label(comp_image, background=0)

    comp_id = all_labels[288][1992]
    total_orange_pix = np.count_nonzero(all_labels == comp_id)

    x = comp_image[comp_id, cv2.CC_STAT_LEFT]
    print(comp_id)

imshow_components()