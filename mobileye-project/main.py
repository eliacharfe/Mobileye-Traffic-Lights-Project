try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2
    from skimage.feature import peak_local_max

except ImportError:
    print("Need to fix the installation")
    raise

BLACK = -0.64
WHITE = 0.36
threshold = 100
CROPPED_PERCENT = 0.6
kernel = np.array([[BLACK,BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK]])


def find_specific_color(image, color: int):
    img_color = image[:, :, color]
    conv = sg.convolve2d(img_color, kernel)
    relevant_list = np.where(conv > 5, conv, 0)
    coordinates = peak_local_max(relevant_list, min_distance=15)
    coordinates -= 5
    return coordinates

def find_tfl_lights(image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use image, kwargs
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    cropped_image = image[:int(image.shape[0] * CROPPED_PERCENT), :]

    red_coordinates = find_specific_color(cropped_image, 0)
    green_coordinates = find_specific_color(cropped_image, 1)
    return red_coordinates[:, 1], red_coordinates[:, 0], green_coordinates[:, 1], green_coordinates[:, 0]


def test_find_tfl_lights(image_path, json_path=None, fig_num=tuple):
    """ Run the attention code """
    image = np.array(plt.imread(image_path))
    plt.imshow(image)
    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def plot_image(image, original_img):
    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)
    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(image)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "test"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()

#   For filtering the image using json
#   def show_image_and_gt(image, objs, fig_num=None):
#     plt.figure(fig_num).clf()
#     plt.imshow(image)
#     labels = set()
#     if objs is not None:
#         for o in objs:
#             poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
#             plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
#             labels.add(o['label'])
#         if len(labels) > 1:
#             plt.legend()
