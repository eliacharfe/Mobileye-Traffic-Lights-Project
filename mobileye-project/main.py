try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2
    import pandas as pd
    from skimage.feature import peak_local_max
except ImportError:
    print("Need to fix the installation")
    raise


# kernel = np.array([[-0.04, -0.04, -0.04, -0.04, -0.04],
#                    [-0.04, 0.04,  0.04, 0.04, -0.04],
#                    [-0.04,  0.04,  0.04, 0.04, -0.04],
#                    [-0.04,  0.04,  0.04,  0.04, -0.04],
#                    [-0.04, -0.04, -0.04, -0.04, -0.04]])
#
#
# kernel = np.array([[-0.64, -0.64, -0.64, -0.64, -0.64],
#                    [-0.64, 1.1377777777,  1.1377777777, 1.1377777777, -0.64],
#                    [-0.64,  1.1377777777,  1.1377777777, 1.1377777777, -0.64],
#                    [-0.64,  1.1377777777,  1.1377777777,  1.1377777777, -0.64],
#                    [-0.64, -0.64, -0.64, -0.64, -0.64]])

#
# kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
#                    [0.04, -0.04,  -0.04, -0.04, 0.04],
#                    [0.04,  -0.04,  -0.04, -0.04, 0.04],
#                    [0.04,  -0.04,  -0.04,  -0.04,  0.04],
#                    [0.04, 0.04, 0.04, 0.04, 0.04]])
#
#
# kernel = np.array([[0.64, 0.64, 0.64, 0.64, 0.64],
#                    [0.64, -0.026666666666,  -0.026666666666, -0.026666666666, 0.64],
#                    [0.64,  -0.026666666666,  -0.026666666666, -0.026666666666, 0.64],
#                    [0.64, -0.026666666666,  -0.026666666666, -0.026666666666,  0.64],
#                    [0.64, 0.64, 0.64, 0.64, 0.64]])

BLACK = -0.52
WHITE = 0.48
kernel = np.array([[BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
                   [BLACK, BLACK, BLACK,  WHITE, WHITE, WHITE, WHITE, BLACK, BLACK, BLACK],
                   [BLACK, BLACK, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, BLACK, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
                   [BLACK, BLACK, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, BLACK, BLACK],
                   [BLACK, BLACK, BLACK,  WHITE, WHITE, WHITE, WHITE, BLACK, BLACK, BLACK],
                   [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK]])


# BLACK = -0.4
# GRAY = -0.1
# WHITE = 0.39183
#
# kernel = np.array([[BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
#                    [BLACK, GRAY,  GRAY,  GRAY,  GRAY,  GRAY,  GRAY,  GRAY,  GRAY, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GRAY,   BLACK],
#                    [BLACK, GRAY,  GRAY, GRAY,   GRAY,  GRAY,  GRAY,  GRAY,  GRAY, GRAY,   BLACK],
#                    [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK]
#                    ])
#

# BLACK = -0.54
# WHITE = 0.26666666667
# kernel = np.array([[BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,  BLACK],
#                    [BLACK, WHITE, WHITE,  WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,  BLACK],
#                    [BLACK, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK],
#                    [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,  BLACK]
#                    ])

threshold = 100
CROPPED_PERCENT = 0.6

tfl_table = pd.DataFrame(columns=['x_coordinate', 'y_coordinate', 'color', 'conv_value', 'image_name'])


def convolve_red(image, h, image_path):
    # red_filtered_image = Image.fromarray(image[:, :, 0])
    red_filtered_image = image[:, :, 0]
    plt.subplot(2, 2, 2, sharex=h, sharey=h)
    red_conv = sg.convolve2d(red_filtered_image, kernel)
    plt.imshow(red_conv)
    #plt.imshow(red_conv > 6)
    plt.title("Convolved red")

    relevant_list = np.where(red_conv > 5.10, red_conv, 0)
    coordinates = peak_local_max(relevant_list, min_distance=35)
    coordinates -= 5

    coordinates_info = list(map(lambda coordinate: [coordinate[0], coordinate[1], "r",
                                                    red_conv[coordinate[0] + 5, coordinate[1] + 5],
                                                    image_path], coordinates))

    return coordinates, coordinates_info


def convolve_green(image, h, image_path):
    green_filtered_image = image[:, :, 1]
    plt.subplot(2, 2, 3, sharex=h, sharey=h)
    green_conv = sg.convolve2d(green_filtered_image, kernel)
    plt.imshow(green_conv)
    #plt.imshow(green_conv > 6)
    plt.title("Convolved green")

    relevant_list = np.where(green_conv >= 4.5, green_conv, 0)
    coordinates = peak_local_max(relevant_list, min_distance=35)
    coordinates -= 5

    coordinates_info = list(map(lambda coordinate: [coordinate[0], coordinate[1], "g",
                                                    green_conv[coordinate[0] + 5, coordinate[1] + 5],
                                                    image_path], coordinates))

    return coordinates, coordinates_info


# def find_specific_color(image, color: int):
#     img_color = image[:, :, color]
#     conv = sg.convolve2d(img_color, kernel)
#     relevant_list = np.where(conv > 5, conv, 0)
#     coordinates = peak_local_max(relevant_list, min_distance=15)
#     coordinates -= 5
#     return coordinates


def find_tfl_lights(image: np.ndarray, *args):
    """
    Detect candidates for TFL lights. Use image, kwargs
    :param image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    plt.imshow(image)
    plt.title("Original image")

    red_coordinates, red_coordinates_info = convolve_red(image, args[0], args[1])
    green_coordinates, green_coordinates_info = convolve_green(image, args[0], args[1])

    global tfl_table
    tfl_table = pd.concat([tfl_table, pd.DataFrame(red_coordinates_info + green_coordinates_info,
                                                   columns=['x_coordinate', 'y_coordinate', 'color', 'conv_value',
                                                            'image_name'])], ignore_index=True)
    return red_coordinates[:, 1], red_coordinates[:, 0], green_coordinates[:, 1], green_coordinates[:, 0]


def test_find_tfl_lights(image_path, json_path=None, fig_num=tuple):
    """ Run the attention code """
    
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    if not objects:
        image = np.array(plt.imread(image_path))
        cropped_image = image[:int(image.shape[0] * CROPPED_PERCENT), :]
        h = plt.subplot(2, 2, 1)

        red_x, red_y, green_x, green_y = find_tfl_lights(cropped_image, h, image_path)
        plt.subplot(2, 2, 4, sharex=h, sharey=h)
        plt.imshow(cropped_image)
        plt.plot(green_x, green_y, 'ro', color='g', markersize=3)
        plt.plot(red_x, red_y, 'ro', color='r', markersize=3)
        plt.title(f"black = {BLACK}, white = {WHITE}")
        plt.show()


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
