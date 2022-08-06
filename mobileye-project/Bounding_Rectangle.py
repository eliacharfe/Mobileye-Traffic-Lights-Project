try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    import pandas as pd
    from scipy import signal as sg
    from scipy import ndimage as ndi
    from scipy.ndimage import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
    # for rectangle draw:
    import matplotlib.patches as patches
    import cv2
    import data
    from skimage.feature import peak_local_max
except ImportError:
    print("Need to fix the installation")
    raise


# def dataFrame_to_csv(df):
#     base_path = 'C:\\leftImg8bit_trainvaltest\\leftImg8bit\\train\\'
#     new_df = pd.DataFrame(columns=['index', 'path'])
#     for i in range(len(df.index)):
#         cur_path = base_path + df.loc[i]['path'].split('_')[0] + '\\' +

X_AXIS = 14
Y_AXIS = 14
HEIGHT = 100


def create_bounding_rectangle(image, tf_details):
    im = image
    rectangle_x = np.array([], dtype='int64')
    rectangle_y = np.array([], dtype='int64')

    for row in tf_details.iterrows():
        tf_x, tf_y, tf_color, zoom = row[1][0:4]
        plt.plot(tf_x, tf_y, 'ro', color='r', markersize=3)
        if tf_color == 'r':
            top_right = (tf_x + X_AXIS*(1-zoom), tf_y - Y_AXIS*(1-zoom))
            bottom_left = (tf_x - X_AXIS*(1-zoom), tf_y + (HEIGHT - Y_AXIS)*(1-zoom))

        else:
            top_right = (tf_x + X_AXIS*(1-zoom), tf_y - (HEIGHT - Y_AXIS)*(1-zoom))
            bottom_left = (tf_x - X_AXIS*(1-zoom), tf_y + Y_AXIS*(1-zoom))

        rectangle_x = np.append(rectangle_x, [top_right[0], bottom_left[0]])
        rectangle_y = np.append(rectangle_y, [top_right[1], bottom_left[1]])

    return rectangle_x, rectangle_y

def main():
    pd.set_option('display.max_columns', None, 'display.max_rows', None)
    df = pd.read_hdf('attention_results.h5')

    path_dict = data.create_data()
    cropped_df = pd.DataFrame(columns=['seq', 'is_true', 'is_ignored', 'path', 'x0', 'x1', 'y0', 'y1', 'color'])
    cropped_df.loc[len(cropped_df.index)] = [0, True, False, 'blabla.png', 52, 12, 43, 23, 'r']

    # image_tf_details - panda contains the images : all traffic lights x, y, color and zoom
    for image_name in path_dict.keys():
        image_tf_details = df.loc[df['path'] == image_name][['x', 'y', 'col', 'zoom']]
        im = plt.imread(path_dict[image_name][0])
        tf_coordinates_x, tf_coordinates_y = create_bounding_rectangle(im, image_tf_details)
        plt.imshow(im)
        plt.plot(tf_coordinates_x, tf_coordinates_y, 'ro', color='y', markersize=3)
        plt.show()

# use this 'aachen_000084_000019_leftImg8bit.png' with tf instead of image_name


if __name__ == '__main__':
    main()
