try:
    import os
    import re
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
    import matplotlib.patches as patches
    import cv2
    import data
    from skimage.feature import peak_local_max
    import skimage.transform as st
    import math
    import consts as C
except ImportError:
    print("Need to fix the installation")
    raise


def create_bounding_rectangle(image, tf_details, temp_cropped_df):
    seq = 0
    rectangle_x = np.array([], dtype='int64')
    rectangle_y = np.array([], dtype='int64')

    for row in tf_details.iterrows():
        tf_x, tf_y, tf_color, zoom = row[1][0:4]
        if math.isnan(tf_x):
            continue
            # plt.plot(tf_x, tf_y, 'ro', color=tf_color, markersize=3)
        if tf_color == C.RED:
            x = tf_x + C.X_AXIS * (1 - zoom)
            y = tf_y - C.Y_AXIS * (1 - zoom)
            top_right = (x, y if y > 0 else 0)
            x = tf_x - C.X_AXIS * (1 - zoom)
            y = tf_y + (C.HEIGHT - C.Y_AXIS) * (1 - zoom)
            bottom_left = (x if x > 0 else 0, y)
        else:
            x = tf_x + C.X_AXIS * (1 - zoom)
            y = tf_y - (C.HEIGHT - C.Y_AXIS) * (1 - zoom)
            top_right = (x, y if y > 0  else 0)
            x = tf_x - C.X_AXIS * (1 - zoom)
            y = tf_y + C.Y_AXIS * (1 - zoom)
            bottom_left = (x if x > 0 else 0, y)

        rectangle_x = np.append(rectangle_x, [top_right[0], bottom_left[0]])
        rectangle_y = np.append(rectangle_y, [top_right[1], bottom_left[1]])

        temp_cropped_df.loc[len(temp_cropped_df.index)] = \
            [seq, False, False, '', top_right[0], bottom_left[0], top_right[1], bottom_left[1], tf_color]
        seq += 1

    return rectangle_x, rectangle_y


def get_rect(gray, pix_range, tf_x, tf_y, threshold, color, op):
    size = -1
    for i in range(pix_range):
        if int(tf_y) + i < gray.shape[0]:
            if gray[int(tf_y) + i if op == '+' else int(tf_y) - i][int(tf_x) if int(tf_x) < 1024 else 1023] < threshold:
                size = i
                break
    for i in range(pix_range):
        if int(tf_y) - i > 0:
            if gray[int(tf_y) - i if op == '+' else int(tf_y) + i][int(tf_x) if int(tf_x) < 1024 else 1023] < threshold:
                size += i
                break

    if color == C.RED:
        top_right = (tf_x + size, tf_y - size if tf_y - size > 0 else 0)
        bottom_left = (tf_x - size if tf_x - size > 0 else 0, tf_y + 4 * size)
    else:
        top_right = (tf_x + 0.8 * size, tf_y - 2.8 * size if tf_y - 2.8 * size > 0 else 0)
        bottom_left = (tf_x - 0.8 * size if tf_x - 0.8 * size > 0 else 0, tf_y + size)
    return top_right, bottom_left, size


def new_bounding_rectangle(image, tf_axis_and_color, temp_cropped_df):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seq = 0
    rectangle_x = np.array([], dtype='int64')
    rectangle_y = np.array([], dtype='int64')

    for row in tf_axis_and_color.iterrows():
        tf_x, tf_y, color = row[1][:3]
        if math.isnan(tf_x):
            continue
        # plt.plot(tf_x, tf_y, 'ro', color=color, markersize=3)
        if color == C.RED:
            top_right, bottom_left, size = get_rect(gray, 25, tf_x, tf_y, 0.300, color, '+')
        else:  # green color
            top_right, bottom_left, size = get_rect(gray, 50, tf_x, tf_y, 0.250, color, '-')

        if size == -1 or size < 3:
            top_right = (tf_x + 5, tf_y - 2*5 if tf_y - 2*5 > 0 else 0)
            bottom_left = (tf_x - 5 if tf_x - 5 > 0 else 0, tf_y + 5)

        rectangle_x = np.append(rectangle_x, [top_right[0], bottom_left[0]])
        rectangle_y = np.append(rectangle_y, [top_right[1], bottom_left[1]])

        temp_cropped_df.loc[len(temp_cropped_df.index)] = \
            [seq, False, False, '', top_right[0], bottom_left[0], top_right[1], bottom_left[1], color]
        seq += 1

    return rectangle_x, rectangle_y


def connected_component(label_image, num_orange_pix, center_point):
    pass


def calculate_percentage(num_orange_pix, total_pix, label_image, center_point: tuple):
    """Return True, False or Ignore"""
    percentage = 100 * float(num_orange_pix)/float(total_pix)
    if percentage < 40:
        return False
    elif percentage >= 60:
        if percentage >= 95:
            return connected_component(label_image, num_orange_pix, center_point)
        return True
    return C.IS_IGNORE


def get_top_rights_bottom_lefts(coordinates_x, coordinates_y):
    top_right_arr = []
    bottom_left_arr = []
    for i in range(len(coordinates_x)):
        tuple_point = (coordinates_x[i], coordinates_y[i])
        if not i % 2:
            top_right_arr.append(tuple_point)
        else:
            bottom_left_arr.append(tuple_point)
    return top_right_arr, bottom_left_arr


def label_calculate(paths_image, coordinates_x, coordinates_y, temp_cropped_df):
    label_im = np.array(Image.open(paths_image[1]).convert('RGB'))
    top_right_arr, bottom_left_arr = get_top_rights_bottom_lefts(coordinates_x, coordinates_y)

    for i, top_right in enumerate(top_right_arr):
        crop_tl = label_im[int(top_right[1]): int(bottom_left_arr[i][1]),
                           int(bottom_left_arr[i][0]): int(top_right[0])]

        count_orange_pixels = np.count_nonzero(np.all(crop_tl == C.ORANGE_PIXEL, axis=2))

        diff_x = int(top_right[0]) - int(bottom_left_arr[i][0])
        diff_y = int(bottom_left_arr[i][1]) - int(top_right[1])
        sum_pixel_crop = diff_x * diff_y

        res = calculate_percentage(count_orange_pixels, sum_pixel_crop, paths_image[1], (diff_x // 2, diff_y // 2))

        if res == C.IS_IGNORE:
            temp_cropped_df.iat[i, C.INDEX_IGNORE] = True
        elif res:
            temp_cropped_df.iat[i, C.INDEX_TRUE] = True
        # plt.imshow(crop_tl)
        # plt.show()


def crop_tf_from_image(image_name: str, image: np.array, temp_cropped_df: pd.DataFrame) -> None:

    if not os.path.exists(C.PATH_CROPPED):
        os.mkdir(C.PATH_CROPPED)

    if not os.path.exists(C.PATH_CROPPED + C.DIRECTORY_TRUE):
        os.mkdir(C.PATH_CROPPED + C.DIRECTORY_TRUE)
    if not os.path.exists(C.PATH_CROPPED + C.DIRECTORY_FALSE):
        os.mkdir(C.PATH_CROPPED + C.DIRECTORY_FALSE)
    if not os.path.exists(C.PATH_CROPPED + C.DIRECTORY_IGNORE):
        os.mkdir(C.PATH_CROPPED + C.DIRECTORY_IGNORE)

    for index in temp_cropped_df.index:
        cropped_image = image[int(temp_cropped_df[C.Y0][index]):int(temp_cropped_df[C.Y1][index]),
                              int(temp_cropped_df[C.X1][index]):int(temp_cropped_df[C.X0][index])]

        cropped_image_name = image_name.replace(C.EXTENSION_IMG, '') + '_' + temp_cropped_df[C.COL][index]

        if temp_cropped_df[C.IS_TRUE][index]:
            cropped_image_name += C.T
            directory = C.DIRECTORY_TRUE
        elif not temp_cropped_df[C.IS_IGNORE][index]:
            cropped_image_name += C.F
            directory = C.DIRECTORY_FALSE
        else:
            cropped_image_name += C.I
            directory = C.DIRECTORY_IGNORE

        cropped_image_name += '_' + str(temp_cropped_df[C.SEQ][index]).zfill(5) + C.PNG
        temp_cropped_df.at[index, C.PATH] = cropped_image_name
        plt.imsave(C.PATH_CROPPED + directory + '/' + cropped_image_name, st.resize(cropped_image, (200, 100)))


def create_pandas_cropped_images():
    df = data.create_data_frame(C.attention_results_h5)
    path_dict = data.create_data()
    cropped_df = pd.DataFrame(columns=[C.SEQ, C.IS_TRUE, C.IS_IGNORE, C.PATH, C.X0, C.X1,
                                       C.Y0, C.Y1, C.COL])

    # image_tf_details - panda contains the images : all traffic lights x, y, color and zoom
    for image_name in path_dict.keys():
        im = plt.imread(path_dict[image_name][0])
        temp_cropped_df = pd.DataFrame(
            columns=[C.SEQ, C.IS_TRUE, C.IS_IGNORE, C.PATH, C.X0, C.X1, C.Y0,
                     C.Y1, C.COL])

        image_tf_details = df.loc[df[C.PATH] == image_name][[C.X, C.Y, C.COL, C.ZOOM]]
        tf_coordinates_x, tf_coordinates_y = create_bounding_rectangle(im, image_tf_details, temp_cropped_df)
        label_calculate(path_dict[image_name], tf_coordinates_x, tf_coordinates_y, temp_cropped_df)

        # image_axis_and_color = df.loc[df['path'] == image_name][['x', 'y', 'col']]
        # tf_coordinates_x, tf_coordinates_y = new_bounding_rectangle(im, image_axis_and_color, temp_cropped_df)
        # label_calculate(path_dict[image_name], tf_coordinates_x, tf_coordinates_y, temp_cropped_df)

        # print(temp_cropped_df)

        crop_tf_from_image(image_name, im, temp_cropped_df)
        cropped_df = pd.concat([cropped_df, temp_cropped_df], ignore_index=True)

        # plt.imshow(im)
        # plt.plot(tf_coordinates_x, tf_coordinates_y, 'mx', color='y', markersize=3)
        # plt.show()

    return cropped_df


def main():
    cropped_df = create_pandas_cropped_images()
    print(cropped_df)


if __name__ == '__main__':
    main()
