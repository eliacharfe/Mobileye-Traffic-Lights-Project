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

X_AXIS = 22
Y_AXIS = 15
HEIGHT = 115

# ORANGE_PIXEL = [0.98, 0.667, 0.118]
# ORANGE_PIXEL = [0.98039216, 0.6666667, 0.11764706]
ORANGE_PIXEL = [250, 170, 30]


def create_bounding_rectangle(image, tf_details, temp_cropped_df):
    seq = 0
    rectangle_x = np.array([], dtype='int64')
    rectangle_y = np.array([], dtype='int64')

    for row in tf_details.iterrows():
        tf_x, tf_y, tf_color, zoom = row[1][0:4]
        plt.plot(tf_x, tf_y, 'ro', color=tf_color, markersize=3)
        if tf_color == 'r':
            top_right = (tf_x + X_AXIS*(1-zoom), tf_y - Y_AXIS*(1-zoom))
            bottom_left = (tf_x - X_AXIS*(1-zoom), tf_y + (HEIGHT - Y_AXIS)*(1-zoom))

        else:
            top_right = (tf_x + X_AXIS*(1-zoom), tf_y - (HEIGHT - Y_AXIS)*(1-zoom))
            bottom_left = (tf_x - X_AXIS*(1-zoom), tf_y + Y_AXIS*(1-zoom))

        rectangle_x = np.append(rectangle_x, [top_right[0], bottom_left[0]])
        rectangle_y = np.append(rectangle_y, [top_right[1], bottom_left[1]])

        temp_cropped_df.loc[len(temp_cropped_df.index)] = \
            [seq, False, False, '', top_right[0], bottom_left[0], top_right[1], bottom_left[1], tf_color]
        seq += 1

    return rectangle_x, rectangle_y


def new_bounding_rectangle(image, tf_axis_and_color, temp_cropped_df):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seq = 0
    rectangle_x = np.array([], dtype='int64')
    rectangle_y = np.array([], dtype='int64')

    for row in tf_axis_and_color.iterrows():
        tf_x, tf_y, color = row[1][:3]
        size = -1
        plt.plot(tf_x, tf_y, 'ro', color=color, markersize=3)
        if color == 'r':
            for i in range(25):
                if int(tf_y) + i < gray.shape[0]:
                    if gray[int(tf_y)+i][int(tf_x)] < 0.300:
                        size = i
                        break
            for i in range(25):
                if int(tf_y) - i > 0:
                    if gray[int(tf_y)-i][int(tf_x)] < 0.300:
                        size += i
                        break
            top_right = (tf_x + size, tf_y - size if tf_y - size > 0 else 0)
            bottom_left = (tf_x - size if tf_x - size > 0 else 0, tf_y + 4 * size)

        else:  # green color
            for i in range(50):
                if int(tf_y) + i < gray.shape[0]:
                    if gray[int(tf_y)-i][int(tf_x)] < 0.250:
                        size = i
                        break
            for i in range(50):
                if int(tf_y) - i > 0:
                    if gray[int(tf_y)+i][int(tf_x)] < 0.250:
                        size += i
                        break
            top_right = (tf_x + 0.8*size , tf_y - 2.8*size if tf_y - 2.8*size > 0 else 0)
            bottom_left = (tf_x - 0.8 * size if tf_x - 0.8 * size > 0 else 0, tf_y + size)

        if size == -1 or size < 3:
            top_right = (tf_x + 5,tf_y - 2*5 if tf_y - 2*5 > 0 else 0)
            bottom_left = (tf_x - 5 if tf_x - 5 > 0 else 0, tf_y + 5)

        rectangle_x = np.append(rectangle_x, [top_right[0], bottom_left[0]])
        rectangle_y = np.append(rectangle_y, [top_right[1], bottom_left[1]])

        temp_cropped_df.loc[len(temp_cropped_df.index)] = \
            [seq, False, False, '', top_right[0], bottom_left[0], top_right[1], bottom_left[1], color]
        seq += 1

    return rectangle_x, rectangle_y


def calculate_percentage(num_orange_pix, total_pix):
    """Return True, False or Ignore"""
    percentage = 100 * float(num_orange_pix)/float(total_pix)
    # print(f"percent: {percentage}")
    if percentage < 40:
        return False
    elif percentage >= 60:
        return True
    return "is_ignored"


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
    label_path = paths_image[1]
    label_im = np.array(Image.open(label_path).convert('RGB'))
    top_right_arr, bottom_left_arr = get_top_rights_bottom_lefts(coordinates_x, coordinates_y)

    for i, top_right in enumerate(top_right_arr):
        crop_tl = label_im[int(top_right[1]): int(bottom_left_arr[i][1]),
                           int(bottom_left_arr[i][0]): int(top_right[0])]

        count_orange_pixels = np.count_nonzero(np.all(crop_tl == ORANGE_PIXEL, axis=2))

        diff_x = int(top_right_arr[i][0]) - int(bottom_left_arr[i][0])
        diff_y = int(bottom_left_arr[i][1]) - int(top_right_arr[i][1])
        sum_pixel_crop = diff_x * diff_y

        res = calculate_percentage(count_orange_pixels, sum_pixel_crop)

        if res == "is_ignored":
            temp_cropped_df["is_ignored"].loc[temp_cropped_df['x0'] == top_right[0]] = True
        elif res:
            temp_cropped_df["is_true"][temp_cropped_df['x0'] == top_right[0]] = True

        # plt.imshow(crop_tl)
        # plt.show()


def create_pandas_cropped_images():
    df = data.create_data_frame('attention_results.h5')
    path_dict = data.create_data()
    cropped_df = pd.DataFrame(columns=['seq', 'is_true', 'is_ignored', 'path', 'x0', 'x1', 'y0', 'y1', 'color'])
    #  cropped_df.loc[len(cropped_df.index)] = [0, False, False,'', 52, 12, 43, 23, 'r']

    # image_tf_details - panda contains the images : all traffic lights x, y, color and zoom
    for image_name in path_dict.keys():
        im = plt.imread(path_dict[image_name][0])
        temp_cropped_df = pd.DataFrame(
            columns=['seq', 'is_true', 'is_ignored', 'path', 'x0', 'x1', 'y0', 'y1', 'color'])

        # image_tf_details = df.loc[df['path'] == image_name][['x', 'y', 'col', 'zoom']]
        # tf_coordinates_x1, tf_coordinates_y1 = create_bounding_rectangle(im, image_tf_details, temp_cropped_df)
        # label_calculate(path_dict[image_name], tf_coordinates_x1, tf_coordinates_y1, temp_cropped_df)

        image_axis_and_color = df.loc[df['path'] == image_name][['x', 'y', 'col']]
        tf_coordinates_x, tf_coordinates_y = new_bounding_rectangle(im, image_axis_and_color, temp_cropped_df)
        label_calculate(path_dict[image_name], tf_coordinates_x, tf_coordinates_y, temp_cropped_df)
        print(temp_cropped_df)

        cropped_df = pd.concat([cropped_df, temp_cropped_df], ignore_index=True)

        plt.imshow(im)
        # plt.plot(tf_coordinates_x1, tf_coordinates_y1, 'mx', color='m', markersize=3)
        plt.plot(tf_coordinates_x, tf_coordinates_y, 'mx', color='y', markersize=3)
        plt.show()

    return cropped_df


def main():
    cropped_df = create_pandas_cropped_images()


if __name__ == '__main__':
    main()
