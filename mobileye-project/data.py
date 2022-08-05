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
    import cv2
    from skimage.feature import peak_local_max
    import re
except ImportError:
    print("Need to fix the installation")
    raise


PATH_HEAD = "C:/leftImg8bit/train/"
EXTENSION_LABEL = "_gtFine_color.png"


def create_data_frame(df_path):
    pd.set_option('display.max_columns', None, 'display.max_rows', None)
    return pd.read_hdf(df_path)


def create_data_structure_images_names_paths(df):
    dictionary_name_image_paths = {}
    for image_name in df["path"]:
        if image_name not in dictionary_name_image_paths.keys():
            list_words = re.split("_", image_name)
            label = '_'.join(list_words[: len(list_words) - 1]) + EXTENSION_LABEL
            dir_name = list_words[0]
            path_image = PATH_HEAD + dir_name + '/' + image_name
            label_image = PATH_HEAD + dir_name + '/' + label
            dictionary_name_image_paths[image_name] = (path_image, label_image)
            # dictionary_name_image_paths.setdefault(image_name, []).append(path_image)
    return dictionary_name_image_paths


def get_values_by_name_key(image_name: str, dictionary: dict):
    return dictionary[image_name]


def create_data():
    df = create_data_frame('attention_results.h5')

    my_dict = create_data_structure_images_names_paths(df)

    # (test)
    name_of_image = "aachen_000001_000019_leftImg8bit.png"
    path_img, label_img = get_values_by_name_key(name_of_image, my_dict)
    print(path_img)
    print(label_img)


if __name__ == '__main__':
    create_data()










