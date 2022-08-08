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
    import consts as const
except ImportError:
    print("Need to fix the installation")
    raise


def create_data_frame(df_path):
    pd.set_option('display.max_columns', None, 'display.max_rows', None)
    return pd.read_hdf(df_path)


def create_data_structure_images_names_paths(df):
    dictionary_name_image_paths = {}
    for image_name in df[const.PATH]:
        if image_name not in dictionary_name_image_paths.keys():
            list_words = re.split("_", image_name)
            label = '_'.join(list_words[: len(list_words) - 1]) +  const.EXTENSION_LABEL
            dir_name = list_words[0]
            path_image = const.PATH_HEAD + dir_name + '/' + image_name
            label_image = const.PATH_HEAD + dir_name + '/' + label
            dictionary_name_image_paths[image_name] = (path_image, label_image)
    return dictionary_name_image_paths


def get_values_by_name_key(image_name: str, dictionary: dict):
    return dictionary[image_name]


def create_data():
    df = create_data_frame(const.attention_results_h5)
    my_dict = create_data_structure_images_names_paths(df)
    return my_dict












