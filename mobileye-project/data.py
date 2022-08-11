from typing import Dict, Tuple

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
    import consts as C
except ImportError:
    print("Need to fix the installation")
    raise


def create_data_frame(df_path: str):
    """
    Get a path to the dataframe and return the dataframe.
    :param df_path: A path to the dataframe.
    :return: The dataframe.
    """
    pd.set_option('display.max_columns', None, 'display.max_rows', None)
    return pd.read_hdf(df_path)


def create_data_structure_images_names_paths(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Get a DataFrame and create a dictionary which the key is name of the image according to the df
    and the value is a tuple of the full path image and the full path to the label image.
    :param df:
    :return: Dictionary
    """
    dictionary_name_image_paths = {}
    for image_name in df[C.PATH]:
        if image_name not in dictionary_name_image_paths.keys():
            list_words = re.split("_", image_name)
            label = '_'.join(list_words[: len(list_words) - 1]) + C.EXTENSION_LABEL
            dir_name = list_words[0]
            path_image = C.PATH_HEAD + dir_name + '/' + image_name
            label_image = C.PATH_HEAD + dir_name + '/' + label
            dictionary_name_image_paths[image_name] = (path_image, label_image)
    return dictionary_name_image_paths


def get_values_by_name_key(image_name: str, dictionary: dict) -> Tuple[str, str]:
    return dictionary[image_name]


def create_data() -> Dict[str, Tuple[str, str]]:
    """
    Create a dictionary which the key is name of the image (taken from the df) and the value is a tuple
    of the full path image and the full path to the label image.
    :return: Dictionary
    """
    df = create_data_frame(C.attention_results_h5)
    my_dict = create_data_structure_images_names_paths(df)
    return my_dict












