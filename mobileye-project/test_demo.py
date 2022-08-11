import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import skimage.transform as st

import consts as C
from data_utils import TrafficLightDataSet, ModelManager, MyNeuralNetworkBase
from mpl_goodies import nn_examiner_example

from torch import nn
from torch.utils.data import Dataset

from data_utils import device

import Bounding_Rectangle
import data


def create_crops_af_image(image_name: str):
    path_to_save_crops_of_image = 'C:/leftImg8bit/temp_crop'
    if not os.path.exists(path_to_save_crops_of_image):
        os.mkdir(path_to_save_crops_of_image)

    attention_df = data.create_data_frame(C.attention_results_h5)
    attention_df = attention_df.dropna()
    my_dict = data.create_data()
    full_path_to_img = my_dict[image_name][0]
    im = plt.imread(full_path_to_img)
    temp_cropped_df = pd.DataFrame(
        columns=[C.SEQ, C.IS_TRUE, C.IS_IGNORE, C.PATH, C.X0, C.X1, C.Y0, C.Y1, C.COL])

    image_tf_details = attention_df.loc[attention_df[C.PATH] == image_name][[C.X, C.Y, C.COL, C.ZOOM]]
    Bounding_Rectangle.create_bounding_rectangle(image_tf_details, temp_cropped_df)

    for index in temp_cropped_df.index:
        cropped_image = im[int(temp_cropped_df[C.Y0][index]):int(temp_cropped_df[C.Y1][index]),
                        int(temp_cropped_df[C.X1][index]):int(temp_cropped_df[C.X0][index])]

        cropped_image_name = image_name.replace(C.EXTENSION_IMG, '') + '_' + temp_cropped_df[C.COL][index]

        if temp_cropped_df[C.IS_TRUE][index]:
            cropped_image_name += C.T
        elif not temp_cropped_df[C.IS_IGNORE][index]:
            cropped_image_name += C.F
        else:
            cropped_image_name += C.I

        cropped_image_name += '_' + str(temp_cropped_df[C.SEQ][index]).zfill(5) + C.PNG
        temp_cropped_df.at[index, C.PATH] = cropped_image_name
        img = Image.fromarray(
            (st.resize(cropped_image, (C.default_crop_w, C.default_crop_h)) * 255).astype(np.uint8))
        img.save(path_to_save_crops_of_image + '/' + cropped_image_name)

    return temp_cropped_df


