import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import skimage.transform as st
import consts as C
import torchvision.transforms as transforms
import Bounding_Rectangle
import data
import data_utils


def create_crops_af_image(path_to_save_crops_of_image: str, image_name: str):
    # path_to_save_crops_of_image = 'C:/leftImg8bit/temp_crop'
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


def test(trained_model_path):

    image_name = 'aachen_000004_000019_leftImg8bit.png'
    path_to_dir = 'C:/leftImg8bit/temp_crop'
    crop_df = create_crops_af_image(path_to_dir, image_name)

    my_model = data_utils.ModelManager.load_model(trained_model_path)
    # # print(my_model)

    for i, row in crop_df.iterrows():
        # im = plt.imread(path_to_dir + '/' + row['path'])
        image = Image.open(path_to_dir + '/' + row['path'])

        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        img_tensor = transform(image)
        print(img_tensor)

        print(img_tensor.shape)

        preds = my_model(img_tensor.shape[:2])
        print(preds)

