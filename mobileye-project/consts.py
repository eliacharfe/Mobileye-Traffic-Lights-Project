# This file contains constants.. Mainly strings.
# It's never a good idea to have a string scattered in your code across different files, so just put them here
import os
import numpy as np

X_AXIS = 22
Y_AXIS = 15
HEIGHT = 115
ORANGE_PIXEL = [250, 170, 30]
INDEX_TRUE = 1
INDEX_IGNORE = 2

PATH_HEAD = "C:/leftImg8bit/train/"
EXTENSION_LABEL = "_gtFine_color.png"

threshold = 100
CROPPED_PERCENT = 0.6

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

X = 'x'
Y = 'y'
RED = 'r'
GRN = 'g'
COLOR = 'color'
SCORE = 'score'
SEQ = 'seq'
I = 'i'
J = 'j'
X0 = 'x0'
Y0 = 'y0'
X1 = 'x1'
Y1 = 'y1'
ZOOM = 'zoom'
PATH = 'path'
FULL_PATH = 'full_path'
CROP_PATH = 'crop_path'
COL = 'col'
LABEL = 'label'
BATCH = 'batch'
IS_TRUE = 'is_true'
IS_IGNORE = 'is_ignore'


default_base_dir = '../../data/v1'
attention_results = 'attention_results'
crops_dir = os.path.join(attention_results, 'crop')
attention_results_h5 = 'attention_results.h5'
crop_results_h5 = 'crop_results.h5'
models_output_dir = 'logs_and_models'
logs_dir = 'logs'

# Crop size:
default_crop_w = 32
default_crop_h = 96
