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
except ImportError:
    print("Need to fix the installation")
    raise


def main():
    pd.set_option('display.max_columns', None, 'display.max_rows', None)
    df = pd.read_hdf('attention_results.h5')
    print(df)


if __name__ == '__main__':
    main()