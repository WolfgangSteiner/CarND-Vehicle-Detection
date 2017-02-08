import skimage.feature
from imageutils import *
import numpy as np


def calc_hog(ch, ppc):
    return skimage.feature.hog(ch, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2),
                               visualise=False, transform_sqrt=True, feature_vector=False, normalise=None).astype(np.float32)


def calc_color_histogram(img_yuv):
    return np.concatenate([np.histogram(img_yuv[:,:,i_ch], bins=32, range=(0,256))[0] for i_ch in range(3)], axis=0).astype(np.float32)


def calc_spatial_color_binning(img_yuv):
    h,w = img_yuv.shape[0:2]
    factor = 8 / h
    return np.concatenate([np.ravel(scale_img(img_yuv[:,:,i_ch],factor)) for i_ch in range(3)], axis=0).astype(np.float32)


def extract_features(img_yuv, window_size, hog_data=None, hog_pos=None):
    ppc = 16  * window_size // 64
    if hog_data is None:
        hog_feature_array = [np.ravel(calc_hog(img_yuv[:,:,i_ch], ppc)) for i_ch in range(3)]
    else:
        hog_window_size = window_size // ppc - 1
        j,i = hog_pos
        hog_feature_array = [np.ravel(hog_data[i_ch][j:j+hog_window_size, i:i+hog_window_size, :, :, :]) for i_ch in range(3)]

    features = []
    features.append(np.concatenate(hog_feature_array, axis=0))
    #features.append(color_hist = calc_color_histogram(img_yuv))
    features.append(calc_spatial_color_binning(img_yuv))
    return np.concatenate(features, axis=0)


