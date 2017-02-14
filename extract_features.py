import skimage.feature
from imageutils import *
import numpy as np


def calc_hog(ch, ppc, sqrt=False, visualize=False, feature_vector=True):
    return skimage.feature.hog(ch, orientations=9, pixels_per_cell=(ppc, ppc),
                                cells_per_block=(2, 2),
                                visualise=visualize, transform_sqrt=sqrt, feature_vector=True,
                                normalise=None)


def visualize_hog(ch, ppc):
    _,img = calc_hog(ch, ppc, visualize=True)
    return expand_channel(img) * 32


def calc_color_histogram(img):
    n_ch = img.shape[2] if len(img.shape) > 2 else 1
    if n_ch > 1:
        return np.concatenate([np.histogram(img[:,:,i_ch], bins=32)[0] for i_ch in range(n_ch)], axis=0)
    else:
        return np.histogram(img[:,:], bins=32)[0]


def calc_spatial_color_binning(img):
    h,w = img.shape[0:2]
    factor = 16 / h
    n_ch = img.shape[2] if len(img.shape) > 2 else 1
    if n_ch > 1:
        return np.concatenate([np.ravel(scale_img(img[:,:,i_ch],factor)) for i_ch in range(n_ch)], axis=0)
    else:
        return np.ravel(scale_img(img[:,:], factor))


def extract_features(img_yuv, window_size, ppc=16, hog_data=None, hog_pos=None):
    ppc = ppc * window_size // 64
    y,u,v = split_channels(img_yuv)
    features = []
    features.append(calc_hog(y, ppc, False))
    features.append(calc_hog(u,ppc, False))
    features.append(calc_hog(v,ppc, False))
    features.append(calc_color_histogram(img_yuv))
    features.append(calc_spatial_color_binning(img_yuv))
    return np.concatenate(features, axis=0)


