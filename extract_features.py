import skimage.feature
from imageutils import *
import numpy as np


def calc_hog(ch, ppc, sqrt=False):
    return skimage.feature.hog(ch, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2),
                               visualise=False, transform_sqrt=sqrt, feature_vector=True,
                               normalise=None).astype(np.float32)


def calc_color_histogram(img_yuv):
    return np.concatenate([np.histogram(img_yuv[:,:,i_ch], bins=32)[0] for i_ch in range(3)], axis=0).astype(np.float32)


def calc_spatial_color_binning(img_yuv):
    h,w = img_yuv.shape[0:2]
    factor = 16 / h
    return np.concatenate([np.ravel(scale_img(img_yuv[:,:,i_ch],factor)) for i_ch in range(3)], axis=0).astype(np.float32)


def extract_features(img_yuv, window_size, ppc=16, hog_data=None, hog_pos=None):
    ppc = ppc * window_size // 64
    y,u,v = split_channels(img_yuv)
    #y = cv2.equalizeHist(y)

    features = []
    features.append(calc_hog(u,ppc, True))
    features.append(calc_hog(y,ppc, False))
    features.append(calc_hog(v,ppc, False))
    features.append(calc_color_histogram(img_yuv))
    features.append(calc_spatial_color_binning(img_yuv))
    return np.concatenate(features, axis=0)


