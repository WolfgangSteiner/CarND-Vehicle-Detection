import skimage.feature
from imageutils import *
import numpy as np


def calc_hog(ch, ppc):
    return skimage.feature.hog(ch, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2),
                               visualise=False, transform_sqrt=True, feature_vector=False, normalise=None).astype(np.float32)


def extract_features(img, window_size):
    ppc = 16  * window_size // 64
    result = [np.ravel(calc_hog(ch, ppc)) for ch in split_yuv(img)]
    return np.concatenate(result, axis=0)


