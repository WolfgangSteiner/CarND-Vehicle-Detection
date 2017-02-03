import cv2
import numpy as np
import cvcolor


def save_img(img, name, path="."):
    file_name = path + '/' + name
    if not (file_name.endswith(".png") or file_name.endswith(".jpg")):
        file_name += '.png'
    cv2.imwrite(file_name, img)


def load_img(name,path="."):
    file_name = path + '/' + name
    if not name.endswith(".png") and not name.endswith(".jpg"):
        name += '.png'
    return cv2.imread(file_name)


def img_size(img):
    return np.flipud(np.array(img.shape[0:2], np.int))


def new_img(size, color=cvcolor.black):
    w,h = size
    if color == cvcolor.black:
        return np.zeros((h, w,3),np.uint8)
    elif color == cvcolor.white:
        return np.ones((h, w,3),np.uint8) * 255
    else:
        img = new_img((size), cvcolor.black)
        img[:,:,:] = color
        return img


def num_channels(img):
    if len(img.shape) == 2:
        return 1
    else:
        return img.shape[2]


def paste_img(target_img, source_img, pos):
    h,w = source_img.shape[0:2]
    n_ch = num_channels(source_img)
    if n_ch == 1:
        for i_ch in range(0,num_channels(target_img)):
            target_img[pos[0]:pos[0]+h,pos[1]:pos[1]+w,i_ch] = source_img
    else:
        target_img[pos[0]:pos[0]+h,pos[1]:pos[1]+w,:] = source_img


def scale_img(img, factor):
    if factor != 1:
        return cv2.resize(img,None,fx=factor,fy=factor, interpolation=cv2.INTER_CUBIC)
    else:
        return img


def show_img(img, title=""):
    cv2.imshow(title, img)
    cv2.waitKey()

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def hls2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HLS2BGR)


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def split_hls(img):
    return split_channels(bgr2hls(img))


def join_hls(h,l,s):
    return hls2bgr(combine_channels(h,l,s))


def bgr2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def yuv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)


def split_yuv(img):
    return split_channels(bgr2yuv(img))


def join_yuv(y,u,v):
    return yuv2bgr(combine_channels(y,u,v))

def split_hsv(img):
    return split_channels(bgr2hsv(img))


def split_channels(img):
    return img[:,:,0], img[:,:,1], img[:,:,2]


def combine_channels(a,b,c):
    return np.stack((a,b,c),axis=2)


def expand_channel(c):
    return np.stack((c,c,c),axis=2).astype(np.uint8)


def expand_mask(m):
    return expand_channel(m) * 255


def AND(*args):
    result = args[0]
    for a in args[1:]:
        result = cv2.bitwise_and(result, a)
    return result

def OR(*args):
    result = args[0]
    for a in args[1:]:
        result = cv2.bitwise_or(result, a)
    return result

def NOT(a):
    return 1 - a
