from imageutils import *
from skimage.feature import hog

class VehicleDetector(object):
    def __init__(self):
        self.scale = 4
        self.crop_y_rel = np.array((0.55,0.90))
        self.crop_y = None


    def process(self,frame):
        img = scale_img(frame, 1 / self.scale)
        self.cropped_img = self.crop_img(img)
        h_ch, l_ch, s_ch = split_hls(img)
        self.hog_data, self.hog_image = hog(l_ch[self.crop_y[0]:self.crop_y[1],:], orientations=8, pixels_per_cell=(8,8), cells_per_block=(3,3), visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
        self.hog_image *= 4#(self.hog_data * 16).astype(np.uint8)
        return frame


    def crop_img(self, img):
        w,h = img_size(img)

        if self.crop_y == None:
            y1,y2 = (self.crop_y_rel * h)
            y1 = y2 - 64 * 4 // self.scale
            self.crop_y = np.array((y1,y2),np.int)

        cropped_img = np.copy(img)
        cropped_img[0:self.crop_y[0],:,:] //= 4
        cropped_img[self.crop_y[1]:h,:,:] //= 4

        return cropped_img
