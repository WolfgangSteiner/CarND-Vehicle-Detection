from skimage.feature import hog

from drawing import *
from imageutils import *


class VehicleDetector(object):
    def __init__(self):
        self.scale = 4
        self.crop_y_rel = np.array((0.55,0.90))
        self.crop_y = None
        self.grid = None


    def process(self,frame):
        img = scale_img(frame, 1 / self.scale)
        self.cropped_img = self.crop_img(img)
        h_ch, l_ch, s_ch = split_hls(self.cropped_img)
        self.hog_data, self.hog_image = hog(l_ch, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                            visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
        self.hog_image *= 4#(self.hog_data * 16).astype(np.uint8)
        self.draw_grid_on_cropped_img()

        return frame

    def draw_grid_on_cropped_img(self):
        if self.grid == None:
            w, h = img_size(self.cropped_img)
            self.grid = new_img((w, h))
            y = h // 2
            print(y)
            for i in range(0, 3):
                draw_line(self.grid, (0, y), (w, y), color=cvcolor.white, thickness=1, antialias=False)
                y //= 2

            vanishing_point = (w // 2, h // 4 - h // 8)
            dw1 = w // 4
            dw2 = w // 2 + w // 4
            w11 = w // 2 - dw1
            w12 = w // 2 + dw1
            w21 = w // 2 - dw2
            w22 = w // 2 + dw2

            draw_line(self.grid, (w11, h), vanishing_point, color=cvcolor.white, antialias=False)
            draw_line(self.grid, (w12, h), vanishing_point, color=cvcolor.white, antialias=False)
            draw_line(self.grid, (w21, h), vanishing_point, color=cvcolor.white, antialias=False)
            draw_line(self.grid, (w22, h), vanishing_point, color=cvcolor.white, antialias=False)

            # draw_line(self.grid, (w11,h), (w11,0), color=cvcolor.white, antialias=False)
            # draw_line(self.grid, (w12,h), (w12,0), color=cvcolor.white, antialias=False)
            # draw_line(self.grid, (w21,h), (w21,0), color=cvcolor.white, antialias=False)
            # draw_line(self.grid, (w22,h), (w22,0), color=cvcolor.white, antialias=False)

        self.cropped_img = blend_img(self.cropped_img, self.grid, 0.25)


    def crop_img(self, img):
        w,h = img_size(img)

        if self.crop_y == None:
            y1,y2 = (self.crop_y_rel * h)
            y1 = y2 - 64 * 4 // self.scale
            self.crop_y = np.array((y1,y2),np.int)

        return img[self.crop_y[0]:self.crop_y[1]]
