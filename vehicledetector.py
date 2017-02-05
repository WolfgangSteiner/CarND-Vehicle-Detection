from skimage.feature import hog
from drawing import *
from imageutils import *
import pickle
from rectangle import Rectangle
from point import Point
import cvcolor
from calibrate_camera import undistort_image
from midicontrol import MidiControlManager, MidiControl, set_logging
from heatmap import HeatMap
from extract_features import calc_hog

set_logging(True)

class VehicleDetector(MidiControlManager):
    def __init__(self):
        super().__init__()
        self.scale = 4
        self.crop_y_rel = np.array((0.55,0.90))
        self.crop_y = None
        self.grid = None
        self.svc,self.scaler = self.load_svc()
        self.threshold = MidiControl(self,"threshold", 80, 0.5, 0.0, 1.0)
        self.heatmap = None


    def load_svc(self):
        with open("svc.pickle", "rb") as f:
            return pickle.load(f)

        raise ValueError
        return None


    def process(self,frame):
        self.poll()
        self.frame = undistort_image(frame)
        img = scale_img(self.frame, 1 / self.scale)
        self.cropped_img = self.crop_img(img)
        self.cropped_img_y, _, _ = split_yuv(self.cropped_img)
        #self.hog_data, self.hog_image = hog(self.cropped_img_y, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        #                                    visualise=True, transform_sqrt=False, feature_vector=False, normalise=None)

        #self.hog_image *= 4
        self.draw_grid_on_cropped_img()

        self.sliding_window()
        self.update_heatmap()
        self.draw_detections()

        return self.frame


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
            self.cropped_image_size = np.array((y2-y1, w))

        return img[self.crop_y[0]:self.crop_y[1]]


    def sliding_window(self):
        self.detections = []
        for size,y in ((64,0), (32,0), (32,8), (32,16), (32,24),(16,0),(16,4),(16,8)):
            self.sliding_window_impl(size,y)


    def sliding_window_impl(self, window_size, y):
        slice = self.cropped_img[y:y+window_size,:]
        ppc = 16 * window_size // 64

        hog_for_slice = [calc_hog(ch,ppc) for ch in split_yuv(slice)]
        i = 0
        delta_i = window_size//ppc - 1
        while i <= hog_for_slice[0].shape[1] - delta_i:
            hog_for_window = [np.ravel(hog_for_slice[ch][:,i:i+window_size//ppc-1,:,:,:]) for ch in range(0,3)]
            feature_vector = self.scaler[window_size].transform(np.concatenate(hog_for_window, axis=0))
            score = self.svc[window_size].decision_function(feature_vector)
            if score > self.threshold.value:
                rect = Rectangle.from_point_and_size(Point(i*ppc, y), Point(window_size, window_size))
                self.detections.append((rect,score))
            i += 1


    def draw_detections(self):
        for r,_ in self.detections:
            offset = Point(0, self.crop_y[0])
            draw_rectangle(self.frame, r.translate(offset)*4, color=cvcolor.green)


    def update_heatmap(self):
        if self.heatmap is None:
            self.heatmap = HeatMap(self.cropped_image_size, 0.25)

        self.heatmap.add_detections(self.detections)
        self.heatmap.update_map()