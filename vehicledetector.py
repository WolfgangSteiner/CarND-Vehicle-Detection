from drawing import *
from imageutils import *
import pickle
from rectangle import Rectangle
from point import Point
import cvcolor
from calibrate_camera import undistort_image
from midicontrol import MidiControlManager, MidiControl, set_logging
from heatmap import HeatMap
from extract_features import calc_hog, extract_features
import Utils

set_logging(True)

class VehicleDetector(MidiControlManager):
    def __init__(self):
        super().__init__()
        self.scale = 2
        self.crop_y_rel = np.array((0.55,0.90))
        self.crop_y = None
        self.grid = None
        self.svc,self.scaler = self.load_svc()
        self.decision_threshold = MidiControl(self,"decision_threshold", 80, 0.0, 0.0, 8.0)
        self.heatmap = None

        for size in (64,32,16):
            Utils.mkdir("false_positives/%d" % size)


    def load_svc(self):
        with open("svc.pickle", "rb") as f:
            return pickle.load(f)

        raise ValueError
        return None


    def process(self, frame, frame_count):
        self.poll()
        self.frame_count = frame_count
        self.frame = undistort_image(frame)
        img = scale_img(self.frame, 1 / self.scale)
        self.cropped_img = self.crop_img(img)
        self.cropped_img_yuv = bgr2yuv(self.cropped_img)
        self.sliding_window()
        self.update_heatmap()
        self.draw_detections()
        self.draw_bboxes()
        self.draw_grid_on_cropped_img()

        return self.frame


    def draw_grid_on_cropped_img(self):
        if self.grid == None:
            w, h = img_size(self.cropped_img)
            self.grid = new_img((w, h))
            y = h // 2
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

            draw_line(self.grid, self.left_edge[0].astype(np.int), self.left_edge[1].astype(np.int), color=cvcolor.pink, antialias=False)

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
            h_cropped = y2 - y1
            self.cropped_image_size = np.array((h_cropped, w))
            self.vanishing_point = Point(w // 2, h_cropped // 4 - h_cropped // 8)
            self.left_edge = [Point(0, h_cropped // 4 + h_cropped // 16), self.vanishing_point]

        return img[self.crop_y[0]:self.crop_y[1]]


    def sliding_window(self):
        self.detections = []
        self.false_positive_count = 0
        for size,y1,y2,delta_y in ((64,0,0,16),(32,0,16,8),(16,0,8,4)):
            self.sliding_window_impl(size, y1, y2, delta_y)


    def sliding_window_impl(self, window_size, y1, y2, delta_y):
        slice = self.cropped_img[y1:y2+window_size,:,:]
        ppc = 16 * window_size // 64
        hog_for_slice = [calc_hog(ch,ppc) for ch in split_yuv(slice)]
        #spacial_colors =

        i = 0
        j = 0
        delta_ij = window_size//ppc - 1
        X = []
        inc_i = 2
        inc_j = delta_y // ppc
        num_windows_y = (hog_for_slice[0].shape[0] - delta_ij)
        num_windows_x = (hog_for_slice[0].shape[1] - delta_ij)
        window_positions = []
        for j in range(0, num_windows_y + 1, inc_j):
            y = y1 + j * ppc
            for i in range(0, num_windows_x + 1, inc_i):
                x = i * ppc
                window_rect = Rectangle.from_point_and_size(Point(i*ppc, y), Point(window_size, window_size))
                window_yuv = bgr2yuv(self.cropped_img[y:y+window_size,x:x+window_size])
                X.append(extract_features(window_yuv, window_size, hog_for_slice, (j,i)))
                window_positions.append(np.array((x,y)))

        X = np.array(X)
        window_positions = np.array(window_positions)
        normalized_feature_vector = self.scaler[window_size].transform(X)
        score = self.svc[window_size].decision_function(normalized_feature_vector)
        pos_window_indexes = np.where(score > self.decision_threshold.value)[0]
        pos_window_positions = window_positions[pos_window_indexes]

        for idx,(x,y) in zip(pos_window_indexes, pos_window_positions):
            window_rect = Rectangle.from_point_and_size(Point(x,y),Point(1.0,1.0)*window_size)
            i_score = score[idx]
            self.detections.append((window_rect,i_score))
            if False and self.frame_count < 125:
                false_positive_img = crop_img(self.cropped_img, window_rect.x1, window_rect.y1, window_rect.x2, window_rect.y2)
                save_img(false_positive_img, "false_positives/%d/%04d-%04d" % (window_size,self.frame_count, self.false_positive_count))
                self.false_positive_count += 1


    def draw_detections(self):
        for r,_ in self.detections:
            offset = Point(0, self.crop_y[0])
            draw_rectangle(self.frame, r.translate(offset)*4, color=cvcolor.gray50)


    def draw_bboxes(self):
        for r in self.heatmap.get_bboxes():
            offset = Point(0, self.crop_y[0])
            draw_rectangle(self.frame, r.translate(offset)*4, color=cvcolor.green)


    def update_heatmap(self):
        if self.heatmap is None:
            self.heatmap = HeatMap(self.cropped_image_size, self)

        self.heatmap.add_detections(self.detections)
        self.heatmap.update_map()