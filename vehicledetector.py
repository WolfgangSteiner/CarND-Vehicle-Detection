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
from multiprocessing.dummy import Pool
from functools import partial
from cardetection import CarDetection
from itertools import product

set_logging(True)

class VehicleDetector(MidiControlManager):
    def __init__(self, save_false_positives=False):
        super().__init__()
        self.scale = 2
        self.crop_y_rel = np.array((0.55,0.90))
        self.crop_y = None
        self.grid = None
        self.svc,self.scaler = self.load_svc()
        self.decision_threshold = MidiControl(self,"decision_threshold", 80, 0.0, 0.0, 8.0)
        self.heatmap = None
        self.pool = Pool(8)
        self.save_false_positives = save_false_positives
        self.detected_cars = []
        self.hog_y_1 = None
        self.hog_y_2 = None

        if self.save_false_positives:
            self.false_positive_dir_name = "false_positives_%s" % Utils.date_file_name().split(".")[0]
            for size in (64,32,16):
                Utils.mkdir("%s/%d" % (self.false_positive_dir_name,size))


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

        self.initialize_scan()
        self.scan_edges()
        self.scan_vehicle_bboxes()
#        self.sliding_window()

        self.update_heatmap()
        self.update_car_detections()
        self.draw_evaluated_windows()
        self.draw_detections()
        self.draw_bboxes()
        self.draw_grid_on_cropped_img()
        #self.calc_test_hog()

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


    def initialize_scan(self):
        self.detections = []
        self.evaluated_windows = []
        self.false_positive_count = 0


    def sliding_window(self):
        for size, y1, y2, delta_y in ((64,0,0,32), (48, 16, 16, 24), (32, 8, 16, 8), (24, 0, 12, 12), (16, 0, 8, 8)):
            result = self.sliding_window_impl(size, y1, y2, delta_y)
            for window_rect, i_score in result:
                self.detections.append((window_rect,i_score))


    def sliding_window_impl(self, window_size, y1, y2, delta_y):
        ppc = 16 * window_size // 64
        X = []
        window_positions = []

        y = y1
        while y <= y2:
            window_positions_row = self.sliding_window_horizontal(None, window_size, ppc, y)
            window_positions.extend(window_positions_row)
            y += delta_y

        return self.evaluate_windows_of_size(window_positions, window_size)


    def sliding_window_horizontal(self, hog_for_slice, window_size, ppc, y):
        h,w = self.cropped_image_size

        X = []
        delta_x = window_size // 2
        x = 0
        window_positions=[]

        while x <= w - window_size:
            window_positions.append(Rectangle(pos=(x,y), size=window_size))
            x+=delta_x

        return window_positions


    def evaluate_window(self, window):
        window_size = window.height()
        window_yuv = self.cropped_img_yuv[int(window.y1):int(window.y2),int(window.x1):int(window.x2)]
        X = np.array(extract_features(window_yuv, window_size))
        normalized_feature_vector = self.scaler[window_size].transform(X)
        return self.svc[window_size].predict(normalized_feature_vector)[0]


    def evaluate_windows_of_size(self, windows, window_size):
        X = []
        for w in windows:
            window_yuv = self.cropped_img_yuv[int(w.y1):int(w.y2),int(w.x1):int(w.x2)]
            X.append(extract_features(window_yuv, window_size))
        X = np.array(X)

        windows = np.array(windows)
        normalized_feature_vector = self.scaler[window_size].transform(X)
        score = self.svc[window_size].predict(normalized_feature_vector)
        pos_window_indexes = np.where(score > self.decision_threshold.value)[0]
        pos_windows = windows[pos_window_indexes]
        pos_window_scores = score[pos_window_indexes]

        result = []
        self.evaluated_windows.extend(windows)
        for r, score in zip(pos_windows, pos_window_scores):
            result.append((r, score))
            if self.save_false_positives and self.is_false_positive_candidate(r):
                window_img = crop_img(self.cropped_img, r.x1, r.y1, r.x2, r.y2)
                save_img(window_img, "%s/%d/%04d-%04d" % (self.false_positive_dir_name, window_size, self.frame_count, self.false_positive_count))
                self.false_positive_count+=1

        return result


    def evaluate_windows(self, windows):
        result = []
        for ws in (64,48,32,24,16):
            windows_of_size = [w for w in windows if int(w.height()) == ws]
            if not windows_of_size:
                continue
            result.extend(self.evaluate_windows_of_size(windows_of_size, ws))
        return result


    def scan_edges(self):
        h, w = self.cropped_image_size
        windows = self.left_edge_window_rects()
        windows.extend([r.mirror_x(w//2) for r in windows])
        result = self.evaluate_windows(windows)
        for window_rect, i_score in result:
            self.detections.append((window_rect, i_score))


    def left_edge_window_rects(self):
        result = []
        for ws,x,y in (48,0,16),(48,0,12),(32,0,16),(32,0,12),(32,0,8),(32,0,4),(32,0,0):
            result.append(Rectangle(pos=(0,y),size=ws))
        return result


    def window_size_for_rect(self, rect):
        for ws in reversed((16,24,32,48,64)):
            if ws <= 1.0 * rect.height():
                return ws
        return 16


    def scan_vehicle_bboxes(self):
        h,w = self.cropped_image_size
        img_rect = Rectangle(pos=(0,0), size=(w,h))
        windows = []
        for d in self.detected_cars:
            d_rect = (d.current_rect() // self.scale).translate((0,-self.crop_y[0]))
            ws = self.window_size_for_rect(d_rect)
            dx = ws // 4
            dy = ws // 4

            pos = d_rect.center()
            rect_w,rect_h = d_rect.size()
            rect_w = max(rect_w, ws)
            rect_h = max(rect_h, ws)
            d_rect = Rectangle(center=pos,size=(rect_w,rect_h))

            y1 = max(0, d_rect.y1 - 2*dy)
            y2 = min(h, d_rect.y2 + 2*dy)
            x1 = max(0, d_rect.x1 - 2*dx)
            x2 = min(w, d_rect.x2 + 2*dx)

            y = y1
            while y <= y2 - ws:
                x = x1
                while x <= x2 - ws:
                    r = Rectangle(pos=(x,y), size=ws)
                    windows.append(r)
                    x+=dx
                y+=dy

        if  self.detected_cars and not windows:
            raise ValueError

        if windows:
            result = self.evaluate_windows(windows)
            for window_rect, i_score in result:
                self.detections.append((window_rect, i_score))


    def is_false_positive_candidate(self, window_rect):
        w,h = img_size(self.frame)
        r = self.transform_rect(window_rect)

        if self.frame_count < 125:
            return True
        elif r.x2 < w - w //3:
            return True
        elif any(map(lambda d: d.current_rect().intersects(r), self.detected_cars)):
            return False
        elif self.frame_count < 150 and r.x1 > w * 3 // 4:
            return False
        elif self.frame_count > 675 and self.frame_count < 700 and r.x1 > w * 3 // 4:
            return False
        return True


    def draw_detections(self):
        for r,_ in self.detections:
            offset = Point(0, self.crop_y[0])
            color = cvcolor.pink if self.is_false_positive_candidate(r) else cvcolor.gray50
            draw_rectangle(self.frame, r.translate(offset)*4, color=color)


    def draw_evaluated_windows(self):
        for w in self.evaluated_windows:
            offset = Point(0, self.crop_y[0])
            draw_rectangle(self.frame, w.translate(offset)*4, color=cvcolor.gray70)


    def draw_bboxes(self):
        for r in map(self.transform_rect, self.heatmap.get_bboxes()):
            draw_rectangle(self.frame, r, color=cvcolor.green)

        for d in self.detected_cars:
            r = d.current_rect()
            w,h = r.size() / 4
            x,y = r.p1()
            draw_rectangle(self.frame, r, color=cvcolor.orange, thickness=2)
            draw_rectangle(self.frame, d.current_rect_of_influence(), color=cvcolor.orange, thickness=1)
            put_text(self.frame,"%d" % w, Point(x+2*w-8,y-24), color=cvcolor.orange)
            put_text(self.frame,"%d" % h, Point(x-16,y+2*h-16), color=cvcolor.orange)


    def update_heatmap(self):
        if self.heatmap is None:
            self.heatmap = HeatMap(self.cropped_image_size, self)

        self.heatmap.add_detections(self.detections)
        self.heatmap.update_map()


    def transform_rect(self, rect):
        offset = Point(0, self.crop_y[0])
        return rect.translate(offset) * self.scale


    def update_car_detections(self):
        rect_list = [self.transform_rect(r) for r in self.heatmap.get_bboxes()]
        [d.tick() for d in self.detected_cars]

        # sort existing car detections by area: bigger rectangles represent closer cars
        self.detected_cars = sorted(self.detected_cars, key=lambda d: d.current_area(), reverse=True)

        for d in self.detected_cars:
            rect_list = d.update(rect_list)

        # remaining rectangles are newly detected vehicles:
        for r in rect_list:
            self.detected_cars.append(CarDetection(r))

        # remove old detections
        self.detected_cars = [d for d in self.detected_cars if d.is_alive()]


    def calc_test_hog(self):
        import skimage.feature

        def my_hog(img):
            return skimage.feature.hog(img, orientations=9, pixels_per_cell=(ppc, ppc),
                                       cells_per_block=(2, 2),
                                       visualise=True, transform_sqrt=False,
                                       feature_vector=False,normalise=None)[1]

        ppc = 16
        ws = 64
        w,h = img_size(self.cropped_img)
        self.hog_y_1 = my_hog(self.cropped_img_yuv[:,:,0])
        self.hog_y_2 = np.zeros_like(self.hog_y_1)
        delta_ij = 3
        for y in range(0,h,ws):
            for x in range(0,w,ws):
                hog = my_hog(self.cropped_img_yuv[y:y+ws,x:x+ws,0])
                self.hog_y_2[y:y+ws,x:x+ws] = hog

        self.hog_y_12 = (np.abs(self.hog_y_1 - self.hog_y_2)*128.0).astype(np.uint8)
        self.hog_y_1 = (self.hog_y_1 * 32).astype(np.uint8)
        self.hog_y_2 = (self.hog_y_2 * 32).astype(np.uint8)
