from drawing import *
from imageutils import *
import pickle
from rectangle import Rectangle
from point import Point
import cvcolor
from midicontrol import MidiControlManager, MidiControl, set_logging
from heatmap import HeatMap
from extract_features import calc_hog, extract_features
import Utils
from multiprocessing.dummy import Pool
from cardetection import CarDetection

set_logging(True)

class VehicleDetector(MidiControlManager):
    def __init__(self, save_false_positives=False, use_multires_classifiers=True, use_hires_classifier=False):
        super().__init__()
        self.use_multires_classifiers = use_multires_classifiers and not use_hires_classifier
        self.use_hires_classifier=use_hires_classifier
        self.scale = 2
        self.crop_y_rel = np.array((0.55,0.90))
        self.crop_y = None
        self.grid = None
        self.decision_threshold = MidiControl(self,"decision_threshold", 80, 0.95, 0.0, 1.0)
        self.load_svc()
        self.heatmap = None
        self.pool = Pool(8)
        self.save_false_positives = save_false_positives
        self.detected_cars = []
        self.hog_y_1 = None
        self.hog_y_2 = None
        self.annotate = True
        self.annotated_heatmap = None

        if self.save_false_positives:
            self.false_positive_dir_name = "false_positives_%s" % Utils.date_file_name().split(".")[0]
            for size in (64,32,16):
                Utils.mkdir("%s/%d" % (self.false_positive_dir_name,size))


    def load_svc(self):
        filename = "svc_multires.pickle" if not self.use_hires_classifier else "svc_hires.pickle"

        with open(filename, "rb") as f:
            self.svc, self.scaler = pickle.load(f)
            self.svc_sizes = sorted(list(self.svc.keys()), reverse=True)
            print(self.svc_sizes)


    def process(self, frame, frame_count):
        self.poll()
        self.frame_count = frame_count
        self.frame = frame
        scaled_frame = scale_img(self.frame, 1 / self.scale)
        self.cropped_frame = self.crop_frame(scaled_frame)
        self.cropped_frame_yuv = bgr2yuv(self.cropped_frame)

        self.initialize_scan()
        self.scan_edges()
        self.scan_vehicle_bboxes()
#        self.sliding_window()

        self.update_heatmap()
        self.update_car_detections()
        self.draw_detected_cars()
        self.draw_evaluated_windows()
        self.draw_detections()
        self.draw_bboxes()
        self.draw_grid_on_cropped_img()
        #self.calc_test_hog()
        self.annotate_heatmap()

        return self.frame


    def draw_grid_on_cropped_img(self):
        if self.grid == None:
            w, h = img_size(self.cropped_frame)
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

        if self.annotate:
            self.cropped_frame = blend_img(self.cropped_frame, self.grid, 0.25)


    def crop_frame(self, scaled_frame):
        w,h = img_size(scaled_frame)
        if self.crop_y == None:
            y1,y2 = (self.crop_y_rel * h)
            h_cropped = 64 * 4 // self.scale
            y1 = y2 - h_cropped
            self.crop_y = np.array((y1,y2),np.int)
            self.cropped_image_size = np.array((h_cropped, w))
            self.cropped_frame_rect = Rectangle(pos=(0, 0), size=(w,h_cropped))
            self.vanishing_point = Point(w // 2, h_cropped // 4 - h_cropped // 8)
            self.left_edge = [Point(0, h_cropped // 4 + h_cropped // 16), self.vanishing_point]

        return scaled_frame[self.crop_y[0]:self.crop_y[1]]


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
        window_yuv = self.cropped_frame_yuv[int(window.y1):int(window.y2), int(window.x1):int(window.x2)]
        X = np.array(extract_features(window_yuv, window_size))
        normalized_feature_vector = self.scaler[window_size].transform(X)
        return self.svc[window_size].predict(normalized_feature_vector)[0]


    def evaluate_windows_of_size(self, windows, window_size):
        X = []
        for w in windows:
            window_yuv = self.cropped_frame_yuv[int(w.y1):int(w.y2), int(w.x1):int(w.x2)]
            if w.width() != window_size or w.height() != window_size:
                window_yuv = cv2.resize(window_yuv, (window_size, window_size))

            ppc = 8 if self.use_hires_classifier else 16
            X.append(extract_features(window_yuv, window_size, ppc=ppc))
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
                window_img = crop_img(self.cropped_frame, r.x1, r.y1, r.x2, r.y2)
                save_img(window_img, "%s/%d/%04d-%04d" % (self.false_positive_dir_name, window_size, self.frame_count, self.false_positive_count))
                self.false_positive_count+=1

        return result


    def evaluate_windows(self, windows):
        result = []
        if self.use_multires_classifiers:
            for ws in self.svc_sizes:
                windows_of_size = [w for w in windows if int(w.height()) == ws]
                if not windows_of_size:
                    continue
                result.extend(self.evaluate_windows_of_size(windows_of_size, ws))

            other_windows = [w for w in windows if not w.height() in self.svc_sizes]
            if other_windows:
                result.extend(self.evaluate_windows_of_size(other_windows, 64))

        else:
            result.extend(self.evaluate_windows_of_size(windows, 64))
        return result


    def scan_edges(self):
        h, w = self.cropped_image_size
        windows = self.left_edge_windows()
        windows.extend([r.mirror_x(w//2) for r in windows])
        windows.extend(self.top_edge_windows())
        result = self.evaluate_windows(windows)
        for window_rect, i_score in result:
            self.detections.append((window_rect, i_score))


    def left_edge_windows(self):
        result = []
        h,w = self.cropped_image_size
        frame_rect = Rectangle(pos=(0,0), size=(w,h))

        for width,height,x,y in (48,48,0,16),(32,32,0,0),(32,32,8,0),(32,32,16,0),(32,32,0,8),(32,32,8,8),(32,32,16,8):
            window = Rectangle(pos=(x, y), size=(width, height)) * 4 // self.scale
            assert frame_rect.contains(window)
            result.append(window)
        return result


    def top_edge_windows(self):
        h,w = self.cropped_image_size
        frame_rect = Rectangle(pos=(0,0), size=(w,h))
        ws = 16 * 4 // self.scale
        result = []
        x1, x2 = 0, w - ws
        x, y = x1, 4 * 4 // self.scale
        while x <= x2:
            window = Rectangle(pos=(x, y), size=ws)
            assert frame_rect.contains(window)
            result.append(window)
            x += ws / 2

        return result


    def window_size_for_rect(self, rect):
        for ws in reversed(self.svc_sizes):
            if ws >= 0.65 * rect.height():
                return ws
        return 64


    def scan_vehicle_bboxes(self):
        h,w = self.cropped_image_size
        windows = []
        for d in self.detected_cars:
            d_rect = (d.current_rect() // self.scale).translate((0,-self.crop_y[0]))

            if d_rect.aspect_ratio() >= 0.75:
                windows.append(d_rect.intersect(self.cropped_frame_rect))

            ws = self.window_size_for_rect(d_rect)
            dx = ws // 4
            dy = ws // 4

            pos = d_rect.center()
            rect_w,rect_h = d_rect.size()
            rect_w = max(rect_w, ws)
            rect_h = max(rect_h, ws)
            w_rect = Rectangle(center=pos,size=(rect_w,rect_h))
            w_rect = w_rect.expand(dx,dy).intersect(self.cropped_frame_rect)

            y = w_rect.y1
            while y <= w_rect.y2 - ws:
                x = w_rect.x1
                while x <= w_rect.x2 - ws:
                    r = Rectangle(pos=(x,y), size=ws)
                    windows.append(r)
                    x+= dx
                y+= dy

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
        if self.annotate:
            for r,_ in self.detections:
                offset = Point(0, self.crop_y[0])
                color = cvcolor.pink if self.is_false_positive_candidate(r) else cvcolor.gray50
                draw_rectangle(self.frame, r.translate(offset)*self.scale, color=color)


    def draw_evaluated_windows(self):
        if self.annotate:
            for w in self.evaluated_windows:
                offset = Point(0, self.crop_y[0])
                draw_rectangle(self.frame, w.translate(offset)*self.scale, color=cvcolor.gray70)


    def draw_bboxes(self):
        if self.annotate:
            for r in map(self.transform_rect, self.heatmap.get_bboxes()):
                draw_rectangle(self.frame, r, color=cvcolor.green)

        for d in self.detected_cars:
            if not d.is_real:
                continue
            r = d.current_rect()
            w,h = r.size() / 4
            x,y = r.p1()
            draw_rectangle(self.frame, r, color=cvcolor.orange, thickness=2)

            if self.annotate:
                put_text(self.frame,"%d" % w, Point(x+2*w-8,y-24), color=cvcolor.orange)
                put_text(self.frame,"%d" % h, Point(x-16,y+2*h-16), color=cvcolor.orange)


    def draw_detected_cars(self):
        i = 0
        for d in self.detected_cars:
            if not d.is_real:
                continue
            r = d.current_rect()
            car_img = crop_img(self.frame, r.x1,r.y1,r.x2,r.y2)
            size, margin = 128, 32
            car_img = cv2.resize(car_img, (size,size))
            paste_img(self.frame,car_img, (margin,(size + margin) * i + margin))
            i+=1


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


    def annotate_heatmap(self):
        r = (self.heatmap.map * 16).astype(np.uint8)
        zero = np.zeros_like(r)
        heatmap_overlay = join_bgr(zero,zero,r)
        self.annotated_heatmap = blend_img(self.cropped_frame, heatmap_overlay, 1.0)


    def calc_test_hog(self):
        import skimage.feature

        def my_hog(img):
            return skimage.feature.hog(img, orientations=9, pixels_per_cell=(ppc, ppc),
                                       cells_per_block=(2, 2),
                                       visualise=True, transform_sqrt=False,
                                       feature_vector=False,normalise=None)[1]

        ppc = 16
        ws = 64
        w,h = img_size(self.cropped_frame)
        self.hog_y_1 = my_hog(self.cropped_frame_yuv[:, :, 0])
        self.hog_y_2 = np.zeros_like(self.hog_y_1)
        delta_ij = 3
        for y in range(0,h,ws):
            for x in range(0,w,ws):
                hog = my_hog(self.cropped_frame_yuv[y:y + ws, x:x + ws, 0])
                self.hog_y_2[y:y+ws,x:x+ws] = hog

        self.hog_y_12 = (np.abs(self.hog_y_1 - self.hog_y_2)*128.0).astype(np.uint8)
        self.hog_y_1 = (self.hog_y_1 * 32).astype(np.uint8)
        self.hog_y_2 = (self.hog_y_2 * 32).astype(np.uint8)
