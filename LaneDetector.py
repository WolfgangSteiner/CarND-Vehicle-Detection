from LaneLine import LaneLine
from drawing import *
import cvcolor
from perspective_transform import *
from ImageThresholding import *
from calibrate_camera import undistort_image
import imageutils
import Utils

class LaneDetector(object):
    def __init__(self, pipeline):
        self.left_lane_line = LaneLine()
        self.right_lane_line = LaneLine()
        self.scale = 4
        self.pipeline = pipeline
        self.frame_size = None
        # relative distance of left/right lane line from
        # left/right edge of bird's eye view
        self.dst_margin_rel = 11.0/32.0
#        self.dst_margin_rel = 0.25
        self.dst_margin_abs = None
        self.is_initialized = False
        self.last_lane_width = 3.7
        self.last_distance_from_center = None
        self.current_distance_from_center = None
        self.current_lane_width = None


    def process(self, frame):
        if not self.is_initialized:
            self.frame_size = np.array(frame.shape[0:2])
            self.input_frame_size = self.frame_size // self.scale
            h,w = self.input_frame_size
            self.dst_margin_abs = int(w * self.dst_margin_rel)

            # meters per pixel in y dimension
            self.ym_per_px =  48 / h

            # meters per pixel in x dimension
            self.xm_per_px = 3.7 / (w - 2.0 * self.dst_margin_abs)

            # anchor point for lane detection
            x_anchor_left = self.dst_margin_abs #+ w // 32
            x_anchor_right = w - self.dst_margin_abs# - w // 32


            self.left_lane_line.initialize(
                self.input_frame_size,
                x_anchor_left,
                self.xm_per_px, self.ym_per_px)

            self.right_lane_line.initialize(
                self.input_frame_size,
                x_anchor_right,
                self.xm_per_px, self.ym_per_px)

            self.is_initialized = True

        self.warped_frame, self.M_inv = perspective_transform(frame, self.dst_margin_rel)
        self.pipeline_input = imageutils.scale_img(self.warped_frame, 1.0/self.scale)
        self.detection_input = self.pipeline.process(self.pipeline_input)

        self.left_lane_line.fit_lane_line(self.detection_input)
        self.right_lane_line.fit_lane_line(self.detection_input)
        self.detection_input *= 255



    def annotate(self, frame):
        composite_img = np.zeros_like(frame, np.uint8)

        self.fill_lane_area(composite_img)

        if self.left_lane_line.has_good_fit():
            self.draw_lane_line(composite_img, self.left_lane_line)

        if self.right_lane_line.has_good_fit():
            self.draw_lane_line(composite_img, self.right_lane_line)

        transformed_composite_img = inv_perspective_transform(composite_img, self.M_inv)
        return cv2.addWeighted(frame, 1, transformed_composite_img, 0.3, 0)



    def annotate_lane_lines(self, img):
        self.left_lane_line.draw_lane_points(img)
        self.right_lane_line.draw_lane_points(img)
        self.left_lane_line.draw_histogram(img)
        self.right_lane_line.draw_histogram(img)
        img = self.left_lane_line.annotate_poly_fit(img)
        img = self.right_lane_line.annotate_poly_fit(img)
        return img


    def get_radii(self):
        R_left = self.left_lane_line.radius.value
        R_right = self.right_lane_line.radius.value
        return R_left, R_right


    def calc_distance_from_center(self):
    #def calc_position_and_lane_width(self):
        d_left = self.left_lane_line.calc_distance_from_center()
        d_right = self.right_lane_line.calc_distance_from_center()

        if d_left == None and d_right == None:
            d = self.last_distance_from_center
            w_lane = 3.7
        elif d_left == None:
            d = self.last_lane_width / 2 - d_right
            w_lane = 3.7
        elif d_right == None:
            d = self.last_lane_width / 2 + d_left
            w_lane = 3.7
        else:
            w_lane = d_right - d_left
            d = w_lane/2 - d_right
            self.last_lane_width = w_lane

        self.last_distance_from_center = d
        if d is None:
            d = 0.0
        return d,w_lane


    def draw_lane_line(self,img,lane_line):
        h,w = img.shape[0:2]
        pts = []
        coords = lane_line.interpolate_line_points(h//self.scale) * self.scale
        if coords is not None:
            thickness = max(1, int(16))
            cv2.polylines(img, [coords], isClosed=False, color=cvcolor.red, thickness=thickness)


    def fill_lane_area(self, img):
        h,w = img.shape[0:2]
        pts = []

        left_pts = self.left_lane_line.interpolate_line_points(h//self.scale)
        right_pts = self.right_lane_line.interpolate_line_points(h//self.scale)

        has_left_pts = left_pts is not None and len(left_pts)
        has_right_pts = right_pts is not None and len(right_pts)

        if has_left_pts and has_right_pts:
            coords = np.stack((left_pts, right_pts[::-1,:]), axis=0).astype(np.int32).reshape((-1,2))
            cv2.fillPoly(img, [coords * self.scale], cvcolor.green)


    def save_screenshots(self):
        dir_name = "output_images/" + Utils.date_file_name()
        Utils.mkdir(dir_name)

        img_names = (
            "input_frame",
            "undistorted_frame",
            "warped_frame",
            "pipeline_input",
            "detection_input",
            "annotated_frame",
            "warped_annotated_frame",
            "annotated_detection_input")

        for n in img_names:
            imageutils.save_img(eval("self."+n), n, path=dir_name)

        pipeline_dir_name = dir_name + '/pipeline'
        Utils.mkdir(pipeline_dir_name)
        for img,title in self.pipeline.intermediates:
            imageutils.save_img(img, title, pipeline_dir_name)

        return dir_name
