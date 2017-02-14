from moviepy.editor import VideoFileClip
import argparse
from drawing import *
from  imageutils import *
from cv2grid import CV2Grid
import pyglet
from vehicledetector import VehicleDetector
from LaneDetector import LaneDetector
from FilterPipeline import HSVPipeline,YUVPipeline
from calibrate_camera import undistort_image


frame_rate = 25
parser = argparse.ArgumentParser()
parser.add_argument('-1', action="store_const", dest="video_file", const="project", default="project")
parser.add_argument('-2', action="store_const", dest="video_file", const="challenge")
parser.add_argument('-3', action="store_const", dest="video_file", const="harder_challenge")
parser.add_argument('-d', action="store_const", dest="delay", const=500, default=1)
parser.add_argument('-dd', action="store_const", dest="delay", const=1000, default=1)
parser.add_argument('-t', action="store", dest="t1", default="0", type=str)
parser.add_argument('-s', action="store", dest="scale", default=4, type=int)
parser.add_argument('--render', action="store_true", dest="render")
parser.add_argument('--annotate', action="store_true", dest="annotate")
parser.add_argument('--save-false-positives', action="store_true", dest="save_false_positives")
parser.add_argument("--no-lane-lines", action="store_false", dest="lane_lines")
parser.add_argument("--no-multires", action="store_false", dest="use_multires")
parser.add_argument("--hires", action="store_true", dest="hires")
parser.add_argument("--frame-skip", action="store", type=int, dest="frame_skip", default=0)
args = parser.parse_args()

t_array = args.t1.split(".")
args.t1 = int(t_array[0]) * frame_rate
if len(t_array) == 2:
    args.t1 += int(t_array[1])

args.video_file += "_video.mp4"

vdetector = VehicleDetector(
    save_false_positives=args.save_false_positives,
    use_multires_classifiers=args.use_multires,
    use_hires_classifier=args.hires,
    frame_skip=args.frame_skip)

vdetector.scale=args.scale
vdetector.annotate = args.annotate

if args.lane_lines:
    pipeline = YUVPipeline()
    lane_detector = LaneDetector(pipeline)

def process_frame(input_frame, fps=None):
    global counter

    input_frame = rgb2bgr(input_frame)
    undistorted_frame = undistort_image(input_frame)
    annotated_frame = vdetector.process(undistorted_frame)

    if args.lane_lines:
        lane_detector.process(undistorted_frame)
        annotated_frame = lane_detector.annotate(annotated_frame)

    if args.render or not args.annotate:
        new_frame = annotated_frame
    else:
        grid = CV2Grid.with_img(out_frame,(3,4))
        grid.paste_img(annotated_frame, (0,0), scale=1.0)

        if args.annotate:
            #grid.paste_img(vdetector.cropped_frame, (0, 4), scale=args.scale / 2.0)
            grid.paste_img((vdetector.annotated_heatmap), (2,0), scale=args.scale/2.0)
            grid.paste_img((vdetector.annotated_thresholded_heatmap), (2,1), scale=args.scale/2.0)
            grid.paste_img((vdetector.heatmap.label_map * 32.0).astype(np.uint8), (2,2), scale=args.scale/2.0)
        new_frame = grid.canvas

    text_grid = CV2Grid.with_img(new_frame, (10,6))
    text_grid.text((0,0), "%02d.%02d"%(vdetector.frame_count // frame_rate, vdetector.frame_count % frame_rate), text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)

    if not args.render and fps is not None:
        text_grid.text((0.75,0), "%5.2ffps"%fps, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)

    if False and args.annotate:
        grid.text((0.0,0.25), "d_thres = %.2f"%vdetector.decision_threshold.value, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)
        grid.text((0.0,0.5), "h_thres = %.2f"%vdetector.heatmap.threshold.value, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)
        grid.text((0.0,0.75), "h_A     = %.2f"%vdetector.heatmap.A.value, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)

    if args.render:
        new_frame = bgr2rgb(new_frame)

    if args.annotate and counter==300:
        save_img(input_frame, "input_frame", "fig")
        save_img(undistorted_frame, "undistorted_frame", "fig")
        save_img(vdetector.cropped_frame, "cropped_frame", "fig")
        save_img(vdetector.sliding_windows_frame, "sliding_windows", "fig")
        save_img(vdetector.detections_frame, "sliding_windows_detections", "fig")
        save_img(vdetector.annotated_heatmap, "heatmap", "fig")
        save_img(vdetector.annotated_thresholded_heatmap, "thresholded_heatmap", "fig")
        save_img(vdetector.annotated_detected_cars, "detected_car", "fig")

    return new_frame


clip = VideoFileClip(args.video_file)
counter = -1
frame_skip = 1
start_frame = args.t1
key_wait = args.delay
width,height = (1920,720) if args.annotate else (1280,720)
out_frame = new_img((width,height))
window = pyglet.window.Window(width=width, height=height)

@window.event
def on_draw():
    image = pyglet.image.ImageData(width,height, 'BGR', out_frame[::-1,:,:].tostring())
    image.blit(0, 0)

if args.render:
     out_file_name = args.video_file.split(".")[0] + "_annotated.mp4"
     annotated_clip = clip.fl_image(process_frame)
     annotated_clip.write_videofile(out_file_name, fps=frame_rate, audio=False)
else:
    for frame in clip.iter_frames():
        counter += 1
        if counter < args.t1:
            continue
        pyglet.clock.tick()
        out_frame = process_frame(frame,fps=pyglet.clock.get_fps())
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()

    if not args.render:
        pyglet.app.exit()
