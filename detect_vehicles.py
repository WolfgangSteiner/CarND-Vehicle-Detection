from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import argparse
import cvcolor
from drawing import *
import Utils
from  imageutils import *
from cv2grid import CV2Grid
#import pygame
import pyglet
import sys
from vehicledetector import VehicleDetector


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
args = parser.parse_args()

t_array = args.t1.split(".")
args.t1 = int(t_array[0]) * frame_rate
if len(t_array) == 2:
    args.t1 += int(t_array[1])

args.video_file += "_video.mp4"

vdetector = VehicleDetector(save_false_positives=args.save_false_positives)
vdetector.scale=args.scale

def process_frame(frame, fps=None):
    global counter

    frame = rgb2bgr(frame)
    annotated_frame = vdetector.process(frame, counter)

    if args.render and not args.annotate:
        new_frame = annotated_frame
    else:
        grid = CV2Grid.with_img(out_frame,(4,5))
        grid.paste_img(annotated_frame, (0,0), scale=1.0)

        if args.annotate:
            grid.paste_img(vdetector.cropped_img, (0,4), scale=args.scale/4.0)
            grid.paste_img((vdetector.heatmap.map * 32).astype(np.uint8), (1,4), scale=args.scale/4.0)
            grid.paste_img((vdetector.heatmap.thresholded_map * 255.0).astype(np.uint8), (2,4), scale=args.scale/4.0)
            grid.paste_img((vdetector.heatmap.label_map * 32.0).astype(np.uint8), (3,4), scale=args.scale/4.0)

    new_frame = grid.canvas

    grid.text((0,0), "%02d.%02d"%(counter // frame_rate, counter % frame_rate), text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)

    if not args.render and fps is not None:
        grid.text((0.3,0), "%5.2ffps"%fps, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)

    if args.annotate:
        grid.text((0.0,0.25), "d_thres = %.2f"%vdetector.decision_threshold.value, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)
        grid.text((0.0,0.5), "h_thres = %.2f"%vdetector.heatmap.threshold.value, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)
        grid.text((0.0,0.75), "h_A     = %.2f"%vdetector.heatmap.A.value, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)

    if args.render:
        new_frame = bgr2rgb(new_frame)

    return new_frame

clip = VideoFileClip(args.video_file)
counter = -1
frame_skip = 1
start_frame = args.t1
key_wait = args.delay
height = 900 if args.annotate else 720
out_frame = new_img((1280,height))
window = pyglet.window.Window(width=1280, height=height)

@window.event
def on_draw():
    image = pyglet.image.ImageData(1280,height, 'BGR', out_frame[::-1,:,:].tostring())
    image.blit(0, 0)


if args.render:
     out_file_name = args.video_file.split(".")[0] + "_annotated.mp4"
     annotated_clip = clip.fl_image(process_frame)
     annotated_clip.write_videofile(out_file_name, fps=frame_rate, audio=False)
else:
    for frame in clip.iter_frames():
        if not args.render:
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
