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


frame_rate = 25
parser = argparse.ArgumentParser()
parser.add_argument('-1', action="store_const", dest="video_file", const="project")
parser.add_argument('-2', action="store_const", dest="video_file", const="challenge")
parser.add_argument('-3', action="store_const", dest="video_file", const="harder_challenge")
parser.add_argument('-d', action="store_const", dest="delay", const=500, default=1)
parser.add_argument('-dd', action="store_const", dest="delay", const=1000, default=1)
parser.add_argument('-t', action="store", dest="t1", default="0", type=str)
parser.add_argument('-s', action="store", dest="scale", default=4, type=int)
parser.add_argument('--render', action="store_true", dest="render")
parser.add_argument('--annotate', action="store_true", dest="annotate")
args = parser.parse_args()

t_array = args.t1.split(".")
args.t1 = int(t_array[0]) * frame_rate
if len(t_array) == 2:
    args.t1 += int(t_array[1])

args.video_file += "_video.mp4"

def process_frame(frame, fps=None):
    global counter

#    if not args.render:
#        pipeline.poll()

#    detector.process(frame)
#    detector.annotate(frame)

    frame = rgb2bgr(frame)

    if args.render and not args.annotate:
        new_frame = detector.annotated_frame
    else:
        pass
        grid = CV2Grid.with_img(out_frame,(6,6))
        grid.paste_img(frame, (0,0), scale=0.5)

    new_frame = grid.canvas

    if fps is not None:
        grid.text((0,0), "%5.2ffps"%fps, text_color=cvcolor.white, horizontal_align="left", vertical_align="top", scale=1.0)


    if True or args.render:
        new_frame = bgr2rgb(new_frame)

    return new_frame


#clip = VideoFileClip(args.video_file)
cap = cv2.VideoCapture(args.video_file)
counter = 0
frame_skip = 1
start_frame = args.t1
key_wait = args.delay

out_frame = new_img((1280,720))

#cv2.startWindowThread()
#cv2.namedWindow("test")
#screen = pygame.display.set_mode((1280,720))


frames = []

#clock = pygame.time.Clock()
#fps = 0
#frame_rate=cap.get(cv2.CAP_PROP_FPS)


window = pyglet.window.Window(width=1280, height=720)

@window.event
def on_draw():
    image = pyglet.image.ImageData(1280,720, 'BGR', out_frame[::-1,:,:].tostring())
    image.blit(0, 0)


while True:
    pyglet.clock.tick()

    has_frame, frame = cap.read()
    if not has_frame:
        pyglet.app.exit()

    out_frame = process_frame(frame,fps=pyglet.clock.get_fps())

    window.switch_to()
    window.dispatch_events()
    window.dispatch_event('on_draw')
    window.flip()

# if args.render:
#     out_file_name = args.video_file.split(".")[0] + "_annotated.mp4"
#     annotated_clip = clip.fl_image(process_frame)
#     annotated_clip.write_videofile(out_file_name, fps=frame_rate, audio=False)
# else:
#     while True:
#         has_frame, frame = cap.read()
#         if not has_frame:
#             break
#
#
#
#
#     #for frame in clip.iter_frames():
#         # if (counter % frame_skip) or (counter < start_frame):
#         #     counter += 1
#         #     continue
#
#         clock.tick()
#         new_frame = process_frame(frame)
#
#         pimage = pygame.image.frombuffer(new_frame.tostring(), (1280,720), "RGB")
#         screen.blit(pimage, (0,0))
#         pygame.display.flip()
#
#         print(clock.get_fps())
#
#         #cv2.imshow("test",new_frame)
#         #key = cv2.waitKey(1)
#         # key = None
#         # if key == ord('.'):
#         #     dir_name = detector.save_screenshots()
#         #     save_img(new_frame, "output_frame", dir_name)
#         #     print("Screenshots saved in %s." % dir_name)
#         # elif key == ord('+'):
#         #     key_wait = max(10, key_wait // 2)
#         # elif key == ord('-'):
#         #     key_wait = min(2000, key_wait * 2)
#         # elif key == ord(' '):
#         #     print("PAUSE...")
#         #     while True:
#         #         key = cv2.waitKey(10)
#         #         if key == ord(' '):
#         #             break
