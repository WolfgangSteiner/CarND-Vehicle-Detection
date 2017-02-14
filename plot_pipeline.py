from cv2grid import CV2Grid
import cvcolor
import numpy as np

def my_paste(filename, scale, pos, title=None, title_style="topcenter"):
    c.paste_img_file(
        "fig/%(filename)s.png" % locals(),
        pos, scale=scale,
        x_anchor="center",
        title=title,
        title_style=title_style)


def pipeline_step(filename, title):
    global y
    c.arrow([(x, y), (x, y + 0.5)])
    c.text_frame((x, y + 0.5), (-1, 1), title, x_anchor="center",
                 y_anchor="top", scale=1.0)
    c.arrow([(x, y + 1.5), (x, y + 2.0)])
    my_paste(filename, 1.0, (x, y+2))
    y += 4.75


(w,h) = np.array((1440,2250))
c = CV2Grid((w,h),color=cvcolor.white, grid=(32 * 3, 8 * 8-14))

x = 16 * 3
y = 0.25
my_paste("input_frame", 0.5, (x,y))
y+=8
c.arrow([(x,y),(x,y+0.5)])
c.text_frame((x,y+0.5), (-1,1), "camera distortion correction", x_anchor="center", y_anchor="top", scale=1.0)
c.arrow([(x,y+1.5),(x,y+2.0)])

y+=2
my_paste("undistorted_frame", 0.5, (x,y))

y+=8
pipeline_step("cropped_frame", "scaling and cropping")
pipeline_step("sliding_windows", "sliding windows")
pipeline_step("sliding_windows_detections", "feature extraction and classification")
pipeline_step("heatmap", "interpolated heat-map")
pipeline_step("thresholded_heatmap", "thresholding")
pipeline_step("detected_car", "bounding box extraction and update")


c.save("fig/pipeline.png")