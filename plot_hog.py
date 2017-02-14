from cv2grid import CV2Grid
import glob
import Utils
import cvcolor
from extract_features import visualize_hog
from imageutils import *

def my_hog(img, factor=1):
    ppc = 32
    return visualize_hog(img, ppc) * factor


def my_paste(img, pos, title, heading=False):
    title_style = "heading" if heading else "bottomleft"
    g.paste_img(img, pos, scale=1, title=title, title_style=title_style, x_anchor="left")


def plot_hog(file_name, col, title):
    v_img = load_img(file_name)
    v_img = scale_img(v_img, 2)
    y, u, v = split_yuv(v_img)
    g.text((col+1,0), title, horizontal_align="center")
    my_paste(v_img, (col + 0.5, 0.25), None, heading=True)
    my_paste(y, (col, 1.25), "Y")
    my_paste(u, (col, 2.25), "U")
    my_paste(v, (col, 3.25), "V")
    my_paste(my_hog(y), (col+1, 1.25), "HOG(Y)")
    my_paste(my_hog(u,4), (col+1, 2.25), "HOG(U)")
    my_paste(my_hog(v,8), (col+1, 3.25), "HOG(V)")


Utils.mkdir("fig")
vehicle_file_names = glob.glob("vehicles/GTI_MiddleClose/*.png")
vehicle_file_name = vehicle_file_names[18]
nonvehicle_file_names = glob.glob("non-vehicles/GTI/*.png")
nonvehicle_file_name = nonvehicle_file_names[99]

g = CV2Grid((512, 1024), (4,8), color=cvcolor.white)
plot_hog(vehicle_file_name, 0, "vehicle")
plot_hog(nonvehicle_file_name, 2, "non vehicle")

g.save("fig/features.png")


