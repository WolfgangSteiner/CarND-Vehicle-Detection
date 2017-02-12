import cv2
import numpy as np


#y1  = int(430)
##x11 = int(620)
#x12 = int(660)

y1  = 450
x11 = 595
x12 = 690

#y1  = 460
#x11 = 580
#x12 = 700

#y1  = 470
#x11 = 565
#x12 = 717

y2  = 720
x21 = 216
x22 = 1115


def perspective_transform(img,dst_margin_rel=11.0/32.0):
    h,w = img.shape[0:2]

    dst_width = w
    dst_height = h
    dst_margin = dst_width * dst_margin_rel

    src = np.array([[x11,y1], [x12,y1], [x22,y2], [x21,y2]], np.float32)
    dst = np.array([[dst_margin,0], [dst_width-dst_margin,0], [dst_width-dst_margin,dst_height], [dst_margin,dst_height]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (dst_width,dst_height), flags=cv2.INTER_LINEAR)
    return warped, M_inv


def inv_perspective_transform(img, M_inv):
    h,w = img.shape[0:2]
    return cv2.warpPerspective(img, M_inv, (w,h), flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    from Drawing import *
    from imageutils import *
    from Color import color
    from cv2grid import CV2Grid
    import glob

    c = CV2Grid(1280/2,720//4*3,(2,3))
    for idx,img_file in enumerate(glob.glob("test_images/straight_lines*")):
        img = load_img(img_file)
        thickness=2
        draw_line(img, (x11,y1), (x21,y2), color=color.red, thickness=thickness)
        draw_line(img, (x12,y1), (x22,y2), color=color.red, thickness=thickness)
        draw_line(img, (x21,y2), (x22,y2), color=color.red, thickness=thickness)
        draw_line(img, (x11,y1), (x12,y1), color=color.red, thickness=thickness)
        c.paste_img(img, (0,idx), scale=0.25)
        c.paste_img(perspective_transform(img)[0], (1,idx), scale=0.25)

    c.save("output_images/perspective_transform")
