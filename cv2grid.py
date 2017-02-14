import cv2
import numpy as np

import cvcolor
import drawing
import imageutils
from rectangle import Rectangle
from point import Point


class CV2Grid(object):
    def __init__(self, size, grid=(8, 8), img=None, color=cvcolor.black):
        if img is None:
            self.canvas = imageutils.new_img(size, color=color)
        else:
            assert np.all(size == imageutils.img_size(img))
            self.canvas = img

        self.grid = np.array(grid, np.int)
        self.grid_size = (self.size() / self.grid).astype(np.int)

        self.tr = drawing.TextRenderer(self.canvas)

    @classmethod
    def with_img(cls, img, grid, color=cvcolor.black):
        return cls(imageutils.img_size(img), grid=grid, color=color, img=img)

    def size(self):
        return imageutils.img_size(self.canvas)

    def width(self):
        return self.size()[0]

    def height(self):
        return self.size()[1]

    def abs_pos(self, grid_pos):
        return (np.array(grid_pos) * self.grid_size).astype(np.int)

    def paste_img(self, img, grid_pos, scale=1.0, title=None, title_style="heading", x_anchor="left"):
        if scale != 1.0:
            img = imageutils.scale_img(img, scale)

        w, h = imageutils.img_size(img)

        offset = np.zeros(2, np.int)
        if x_anchor == "center":
            offset[0] -= w // 2

        x1, y1 = self.abs_pos(grid_pos) + offset
        x2, y2 = self.abs_pos(grid_pos) + imageutils.img_size(img) + offset

        imageutils.paste_img(self.canvas, img, (y1, x1))

        if title is not None:
            if title_style == "heading":
                pos = np.array(((x1 + x2) // 2, y1), np.int)
                horizontal_align = "center"
                vertical_align = "bottom"
                text_color = cvcolor.black
                scale = 1.0
            elif title_style == "topleft":
                pos = np.array((x1, y1))
                horizontal_align = "left"
                vertical_align = "top"
                text_color = cvcolor.white
                scale = 1.0
            elif title_style == "topcenter":
                pos = np.array(((x1 + x2) // 2, y1), np.int)
                horizontal_align = "center"
                vertical_align = "top"
                text_color = cvcolor.white
                scale = 1.0
            elif title_style == "bottomleft":
                pos = np.array((x1, y2))
                horizontal_align = "left"
                vertical_align = "bottom"
                text_color = cvcolor.white
                scale = 1.0
            elif title_style == "bottomcenter":
                pos = np.array(((x1 + x2) // 2, y2), np.int)
                horizontal_align = "center"
                vertical_align = "bottom"
                text_color = cvcolor.white
                scale = 1.0


            if not title_style == "heading":
                (p1, p2) = self.tr.calc_bounding_box(
                    title,
                    pos,
                    horizontal_align=horizontal_align,
                    vertical_align=vertical_align,
                    scale=scale,
                    margin=[5, 5],
                    font=cv2.FONT_HERSHEY_PLAIN)

                self.canvas[p1[1]:p2[1], p1[0]:p2[0], :] //= 2

            # if not title_style.startswith("heading"):
            #     self.tr.text_at(
            #         title,
            #         pos + np.ones(2,np.int) * 2,
            #         horizontal_align=horizontal_align,
            #         vertical_align=vertical_align,
            #         color=cvcolor.gray20,
            #         scale=scale)




            # self.tr.text_at(
            #     title,
            #     pos - np.ones(2,np.int) * 2,
            #     horizontal_align=horizontal_align,
            #     vertical_align=vertical_align,
            #     color=cvcolor.gray40,
            #     scale=scale)

            (p1, p2) = self.tr.text_at(
                title,
                pos,
                horizontal_align=horizontal_align,
                vertical_align=vertical_align,
                color=text_color,
                scale=scale,
                margin=[5, 5],
                font=cv2.FONT_HERSHEY_PLAIN)

    def paste_img_file(self, file_name, pos, scale=1.0,title=None,title_style="heading", x_anchor="left"):
        img = imageutils.load_img(file_name)
        self.paste_img(img, pos, scale,title=title, title_style=title_style, x_anchor=x_anchor)


    def arrow(self, pts, color=cvcolor.black, start_margin=0, end_margin=0):
        arrow_length = 10
        arrow_width = 5

        abs_pts = []
        for idx, p in enumerate(pts):
            p = self.abs_pos(p)
            if idx == 0:
                p1 = p
                p2 = self.abs_pos(pts[1])
                v = (p2 - p1) / np.linalg.norm(p2 - p1)
                p = p1 + v * start_margin
            elif idx == len(pts) - 1:
                p1 = self.abs_pos(pts[-2])
                p2 = p
                v = (p2 - p1) / np.linalg.norm(p2 - p1)
                p = p2 - v * end_margin

            abs_pts.append(p)

        p1 = abs_pts[-2]
        p2 = abs_pts[-1]
        v = (p2 - p1) / np.linalg.norm(p2 - p1)
        nv = np.array((v[1], -v[0]))
        p3 = (p2 - v * (arrow_length))

        head_coords = []
        head_coords.append(p3 + nv * arrow_width)
        head_coords.append(p3 - nv * arrow_width)
        head_coords.append(p2)

        cv2.polylines(self.canvas, [np.array(abs_pts, np.int)], isClosed=False, color=color, thickness=2)
        cv2.fillPoly(self.canvas, [np.array(head_coords).astype(int)], color, lineType=cv2.LINE_AA)

    def line(self, pts, color=cvcolor.black, start_margin=0, end_margin=0):
        abs_pts = []
        for idx, p in enumerate(pts):
            p = self.abs_pos(p)
            if idx == 0:
                p1 = p
                p2 = self.abs_pos(pts[1])
                v = (p2 - p1) / np.linalg.norm(p2 - p1)
                p = p1 + v * start_margin
            elif idx == len(pts) - 1:
                p1 = self.abs_pos(pts[-2])
                p2 = p
                v = (p2 - p1) / np.linalg.norm(p2 - p1)
                p = p2 - v * end_margin

            abs_pts.append(p)

        cv2.polylines(self.canvas, [np.array(abs_pts, np.int)], isClosed=False, color=color, thickness=2)

    def save(self, filename):
        imageutils.save_img(self.canvas, filename)

    def show(self):
        imageutils.show_img(self.canvas)

    def set_text_color(self, color):
        self.tr.color = color

    def draw_grid(self, color=cvcolor.gray80):
        for x in np.arange(0, self.grid[0] + 1) * self.grid_size[0]:
            cv2.line(self.canvas, (x, 0), (x, self.height()), color=color)

        for y in np.arange(0, self.grid[1] + 1) * self.grid_size[1]:
            cv2.line(self.canvas, (0, y), (self.width(), y), color=color)

    def draw_frame(self, pos, size, fill_color=cvcolor.white, border_color=cvcolor.black, x_anchor="left",
                   y_anchor="top"):
        offset = np.zeros(2, np.int)
        abs_size = self.abs_pos(size)
        offset = np.zeros(2, np.int)

        if x_anchor == "center":
            offset[0] -= abs_size[0] // 2

        if y_anchor == "center":
            offset[1] -= abs_size[1] // 2

        p1 = self.abs_pos(pos) + offset

        drawing.bordered_rectangle(self.canvas, Rectangle(pos=p1, size=abs_size), fill_color, border_color, thickness=2)
        return p1, abs_size

    def text_frame(self, pos, size, text, text_color=cvcolor.black, fill_color=cvcolor.white,
                   border_color=cvcolor.black, x_anchor="left", y_anchor="top", scale=1.0, margin=[20, 20]):
        size = np.array(size)
        if size[0] == -1:
            (tw, th), baseline = self.tr.calc_text_size(text, scale=scale)
            size[0] = float(tw + 2 * margin[0]) / self.grid_size[0]

        abs_pos, abs_size = self.draw_frame(pos, size, fill_color, border_color, x_anchor, y_anchor)
        center = (abs_pos + abs_size // 2).astype(np.int)
        self.tr.text_at(text, center, horizontal_align="center", vertical_align="center", color=text_color, scale=scale)

    def text(self, pos, text, text_color=cvcolor.black, horizontal_align="left", vertical_align="top", scale=1.0):
        abs_pos = self.abs_pos(pos)
        self.tr.text_at(text, abs_pos, horizontal_align=horizontal_align, vertical_align=vertical_align,
                        color=text_color, scale=scale)
