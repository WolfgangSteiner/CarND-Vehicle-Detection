from point import Point
import numpy as np

class Rectangle(object):
    def __init__(self, *args, **kwargs):
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

        if len(args) == 4:
            self.x1 = min(args[0], args[2])
            self.y1 = min(args[1], args[3])
            self.x2 = max(args[0], args[2])
            self.y2 = max(args[1], args[3])
            return
        elif len(args) == 2:
            self.set_p1(args[0])
            self.set_p2(args[1])
        elif 'pos' in kwargs:
            if not 'size' in kwargs:
                raise ValueError
            self.set_p1(kwargs['pos'])
            size = kwargs['size']
            if type(size) in (int,float):
                size = (size,size)
            self.x2 = self.x1 + size[0]
            self.y2 = self.y1 + size[1]
            return
        elif 'center' in kwargs:
            if not 'size' in kwargs:
                raise ValueError
            center = kwargs['center']
            size = kwargs['size']
            if type(size) in (int,float):
                size = (size,size)
            self.x1 = center[0] - size[0] / 2
            self.x2 = center[0] + size[0] / 2
            self.y1 = center[1] - size[1] / 2
            self.y2 = center[1] + size[1] / 2
            return

        if 'x1' in kwargs:
            self.x1 = kwargs['x1']
        if 'x2' in kwargs:
            self.x2 = kwargs['x2']
        if 'y1' in kwargs:
            self.y1 = kwargs['y1']
        if 'y2' in kwargs:
            self.y2 = kwargs['y2']
        if 'x' in kwargs:
            self.x1 = kwarg['x']
        if 'y' in kwargs:
            self.y1 = kwargs['y']
        if 'size' in kwargs:
            size = kwargs['size']
            if type(size) in (int,float):
                size = (size,size)
            if self.x1 is not None:
                self.x2 = self.x1 + size[0]
            elif self.x2 is not None:
                self.x1 = self.x2 - size[0]

            if self.y1 is not None:
                self.y2 = self.y1 + size[1]
            elif self.y2 is not None:
                self.y1 = self.y2 - size[1]

        if any([c is None for c in (self.x1,self.x2,self.y1,self.y2)]):
            raise ValueError


    def set_p1(self, p):
        self.x1 = p[0]
        self.y1 = p[1]


    def set_p2(self, p):
        self.x2 = p[0]
        self.y2 = p[1]


    def p1(self):
        return Point(self.x1, self.y1)


    def p2(self):
        return Point(self.x2, self.y2)


    def unscale(self, scale):
        return Rectangle(self.x1 / scale.x, self.y1 / scale.y, self.x2 / scale.x, self.y2 / scale.y)


    def scale(self, scale):
        return Rectangle(self.x1 * scale.x, self.y1 * scale.y, self.x2 * scale.x, self.y2 * scale.y)


    def as_array(self):
        return [self.x1, self.y1, self.x2, self.y2]


    def intersects_horizontally(self, other_rect):
        return not self.x2 < other_rect.x1 and not self.x1 > other_rect.x2


    def intersects_vertically(self, other_rect):
        return not self.y2 < other_rect.y1 and not self.y1 > other_rect.y2


    def intersects(self, other_rect):
        return self.intersects_horizontally(other_rect) and self.intersects_vertically(other_rect)


    def calc_overlap(self, other_rect):
        if not self.intersects(other_rect) or self.area() == 0.0:
            return 0.0
        else:
            return float(self.intersect(other_rect).area()) / self.area()


    def calc_vertical_overlap(self, r):
        if not self.intersects_vertically(r) or self.height() == 0:
            return 0.0
        else:
            ri = self.intersect(r)
            return float(ri.height()) / self.height()


    def contains(self, r):
        return self.x1 <= r.x1 and self.x2 >= r.x2 and self.y1 <= r.y1 and self.y2 >= r.y2


    def contains_point(self, p):
        return self.x1 <= p.x and self.x2 >= p.x and self.y1 <= p.y and self.y2 >= p.y


    def contains_vertically(self, r):
        return self.y1 <= r.y1 and self.y2 >= r.y2


    def shrink(self, point):
        return Rectangle(self.x1 + point.x, self.y1 + point.y, self.x2 - point.x, self.y2 - point.y)


    def expand(self, *args):
        if len(args)==2:
            return Rectangle(self.x1 - args[0], self.y1 - args[1], self.x2 + args[0], self.y2 + args[1])
        elif len(args)==4:
            return Rectangle(self.x1 - args[0], self.y1 - args[1], self.x2 + args[2], self.y2 + args[3])
        else:
            e = args[0]
            if type(e) == Point:
                return Rectangle(self.x1 - e.x, self.y1 - e.y, self.x2 + e.x, self.y2 + e.y)
            elif type(e) == int:
                return Rectangle(self.x1 - e, self.y1 - e, self.x2 + e, self.y2 + e)


    def shrink_with_factor(self, point):
        return Rectangle.from_center_and_size(self.center(), self.size().scale(point))


    def union_with(self, other_rect):
        self.x1 = min(self.x1, other_rect.x1)
        self.x2 = max(self.x2, other_rect.x2)
        self.y1 = min(self.y1, other_rect.y1)
        self.y2 = max(self.y2, other_rect.y2)


    def intersect_with(self, other_rect):
        if self.intersects(other_rect):
            self.x1 = max(self.x1, other_rect.x1)
            self.x2 = min(self.x2, other_rect.x2)
            self.y1 = max(self.y1, other_rect.y1)
            self.y2 = min(self.y2, other_rect.y2)
        else:
            self.x1 = self.x2 = self.y1 = self.y2 = 0


    def union(self, other_rect):
        r = Rectangle(self.x1,self.y1,self.x2,self.y2)
        r.union_with(other_rect)
        return r


    def intersect(self, other_rect):
        r = Rectangle(self.x1,self.y1,self.x2,self.y2)
        r.intersect_with(other_rect)
        return r


    def size(self):
        return self.p2() - self.p1()


    def width(self):
        return self.size().x


    def height(self):
        return self.size().y


    def aspect_ratio(self):
        if self.height() == 0:
            return 0
        return self.width() / self.height()


    def center(self):
        return 0.5 * (self.p1() + self.p2())


    def area(self):
        s = self.size()
        return s.x * s.y


    def translate(self, p):
        if not type(p) == Point:
            p = Point(p[0],p[1])
        return self + p


    def translate_to_fit_into_rect(self, r):
        w,h = self.size()
        if self.x1 < r.x1:
            x1,x2 = r.x1, r.x1 + w
        elif self.x2 > r.x2:
            x1,x2 = r.x2 - w, r.x2
        else:
            x1,x2 = self.x1,self.x2

        if self.y1 < r.y1:
            y1,y2 = r.y1, r.y1 + h
        elif self.y2 > r.y2:
            y1,y2 = r.y2 - h, r.y2
        else:
            y1,y2 = self.y1,self.y2

        return Rectangle(x1,y1,x2,y2)


    def mirror_x(self, x=0):
        return Rectangle((x - self.x2) + x, self.y1, (x - self.x1) + x, self.y2)


    def mirror_x(self, x=0):
        return Rectangle((x - self.x2) + x, self.y1, (x - self.x1) + x, self.y2)


    def mirror_y(self, y=0):
        return Rectangle(self.x1, (y - self.y2) + y, self.x2, (y - self.y1) + y)


    def __add__(self, a):
        if type(a) == Point:
            return Rectangle(self.x1+a.x, self.y1+a.y, self.x2+a.x, self.y2+a.y)
        else:
            raise ValueError


    def __imul__(self, a):
        self.x1 *= a
        self.x2 *= a
        self.y1 *= a
        self.y2 *= a
        return self

    def __mul__(self, factor):
        return Rectangle(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)

    __rmul__ = __mul__


    def __repr__(self):
        return "(%.2f, %.2f, %.2f, %.2f)" % (self.x1,self.y1,self.x2,self.y2)


    def __floordiv__(self, factor):
        return Rectangle(self.x1 // factor, self.y1 // factor, self.x2 // factor,self.y2 // factor)


    def __realdiv__(self, factor):
        return Rectangle(self.x1 / factor, self.y1 / factor, self.x2 / factor,self.y2 / factor)


    def is_vertical_edge_intersected_by_line(self, x, line_p1, line_p2):
        delta = (line_p2 - line_p1).astype(np.float32)
        vec = delta / np.linalg.norm(delta)
        if vec[0] == 0.0:
            return False
        t = (x - line_p1.x) / vec[0]
        y = t * vec[1]
        return y >= self.y1 and y <= self.y2


    def is_horizontal_edge_intersected_by_line(self, y, line_p1, line_p2):
        delta = (line_p2 - line_p1).astype(np.float32)
        vec = delta / np.linalg.norm(delta)
        if vec[1] == 0.0:
            return False
        t = (y - line_p1.y) / vec[1]
        x = t * vec[0]
        return x >= self.x1 and x <= self.x2


    def is_intersected_by_line(self, line_p1, line_p2):
        for x in self.x1,self.x2:
            if self.is_vertical_edge_intersected_by_line(x, line_p1, line_p2):
                return True

        for y in self.y1,self.y2:
            if self.is_horizontal_edge_intersected_by_line(y, line_p1, line_p2):
                return True

        return False
