from point import Point
from rectangle import Rectangle

class CarDetection(object):
    def __init__(self, rect):
        self.current_position = rect.center()
        self.current_size = rect.size()
        self.A = 0.125
        self.ticks_since_last_update = 0
        self.num_detections = 0
        self.age = 0
        self.is_real = False


    def position(self):
        return self.current_rect.center()


    def tick(self):
        self.ticks_since_last_update += 1
        self.age += 1
        if self.num_detections > 5:
            self.is_real = True


    def update(self, rect_list):
        if self.ticks_since_last_update == 0 or not rect_list:
            return rect_list

        sorted_list = sorted(rect_list, key=lambda r: self.current_rect().calc_overlap(r))

        r = sorted_list[-1]
        if self.current_rect().calc_overlap(r) > 0.0:
            self.current_position = self.A * r.center() + (1.0 - self.A) * self.current_position
            self.current_size = self.A * r.size() + (1.0 - self.A) * self.current_size
            self.ticks_since_last_update = 0
            self.num_detections += 1
            return sorted_list[0:-1]
        else:
            return rect_list


    def current_rect(self):
        return Rectangle(center=self.current_position, size=self.current_size)


    def current_rect_of_influence(self):
        return self.current_rect().expand(0)


    def is_rect_candidate(self, rect):
        return rect.calc_overlap(self.current_rect_of_influence()) > 0.0


    def is_alive(self):
        return (self.ticks_since_last_update < 25 and self.is_real) or (self.ticks_since_last_update < 5 and not self.is_real)


    def current_area(self):
        return self.current_rect().area()
