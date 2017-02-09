from point import Point
from rectangle import Rectangle

class CarDetection(object):
    def __init__(self, rect):
        self.current_position = rect.center()
        self.current_size = rect.size()
        self.A = 0.25
        self.ticks_since_last_update = 0
        self.age = 0


    def position(self):
        return self.current_rect.center()


    def tick(self):
        self.ticks_since_last_update += 1
        self.age += 1


    def update(self, rect_list):
        if self.ticks_since_last_update == 0 or not rect_list:
            return rect_list

        sorted_list = sorted(rect_list, key=lambda r: self.current_rect().calc_overlap(r))

        r = sorted_list[-1]
        if self.current_rect().calc_overlap(r) > 0.0:
            self.current_position = self.A * r.center() + (1.0 - self.A) * self.current_position
            self.current_size = self.A * r.size() + (1.0 - self.A) * self.current_size
            self.ticks_since_last_update = 0
            return sorted_list[0:-1]
        else:
            return rect_list


    def current_rect(self):
        return Rectangle.from_center_and_size(self.current_position, self.current_size)


    def current_rect_of_influence(self):
        return self.current_rect().expand(0)


    def is_rect_candidate(self, rect):
        return rect.calc_overlap(self.current_rect_of_influence()) > 0.0


    def is_alive(self):
        return self.ticks_since_last_update < 5


    def current_area(self):
        return self.current_rect().area()
