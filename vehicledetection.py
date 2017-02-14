from point import Point
from rectangle import Rectangle

class VehicleDetection(object):
    def __init__(self, rect, frame_skip=0):
        self.current_position = rect.center()
        self.current_size = rect.size()
        self.frame_skip = frame_skip

        if frame_skip > 1:
            self.A = min(1.0, 0.25 * frame_skip)
        else:
            self.A = 0.25

        self.ticks_since_last_update = 0
        self.num_detections = 0
        self.age = 0
        self.is_real = False
        self.delta_pos = Point(0,0)
        self.delta_size = Point(0,0)


    def position(self):
        return self.current_rect.center()


    def tick(self):
        self.ticks_since_last_update += 1
        self.age += 1
        if self.num_detections > 2 and self.current_rect().area() > 32*32:
            self.is_real = True


    def interpolate(self):
        self.age += 1
        self.current_position += self.delta_pos
        self.current_size += self.delta_size

    def update(self, rect_list):
        if self.ticks_since_last_update == 0 or not rect_list:
            return rect_list

        sorted_list = sorted(rect_list, key=lambda r: self.current_rect().calc_overlap(r))

        r = sorted_list[-1]
        if self.current_rect().calc_overlap(r) > 0.0:
            if self.frame_skip == 1:
                self.current_position = self.A * r.center() + (1.0 - self.A) * self.current_position
                self.current_size = self.A * r.size() + (1.0 - self.A) * self.current_size
            else:
                new_position = r.center()
                new_delta_pos = (new_position - self.current_position) / self.frame_skip

                new_size = r.size()
                new_delta_size = (new_size - self.current_size) / self.frame_skip

                self.delta_pos = new_delta_pos * self.A + (1.0 - self.A) * self.delta_pos
                self.delta_size = new_delta_size * self.A + (1.0 - self.A) * self.delta_size

                self.current_position += self.delta_pos
                self.current_size += self.delta_size

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
        r = self.current_rect()
        is_size_ok = r.width() > 16 and r.height() > 16
        return (self.ticks_since_last_update < 25 and self.is_real and is_size_ok) or (self.ticks_since_last_update < 2 and not self.is_real)


    def current_area(self):
        return self.current_rect().area()
