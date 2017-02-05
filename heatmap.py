import numpy as np

class HeatMap(object):
    def __init__(self, size, A=0.1):
        self.map = np.zeros(size.astype(np.int), np.float32)
        self.new_map = np.zeros_like(self.map)
        self.thresholded_map = np.zeros_like(self.map)
        self.A = A
        self.B = 1.0 - self.A
        self.threshold = 4.0


    def add_detections(self, detections):
        for (r,confidence) in detections:
            x1,y1 = r.p1().astype(np.int)
            x2,y2 = r.p2().astype(np.int)
            self.new_map[y1:y2, x1:x2] += confidence


    def update_map(self):
        self.map = self.A * self.new_map + self.B * self.map
        self.new_map[:,:] = 0.0
        self.thresholded_map[::] = 0.0
        self.thresholded_map[self.map > self.threshold] = 1.0

