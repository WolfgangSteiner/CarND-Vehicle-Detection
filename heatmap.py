import numpy as np
import scipy.ndimage.measurements
from rectangle import Rectangle
from midicontrol import MidiControl

class HeatMap(object):
    def __init__(self, size, midimanager):
        self.map = np.zeros(size.astype(np.int), np.float32)
        self.new_map = np.zeros_like(self.map)
        self.thresholded_map = np.zeros_like(self.map)
        self.A = MidiControl(midimanager,"heatmap_A", 82, 0.1, 0.0, 1.0)
        self.threshold = MidiControl(midimanager,"heatmap_threshold", 81, 1.25, 0.0, 8.0)


    def add_detections(self, detections):
        for (r,confidence) in detections:
            x1,y1 = r.p1().astype(np.int)
            x2,y2 = r.p2().astype(np.int)
            self.new_map[y1:y2, x1:x2] += 1.0


    def update_map(self):
        self.map = self.A.value * self.new_map + (1.0 - self.A.value) * self.map
        self.new_map[:,:] = 0.0
        self.thresholded_map[::] = 0.0
        self.thresholded_map[self.map > self.threshold.value] = 1.0


    def get_bboxes(self):
        self.label_map, num_detections = scipy.ndimage.measurements.label(self.thresholded_map)
        result = []
        for i in range(1, num_detections+1):
            nonzero = (self.label_map == i).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            result.append(Rectangle(np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)))

        return result


