import numpy as np
from collections import deque

class OrientationFilter:
    def __init__(self, orientation, timestamp, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)  # Each entry is (timestamp, angle)

    def add_measurement(self, angle_rad, timestamp):
        self.buffer.append((timestamp, angle_rad))

    def get_filtered_orientation(self):
        if not self.buffer:
            return None

        # Extract angles
        angles = np.array([angle for _, angle in self.buffer])
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        filtered_angle = np.arctan2(sin_sum, cos_sum)

        # Return most recent timestamp with filtered angle
        latest_timestamp = self.buffer[-1][0]
        return filtered_angle
