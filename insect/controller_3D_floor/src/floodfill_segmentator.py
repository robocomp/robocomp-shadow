import cv2
import numpy as np
class Floodfill_Segmentator():
    def __init__(self):
        pass
    def process(self, frame):
        height, width = frame.shape[:2]
        point = (width//2, height-5)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        connectivity = 4
        flags = connectivity | (255 << 8)
        flags |= cv2.FLOODFILL_FIXED_RANGE
        mask = np.zeros((frame_hsv.shape[0] + 2, frame_hsv.shape[1] + 2), np.uint8)
        retval, image, mask, rect = cv2.floodFill(frame_hsv, mask, point, (128, 200, 150), (20, 20, 200),
                                                   (20, 20, 200), flags)
        mask = mask[:-2, :-2]
        segmented = np.zeros((frame_hsv.shape[0], frame_hsv.shape[1]), np.uint8)
        segmented[mask == 255] = 3
        return mask, segmented, []
