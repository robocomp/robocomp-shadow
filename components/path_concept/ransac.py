# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import sys
import time

import cv2
import numpy as np

"""
   Implementation for 2D Lane RANSAC
"""
def flood_fill(frame, tol_h, tol_s, tol_v):
    point = (310, 370)
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    connectivity = 4
    flags = connectivity | (255 << 8)
    flags |= cv2.FLOODFILL_FIXED_RANGE
    mask = np.zeros((frameHSV.shape[0]+2, frameHSV.shape[1]+2), np.uint8)
    retval, image, mask, rect = cv2.floodFill(frameHSV, mask, point, (128, 200, 150), (tol_h, tol_s, tol_v), (tol_h, tol_s, tol_v), flags)
    # if mask.size>0:
    #     cv2.imshow("flood", mask)
    #     cv2.waitKey(5)
    frame2 = cv2.cvtColor(frameHSV, cv2.COLOR_HSV2BGR)
    return frame2, mask

def sampler(segmented, y_offset, middle_x):
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    n_samples = 6
    for it in range(1000):
        n_points = len(segmented)
        id_samples = random.sample(range(0, n_points), n_samples)
        samples = segmented[id_samples]
        if max(samples, key=lambda x: x[1])[1] > y_offset:     # closeness start constraint
            continue
        # sort samples by x coordinate
        samples = sorted(samples, key=lambda x: x[0])
        # sort left and right sets by y coordinate
        left_x_samples = np.array(sorted(samples[:n_samples//2], key=lambda x: x[1]))
        right_x_samples = np.array(sorted(samples[n_samples//2:], key=lambda x: x[1]))
        dists = np.linalg.norm(np.subtract(left_x_samples, right_x_samples), axis=1)
        left_dists = np.linalg.norm(left_x_samples[1:] - left_x_samples[:-1], axis=1)
        right_dists = np.linalg.norm(right_x_samples[1:] - right_x_samples[:-1], axis=1)
        min_width = 200     # of the lowest part of the lane
        max_diff = 100      # between lateral segments of the lane
        if is_sorted(dists) and dists[-1] > min_width:     # perspective constraint
            if np.all(left_dists > max_diff) and np.all(right_dists > max_diff):   # tube constraint
                # return as a polygon counterclockwise
                return np.concatenate((left_x_samples, np.flip(right_x_samples, axis=0)), axis=0), True
    return [], False

def ransac(frame, segmented_edges, mask, prev_poly):
    middle_x = frame.shape[1]//2
    end_y = frame.shape[0]
    y_offset = int(end_y - (end_y/6))
    y_offset = int(end_y)
    alpha = 0.4
    best_inliers = 0
    first_time = True
    for it in range(1000):
        # initialize from previous solution
        if first_time and prev_poly.size > 0:
            #print("FROM ANT")
            poly = prev_poly
            first_time = False
        else:
            poly, ret = sampler(segmented_edges, y_offset, middle_x)
            if not ret:
                #print("CANNOT FIND VALID SAMPLE")
                break

        # count inliers
        #inliers = np.count_nonzero([cv2.pointPolygonTest(poly, tuple(x), False) == 1 for x in segmented])
        img_poly = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(img_poly, pts=[poly], color=255)
        result = cv2.bitwise_and(mask, img_poly)
        inliers = np.count_nonzero(result)

        # check results
        if inliers > best_inliers:
            best_inliers = inliers
            prev_poly = poly
            temp = frame.copy()
            cv2.fillConvexPoly(temp, poly, (0, 200, 0))
            for i in range(len(poly//2)):
                cv2.putText(temp, str(i+1), tuple(poly[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            for i in range(len(poly // 2), len(poly)):
                cv2.putText(temp, str(len(poly)-i+1), tuple(poly[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            frame_new = cv2.addWeighted(temp, alpha, frame, 1 - alpha, 0)
            cv2.imshow("Ransac", frame_new)
            cv2.waitKey(1)
            percent = int(best_inliers*100/np.count_nonzero(mask))

        print(inliers, best_inliers, it, percent, '%')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if percent > 70:
            print("SUCCESS")
            break

    return prev_poly

def main():
    #frame = cv2.imread('road.png')
    cap = cv2.VideoCapture('track6.mp4')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        sys.exit()

    winname = "Ransac"
    cv2.namedWindow(winname)

    # get segmentation color
    #color = frame[495, 350]
    color = (150, 32, 63)       # BGR for (128, 200, 150) HSV
    best_poly = np.empty(0)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (700, 500))

            frame2, mask = flood_fill(frame, 20, 20, 200)

            # cv2.imshow(winname, frame2)
            # cv2.waitKey(5)
            # get all segmented pixels
            #mask = cv2.inRange(frame2, color, color)
            #segmented = cv2.findNonZero(mask)
            # compute the edges of segmented image
            segmented_edges = cv2.findNonZero(cv2.Canny(mask, 100, 200))
            # get not-segmented pixels
            #mask = 1 - mask
            #not_segmented = cv2.findNonZero(mask)

            # RANSAC
            if mask.size > 6:
                best_poly = ransac(frame, segmented_edges.squeeze(), mask, best_poly)

    while cv2.waitKey(25) & 0xFF != ord('q'):
        pass

    # Closes all the frames
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


