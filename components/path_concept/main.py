# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import time
from pylsd import lsd   # https://github.com/AndranikSargsyan/pylsd-nova

tol_h = 20
tol_s = 20
tol_v = 250
point = (350, 250)
cap = cv2.VideoCapture('track6.mp4')
video_pos = 0

def mouse_clic(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print(point)

def flood_fill(frame, tol_h, tol_s, tol_v):
    global point
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    connectivity = 4
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(frameHSV, None, point, (128, 200, 150), (tol_h, tol_s, tol_v), (tol_h, tol_s, tol_v), flags)
    frame2 = cv2.cvtColor(frameHSV, cv2.COLOR_HSV2BGR)
    cv2.circle(frame2, point, 30, (255, 0, 0))
    return frame2

def hough(frame):
    dst = cv2.Canny(frame, 50, 200, None, 3)
    gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray, 50, 200, None, 3)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            print(linesP[i])
            # if np.abs(linesP[i][0]) < 0.3:
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

def line_segment_detector(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segments = lsd(frame_gray, scale=0.5)

    for i in range(segments.shape[0]):
        pt1 = (int(segments[i, 0]), int(segments[i, 1]))
        pt2 = (int(segments[i, 2]), int(segments[i, 3]))
        width = segments[i, 4]
        angle = np.arctan2(pt1[1]-pt2[1], pt1[0]-pt2[0])
        print(angle)
        if np.abs(angle) > np.pi/2 + 0.1:  #vertical lines
            cv2.line(frame, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

def trackbar_value_H(value):
    global tol_h
    tol_h = value

def trackbar_value_S(value):
    global tol_s
    tol_s = value

def trackbar_value_V(value):
    global tol_v
    tol_v = value

def onChange(trackbarValue):
    global cap
    global video_pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)

def main():
    global cap, video_pos
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    winname = "Flood"
    cv2.namedWindow(winname)
    cv2.createTrackbar('Threshold H', winname, tol_h, 300, trackbar_value_H)
    cv2.createTrackbar('Threshold S', winname, tol_s, 300, trackbar_value_S)
    cv2.createTrackbar('Threshold V', winname, tol_v, 300, trackbar_value_V)
    cv2.createTrackbar('Video', winname, 0, length, onChange)
    cv2.setMouseCallback(winname, mouse_clic)

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (700, 500))
            frame2 = flood_fill(frame, tol_h, tol_s, tol_v)
            video_pos += 1
            cv2.setTrackbarPos('Video', winname, video_pos)

            #hough(frame2)
            #line_segment_detector(frame2)

            # Display the resulting frame
            cv2.imshow(winname, frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('segmented_file.png', frame2)
                break
            time.sleep(0.1)

        # Break the loop
        else:
            print(ret)
            break

    # When everything done, release the video capture object
    # while cv2.waitKey(25) & 0xFF != ord('q'):
    #     pass

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


