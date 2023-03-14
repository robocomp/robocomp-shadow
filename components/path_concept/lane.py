# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import time
from itertools import islice
# from pylsd import lsd   # https://github.com/AndranikSargsyan/pylsd-nova

curvature = 0   # mapped from 0..max_curvature to -max_curvature/2 to max_curvature/2
max_curvature = 500
curvature_limit = 140
arc = 350
max_arc = 500
lane_width = 50
max_lane_width = 300
projection = 5
max_projection = 10
min_projection = 0.4

winname = "Lane"
number_of_points = 5

def mouse_clic(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pass

def trackbar_curvature(value):
    global curvature, max_curvature
    curvature = value-max_curvature//2

def trackbar_lane_width(value):
    global lane_width
    lane_width = value

def trackbar_arc(value):
    global arc
    arc = value

def trackbar_projection(value):
    global projection
    projection = value

def draw_lanes(img):
    global winname, curvature, arc, max_curvature, lane_width, number_of_points, curvature_limit

    halfcurv = max_curvature//2
    height, width = img.shape[0:2]
    height = height - 50     # margin from bottom margin
    hwidth = width//2
    left_points = []
    right_points = []
    points = []
    if curvature == 0:
        left_points.append((hwidth-lane_width//2, height))
        left_points.append((hwidth-lane_width//2, height-arc))
        right_points.append((hwidth+lane_width//2, height))
        right_points.append((hwidth + lane_width // 2, height - arc))
    elif curvature > -curvature_limit and curvature < curvature_limit:      # limits in curvature
        k = np.log(10000) / halfcurv     # mapping halfcurv -> 0 to 10.000 -> 0
        if curvature > 0:
            left_radius = halfcurv - curvature   # goes from 250 to 0. High radius means low curvature. The exp maps from 10000 to 0
            left_radius = np.exp(left_radius*k)
            right_radius = halfcurv - curvature - lane_width/20  # goes from 250 to 0. High radius means low curvature. The exp maps from 10000 to 0
            right_radius = np.exp(right_radius * k)
        else:
            left_radius = -curvature - halfcurv
            left_radius = -np.exp(-left_radius * k)
            right_radius = -curvature - halfcurv - lane_width/20
            right_radius = -np.exp(-right_radius * k)

        # left
        max_angle_left = arc / left_radius
        for t in np.linspace(np.pi, np.pi+max_angle_left, number_of_points):
            p = (int(left_radius*np.cos(t)+left_radius+hwidth-lane_width//2), int(left_radius*np.sin(t))+height)
            left_points.append(p)

        # right
        max_angle_right = arc / right_radius
        for t in np.linspace(np.pi, np.pi + max_angle_right, number_of_points):
            p = (int(right_radius*np.cos(t)+right_radius+hwidth+lane_width//2), int(right_radius*np.sin(t))+height)
            right_points.append(p)


    # fake projection on polygon points into image
    # we want to reduce the current separation between lines at the middle length
    # down to its half, and recompute the points according to a linear law
    # starting from the base (which is not changed) and up to the end.
    proj = ((1-0.4)/max_projection)*projection + 0.4
    left_proj_points = []
    right_proj_points = []
    proj_points = []
    for i in range(len(left_points)):
        length_reduc = -((1-proj)/len(left_points))*i + 1        # this is the reduction in length for each pair of opposite points
        lp = np.array(left_points[i])
        rp = np.array(right_points[i])
        left_proj_points.append((lp + ((rp-lp) * length_reduc)))
        right_proj_points.append((rp + ((lp-rp) * length_reduc)))

    left_proj_points = [(int(x), int(y)) for x, y in left_proj_points]
    right_proj_points = [(int(x), int(y)) for x, y in right_proj_points]
    points.extend(left_points)
    points.extend(right_points[::-1])
    proj_points.extend(left_proj_points)
    proj_points.extend(right_proj_points[::-1])

    return points, proj_points

def main():

    cv2.namedWindow(winname)
    cv2.createTrackbar('curvature', winname, curvature, max_curvature, trackbar_curvature)
    cv2.setTrackbarPos('curvature', winname, max_curvature//2)
    cv2.createTrackbar('lane_width', winname, lane_width, max_lane_width, trackbar_lane_width)
    cv2.setTrackbarPos('lane_width', winname, max_lane_width//2)
    cv2.createTrackbar('arc', winname, arc, max_arc, trackbar_arc)
    cv2.setTrackbarPos('arc', winname, arc)
    cv2.createTrackbar('projection', winname, projection, max_projection, trackbar_projection)
    cv2.setTrackbarPos('projection', winname, projection)
    cv2.setMouseCallback(winname, mouse_clic)

    while True:
        img = np.full((500, 500, 1), 255, np.uint8)
        points, proj_points = draw_lanes(img)

        # draw points and lines
        # for p in points:
        #     cv2.circle(img, p, 5, 0)
        # if points:
        #     points.append(points[0])
        # for i in range(len(points) - 2 + 1):
        #     cv2.line(img, points[i], points[i + 1], 0, 2)

        # draw projected points and lines
        for p in proj_points:
            cv2.circle(img, tuple(p), 5, 0)
        if proj_points:
            proj_points.append(proj_points[0])
        for i in range(len(proj_points) - 2 + 1):
            cv2.line(img, proj_points[i], proj_points[i + 1], 0, 2)

        # draw mask
        mask_poly = np.zeros((500, 500), np.uint8)
        cv2.fillPoly(mask_poly, pts=np.array([proj_points]), color=255)

        cv2.imshow(winname, img)
        cv2.imshow("mask", mask_poly)
        cv2.waitKey(5)


    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# def draw_lane_2():
#     global winname, r_left
#     width = 500
#     height = 500
#     img = np.full((height, width, 1), 255, np.uint8)
#     k = np.log(10000)/100
#     radius = int(np.exp(r_left * k))
#     center = (radius+width//2, height//2)
#     if r_left > 0:
#         min_ang = 90
#         max_ang = angle
#     else:
#         min_ang = 360
#         max_ang = (angle + 90)
#
#     cv2.ellipse(img, center, (abs(radius), abs(radius)), 180, 0, angle, 0, 5)
#     #cv2.ellipse(img, center, (abs(r_left), height), 0, 90, angle, 0, 5)
#     #cv2.ellipse(img, center, (abs(r_left), height), 0, 360, angle, 0, 5)
#
#     cv2.imshow(winname, img)
#     cv2.waitKey(5)
# def draw_lane():
#     global winname
#     width = 500
#     height = 500
#     img = np.full((height, width, 1), 255, np.uint8)
#     center = (r_left+width//2, height)
#     if r_left > 0:
#         min_ang = 90
#         max_ang = angle
#     else:
#         min_ang = 360
#         max_ang = (angle + 90)
#     cv2.ellipse(img, center, (abs(r_left), height), 0, min_ang, max_ang, 0, 5)
#     #cv2.ellipse(img, center, (abs(r_left), height), 0, 90, angle, 0, 5)
#     #cv2.ellipse(img, center, (abs(r_left), height), 0, 360, angle, 0, 5)
#
#     cv2.imshow(winname, img)
#     cv2.waitKey(5)
#


# project on image. We assume 3D coordinates in -2500, 2500 mm range (x10 from image)
        # img_proj = np.full((500, 500, 1), 255, np.uint8)
        # focal = 400
        # z = -1500
        # for p in points:
        #     x = (p[0] - 250)*10
        #     y = (500 - p[1])*10
        #     pp = (int(focal*x/y), int(focal*z/y))
        #     cv2.circle(img_proj, pp, 5, 0)
        #     print("pp", pp)
        # print("------------")
        # cv2.imshow("Projection", img_proj)
        # cv2.waitKey(2)
