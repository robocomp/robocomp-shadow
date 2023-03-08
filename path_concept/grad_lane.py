''' Computes the grounding of the lane model by optimization '''

import cv2
import numpy as np
import time, random
import sys
import threading
from queue import SimpleQueue
#from nevergrad_proxy import Nevergrad_Proxy
#from random_optimizer import Random_Optimizer
from mask2former import Mask2Former
from scipy import optimize
from dwa_optimizer import DWA_Optimizer
from scipy.optimize import Bounds

# from pylsd import lsd   # https://github.com/AndranikSargsyan/pylsd-nova

curvature_limit = 130
max_curvature = 150
min_curvature = -max_curvature
max_arc = 400
max_lane_width = 500
max_projection = 10
min_projection = 0.4
number_of_points = 5
frame = []

mask2former = Mask2Former()
cap = cv2.VideoCapture('hall.mp4')
#cap = cv2.VideoCapture('track6.mov')
#cap = cv2.VideoCapture('hall.mp4')
video_frame = 0

def trackbar_video_slider(value):
    global cap, video_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, value)
    video_frame = value

def flood_fill(frame, tol_h, tol_s, tol_v):
    point = (220, 260)
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    connectivity = 4
    flags = connectivity | (255 << 8)
    flags |= cv2.FLOODFILL_FIXED_RANGE
    mask = np.zeros((frameHSV.shape[0]+2, frameHSV.shape[1]+2), np.uint8)
    retval, image, mask, rect = cv2.floodFill(frameHSV, mask, point, (128, 200, 150), (tol_h, tol_s, tol_v), (tol_h, tol_s, tol_v), flags)
    #     cv2.imshow("flood", mask)
    #     cv2.waitKey(5)
    frame2 = cv2.cvtColor(frameHSV, cv2.COLOR_HSV2BGR)
    cv2.circle(frame2, point, 20, (0, 0, 255))
    return frame2, mask

def sample_lane_model(frame_shape, curvature, arc, lane_width, projection):
    global curvature_limit, max_curvature, max_arc, max_lane_width, max_projection, min_projection, number_of_points

    height, width = frame_shape[0:2]
    halfcurv = max_curvature//2
    height = height - 10    # margin from bottom margin
    hwidth = width//2
    left_points = []
    right_points = []
    points = []
    if curvature == 0:
        left_points.append((hwidth-lane_width//2, height))
        left_points.append((hwidth-lane_width//2, height-arc))
        right_points.append((hwidth+lane_width//2, height))
        right_points.append((hwidth + lane_width // 2, height - arc))
    elif curvature > -curvature_limit and curvature < curvature_limit:      # limits in curvature. Units are wrong
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
    target = (np.array(left_proj_points)[-1] + np.array(right_proj_points)[-1])/2
    points.extend(left_points)
    points.extend(right_points[::-1])
    proj_points.extend(left_proj_points)
    proj_points.extend(right_proj_points[::-1])

    mask_poly = np.zeros(frame_shape, np.uint8)
    if proj_points:
        cv2.fillPoly(mask_poly, pts=np.array([proj_points]), color=255)

    return mask_poly, (int(target[0]), int(target[1]))

def draw(winname, frame, params):
    #segmented_frame, mask_img = flood_fill(frame, 20, 20, 200)  # tol_h, tol_s, tol_v)
    mask_poly, target = sample_lane_model(frame.shape[0:2], params[0], params[1], 150, params[2])
    alpha = 0.8
    color_lane = cv2.cvtColor(mask_poly, cv2.COLOR_GRAY2BGR)  # remove 2 last columns
    color_lane[np.all(color_lane == (255, 255, 255), axis=-1)] = (0, 255, 0)    # green
    frame_new = cv2.addWeighted(frame, alpha, color_lane, 1 - alpha, 0)
    cv2.circle(frame_new, target, 5, (255, 0, 0), cv2.FILLED)
    cv2.imshow(winname, frame_new)
    cv2.waitKey(2)

def draw_frame(winname, frame, mask_poly, alternatives):
    alpha = 0.8
    color_lane = cv2.cvtColor(mask_poly, cv2.COLOR_GRAY2BGR)
    color_lane[np.all(color_lane == (255, 255, 255), axis=-1)] = (0, 255, 0)    # green
    frame_new = cv2.addWeighted(frame, alpha, color_lane, 1 - alpha, 0)
    #cv2.circle(frame_new, target, 5, (255, 0, 0), cv2.FILLED)

    if len(alternatives) > 0:
        alt_lane = cv2.cvtColor(alternatives[0], cv2.COLOR_GRAY2BGR)
        alt_lane[np.all(color_lane == (255, 255, 255), axis=-1)] = (255, 0, 0)  # green
        frame_new = cv2.addWeighted(frame_new, alpha, alt_lane, 1 - alpha, 0)

    cv2.imshow(winname, frame_new)
    cv2.waitKey(2)

def target_function(params, other=()):
    global frame
    mask_img = other
    #segmented_frame, mask_img = flood_fill(frame, 20, 20, 200)  # tol_h, tol_s, tol_v)
    mask_poly, _ = sample_lane_model(mask_img.shape, params[0], params[1], 150, params[2])
    result = cv2.bitwise_and(mask_img, mask_poly)
    lane_size = np.count_nonzero(mask_poly)
    segmented_size = np.count_nonzero(mask_img)
    inliers = np.count_nonzero(result)
    # loss function: number of inliers out + number of outliers in
    #loss = (segmented_size - inliers) + (lane_size - inliers)
    loss = abs(segmented_size - lane_size) + 5*abs(lane_size-inliers)
    #loss = (segmented_size - inliers) #+ abs(lane_size-segmented_size)
    return float(loss)

def target_function_mask(mask_img, mask_poly):
    result = cv2.bitwise_and(mask_img, mask_poly)
    lane_size = np.count_nonzero(mask_poly)
    segmented_size = np.count_nonzero(mask_img)
    inliers = np.count_nonzero(result)
    loss = abs(segmented_size - lane_size) + 5*abs(lane_size-inliers)
    return float(loss)

def thread_frame_capture(cap, frame_queue, winname, video_frame, mask2former):
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (400, 400))
        mask_img = mask2former.process(frame)
        frame_queue.put([frame, mask_img])

def main():
    global frame, cap, video_frame

    random.seed()
    curvature = 0   # mapped from 0..max_curvature to -max_curvature/2 to max_curvature/2
    arc = 350
    lane_width = 50
    projection = 5
    #curv_range = (-max_curvature, max_curvature)
    curv_range = (-30, 30, 1)
    arc_range = (1, max_arc, 5)
    lane_range = (100, max_lane_width*2, 5)
    proj_range = (0, max_projection-7, 1)

    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit()

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    winname = "Lane"
    cv2.namedWindow(winname)
    cv2.createTrackbar('FF', winname, 0, video_length, trackbar_video_slider)

    dwa_optimizer = DWA_Optimizer()
    #while cv2.waitKey(25) & 0xFF != ord('q'):
    #    pass
    #sys.exit()

    # init optimizer
    #bounds = Bounds([-15, 1, 149, 0], [15, 390, 151, 5])
    #bounds = Bounds([-22, 1, 0], [22, 390, 5])
    #optimum = [0, 100, 150, 5]
    optimum = [0, 100, 5]

    # start frame thread
    frame_queue = SimpleQueue()
    thread_frame = threading.Thread(target=thread_frame_capture, args=(cap, frame_queue, winname, video_frame, mask2former), daemon=True)
    thread_frame.start()
    print("Video frame started")

    while True:
        now = time.time()
        frame, mask_img = frame_queue.get()
        # result = optimize.differential_evolution(target_function, bounds, args=[mask_img], x0=optimum,
        #                                            updating='immediate', workers=1,
        #                                            maxiter=5, strategy='best2bin')
        #
        now2 = time.time()

        loss, mask, alternatives = dwa_optimizer.optimize(loss=target_function_mask, mask_img=mask_img)
        #draw(winname, frame, result.x)
        draw_frame(winname, frame, mask, alternatives)
        #optimum = result.x
        print("Loss:", loss, "Elapsed:", now2-now, time.time()-now2, time.time()-now, len(alternatives))

    while cv2.waitKey(25) & 0xFF != ord('q'):
        pass
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

 # best_params = np.array([0, 0, 0, 0])
    # while cap.isOpened():
    # #for k in range(1):
    #     ret, frame = cap.read()
    #     if ret:
    #         frame = cv2.resize(frame, (400, 400))
    #         segmented_frame, mask_img = flood_fill(frame, 20, 20, 200)   # tol_h, tol_s, tol_v)
    #         last_inliers_count = 0
    #         last_outliers_count = mask_img.size
    #         mask_poly_size = 1
    #         mask_image_size = np.count_nonzero(mask_img)
    #         best_score = 0
    #         first_time = True
    #         now = time.time()
    #         # while 100*last_inliers_count/mask_poly_size < 80:
    #         for k in range(2000):
    #             #print(100.0*last_inliers_count/mask_poly_size, "%")
    #             #print("b ", last_inliers_count, mask_poly_size)
    #             if first_time:
    #                 mask_poly = sample_lane_model(mask_img.shape, best_params[0], best_params[1], best_params[2], best_params[3])  # width, height
    #                 result = cv2.bitwise_and(mask_img, mask_poly)
    #                 first_time = False
    #             else:
    #                 curvature, arc, lane_width, projection = random_sampler([curv_range, arc_range, lane_range, proj_range])
    #                 mask_poly = sample_lane_model(mask_img.shape, curvature, arc, lane_width, projection)
    #                 result = cv2.bitwise_and(mask_img, mask_poly)
    #             inliers = np.count_nonzero(result)
    #             outliers = np.count_nonzero(mask_poly) - inliers
    #             if (inliers - outliers*4) > best_score:
    #                 best_score = (inliers - 4*outliers)
    #                 last_inliers_count = inliers
    #                 last_outliers_count = outliers
    #                 mask_poly_size = np.count_nonzero(mask_poly)
    #
    #                 # substract from previous set
    #                 diff_params = np.subtract(np.array(best_params), np.array([curvature, arc, lane_width, projection]))
    #                 new_params = np.array(best_params) - 0.05 * diff_params
    #                 mask_poly = sample_lane_model(mask_img.shape, new_params[0], new_params[1], new_params[2], new_params[3])
    #                 best_params = new_params
    #
    #                 #best_params = (curvature, arc, lane_width, projection)
    #
    #                 alpha = 0.4
    #                 color_mask = cv2.cvtColor(mask_poly[:-2, :-2], cv2.COLOR_GRAY2BGR)  # remove 2 last columns
    #                 color_mask[np.all(color_mask == (255, 255, 255), axis=-1)] = (0, 255, 0)    # green
    #                 frame_new = cv2.addWeighted(frame, alpha, color_mask, 1 - alpha, 0)
    #                 cv2.imshow("Detector", frame_new)
    #                 cv2.imshow("Image Mask", mask_img)
    #                 print(100.0 * last_inliers_count / mask_poly_size, "%", best_score, last_inliers_count,
    #                       last_outliers_count, mask_poly_size, mask_image_size)
    #                 #cv2.imshow("mask", mask_poly)
    #                 cv2.waitKey(2)
    #         end = time.time()
    #         #print(end-now)

        #cv2.waitKey(10)


# nevergrad
# ngrad = Nevergrad_Proxy(target_function)
# rndopt = Random_Optimizer(target_function, [])