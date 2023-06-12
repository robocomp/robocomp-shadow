#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide6.QtCore import *
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPolygonF
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import cv2
import time
import traceback
from shapely.geometry import Point, Polygon
from dataclasses import dataclass
from collections import deque
from typing import List, Dict
from numpy.typing import NDArray
from typing import Any

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50
        self.A = 0
        self.B = 0
        self.C = 0
        if startup_check:
            self.startup_check()
        else:

            self.window_name = "DWA"
            cv2.namedWindow(self.window_name)
            cv2.createTrackbar('A VALUE', self.window_name, 0, 10, self.A_on_change)
            cv2.createTrackbar('B VALUE', self.window_name, 0, 10, self.B_on_change)
            cv2.createTrackbar('C VALUE', self.window_name, 0, 10, self.C_on_change)
            self.z_lidar_height = 1250
            self.z_threshold = self.z_lidar_height

            #image size
            self.width = 700
            self.height = 700
            self.world_width = 5000 #mm
            self.world_height = 4000  # mm
            self.gfactor_x = self.width / self.world_width
            self.gfactor_y = self.height / self.world_height

            # Hz
            self.cont = 1
            self.last_time = time.time()
            self.fps = 0

            # dynamic params
            @dataclass
            class TDynamic:
                adv_max_accel: float = 600
                rot_max_accel: float = 0.8
                time_ahead: float = 1.5
                step_along_ang: float = 0.05
                advance_step: float = 100
                rotation_step: float = 0.2
                max_distance_ahead: float = adv_max_accel * time_ahead * 1.5    # excess factor
            self.dyn = TDynamic()

            # Initial values
            self.candidates = self.create_candidates_differential()

            # target
            self.target = None
            self.prev_fovea_error = 0
            self.queue_fovea_error = deque(maxlen=5)

            # optimal
            self.previous_choice = np.zeros(2)

            self.timer.timeout.connect(self.compute)
            #self.timer.setSingleShot(True)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True

    @QtCore.Slot()
    def compute(self):
        now = time.time()
        ldata_set = self.read_lidar_data()
        #print("t1", time.time() - now)

        # discard_occupied_lanes
        safe_lanes = self.discard_occupied_lanes(ldata_set, self.candidates)
        #print("t2", time.time() - now)

        # base image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # filas, columnas

        # draw lidar points
        self.draw_lidar_points(ldata_set, img)

        # draw target
        # if self.target:
        #     print(self.target.depth)

        # select optimal lane
        optimal = self.select_optimal_lane(safe_lanes, self.target, ldata_set)
        #print(optimal[1]['params'])
        
        # control.
        if optimal is not None:
            self.control(optimal[1]['tip'])
            self.draw_target(img, optimal[1]['tip'])

        # draw candidates
        self.draw_candidates(safe_lanes, img, optimal)

        cv2.imshow(self.window_name, img)
        cv2.waitKey(2)
        self.show_fps()

    def create_candidates_differential(self):
        current_adv_speed = 0
        current_rot_speed = 0
        max_reachable_adv_speed = self.dyn.adv_max_accel * self.dyn.time_ahead
        max_reachable_rot_speed = self.dyn.rot_max_accel * self.dyn.time_ahead
        robot_semi_width = 200

        params = []
        tips = []
        trajectories = []
        for v in np.arange(100, max_reachable_adv_speed, self.dyn.advance_step):
            for w in np.arange(0, max_reachable_rot_speed, self.dyn.rotation_step):
                new_advance = current_adv_speed + v
                new_rotation = current_rot_speed + w
                arc_length = new_advance * self.dyn.time_ahead
                if w > 0:
                    r = new_advance / new_rotation
                    ang_length = arc_length / r

                    # compute LEFT arcs corresponding to r - 100 and r + 100
                    points = []
                    for t in np.arange(0, ang_length, self.dyn.step_along_ang):
                        xl = (r - robot_semi_width) * np.cos(t) - r
                        yl = (r - robot_semi_width) * np.sin(t)
                        points.append([xl, yl])
                    ipoints = []
                    for t in np.arange(0, ang_length, self.dyn.step_along_ang):  # inverse order to build a proper polygon
                        xh = (r + robot_semi_width) * np.cos(t) - r
                        yh = (r + robot_semi_width) * np.sin(t)
                        ipoints.append([xh, yh])
                    ipoints.reverse()
                    # add to trajectories
                    if len(points) > 2 and len(ipoints) > 2:
                        points.extend(ipoints)
                        trajectories.append(points)
                        params.append([new_advance, -new_rotation, -r])
                        tips.append([r * np.cos(ang_length-self.dyn.step_along_ang) - r, r * np.sin(ang_length-self.dyn.step_along_ang)])

                    # now compute RIGHT arcs corresponding to r - 100 and r + 100
                    points = []
                    for t in np.arange(0, ang_length, self.dyn.step_along_ang):
                        xl = r - (r - robot_semi_width) * np.cos(t)
                        yl = (r - robot_semi_width) * np.sin(t)
                        points.append([xl, yl])
                    ipoints = []
                    for t in np.arange(0, ang_length, self.dyn.step_along_ang):  # inverse order to build a proper polygon
                        xh = r - (r + robot_semi_width) * np.cos(t)
                        yh = (r + robot_semi_width) * np.sin(t)
                        ipoints.append([xh, yh])
                    ipoints.reverse()
                    # add to trajectories
                    if len(points) > 2 and len(ipoints) > 2:
                        points.extend(ipoints)
                        trajectories.append(points)
                        params.append([new_advance, new_rotation, r])
                        tips.append([r - r*np.cos(ang_length-self.dyn.step_along_ang), r*np.sin(ang_length-self.dyn.step_along_ang)])

                else:  # avoid division by zero for straight lanes
                    points = []
                    points.append([-robot_semi_width, 0])
                    points.append([-robot_semi_width, arc_length])
                    points.append([robot_semi_width, arc_length])
                    points.append([robot_semi_width, 0])
                    trajectories.append(points)
                    params.append([new_advance, 0.0, np.inf])
                    tips.append([0, arc_length])

        params = np.array(params)
        params[:, 2] = 100 * np.reciprocal(params[:, 2])  # compute curvature from radius
        candidates = []
        kid = 0
        for pa, tr, tip in zip(params, trajectories, tips):
            candidate = {}
            candidate["polygon"] = tr
            candidate["params"] = pa
            candidate["tip"] = tip
            candidate["id"] = kid
            candidates.append(candidate)
            kid += 1
        #print("Created ", len(candidates), " candidates")
        return candidates

    def read_lidar_data(self) -> NDArray[[float, float]]:
        ldata_set = []
        try:
            ldata = self.lidar3d_proxy.getLidarData(787, 225)
            # remove points 30cm from floor and above robot
            ldata_set = [(l.x, l.y) for l in ldata
                         if l.z > (400 - self.z_lidar_height)
                         and l.z < 300                              # robot's height
                         and np.linalg.norm((l.x, l.y)) > 400       # robot's body]
                         and np.linalg.norm((l.x, l.y)) < 3000]
                         #and np.linalg.norm((l.x, l.y)) < self.dyn.max_distance_ahead]     # too far away. TODO: add current speed
            #print(len(ldata), len(ldata_set))
        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error connecting to Lidar3D")
        return ldata_set

    def discard_occupied_lanes(self, ldata, lanes):
        safe_lanes = []
        qp_lanes = [(QPolygonF([QPointF(p[0], p[1]) for p in l["polygon"]]), l) for l in lanes]
        for qp, l in qp_lanes:
            if all(not qp.containsPoint(QPointF(point[0], point[1]), Qt.OddEvenFill) for point in ldata):
                safe_lanes.append(l)
        return safe_lanes

    def discard_occupied_lanes2(self, ldata, lanes):
        safe_lanes = []
        polygons = [Polygon([Point(p[0], p[1]) for p in l["polygon"]]) for l in lanes]
        for poly, l in zip(polygons, lanes):
            if not any(poly.contains(Point(point[0], point[1])) for point in ldata):
                safe_lanes.append(l)
        return safe_lanes

    def draw_candidates(self, candidates, img, optimal):

        for c in candidates:
            pol = (np.array(c["polygon"]) * np.array([self.gfactor_x, self.gfactor_y])).astype(int)
            pol[:, 1] = self.height - pol[:, 1]
            pol[:, 0] += self.width//2
            if optimal is not None and c["id"] == optimal[1]["id"]:
                color = (0, 0, 255)
                thick = 4
            else:
                color = (255, 255, 255)
                thick = 1
            cv2.polylines(img, [pol], True, color, 1, cv2.LINE_8)
            #print(c["tip"])
            tip = np.array(c["tip"]) * np.array([self.gfactor_x, self.gfactor_y])
            cv2.circle(img, np.array([tip[0] + self.width//2, self.height - tip[1]]).astype(int), 5, color, thick)

    def draw_lidar_points(self, ldata_set, img):
        img_points = []
        points = []
        for l in ldata_set:
            p = np.array([int(l[0]*self.gfactor_x) + self.width//2, self.height-int(l[1]*self.gfactor_y)])
            if(p[0] > 0 and p[0] < self.width and p[1]> 0 and p[1] < self.height):
                #img_points.append(p)
                #points.append([l[0], l[1]])
                cv2.rectangle(img, p-(2, 2), p+(2, 2), (0, 255, 0))
        # if img_points and points:
        #     x, y, w, h = cv2.boundingRect(np.array(img_points))
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))
        #     if self.target:
        #         # points = np.array(points)
        #         # self.target.x = sum(points[:, 0] / len(points))
        #         # self.target.y = sum(points[:, 1] / len(points))
        #         # self.target.depth = np.linalg.norm([self.target.x, self.target.y])
        #         dist = str(int(self.target.depth)) + " mm"
        #         cv2.putText(img, dist, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)

    def draw_target(self, img, optimal):
        if self.target is not None:
            size = np.array((10, 10))
            cx = int(self.target.x*self.gfactor_x)+self.width//2
            cy = self.height-int(self.target.y*self.gfactor_y)
            p = np.array((cx, cy))
            cv2.rectangle(img, p-size, p+size, (255, 0, 128), 3)

            if optimal:
                dist = str(int(optimal[0])) + " " + str(int(self.target.depth)) + " " + str(int(optimal[1]))
            else:
                dist = str(int(self.target.depth))
            cv2.putText(img, dist, p + (20, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 128), thickness=1)
            #print(self.target.x, self.target.y, (int(self.target.x*self.gfactor_x+self.width//2),
            #                                     self.height-int(self.target.y*self.gfactor_y)))

    def select_optimal_lane(self, lanes: List[Any], target, points):
        if not target:
            return

        A = self.A   # dist to target
        B = self.B   # dist to previous action
        C = self.C   # dist to obstacles
        points = np.array(points)
        target = np.array((target.x, target.y))

        tips = np.zeros(shape=(len(lanes), 2))
        for i, l in enumerate(lanes):
            tips[i] = np.array(l["tip"])

        # distance to target
        l_tar = np.linalg.norm(tips - target, axis=1)
        l_tar = l_tar / np.max(l_tar)

        # distance to previous choice
        l_prev = np.linalg.norm(tips - self.previous_choice, axis=1)
        l_prev = l_prev / np.max(l_prev)

        # distance to obstacles
        l_obs = np.zeros(shape=(len(lanes)))
        if points.size > 0:
            dist_threshold = 1000   # max evaluated distance between obstacles and tips
            for i in range(len(lanes)):
                dists = np.linalg.norm(points - tips[i], axis=1)
                dists = np.clip(dists, 0, dist_threshold)
                l_obs[i] = (np.min(dists) * (-1/dist_threshold)) + 1

        suma = (A*l_tar) + (B*l_prev) + (C*l_obs)

        min_index = suma.argsort()[0]
        #c = cum[cum[:, 0].argsort()]
        self.previous_choice = np.array(lanes[min_index]["tip"])
        return suma[min_index], lanes[min_index]

    def normalize(self, value, max):
        return value / max

    def control(self, local_target):    # get local_target from DWA as next place to go
        MAX_ADV_SPEED = 1000
        MAX_ROT_SPEED = 2
        rot = np.arctan2(local_target[0], local_target[1])
        if self.target is not None:
            if self.target.depth < 800:

                self.stop_robot()
                self.target = None
                print("Control: Target achieved")
                # self.omnirobot_proxy.setSpeedBase(0, 0, rot)
                pass
            else:
                dist = np.linalg.norm(local_target)
                # rot = np.arctan2(local_target[0], local_target[1])
                # rot_error = self.target.roi.xcenter - 500    # full image half size
                # rot_error_der = rot_error - self.prev_fovea_error
                # self.queue_fovea_error.append(rot_error)
                # sum = 0
                # for i in range(len(self.queue_fovea_error)):
                #     sum += self.queue_fovea_error[i]
                # PID
                #rot_control = rot_error * (MAX_ROT_SPEED/400) + self.prev_fovea_error * (MAX_ROT_SPEED/450) -sum * (MAX_ROT_SPEED/5000)
                # rot_control = rot * (MAX_ROT_SPEED / 600)

                adv = MAX_ADV_SPEED * self.sigmoid(local_target[1])
                side = MAX_ADV_SPEED * self.sigmoid_side(local_target[0]) * 1.5
                #print("kkk", local_target[0], side)
                # print("dist: {:.2f} l_dist: {:.2f} side: {:.2f} adv: {:.2f} "
                #        "rot: {:.2f} rerror: {:.2f} rerror_der: {:.2f}".format(self.target.depth, dist, side, adv, rot_control, rot_error, rot_error_der))
                self.prev_fovea_error = rot

                try:
                    self.omnirobot_proxy.setSpeedBase(0, adv, rot) #rot_control)
                    # self.omnirobot_proxy.setSpeedBase(0, 0, 0) #rot_control)
                except Ice.Exception as e:
                    traceback.print_exc()
                    print(e, "Error connecting to omnirobot")
        else:
            self.stop_robot()
            print("Control: stopping")
            return

    def sigmoid(self, x):   # top = 1, front = 2
        x = x/1000  #  to m
        x0, k = [1.05425285, 3.72219712]
        #x0, k = [1.93401352, 1.9522765]
        return 1 / (1 + np.exp(-k * (x - x0)))

    def sigmoid_side(self, x):  # top = 1, front = 2
        x = x / 1000  # to m
        x0, k = [0.03549303, 1.62201825]
        r = (2 / (1 + np.exp(-k * (x - x0)))) - 1
        print(r)
        return r

    def stop_robot(self):
        try:
            print("STOPPING THE ROBOT")
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error connecting to omnirobot")

    def loss_function(self, mask_img, mask_path, distance_to_target_object=0):
        result = cv2.bitwise_and(mask_img, mask_path)
        lane_size = np.count_nonzero(mask_path)
        segmented_size = np.count_nonzero(mask_img)
        inliers = np.count_nonzero(result)
        loss = abs(segmented_size - lane_size) + 5 * abs(lane_size - inliers) + distance_to_target_object
        return float(loss)

    def create_masks(self, shape, candidates):
        # masks = []
        height, width, _ = shape
        for c in candidates:  # [xl,yl]
            t = c["projected_polygon"]
            nt = np.array(t).copy()
            mask_poly = np.zeros((shape[0], shape[1]), np.uint8)
            cv2.fillPoly(mask_poly, pts=[nt.astype(int)], color=255)
            # masks.append(mask_poly)
            c["mask"] = mask_poly
            # cv2.polylines(mask_poly, pts=[nt.astype(int)], isClosed=True, color=255)
            # cv2.imshow("mask", mask_poly)
            # cv2.waitKey(200)
        return candidates

    def project_polygons(self, candidates):

        # get 3D points in robot CS, transform to camera CS and project with projection equations
        # transform to camera CS
        frame_width, frame_height, _ = self.frame_shape
        for c in candidates:
            po = c["polygon"]
            extra_cols = np.array([np.zeros(len(po)), np.ones(len(po))]).transpose()
            npo = np.array(po)
            npo = np.append(npo, extra_cols, axis=1).transpose()
            cam_poly = self.camera_matrix.dot(npo).transpose()  # n x 4
            # project on camera
            xs = np.array([((cam_poly[:, 0] * self.focaly) / cam_poly[:, 1]) + (frame_width / 2),
                           ((cam_poly[:, 2] * -self.focalx) / cam_poly[:, 1]) + (frame_height / 2)]).transpose()
            c["projected_polygon"] = xs.astype(int)  # now in camera CS

        return candidates

    def show_fps(self):
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000./self.cont)
            #delta = (-1 if (period - cur_period) < -1 else (1 if (period - cur_period) > 1 else 0))
            print("Freq:", self.cont, "Hz. Curr period:", cur_period)
            #self.thread_period = np.clip(self.thread_period+delta, 0, 200)
            self.cont = 1
        else:
            self.cont += 1

    def z_on_change(self, val):
        print(val)
        self.z_threshold = val - self.z_lidar_height

    def A_on_change(self, val):
        print(val)
        self.A = val

    def B_on_change(self, val):
        print(val)
        self.B = val

    def C_on_change(self, val):
        print(val)
        self.C = val

    ################################################################
    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    #########################################################################
    # SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface3
    #########################################################################

    def SegmentatorTrackingPub_setTrack(self, target):
        #print("In callback: ", target)
        # if target.depth == 0:
        #     #print("Warning: target at depth 0")
        #     pass
        if target.id == -1:
            self.target = None
        else:
            #target.x = -(target.x + 800)
            #target.y /= 1.7
            self.target = target
            #print(self.target.depth)

# self.params = np.array(candidates["params"])
# self.camera_matrix = camera_matrix
# self.focalx = focalx
# self.focaly = focaly
# self.frame_shape = frame_shape
# project polygons on image

# self.candidates = self.project_polygons(self.candidates)
# self.candidates = self.create_masks(frame_shape, self.candidates)
