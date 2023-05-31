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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import cv2
import time
import traceback

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100
        if startup_check:
            self.startup_check()
        else:

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # Initial values
            self.candidates = self.create_candidates()


            # self.params = np.array(candidates["params"])
            # self.camera_matrix = camera_matrix
            # self.focalx = focalx
            # self.focaly = focaly
            # self.frame_shape = frame_shape
            # project polygons on image

            #self.candidates = self.project_polygons(self.candidates)
            #self.candidates = self.create_masks(frame_shape, self.candidates)

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

        try:
            self.lidar3d_proxy.getLidarData(0, 900)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error connecting to Lidar3D")

        img = np.zeros((700, 700, 3), dtype=np.uint8)
        self.draw_candidates(self.candidates, img)
        cv2.imshow("Candidates", img)
        cv2.waitKey(2)

        self.show_fps()

    def create_candidates(self):
        adv_max_accel = 600
        rot_max_accel = 0.4
        time_ahead = 1.5
        step_along_ang = 0.05
        advance_step = 100
        rotation_step = 0.2
        current_adv_speed = 0
        current_rot_speed = 0
        max_reachable_adv_speed = adv_max_accel * time_ahead
        max_reachable_rot_speed = rot_max_accel * time_ahead
        robot_semi_width = 200

        params = []
        targets = []
        trajectories = []
        for v in np.arange(100, max_reachable_adv_speed, advance_step):
            for w in np.arange(0, max_reachable_rot_speed, rotation_step):
                new_advance = current_adv_speed + v
                new_rotation = current_rot_speed + w
                arc_length = new_advance * time_ahead
                if w > 0:
                    r = new_advance / new_rotation
                    ang_length = arc_length / r

                    # compute LEFT arcs corresponding to r - 100 and r + 100
                    points = []
                    central = []
                    for t in np.arange(0, ang_length, step_along_ang):
                        xl = (r - robot_semi_width) * np.cos(t) - r
                        yl = (r - robot_semi_width) * np.sin(t)
                        points.append([xl, yl])
                    ipoints = []
                    for t in np.arange(0, ang_length, step_along_ang):  # inverse order to build a proper polygon
                        xh = (r + robot_semi_width) * np.cos(t) - r
                        yh = (r + robot_semi_width) * np.sin(t)
                        ipoints.append([xh, yh])
                    ipoints.reverse()
                    # add to trajectories
                    if len(points) > 2 and len(ipoints) > 2:
                        points.extend(ipoints)
                        trajectories.append(points)
                        params.append([new_advance, -new_rotation, -r])

                    # now compute RIGHT arcs corresponding to r - 100 and r + 100
                    points = []
                    central = []
                    for t in np.arange(0, ang_length, step_along_ang):
                        xl = r - (r - robot_semi_width) * np.cos(t)
                        yl = (r - robot_semi_width) * np.sin(t)
                        points.append([xl, yl])
                    ipoints = []
                    for t in np.arange(0, ang_length, step_along_ang):  # inverse order to build a proper polygon
                        xh = r - (r + robot_semi_width) * np.cos(t)
                        yh = (r + robot_semi_width) * np.sin(t)
                        ipoints.append([xh, yh])
                    ipoints.reverse()
                    # add to trajectories
                    if len(points) > 2 and len(ipoints) > 2:
                        points.extend(ipoints)
                        trajectories.append(points)
                        params.append([new_advance, new_rotation, r])

            else:  # avoid division by zero
                points = []
                central = []
                points.append([-robot_semi_width, 0])
                points.append([-robot_semi_width, arc_length])
                points.append([robot_semi_width, arc_length])
                points.append([robot_semi_width, 0])
                trajectories.append(points)
                params.append([new_advance, 0.0, np.inf])

        params = np.array(params)
        params[:, 2] = 100 * np.reciprocal(params[:, 2])  # compute curvature from radius
        candidates = []
        for pa, tr in zip(params, trajectories):
            candidate = {}
            candidate["polygon"] = tr
            candidate["params"] = pa
            candidates.append(candidate)
        print("Created ", len(candidates), " candidates")
        return candidates

    def draw_candidates(self, candidates, img):

        height = img.shape[1]
        for c in candidates:
            pol = (np.array(c["polygon"]) * 0.5).astype(int)
            pol[:, 1] = height - pol[:, 1]
            pol[:, 0] += 350
            cv2.polylines(img, [pol], True, (255, 255, 255), 1, cv2.LINE_8)

    def discard(self, mask_img):
        # discard all paths with less than 95% occupancy
        survivors = []
        for c in self.candidates:
            mask = c["mask"]
            result = cv2.bitwise_and(mask_img, mask)
            lane_size = np.count_nonzero(mask)
            inliers = np.count_nonzero(result)
            segmented_size = np.count_nonzero(mask_img)
            if lane_size == 0:
                continue
            # print(inliers * 100 / lane_size)
            if inliers * 100 / lane_size > 90:
                loss = abs(segmented_size - lane_size) + 5 * abs(lane_size - inliers)
                survivor = {"mask": mask, "polygon": c["polygon"],
                            "params": c["params"], "loss": loss,
                            "projected_polygon": c["projected_polygon"],
                            "trajectory": c["trajectory"]}
                survivors.append(survivor)
        return survivors

    def optimize(self, mask_img, target_object, x0=None):
        curvature_threshold = 0.025  # range: -1, 1
        losses = []
        for i, mask in enumerate(self.masks):
            distance_to_target_object = 0
            if target_object is not None:
                target_center = np.array(
                    [target_object.right - target_object.left, target_object.bot - target_object.top])
                tip_of_path = self.get_tip_of_path(self.polygons[i])
                distance_to_target_object = np.linalg.norm(target_center - tip_of_path)
            loss = self.loss_function(mask_img, mask, distance_to_target_object)
            losses.append(loss)
        losses = np.array(losses)
        # sorted_loss_index = np.argsort(losses)
        params = self.params
        params = np.c_[params, losses]


        # return path_set, curvatures, selected_target_trajs, controls
        return params, self.targets

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
            print("Freq:", self.cont, "ms. Curr period:", cur_period)
            #self.thread_period = np.clip(self.thread_period+delta, 0, 200)
            self.cont = 0
        else:
            self.cont += 1
    ################################################################
    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)





