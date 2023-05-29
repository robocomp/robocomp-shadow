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
import traceback

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import cv2
import time

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
sys.path.append('/home/robocomp/robocomp/components/robocomp-shadow/insect/byte_tracker/ByteTrack')
# from yolox.tracker.byte_tracker_depth import BYTETracker as BYTETrackerDepth
from yolox.tracker.byte_tracker import BYTETracker
import json
from dataclasses import dataclass

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # params
        self.display = False
        self.lidar3D = False

        # ROI parameters. Must be filled up
        self.final_xsize = 0
        self.final_ysize = 0
        self.roi_xcenter = 0
        self.roi_ycenter = 0
        self.roi_xsize = 0
        self.roi_ysize = 0
        # self.cam_to_lidar = self.make_matrix_rt(0, 0, 0, 0, 0,
        #                                         305.65)  # converts points in omnicamera (coppelia) to velodyne (coppelia)
        # self.cam_to_lidar = self.make_matrix_rt(0, 0, 0, 0, 0,
        #                                         108.51)
        self.cam_to_lidar = self.make_matrix_rt(0, 0, 0, 0, 0,
                                                108.51)
        self.lidar_to_cam = np.linalg.inv(self.cam_to_lidar)
        self.focal_x = 156
        self.focal_y = 156

        # ROI offsets respect the original image
        self.x_roi_offset = 0
        self.y_roi_offset = 0

        if startup_check:
            self.startup_check()
        else:
            # @dataclass
            # class TRoi:
            #     final_xsize: int = 0
            #     final_ysize: int = 0
            #     xcenter: int = 0
            #     ycenter: int = 0
            #     xsize: int = 0
            #     ysize: int = 0
            # self.roi = TRoi()

            self.objects_read = []
            self.objects_write = []
            self.display = False

            # Hz
            self.cont = 1
            self.last_time = time.time()
            self.fps = 0

            # read test image to get sizes
            started_camera = False
            init_time = time.time()
            while not started_camera and (time.time() - init_time) < 10:
                try:
                    rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
                    # self.center_x = rgb.width // 2
                    # self.center_y = rgb.height // 2
                    print("Camera specs:")
                    print(" width:", rgb.width)
                    print(" height:", rgb.height)
                    print(" depth", rgb.depth)
                    print(" focalx", rgb.focalx)
                    print(" focaly", rgb.focaly)
                    print(" period", rgb.period)
                    print(" ratio {:.2f}.format(image.width/image.height)")

                    started_camera = True
                except Ice.Exception as e:
                    traceback.print_exc()
                    print(e, "Error connecting to camera. Trying again...")
                    time.sleep(1)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            print("Params: ", params)
            if params["depth_flag"] == "true" or params["depth_flag"] == "True":
                self.tracker = BYTETrackerDepth(frame_rate=30)
            else:
                self.tracker = BYTETracker(frame_rate=30)
            if params["display"] == "true" or params["display"] == "True":
                self.display = True
            if params["lidar3D"] == "true" or params["lidar3D"] == "True":
                self.lidar3D = True

        except:
            traceback.print_exc()
            print("Error reading config params")

        return True


    @QtCore.Slot()
    def compute(self):
        # read image
        if self.display:
            img = self.read_image()
            img = self.display_data(img, processed_data)
            if img is not None:
                cv2.imshow("Image", img)
                cv2.waitKey(1)

        self.show_fps()

    #########################################################################################################
    def read_image(self):
        # rgb = self.camera360rgb_proxy.getROI(self.center_x, self.center_y, 512, 430, 640, 640)
        rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
        rgb_frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))
        return rgb_frame

    def display_data(self, image, objects):
        # for i in self.lidar_in_image:
        #     # print(i)
        #     try:
        #         cv2.circle(image, (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)
        #     except:
        #         print("PUNTO NO VÁLIDO")
        if len(objects) == 0:
            return image
        for element in objects:
            final_xsize = element.roi.finalxsize
            final_ysize = element.roi.finalysize
            roi_xcenter = element.roi.xcenter
            roi_ycenter = element.roi.ycenter
            roi_xsize = element.roi.xsize
            roi_ysize = element.roi.ysize
            x_roi_offset = roi_xcenter-roi_xsize/2
            y_roi_offset = roi_ycenter-roi_ysize/2
            x_factor = roi_xsize / final_xsize
            y_factor = roi_ysize / final_ysize

            x0 = int(element.left*x_factor + x_roi_offset)
            y0 = int(element.top*y_factor + y_roi_offset)
            x1 = int(element.right*x_factor + x_roi_offset)
            y1 = int(element.bot*y_factor + y_roi_offset)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            resultados, min_distance = self.points_in_bbox(self.lidar_in_image, x0, y0, x1, y1)
            for i in resultados:
                cv2.circle(image, (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)
            # text = 'Class: {} - Score: {:.1f}% - ID: {}'.format(element.type, element.score*100, element.id)
            text = 'Class: {} - Score: {:.1f}% - ID: {} -Dist: {}'.format(element.type, element.score * 100, element.id,
                                                                          min_distance / 1000)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                (255, 0, 0),
                -1
            )
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, (0,255,0), thickness=1)
        return image

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

    ################################################################################################
    ### Methods related to external calls
    ################################################################################################

    def process_visual_objects(self, visual_objects):
        data = {"scores": [], "boxes": [], "clases": [], "roi": []}
        try:
            for object in visual_objects:
                data["scores"].append(object.score)
                data["boxes"].append([object.left, object.top, object.right, object.bot])
                data["clases"].append(object.type)
                data["roi"].append(object.roi)

            # get roi params from firs visual object since all are the same
            # if visual_objects:
            #     roi = visual_objects[0].roi
            #     self.final_xsize = roi.finalxsize
            #     self.final_ysize = roi.finalysize
            #     self.roi_xcenter = roi.xcenter
            #     self.roi_ycenter = roi.ycenter
            #     self.roi_xsize = roi.xsize
            #     self.roi_ysize = roi.ysize
            #     self.x_roi_offset = self.roi_xcenter-self.roi_xsize/2
            #     self.y_roi_offset = self.roi_ycenter-self.roi_ysize/2
            #     self.x_factor = self.roi_xsize / self.final_xsize
            #     self.y_factor = self.roi_ysize / self.final_ysize
            #     print(self.final_xsize, self.final_ysize)
            #     print(self.roi_xcenter, self.roi_ycenter)
            #     print(self.roi_xsize, self.roi_ysize)
            #     print(self.x_roi_offset, self.y_roi_offset)
            #     print(self.x_factor, self.y_factor)

            #Compute alpha start
            width_start = self.roi_xcenter - (self.roi_xsize//2)
            # width_final = self.roi_xcenter + (self.roi_xsize//2)
            start_angle = (self.calculate_start_angle(1024, width_start) + 360)*900/360
            # final_angle = self.calculate_end_angle(1024, width_final) + 900
            # print(final_angle-start_angle)
            #print(start_angle)

            if self.lidar3D:
                self.lidar_in_image = self.lidar_points(int(start_angle), int(abs(start_angle)*2))

        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error reading from Visual Objects interface")
        return data

    def to_visualelements_interface(self, tracks):
        self.objects_write = ifaces.RoboCompVisualElements.TObjects()
        for track in tracks:
            target = ifaces.RoboCompVisualElements.TObject()
            target.roi = track.roi
            target.id = int(track.track_id)
            target.score = track.score
            target.left = int(track.tlwh[0])
            target.top = int(track.tlwh[1])
            target.right = int((track.tlwh[0]+track.tlwh[2]))
            target.bot = int((track.tlwh[1]+track.tlwh[3]))
            target.type = track.clase

            self.objects_write.append(target)

        # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write
        return self.objects_read

    def calculate_start_angle(self, original_width, start_roi_angle):
        # La imagen original cubre 360 grados
        original_fov = 360
        # original_fov = 900

        # Calculamos la proporción de la posición de inicio con respecto al ancho original
        start_ratio = start_roi_angle / original_width

        # Calculamos el ángulo de inicio
        start_angle = original_fov * start_ratio - 180
        # start_angle = original_fov * start_ratio - 450

        return start_angle

    def calculate_end_angle(self, original_width, final_roi_angle):
        # La imagen original cubre 360 grados
        # original_fov = 360
        original_fov = 900

        # Calculamos la proporción de la posición de inicio con respecto al ancho original
        start_ratio = final_roi_angle / original_width

        # Calculamos el ángulo de inicio
        #start_angle = original_fov * start_ratio - 180
        start_angle = original_fov * start_ratio + 450

        return start_angle
    def lidar_points(self, alpha_roi, roi_width):
        points = self.lidar3d_proxy.getLidarData(alpha_roi, roi_width)
        lidar_points = np.array([[i.x*1000, i.y*1000, i.z*1000, 1] for i in points ])
        lidar_points = np.dot(lidar_points, self.lidar_to_cam.T)[:, :3]  # Eliminar la última columna
        lidar_in_image = np.column_stack([
            (self.focal_x * lidar_points[:, 0] / lidar_points[:, 1]) + 512,
            (-self.focal_y * lidar_points[:, 2] / lidar_points[:, 1]) + 256,
            np.linalg.norm(lidar_points, axis=1)
        ])
        return lidar_in_image

    def make_matrix_rt(self, roll, pitch, heading, x0, y0, z0):
        a = roll
        b = pitch
        g = heading
        mat = np.array([[np.cos(b) * np.cos(g),
                         (np.sin(a) * np.sin(b) * np.cos(g) + np.cos(a) * np.sin(g)), (np.sin(a) * np.sin(g) -
                                                                                       np.cos(a) * np.sin(b) * np.cos(
                        g)), x0],
                        [-np.cos(b) * np.sin(g), (np.cos(a) * np.cos(g) - np.sin(a) * np.sin(b) * np.sin(g)),
                         (np.sin(a) * np.cos(g) + np.cos(a) * np.sin(b) * np.sin(g)), y0],
                        [np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), z0],
                        [0, 0, 0, 1]])
        return mat

    def points_in_bbox(self, points, x1, y1, x2, y2):
        resultados = []
        min_distancia = float('inf')
        for punto in points:
            x, y, distancia = punto
            if x1 <= x <= x2 and y1 <= y <= y2:
                resultados.append(punto)
                if distancia < min_distancia:
                    min_distancia = distancia
        return resultados, min_distancia
    ##############################################################################################

    ##############################################################################################

    def startup_check(self):
        print(f"Testing RoboCompYoloObjects.TBox from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TBox()
        print(f"Testing RoboCompYoloObjects.TKeyPoint from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TKeyPoint()
        print(f"Testing RoboCompYoloObjects.TPerson from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TPerson()
        print(f"Testing RoboCompYoloObjects.TConnection from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TConnection()
        print(f"Testing RoboCompYoloObjects.TJointData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TJointData()
        print(f"Testing RoboCompYoloObjects.TData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TData()
        QTimer.singleShot(200, QApplication.instance().quit)

    ##############################################################################################
    # IMPLEMENTATION of getVisualObjects method from VisualElements interface
    ##############################################################################################

    def VisualElements_getVisualObjects(self):
        return self.objects_read

    ##############################################################################################
    # IMPLEMENTATION of allTargets method from ByteTrack interface
    ##############################################################################################

    def ByteTrack_allTargets(self):
        ret = ifaces.RoboCompByteTrack.OnlineTargets()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of getTargets method from ByteTrack interface
    #
    def ByteTrack_getTargets(self, objects):
        # Read visual elements from segmentator
        data = self.process_visual_objects(objects)
        # Get tracks from Bytetrack and convert data to VisualElements interface
        return self.to_visualelements_interface(self.tracker.update_original(np.array(data["scores"]),
                                                                                       np.array(data["boxes"]),
                                                                                       np.array(data["clases"]),
                                                                                       np.array(data["roi"])))
    #
    # IMPLEMENTATION of getTargetswithdepth method from ByteTrack interface
    #
    def ByteTrack_getTargetswithdepth(self, objects, depth):
        ret = ifaces.RoboCompByteTrack.OnlineTargets()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of setTargets method from ByteTrack interface
    #
    def ByteTrack_setTargets(self, objects, sender):
    
        #
        # write your CODE here
        #
        pass

    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompCamera360RGB you can call this methods:
    # self.camera360rgb_proxy.getROI(...)

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets
    #
    # def ByteTrack_getTargets(self, ps, pb, clases):
    #     ret = ifaces.RoboCompByteTrack.OnlineTargets()
    #     scores = np.array(ps)
    #     boxes = np.array(pb)
    #     clases = np.array(clases)
    #     for i in self.tracker.update_original(scores, boxes, clases):
    #         target = ifaces.RoboCompByteTrack.Targets()
    #         tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
    #         target.trackid = i.track_id
    #         target.score = i.score
    #         target.tlwh = tlwh
    #         target.clase = i.clase
    #         ret.append(target)
    #     return ret
    # #
    # # IMPLEMENTATION of getTargetswithdepth method from ByteTrack interface
    # #
    # def ByteTrack_getTargetswithdepth(self, ps, pb, depth, clases):
    #     ret = ifaces.RoboCompByteTrack.OnlineTargets()
    #     depth = np.frombuffer(depth.depth, dtype=np.float32).reshape(depth.height, depth.width, 1)
    #     scores = np.array(ps)
    #     boxes = np.array(pb)
    #     clases = np.array(clases)
    #     for i in self.tracker.update2(scores, boxes, depth, clases):
    #         target = ifaces.RoboCompByteTrack.Targets()
    #         tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
    #         target.trackid = i.track_id
    #         target.score = i.score
    #         target.tlwh = tlwh
    #         target.clase = i.clase
    #         ret.append(target)
    #     return ret
    # # ===================================================================
    # # ===================================================================
    #
    # def ByteTrack_setTargets(self, ps, pb, clases, sender):
    #
    #
    # def ByteTrack_allTargets(self):
    #     return self.read_tracks

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets


# def VisualElements_setVisualObjects(self, visualObjects, publisher):
    #     target_data_dict = {"scores" : [], "boxes" : [], "clases" : []}
    #     for object in visualObjects:
    #         target_data_dict["scores"].append(object.score)
    #         target_data_dict["boxes"].append([object.left, object.top, object.right, object.bot])
    #         target_data_dict["clases"].append(object.type)
    #     self.process_queue.put(target_data_dict)
    #     self.publisher = publisher
    #     if publisher == 0:
    #         self.image_height = 640
    #         self.image_width = 640
    #     elif publisher == 1:
    #         self.image_height = 384
    #         self.image_width = 384
    #