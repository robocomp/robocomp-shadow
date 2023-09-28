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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
from threading import Thread
import interfaces as ifaces
# Import other dependencies
import numpy as np
import sys
import cv2
import queue
from flask import Flask, Response, render_template, request, jsonify
import time
import threading
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
        self.Period = 33
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        # Create queues and locks
        self.frame_queue = queue.Queue()
        self.frame_lock = threading.Lock()
        self.MAX_QUEUE_SIZE = 2  # Adjust as necessary
        self.box_queue = queue.Queue()
        self.box_lock = threading.Lock()
        self.frame_box_queue = queue.Queue()  # This queue contains tuples of (frame, boxes)
        self.frame_box_lock = threading.Lock()
        self.bbox_queue = queue.Queue()
        self.bbox_lock = threading.Lock()
        self.cont = 0
        self.last_time = time.time()
        self.target = -1
        self.fps = 0
        # Iniciar el servidor Flask en un nuevo hilo
        self.app = Flask(__name__)

        @self.app.route('/')
        def index():
            width = request.args.get('width')
            height = request.args.get('height')
            return render_template('index.html', width=width, height=height)

        @self.app.route('/video_feed/<int:width>/<int:height>')
        def video_feed(width, height):
            resolution = (width, height)
            return Response(self.generate(resolution), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/get_boxes')
        def get_boxes():
            try:
                with self.bbox_lock:
                    boxes = self.bbox_queue.get_nowait()
                    # print(boxes)
            except queue.Empty:
                boxes = []
            return jsonify(boxes)

        @self.app.route('/select_person', methods=['POST'])
        def select_person():
            if not request.is_json:
                return jsonify({'status': 'failure', 'reason': 'Expected JSON'}), 400

            box = request.json.get('box')
            if not box:
                # print("ENTRAAAAA3")
                # print("NO TARGET")
                self.target = -1
                # self.segmentatortrackingpub_proxy.setTrack(target)
                return jsonify({'status': 'failure', 'reason': 'No box data'}), 400

            for b in self.boxes:
                # print("id box", b.id)
                # print("id BOXES", str(box.get('id')))
                if b.id == box.get('id'):
                    self.target = b.id
                    # print("HAY TARGET", target)

            # try:
            #     if target:
            #         self.segmentatortrackingpub_proxy.setTrack((target))
            #     else:
            #         print("NO TARGET")
            #         target = ifaces.RoboCompVisualElements.TObject()
            #         target.id = -1
            #         self.segmentatortrackingpub_proxy.setTrack(target)

            # except Ice.Exception as e:
            #     traceback.print_exc()
            #     print(e)
            #     return jsonify({'status': 'failure', 'reason': "str(e)"}), 500

            return jsonify({'status': 'success'})
        self.flask_thread = Thread(target=self.app.run, kwargs={'host': '192.168.50.249', 'port': 5000}) # '192.168.50.153' orin ip
        self.flask_thread.start()

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
        t1 = time.time()
        image = None

        try:
            t2 = time.time()
            image = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
        except:
            print("Camara no responde")

        try:
            t3 = time.time()
            self.boxes = self.visualelements_proxy.getVisualObjects([])
            boxes = self.adapt_bbox(self.boxes, image)
        except:
            print("No Objects")

        for b in self.boxes:
            # print("id box", b.id)
            # print("id BOXES", str(box.get('id')))
            if b.id == self.target:
                target = b
                # print("HAY TARGET", target)

        try:
            if self.target != -1:
                self.segmentatortrackingpub_proxy.setTrack((target))
            else:
                print("NO TARGET")
                target = ifaces.RoboCompVisualElements.TObject()
                target.id = -1
                self.segmentatortrackingpub_proxy.setTrack(target)

        except Ice.Exception as e:
            traceback.print_exc()
            print(e)

        t4 = time.time()
        self.create_queues(image, boxes)
        #print("TIEMPO", (time.time()-t1) *1000)

        return True


    def adapt_bbox(self, objects, image):
        boxes = []
        for obj in objects:
            # Obten la informacion de la ROI
            
            roi = obj.image.roi
            final_xsize = roi.finalxsize
            final_ysize = roi.finalysize
            roi_xcenter = roi.xcenter
            roi_ycenter = roi.ycenter
            roi_xsize = roi.xsize
            roi_ysize = roi.ysize
            # Calcula el factor de escala y offset
            x_roi_offset = roi_xcenter - roi_xsize / 2
            y_roi_offset = roi_ycenter - roi_ysize / 2
            x_factor = roi_xsize / final_xsize
            y_factor = roi_ysize / final_ysize

            # Redimensiona las coordenadas del bounding box
            left = int(obj.left * x_factor + x_roi_offset) % image.width
            right = int(obj.right * x_factor + x_roi_offset) % image.width
            top = int(obj.top * y_factor + y_roi_offset)
            bot = int(obj.bot * y_factor + y_roi_offset)

            # Crea una nueva caja redimensionada y añádela a la lista
            new_box = {'x': left, 'y': top, 'width': right - left, 'height': bot - top, 'id': obj.id}
            boxes.append(new_box)
        return boxes

    def create_queues(self, image, boxes):
        with self.frame_box_lock:
            if self.frame_box_queue.qsize() < self.MAX_QUEUE_SIZE:
                frame = np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3))
                self.frame_box_queue.put((frame, boxes))
            else:
                # descarta la imagen más antigua si la cola está demasiado llena
                self.frame_box_queue.get()
                frame = np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3))
                self.frame_box_queue.put((frame, boxes))

    def generate(self, resolution):
        while True:
            try:
                boxes_rescale = []
                with self.frame_box_lock:
                    frame, boxes = self.frame_box_queue.get_nowait()

                # Obtener el tamaño original de la imagen
                original_height, original_width = frame.shape[:2]

                # Redimensionar el marco a la resolución deseada
                frame = cv2.resize(frame, resolution)

                for box in boxes:
                    # Redimensionar las coordenadas del bounding box
                    x = int(box['x'] * resolution[0] / original_width)
                    y = int(box['y'] * resolution[1] / original_height)
                    width = int(box['width'] * resolution[0] / original_width)
                    height = int(box['height'] * resolution[1] / original_height)
                    # Redimensiona solo a 640 (Sergio)
                    # box['x'] = int(box['x'] * 640 / original_width)
                    # box['y'] = int(box['y'] * 640 / original_height)
                    # box['width'] = int(box['width'] * 640 / original_width)
                    # box['height'] = int(box['height'] * 640 / original_height)

                    # Redimensiona a la resolución especificada
                    box['x'] = int(box['x'] * resolution[0] / original_width)
                    box['y'] = int(box['y'] * resolution[1] / original_height)
                    box['width'] = int(box['width'] * resolution[0] / original_width)
                    box['height'] = int(box['height'] * resolution[1] / original_height)

                    boxes_rescale.append(box)

                    # Dibujar el bounding box en la imagen redimensionada
                    frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                with self.bbox_lock:
                    if self.bbox_queue.qsize() < self.MAX_QUEUE_SIZE:
                        self.bbox_queue.put(boxes_rescale)
                    else:
                        # descarta la imagen más antigua si la cola está demasiado llena
                        self.bbox_queue.get()

                        self.bbox_queue.put(boxes_rescale)
            except queue.Empty:
                continue

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

    def startup_check(self):
        print(f"Testing RoboCompCamera360RGB.TRoi from ifaces.RoboCompCamera360RGB")
        test = ifaces.RoboCompCamera360RGB.TRoi()
        print(f"Testing RoboCompCamera360RGB.TImage from ifaces.RoboCompCamera360RGB")
        test = ifaces.RoboCompCamera360RGB.TImage()
        print(f"Testing RoboCompMPC.Point from ifaces.RoboCompMPC")
        test = ifaces.RoboCompMPC.Point()
        print(f"Testing RoboCompMPC.Control from ifaces.RoboCompMPC")
        test = ifaces.RoboCompMPC.Control()
        print(f"Testing RoboCompMaskElements.TRoi from ifaces.RoboCompMaskElements")
        test = ifaces.RoboCompMaskElements.TRoi()
        print(f"Testing RoboCompMaskElements.TMask from ifaces.RoboCompMaskElements")
        test = ifaces.RoboCompMaskElements.TMask()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompVisualElements.TRoi from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TRoi()
        print(f"Testing RoboCompVisualElements.TObject from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TObject()
        QTimer.singleShot(200, QApplication.instance().quit)

    ######################
    # From the RoboCompCamera360RGB you can call this methods:
    # self.camera360rgb_proxy.getROI(...)

    ######################
    # From the RoboCompCamera360RGB you can use this types:
    # RoboCompCamera360RGB.TRoi
    # RoboCompCamera360RGB.TImage

    ######################
    # From the RoboCompMPC you can call this methods:
    # self.mpc_proxy.newPath(...)

    ######################
    # From the RoboCompMPC you can use this types:
    # RoboCompMPC.Point
    # RoboCompMPC.Control

    ######################
    # From the RoboCompMaskElements you can call this methods:
    # self.maskelements_proxy.getMasks(...)
    # self.maskelements_proxy.getNamesofCategories(...)

    ######################
    # From the RoboCompMaskElements you can use this types:
    # RoboCompMaskElements.TRoi
    # RoboCompMaskElements.TMask

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # self.omnirobot_proxy.correctOdometer(...)
    # self.omnirobot_proxy.getBasePose(...)
    # self.omnirobot_proxy.getBaseState(...)
    # self.omnirobot_proxy.resetOdometer(...)
    # self.omnirobot_proxy.setOdometer(...)
    # self.omnirobot_proxy.setOdometerPose(...)
    # self.omnirobot_proxy.setSpeedBase(...)
    # self.omnirobot_proxy.stopBase(...)

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompVisualElements you can call this methods:
    # self.visualelements_proxy.getVisualObjects(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

    ######################
    # From the RoboCompVisualElements you can call this methods:
    # self.visualelements1_proxy.getVisualObjects(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

    ######################
    # From the RoboCompSegmentatorTrackingPub you can publish calling this methods:
    # self.segmentatortrackingpub_proxy.setTrack(...)