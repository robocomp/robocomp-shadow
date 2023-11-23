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

        self.Period = 50

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

        self.act_roi = None

        # Iniciar el servidor Flask en un nuevo hilo
        self.app = Flask(__name__)

        self.labels = []
        self._COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)

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
                self.target = -1
                return jsonify({'status': 'failure', 'reason': 'No box data'}), 400

            for b in self.boxes:
                if b.id == box.get('id'):
                    self.target = b.id
            print("########### CHOSEN ID :", self.target)
            return jsonify({'status': 'success'})
        
        #                                                                Caja-blanca IP
        self.flask_thread = Thread(target=self.app.run, kwargs={'host': '192.168.50.240', 'port': 5000}) # '192.168.50.153' orin ip
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
        image = None
        try:
            image = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
        except:
            print("Camara no responde")

        try:
            self.labels = []
            self.visual_elements = self.visualelements_proxy.getVisualObjects()
            self.boxes = self.visual_elements.objects
            boxes = self.adapt_bbox(self.boxes, image)
        except:
            print("No Objects")

        target = ifaces.RoboCompVisualElements.TObject()
        target.id = self.target
        for b in self.boxes:
            if b.id == self.target:
                target = b
        try:
            self.segmentatortrackingpub_proxy.setTrack(target)

        except Ice.Exception as e:
            traceback.print_exc()
            print(e)

        self.create_queues(image, boxes)

        return True


    def adapt_bbox(self, objects, image):
        boxes = []
        labels = []
        for i, obj in enumerate(objects):
            # Obten la informacion de la ROI
            if i == 0:
                self.act_roi = obj.image.roi
            roi = obj.image.roi
            final_xsize = roi.finalxsize
            final_ysize = roi.finalysize
            roi_xcenter = roi.xcenter
            roi_ycenter = roi.ycenter
            roi_xsize = roi.xsize
            roi_ysize = roi.ysize
            print(obj.metrics)

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
            self.labels.append({'id': obj.id, 'orientation': obj.person.orientation, 'x': round(obj.x, 0), 'y': round(obj.y, 0), 'z': round(obj.z, 0),
                           'top': top, 'bot': bot, 'right': right, 'left': left, 'type': obj.type})

        return boxes

    def create_queues(self, image, boxes):
        with self.frame_box_lock:
            if self.frame_box_queue.qsize() < self.MAX_QUEUE_SIZE:
                frame = np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3))
                self.frame_box_queue.put((frame, boxes))
            else:
                # Discard the oldest image if the queue is full
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

                    # Redimensiona a la resolución especificada
                    box['x'] = int(box['x'] * resolution[0] / original_width)
                    box['y'] = int(box['y'] * resolution[1] / original_height)
                    box['width'] = int(box['width'] * resolution[0] / original_width)
                    box['height'] = int(box['height'] * resolution[1] / original_height)

                    boxes_rescale.append(box)

                    # Dibujar el bounding box en la imagen redimensionada
                    # frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                frame = self.display_data_tracks(frame, [], resolution, original_height, original_width)
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

    def display_data_tracks(self, img, elements, resolution, original_height, original_width):  # Optimizado
        """
        This function overlays bounding boxes and object information on the image for tracked objects.

        Args:
            img (numpy array): The image to display object data on.
            elements (list): Tracked objects with bounding box coordinates, scores, and class indices.
            class_names (list, optional): Names of the classes.

        Returns:
            img (numpy array): The image with overlaid object data.
        """
        reescale_x = resolution[0] / original_width
        reescale_y = resolution[1] / original_height
        # print(resolution[0],original_width)
        # print(resolution[1],original_height)
        # print("reescale",reescale_x,reescale_y)

        if self.act_roi != None:
            left = round((self.act_roi.xcenter - (self.act_roi.xsize // 2))*reescale_x)
            right = round((self.act_roi.xcenter + (self.act_roi.xsize // 2))*reescale_x)
            top = round((self.act_roi.ycenter - (self.act_roi.ysize // 2))*reescale_y)
            bot = round((self.act_roi.ycenter + (self.act_roi.ysize // 2))*reescale_y)
            print(left,right,top,bot)
            cv2.rectangle(img, (left, top), (right, bot), (255, 0, 0), 2)

        for label in self.labels:
            # x0, y0, x1, y1 = map(int, [element.left, element.top, element.right, element.bot])
            x0, y0, x1, y1 = label['left'], label['top'], label['right'], label['bot']

            x0 = round(x0 * reescale_x)
            y0 = round(y0 * reescale_y)
            x1 = round(x1 * reescale_x)
            y1 = round(y1 * reescale_y)
            
            # print("x0,y0,x1,y1",x0,y0,x1,y1)

            cls_ind = label['type']
            color = (self._COLORS[cls_ind] * 255).astype(np.uint8).tolist()
            # text = f'Class: {class_names[cls_ind]} - Score: {element.score * 100:.1f}% - ID: {element.id}'

            element_x = label['x']
            element_y = label['y']
            element_z = label['z']
            element_id = label['id']
            element_orientation = round(label['orientation'],2)

            text = f'{element_x} - {element_y} - {element_z} - {element_id} - {element_orientation}'
            txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_ind]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

            # Show the humans boundin boxes
            txt_bk_color = (self._COLORS[cls_ind] * 255 * 0.7).astype(np.uint8).tolist()

            # Change the target bounding box color
            if self.target == element_id:
                color = txt_bk_color = (49,86,28) # Dark green
            
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

    def show_fps(self):
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000./self.cont)
            print("Freq:", self.cont, "ms. Curr period:", cur_period)
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
