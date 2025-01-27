#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
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
import interfaces as ifaces
from difflib import SequenceMatcher

import cv2
import sys
import pyzed.sl as sl
import numpy as np
import math

from collections import deque

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

_COLORS = np.array(
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

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 16
        if startup_check:
            self.startup_check()
        else:

            self.initialize_zed_camera()
            self.bodies = sl.Bodies()
            self.image = sl.Mat()
            self.pose = sl.Pose()
            self.sensors = sl.SensorsData()

            self.click_image = np.zeros((int(720), int(1280), 3), dtype=np.uint8)
            # Set up the OpenCV window and mouse callback
            cv2.namedWindow("ZED Camera View")
            cv2.setMouseCallback("ZED Camera View", self.mouse_callback)

            self.selected_id = -1
            self.selected_body = None
            self.body_list_write, self.body_list_read = [], []

            # Control parameters
            self.max_vel = 900
            self.min_vel = -300
            self.stop_speed_margin = 1000
            self.max_distance = 1000
            self.rotation_gain = 1.6
            self.gaussian_x = 0.5
            self.gaussian_y = 0.5

            self.voice_commands = deque(maxlen=1)
            self.commands = ["sígueme", "espera"]


            self.timer.timeout.connect(self.compute)
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
            # Grab an image
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
                py_translation = sl.Translation()
                tx = round(self.pose.get_translation(py_translation).get()[0], 3)
                ty = round(self.pose.get_translation(py_translation).get()[1], 3)
                tz = round(self.pose.get_translation(py_translation).get()[2], 3)
                # print("Translation: tx: {0}, ty:  {1}, tz:  {2}, timestamp: {3}\n".format(tx, ty, tz, self.zed.get_timestamp))
                # Display orientation quaternion
                py_orientation = sl.Orientation()
                ox = round(self.pose.get_orientation(py_orientation).get()[0], 3)
                oy = round(self.pose.get_orientation(py_orientation).get()[1], 3)
                oz = round(self.pose.get_orientation(py_orientation).get()[2], 3)
                ow = round(self.pose.get_orientation(py_orientation).get()[3], 3)
                # cOnvert quaternion to euler angles
                roll, pitch, yaw = self.quaternion_to_euler(ox, oy, oz, ow)

                # print("Orientation: ox: {0}, oy:  {1}, oz: {2}\n".format(roll, pitch, yaw))

                self.zed.get_sensors_data(self.sensors, sl.TIME_REFERENCE.IMAGE)  # Retrieve only frame synchronized data

                # Extract IMU data
                imu_data = self.sensors.get_imu_data()

                # Retrieve linear acceleration and angular velocity
                linear_acceleration = imu_data.get_linear_acceleration()
                angular_velocity = imu_data.get_angular_velocity()
                # print("Linear acceleration: x: {0}, y:  {1}, z:  {2}\n".format(linear_acceleration[0], linear_acceleration[1], linear_acceleration[2]))
                # print("Angular velocity: x: {0}, y:  {1}, z:  {2}\n".format(angular_velocity[0], angular_velocity[1], angular_velocity[2]))

                # Retrieve left image
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU)

                # Copy image and reduce size
                image_ocv = self.image.get_data()
                # Retrieve bodies
                self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
                self.body_list_read = self.bodies.body_list
                self.body_list_read.sort(key=lambda x: np.linalg.norm(x.position))
                # Check if there are voice commands
                if len(self.voice_commands) > 0:
                    command = self.voice_commands.popleft()
                    self.check_voice_commands(command)

                # Update self.selected_body if the selected ID is valid
                for body in self.body_list_read:
                    if body.id == self.selected_id:
                        self.selected_body = body
                        break
                if self.selected_body is not None:
                    if self.selected_id == -1 or self.selected_body.tracking_state == sl.OBJECT_TRACKING_STATE.TERMINATE:
                        # self.omnirobot_proxy.setSpeedBase(0, 0, 0)
                        self.selected_body = None


                # self.body_to_visual_element_pub(self.selected_body)
                self.send_control_commands(self.selected_body)


                self.display_data_tracks(image_ocv, self.bodies.body_list)

        except KeyboardInterrupt:
            self.event.set()

    def quaternion_to_euler(self, x, y, z, w):
        """
        Converts a quaternion to Euler angles (roll, pitch, yaw).

        Parameters:
            x (float): The x component of the quaternion.
            y (float): The y component of the quaternion.
            z (float): The z component of the quaternion.
            w (float): The w (scalar) component of the quaternion.

        Returns:
            tuple: A tuple of Euler angles (roll, pitch, yaw) in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def initialize_zed_camera(self):
        # Create a Camera object
        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
        init_params.camera_fps = 60   
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Set coordinate units
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Enable Positional tracking (mandatory for object detection)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances
        # positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        body_param = sl.BodyTrackingParameters()
        body_param.enable_tracking = True  # Track people across images flow
        body_param.enable_body_fitting = False  # Smooth skeleton move
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use

        # Enable Object Detection module
        self.zed.enable_body_tracking(body_param)

        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 40

        # # Get ZED camera information
        # camera_info = zed.get_camera_information()
        # # 2D viewer utilities
        # display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
        #                                    min(camera_info.camera_configuration.resolution.height, 720))
        # image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
        #     , display_resolution.height / camera_info.camera_configuration.resolution.height]

    def mouse_callback(self, event, x, y, flags, param):
        """
        Handles mouse events.
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            self.selected_id = self.get_bbox_id(x, y, self.body_list_read)
            print(f"Selected ID: {self.selected_id}")
            # update_display()

    def get_bbox_id(self, x, y, bodies):
        """
        Checks if the click is inside any bounding box.
        Returns the ID of the bounding box or None.
        """
        for body in bodies:
            bx, by, bw, bh = map(int, [body.bounding_box_2d[0][0], body.bounding_box_2d[0][1], body.bounding_box_2d[2][0], body.bounding_box_2d[2][1]])
            if bx <= x <= bw and by <= y <= bh:
                return body.id
        return -1

    def body_to_visual_element_pub(self, body):
        """
        Convert a Body object to a VisualElement object.
        """
        visual_element = ifaces.RoboCompVisualElementsPub.TObject()
        if body is None:
            visual_element.id = -1
            self.segmentatortrackingpub_proxy.setTrack(visual_element)
            return

        visual_element.id = body.id
        visual_element.type = 0
        generic_attrs = {
            "x_pos": str(round(body.position[0], 2)),
            "y_pos": str(round(-body.position[2], 2)),
        }
        visual_element.attributes = generic_attrs
        print(generic_attrs)
        self.segmentatortrackingpub_proxy.setTrack(visual_element)

        return visual_element
    
    def send_control_commands(self, body):
        if body is None:
            try:
                self.omnirobot_proxy.setSpeedBase(0, 0, 0)
            except Exception as e:
                print(f"Error sending control commands: {e}")
            return
        if body.tracking_state == sl.OBJECT_TRACKING_STATE.SEARCHING:
            print("Body lost")
            try:
                self.omnirobot_proxy.setSpeedBase(0, 0, 0)
            except Exception as e:
                print(f"Error sending control commands: {e}")
            return
        # Get the 3D position of the body
        body_position = body.position
        # Get target at self.max_distance millimeters before the body in the line between the robot and the body
        target = body_position - self.max_distance * body_position / np.linalg.norm(body_position)

        # Get distance to the body
        distance = np.linalg.norm(body_position) if np.linalg.norm(body_position) > self.max_distance else 0
        # Get angle to body
        angle = np.arctan2(body_position[0], -body_position[2])
        # print(f"Distance: {distance} - Angle: {angle}")

        rotation_vel = np.clip(angle * self.rotation_gain, -2, 2)
        advance_vel = np.clip(self.max_vel * self.gaussian(rotation_vel) * (distance  / (self.max_distance + self.stop_speed_margin)), self.min_vel, self.max_vel)
        print(f"Advance vel: {advance_vel} - Rotation vel: {rotation_vel}")
        try:
            # plan = ifaces.RoboCompGridPlanner.TPlan(valid=True, controls=ifaces.RoboCompGridPlanner.Control([ifaces.RoboCompGridPlanner.TControl(adv=advance_vel, rot=rotation_vel)]))
            # self.gridplanner_proxy.setPlan(plan)

            self.omnirobot_proxy.setSpeedBase(0, advance_vel, rotation_vel)
        except Exception as e:
            print(f"Error sending control commands: {e}")


    def gaussian(self, x):
        """
        Compute the value of a Gaussian function.

        Parameters:
            x (float): The input value.
            xset (float): The parameter controlling the width of the Gaussian.
            yset (float): The parameter controlling the height of the Gaussian.

        Returns:
            float: The computed Gaussian value.
        """
        s = -self.gaussian_x * self.gaussian_x / math.log(self.gaussian_y)
        return math.exp(-x * x / s)

    def display_data_tracks(self, img, elements): #Optimizado
        """
        This function overlays bounding boxes and object information on the image for tracked objects.

        Args:
            img (numpy array): The image to display object data on.
            elements (list): Tracked objects with bounding box coordinates, scores, and class indices.
            class_names (list, optional): Names of the classes.

        Returns:
            img (numpy array): The image with overlaid object data.
        """

        img = img.astype(np.uint8)
        for element in elements:
            x0, y0, x1, y1 = map(int, [element.bounding_box_2d[0][0], element.bounding_box_2d[0][1], element.bounding_box_2d[2][0], element.bounding_box_2d[2][1]])
            cls_ind = 0
            if element.id == self.selected_id:
                color = (255, 255, 255)
            else:
                color = (_COLORS[cls_ind] * 255).astype(np.uint8).tolist()
            # text = f'Class: {class_names[cls_ind]} - Score: {element.score * 100:.1f}% - ID: {element.id}'
            text = f'{round(element.position[0], 0)} - {round(-element.position[2], 0)} - {element.id}'
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_ind]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

            # Check if the img array is read-only
            if not img.flags.writeable:
                # Create a writable copy of the img array
                img = img.copy()

            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[cls_ind] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.imshow("ZED Camera View", img)
        cv2.waitKey(1)

    def calculate_asr_confidence(self, asr_text, keywords):
        """
        Calcula el grado de confianza de las coincidencias entre las palabras clave y el texto de ASR.

        :param asr_text: Cadena generada por el sistema ASR.
        :param keywords: Lista de palabras clave a buscar en el texto.
        :return: Diccionario con las palabras clave y su grado de confianza.
        """
        confidences = {}
        asr_text_lower = asr_text.lower()  # Convertir el texto a minúsculas para comparación insensible a mayúsculas.

        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Usamos SequenceMatcher para calcular la similitud.
            similarity = SequenceMatcher(None, asr_text_lower, keyword_lower).ratio()
            confidences[keyword] = similarity

        return confidences

    def check_voice_commands(self, command):
        # Check if any word in follow_commands is in the command
        follow_word_confidences = self.calculate_asr_confidence(command, self.commands)
        print(f"Follow word confidences: {follow_word_confidences}")
        if follow_word_confidences["sígueme"] > 0.5:
            print("Follow me command received")
            if len(self.body_list_read) > 0:
                self.selected_id = self.body_list_read[0].id
                self.selected_body = self.body_list_read[0]
                self.speech_proxy.say("Okey, estoy siguiéndote", False)
            else:
                self.speech_proxy.say("No tengo a nadie a quién seguir", False)
        elif follow_word_confidences["espera"] > 0.5:
            print("Stop command received")
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
            self.selected_body = None
            self.selected_id = -1
            self.speech_proxy.say("Vale, dejo de seguirte", False)


    def startup_check(self):
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to OnMessageTranscribed method from WhisperStream interface
    #
    def WhisperStream_OnMessageTranscribed(self, message):
        print(f"Received message: {message}")
        self.voice_commands.append(message)



    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompGridPlanner you can call this methods:
    # self.gridplanner_proxy.modifyPlan(...)
    # self.gridplanner_proxy.setPlan(...)

    ######################
    # From the RoboCompGridPlanner you can use this types:
    # RoboCompGridPlanner.TPoint
    # RoboCompGridPlanner.TControl
    # RoboCompGridPlanner.TPlan

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
    # From the RoboCompSpeech you can call this methods:
    # self.speech_proxy.isBusy(...)
    # self.speech_proxy.say(...)

    ######################
    # From the RoboCompSegmentatorTrackingPub you can publish calling this methods:
    # self.segmentatortrackingpub_proxy.setTrack(...)


