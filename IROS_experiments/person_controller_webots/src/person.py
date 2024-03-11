import os
import sys
sys.path.append('/usr/local/webots/lib/controller/python')
from controller import Supervisor, Robot, Camera
import ctypes
import numpy as np
import math
import time
import cv2


class Person(Supervisor):
    def __init__(self, person_number):
        parent = os.path.dirname(os.path.realpath(__file__))
        main_directory = os.path.abspath(os.path.join(parent, os.pardir))
        print(main_directory)
        Robot.__init__(self)
        self.person_number = person_number
        self.timeStep = int(self.getBasicTimeStep())
        self.controller_functions = ctypes.CDLL('/usr/local/webots/lib/controller/libController.so',
                                                mode=ctypes.RTLD_GLOBAL)
        self.bvh_functions = ctypes.CDLL('/usr/local/webots/projects/humans/skin_animated_humans/libraries/bvh_util/libbvh_util.so', mode=ctypes.RTLD_GLOBAL)
        self.bvh_animation_functions = ctypes.CDLL( main_directory + '/bvh_animation_mod.so')
        self.bvh_animation_functions.load_motion_data(person_number)

        self.moving = False

        self.person_node = self.getFromDef("HUMAN_" + str(person_number))
        self.person_camera = self.getDevice("person_camera")
        self.person_camera.enable(33)
        # controller_field = self.person_node.getField('controller')
        # controller_field.setSFString("<extern>")

        self.traslation_field = self.person_node.getField('translation')
        self.start_pose = self.traslation_field.getSFVec3f()
        self.rotation_field = self.person_node.getField('rotation')
        self.start_rotation = self.rotation_field.getSFRotation()

        self.last_time_sent_velocity = time.time()
        self.time_between_velocity_commands = 0.1

    def set_speed(self, data):
        ########## In case the receibed speed is in the person frame ##########
        # if time.time() - self.last_time_sent_velocity > self.time_between_velocity_commands:
        #     self.last_time_sent_velocity = time.time()
        person_orientation = self.person_node.getOrientation()
        orientation = math.atan2(person_orientation[0], person_orientation[1]) - math.pi / 2
        rotation_matrix = np.array(
            [[math.cos(orientation), -math.sin(orientation)], [math.sin(orientation), math.cos(orientation)]])
        lin_speed = np.array([data.axes[1].value * 2, 0])
        # if data.axes[1] > 0.1 or data.axes[1] < -0.1:
        #     self.moving = True
        # else:
        #     self.moving = False
        converted_speed = np.matmul(rotation_matrix, lin_speed)
        self.person_node.setVelocity(
            [converted_speed[0] / 2.5, converted_speed[1] / 2.5, 0, 0, 0, -data.axes[0].value * math.pi / 2])

    def set_initial_pose(self, data):
        if data.buttons[3]:
            self.traslation_field.setSFVec3f(self.start_pose)
            self.rotation_field.setSFRotation(self.start_rotation)

    def get_camera_image(self):
        color = self.person_camera.getImage()
        color_image = cv2.cvtColor(cv2.cvtColor(
            np.frombuffer(color, np.uint8).reshape(self.person_camera.getHeight(), self.person_camera.getWidth(), 4),
            cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB)
        cv2.imshow("color", color_image)
        cv2.waitKey(1)