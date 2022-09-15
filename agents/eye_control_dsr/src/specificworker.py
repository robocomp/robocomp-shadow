#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 by YOUR NAME HERE
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
from PySide2.QtGui import QImage
from rich.console import Console
from genericworker import *
import time
import interfaces as ifaces
import numpy as np
import cv2, json
sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
import matplotlib.pyplot as plt
from pydsr import *
import math
from scipy.stats import norm
from collections import deque



# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

MIN_VELOCITY = 0.05
TOLERANCE = 0.02

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.last_timestamp = 0
        self.last_person_position = 240
        self.Period = 100

        QObject.connect(self.ui.horizontalSlider_pos, QtCore.SIGNAL('valueChanged(int)'), self.slot_change_pos)
        QObject.connect(self.ui.horizontalSlider_max_speed, QtCore.SIGNAL('valueChanged(int)'), self.slot_change_max_speed)
        QObject.connect(self.ui.pushButton_center, QtCore.SIGNAL('clicked()'), self.slot_center)
        QObject.connect(self.ui.pushButton, QtCore.SIGNAL('clicked()'), self.slot_track)
        self.motor = ifaces.RoboCompJointMotorSimple.MotorState()
        self.last_goal = ifaces.RoboCompJointMotorSimple.MotorGoalPosition()
        self.last_goal.maxSpeed = 0
        self.last_goal.pos = 0
        self.last_puntoMedioX = 0
        self.current_max_speed = 0.0
        self.ui.horizontalSlider_pos.setSliderPosition(self.motor.pos)
        self.rad_old = self.motor.pos

        self.faceList = ["0", "1", "2", "3", "4"]
        self.hipList = ["12", "13"]
        self.chestList = ["6", "7"]

        self.k1 = 0.8
        self.k2 = 0.6
        self.k3 = 1
        self.k4 = 0.4
        self.k5 = 4
        self.k6 = 1
        self.k7 = 1

        self.error_ant = 0
        self.rad_old = 0
        self.last_motor_pos = 0

        self.last_person_azimut = 0

        self.integral_error = 0
        self.max_integral_error_value = 5

        self.distance_avg = 0
        self.distance_limit = 1000

        self.rotational_speed_coefficients=[0,0,0]
        self.rotational_speed_avg=0

        self.lineal_speed_coefficients=[0,0,0]
        self.lineal_speed_avg = 0

        self.act_chased_person = None
        self.intention_id = None

        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

        self.sacadic = False

        # plt.ion()
        # self.visible = 120
        # self.d_camera_position_error = deque(np.zeros(self.visible), self.visible)
        # self.d_camera_audio_angle_error = deque(np.zeros(self.visible), self.visible)
        # self.dx = deque(np.zeros(self.visible), self.visible)
        # self.data_length = np.linspace(0, 121, num=120)
        # self.fig, self.ax = plt.subplots()
        # self.fig, (self.ah1, self.ah2, self.ax) = plt.subplots(3)
        # self.ah1 = self.fig.add_subplot()
        # self.ax = self.fig.add_subplot()
        # plt.margins(x=0.01)
        # self.ah1.set_xlabel("Camera position error", fontsize=14)
        # self.ah2.set_xlabel("Camera audio angle error", fontsize=14)
        # self.ax.set_ylabel("y", fontsize=14)
        # self.ax.set_xlabel("x", fontsize=14)
        # self.ax.axis([-3, 3, -3, 3])

        self.needed_joints = ["0", "1", "2", "3", "4", "5", "6", "11", "12", "17"]

        self.past_pos = np.array([0, 0, 0])

        # self.camera_position_error, = self.ah1.plot(self.dx, self.d_camera_position_error, color='green', label="Closing (x10)", linewidth=1.0)
        # self.camera_audio_angle_error, = self.ah2.plot(self.dx, self.d_camera_audio_angle_error, color='green', label="Closing (x10)", linewidth=1.0)

        self.x_data = 0

        # Dictionary for error saving

        # self.error_dict = {
        #     "camera_position_error" : [],
        #     "camera_audio_angle_error" : [],
        # }


        with open('human_pose.json', 'r') as f:
            self.human_pose = json.load(f)

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 911
        self.g = DSRGraph(0, "eye_control", self.agent_id)
        self.rt_api = rt_api(self.g)
        self.inner_api = inner_api(self.g)

        try:
            # signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        if startup_check:
            self.startup_check()
        else:
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
        #obtenemos datos

        camera_node = self.g.get_node('giraff_camera_realsense')
        image_data = camera_node.attrs['cam_rgb'].value
        image_width = camera_node.attrs['cam_rgb_width'].value
        image_height = camera_node.attrs['cam_rgb_height'].value
        image = np.frombuffer(image_data, np.uint8).reshape(image_height, image_width, 3)
        self.refesco_ventana(image_width, image_height, image)
        # self.plot_data()


    def draw_error_data(self):
        self.d_camera_position_error.extend([self.error_dict["camera_position_error"][-1]])
        self.d_camera_audio_angle_error.extend([self.error_dict["camera_audio_angle_error"][-1]])
        self.dx.extend([self.x_data])
        self.camera_position_error.set_ydata(self.d_camera_position_error)
        self.camera_position_error.set_xdata(self.dx)
        self.camera_audio_angle_error.set_ydata(self.d_camera_audio_angle_error)
        self.camera_audio_angle_error.set_xdata(self.dx)

        # set axes
        self.ah1.set_ylim(0, 360)
        self.ah1.set_xlim(self.x_data-self.visible, self.x_data)
        self.ah2.set_ylim(0, 360)
        self.ah2.set_xlim(self.x_data-self.visible, self.x_data)

        self.x_data += 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def tracker_camera(self, data):
        goal = ifaces.RoboCompJointMotorSimple.MotorGoalPosition()
        if abs(data["error_rads"]) > 0.1:
            self.sacadic = True
            print("SACADIC IN")
            goal.maxSpeed = 0
        else:
            # self.sacadic = False
            print("SACADIC OUT")
            goal.maxSpeed = abs(abs((((data["error_rads"] + MIN_VELOCITY) * 2.5)) * (1.5/(data["distance"]/1000))) - 0.3 * abs(data["act_rot_speed"]))
            # goal.maxSpeed = 0

        goal.position = data["goal_rads"]

        return goal

    def obtencion_datos(self):
        try:
            servo_node = self.g.get_node("servo")
            servo_pos = servo_node.attrs["servo_pos"].value
            servo_speed = servo_node.attrs["servo_speed"].value
            servo_isMoving = servo_node.attrs["servo_moving"].value
            self.motor.pos = servo_pos
            self.motor.vel = servo_speed
            self.motor.isMoving = servo_isMoving
        except:
            print("CAN'T GET SERVO DATA")

    def refesco_ventana(self, height, width, image):
        qt_image = QImage(image, height, width, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image).scaled(self.ui.label_image.width(), self.ui.label_image.height())
        self.ui.label_image.setPixmap(pix)
        # image = np.frombuffer(color.image, np.uint8).reshape(color.height, color.width, color.depth)

        # self.ui.lcdNumber_pos.display(self.motor.pos.value)
        # self.ui.lcdNumber_speed.display(self.motor.vel.value)
        # self.ui.lcdNumber_temp.display(self.motor.temperature)
        # self.ui.lcdNumber_max_speed.display(self.current_max_speed)
        # if self.motor.isMoving.value:
        #     self.ui.radioButton_moving.setChecked(True)
        # else:
        #     self.ui.radioButton_moving.setChecked(False)

    def plot_data(self):
        self.ax.cla()
        self.ax.axis([-3, 3, -3, 3])
        people_nodes = self.g.get_nodes_by_type('person')
        # world_node = self.g.get_node('world')
        robot_node = self.g.get_node('robot')
        # robot_edge_rt = self.rt_api.get_edge_RT(world_node, robot_node.id)
        # robot_tx, robot_ty, robot_tz = robot_edge_rt.attrs['rt_translation'].value
        # robot_rx, robot_ry, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
        # robot_rz = robot_rz + 180

        # vector = np.array([0, -100, 0])

        # self.ax.scatter(robot_tx/1000, robot_ty/1000, marker='o', label='Robot', color='green')
        for person in people_nodes:
            edge_rt = self.rt_api.get_edge_RT(robot_node, person.id)
            # try:
            #     robot_matrix = np.array([[math.cos(robot_rz), -math.sin(robot_rz), 0],
            #                              [math.sin(robot_rz), math.cos(robot_rz), 0],
            #                              [0, 0, 1]])
            #     robot_front_vector = vector.dot(robot_matrix) / 1000
            #     # self.ax.arrow(robot_tx/1000, robot_ty/1000, rotated_person_point_sound[0]-robot_tx/1000, rotated_person_point_sound[1]-robot_ty/1000,
            #     #               fc='lightblue', ec='black')
            # except:
            #     print("No sound_azimut")
            #
            # image_azimut = person.attrs['azimut_refered_to_robot_image'].value
            # print("AZIMUT ANGLE: ", image_azimut)
            # z_axis_rotation_matrix = np.array([[math.cos(image_azimut), -math.sin(image_azimut), 0],
            #                                    [math.sin(image_azimut), math.cos(image_azimut), 0],
            #                                    [0, 0, 1]])
            # rotated_person_point_image = vector.dot(z_axis_rotation_matrix)/1000
            #
            # self.ax.arrow(robot_tx/1000, robot_ty/1000, rotated_person_point_image[0]-robot_tx/1000, rotated_person_point_image[1]-robot_ty/1000,
            #               fc='lightblue', ec='black')

            try:
                tx, ty, tz = edge_rt.attrs['rt_translation'].value

                self.ax.scatter(tx/1000, ty/1000, marker='x', label='Person', color=['blue'])
            except:
                print("No person_pos", edge_rt, person, [ k for k in world_node.edges.items() ])

    def get_coords(self, person):
        x_pos = []
        z_pos = []
        y_pos = []

        z_axis_rotation_matrix = np.array([[math.cos(self.motor.pos), -math.sin(self.motor.pos), 0],
                                           [math.sin(self.motor.pos), math.cos(self.motor.pos), 0],
                                           [0, 0, 1]])

        x_axis_rotation_matrix = np.array([[1, 0, 0],
                                           [0, math.cos(18 * math.pi / 180), -math.sin(18 * math.pi / 180)],
                                           [0, math.sin(18 * math.pi / 180), math.cos(18 * math.pi / 180)]])

        to_robot_reference_matrix = np.array([[1, 0, 0],
                                              [0, 0, 1],
                                              [0, 1, 0]])

        traslation_1_array = np.array([0, -0.06, -0.12])
        traslation_2_array = np.array([0, -0.04, -1.55])

        for joint in list(person.joints.keys()):
            if person.joints[joint].z != 0 and person.joints[joint].x != 0 and person.joints[joint].y != 0 and \
                    person.joints[joint].i != 0 and person.joints[joint].j != 0:
                # key_point = RoboCompHumanCameraBody.KeyPoint()
                if joint in self.needed_joints:
                    vect = np.array([person.joints[joint].x, person.joints[joint].y, person.joints[joint].z])
                    x_y_z = to_robot_reference_matrix.dot(vect)
                    converted_coords_1 = z_axis_rotation_matrix.dot(x_y_z)
                    translated_1 = converted_coords_1 + traslation_1_array
                    converted_coords_2 = x_axis_rotation_matrix.dot(translated_1)
                    world_position = np.around(converted_coords_2 + traslation_2_array, 2)
                    # self.ax.cla()
                    # self.ax.axis([-3, 3, 0, 3])

                    pos = np.array([world_position[0], world_position[1], world_position[2]])
                    # print("POS: ", pos)
                    filtered_pos = 0.75 * pos + 0.25 * self.past_pos
                    # print("FILTERED POS: ", filtered_pos)

                    self.past_pos = pos
                    z_pos.append(filtered_pos[2])
                    x_pos.append(filtered_pos[0])
                    y_pos.append(filtered_pos[1])

        if len(x_pos) > 0 and len(y_pos) > 0 and len(z_pos) > 0:
            self.x_pos = sum(x_pos) / len(x_pos)
            self.y_pos = sum(y_pos) / len(y_pos)
            self.z_pos = sum(z_pos) / len(z_pos)

        # self.ah1.cla()
        # self.ah1.axis([-3, 3, 0, 5])
        # x = np.arange(-5, 5, 0.001)
        # if(len(x_pos) > 0):
        #     x_mean = sum(x_pos)/len(x_pos)
        #     self.x_temp_pose = []
        #
        #     x_std = np.std(x_pos)
        #     # print("X MEAN: ", round(x_mean,2))
        #     # print("X STD: ", round(x_std,2))
        #     self.ah1.plot(x, norm.pdf(x, x_mean, x_std), label='x', color='gold')
        # if(len(y_pos) > 0):
        #     y_mean = sum(y_pos)/len(y_pos)
        #     self.y_temp_pose = []
        #     y_std = np.std(y_pos)
        #     # print("Y MEAN: ", round(y_mean,2))
        #     # print("Y STD: ", round(y_std,2))
        #     self.ah1.plot(x, norm.pdf(x, y_mean, y_std), label='y', color='blue')
        # if (len(z_pos) > 0):
        #     z_mean = sum(z_pos) / len(z_pos)
        #     self.z_temp_pose = []
        #     z_std = np.std(z_pos)
        #     # print("Z MEAN: ", round(z_mean,2))
        #     # print("Z STD: ", round(z_std,2))
        #     self.ah1.plot(x, norm.pdf(x, z_mean, z_std), label='z', color='black')
        #
        # self.ah1.legend(title='Parameters')

    @QtCore.Slot()
    def slot_change_pos(self, pos):   # comes in degrees -150 .. 150. Sent in radians -2.62 .. 2.62
        servo_node = self.g.get_node("servo")
        servo_node.attrs['servo_ref_pos'] = Attribute(float(0.1), self.agent_id)
        servo_node.attrs['servo_ref_speed'] = Attribute(float(0.0), self.agent_id)
        self.g.update_node(servo_node)

    @QtCore.Slot()
    def slot_change_max_speed(self, max_speed):
        self.current_max_speed = max_speed*0.111/60*np.pi*2.0

    @QtCore.Slot()
    def slot_center(self):
        self.ui.horizontalSlider_pos.setSliderPosition(0)
        self.slot_change_pos(0)

    @QtCore.Slot()
    def slot_track(self):
        self.track = not self.track
        if not self.track:
            print("Guardando")
            with open("error_data.json", 'w+') as file:
                json.dump(self.error_dict, file, indent=4)
        print("state track", self.track)

    def print_angle_ball(self, image):
        ball_center_pos = np.array([0, -120, 0])
        z_axis_rotation_matrix = np.array([[math.cos(self.motor.pos), -math.sin(self.motor.pos), 0],
                                           [math.sin(self.motor.pos), math.cos(self.motor.pos), 0],
                                           [0, 0, 1]])
        ball_pos = ball_center_pos.dot(z_axis_rotation_matrix)
        cv2.circle(image, (int(ball_pos[0] + 240), int(ball_pos[1] + 480)), 20, (255, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(round(self.motor.pos,2)), (int(ball_pos[0] + 220), int(ball_pos[1] + 450)), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def startup_check(self):
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompDifferentialRobot.TMechParams from ifaces.RoboCompDifferentialRobot")
        test = ifaces.RoboCompDifferentialRobot.TMechParams()
        print(f"Testing RoboCompHumanCameraBody.TImage from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.TImage()
        print(f"Testing RoboCompHumanCameraBody.TGroundTruth from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.TGroundTruth()
        print(f"Testing RoboCompHumanCameraBody.KeyPoint from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.KeyPoint()
        print(f"Testing RoboCompHumanCameraBody.Person from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.Person()
        print(f"Testing RoboCompHumanCameraBody.PeopleData from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.PeopleData()
        print(f"Testing RoboCompJointMotorSimple.MotorState from ifaces.RoboCompJointMotorSimple")
        test = ifaces.RoboCompJointMotorSimple.MotorState()
        print(f"Testing RoboCompJointMotorSimple.MotorParams from ifaces.RoboCompJointMotorSimple")
        test = ifaces.RoboCompJointMotorSimple.MotorParams()
        print(f"Testing RoboCompJointMotorSimple.MotorGoalPosition from ifaces.RoboCompJointMotorSimple")
        test = ifaces.RoboCompJointMotorSimple.MotorGoalPosition()
        print(f"Testing RoboCompJointMotorSimple.MotorGoalVelocity from ifaces.RoboCompJointMotorSimple")
        test = ifaces.RoboCompJointMotorSimple.MotorGoalVelocity()
        print(f"Testing RoboCompMoveTowards.Command from ifaces.RoboCompMoveTowards")
        test = ifaces.RoboCompMoveTowards.Command()
        print(f"Testing RoboCompSoundRotation.Position from ifaces.RoboCompSoundRotation")
        test = ifaces.RoboCompSoundRotation.Position()
        print(f"Testing RoboCompSoundRotationPub.Position from ifaces.RoboCompSoundRotationPub")
        test = ifaces.RoboCompSoundRotationPub.Position()
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to getAngle method from SoundRotationPub interface
    #
    def SoundRotationPub_getAngle(self, ang):
    
        #
        # write your CODE here
        #
        pass


    #
    # SUBSCRIPTION to getKeyWord method from SoundRotationPub interface
    #
    def SoundRotationPub_getKeyWord(self, word):
    
        #
        # write your CODE here
        #
        pass


    #
    # SUBSCRIPTION to getPositions method from SoundRotationPub interface
    #
    def SoundRotationPub_getPositions(self, pos):
    
        #
        # write your CODE here
        #
        pass


    #
    # SUBSCRIPTION to personFound method from SoundRotationPub interface
    #
    def SoundRotationPub_personFound(self, found):
    
        #
        # write your CODE here
        #
        pass


    # ===================================================================
    # ===================================================================


    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of setState method from Follower interface
    #
    def Follower_setState(self, state):
    
        #
        # write your CODE here
        #
        pass


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompCameraRGBDSimple you can call this methods:
    # self.camerargbdsimple_proxy.getAll(...)
    # self.camerargbdsimple_proxy.getDepth(...)
    # self.camerargbdsimple_proxy.getImage(...)

    ######################
    # From the RoboCompCameraRGBDSimple you can use this types:
    # RoboCompCameraRGBDSimple.TImage
    # RoboCompCameraRGBDSimple.TDepth
    # RoboCompCameraRGBDSimple.TRGBD

    ######################
    # From the RoboCompDifferentialRobot you can call this methods:
    # self.differentialrobot_proxy.correctOdometer(...)
    # self.differentialrobot_proxy.getBasePose(...)
    # self.differentialrobot_proxy.getBaseState(...)
    # self.differentialrobot_proxy.resetOdometer(...)
    # self.differentialrobot_proxy.setOdometer(...)
    # self.differentialrobot_proxy.setOdometerPose(...)
    # self.differentialrobot_proxy.setSpeedBase(...)
    # self.differentialrobot_proxy.stopBase(...)

    ######################
    # From the RoboCompDifferentialRobot you can use this types:
    # RoboCompDifferentialRobot.TMechParams

    ######################
    # From the RoboCompHumanCameraBody you can call this methods:
    # self.humancamerabody_proxy.newPeopleData(...)

    ######################
    # From the RoboCompHumanCameraBody you can use this types:
    # RoboCompHumanCameraBody.TImage
    # RoboCompHumanCameraBody.TGroundTruth
    # RoboCompHumanCameraBody.KeyPoint
    # RoboCompHumanCameraBody.Person
    # RoboCompHumanCameraBody.PeopleData

    ######################
    # From the RoboCompJointMotorSimple you can call this methods:
    # self.jointmotorsimple_proxy.getMotorParams(...)
    # self.jointmotorsimple_proxy.getMotorState(...)
    # self.jointmotorsimple_proxy.setPosition(...)
    # self.jointmotorsimple_proxy.setVelocity(...)
    # self.jointmotorsimple_proxy.setZeroPos(...)

    ######################
    # From the RoboCompJointMotorSimple you can use this types:
    # RoboCompJointMotorSimple.MotorState
    # RoboCompJointMotorSimple.MotorParams
    # RoboCompJointMotorSimple.MotorGoalPosition
    # RoboCompJointMotorSimple.MotorGoalVelocity

    ######################
    # From the RoboCompMoveTowards you can call this methods:
    # self.movetowards_proxy.move(...)

    ######################
    # From the RoboCompMoveTowards you can use this types:
    # RoboCompMoveTowards.Command

    ######################
    # From the RoboCompSoundRotation you can call this methods:
    # self.soundrotation_proxy.getAngle(...)
    # self.soundrotation_proxy.getKeyWord(...)
    # self.soundrotation_proxy.getPositions(...)
    # self.soundrotation_proxy.personFound(...)

    ######################
    # From the RoboCompSoundRotation you can use this types:
    # RoboCompSoundRotation.Position

    ######################
    # From the RoboCompSoundRotationPub you can use this types:
    # RoboCompSoundRotationPub.Position



    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        pass

    def update_node(self, id: int, type: str):
        if type == "intention":
            self.intention_id = id
            intention_node = self.g.get_node(id)
            intention_data = json.loads(intention_node.attrs['current_intention'].value)
            if(list(intention_data.keys())[0] == "FOLLOW_PEOPLE"):
                self.act_chased_person = int(intention_data["FOLLOW_PEOPLE"]["person_node_id"])
            if (list(intention_data.keys())[0] == "TALKING_WITH_PEOPLE"):
                self.act_chased_person = int(intention_data["TALKING_WITH_PEOPLE"]["person_node_id"])

        if type == "person":
            # try:
            self.obtencion_datos()
            person_node = self.g.get_node(id)
            if (self.act_chased_person != None) and (person_node!= None) and (person_node.id == self.act_chased_person):
                print("ENTRANDO")
                puntoMedioX = person_node.attrs['person_pixel_x'].value
                if(puntoMedioX!=self.last_puntoMedioX):
                    self.last_puntoMedioX = puntoMedioX
                    distance = person_node.attrs['distance_to_robot'].value

                    # print("POSICION PERSONA EN IMAGEN: ", puntoMedioX)
                    # print("POSICION SERVO: ", self.motor.pos)
                    # print("POSICION ASIGNADA A SERVO ANTERIOR: ", self.rad_old)

                    robot_node = self.g.get_node("robot")

                    error = puntoMedioX - 240

                    error_rads = np.arctan2(error, 382)

                    # Rotational speed given by odometry
                    act_rot_speed = robot_node.attrs["robot_ref_rot_speed"].value

                    goal_rad = self.motor.pos - error_rads
                    rad_seg = self.k5 * error_rads
                    # print("POSICION ASIGNADA A SERVO: ", goal_rad)

                    tracker_data = {
                        "goal_rads" : goal_rad,
                        "error_rads": rad_seg,
                        "act_rot_speed": act_rot_speed,
                        "distance": distance,
                    }

                    goal = self.tracker_camera(tracker_data)
                    if abs(self.last_goal.position-goal.position)>0.02 and abs(goal.position) < (math.pi/2):
                        self.last_goal = goal
                        servo_node = self.g.get_node("servo")
                        servo_node.attrs['servo_ref_pos'] = Attribute(float(goal.position), self.agent_id)
                        servo_node.attrs['servo_ref_speed'] = Attribute(float(goal.maxSpeed), self.agent_id)
                        self.g.update_node(servo_node)
                        self.error_ant = error
                        self.rad_old = goal_rad
                else:
                    print("El punto no ha sido modificado")
            # except:
            #     print("DATA CANT BE OBTAINED")

    def delete_node(self, id: int):
        if id == self.intention_id:
            self.act_chased_person = None
            self.intention_id = None
            self.slot_change_pos(0)
        console.print(f"DELETE NODE:: {id} ", style='green')


    def update_edge(self, fr: int, to: int, type: str):
        if(type == "interacting"):
            self.act_chased_person = to
            print("PERSONA SEGUIDA:", self.act_chased_person)
        # console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        if(type == "interacting" and self.g.get_edge(fr, to, "following_action") == None):
            self.act_chased_person = None
            self.slot_change_pos(0)
