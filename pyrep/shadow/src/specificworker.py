#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2020 by YOUR NAME HERE
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

from genericworker import *
import time
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.robots.mobiles.shadow import Shadow
import numpy as np
import numpy_indexed as npi
import cv2
import itertools as it
from math import *
import pprint
import traceback
from sys import getsizeof

_OBJECT_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep',
                 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                 'chair', 'couch',
                 'potted_plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']

class TimeControl:
    def __init__(self, period_):
        self.counter = 0
        self.start = time.time()  # it doesn't exist yet, so initialize it
        self.start_print = time.time()  # it doesn't exist yet, so initialize it
        self.period = period_

    def wait(self):
        elapsed = time.time() - self.start
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self.start = time.time()
        self.counter += 1
        if time.time() - self.start_print > 1:
            print("Shadow PyRep - Freq -> ", self.counter, "Hz")
            self.counter = 0
            self.start_print = time.time()



class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        super(SpecificWorker, self).__init__(proxy_map)

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        SCENE_FILE = params["scene_file"]
        self.WITH_BILL = False
        if "bill" in SCENE_FILE:
            self.WITH_BILL = True

        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()

        # robot

        self.robot = Shadow()
        self.robot_object = Shape("/Shadow")
        self.ShadowBase_WheelRadius = 44  # mm coppelia
        self.ShadowBase_DistAxes = 380.  # mm
        self.ShadowBase_AxesLength = 422.  # mm
        self.ShadowBase_Rotation_Factor = 8.1  # it should be (DistAxes + AxesLength) / 2
        self.speed_robot = []
        self.speed_robot_ant = []
        self.bState = RoboCompGenericBase.TBaseState()

        # cameras
        self.cameras_write = {}
        self.cameras_read = {}

        # self.tablet_camera_name = "camera_tablet"
        # cam = VisionSensor(self.tablet_camera_name)
        # self.cameras_write[self.tablet_camera_name] = { "handle": cam,
        #                                                 "id": 0,
        #                                                 "angle": np.radians(cam.get_perspective_angle()),
        #                                                 "width": cam.get_resolution()[0],
        #                                                 "height": cam.get_resolution()[1],
        #                                                 "focalx": (cam.get_resolution()[0] / 2) / np.tan(
        #                                                  np.radians(cam.get_perspective_angle() / 2)),
        #                                                 "focaly": (cam.get_resolution()[1] / 2) / np.tan(
        #                                                     np.radians(cam.get_perspective_angle() / 2)),
        #                                                 "rgb": np.array(0),
        #                                                 "depth": np.ndarray(0),
        #                                                 "is_ready": False,
        #                                                 "is_rgbd": False,
        #                                                 "rotated": False,
        #                                                 "has_depth": False
        #                                         }

        self.top_camera_name = "/Shadow/camera_top"
        try:
            cam = VisionSensor(self.top_camera_name)
        except:
            traceback.print_exc()

        self.cameras_write[self.top_camera_name] = {"handle": cam,
                                                     "id": 0,
                                                     "angle": np.radians(cam.get_perspective_angle()),
                                                     "width": cam.get_resolution()[0],
                                                     "height": cam.get_resolution()[1],
                                                     "focalx": (cam.get_resolution()[0] / 2) / np.tan(
                                                        np.radians(cam.get_perspective_angle() / 2.0)),
                                                     "focaly": (cam.get_resolution()[1] / 2) / np.tan(
                                                         np.radians(cam.get_perspective_angle() / 2)),
                                                     "rgb": np.array(0),
                                                     "depth": np.ndarray(0),
                                                     "is_ready": False,
                                                     "is_rgbd": True,
                                                     "rotated": True,
                                                     "has_depth": True,
                                                     "has_points": False
                                                    }

        self.omni_camera_rgb_name = "/Shadow/omnicamera/sensorRGB"
        try:
            cam = VisionSensor(self.omni_camera_rgb_name)
            self.cameras_write[self.omni_camera_rgb_name] = {"handle": cam,
                                                             "id": 0,
                                                             "angle": np.radians(cam.get_perspective_angle()),
                                                             "width": cam.get_resolution()[0],
                                                             "height": cam.get_resolution()[1],
                                                             "focalx": (cam.get_resolution()[0] / 2) / np.tan(
                                                                 np.radians(cam.get_perspective_angle() / 2.0)),
                                                             "focaly": (cam.get_resolution()[1] / 2) / np.tan(
                                                                 np.radians(cam.get_perspective_angle() / 2)),
                                                             "rgb": np.array(0),
                                                             "depth": np.ndarray(0),
                                                             "is_ready": False,
                                                             "is_rgbd": False,
                                                             "rotated": False,
                                                             "has_depth": False
                                                             }
        except:
            print("Camera OMNI sensorRGB  not found in Coppelia")

        self.omni_camera_depth_name = "/Shadow/omnicamera/sensorDepth"
        try:
            cam = VisionSensor(self.omni_camera_depth_name)
            self.cameras_write[self.omni_camera_depth_name] = {"handle": cam,
                                                               "id": 0,
                                                               "angle": np.radians(cam.get_perspective_angle()),
                                                               "width": cam.get_resolution()[0],
                                                               "height": cam.get_resolution()[1],
                                                               "focalx": (cam.get_resolution()[0] / 2) / np.tan(
                                                                   np.radians(cam.get_perspective_angle() / 2.0)),
                                                               "focaly": (cam.get_resolution()[1] / 2) / np.tan(
                                                                   np.radians(cam.get_perspective_angle() / 2)),
                                                               "rgb": np.array(0),
                                                               "depth": np.ndarray(0),
                                                               "is_ready": False,
                                                               "is_rgbd": False,
                                                               "rotated": False,
                                                               "has_depth": False
                                                               }
        except:
            print("Camera OMNI sensorDEPTH not found in Coppelia")

        self.cameras_read = self.cameras_write.copy()

        # Read existing people
        self.people = {}
        if Dummy.exists("Bill_base"):
            self.people["Bill"] = Dummy("Bill_base")
        elif Dummy.exists("Bill"):
            self.people["Bill"] = Dummy("Bill")

        for i in range(0, 2):
            name = "Bill#" + str(i)
            name_base = "Bill_base#" + str(i)
            if Dummy.exists(name_base):
                self.people[name] = Dummy(name_base)
            elif Dummy.exists(name):
                self.people[name] = Dummy(name)

        # laser
        # self.lasers = {}
        # self.hokuyo_front_left_name = "Hokuyo_sensor2"
        # cam = VisionSensor(self.hokuyo_front_left_name)
        # self.lasers[self.hokuyo_front_left_name] = { "handle": cam,
        #                                               "id": 0,
        #                                               "angle": np.radians(cam.get_perspective_angle()),
        #                                               "width": cam.get_resolution()[0],
        #                                              "semiwidth": cam.get_resolution()[0] / 2.0,
        #                                               "height": cam.get_resolution()[1],
        #                                               "focal": (cam.get_resolution()[0] / 2) / np.tan(
        #                                                   np.radians(cam.get_perspective_angle() / 2)),
        #                                               "rgb": np.array(0),
        #                                               "depth": np.ndarray(0),
        #                                               "offset_angle": -np.pi/3.0
        #                                              }
        # self.hokuyo_front_right_name = "Hokuyo_sensor1"
        # cam = VisionSensor(self.hokuyo_front_right_name)
        # self.lasers[self.hokuyo_front_right_name] = { "handle": cam,
        #                                               "id": 0,
        #                                               "angle": np.radians(cam.get_perspective_angle()),
        #                                               "width": cam.get_resolution()[0],
        #                                               "semiwidth": cam.get_resolution()[0]/2.0,
        #                                               "height": cam.get_resolution()[1],
        #                                               "focal": (cam.get_resolution()[0] / 2) / np.tan(
        #                                                 np.radians(cam.get_perspective_angle() / 2)),
        #                                               "rgb": np.array(0),
        #                                               "depth": np.ndarray(0),
        #                                               "offset_angle": np.pi / 3.0
        #                                             }
        # self.ldata_write = []
        # self.ldata_read = []
        
        # PoseEstimation
        self.robot_full_pose_write = RoboCompFullPoseEstimation.FullPoseEuler()
        self.robot_full_pose_read = RoboCompFullPoseEstimation.FullPoseEuler()

        # JoyStick
        self.joystick_newdata = []
        self.last_received_data_time = 0

        # Tablet tilt motor
        # self.tablet_motor = Joint("tablet_joint")
        # self.tablet_new_pos = None

        # Eye pan motor
        self.eye_motor = Joint("/Shadow/camera_pan_joint")
        self.eye_new_pos = None

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    def compute(self):
        tc = TimeControl(0.05)
        while True:
            self.pr.step()
            self.read_robot_pose()
            self.move_robot()
            self.read_cameras([self.omni_camera_rgb_name, self.omni_camera_depth_name, self.top_camera_name])
            #ksself.read_people()
            self.read_joystick()
            self.move_eye()
            tc.wait()

    ###########################################
    ### PEOPLE get and publish people position
    ###########################################
    def read_people(self):
        people_data = RoboCompHumanToDSRPub.PeopleData()
        people_data.timestamp = time.time()
        people = []  # RoboCompHumanToDSRPub.People()
        for name, handle in self.people.items():
            pos = handle.get_position()
            rot = handle.get_orientation()
            person = RoboCompHumanToDSRPub.Person(len(people), pos[0] * 1000, pos[1] * 1000, pos[2] * 1000,
                                                  pi - rot[2] - pi / 2,
                                                  {})
            people.append(person)
        try:
            people_data.peoplelist = people
            self.humantodsrpub_proxy.newPeopleData(people_data)
        except Ice.Exception as e:
            print(e)

    def read_laser(self):
        data = self.pr.script_call("get_depth_data@Hokuyo", 1)
        if len(data[1]) > 0:
            self.hokuyo = Shape("Hokuyo")
            h_pos = self.hokuyo.get_position()
            polar = np.zeros(shape=(int(len(data[1])/3), 2))
            i = 0
            for x, y, z in self.grouper(data[1], 3):                      # extract non-intersecting groups of 3
                # translate to the robot center
                #x += h_pos[0]
                #y += h_pos[1]
                polar[i] = [-np.arctan2(y, x), np.linalg.norm([x, y])]    # add to list in polar coordinates
                i += 1

            angles = np.linspace(-np.radians(120), np.radians(120), 360)  # create regular angular values
            positions = np.searchsorted(angles, polar[:, 0])  # list of closest position in polar for each laser measurement
            self.ldata_write = [RoboCompLaser.TData(a, 0) for a in angles]  # create empty 240 angle array
            pos, medians = npi.group_by(positions).median(polar[:, 1])  # group by repeated positions
            for p, m in it.zip_longest(pos, medians):  # fill the angles with measures
                if p < len(self.ldata_write):
                    self.ldata_write[p].dist = int(m * 1000)  # to millimeters
            if self.ldata_write[0] == 0:
               self.ldata_write[0] = 200  # half robot width
            del self.ldata_write[-3:]
            del self.ldata_write[:3]
            for i in range(1, len(self.ldata_write)):
               if self.ldata_write[i].dist == 0:
                   self.ldata_write[i].dist = self.ldata_write[i - 1].dist


            self.ldata_read, self.ldata_write = self.ldata_write, self.ldata_read

            # try:
            #     self.laserpub_proxy.pushLaserData(self.ldata_read)
            # except Ice.Exception as e:
            #     print(e)

    def grouper(self, inputs, n, fillvalue=None):
        iters = [iter(inputs)] * n
        return it.zip_longest(*iters, fillvalue=fillvalue)

    ###########################################
    ### CAMERAS get and publish cameras data
    ###########################################
    def read_cameras(self, camera_names):

         if self.top_camera_name in camera_names:  # RGBD rotated
            cam = self.cameras_write[self.top_camera_name]
            image_float = cam["handle"].capture_rgb()
            image = cv2.normalize(src=image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth = cam["handle"].capture_depth(True)  # meters
            depth = np.frombuffer(depth, dtype=np.float32).reshape((cam["height"], cam["width"]))
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # we change width and height here to follow the rotation operation
            cam["depth"] = RoboCompCameraRGBDSimple.TDepth( cameraID=cam["id"],
                                                            width=cam["height"],  # cambiados
                                                            height=cam["width"],
                                                            focalx=cam["focaly"],
                                                            focaly=cam["focalx"],
                                                            alivetime=int(time.time()*1000),
                                                            period=50, # ms
                                                            depthFactor=1.0,
                                                            depth=depth.tobytes())
            cam["rgb"] = RoboCompCameraRGBDSimple.TImage( cameraID=cam["id"],
                                                          width=cam["height"],          #cambiados
                                                          height=cam["width"],
                                                          depth=3,
                                                          focalx=cam["focaly"],
                                                          focaly=cam["focalx"],
                                                          alivetime=int(time.time()*1000),
                                                          period=50,  # ms
                                                          image=image.tobytes(),
                                                          compressed=False)
            cam["is_ready"] = True

         if self.omni_camera_rgb_name in camera_names:  # RGB not-rotated
             cam = self.cameras_write[self.omni_camera_rgb_name]
             image_float = cam["handle"].capture_rgb()
             image = cv2.normalize(src=image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             cam["rgb"] = RoboCompCameraRGBDSimple.TImage(cameraID=cam["id"],
                                                          width=cam["width"],
                                                          height=cam["height"],
                                                          depth=3,
                                                          focalx=cam["focalx"],
                                                          focaly=cam["focaly"],
                                                          alivetime=int(time.time() * 1000),
                                                          period=50,  # ms
                                                          image=image.tobytes(),
                                                          compressed=False)

             cam["is_ready"] = True

         if self.omni_camera_depth_name in camera_names:  # RGB not-rotated
             cam = self.cameras_write[self.omni_camera_depth_name]
             image_float = cam["handle"].capture_rgb()
             image = cv2.normalize(src=image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             cam["rgb"] = RoboCompCameraRGBDSimple.TImage(cameraID=cam["id"],
                                                          width=cam["width"],
                                                          height=cam["height"],
                                                          depth=3,
                                                          focalx=cam["focalx"],
                                                          focaly=cam["focaly"],
                                                          alivetime=int(time.time() * 1000),
                                                          period=50,  # ms
                                                          image=image.tobytes(),
                                                          compressed=False)

             cam["is_ready"] = True

         self.cameras_write, self.cameras_read = self.cameras_read, self.cameras_write

    ###########################################
    ### JOYSITCK read and move the robot
    ###########################################
    def read_joystick(self):
        if self.joystick_newdata:  # and (time.time() - self.joystick_newdata[1]) > 0.1:
            adv = 0.0         # CHANGE THIS INITIALIZATION TO FIX THE MOVING PROBLEM
            rot = 0.0
            side = 0.0
            left_pan = 0.0
            right_pan = 0.0

            for x in self.joystick_newdata[0].axes:
                if x.name == "advance":
                    adv = x.value if np.abs(x.value) > 0.1 else 0  # mm/sg
                if x.name == "rotate":
                    rot = x.value if np.abs(x.value) > 0.1 else 0  # rads/sg
                if x.name == "side":
                    side = x.value if np.abs(x.value) > 0.1 else 0
                if x.name == "left_pan":
                    left_pan = x.value
                if x.name == "right_pan":
                    right_pan = x.value

            converted = self.convert_base_speed_to_radians(adv*1.8, side*1.8, rot*1.2)  # temptative values to match real velocities. Adjust mechanical parameters instead

            # convert velocities in world reference system to local robot coordinates: adv, side and rot
            # linear_vel, ang_vel = self.robot_object.get_velocity()
            # ang = self.robot_object.get_orientation()[2]
            # adv_vel = np.transpose(np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])) @ (np.array([linear_vel[0], linear_vel[1]]))
            #print("ADV", adv_vel[1]*1000, "SIDE", adv_vel[0]*1000)
            self.robot.set_base_angular_velocites(converted)
            #
            if left_pan > right_pan:
                self.eye_new_pos = left_pan
            else:
                self.eye_new_pos = -right_pan

            self.joystick_newdata = None
            self.last_received_data_time = time.time()
        else:
            elapsed = time.time() - self.last_received_data_time
            if elapsed > 2 and elapsed < 3:   # safety break
                self.robot.set_base_angular_velocites([0, 0, 0])

        # dummy = Dummy("viriato_head_pan_tilt_nose_target")
        # pantilt = Dummy("viriato_head_camera_pan_tilt")
        # pose = dummy.get_position(pantilt)
        # dummy.set_position([pose[0], pose[1] - pan / 10, pose[2] + tilt / 10], pantilt)

    def convert_base_speed_to_radians(self, adv, side, rot):
        return [adv / self.ShadowBase_WheelRadius, side / self.ShadowBase_WheelRadius,
                rot * self.ShadowBase_Rotation_Factor]

    ###########################################
    ### Get ROBOT POSE from Coppelia
    ###########################################
    def read_robot_pose(self):

        pose = self.robot_object.get_position()
        rot = self.robot_object.get_orientation()
        linear_vel, ang_vel = self.robot_object.get_velocity()

        isMoving = np.abs(linear_vel[0]) > 0.01 or np.abs(linear_vel[1]) > 0.01 or np.abs(ang_vel[2]) > 0.01
        ang = rot[2]
        robot_vel = np.transpose(np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])) @ (np.array([linear_vel[0], linear_vel[1]]))
        self.bState = RoboCompGenericBase.TBaseState(x=pose[0] * 1000,
                                                     z=pose[1] * 1000,
                                                     alpha=rot[2],
                                                     advVx=robot_vel[0] * 1000,
                                                     advVz=robot_vel[1] * 1000,
                                                     rotV=ang_vel[2],
                                                     isMoving=isMoving)
        
        self.robot_full_pose_write.x = pose[0] * 1000
        self.robot_full_pose_write.y = pose[1] * 1000
        self.robot_full_pose_write.z = pose[2] * 1000
        self.robot_full_pose_write.rx = rot[0]
        self.robot_full_pose_write.ry = rot[1]
        self.robot_full_pose_write.rz = rot[2]
        self.robot_full_pose_write.vx = linear_vel[0] * 1000.0
        self.robot_full_pose_write.vy = linear_vel[1] * 1000.0
        self.robot_full_pose_write.vz = linear_vel[2] * 1000.0
        self.robot_full_pose_write.vrx = ang_vel[0]
        self.robot_full_pose_write.vry = ang_vel[1]
        self.robot_full_pose_write.vrz = ang_vel[2]
        self.robot_full_pose_write.adv = robot_vel[1]*1000.0
        self.robot_full_pose_write.side = robot_vel[0]*1000.0
        self.robot_full_pose_write.rot = ang_vel[2]

        # swap
        self.robot_full_pose_write, self.robot_full_pose_read = self.robot_full_pose_read, self.robot_full_pose_write

    ###########################################
    ### MOVE ROBOT from Omnirobot interface
    ###########################################
    def move_robot(self):
        if self.speed_robot != self.speed_robot_ant:  # or (isMoving and self.speed_robot == [0,0,0]):
            self.robot.set_base_angular_velocites(self.speed_robot)
            #print("Velocities sent to robot:", self.speed_robot)
            self.speed_robot_ant = self.speed_robot

    ###########################################
    ### MOVE EYE
    ###########################################

    def move_eye(self):
        if self.eye_new_pos:
            self.eye_motor.set_joint_position(self.eye_new_pos)  # radians
            self.eye_new_pos = None

    #################################################################
    ##################################################################################
    # SUBSCRIPTION to sendData method from JoystickAdapter interface
    ###################################################################################
    def JoystickAdapter_sendData(self, data):
        self.joystick_newdata = [data, time.time()]

    ##################################################################################
    #                       Methods for CameraRGBDSimple
    # ===============================================================================
    #
    # getAll
    #
    def CameraRGBDSimple_getAll(self, camera):
        if camera in self.cameras_read.keys() \
                and self.cameras_read[camera]["is_ready"] \
                and self.cameras_read[camera]["is_rgbd"]:
            return RoboCompCameraRGBDSimple.TRGBD(self.cameras_read[camera]["rgb"], self.cameras_read[camera]["depth"])
        else:
            ex = RoboCompCameraRGBDSimple.HardwareFailedException()
            ex.what = "No camera found with this name or with depth attributes: " + camera
            raise e

    def CameraRGBDSimple_getDepth(self, camera):
        if camera in self.cameras_read.keys() \
                and self.cameras_read[camera]["is_ready"] \
                and self.cameras_read[camera]["has_depth"]:
            return self.cameras_read[camera]["depth"]
        else:
            ex = RoboCompCameraRGBDSimple.HardwareFailedException()
            ex.what = "No camera found with this name or with depth attributes: " + camera
            raise e

    def CameraRGBDSimple_getImage(self, camera):
        if camera in self.cameras_read.keys() and self.cameras_read[camera]["is_ready"]:
            return self.cameras_read[camera]["rgb"]
        else:
            ex = RoboCompCameraRGBDSimple.HardwareFailedException()
            ex.what = "No camera found with this name: " + camera
            raise e

    def CameraRGBDSimple_getPoints(self, camera):
        if camera in self.cameras_read.keys() and self.cameras_read[camera]["is_ready"]\
                and self.cameras_read[camera]["has_points"]:
            return self.cameras_read[camera]["points"]
        else:
            ex = RoboCompCameraRGBDSimple.HardwareFailedException()
            ex.what = "No camera found with this name: " + camera
            raise e

    ##############################################
    ### Omnibase
    #############################################
    def OmniRobot_correctOdometer(self, x, z, alpha):
        pass

    def OmniRobot_getBasePose(self):
        x = self.bState.x
        z = self.bState.z
        alpha = self.bState.alpha
        return [x, z, alpha]

    def OmniRobot_getBaseState(self):
        return self.bState

    def OmniRobot_resetOdometer(self):
        pass

    def OmniRobot_setOdometer(self, state):
        pass

    def OmniRobot_setOdometerPose(self, x, z, alpha):
        pass

    def OmniRobot_setSpeedBase(self, advx, advz, rot):
        self.speed_robot = self.convert_base_speed_to_radians(advz, advx, rot)
        print("Received speed command" , self.speed_robot)

    def OmniRobot_stopBase(self):
        self.speed_robot = [0, 0, 0]

    # ===================================================================
    # CoppeliaUtils
    # ===================================================================
    def CoppeliaUtils_addOrModifyDummy(self, type, name, pose):
        if not Dummy.exists(name):
            dummy = Dummy.create(0.1)
            # one color for each type of dummy
            if type == RoboCompCoppeliaUtils.TargetTypes.Info:
                pass
            if type == RoboCompCoppeliaUtils.TargetTypes.Hand:
                pass
            if type == RoboCompCoppeliaUtils.TargetTypes.HeadCamera:
                pass
            dummy.set_name(name)
        else:
            dummy = Dummy(name)
            parent_frame_object = None
            if type == RoboCompCoppeliaUtils.TargetTypes.HeadCamera:
                parent_frame_object = Dummy("viriato_head_camera_pan_tilt")
            # print("Coppelia ", name, pose.x/1000, pose.y/1000, pose.z/1000)
            dummy.set_position([pose.x / 1000., pose.y / 1000., pose.z / 1000.], parent_frame_object)
            dummy.set_orientation([pose.rx, pose.ry, pose.rz], parent_frame_object)

    # =============== Methods for FULLPOSEESTIMATION ==================
    # ===================================================================

    #
    # IMPLEMENTATION of getFullPose method from FullPoseEstimation interface
    #
    def FullPoseEstimation_getFullPoseEuler(self):
        return self.robot_full_pose_read

    def FullPoseEstimation_getFullPoseMatrix(self): # NO USAR
        t = self.tm.get_transform("origin", "robot")
        m = RoboCompFullPoseEstimation.FullPoseMatrix()
        m.m00 = t[0][0]
        m.m01 = t[0][1]
        m.m02 = t[0][2]
        m.m03 = t[0][3]
        m.m10 = t[1][0]
        m.m11 = t[1][1]
        m.m12 = t[1][2]
        m.m13 = t[1][3]
        m.m20 = t[2][0]
        m.m21 = t[2][1]
        m.m22 = t[2][2]
        m.m23 = t[2][3]
        m.m30 = t[3][0]
        m.m31 = t[3][1]
        m.m32 = t[3][2]
        m.m33 = t[3][3]
        return m

    def FullPoseEstimation_setInitialPose(self, x, y, z, rx, ry, rz):

        # should move robot in Coppelia to designated pose
        self.tm.add_transform("origin", "world",
                               pytr.transform_from(pyrot.active_matrix_from_intrinsic_euler_xyz([rx, ry, rz]), [x, y, z])
        )

    ###################################################################
    # IMPLEMENTATION Ultrasound interface
    ################################################################
    def Ultrasound_getAllSensorDistances(self):
        ret = RoboCompUltrasound.SensorsState()
        #
        # write your CODE here
        #
        return ret

    def Ultrasound_getAllSensorParams(self):
        ret = RoboCompUltrasound.SensorParamsList()
        #
        # write your CODE here
        #
        return ret

    #
    # IMPLEMENTATION of getBusParams method from Ultrasound interface
    #
    def Ultrasound_getBusParams(self):
        ret = RoboCompUltrasound.BusParams()
        #
        # write your CODE here
        #
        return ret

    #
    # IMPLEMENTATION of getSensorDistance method from Ultrasound interface
    #
    def Ultrasound_getSensorDistance(self, sensor):
        ret = int()
        #
        # write your CODE here
        #
        return ret

    #
    # IMPLEMENTATION of getSensorParams method from Ultrasound interface
    #
    def Ultrasound_getSensorParams(self, sensor):
        ret = RoboCompUltrasound.SensorParams()
        #
        # write your CODE here
        #
        return ret

    ###################################################################
    # IMPLEMENTATION RSSI interface
    ################################################################
    def RSSIStatus_getRSSIState(self):
        ret = RoboCompRSSIStatus.TRSSI()
        ret.percentage = 100;
        return ret

    #
    # IMPLEMENTATION of getBatteryState method from BatteryStatus interface
    #
    def BatteryStatus_getBatteryState(self):
        ret = RoboCompBatteryStatus.TBattery()
        ret.percentage = 100
        return ret
    #
    #######################################################
    #### Laser Interface
    #######################################################
    #
    # getLaserAndBStateData
    #
    def Laser_getLaserAndBStateData(self):
        bState = RoboCompGenericBase.TBaseState()
        return self.ldata_read, bState

   #
   # getLaserConfData
   #

    def Laser_getLaserConfData(self):
        ret = RoboCompLaser.LaserConfData()
        return ret

    #
    # getLaserData
    #

    def Laser_getLaserData(self):
        return self.ldata_read

    # ===================================================================
    # IMPLEMENTATION of  JointMotorSimple interface
    # ===================================================================

    def JointMotorSimple_getMotorParams(self, motor):
        ret = RoboCompJointMotorSimple.MotorParams()
        return ret

    #
    # IMPLEMENTATION of getMotorState method from JointMotorSimple interface
    #
    def JointMotorSimple_getMotorState(self, motor):
        if motor == "camera_pan_joint":
            ret = RoboCompJointMotorSimple.MotorState(self.eye_motor.get_joint_position())  # radians
        return ret

    #
    # IMPLEMENTATION of setPosition method from JointMotorSimple interface
    #
    def JointMotorSimple_setPosition(self, name, goal):
        print("JointMotorSimple_setPosition: ", name, goal)
        # check position limits -10 to 80
        if name == "tablet":
            self.tablet_new_pos = goal.position
        elif name == "camera_pan_joint":
            self.eye_new_pos = goal.position
        else: print("Unknown motor name", name)

    #
    # IMPLEMENTATION of setVelocity method from JointMotorSimple interface
    #
    def JointMotorSimple_setVelocity(self, name, goal):
        pass

    #
    # IMPLEMENTATION of setZeroPos method from JointMotorSimple interface
    #
    def JointMotorSimple_setZeroPos(self, name):

        #
        # write your CODE here
        #
        pass

    # =====================================================================
    # IMPLEMENTATION of CameraSimple interface
    #######################################################################
    def CameraSimple_getImage(self):
        camera = self.tablet_camera_name
        if camera in self.cameras_read.keys() \
                and self.cameras_read[camera]["is_ready"]\
                and not self.cameras_read[camera]["is_rgbd"]:
                    return self.cameras_read[camera]["rgb"]
        else:
            e = RoboCompCameraSimple.HardwareFailedException()
            e.what = "No (no RGBD) camera found with this name: " + camera
            raise e

    # ===================================================================
    # IMPLEMENTATION of getPose method from BillCoppelia interface
    # ###################################################################
    def BillCoppelia_getPose(self):
        ret = RoboCompBillCoppelia.Pose()
        bill = Dummy("/Bill/Bill")
        #bill = Dummy("Bill")
        pos = bill.get_position()
        print(pos)
        ret.x = pos[0] * 1000.0
        ret.y = pos[1] * 1000.0
        linear_vel, ang_vel = bill.get_velocity()
        ret.vx = linear_vel[0] * 1000.0
        ret.vy = linear_vel[1] * 1000.0
        ret.vo = ang_vel[2]
        ret.orientation = bill.get_orientation()[2]
        return ret

    #
    # IMPLEMENTATION of setSpeed method from BillCoppelia interface
    #
    def BillCoppelia_setSpeed(self, adv, rot):
        pass

    #
    # IMPLEMENTATION of setTarget method from BillCoppelia interface
    #
    def BillCoppelia_setTarget(self, tx, ty):
        bill_target = Dummy("Bill_goalDummy")
        current_pos = bill_target.get_position()
        bill_target.set_position([tx/1000.0, ty/1000.0, current_pos[2]])

    # ===================================================================


