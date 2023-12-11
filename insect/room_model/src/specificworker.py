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

console = Console(highlight=False)
from model_manager import Model_Manager

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100
        self.real_data = ifaces.RoboCompVisualElementsPub.TData()
        self.real_data.objects = []
        self.synth_data = ifaces.RoboCompVisualElementsPub.TData()
        self.synth_data.objects = []
        self.joy_data = ifaces.RoboCompJoystickAdapter.TData()

        if startup_check:
            self.startup_check()
        else:
            # model_manager
            self.model_manager = Model_Manager()

            self.robot = self.model_manager.add_robot()

            # room_width = 6.0
            # room_height = 1.5
            # room_depth = 8.0
            # room_center = (0, 0, 1.5)
            #
            # # Define the door specifications for each wall (wall index: (door_width, door_height, door_position))
            # door_specs = {
            #     0: (1.0, 1.0, 1.5),  # Wall 0 has a door of width 1.0, height 2.0 at position 1.5
            #     1: (1.0, 1.0, 2.3),  # Wall 2 has a door of width 1.0, height 2.0 at position 2.5
            #     2: (1.0, 1.0, 1),  # Wall 2 has a door of width 1.0, height 2.0 at position 2.5
            #     3: (1.0, 1.0, 2.8),  # Wall 2 has a door of width 1.0, height 2.0 at position 2.5
            #     # Walls 1 and 3 have no doors
            # }

            # Create the room with doors
            #wall_parts_ids = self.model_manager.create_room_with_doors(room_width, room_depth, room_height, room_center, door_specs)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        # read visual_objects from detector and model
        real_data = self.real_data

        # compute error between real_objects and synth_objects
        self.compute_prediction_error(real_data, self.synth_data)

        # compute synth_objects from model
        self.synth_data = self.compute_synth_data()

        # check efferent motor commands from joystick
        self.check_efferent_motor_commands(self.joy_data)

        self.model_manager.step_simulation()

    ###room_width, room_width,#####################################################################
    def compute_prediction_error(self, real_data, synth_data):
        # compute error between real_objects and synth_objects
        # error is a TObjects from VisualElements. Zero values will be ignored in model
        # period error
        # timestamp error: real_TObjects.timestampimage and synth_TObjects.timestampimage must match
        # x,y,z error: translation of object in 3D world must match
        # rx,ry,rz error: rotation of object in 3D world must match
        # vx,vy,vz error: pose of object in 3D world must match
        # etc

        if len(real_data.objects) > len(synth_data.objects):
            # add new objects to synth_data
            door_specs = {}
            k = 0
            for obj in real_data.objects:
                if obj.type == 6:   # "door":
                    door_width = float(obj.attributes["width"])/1000
                    door_height = float(obj.attributes["height"])/1000
                    door_pos = float(obj.attributes["position"])/1000
                    door_specs[k] = (door_width, door_height, door_pos)
                    print("Adding new door: id ", k, door_width, door_height, door_pos)
                    k += 1

            for i, obj in enumerate(real_data.objects):
                if obj.type == 5:   # "room"
                    attr = obj.attributes
                    room_width = float(attr["width"])/1000
                    room_depth = float(attr["depth"])/1000
                    room_height = float(attr["height"])/1000
                    room_rotation = float(attr["rotation"])
                    room_center = [float(attr["center_x"])/1000, float(attr["center_y"])/1000,  0]
                    print("Adding new room:", room_width, room_depth, room_rotation, room_center)
                    # self.model_manager.create_room_with_doors(room_width=room_width,
                    #                                           room_depth=room_depth,
                    #                                           room_height=room_height,
                    #                                           room_center=room_center,
                    #                                           room_rotation=room_rotation,
                    #                                           doors_specs=door_specs)
                    self.model_manager.create_rectangular_room(width=room_width,
                                                               depth=room_depth,
                                                               height=room_height,
                                                               orientation=room_rotation)

                    break  # only one room
        else:
            print("Prediction OK")
    def compute_synth_data(self):
        synth_data = ifaces.RoboCompVisualElementsPub.TData()
        synth_data.objects = []
        room_attr = self.model_manager.get_room()
        if room_attr:
            obj = ifaces.RoboCompVisualElementsPub.TObject()
            obj.type = 5
            obj.attributes = room_attr
            synth_data.objects.append(obj)
        doors_attr_list = self.model_manager.get_doors()
        synth_data.objects.append(ifaces.RoboCompVisualElementsPub.TObject())
        synth_data.objects.append(ifaces.RoboCompVisualElementsPub.TObject())

        # for doors_attr in doors_attr_list:
        #      obj = ifaces.RoboCompVisualElementsPub.TObject()
        #      obj.type = 5
        #      obj.attributes = doors_attr
        #      synth_data.objects.append(obj)
        return synth_data
    def check_efferent_motor_commands(self, joy_data):
        # check efferent motor commands from joystick if any, send them to model
        side = adv = rot = 0
        if joy_data is not None and joy_data.axes is not None:
            for axis in joy_data.axes:
                if "advance" == axis.name:
                    adv = axis.value
                if "rotate" == axis.name:
                    rot = axis.value
                if "side" == axis.name:
                    side = axis.value
            self.model_manager.set_robot_velocity(side, adv, rot)
    #######################################################################3
    def startup_check(self):
        print(f"Testing RoboCompVisualElements.TRoi from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TRoi()
        print(f"Testing RoboCompVisualElements.TObject from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TObject()
        print(f"Testing RoboCompVisualElements.TRoi from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TRoi()
        print(f"Testing RoboCompVisualElements.TObject from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TObject()
        QTimer.singleShot(200, QApplication.instance().quit)

    #######################################################################3
    # =============== Interface methods: SubscribesTo ================
    # SUBSCRIPTION to setVisualObjects method from VisualElementsPub interface

    def VisualElementsPub_setVisualObjects(self, data):
        #print("Received: ", data)
        self.real_data = data

    #
    # SUBSCRIPTION to sendData method from JoystickAdapter interface
    #
    def JoystickAdapter_sendData(self, data):
        self.joy_data = data

