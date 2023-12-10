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

        self.model_manager.step_simulation()

    ########################################################################
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
            attr = real_data.objects[0].attributes
            width = float(attr["width"])/1000
            depth = float(attr["depth"])/1000
            height = float(attr["height"])/1000
            rotation = float(attr["rotation"])
            centre = [float(attr["center_x"])/1000, float(attr["center_y"])/1000,  0]
            print("Adding new room:", width, depth, rotation, centre)

            self.model_manager.create_room_with_doors(room_width=width, room_depth=depth, room_height=height,
                                                      room_center=centre, room_rotation=rotation, doors_specs={})
        else:
            print("Prediction OK")

    def compute_synth_data(self):
        synth_data = ifaces.RoboCompVisualElementsPub.TData()
        synth_data.objects = []
        floor = self.model_manager.get_room()
        if floor:
            obj = ifaces.RoboCompVisualElementsPub.TObject()
            obj.name = "room"
            obj.attributes = self.model_manager.get_room()
            synth_data.objects.append(object)
        return synth_data

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



