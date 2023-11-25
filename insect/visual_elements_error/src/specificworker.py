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

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100
        self.empty = ifaces.RoboCompVisualElements.TObjects()

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
        # read visual_objects from detector and model
        real_objects = self.read_real_objects()
        synth_objects = self.read_synth_objects()

        # compute error between real_objects and synth_objects
        # error is a TObjects from VisualElements. Zero values will be ignored in model
            # period error
            # timestamp error: real_TObjects.timestampimage and synth_TObjects.timestampimage must match
            # x,y,z error: translation of object in 3D world must match
            # rx,ry,rz error: rotation of object in 3D world must match
            # vx,vy,vz error: pose of object in 3D world must match
            # etc

        # send error to model
        # self.visualelements2_proxy.setVisualObjects(error)

        return True

    ########################################################################
    def read_real_objects(self):
        objects = ifaces.RoboCompVisualElements.TObjects()
        try:
            objects = self.visualelements_proxy.getVisualObjects(self.empty)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e)
        return objects

    def read_synth_objects(self):
        objects = ifaces.RoboCompVisualElements.TObjects()
        try:
            objects = self.visualelements1_proxy.getVisualObjects(self.empty)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e)
        return objects
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

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
    #
    def SegmentatorTrackingPub_setTrack(self, target):
    
        #
        # write your CODE here
        #
        pass

    # ===================================================================
    # =============== Methods for Component Implements ==================
    # ===================================================================
    #
    # IMPLEMENTATION of getVisualObjects method from VisualElements interface
    #
    def VisualElements_getVisualObjects(self, objects):
        return ifaces.RoboCompVisualElements.TObjects()
    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompVisualElements you can call this methods:
    # self.visualelements_proxy.getVisualObjects(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject


