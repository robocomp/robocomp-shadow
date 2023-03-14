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
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
from dr_spaam.detector import Detector

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1000
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

        ckpt = 'ckpt_jrdb_ann_ft_dr_spaam_e20.pth'
        #ckpt = 'ckpt_jrdb_ann_ft_drow3_e40.pth'
        self.detector = Detector(
            ckpt,
            #model="DROW3",  # Or DR-SPAAM
            model="DR-SPAAM",  # Or DR-SPAAM
            gpu=True,  # Use GPU
            stride=1,  # Optionally downsample scan for faster inference
            panoramic_scan=True  # Set to True if the scan covers 360 degree
        )
        # tell the detector field of view of the LiDAR
        laser_fov_deg = 360
        self.detector.set_laser_fov(laser_fov_deg)

        self.ldata = None
        self.ret = None
        self.done = False
        return True


    @QtCore.Slot()
    def compute(self):
        pass

    def startup_check(self):
        print(f"Testing RoboCompLegDetector2DLidar.Leg from ifaces.RoboCompLegDetector2DLidar")
        test = ifaces.RoboCompLegDetector2DLidar.Leg()
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Interface Methods ==================
    # ===================================================================
    #
    # IMPLEMENTATION of getLegs method from LegDetector2DLidar interface
    #
    def LegDetector2DLidar_getLegs(self, ldata):
        scan = [d.dist / 1000.0 for d in ldata]
        scan = np.asarray(scan)
        dets_xy, dets_cls, instance_mask = self.detector(scan)  # (M, 2), (M,), (N,)
        #
        # # confidence threshold
        cls_thresh = 0.7
        cls_mask = dets_cls > cls_thresh
        dets_xy = dets_xy[cls_mask]
        dets_cls = dets_cls[cls_mask]
        return [ifaces.RoboCompLegDetector2DLidar.Leg(y, x) for x, y in dets_xy]
    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompLegDetector2DLidar you can use this types:
    # RoboCompLegDetector2DLidar.Leg


