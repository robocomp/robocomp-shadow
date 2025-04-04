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

import sys, Ice, os

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'


Ice.loadSlice("-I ./src/ --all ./src/CommonBehavior.ice")
import RoboCompCommonBehavior
Ice.loadSlice("-I ./src/ --all ./src/CameraRGBDSimple.ice")
import RoboCompCameraRGBDSimple
Ice.loadSlice("-I ./src/ --all ./src/Camera360RGB.ice")
import RoboCompCamera360RGB
Ice.loadSlice("-I ./src/ --all ./src/CameraSimple.ice")
import RoboCompCameraSimple
Ice.loadSlice("-I ./src/ --all ./src/GenericBase.ice")
import RoboCompGenericBase 
Ice.loadSlice("-I ./src/ --all ./src/JoystickAdapter.ice")
import RoboCompJoystickAdapter 
Ice.loadSlice("-I ./src/ --all ./src/Laser.ice")
import RoboCompLaser
Ice.loadSlice("-I ./src/ --all ./src/OmniRobot.ice")
import RoboCompOmniRobot
Ice.loadSlice("-I ./src/ --all ./src/CoppeliaUtils.ice")
import RoboCompCoppeliaUtils
Ice.loadSlice("-I ./src/ --all ./src/BillCoppelia.ice")
import RoboCompBillCoppelia
Ice.loadSlice("-I ./src/ --all ./src/FullPoseEstimation.ice")
import RoboCompFullPoseEstimation
Ice.loadSlice("-I ./src/ --all ./src/BatteryStatus.ice")
import RoboCompBatteryStatus
Ice.loadSlice("-I ./src/ --all ./src/RSSIStatus.ice")
import RoboCompRSSIStatus
Ice.loadSlice("-I ./src/ --all ./src/JointMotorSimple.ice")
import RoboCompJointMotorSimple
Ice.loadSlice("-I ./src/ --all ./src/Lidar3D.ice")
import RoboCompLidar3D

import camerargbdsimpleI
import camerasimpleI
import camera360rgbI
import laserI
import omnirobotI
import joystickadapterI
import coppeliautilsI
import fullposeestimationI
import batterystatusI
import rssistatusI
import jointmotorsimpleI
import billcoppeliaI
import lidar3dI

class GenericWorker():

    #kill = QtCore.Signal()

    def __init__(self, mprx):
        super(GenericWorker, self).__init__()



