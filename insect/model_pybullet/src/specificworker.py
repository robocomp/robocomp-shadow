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
import interfaces as ifaces
import pybullet as p
import pybullet_data
import time
import numpy as np
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
sys.path.append('/home/robocomp/robocomp/lib')
console = Console(highlight=False)
import cv2

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100
        if startup_check:
            self.startup_check()
        else:

            # Start PyBullet in GUI mode
            self.physicsClient = p.connect(p.GUI)

            # Set the path to PyBullet data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            #plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            #print("plugin=", plugin)
            #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

            self.personUid = p.loadURDF("girl.urdf", basePosition=[0, 0, 0])
                                   #baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
            p.loadURDF("plane.urdf", [0, 0, -1])
            #self.personUid = p.loadURDF("r2d2.urdf", baseOrientation=p.getQuaternionFromEuler([0, np.pi / 2, 0]), basePosition=[0, 0, 0])
            # Simulation parameters
            self.simulationTime = 5  # in seconds
            self.timeStep = 1. / 240.  # time step for the simulation

            # Define a steady forward speed
            self.forwardSpeed = 0.05  # Adjust this value as needed

            self.camDistance = 2
            self.yaw = 10

            #cv2.namedWindow('image')
            #cv2.setMouseCallback('image', self.mouse_click)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        pos, ori = p.getBasePositionAndOrientation(self.personUid)

        # Calculate the forward direction based on the orientation
        pos, ori = p.getBasePositionAndOrientation(self.personUid)
        forwardDir = p.getMatrixFromQuaternion(ori)[0:3]  # Forward direction in world coordinates

        # Calculate the new position
        newPos = [pos[0] + self.forwardSpeed * forwardDir[0] * self.timeStep,
                  pos[1] + self.forwardSpeed * forwardDir[1] * self.timeStep,
                  pos[2] + self.forwardSpeed * forwardDir[2] * self.timeStep]

        # Set the new position of the person
        p.resetBasePositionAndOrientation(self.personUid, newPos, ori)

        # Step the simulation
        p.stepSimulation()

        # pixelWidth = 640
        # pixelHeight = 480
        # camTargetPos = [0, 0, 0]
        # pitch = -10.0
        # roll = 0
        # upAxisIndex = 2
        # viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, self.camDistance, self.yaw, pitch, roll,
        #                                                  upAxisIndex)
        # projectionMatrix = [
        #     1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        #     -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
        # ]
        #
        # _, _, rgba, _, _ = p.getCameraImage(pixelWidth,
        #                            pixelHeight,
        #                            viewMatrix=viewMatrix,
        #                            projectionMatrix=projectionMatrix,
        #                            shadow=1,
        #                            lightDirection=[1, 1, 1])
        #
        # # Convert PyBullet's RGBA to OpenCV's BGR format
        # rgba_array = np.array(rgba, dtype=np.uint8)
        # rgba_array = np.reshape(rgba_array, (pixelHeight, pixelWidth, 4))
        # bgr_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGR)
        #
        # # Display the image using OpenCV
        # #cv2.imshow('image', bgr_array)
        # #cv2.waitKey(2)
        #print("Caca")

    def mouse_click(self, event, x, y, flags, param):

        # to check if left mouse  button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            self.yaw += 1
        if event == cv2.EVENT_RBUTTONDOWN:
            self.yaw -= 1


    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)





