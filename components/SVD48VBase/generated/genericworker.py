#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 20252025 by YOUR NAME HERE
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
from PySide6 import QtWidgets, QtCore



class GenericWorker(QtCore.QObject):

    kill = QtCore.Signal()

    def __init__(self, mprx, configData):
        super(GenericWorker, self).__init__()

        self.fullposeestimationpub_proxy = mprx["FullPoseEstimationPub"]
        self.fullposeestimationpub_proxy = mprx["FullPoseEstimationPub"]

        self.configData = configData

        self.Period = 30
        self.timer = QtCore.QTimer(self)

    @QtCore.Slot()
    def killYourSelf(self):
        rDebug("Killing myself")
        self.kill.emit()

    # \brief Change compute period
    # @param per Period in ms
    @QtCore.Slot(int)
    def setPeriod(self, p):
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
