#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 by YOUR NAME HERE
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
from PySide2 import QtWidgets, QtCore

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'

Ice.loadSlice("-I ./src/ --all ./src/CommonBehavior.ice")
import RoboCompCommonBehavior




class GenericWorker(QtCore.QObject):

    kill = QtCore.Signal()

    def __init__(self, mprx):
        """
        Initializes a `GenericWorker` object, setting up proxies for a
        `Camera360RGBDProxy` and a `VisualElementsProxy`, as well as creating a
        mutex and timer for periodic processing.

        Args:
            mprx (`PythonProxy`.): Proxy Manager, which provides proxies for the
                `Camera360RGBDProxy` and `VisualElementsProxy` classes in the
                initialization of the `GenericWorker` class.
                
                		- `Camera360RGBDProxy`: This property represents a proxy class
                for interacting with the `Camera360RGBD` interface.
                		- `VisualElementsProxy`: This property represents a proxy class
                for interacting with the `VisualElements` interface.
                		- `mutex`: A mutual exclusion lock used to protect shared resources.
                		- `Period`: The period of time between updates, set to 30 in
                this example.
                		- `timer`: A timer object used to trigger updates at a specific
                interval.

        """
        super(GenericWorker, self).__init__()

        self.camera360rgbd_proxy = mprx["Camera360RGBDProxy"]
        self.visualelements_proxy = mprx["VisualElementsProxy"]

        self.mutex = QtCore.QMutex(QtCore.QMutex.Recursive)
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
        """
        Updates the instance variable `Period`, and then starts a timer that fires
        every period (`self.Period`) using the passed time as the timer interval.

        Args:
            p (int): duration of the timer, and by assigning it to the `self.Period`
                instance variable and starting the `timer.start()` method, the
                function sets the timer to count down for the specified duration.

        """
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
