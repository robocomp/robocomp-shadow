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
        initializes a `GenericWorker` object, defining its mutex and timer variables
        for managing tasks within a specified period (30 seconds).

        Args:
            mprx (`object` because there's no explicit indication of a specific
                data type.): QMetaObject::PropertyResourceReference of an object
                that holds the state of the worker.
                
                		- `super(GenericWorker, self).__init__()` calls the constructor
                of the parent class `GenericWorker`.
                		- `self.mutex` is a `QtCore.QMutex` object with recursive locking
                semantics, which allows only one thread to access the worker's
                methods at a time.
                		- `self.Period` represents the interval between repeated actions,
                set to 30 in this example.
                		- `self.timer` is a `QtCore.QTimer` instance that is owned by
                the worker object and will be used to schedule repetitions of the
                action.
                

        """
        super(GenericWorker, self).__init__()


        self.mutex = QtCore.QMutex(QtCore.QMutex.Recursive)
        self.Period = 30
        self.timer = QtCore.QTimer(self)


    @QtCore.Slot()
    def killYourSelf(self):
        """
        emits the `kill` signal, which has unspecified consequences.

        """
        rDebug("Killing myself")
        self.kill.emit()

    # \brief Change compute period
    # @param per Period in ms
    @QtCore.Slot(int)
    def setPeriod(self, p):
        """
        updates its instance variable ` Period` and starts the timer using that value.

        Args:
            p (int): new period for which the timer should be set, and it is used
                to update the ` Period ` attribute of the object and start the
                timer accordingly.

        """
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
