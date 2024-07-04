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


try:
    from ui_mainUI import *
except:
    print("Can't import UI file. Did you run 'make'?")
    sys.exit(-1)



class GenericWorker(QtWidgets.QWidget):

    """
    Manages a timer and a signal to stop its own execution. It has methods to
    change the timer period and to emit the signal to stop itself.

    Attributes:
        kill (QtCoreQObjectSlot): Used to emit a signal that can be caught by any
            connected slots to stop the worker's execution.
        ui (Ui_guiDlg): Used to setup the user interface of the class.
        mutex (QMutex): Used to protect the worker's state from concurrent access.
        Period (int): Used to set the time interval for the timer signal emitted
            by the `setPeriod()` method, which changes its value on each call.
        timer (QtCoreQTimer): Used to start a timer that emits the `kill` signal
            after a specified period.

    """
    kill = QtCore.Signal()

    def __init__(self, mprx):
        """
        Initializes an object of the `GenericWorker` class, setting up a UI widget,
        creating a mutex for synchronization, and defining a timer with a period
        of 500 milliseconds.

        Args:
            mprx (Ui_guiDlg): Used as the parent widget for the GenericWorker
                object's UI.

        """
        super(GenericWorker, self).__init__()


        self.ui = Ui_guiDlg()
        self.ui.setupUi(self)
        # self.show()

        self.mutex = QtCore.QMutex(QtCore.QMutex.Recursive)
        self.Period = 500
        self.timer = QtCore.QTimer(self)


    @QtCore.Slot()
    def killYourSelf(self):
        """
        Emits the `kill` signal, indicating that the object should be terminated.

        """
        rDebug("Killing myself")
        self.kill.emit()

    # \brief Change compute period
    # @param per Period in ms
    @QtCore.Slot(int)
    def setPeriod(self, p):
        """
        Sets the period of a timer and updates the internal variable `Period`.

        Args:
            p (int): Used to set the new period for the timer.

        """
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
