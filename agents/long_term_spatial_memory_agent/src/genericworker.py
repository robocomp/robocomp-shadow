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
    Provides a mechanism for killing itself and setting a period for a timer. It
    has a signal `kill` that is emitted when the object is killed, and a method
    `setPeriod` to set the timer period.

    Attributes:
        kill (QtCoreSignal): Used to signal the object's termination.
        ui (Ui_guiDlg): Used to store the user interface of a GUI dialog box.
        mutex (QMutex): Used to protect the internal state of the worker object.
        Period (int): 500 milliseconds by default, used to set the time interval
            for the timer event emitted by the `setPeriod()` method.
        timer (QtCoreQTimer): Used to start a timer that calls the `setPeriod`
            slot when it expires.

    """
    kill = QtCore.Signal()

    def __init__(self, mprx):
        """
        Initializes an instance of the `GenericWorker` class by setting up the UI,
        creating a mutex for thread-safe access to the `Period` variable, and
        starting a timer with a 500ms delay.

        Args:
            mprx (Ui_guiDlg): Used to set up the user interface for the GenericWorker
                class.

        """
        super(GenericWorker, self).__init__()


        self.ui = Ui_guiDlg()ss
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
        Sets the period of a timer and prints a message to the console when it changes.

        Args:
            p (int): Used to set the new period value for the application's timer.

        """
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
