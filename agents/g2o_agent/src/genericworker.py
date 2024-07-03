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
    Manages a worker process with a periodic timer and provides a signal for
    termination. It also has a method to set the period of the timer.

    Attributes:
        kill (QtCoreSignal): Used to emit a signal when the object needs to be killed.
        ui (Ui_guiDlg): Used to initialize and access the user interface of the widget.
        mutex (QMutex): Used to protect access to the internal state of the worker
            object, particularly the timer and kill signal.
        Period (int): 30 milliseconds by default, which represents the time interval
            for the timer to run.
        timer (QTimer): Used to schedule a call to the `killYourSelf` slot after
            a specified period of time.

    """
    kill = QtCore.Signal()

    def __init__(self, mprx):
        """
        Initializes an instance of the `GenericWorker` class, setting up a GUI
        dialog and creating a mutex for managing access to the timer. It also sets
        the period of the timer to 30 seconds.

        Args:
            mprx (Ui_guiDlg): Used as an argument for the setupUi method.

        """
        super(GenericWorker, self).__init__()


        self.ui = Ui_guiDlg()
        self.ui.setupUi(self)
        # self.show()

        self.mutex = QtCore.QMutex(QtCore.QMutex.Recursive)
        self.Period = 30
        self.timer = QtCore.QTimer(self)


    @QtCore.Slot()
    def killYourSelf(self):
        """
        Emits the `kill` signal, indicating that the instance should be destroyed.

        """
        rDebug("Killing myself")
        self.kill.emit()

    # \brief Change compute period
    # @param per Period in ms
    @QtCore.Slot(int)
    def setPeriod(self, p):
        """
        Updates the `Period` attribute and starts a timer with the new period value
        using the `timer.start()` method.

        Args:
            p (int): Used to set the new period for the timer.

        """
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
