#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
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
from melo.api import TTS
from collections import deque
import os

import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1000
        if startup_check:
            self.startup_check()
        else:
            # Speed is adjustable
            self.speed = 1.0

            # CPU is sufficient for real-time inference.
            # You can also change to cuda:0
            self.device = 'cuda:0'

            self.model = TTS(language='ES', device=self.device)
            self.speaker_ids = self.model.hps.data.spk2id

            self.output_path = 'es.wav'

            self.text_deque = deque(maxlen=5)

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
        if len(self.text_deque) > 0:
            text = self.text_deque.popleft()
            self.model.tts_to_file(text, self.speaker_ids['ES'], self.output_path, speed=self.speed)
            os.system(f"ffplay -nodisp -autoexit {self.output_path}")

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)



    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of isBusy method from Speech interface
    #
    def Speech_isBusy(self):
        ret = bool()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of say method from Speech interface
    #
    def Speech_say(self, text, overwrite):
        self.text_deque.append(text)
        print(f"Text received: {text}")
        return True
    # ===================================================================
    # ===================================================================



