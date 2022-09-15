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
import struct
import pyaudio
import pvporcupine
from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 2000
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if self.dev:
            self.Mic_tuning = Tuning(self.dev)
        else:
            print("################## NO MIC FOUND. CHECK IF CONNECTED ##################")
        self.dir_list = []
        # self.porcupine = pvporcupine.create(keywords=["picovoice", "blueberry"],
        #                                keyword_paths=["Hey-Giraffe_en_linux_v2_1_0.ppn"],
        #                                access_key="eeegi+PSnbxq38fvCQJOCLjx280A8CNzdf4Q5NkCCxyZYFbNURErfw==")
        #
        # self.pa = pyaudio.PyAudio()
        # for i in range(self.pa.get_device_count()):
        #     print(self.pa.get_device_info_by_index(i))
        # self.audio_stream = self.pa.open(
        #     input_device_index=1,
        #     rate=self.porcupine.sample_rate,
        #     channels=1,
        #     format=pyaudio.paInt16,
        #     input=True,
        #     frames_per_buffer=self.porcupine.frame_length)

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
        porcupine = pvporcupine.create(keywords=["picovoice", "blueberry"],
                                       keyword_paths=["Hey-Giraffe_en_linux_v2_1_0.ppn"],
                                       access_key="zvBFLun7pJBxuAqmhqhbfm0y1xICq0MW3YQhDkrSphWvqwLLbHgJfA==")

        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            print(pa.get_device_info_by_index(i))
            if "ReSpeaker 4 Mic Array" in pa.get_device_info_by_index(i)["name"]:
                dev_index = i
                break
        audio_stream = pa.open(
            input_device_index=dev_index,
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)
        while True:
            try:
                dir = self.Mic_tuning.direction
                if 270 >= dir >= 90:
                    dir = -(dir - 90)
                elif dir < 90:
                    dir = 90 - dir
                else:
                    dir = -(dir - 450)

                if len(self.dir_list) > 9:
                    self.dir_list.pop(0)
                self.dir_list.append(dir)
                print(dir)
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    print("KEYWORD")
                    print(sum(self.dir_list)/len(self.dir_list))
                    self.soundrotation_proxy.gotKeyWord(sum(self.dir_list)/len(self.dir_list))
            except:
                print("PROBLEMA CON EL MICRO")

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)




    ######################
    # From the RoboCompSoundRotation you can call this methods:
    # self.soundrotation_proxy.getKeyWord(...)
    # self.soundrotation_proxy.personFound(...)


