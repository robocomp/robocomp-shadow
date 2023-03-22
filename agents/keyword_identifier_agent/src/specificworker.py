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
import math

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 2000

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 666
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

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
        self.start = True
        self.recording = True
        self.path_id = None
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
        print("RECORDING:", self.start, self.recording)
        if self.start:
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
            while self.recording:
                # try:
                dir = self.Mic_tuning.direction
                if 270 >= dir and dir >= 90:
                    dir = -(dir - 270)
                elif dir < 90:
                    dir = -(90 + dir)
                else:
                    dir = -(dir - 270)
                print(dir)
                if len(self.dir_list) > 9:
                    self.dir_list.pop(0)
                self.dir_list.append(dir)
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    print("KEYWORD")
                    print(sum(self.dir_list)/len(self.dir_list))
                    self.got_keyword(int((sum(self.dir_list)/len(self.dir_list))))
            pa.terminate()
            audio_stream.close()

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def got_keyword(self, positionAngle):
        virtual_node = Node(agent_id=self.agent_id, type='virtual_person', name="listened_person")
        virtual_node.attrs['position_deg'] = Attribute(positionAngle, self.agent_id)
        self.g.insert_node(virtual_node)
        looking_for_edge = Edge(virtual_node.id,self.g.get_node("robot").id, 'looking_for', self.agent_id)
        self.g.insert_or_assign_edge(looking_for_edge)
        has_edge = Edge(virtual_node.id, self.g.get_node("mind").id, 'has', self.agent_id)
        self.g.insert_or_assign_edge(has_edge)



    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        pass

    def update_node(self, id: int, type: str):
        if type == "path_to_target":
            if len(self.g.get_edges_by_type("looking_for")) > 0:
                print("##################### PAAAAAAAATH #####################")
                self.path_id = id

    def delete_node(self, id: int):
        if self.path_id == id:
            listened_person = self.g.get_node("listened_person")
            if listened_person != None:                
                self.g.delete_edge(self.g.get_node("robot").id, listened_person.id, "looking_for")
                self.g.delete_edge(self.g.get_node("mind").id, listened_person.id, "has")
                self.g.delete_node("listened_person")

    def update_edge(self, fr: int, to: int, type: str):
        if type == "interacting":
            self.start = False
            self.recording = False
            
    
    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass

    def delete_edge(self, fr: int, to: int, type: str):
        if type == "interacting":
            self.start = True
            self.recording = True
