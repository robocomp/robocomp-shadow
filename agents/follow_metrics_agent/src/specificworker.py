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
import csv
import time
from datetime import datetime
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
        self.agent_id = 897
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)
        self.rt_api = rt_api(self.g)

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
        
            
        self.start = False
        self.recording = False
        self.followed_person_id = 0
        
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
        if self.start:
            print("STARTÇING")
            now = datetime.now()
            with open(now.strftime("%m-%d-%Y-%H-%M-%S")+'.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['Timestamp', 'Distance_to_person', 'robot_tx', 'robot_ty', 'robot_tz', 'Image_x_center_difference'])
                robot_node = self.g.get_node("robot")
                
                while self.recording:
                    following_person_node = self.g.get_node(self.followed_person_id)
                    if following_person_node != None and robot_node != None:
                        robot_tx, robot_ty, robot_tz = self.rt_api.get_translation(robot_node.id, following_person_node.id)
                        print("TRASLACION:",robot_tx, robot_ty, robot_tz)
                        person_x_pixel_pos = following_person_node.attrs["person_pixel_x"].value
                        person_distance = following_person_node.attrs["distance_to_robot"].value
                        print(person_distance, robot_tx, robot_ty, robot_tz, person_x_pixel_pos)
                        spamwriter.writerow([time.time(), person_distance, robot_tx, robot_ty, robot_tz, person_x_pixel_pos])
                        time.sleep(0.2)

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)






    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        pass
        # console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        pass
        # console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        pass
        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        if type == "following_action":
            self.followed_person_id = to
            self.recording = True
            self.start = True            

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass
        # console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        if type == "following_action":
            self.recording = False
            self.start = False    
