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
#
import numpy as np
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 2000

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = "_CHANGE_THIS_ID_"
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        try:
            #signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            #signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            #signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            #signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            #signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        if startup_check:
            self.startup_check()
        else:
            self.rt_api = rt_api(self.g)
            self.inner_api = inner_api(self.g)
            self.room_initialized = False

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True


    @QtCore.Slot()
    def compute(self):

        # check it there is an active goto edge connecting the robot and a door
            # if so, then check the robot position. If it is at the door
                # signal that the current room is not anymore by removing self-edge
                # wait for a new room to be created
                # when new room is created, check for the first door in it.  Should be the door used to get in
                # connect both doors with a new edge "same"
                ### loop-closure  PARA LA SEGUNDA VUELTA  (gist es una imagen 360 de la habitación usada para reconocerla rápidamente)
                # check if the current room "gist" is similar to the gists of the other rooms in the agent's internal graph.
                # if so, then replace the current room by the matching nominal room in the internal graph
                    # and adjust door connections
        # if not, then check if there is a room not marked as "current" that has been there for at least 1 minute
            # if so, then replace the not-current room by a proxy empty room node with the doors as children nodes
                # save the old room in the agent's internal graph
                # save a gist of the room in the agent's internal graph
                # connect the door connecting both rooms with a new edge "same"

        # check it there is an active goto edge connecting the robot and a door
        goto_edges = self.g.get_edges_by_type("goto")
        if len(goto_edges) > 0:
            # search if one of the edges in goto_edges goes to a door
            for edge in goto_edges:
                if self.g.get_node_type(edge.fr) == "robot" and self.g.get_node_type(edge.to) == "door":
                    robot_id = edge.fr
                    door_id = edge.to
                    # get door coordinates transformed to robot coordinates are smaller than 100mm
                    door_coords_in_robot = self.inner_api(robot_id, door_id)
                    # check that door_coords_in_robot are smaller than 100mm
                    if np.sqrt(np.power(door_coords_in_robot[0], 2) + np.power(door_coords_in_robot[1], 2)) < 100:
                        # signal that the current room is not anymore by removing self-edge
                        self.g.remove_edge(robot_id, door_id)
                        # wait for a new room to be created


    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):

        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
