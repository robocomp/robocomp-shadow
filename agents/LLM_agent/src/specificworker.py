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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import ast
from collections import deque
from src.follow_mission import FollowMission
from src.interact_mission import InteractMission
from src.idle_mission import IdleMission
import json

import cv2
import numpy as np
import mss
import time
from datetime import datetime


sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        self.hide()

        self.state = "idle"

        self.new_missions_deque = deque(maxlen=3)

        self.mission_active = None
        self.mission_id = None
        self.mission_stop_flag = False
        self.last_active_mission_result = None

        try:
            with open("src/main_prompts.json", "r", encoding="utf-8") as f:
                self.main_prompts = json.load(f)
        except Exception as e:
            print(e)
            exit(0)

        # Robot node variables
        robot_nodes = self.g.get_nodes_by_type("omnirobot") + self.g.get_nodes_by_type("robot")
        if len(robot_nodes) == 0:
            print("No robot node found in the graph. Exiting")
            exit(0)
        robot_node = robot_nodes[0]

        self.robot_name = robot_node.name
        self.robot_id = robot_node.id
        self.start_timestamp = robot_node.attrs["timestamp_alivetime"].value / 1000
        self.idle_mission = IdleMission(self.g, self.agent_id, self.main_prompts["idle"], self.start_timestamp)

        LLM_node = self.g.get_node("LLM")
        self.llm_node_id = self.create_LLM_node() if LLM_node is None else LLM_node.id

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def create_LLM_node(self):
        robot_node = self.g.get_node(self.robot_id)
        llm_node = Node(agent_id=self.agent_id, type="LLM",
                                         name="LLM")
        llm_node.attrs['pos_x'] = Attribute(float(robot_node.attrs["pos_x"].value) + 50,
                                                             self.agent_id)
        llm_node.attrs['pos_y'] = Attribute(float(robot_node.attrs["pos_y"].value) + 25,
                                                             self.agent_id)
        llm_node.attrs['level'] = Attribute(robot_node.attrs['level'].value + 1,
                                                             self.agent_id)
        llm_node.attrs['parent'] = Attribute(int(robot_node.id), self.agent_id)
        llm_node.attrs['LLM_response'] = Attribute(str(""), self.agent_id)
        self.g.insert_node(llm_node)
        has_edge = Edge(llm_node.id, self.robot_id, "has", self.agent_id)
        self.g.insert_or_assign_edge(has_edge)

        return llm_node.id

    def __del__(self):
        """Destructor"""


    @QtCore.Slot()
    def compute(self):
        # 1 - Check if any mission is active
        # 2 - Check if there is any pending mission
        match self.state:
            case "idle":
                self.idle_mission.monitor(self.last_active_mission_result)
            case "new_mission":
                # 3 - Get the first mission from the deque
                mission = self.new_missions_deque.popleft()
                console.print(f"Processing new mission: {mission}", style='green')
                match mission["mission"]:
                    case "follow":
                        self.mission_active = FollowMission(self.g, self.agent_id, mission, self.start_timestamp)
                        self.state = "active_mission"
                    case "interact":
                        self.mission_active = InteractMission(self.g, self.agent_id, mission, self.start_timestamp)
                        self.state = "active_mission"
                    case _:
                        console.print(f"Mission type not recognized. Ignoring it and cleaning last mission result.", style='red')
                        self.last_active_mission_result = {}
                        self.state = "idle"
            case "active_mission":
                monitor_result = self.mission_active.monitor()
                # Check if monitor result is different form empty dict
                if monitor_result != {}:
                    self.last_active_mission_result = monitor_result
                self.idle_mission.monitor(self.last_active_mission_result)
            case "stop_mission":
                # self.recorder.stop()
                console.print("Mission stop flag is set. Stopping the mission.", style='red')
                # self.mission_active.store_as_dataset()
                self.mission_active = None
                self.mission_stop_flag = False
                self.last_active_mission_result = {}
                self.state = "idle"
        # print("Active mission result:", self.last_active_mission_result)


    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        if "words_to_say" in attribute_names:
            asr_node = self.g.get_node(id)
            if asr_node is not None:
                print(f"ASR node updated: {asr_node}")
                words = asr_node.attrs["words_to_say"].value
                if type(self.mission_active) is InteractMission:
                    self.mission_active.set_ASR_words(words)
                    console.print(f"New ASR words received: {words}", style='green')

    def update_node(self, id: int, type: str):
        if type == "mission":
            self.mission_id = id
            mission_node = self.g.get_node(id)
            if mission_node == None:
                console.print(f"Mission node {id} not found in the graph.", style='red')
                return
            console.print(f"New mission generated", style='green')
            mission = mission_node.attrs["plan"].value
            parsed_mission_plan = ast.literal_eval(mission)
            if len(parsed_mission_plan) == 0:
                console.print(f"Mission plan is empty. Ignoring it.", style='red')
                return
            self.new_missions_deque.append(parsed_mission_plan[0])
            self.state = "new_mission"

    def delete_node(self, id: int):
        if id == self.mission_id:
            console.print(f"Mission {id} deleted", style='red')
            self.state = "stop_mission"

    def update_edge(self, fr: int, to: int, type: str):
        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
