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

from PySide6.QtCore import QTimer, QStringListModel
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import os
import time
import ast

from collections import deque

from src.long_term_graph import LongTermGraph

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from src.goto_mission import GOTOMission
from src.follow_mission import FollowMission
from src.guide_mission import GuideMission
from src.interact_mission import InteractMission

from pydsr import *


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            # signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        # Robot node variables
        robot_nodes = self.g.get_nodes_by_type("omnirobot") + self.g.get_nodes_by_type("robot")
        if len(robot_nodes) == 0:
            print("No robot node found in the graph. Exiting")
            exit(0)
        robot_node = robot_nodes[0]

        self.robot_name = robot_node.name
        self.robot_id = robot_node.id

        print("Robot node found:", self.robot_name, "with id", self.robot_id)

        home_dir = os.path.expanduser("~")
        self.graph_path = os.path.join(home_dir, "igraph_LTSM", "graph.pkl")
        # Global map variables
        self.long_term_graph = LongTermGraph(self.graph_path)

        self.missions_to_process = deque(maxlen=2)
        self.missions = deque(maxlen=2)
        self.pending_missions = deque(maxlen=5)
        self.has_intentions_to_check = deque(maxlen=2)
        self.active_mission = None

        self.missions_history = []

        self.possible_missions = ["follow", "goto", "guide", "interact"]
        self.required_proposed_mission_keys = ["control_command", "mission", "target", "priority", "is_a_submission"]

        self.create_mission_history_node()

        # Crear un modelo con la lista de habitaciones
        model = QStringListModel()
        model.setStringList(self.possible_missions)

        # Asignar el modelo al QListView
        self.ui.mission_list.setModel(model)

        self.ui.execute_button.clicked.connect(self.handle_execute_button)
        self.ui.stop_button.clicked.connect(self.handle_stop_button)
        self.ui.wait_button.clicked.connect(self.handle_wait_button)
        self.ui.abort_button.clicked.connect(self.abort_mission)

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def create_mission_history_node(self):
        # Create plan node connected to robot node
        mission_node = self.g.get_node("mission_history")
        if mission_node is None:
            mission_node = Node(agent_id=self.agent_id, type="mission", name="mission_history")
            # Convert self.active mission to a string representation
            mission_node.attrs['plan'] = Attribute(str(""), self.agent_id)
            mission_node.attrs['pending_missions'] = Attribute(str(list(self.pending_missions)), self.agent_id)
            robot_node = self.g.get_node(self.robot_id)
            robot_node_x_pose = robot_node.attrs["pos_x"].value
            robot_node_y_pose = robot_node.attrs["pos_y"].value
            if robot_node_x_pose is None or robot_node_y_pose is None:
                console.print("Robot node has no position attributes. Cannot create mission node.", style='red')
                return
            mission_node.attrs['pos_x'] = Attribute(float(robot_node_x_pose + 50), self.agent_id)
            mission_node.attrs['pos_y'] = Attribute(float(robot_node_y_pose + 50), self.agent_id)
            mission_node.attrs['parent'] = Attribute(int(self.robot_id), self.agent_id)
            robot_node_level = robot_node.attrs["level"].value
            if robot_node_level:
                mission_node.attrs['level'] = Attribute(robot_node_level + 1, self.agent_id)
                status = self.g.insert_node(mission_node)
                print("Mission node created with status:", status)
                mission_node = self.g.get_node("mission_history")
                has_mission_edge = Edge(mission_node.id, robot_node.id, "has", self.agent_id)
                result = self.g.insert_or_assign_edge(has_mission_edge)
                print("Has edge created with result:", result)
            else:
                console.print("Robot node has no level attribute. Cannot create mission node.", style='red')
                return

    def update_lists(self):
        selected_index = self.ui.mission_list.currentIndex()
        selected_action = selected_index.data() if selected_index.isValid() else ""

        new_people = [node.name for node in self.g.get_nodes_by_type("person")]
        igraph_rooms = self.long_term_graph.get_igraph_rooms()
        igraph_elements = self.long_term_graph.get_igraph_elements()

        # # --- Actualizar lista de destinos ---
        # if selected_action == "goto" or selected_action == "guide" or selected_action == "interact":
        #     new_destinations = igraph_rooms + igraph_elements + new_people
        # else:
        #     new_destinations = []

        new_destinations = igraph_rooms + igraph_elements + new_people

        current_dest_model = self.ui.destinations_list.model()
        current_dest_list = current_dest_model.stringList() if isinstance(current_dest_model, QStringListModel) else []

        if new_destinations != current_dest_list:
            dest_model = QStringListModel()
            dest_model.setStringList(new_destinations)
            self.ui.destinations_list.setModel(dest_model)

        # # --- Actualizar lista de personas ---
        # if selected_action == "follow" or selected_action == "guide" or selected_action == "interact":
        #     new_people = [node.name for node in self.g.get_nodes_by_type("person")]
        # else:
        #     new_people = []
        #
        # current_people_model = self.ui.people_list.model()
        # current_people_list = current_people_model.stringList() if isinstance(current_people_model,
        #                                                                       QStringListModel) else []

        # if new_people != current_people_list:
        #     people_model = QStringListModel()
        #     people_model.setStringList(new_destinations)
        #     self.ui.people_list.setModel(people_model)

    @QtCore.Slot()
    def compute(self):
        self.update_lists()
        # Checking missions comming from LLM or UI
        if self.missions_to_process:
            mission_data = self.missions_to_process.popleft()
            print(f"Processing mission: {mission_data}")
            match mission_data["control_command"]:
                case "stop" :
                    if self.active_mission is None:
                        console.print(f"There is not active mission for stopping. returning.", style='red')
                    elif mission_data["mission"] == self.active_mission.type:
                        console.print(f"Aborting current mission.", style='red')
                        self.abort_mission()
                    elif mission_data["mission"] == self.active_mission.current_mission["mission"]:
                        console.print(f"Aborting current submission.", style='red')
                        self.active_mission.abort_current_submission()
                case "start":
                    console.print(f"New mission created.", style='green')
                    self.create_mission(mission_data)
                case "wait":
                    console.print(f"Waiting command received, no action taken.", style='blue')
                    self.setting_waiting_mission(mission_data)


        # Check if there is any processed mission to be executed as active_mission or queued in pending_missions
        if self.missions:
            new_mission = self.missions.popleft()
            if self.active_mission:
                if new_mission["is_a_submission"] == 1:
                    if new_mission["mission"] == self.active_mission.current_mission["mission"]:
                        console.print(f"Proposed submission is the current mission. Ignoring.", style='red')
                        return
                    console.print(f"Subbmission to add to the main mission.", style='yellow')
                    self.active_mission.abort_and_store_current_submission()
                    self.active_mission.insert_new_submission(new_mission["missions"][0])
                    return
                # Check if current active mission is not the same as new_mission
                if self.active_mission.type == new_mission["mission"] and self.active_mission.target == new_mission["target"]:
                    console.print(f"New mission similar to the current one.", style='yellow')

                    return
                if self.active_mission.type == "interact" and new_mission["mission"] == "follow" and self.active_mission.target == new_mission["target"]:
                    print("New mission is follow the same person as the current interact mission, not adding to pending mission.")
                    console.print(f"New mission is follow the same person as the current interact mission, not adding to pending mission.", style='yellow')
                    self.abort_mission(True)
                    console.print(f"Executing mission: {new_mission}", style='blue')
                    self.insert_mission_node_in_dsr(new_mission)
                    self.set_active_mission(new_mission)
                    return
                # Save current active mission in pending missions
                active_mission_dict = {"mission": self.active_mission.type, "target": self.active_mission.target,
                                       "missions": self.active_mission.missions}
                if "priority" in new_mission.keys():
                    if new_mission["priority"] == 1:
                        self.pending_missions.append(active_mission_dict)
                        self.abort_mission(True)
                        console.print(f"Executing mission: {new_mission}", style='blue')
                        self.insert_mission_node_in_dsr(new_mission)
                        self.set_active_mission(new_mission)
                    else:
                        print(f"New mission {new_mission} received, adding to pending missions.")
                        self.pending_missions.append(new_mission)
                else:
                    print(f"New mission {new_mission} received, adding to pending missions.")
                    self.pending_missions.append(new_mission)
                for pending in self.pending_missions:
                    print(f"##### Pending mission: {pending}")
            else:
                print(f"No active mission. Processing  {new_mission} as new high priority mission.")
                self.insert_mission_node_in_dsr(new_mission)
                self.set_active_mission(new_mission)

        # Monitor mission if any is active
        if self.active_mission:
            if self.active_mission.monitor():
                console.print("Mission completed successfully.", style='green')
                # Delete mission node from the graph
                mission_node = self.g.get_node("mission")
                if mission_node is None:
                    console.print("No active mission node found in the graph.", style='red')
                    self.active_mission = None
                    return
                # Remove the has edge from robot to mission node
                has_edge = self.g.get_edge(mission_node.id, self.robot_id, "has")
                if has_edge:
                    self.g.delete_edge(has_edge.origin, has_edge.destination, has_edge.type)
                    console.print("Has edge deleted from the graph.", style='green')
                if mission_node:
                    self.g.delete_node(mission_node.id)
                    console.print("Mission node deleted from the graph.", style='green')
                else:
                    console.print("Mission node not found in the graph.", style='red')

                robot_node = self.g.get_node(self.robot_id)
                robot_timestamp = robot_node.attrs["timestamp_alivetime"].value if robot_node and "timestamp_alivetime" in robot_node.attrs else time.time()
                self.missions_history.append(
                        {"mission": self.active_mission.type, "target": self.active_mission.target, "init_timestamp" : self.active_mission.init_timestamp, "end_timestamp": robot_timestamp})
                print("Mission history:", self.missions_history)
                mission_history_node = self.g.get_node("mission_history")
                if mission_history_node:
                    mission_history_node.attrs['plan'] = Attribute(str(self.missions_history), self.agent_id)
                    mission_history_node.attrs['pending_missions'] = Attribute(str(list(self.pending_missions)), self.agent_id)
                    self.g.update_node(mission_history_node)
                self.active_mission = None

        elif self.pending_missions:
            print("Pending missions available, processing next one.")
            active_mission = self.pending_missions.popleft()
            self.insert_mission_node_in_dsr(active_mission)
            self.set_active_mission(active_mission)

        if self.has_intentions_to_check:
            destination_node = self.has_intentions_to_check.popleft()
            self.check_intention_creation(destination_node)

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def handle_execute_button(self):
        mission_index = self.ui.mission_list.currentIndex()
        selected_mission = mission_index.data()
        destination_index = self.ui.destinations_list.currentIndex()
        selected_destination = destination_index.data()
        person_index = self.ui.people_list.currentIndex()
        selected_person = person_index.data()
        if mission_index.isValid() and (destination_index.isValid() or person_index.isValid()):
            selected_target = {"control_command" : "start", "mission" : selected_mission, "target": selected_destination, "priority" : self.ui.priority_check.isChecked(), "is_a_submission" : self.ui.submission_check.isChecked()}
            print(f"Misión seleccionada: {selected_target}")
            self.missions_to_process.append(selected_target)

    def handle_stop_button(self):
        mission_index = self.ui.mission_list.currentIndex()
        selected_mission = mission_index.data()
        destination_index = self.ui.destinations_list.currentIndex()
        selected_destination = destination_index.data()
        person_index = self.ui.people_list.currentIndex()
        selected_person = person_index.data()
        if mission_index.isValid() and (destination_index.isValid() or person_index.isValid()):
            selected_target = {"control_command" : "stop", "mission" : selected_mission, "target": selected_destination, "priority" : self.ui.priority_check.isChecked(), "is_a_submission" : self.ui.submission_check.isChecked()}
            print(f"Misión seleccionada: {selected_target}")
            self.missions_to_process.append(selected_target)

    def handle_wait_button(self):
        mission_index = self.ui.mission_list.currentIndex()
        selected_mission = mission_index.data()
        destination_index = self.ui.destinations_list.currentIndex()
        selected_destination = destination_index.data()
        person_index = self.ui.people_list.currentIndex()
        selected_person = person_index.data()
        if mission_index.isValid() and (destination_index.isValid() or person_index.isValid()):
            selected_target = {"control_command" : "wait", "mission" : selected_mission, "target": selected_destination, "priority" : self.ui.priority_check.isChecked(), "is_a_submission" : self.ui.submission_check.isChecked()}
            print(f"Misión seleccionada: {selected_target}")
            self.missions_to_process.append(selected_target)

    def insert_mission_node_in_dsr(self, active_mission_data):
        # Create plan node connected to robot node
        mission_str = str(active_mission_data["missions"])
        mission_node = self.g.get_node("mission")
        if mission_node is None:
            mission_node = Node(agent_id=self.agent_id, type="mission", name="mission")
            # Convert self.active mission to a string representation
            mission_node.attrs['plan_type'] = Attribute(str(active_mission_data["mission"]), self.agent_id)
            mission_node.attrs['plan_target'] = Attribute(str(active_mission_data["target"]), self.agent_id)
            mission_node.attrs['plan'] = Attribute(str(mission_str), self.agent_id)
            mission_node.attrs['pending_missions'] = Attribute(str(self.pending_missions), self.agent_id)
            robot_node = self.g.get_node(self.robot_id)
            robot_node_x_pose = robot_node.attrs["pos_x"].value
            robot_node_y_pose = robot_node.attrs["pos_y"].value
            if robot_node_x_pose is None or robot_node_y_pose is None:
                console.print("Robot node has no position attributes. Cannot create mission node.", style='red')
                return
            mission_node.attrs['pos_x'] = Attribute(float(robot_node_x_pose + 50), self.agent_id)
            mission_node.attrs['pos_y'] = Attribute(float(robot_node_y_pose + 50), self.agent_id)
            mission_node.attrs['parent'] = Attribute(int(self.robot_id), self.agent_id)
            robot_node_level = robot_node.attrs["level"].value
            if robot_node_level:
                mission_node.attrs['level'] = Attribute(robot_node_level + 1, self.agent_id)
                status = self.g.insert_node(mission_node)
                print("Mission node created with status:", status)
                mission_node = self.g.get_node("mission")
                has_mission_edge = Edge(mission_node.id, robot_node.id, "has", self.agent_id)
                result = self.g.insert_or_assign_edge(has_mission_edge)
                print("Has edge created with result:", result)
            else:
                console.print("Robot node has no level attribute. Cannot create mission node.", style='red')
                return
        else:
            mission_node.attrs['plan_type'] = Attribute(str(active_mission_data["mission"]), self.agent_id)
            mission_node.attrs['plan_target'] = Attribute(str(active_mission_data["target"]), self.agent_id)
            mission_node.attrs['plan'] = Attribute(str(mission_str), self.agent_id)
            self.g.update_node(mission_node)

    def set_active_mission(self, mission):
        mission_type = mission["mission"] if isinstance(mission, dict) else None
        match mission_type:
            case "goto":
                self.active_mission = GOTOMission(self.g, mission, self.agent_id)
            case "follow":
                self.active_mission = FollowMission(self.g, mission, self.agent_id)
            case "guide":
                self.active_mission = GuideMission(self.g, mission, self.agent_id)
            case "interact":
                self.active_mission = InteractMission(self.g, mission, self.agent_id)

    def create_mission(self, mission_data):
        missions = []
        robot_node = self.g.get_node(self.robot_id)
        mission_data["timestamp"] = robot_node.attrs["timestamp_alivetime"].value
        match mission_data["mission"]:
            case "goto":
                # Get current edges from graph
                current_edges = self.g.get_edges_by_type("current")
                if current_edges:
                    actual_room = self.g.get_node(current_edges[0].origin)
                    print("El robot está en la habitación:", actual_room.name)
                    try:
                        # Get the path to the selected room
                        path = self.long_term_graph.find_path_between_nodes(actual_room.name, mission_data["target"])
                        print("Path to selected room:", path)
                        # Structure as set of missions ("CROSS", "door_x_y_z")

                        for node in path:
                            # Considering node is "door_x_y_z", append to missions "("cross", "door", "x_y_z")"
                            if node.startswith("door_"):
                                # door_parts = node.split("_")
                                # if len(door_parts) >= 3:
                                #     door_name = "_".join(door_parts[1:])
                                missions.append({"mission" : "cross", "target" : node, "timestamp" : robot_node.attrs["timestamp_alivetime"].value})
                                # else:
                                #     print(f"Invalid door format: {node}")
                            else:
                                missions.append({"mission" : "goto", "target" : node, "timestamp" : robot_node.attrs["timestamp_alivetime"].value})
                    except:
                        console.print(f"Target not found in LTSM memory. Check if target is available.", style='green')
                else:
                    print("No hay ninguna habitación como current.")
            case "follow":
                person_node = self.g.get_node(str(mission_data["target"]))
                if person_node:
                    missions.append({"mission" : "follow", "target" : mission_data["target"], "timestamp" : robot_node.attrs["timestamp_alivetime"].value})
            case "interact":
                person_node = self.g.get_node(str(mission_data["target"]))
                if person_node:
                    missions.append({"mission" : "interact", "target" : mission_data["target"], "timestamp" : robot_node.attrs["timestamp_alivetime"].value})
        mission_data["missions"] = missions
        self.missions.append(mission_data)
            # case "guide":
            #     # Get current edges from graph
            #     person_node = self.g.get_node(str(target["person"]))
            #     if person_node:
            #         current_edges = self.g.get_edges_by_type("current")
            #         if current_edges:
            #             actual_room = self.g.get_node(current_edges[0].origin)
            #             print("El robot está en la habitación:", actual_room.name)
            #             # Get the path to the selected room
            #             path = self.long_term_graph.find_path_between_rooms(actual_room.name, target["destination"])
            #             print("Path to selected room:", path)
            #             # Structure as set of missions ("CROSS", "door_x_y_z")
            #
            #             for node in path:
            #                 # Considering node is "door_x_y_z", append to missions "("cross", "door", "x_y_z")"
            #                 if node.startswith("door_"):
            #                     door_parts = node.split("_")
            #                     if len(door_parts) >= 3:
            #                         door_name = "_".join(door_parts[1:])
            #                         missions.append(("follow", target["person"], robot_node.attrs["timestamp_alivetime"].value))
            #                         missions.append(("cross", "door", door_name, robot_node.attrs["timestamp_alivetime"].value))
            #
            #                     else:
            #                         print(f"Invalid door format: {node}")
            #             self.missions.append({"type": "guide", "target" : "room_2", "missions": missions})
            #         else:
            #             print("No hay ninguna habitación como current.")
            #
            #             missions.append(("follow", target["person"], robot_node.attrs["timestamp_alivetime"].value))
            #             self.missions.append({"type": "guide", "target" : "room_2", "missions": missions})

    def setting_waiting_mission(self, mission_data):
        mission_nodes = self.g.get_nodes_by_type("mission")
        for mission_node in mission_nodes:
            mission_value = mission_node.attrs["plan_type"].value
            target_value = mission_node.attrs["plan_target"].value
            if mission_value is not None and target_value is not None:
                if mission_value == mission_data["mission"] and target_value == mission_data["target"]:
                    print("Setting current mission as waiting")

    def abort_mission(self, set_in_pending=False):
        if self.active_mission:
            # Remove the mission node from the graph
            mission_node = self.g.get_node("mission")
            if mission_node:
                has_edge = self.g.get_edge(mission_node.id, self.robot_id, "has")
                if has_edge:
                    self.g.delete_edge(has_edge.origin, has_edge.destination, has_edge.type)
                    print("Has edge deleted from the graph.")
                self.g.delete_node(mission_node.id)
                self.active_mission.abort_mission(set_in_pending)
                print("Mission node deleted from the graph.")
                robot_node = self.g.get_node(self.robot_id)
                robot_timestamp = robot_node.attrs["timestamp_alivetime"].value if robot_node and "timestamp_alivetime" in robot_node.attrs else time.time()
                self.missions_history.append(
                        {"mission": self.active_mission.type, "target": self.active_mission.target, "init_timestamp" : self.active_mission.init_timestamp, "end_timestamp": robot_timestamp})
                print("Mission history:", self.missions_history)
                mission_history_node = self.g.get_node("mission_history")
                if mission_history_node:
                    mission_history_node.attrs['plan'] = Attribute(str(self.missions_history), self.agent_id)
                    mission_history_node.attrs['pending_missions'] = Attribute(str(list(self.pending_missions)), self.agent_id)
                    self.g.update_node(mission_history_node)
                    print("Mission history node updated in the graph.")
            else:
                print("No active mission to abort in graph.")
            self.active_mission = None
        else:
            print("No active mission to abort.")

    def check_intention_creation(self, destination_node):
        # TODO: probably check lambda counter
        if destination_node.type == "door":
            other_side_room = destination_node.attrs["connected_room_name"].value
            if other_side_room:
                door_name = destination_node.name
                # Check if any mission similar is active in self.active_mission.missions
                if self.active_mission and isinstance(self.active_mission, FollowMission):
                    print("Current submissions in FollowMission:", self.active_mission.missions)
                    if len([mission for mission in self.active_mission.missions if mission["mission"] == "cross" and mission["target"] == door_name]) > 0:
                        # print("Crossing to room:", other_side_room, "already in active mission.")
                        return
                    # Check if self.active_mission has a person_id value and check if the
                    # print("Person crossing to room:", other_side_room, "setting crossing to room as affordance")
                    robot_node = self.g.get_node(self.robot_id)
                    self.active_mission.insert_new_submission({"mission": "cross", "target": door_name, "timestamp": robot_node.attrs["timestamp_alivetime"].value})


                    # pending_mission = {"type": self.active_mission.mission_type, "target" : self.active_mission.person_id, "missions": self.active_mission.missions}
                    # self.pending_missions.append(pending_mission)
                    # self.abort_mission(True)
                    # self.missions_to_process.append(("goto", other_side_room))

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        if "LLM_response" in attribute_names:
            node = self.g.get_node(id)
            if node is not None:
                llm_response = node.attrs["LLM_response"].value
                # print(f"LLM response received: {llm_response}")
                llm_response_parsed = ast.literal_eval(llm_response)
                # Check if key "proposed_mission" exists in llm_response_parsed
                if "proposed_mission" in llm_response_parsed and all(k in llm_response_parsed["proposed_mission"] for k in self.required_proposed_mission_keys):
                    if llm_response_parsed["proposed_mission"]["mission"] in self.possible_missions:
                        mission_data = llm_response_parsed["proposed_mission"]
                        self.missions_to_process.append(mission_data)
                        console.print(f"New mission from LLM received: {mission_data}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        if type == "has_intention" and isinstance(self.active_mission, FollowMission):
            origin_node = self.g.get_node(fr)
            if origin_node.type == "person":
                if origin_node.attrs["person_id"].value == self.active_mission.person_id:
                    destination_node = self.g.get_node(to)
                    print("Followed person is interested in", destination_node.name, time.time())
                    self.has_intentions_to_check.append(destination_node)

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
