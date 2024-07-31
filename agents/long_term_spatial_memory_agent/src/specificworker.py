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
import igraph as ig
import interfaces as ifaces
import matplotlib.pyplot as plt
import time
import setproctitle
import math
import pickle
import subprocess
from collections import deque
import json

from long_term_graph import LongTermGraph

import cv2

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

try:
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
    print("Process title set to", os.path.basename(os.getcwd()))
except:
    pass

from pydsr import *

class SpecificWorker(GenericWorker):
    """
    Simulates a robot navigating through a virtual environment by traversing a
    graph, updating node and edge attributes, and performing tasks such as door
    associations, room creation, and path planning.

    Attributes:
        Period (None): 100 by default, indicating a period of time (in milliseconds)
            for computing tasks. It is used to set the timer interval for periodic
            computations.
        agent_id (int): 13, which represents the ID of a specific agent in the
            robotic environment. It is used as an identifier for the agent's
            internal graph.
        g (igraphGraph): Used to interact with the graph library igraph. It contains
            methods to insert, delete, update nodes and edges in the graph, as
            well as traverse the graph.
        startup_check (None|bool): Set by default to None. It is a simple flag
            that is used for checking if the startup process has completed successfully.
        rt_api (object): Used to get edge RT (Rotation and Translation) from the
            `room_exit_door_id` to the robot node.
        inner_api (object): Used for internal API operations, likely related to
            robot pose transformations, door connections, or other internal
            mechanisms. The exact functionality depends on the implementation details.
        testing (Attribute): 2. It seems to be used as a flag for testing mode,
            controlling certain behavior or actions within the class.
        robot_name (str): Used as a name for the robot node in the graph. It appears
            to be used to identify the robot's location within the graph.
        robot_id (int): 13, as initialized in the `__init__` method. It appears
            to be a unique identifier for the robot node in the graph.
        last_robot_pose (npfloat64): 3-dimensional, representing the last known
            pose (position and orientation) of the robot in the room. It is used
            to store the previous position of the robot before updating it with
            new data.
        robot_exit_pose (npndarray[npfloat64,1D]): 3-dimensional representing a
            pose (x, y, z) of the robot when exiting a room. It stores the final
            affordance pose transformed to the global reference system.
        state (str|None): Used to keep track of the current state of the worker,
            such as "idle", "crossing", "known_room", etc., which determines how
            it should behave in different situations.
        affordance_node_active_id (int): 13 by default, which corresponds to the
            agent's ID. This variable seems to keep track of the active affordance
            node in the graph.
        exit_door_id (int): Used as a reference to identify the ID of the door
            that represents the exit from a room.
        room_exit_door_id (int): 13, as specified in the class initialization
            method (`__init__`). It represents the ID of the exit door node in the
            graph.
        enter_room_node_id (int): Set when a new room is entered. It holds the ID
            of the node that represents the current room being explored by the robot.
        vertex_size (int): 0 initially. It increments by 1 each time a new vertex
            (node) is inserted into the graph, keeping track of the number of
            vertices created so far.
        not_required_attrs (List[str]): Used to store the names of attributes that
            are not required for vertex objects in the graph, such as "parent",
            "timestamp_alivetime", etc. These attributes are ignored when inserting
            vertices into the graph.
        last_save_time (float): Used to store the timestamp when the worker's state
            was last saved or updated.
        long_term_graph (igraphGraph): Used to store and manipulate the long-term
            graph data structure, which represents the agent's internal model of
            its environment.
        room_number (int): 1-based indexing for the room ID. It represents the
            unique identifier of a room node in the graph data structure.
        room_number_limit (None|int): 0 by default. It seems to be related to the
            limit of rooms in a generated apartment, used for procedural room generation.
        insert_current_edge (None): Used to set a current edge between two nodes
            in the graph, updating the room state to idle.
        pending_doors_to_stabilize (List[tuple[str,str]]): Initialized with a
            deque(maxlen=10) to store door names to be stabilized later. It keeps
            track of doors that need to be connected in the graph based on certain
            conditions.
        timer (QTimer): Used for scheduling the execution of a method after a
            certain time delay or interval (set by the `start` method).
        compute (None|Callable[[Any],Any]): Used as a slot for handling timer
            events. It contains a method that gets called at regular intervals,
            performing specific tasks depending on the state of the worker.
        update_node (None|int,str): Used to update a node with the given id and type.
        update_edge (NoneNonefrint,toint,typestr): Used to update edges in the
            graph, specifically when the edge's destination is the robot ID, source
            node is not the room exit door, type is "RT" and there are no current
            edges.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes various attributes, sets up graph data structures, and connects
        signals to methods for updating nodes and edges. It also starts a timer
        that triggers the compute method at regular intervals.

        Args:
            proxy_map (object): Passed to the superclass constructor using the
                `super()` function, indicating that it should be used for
                initialization. Its purpose and content are not specified within
                this code block.
            startup_check (bool): Set to False by default. It determines whether
                a startup check should be performed or not. If set to True, it
                calls the `startup_check` method; otherwise, it initializes the
                other parts of the class.

        """
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 13
        self.g = DSRGraph(0, "LongTermSpatialMemory_agent", self.agent_id)

        if startup_check:
            self.startup_check()
        else:
            self.rt_api = rt_api(self.g)
            self.inner_api = inner_api(self.g)

            self.testing = False

            # Robot node variables
            self.robot_name = "Shadow"
            self.robot_id = self.g.get_node(self.robot_name).id
            self.last_robot_pose = [0, 0, 0]
            self.robot_exit_pose = [0, 0, 0]

            # Variable for designing the state machine
            self.state = "idle"
            print("Starting in IDLE state")

            # ID variables
            self.affordance_node_active_id = None # Affordance node ID
            self.exit_door_id = None # Exit door node ID

            # Room nodes variables
            self.room_exit_door_id = -1 # Exit door node ID
            self.enter_room_node_id = None # Enter room node ID

            # Graph variables
            self.vertex_size = 0
            self.not_required_attrs = ["parent", "timestamp_alivetime", "timestamp_creation", "rt", "valid", "obj_checked", "name", "id"]

            self.last_save_time = time.time()
            # Global map variables
            self.long_term_graph = LongTermGraph("graph.pkl")
            # Check if self.long_term_graph.g is empty
            # if self.long_term_graph.g.vcount() != 0:
            #     # Draw graph from file
            #     self.long_term_graph.draw_graph(False)
            #     # Compute metric map and draw it
            #     g_map = self.long_term_graph.compute_metric_map("room_1")
            #     self.long_term_graph.draw_metric_map(g_map)
            #     self.initialize_room_from_igraph()
            #     self.update_robot_pose_in_igraph()

            if self.testing:
                # Get ground truth map from json
                root_node = self.g.get_node("root")
                path_attribute = root_node.attrs["path"].value
                if path_attribute:
                    # Get string between "generatedRooms/" and "/ApartmentFloorPlan.stl"
                    self.room_number = path_attribute.split("generatedRooms/")[1].split("/ApartmentFloorPlan.stl")[0]
                    print("Room number", self.room_number)
                    # Get data from apartmentData.json file in '/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms'
                    with open(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{self.room_number}/apartmentData.json") as f:
                        data = json.load(f)
                        self.room_number_limit = len(data["rooms"])
               # Print the number of rooms in the ground truth map
                print("Room number limit", self.room_number_limit)


            # In case the room node exists but the current edge is not set, set it
            room_nodes = self.g.get_nodes_by_type("room")
            current_room_nodes = [node for node in room_nodes if self.g.get_edge(node.id, node.id, "current")]
            if len(current_room_nodes) == 0 and len(room_nodes) == 1:
                print("Room node exists but no current edge. Setting as current")
                if not "measured" in room_nodes[0].name:
                    self.insert_current_edge(room_nodes[0].id)

            self.pending_doors_to_stabilize = deque(maxlen=10)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

            try:
                # signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
                signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
                # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
                signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
                # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
                # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
                console.print("signals connected")
            except RuntimeError as e:
                print(e)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        """
        Sets parameters for a scenario and performs subsequent actions on doors,
        including removing self-edges, storing door IDs, updating door attributes,
        and adding an attribute to an entrance door node.

        Args:
            params (Dict[any, any]): Expected to contain various parameters used
                for setting up the current room state in the game.

        Returns:
            bool: True.

        """
        return True



        # PROTOcode
        # Check if there is a node of type aff_cross and it has it's valid attribute to true
        # if so, check if aff_cross status attribute is completed, was active and robot pose is outside the room polygon,

        # - then remove "current" self-edge from the room
        # - Store in a variable the ID of the exit door
        # - wait until a new room is stabilized
        # - when new room is stabilized, check for the door used to get inside
        # - store in both doors the other_side_door name attribute
        # - Read exit door node and add an attribute other_side_door with the name of the entrance door in the new room
        # - Read entrance door node and add an attribute other_side_door with the name of the exit door in the new room
    @QtCore.Slot()
    def compute(self):
        # Exectue function self.update_robot_pose_in_igraph each second
        # if time.time() - self.last_save_time > 1:
        #     self.update_robot_pose_in_igraph()

        """
        Computes the position and orientation of rooms and doors, generates JSON
        data for each room, saves it to a file, and then terminates the process
        if testing mode is enabled.

        """
        if self.testing:
            # # Get room nodes number in igraph
            try:
                room_nodes = self.long_term_graph.g.vs.select(type_eq="room")
                door_nodes = self.long_term_graph.g.vs.select(type_eq="door")
                # Check if every door node has a "connected_room_name" attribute
                connected = True
                for door in door_nodes:
                     # Check if door has "connected_room_name" attribute and return if is None
                    if not door["other_side_door_name"]:
                        connected = False
                        print("Door", door["name"], "has no connected_room_name attribute")
                        break
                if len(room_nodes) == self.room_number_limit and connected:
                    dict = {}
                    # Create a dictionary with the room names. Every room name is a key which value is a dictionary with the room attributes
                    try:
                        dict["rooms"] = [{"name" : room["name"], "x" : room["width"], "y" : room["depth"], "room_id" : room["room_id"]} for room in room_nodes]

                        # For each room, insert an attribute with the room center in the global reference system
                        for room in dict["rooms"]:
                            room_center = self.long_term_graph.compute_element_pose(np.array([0., 0., 0.], dtype=np.float64), "room_1", room["name"])
                            print("ROOM CENTER POSE VALUE", room_center)
                            room["global_center"] = (room_center[0], room_center[1])
                            if (2 * math.pi/6)  < abs(room_center[2]) < (4*math.pi/6):
                                room["x"], room["y"] = room["y"], room["x"]
                        dict["doors"] = [{"name" : door["name"], "width" : door["width"], "room_id" : door["room_id"], "other_side_door_name" : door["other_side_door_name"], "connected_room_name" : door["connected_room_name"]} for door in self.long_term_graph.g.vs.select(type_eq="door")]
                        for door in dict["doors"]:
                            door_center = self.long_term_graph.compute_element_pose(np.array([0., 0., 0.], dtype=np.float64), "room_1", door["name"])
                            door["global_center"] = (door_center[0], door_center[1])
                            # Divide door name by "_" and get the second and the forth element to get the wall_id
                            wall_id = "wall_" + door["name"].split("_")[1] + "_" + door["name"].split("_")[3]
                            door_center = self.long_term_graph.compute_element_pose(np.array([0., 0., 0.], dtype=np.float64), wall_id, door["name"])
                            door["pose"] = (door_center[0], door_center[1])
                        print(dict)
                        # print(dict["doors"])
                        # Get file number in "/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{self.room_number} "
                        file_number = len([name for name in os.listdir(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{self.room_number}") if os.path.isfile(os.path.join(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{self.room_number}", name))])
                        # Save the dictionary in a .json file inside "tests" folder. The name of the file is dependent of the number of files in "tests"
                        with open(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{self.room_number}/generated_data_" + str(file_number) + ".json", "w+") as f:
                            json.dump(dict, f)
                        self.kill_everything()
                    except Exception as e:
                        print(e)
                        exit(0)

            except Exception as e:
                print(e)
                pass
        # print(self.state)
        match self.state:
            case "idle":
                self.idle()
            case "crossing":
                pass
            case "crossed":
                self.crossed()
            case "initializing_room":
                self.initializing_room()
            case "known_room":
                self.known_room()
            case "initializing_doors":
                self.initializing_doors()
            case "store_graph":
                self.store_graph()
            case "removing":
                self.removing()

    def initialize_room_from_igraph(self):
        # Remove RT edge between "root" and "Shadow"
        """
        Initializes a room structure from an igraph object, creates a root node
        and edges, updates node attributes, and inserts or assigns new edges into
        the graph.

        """
        self.g.delete_edge(100, self.robot_id, "RT")
        # Check in igraph the Shadow parent node
        robot_node = self.long_term_graph.g.vs.find(name=self.robot_name)
        robot_node_neighbors = self.long_term_graph.g.neighbors(robot_node)
        # Get first value and get node
        actual_room_igraph = self.long_term_graph.g.vs(robot_node_neighbors[0])[0]
        # Insert the room node in the DSR graph
        self.insert_dsr_vertex("root", actual_room_igraph)
        self.insert_dsr_edge(None, actual_room_igraph)
        self.traverse_igraph(actual_room_igraph)

        # Get room node from graph
        room_node = self.g.get_node(actual_room_igraph["name"])
        new_edge = Edge(self.robot_id, room_node.id, "RT", self.agent_id)
        # Get igraph edge from room to robot
        robot_rt_igraph = self.long_term_graph.g.es.find(_source=actual_room_igraph.index, _target=robot_node.index)
        # Get "translation" and "rotation" attributes
        rt_robot = robot_rt_igraph["traslation"]
        door_rotation = robot_rt_igraph["rotation"]
        new_edge.attrs["rt_translation"] = Attribute(np.array(rt_robot, dtype=np.float32), self.agent_id)
        # Get z rotation value and substract 180 degrees. then, keep the value between -pi and pi

        new_edge.attrs["rt_rotation_euler_xyz"] = Attribute(
            np.array(door_rotation, dtype=np.float32),
            self.agent_id)
        self.g.insert_or_assign_edge(new_edge)
        robot_node = self.g.get_node(self.robot_name)
        # Modify parent attribute of robot node
        robot_node.attrs["parent"] = Attribute(room_node.id, self.agent_id)
        self.g.update_node(robot_node)

    def update_robot_pose_in_igraph(self):
        # # Check if graph exists
        """
        Updates the robot's pose in an igraph graph by reflecting changes from a
        long-term graph to the current graph and saving the updated graph. It
        handles cases where the robot moves between rooms or when it is not found
        in the igraph.

        """
        if self.long_term_graph.g:
            # Get room with "current" edge
            current_edges = [edge for edge in self.g.get_edges_by_type("current") if self.g.get_node(edge.destination).type == "room" and self.g.get_node(edge.origin).type == "room"]
            if len(current_edges) == 1:
                actual_room_node = self.g.get_node(current_edges[0].origin)
                # Get robot pose
                robot_rt = self.rt_api.get_edge_RT(actual_room_node, self.robot_id)
                # Check if robot node exists in graph
                try:
                    robot_node = self.long_term_graph.g.vs.find(name=self.robot_name)
                    # Get robot antecessor
                    robot_node_neighbors = self.long_term_graph.g.neighbors(robot_node)
                    print("Robot node neighbors", robot_node_neighbors)
                    # Get first value and get node
                    actual_room_igraph = self.long_term_graph.g.vs.find(robot_node_neighbors[0])

                    try:
                        robot_rt_igraph = self.long_term_graph.g.es.find(_source=actual_room_igraph.index, _target=robot_node.index)
                        # Print rt and rotation values
                        print("RT edge from room to robot", actual_room_igraph["name"], robot_node["name"])
                        print("Robot RT", robot_rt_igraph["traslation"], robot_rt_igraph["rotation"])
                        if actual_room_igraph["name"] != actual_room_node.name:
                            print("Robot node is not in the same room as the robot")
                            # Get new room node
                            try:
                                new_room_node = self.long_term_graph.g.vs.find(name=actual_room_node.name)
                                self.long_term_graph.g.delete_edges([robot_rt_igraph])
                                self.insert_igraph_edge(robot_rt)
                            except:
                                print("No room node found in igraph. waiting")
                            # Get igraph edge
                        else:
                            pass
                        # Update rt data
                        robot_rt_igraph["traslation"] = robot_rt.attrs["rt_translation"].value
                        robot_rt_igraph["rotation"] = robot_rt.attrs["rt_rotation_euler_xyz"].value

                    except Exception as e:
                        print(e)
                except Exception as e:
                    print(e)
                    print("No robot node found in igraph. Inserting")
                    robot_node_dsr = self.g.get_node(self.robot_name)
                    self.insert_igraph_vertex(robot_node_dsr)
                    self.insert_igraph_edge(robot_rt)
            self.long_term_graph.draw_graph(False)
            # Save graph to file
            with open("graph.pkl", "wb") as f:
                pickle.dump(self.long_term_graph.g, f)
            self.last_save_time = time.time()

    def idle(self):
        # Check if there is a node of type aff_cross and it has it's valid attribute to true using comprehension list
        """
        Handles affordance nodes, stabilizes doors, and computes robot poses for
        crossing between rooms. It updates the graph state and prints relevant
        information during its execution.

        """
        aff_cross_nodes = [node for node in self.g.get_nodes_by_type("affordance") if node.attrs["active"].value == True]
        # Check if not empty
        if len(aff_cross_nodes) == 0 or len(aff_cross_nodes) > 1:
            # print("No aff_cross nodes with valid attribute or more than one valid affordance")
            return
        else:
            # Check if any "current" edge exists
            current_edges = [edge for edge in self.g.get_edges_by_type("current") if self.g.get_node(edge.destination).type == "room" and self.g.get_node(edge.origin).type == "room"]
            to_stabilize_doors = [node for node in self.g.get_nodes_by_type("door") if "pre" in node.name]
            if len(current_edges) == 1 and to_stabilize_doors == []:
                # From current edge, get the origin of the edge to get room node id
                self.room_exit_door_id = current_edges[0].origin
                exit_room_node = self.g.get_node(self.room_exit_door_id)
                # Store DSR graph in igraph
                self.store_graph()
                # Check if there are pending doors to stabilize
                print("Pending doors to stabilize", self.pending_doors_to_stabilize)
                if self.pending_doors_to_stabilize:
                    pending_door = self.pending_doors_to_stabilize.popleft()
                    print("Pending door to stabilize", pending_door)
                    # Try to associate
                    try:
                        self.associate_doors(pending_door[0], pending_door[1])
                    except:
                        print("Error associating doors. One of the doors was not found in the global map")
                        self.pending_doors_to_stabilize.append(pending_door)

                # Add data to json file

                # Draw graph from file
                self.long_term_graph.draw_graph(False)
                # Compute metric map and draw it
                g_map = self.long_term_graph.compute_metric_map("room_1")
                self.long_term_graph.draw_metric_map(g_map, exit_room_node.name)
                # Get affordance node
                self.affordance_node_active_id = aff_cross_nodes[0].id
                affordance_node = self.g.get_node(self.affordance_node_active_id)
                if not affordance_node.attrs["parent"].value:
                    print("Affordance node has no parent")
                    return
                else:
                    # Considering that the final affordance pose at crossing a door is at point at 1000 meters in the y axis
                    # in the normal to the door pointing to the center of the new room, transform this pose to the
                    # global reference system
                    self.exit_door_id = affordance_node.attrs["parent"].value
                    exit_door_node = self.g.get_node(self.exit_door_id)
                    final_robot_affordance_pose = self.inner_api.transform(exit_room_node.name,
                                                        np.array([0., 1000., 0.], dtype=np.float64),
                                                        exit_door_node.name)
                    wall_room_rt = self.rt_api.get_edge_RT(exit_room_node, exit_door_node.attrs["parent"].value)
                    wall_room_rotation = wall_room_rt.attrs["rt_rotation_euler_xyz"].value
                    robot_pose = [final_robot_affordance_pose[0], final_robot_affordance_pose[1], wall_room_rotation[2]]

                    # Transform final affordance pose to global reference
                    print("Final affordance pose in room reference", robot_pose)
                    final_robot_affordance_pose_in_room_reference = self.long_term_graph.compute_element_pose(robot_pose, "room_1",
                    exit_room_node.name)

                    print("Final affordance pose in global reference", final_robot_affordance_pose_in_room_reference)
                    pose_point = QPoint(final_robot_affordance_pose_in_room_reference[0],
                                         final_robot_affordance_pose_in_room_reference[1])

                    # Check if robot is in a room in the global map
                    other_side_room = self.long_term_graph.check_point_in_map(g_map, pose_point)
                    # Draw the transformed point in global map
                    second_point = QPoint(pose_point.x() - (250 * np.sin(final_robot_affordance_pose_in_room_reference[2])), pose_point.y() + (250 * np.cos(final_robot_affordance_pose_in_room_reference[2])))
                    print("Pose point", pose_point, "Second point", second_point)
                    self.long_term_graph.draw_point(pose_point, second_point)
                    # In case the robot is going to cross to a known room...
                    if other_side_room != None:
                        # Get exit door center pose
                        exit_door_pose = self.inner_api.transform(exit_room_node.name,
                                                                  exit_door_node.name)
                        # Convert to np.array
                        exit_door_pose = np.array(exit_door_pose, dtype=np.float32)
                        print("Exit door pose", exit_door_pose)
                        # Transform exit door pose to other_side_room reference
                        exit_door_in_room_reference = self.long_term_graph.compute_element_pose(exit_door_pose,
                                                                                              other_side_room,
                                                                                              exit_room_node.name)

                        self.robot_exit_pose = self.long_term_graph.compute_element_pose(robot_pose,
                                                                                              other_side_room,
                                                                                              exit_room_node.name)
                        # Get door nodes connected to room other_side_room
                        doors = self.long_term_graph.get_room_objects_transform_matrices_with_name(other_side_room, "door")
                        closer_pose = ("", np.finfo(np.float32).max) # Variable to set the closest door to the exit one
                        # Iterate over known room doors
                        for i in doors:
                            # Get difference pose between robot and door
                            door_pose = i[1].t
                            pose_difference = math.sqrt((exit_door_in_room_reference[0] - door_pose[0]) ** 2 + (
                                    exit_door_in_room_reference[1] - door_pose[1]) ** 2)
                            if pose_difference < closer_pose[1] and pose_difference < 1200:
                                closer_pose = (i[0], pose_difference)
                        if closer_pose[0] != "":
                            # Associate both doors data in igraph
                            self.associate_doors((closer_pose[0], other_side_room),
                                                 (exit_door_node.name, exit_room_node.name))
                        # Set to the exit door DSR node the attributes of the matched door in the new room
                        exit_door_node.attrs["other_side_door_name"] = Attribute(closer_pose[0], self.agent_id)
                        # Set to the exit door DSR node the connected room name
                        exit_door_node.attrs["connected_room_name"] = Attribute(other_side_room,
                                                                                self.agent_id)
                        self.g.update_node(exit_door_node)
                    self.state = "crossing"
                    print("CROSSING")
            else:
                print("No current room")
                return

    def crossed(self):
        # Get parent node of affordance node
        """
        Updates the room state and door connections based on the current affordance
        node and exit door node in the graph.

        """
        affordance_node = self.g.get_node(self.affordance_node_active_id)
        if not affordance_node.attrs["parent"].value:
            # print("Affordance node has no parent")
            return
        else:
            self.exit_door_id = affordance_node.attrs["parent"].value
            exit_door_id_node = self.g.get_node(self.exit_door_id)
            # Remove "current" self-edge from the room
            self.g.delete_edge(self.room_exit_door_id, self.room_exit_door_id, "current")
            if exit_door_id_node:
                try:
                    if exit_door_id_node.attrs["connected_room_name"].value:
                        self.state = "known_room"
                        print("INSERTING KNOWN ROOM")
                except:
                    self.state = "initializing_room"
                    print("INITIALIZING ROOM")

    def initializing_room(self):

        # Get room nodes
        """
        Retrieves a list of room nodes, selects the first node, sets it as the
        current room, and inserts an edge representing this change into a graph.
        The method then changes its internal state to "initializing_doors" and
        prints a status message.

        """
        room_nodes = [node for node in self.g.get_nodes_by_type("room") if node.id != self.room_exit_door_id and not "measured" in node.name]
        if len(room_nodes) == 0:
            # print("No room nodes different from the exit one found")
            return
        else:
            # Get the enter room node id
            self.enter_room_node_id = room_nodes[0].id
            self.insert_current_edge(self.enter_room_node_id)
            self.state = "initializing_doors"
            print("INITIALIZING DOORS")
    #
    def known_room(self):
        # Get other side door name attribute
        """
        Navigates from the current room to another connected room through an exit
        door, updates the graph representation of the new room and door, and adjusts
        the robot's position and orientation accordingly.

        """
        other_side_door_node = self.g.get_node(self.exit_door_id)
        other_side_room_name = other_side_door_node.attrs["connected_room_name"].value # TODO: Get directly the connected_room_name
        # Search in self.long_term_graph.g for the node with the name of the other side door
        print("known room", other_side_room_name)
        try:
            # Search in self.long_term_graph.g for the node with the room_id of the other side door
            other_side_door_room_node = self.long_term_graph.g.vs.find(name=other_side_room_name)
            print("other_side_room_graph_name", other_side_door_room_node["name"])

            # Insert the room node in the DSR graph
            self.insert_dsr_vertex("root", other_side_door_room_node)
            self.insert_dsr_edge(None, other_side_door_room_node)
            print("TRAVERSING GRAPH")
            self.traverse_igraph(other_side_door_room_node)
            print("TRAVERSED GRAPH")
            exit_room_node = self.g.get_node(self.room_exit_door_id)

            # Delete RT edge from room node t oShadow
            self.g.delete_edge(self.room_exit_door_id, self.robot_id, "RT")
            new_room_id = self.g.get_node(other_side_door_room_node["name"]).id
            new_edge = Edge(self.robot_id, new_room_id, "RT", self.agent_id)
            # Get new door name
            new_door_name = other_side_door_node.attrs["other_side_door_name"].value
            if new_door_name == "":
                print("No new door name found. Probably the associated door was not found in the global map. Take into account this as a future mission")
                print("Setting as objetive the last affordance pose transformed to the global reference system")

                # final_robot_affordance_pose = self.inner_api.transform(exit_room_node.name,
                #                                                        np.array([0., 1000., 0.], dtype=np.float64),
                #                                                        other_side_door_node.name)
                # wall_room_rt = self.rt_api.get_edge_RT(exit_room_node, exit_door_node.attrs["parent"].value)
                # wall_room_rotation = wall_room_rt.attrs["rt_rotation_euler_xyz"].value
                # # Append a 1 to the last_robot_pose array to make it a 3D array
                # final_robot_pose = [final_robot_affordance_pose[0], final_robot_affordance_pose[1], wall_room_rotation[2]]
                # # Transform final affordance pose to global reference
                # final_robot_affordance_pose_in_room_reference = self.long_term_graph.compute_element_pose(
                #     final_robot_pose, "room_1",
                #     exit_room_node.name)
                exit_robot_pose = self.robot_exit_pose
                rt_robot = np.array([exit_robot_pose[0], exit_robot_pose[1], 0.0], dtype=np.float64)
                door_rotation = [0., 0., exit_robot_pose[2]]

            else:
                print("Going out door", new_door_name)
                try:
                    new_door_node = self.long_term_graph.g.vs.find(name=new_door_name)
                    rt_robot = self.inner_api.transform(other_side_door_room_node["name"], np.array([0. , -1000., 0.], dtype=np.float64), new_door_node["name"])
                    door_node = self.g.get_node(new_door_node["name"])
                    door_parent_id = door_node.attrs["parent"].value
                    door_parent_node = self.g.get_node(door_parent_id)
                    print("Door parent name ", door_parent_node.name)
                    # get door parent node
                    # get rt from room node to door parent node
                    rt_room_wall = self.rt_api.get_edge_RT(self.g.get_node(other_side_door_room_node["name"]),
                                                           door_parent_id)
                    # get rt_rotation_euler_xyz from rt_room_wall
                    door_rotation = rt_room_wall.attrs["rt_rotation_euler_xyz"].value
                    new_z_value = (door_rotation[2] - math.pi)
                    if new_z_value > math.pi:
                        new_z_value = new_z_value - 2 * math.pi
                    elif new_z_value < -math.pi:
                        new_z_value = new_z_value + 2 * math.pi
                    door_rotation[2] = new_z_value
                    print("WALL ROTATION", door_rotation)
                except:
                    print("No door node found")
                    return


            new_edge.attrs["rt_translation"] = Attribute(np.array(rt_robot, dtype=np.float32), self.agent_id)
            # Get z rotation value and substract 180 degrees. then, keep the value between -pi and pi

            new_edge.attrs["rt_rotation_euler_xyz"] = Attribute(
                np.array(door_rotation, dtype=np.float32),
                self.agent_id)
            print("FIRST ROBOT RT", rt_robot, door_rotation)
            self.g.insert_or_assign_edge(new_edge)
            robot_node = self.g.get_node(self.robot_name)
            # Modify parent attribute of robot node
            robot_node.attrs["parent"] = Attribute(new_room_id, self.agent_id)
            self.g.update_node(robot_node)

            # Insert current edge
            self.insert_current_edge(new_room_id)


        except Exception as e:
            print("No other side door room node found")
            print(e)
            return
        self.state = "removing"

    def initializing_doors(self):
        # Check if node called "room_entry" of type room exists
        """
        Initializes and associates exit doors with their corresponding rooms,
        storing the door names and connected room names as attributes. It also
        updates the graph and sets a state to "removing".

        """
        exit_edges = [edge for edge in self.g.get_edges_by_type("exit") if edge.destination == self.exit_door_id]
        if len(exit_edges) > 0:
            # Check if edge of type "same" exists between door_entry and enter_room_node
            same_edges = self.g.get_edges_by_type("match")
            if len(same_edges) == 0:
                # print("No same edges found")
                return
            else:
                # Get the other side door id TODO: the edge comes from door_entry to nominal door (set in door_detector)
                other_side_door_id = same_edges[0].origin
                other_side_door_node = self.g.get_node(other_side_door_id)
                exit_door_id = same_edges[0].destination
                exit_door_node = self.g.get_node(exit_door_id)

                print(other_side_door_node.name, exit_door_node.name)

                # Read exit door node and add an attribute other_side_door with the name of the entrance door in the new room

                connected_room_name_exit_door = self.g.get_node(self.g.get_node(other_side_door_node.attrs["parent"].value).attrs["parent"].value).name
                connected_room_name_enter_door = self.g.get_node(self.room_exit_door_id).name

                exit_door_node.attrs["other_side_door_name"] = Attribute(other_side_door_node.name, self.agent_id)
                # Insert the last number in the name of the room to the connected_room_id attribute
                exit_door_node.attrs["connected_room_name"] = Attribute(connected_room_name_exit_door, self.agent_id)

                # Read entrance door node and add an attribute other_side_door with the name of the exit door in the new room

                other_side_door_node.attrs["other_side_door_name"] = Attribute(exit_door_node.name, self.agent_id)
                other_side_door_node.attrs["connected_room_name"] = Attribute(connected_room_name_enter_door, self.agent_id)
                self.g.update_node(exit_door_node)
                self.g.update_node(other_side_door_node)
                self.associate_doors((other_side_door_node.name, connected_room_name_exit_door), (exit_door_node.name, connected_room_name_enter_door))
                # Find each door in igraph
                self.state = "removing"

    def associate_doors(self, door_1, door_2):
        # Find each door in igraph and update attributes
        """
        Associates two doors by adding an edge between their corresponding nodes
        in the long-term graph, and sets additional attributes on each node to
        store information about the other door and connected room.

        Args:
            door_1 (Tuple[str, str]): Expected to represent a pair of door names
                with connected rooms' names as its elements.
            door_2 (Tuple[str, str]): Expected to contain the name of the second
                door and its connected room. It is used to find the node corresponding
                to this door in the graph and associate it with the first door.

        """
        try:
            door_1_node = self.long_term_graph.g.vs.find(name=door_1[0])
        except:
            print("No door node found in igraph", door_1[0])
            self.pending_doors_to_stabilize.append((door_1, door_2))
            print("Pending doors to stabilize", self.pending_doors_to_stabilize)
            return
        try:
            door_2_node = self.long_term_graph.g.vs.find(name=door_2[0])
        except:
            print("No door node found in igraph", door_2[0])
            self.pending_doors_to_stabilize.append((door_1, door_2))
            print("Pending doors to stabilize", self.pending_doors_to_stabilize)
            return
        self.long_term_graph.g.add_edge(door_1_node, door_2_node)
        door_1_node["other_side_door_name"] = door_2[0]
        door_1_node["connected_room_name"] = door_2[1]
        door_2_node["other_side_door_name"] = door_1[0]
        door_2_node["connected_room_name"] = door_1[1]


    def store_graph(self):
        """
        Stores the current state of a graph, represented by an igraph object, into
        a pickle file named "graph.pkl". It first retrieves and verifies the
        existence of a room node in the graph before storing it.

        """
        actual_room_node = self.g.get_node(self.room_exit_door_id)
        # Check if node in igraph with the same name exists
        try:
            room_node = self.long_term_graph.g.vs.find(name=actual_room_node.name)
            print("Room node found in igraph")
        except Exception as e:
            print("No room node found in igraph. Inserting room")
            self.traverse_graph(self.room_exit_door_id)

        # Save graph to file
        with open("graph.pkl", "wb") as f:
            pickle.dump(self.long_term_graph.g, f)

    def removing(self):
        # # Get last number in the name of the room
        """
        Deletes nodes and edges from the graph based on certain conditions. It
        first identifies nodes connected to a specified room exit door, then removes
        nodes with no outgoing edges and finally removes destinations of remaining
        edges sorted by their values in descending order.

        """
        room_number = self.g.get_node(self.room_exit_door_id).attrs["room_id"].value
        # # Get all RT edges
        rt_edges = self.g.get_edges_by_type("RT")
        # # Get all RT edges which last number in name is the same as the room number and generate a dictionary with the origin node as key and the origin node level as value
        old_room_rt_edges = [edge for edge in rt_edges if self.check_element_room_number(edge.origin) == room_number or self.check_element_room_number(edge.destination) == room_number]
        has_edges = self.g.get_edges_by_type("has")
        old_room_has_edges = [edge for edge in has_edges if self.check_element_room_number(edge.origin) == room_number]
        for edge in old_room_has_edges:
            self.g.delete_node(edge.destination)

        old_room_dict = {edge: int(self.check_element_room_number(edge.destination)) for edge in old_room_rt_edges}
        # Order dictionary by level value in descending order
        old_room_dict = dict(sorted(old_room_dict.items(), key=lambda item: item[1], reverse=True))
        # iterate over the dictionary in descending order
        for item in old_room_dict:
            # self.g.delete_node(self.g.get_node(item.origin).id)
            if item.origin == 200 or item.destination == 200:
                print("SHADOW NODES")
                continue

            self.g.delete_node(item.destination)
        self.long_term_graph.draw_graph(False)
        self.state = "idle"

    def traverse_graph(self, node_id):
        # Mark the current node as visited and print it
        """
        Recursively traverses a graph, starting from a specified node, and inserts
        nodes and edges into an igraph object. It identifies RT-type edges with
        destinations other than the robot ID and explores them further.

        Args:
            node_id (int): Expected to be an identifier for a node within the graph
                data structure (`self.g`).

        """
        node = self.g.get_node(node_id)
        rt_children = [edge for edge in self.g.get_edges_by_type("RT") if edge.origin == node_id and edge.destination != self.robot_id]
        self.insert_igraph_vertex(node)
        # Recur for all the vertices adjacent to this vertex
        for i in rt_children:
            self.traverse_graph(i.destination)
            self.insert_igraph_edge(i)

    def traverse_igraph(self, node):
        """
        Traverses the long-term graph, starting from a given node, and recursively
        explores its successors, inserting new vertices and edges into a data
        structure whenever it finds a suitable successor with higher level and
        same room_id as the current node.

        Args:
            node (igraph.vs): Passed to this function. It represents an individual
                vertex (node) within the graph and contains information about the
                node, such as its index, name, room_id, and level.

        """
        vertex_successors = self.long_term_graph.g.successors(node.index)
        # Recur for all the vertices adjacent to thisvertex
        for i in vertex_successors:
            sucessor = self.long_term_graph.g.vs[i]
            if sucessor["room_id"] == node["room_id"] and sucessor["level"] > node["level"]:
                self.insert_dsr_vertex(node["name"], sucessor)
                # Check if node with id i room_id attribute is the same as the room_id attribute of the room node
                self.insert_dsr_edge(node, sucessor)
                self.traverse_igraph(sucessor)
            else:
                continue

    def insert_igraph_vertex(self, node):
        """
        Adds a vertex to an igraph graph object, long_term_graph.g, with attributes
        from a given node object. It also checks for specific attributes and adds
        edges between vertices if necessary.

        Args:
            node (object): Assumed to have attributes such as `name`, `id`, `type`,
                and possibly others like `attrs`. The `node` parameter is used to
                add vertices to an igraph graph and also potentially establish
                edges between them.

        """
        self.long_term_graph.g.add_vertex(name=node.name, id=node.id, type=node.type)
        # print("Inserting vertex", node.name, node.id)
        for attr in node.attrs:
            if attr in self.not_required_attrs:
                continue
            self.long_term_graph.g.vs[self.vertex_size][attr] = node.attrs[attr].value
            # Check if current attribute is other_side_door_name and, if it has value, check if the node with that name exists in the graph
            if attr == "other_side_door_name" and node.attrs[attr].value:
                try:
                    print("Matched other_side_door_name", node.attrs[attr].value)
                    origin_node = self.long_term_graph.g.vs.find(id=node.id)
                    try:
                        other_side_door_node = self.long_term_graph.g.vs.find(name=node.attrs[attr].value)
                        try:
                            self.long_term_graph.g.add_edge(origin_node, other_side_door_node)
                            print("Matched other_side_door_name", node.attrs[attr].value, other_side_door_node)
                        except Exception as e:
                            print("No other_side_door_name node found", node.attrs[attr].value)
                            print(e)
                    except:
                        print("No other_side_door_name node found", node.attrs[attr].value)
                except:
                    print("No origin node found")

            # Check if current attribute is connected_room_name and, if it has value, check if the node with that name exists in the graph
            # if attr == "connected_room_name" and node.attrs[attr].value:
            #     try:
            #         connescted_room_node = self.long_term_graph.g.vs.find(name=node.attrs[attr].value)
            #         if connected_room_node:
            #             self.long_term_graph.g.add_edge(self.vertex_size, connected_room_node.id)
            #     except:
            #         print("No connected_room_name attribute found")
        self.vertex_size += 1

    def insert_dsr_vertex(self, parent_name, node):
        # print("Inserting vertex", node["name"], node["type"])
        """
        Inserts a new node into a graph, creating it from a given dictionary and
        linking it to an existing parent node with the provided name. It also
        copies over some attributes from the original node.

        Args:
            parent_name (str): Used to get an existing node from the graph using
                its name, which is then used as the parent for the new vertex being
                inserted into the graph.
            node (Dict): Expected to contain attributes such as "type", "name"
                that are used to create a new Node object. The values in this
                dictionary are used to set the properties of the newly created node.

        """
        new_node = Node(agent_id=self.agent_id, type=node["type"], name=node["name"])
        # Check if the node is a room node

        parent_node = self.g.get_node(parent_name)
        new_node.attrs['parent'] = Attribute(int(parent_node.id), self.agent_id)

        # Iterate over the attributes of the node
        for attr in node.attributes():
            if node[attr] is not None and attr not in self.not_required_attrs:
                # Add the attribute to the node
                new_node.attrs[attr] = Attribute(node[attr], self.agent_id)
        id_result = self.g.insert_node(new_node)
    def insert_igraph_edge(self, edge):
        """
        Inserts an edge into the long-term graph based on the origin and destination
        nodes, considering translation and rotation attributes from the provided
        edge. It also applies special handling for door destinations.

        Args:
            edge (object): Likely an instance of a class representing an edge in
                a graph, possibly containing information about its origin,
                destination, translation, rotation, and other attributes.

        """
        origin_node_dsr = self.g.get_node(edge.origin)
        destination_node_dsr = self.g.get_node(edge.destination)

        # Search for the origin and destination nodes in the graph
        origin_node = self.long_term_graph.g.vs.find(name=origin_node_dsr.name)
        destination_node = self.long_term_graph.g.vs.find(name=destination_node_dsr.name)
        # Add the edge to the graph
        if destination_node_dsr.name != self.robot_name:
            # Check if destination_node["type"] == "door"
            traslation = edge.attrs["rt_translation"].value
            if destination_node["type"] == "door":
                traslation[1] += 100
            self.long_term_graph.g.add_edge(origin_node, destination_node, rt=traslation, rotation=edge.attrs["rt_rotation_euler_xyz"].value)
        else:
            self.long_term_graph.g.add_edge(origin_node, destination_node, traslation=edge.attrs["rt_translation"].value, rotation=edge.attrs["rt_rotation_euler_xyz"].value)
        # print("Inserting igraph edge", origin_node["name"], destination_node["name"])
        # print("RT", edge.attrs["rt_translation"].value)
        # print("Rotation", edge.attrs["rt_rotation_euler_xyz"].value)
        # Print origin and destination nodes

    def insert_dsr_edge(self, org, dest):
        # print("ORG::", org)
        # self.insert_dsr_vertex(dest)
        """
        Inserts or updates an edge between two nodes in a graph based on whether
        one node (org) exists or not. It retrieves relevant data, creates a new
        edge with attributes and inserts/assigns it to the graph.

        Args:
            org (object | None): Used to specify whether a root node should be
                created or an existing node from which to create a new edge.
            dest (Dict[str, int]): Used to represent a destination node in the
                graph. It contains a "name" key with a string value representing
                the name of the node, and an "index" key with an integer value
                representing the index of the node.

        """
        if org is None:
            root_node = self.g.get_node("root")
            org_id = root_node.id
            rt_value = [0, 0, 0]
            orientation = [0, 0, 0]
        else:
            # print("Inserting DSR edge", org["name"], dest["name"])
            edge_id = self.long_term_graph.g.get_eid(org.index, dest.index)
            edge = self.long_term_graph.g.es[edge_id]
            rt_value = edge["rt"]
            orientation = edge["rotation"]
            org_name = org["name"]
            org_id = self.g.get_node(org_name).id


        # print("RT_VALUE", rt_value)
        # print(dest["name"], org["name"], "RT")
        dest_id = self.g.get_node(dest["name"]).id
        new_edge = Edge(dest_id, org_id, "RT", self.agent_id)
        new_edge.attrs["rt_translation"] = Attribute(np.array(rt_value, dtype=np.float32), self.agent_id)
        new_edge.attrs["rt_rotation_euler_xyz"] = Attribute(np.array(orientation, dtype=np.float32), self.agent_id)


        # print("RT", rt_value)
        # print("Rotation", orientation)
        self.g.insert_or_assign_edge(new_edge)

    def check_element_room_number(self, node_id):
        """
        Retrieves the room ID associated with a node identified by its node ID
        from a graph, and returns it if found; otherwise, it returns -1. The method
        handles potential exceptions that may occur during the retrieval process.

        Args:
            node_id (int | str): Required for this function. It represents an
                identifier that uniquely identifies a node within a graph data structure.

        Returns:
            str|int: Either a room ID associated with a node or -1 if no room ID
            is found for that node.

        """
        node = self.g.get_node(node_id)
        try:
            room_id = node.attrs["room_id"].value
            return room_id
        except:
            # print("No room_id attribute found")
            return -1

    def check_element_level(self, node_id):
        """
        Retrieves an element level from the attributes of a node with the given
        ID in the graph, and returns it if found; otherwise, prints an error message
        and returns -1.

        Args:
            node_id (object): Expected to be the ID of a node stored in the graph
                `self.g`. It is used to retrieve information about the node from
                the graph.

        Returns:
            int|str: 1) the value of "level" attribute of node with given id if
            present, or 2) -1 and a print statement indicating that no such attribute
            was found.

        """
        node = self.g.get_node(node_id)
        try:
            element_level = node.attrs["level"].value
            return element_level
        except:
            print("No element_level attribute found")
            return -1





        # # check it there is an active goto edge connecting the robot and a door
        #     # if so, then check the robot position. If it is at the door
        #         # signal that the current room is not anymore by removing self-edge
        #         # wait for a new room to be created
        #         # when new room is created, check for the first door in it.  Should be the door used to get in
        #         # connect both doors with a new edge "same"
        #         ### loop-closure  PARA LA SEGUNDA VUELTA  (gist es una imagen 360 de la habitación usada para reconocerla rápidamente)
        #         # check if the current room "gist" is similar to the gists of the other rooms in the agent's internal graph.
        #         # if so, then replace the current room by the matching nominal room in the internal graph
        #             # and adjust door connections
        # # if not, then check if there is a room not marked as "current" that has been there for at least 1 minute
        #     # if so, then replace the not-current room by a proxy empty room node with the doors as children nodes
        #         # save the old room in the agent's internal graph
        #         # save a gist of the room in the agent's internal graph
        #         # connect the door connecting both rooms with a new edge "same"
        #
        # # check it there is an active goto edge connecting the robot and a door
        # goto_edges = self.g.get_edges_by_type("goto")
        # if len(goto_edges) > 0:
        #     # search if one of the edges in goto_edges goes to a door
        #     for edge in goto_edges:
        #         if edge.fr.name == "robot" and edge.to.name == "door":
        #             # get door coordinates transformed to robot coordinates are smaller than 100mm
        #             door_coords_in_robot = self.inner_api(edge.fr.name, edge.to.name)
        #             # check that door_coords_in_robot are smaller than 100mm
        #             if np.sqrt(np.power(door_coords_in_robot[0], 2) + np.power(door_coords_in_robot[1], 2)) < 100:
        #                 # signal that the current room is not anymore by removing self-edge
        #                 self.g.remove_edge(edge.fr.name, edge.to.name, "current")
        #                 # wait for a new room to be created

    def generate_room_picture(self, room_node_id):

        """
        Retrieves information about a given room node, identifies edges of type
        "RT" that are connected to it, and extracts translation data from these edges.

        Args:
            room_node_id (int): Used to retrieve a specific node from a graph
                object (`self.g`). This node represents a room, and its id is
                needed to generate a picture for this room.

        """
        room_node = self.g.get_node(room_node_id)
        # Get room node room_id attribute
        room_id = room_node.attrs["room_id"].value
        # Get RT edges from the room node
        old_room_rt_edges = self.g.get_edges_by_type("RT")
        #Iterate over the RT edges
        for edge in old_room_rt_edges:
            print(edge.origin, edge.destination)
            # Get destination node
            origin_node = self.g.get_node(edge.origin)
            # Get room_id attribute
            try:
                print(origin_node.attrs)
                origin_room_id = origin_node.attrs["room_id"].value
                # Check if the room_id attribute is the same as the room_id attribute of the room node
                if origin_room_id == room_id:
                    # Get translation attribute
                    translation = edge.attrs["rt_translation"].value
                    print(translation)
            except:
                print("No room_id attribute found")


        # # Create a black image which size is proportional to the room size
        # room_image = np.zeros((int(corners[1] - corners[3]), int(corners[0] - corners[2]), 3), np.uint8)
        # # Draw the room polygon
        # cv2.fillPoly(room_image, [np.array(corners, np.int32)], (255, 255, 255))
        # # Draw the doors
        # for door in doors:
        #     cv2.circle(room_image, (int(door[0]), int(door[1])), 5, (0, 0, 255), -1)
        # # Save the image
        # cv2.imwrite("room_image.jpg", room_image)

    def insert_current_edge(self, room_id):
        # Insert current edge to the room
        """
        Inserts or updates an edge in a graph (self.g). The edge represents a
        current connection between two rooms with given IDs, identified by the
        agent ID. It is created based on room_id and its mirrored version.

        Args:
            room_id (int): Required for the creation of an `Edge` object. It
                represents the ID of a room that will be part of the edge being
                inserted into the graph.

        """
        current_edge = Edge(room_id, room_id, "current", self.agent_id)
        self.g.insert_or_assign_edge(current_edge)

    def kill_everything(self):
        # Ruta al script que deseas ejecutar
        """
        Executes a shell script, sets its executable permissions, runs it with the
        argument 'true', and captures any output or errors.

        """
        script_path = '/home/robocomp/robocomp/components/robocomp-shadow/tools/room_and_door_kill_and_restart_dsr.sh'  # Asegúrate de que el script tenga permisos de ejecución
        # Ejecutar el script
        subprocess.run(['chmod', '+x', script_path])
        result = subprocess.run([script_path, 'true'], capture_output=True, text=True)
        # Imprimir la salida del scriptprint('Salida del script:')print(result.stdout)
        # Imprimir los errores (si los hay)
        if result.stderr:
            print('Errores del script:')
            print(result.stderr)

    # def add_data_to_json(self, room_node_id):

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        """
        Updates the node with given id and type in the graph. If the type is "door",
        it inserts the door node into an igraph graph and adds edges between nodes.
        If the id matches the affordance node active id, it checks the state of
        the affordance node and changes its state if necessary.

        Args:
            id (int): Used as an identifier for nodes in a graph, specifically to
                identify either a door node or an affordance node based on the
                value of the `type` parameter.
            type (str): Used to determine what action should be taken when updating
                a node with the given id. The valid values for this parameter are
                "door" or other types that may be added in the future.

        """
        if type == "door":
            # Get door name
            door_node = self.g.get_node(id)
            if not "pre" in door_node.name:
                # Check if door exists in igraph yet
                try:
                    door_igraph = self.long_term_graph.g.vs.find(name=door_node.name)
                    # print("Door node found in igraph. Returning.")
                    return
                except:
                    # print("No door node found in igraph. Checking if room node exists")
                    # Get room node
                    room_id = door_node.attrs["room_id"].value
                    try:
                        room_node = self.long_term_graph.g.vs.find(name="room_" + str(room_id))
                        print("Room node found in igraph. Inserting door")
                        parent_id = door_node.attrs["parent"].value
                        # print("Parent id", parent_id)
                        door_parent_node = self.g.get_node(parent_id)
                        # print("Door parent name", door_parent_node.name)
                        # Insert door node in igraph
                        self.insert_igraph_vertex(door_node)
                        # print("Door inserted in igraph")
                        # Get RT from door_parent to door
                        # rt_door = self.rt_api.get_edge_RT(door_parent_node, door_node.id)

                        rt_door = self.g.get_edge(door_parent_node.id, door_node.id, "RT")
                        # print("RT DOOR", rt_door.attrs["rt_translation"].value, rt_door.attrs["rt_rotation_euler_xyz"].value)
                        # print("Arigin name", door_parent_node.name, "Destination name", door_node.name)
                        # print("IDS", door_parent_node.id, door_node.id)
                        self.insert_igraph_edge(rt_door)
                        with open("graph.pkl", "wb") as f:
                            pickle.dump(self.long_term_graph.g, f)
                        # print("Door inserted in igraph")
                        self.long_term_graph.draw_graph(False)
                    except Exception as e:
                        # print("No room node found in igraph. Not possible to insert door")
                        # print(e)
                        return

        if id == self.affordance_node_active_id:
            print("Affordance node is active")
            affordance_node = self.g.get_node(id)
            print(affordance_node.attrs["bt_state"].value, affordance_node.attrs["active"].value)
            if affordance_node.attrs["bt_state"].value == "completed" and affordance_node.attrs[
                "active"].value == False:
                print("Affordance node is completed and not active. Go to crossed state")
                # Remove "current" self-edge from the room
                self.state = "crossed"


    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        # Insert current edge to the room
        # TODO: Posible problema si tocas el nodo room en la interfaz gráfica

        """
        Updates the current edge when a specific condition is met: if there are
        no current edges, the update edge originates from the room exit door, and
        its type is "RT". It inserts the edge into the graph and prints a message.

        Args:
            fr (int): Likely to represent the from ID, indicating the starting
                point of an edge in the graph.
            to (int): Used to specify the destination of an edge in a graph, which
                represents the ID of another node in the graph.
            type (str): Expected to be set with value "RT". This indicates that
                it is an exit door edge, which leads to the robot's current room.

        """
        if to == self.robot_id and fr != self.room_exit_door_id and type == "RT" and len(self.g.get_edges_by_type("current")) == 0:
            print(self.room_exit_door_id)
            # if len(self.g.get_nodes_by_type("room")) == 1 and type == "room" and not "measured" in self.g.get_node(id).name:
            self.insert_current_edge(fr)
            print("Room node exists but no current edge. Setting as current")

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
