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
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 13
        self.g = DSRGraph(0, "LongTermSpatialMemory_agent", self.agent_id)

        try:
            #signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
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

            # Global map variables
            self.long_term_graph = LongTermGraph("graph.pkl")
            if self.long_term_graph.g:
                print("Graph exists")
                self.graph = self.long_term_graph.g
                self.long_term_graph.draw_graph(False)
                # Compute metric map and draw it
                g_map = self.long_term_graph.compute_metric_map("room_1")
                self.long_term_graph.draw_metric_map(g_map)
            else:
                print("Graph does not exist. Creating a new one")
                self.graph = ig.Graph()

            # In case the room node exists but the current edge is not set, set it
            room_nodes = self.g.get_nodes_by_type("room")
            current_room_nodes = [node for node in room_nodes if self.g.get_edge(node.id, node.id, "current")]
            if len(current_room_nodes) == 0 and len(room_nodes) == 1:
                print("Room node exists but no current edge. Setting as current")
                if not "measured" in room_nodes[0].name:
                    self.insert_current_edge(room_nodes[0].id)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
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
        # Check if graph exists
        if self.long_term_graph.g:
            # Get room with "current" edge
            current_edges = [edge for edge in self.g.get_edges_by_type("current") if self.g.get_node(edge.destination).type == "room" and self.g.get_node(edge.origin).type == "room"]
            if len(current_edges) == 1:
                actual_room_node = self.g.get_node(current_edges[0].origin)
                # Get robot pose
                robot_rt = self.rt_api.get_edge_RT(actual_room_node, self.robot_id)
                # Check if robot node exists in graph
                try:
                    robot_node = self.graph.vs.find(name=self.robot_name)
                    # Get robot antecessor
                    robot_node_successors = self.g.successors(robot_node)

                except:
                    print("No robot node found in igraph. Inserting")
                    robot_node_dsr = self.g.get_node(self.robot_name)
                    self.insert_igraph_vertex(robot_node_dsr)
                    self.insert_igraph_edge(robot_rt)
            self.long_term_graph.draw_graph(False)



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


    def idle(self):
        # Check if there is a node of type aff_cross and it has it's valid attribute to true using comprehension list
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
                # Load graph from file
                self.long_term_graph.g = self.long_term_graph.read_graph("graph.pkl")
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
        other_side_door_node = self.g.get_node(self.exit_door_id)
        other_side_room_name = other_side_door_node.attrs["connected_room_name"].value # TODO: Get directly the connected_room_name
        # Search in self.graph for the node with the name of the other side door
        print("known room", other_side_room_name)
        try:
            # Search in self.graph for the node with the room_id of the other side door
            other_side_door_room_node = self.graph.vs.find(name=other_side_room_name)
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
                    new_door_node = self.graph.vs.find(name=new_door_name)
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
        try:
            door_1_node = self.graph.vs.find(name=door_1[0])
        except:
            print("No door node found in igraph", door_1[0])
            return
        try:
            door_2_node = self.graph.vs.find(name=door_2[0])
        except:
            print("No door node found in igraph", door_2[0])
            return
        self.graph.add_edge(door_1_node, door_2_node)
        door_1_node["other_side_door_name"] = door_2[0]
        door_1_node["connected_room_name"] = door_2[1]
        door_2_node["other_side_door_name"] = door_1[0]
        door_2_node["connected_room_name"] = door_1[1]


    def store_graph(self):
        actual_room_node = self.g.get_node(self.room_exit_door_id)
        # Check if node in igraph with the same name exists
        try:
            room_node = self.graph.vs.find(name=actual_room_node.name)
            print("Room node found in igraph")
        except Exception as e:
            print("No room node found in igraph. Inserting room")
            self.traverse_graph(self.room_exit_door_id)

        # Save graph to file
        with open("graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

    def removing(self):
        # # Get last number in the name of the room
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
        node = self.g.get_node(node_id)
        rt_children = [edge for edge in self.g.get_edges_by_type("RT") if edge.origin == node_id and edge.destination != self.robot_id]
        self.insert_igraph_vertex(node)
        # Recur for all the vertices adjacent to this vertex
        for i in rt_children:
            self.traverse_graph(i.destination)
            self.insert_igraph_edge(i)

    def traverse_igraph(self, node):
        vertex_successors = self.graph.successors(node.index)
        # Recur for all the vertices adjacent to thisvertex
        for i in vertex_successors:
            sucessor = self.graph.vs[i]
            if sucessor["room_id"] == node["room_id"] and sucessor["level"] > node["level"]:
                self.insert_dsr_vertex(node["name"], sucessor)
                # Check if node with id i room_id attribute is the same as the room_id attribute of the room node
                self.insert_dsr_edge(node, sucessor)
                self.traverse_igraph(sucessor)
            else:
                continue

    def insert_igraph_vertex(self, node):
        self.graph.add_vertex(name=node.name, id=node.id, type=node.type)
        # print("Inserting vertex", node.name, node.id)
        for attr in node.attrs:
            if attr in self.not_required_attrs:
                continue
            self.graph.vs[self.vertex_size][attr] = node.attrs[attr].value
            # Check if current attribute is other_side_door_name and, if it has value, check if the node with that name exists in the graph
            if attr == "other_side_door_name" and node.attrs[attr].value:
                try:
                    print("Matched other_side_door_name", node.attrs[attr].value)
                    origin_node = self.graph.vs.find(id=node.id)
                    try:
                        other_side_door_node = self.graph.vs.find(name=node.attrs[attr].value)
                        try:
                            self.graph.add_edge(origin_node, other_side_door_node)
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
            #         connescted_room_node = self.graph.vs.find(name=node.attrs[attr].value)
            #         if connected_room_node:
            #             self.graph.add_edge(self.vertex_size, connected_room_node.id)
            #     except:
            #         print("No connected_room_name attribute found")
        self.vertex_size += 1

    def insert_dsr_vertex(self, parent_name, node):
        # print("Inserting vertex", node["name"], node["type"])
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
        origin_node = self.g.get_node(edge.origin)
        destination_node = self.g.get_node(edge.destination)

        # Search for the origin and destination nodes in the graph
        origin_node = self.graph.vs.find(name=origin_node.name)
        destination_node = self.graph.vs.find(name=destination_node.name)
        # Add the edge to the graph
        self.graph.add_edge(origin_node, destination_node, rt=edge.attrs["rt_translation"].value, rotation=edge.attrs["rt_rotation_euler_xyz"].value)
        # print("Inserting igraph edge", origin_node["name"], destination_node["name"])
        # print("RT", edge.attrs["rt_translation"].value)
        # print("Rotation", edge.attrs["rt_rotation_euler_xyz"].value)
        # Print origin and destination nodes

    def insert_dsr_edge(self, org, dest):
        # print("ORG::", org)
        # self.insert_dsr_vertex(dest)
        if org is None:
            root_node = self.g.get_node("root")
            org_id = root_node.id
            rt_value = [0, 0, 0]
            orientation = [0, 0, 0]
        else:
            # print("Inserting DSR edge", org["name"], dest["name"])
            edge_id = self.graph.get_eid(org.index, dest.index)
            edge = self.graph.es[edge_id]
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

    def draw_graph(self):
        self.ax.clear()
        # Obtener las coordenadas de los vértices
        layout = self.graph.layout("kamada_kawai")  # Utiliza el layout Kamada-Kawai
        # Dibujar los vértices
        x, y = zip(*layout)
        self.ax.scatter(x, y, s=100)  # Ajustar el tamaño de los vértices con el parámetro 's'
        # Dibujar las aristas
        for edge in self.graph.get_edgelist():
            # Print rt_translation attribute
            # Get edge data
            # edge_data = self.graph.es[self.graph.get_eid(edge[0], edge[1])]
            # print(edge_data["rt"])
            self.ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color="grey")
            # add arrow to the edge
            self.ax.annotate("", xy=(x[edge[1]], y[edge[1]]), xytext=(x[edge[0]], y[edge[0]]), arrowprops=dict(arrowstyle="->", lw=2))

        for i, txt in enumerate([f"Node {i}" for i in range(self.graph.vcount())]):
            # Get name attribute
            name = self.graph.vs[i]["name"]
            self.ax.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        # Adapt ax to the graph
        self.ax.set_xlim([min(x) - 2, max(x) + 2])
        self.ax.set_ylim([min(y) - 2, max(y) + 2])

    def check_element_room_number(self, node_id):
        node = self.g.get_node(node_id)
        try:
            room_id = node.attrs["room_id"].value
            return room_id
        except:
            # print("No room_id attribute found")
            return -1

    def check_element_level(self, node_id):
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
        current_edge = Edge(room_id, room_id, "current", self.agent_id)
        self.g.insert_or_assign_edge(current_edge)

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        if type == "door":
            # Get door name
            door_node = self.g.get_node(id)
            if not "pre" in door_node.name:
                # Check if door exists in igraph yet
                try:
                    door_igraph = self.graph.vs.find(name=door_node.name)
                    # print("Door node found in igraph. Returning.")
                    return
                except:
                    print("No door node found in igraph. Checking if room node exists")
                    # Get room node
                    room_id = door_node.attrs["room_id"].value
                    try:
                        room_node = self.graph.vs.find(name="room_" + str(room_id))
                        print("Room node found in igraph. Inserting door")
                        parent_id = door_node.attrs["parent"].value
                        print("Parent id", parent_id)
                        door_parent_node = self.g.get_node(parent_id)
                        print("Door parent name", door_parent_node.name)
                        # Insert door node in igraph
                        self.insert_igraph_vertex(door_node)
                        print("Door inserted in igraph")
                        # Get RT from door_parent to door
                        # rt_door = self.rt_api.get_edge_RT(door_parent_node, door_node.id)

                        rt_door = self.g.get_edge(door_parent_node.id, door_node.id, "RT")
                        print("RT DOOR", rt_door.attrs["rt_translation"].value, rt_door.attrs["rt_rotation_euler_xyz"].value)
                        print("Arigin name", door_parent_node.name, "Destination name", door_node.name)
                        print("IDS", door_parent_node.id, door_node.id)
                        self.insert_igraph_edge(rt_door)
                        with open("graph.pkl", "wb") as f:
                            pickle.dump(self.graph, f)
                        print("Door inserted in igraph")
                        self.long_term_graph.draw_graph(False)
                    except Exception as e:
                        print("No room node found in igraph. Not possible to insert door")
                        print(e)
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

        if to == self.robot_id and fr != self.room_exit_door_id and type == "RT" and len(self.g.get_edges_by_type("current")) == 0:
            print(self.room_exit_door_id)
            # if len(self.g.get_nodes_by_type("room")) == 1 and type == "room" and not "measured" in self.g.get_node(id).name:
            self.insert_current_edge(fr)
            print("Room node exists but no current edge. Setting as current")

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
