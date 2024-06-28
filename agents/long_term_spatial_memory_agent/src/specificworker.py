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
    Manages and updates a specific agent's internal graph, handling various events
    such as node and edge creation, deletion, and attribute changes. It also
    generates room images and performs startup checks.

    Attributes:
        Period (int): 200, indicating that the worker will take a break every 200
            seconds (3 minutes) before continuing to work.
        agent_id (int): Used to identify the agent that owns the graph. It is used
            to determine which edges are current edges and which nodes to update
            when updating node attributes or adding new nodes and edges.
        g (Graph): Used to represent the graph of the environment. It stores all
            the nodes, edges, and attributes of the environment, and is used for
            traversing, searching, and manipulating the environment.
        update_node (int): Used to update the state of a node based on its type,
            such as "completed" or "crossed". It takes two arguments: `id` (int)
            and `type` (str), which represent the ID of the node and the type of
            update respectively.
        update_edge (Edge): Used to update the edge attributes based on the type
            of edge and the specified attribute names. It updates the edge attributes
            by setting the `type` field of the Edge object to the specified type
            and updating the `attribute_names` field with the desired attribute names.
        startup_check (QTimersingleShot): Used to start the quit process after 200
            milliseconds of starting the worker.
        rt_api (str): Used to specify the API for real-time updates on the worker's
            graph.
        inner_api (internal): Used to access the inner API of the agent's environment,
            which allows for retrieving information about the current room, doors,
            and other nodes in the graph.
        room_initialized (int): 0 by default, indicating that the room has not
            been initialized yet. It is used to keep track of whether a room has
            been initialized or not.
        robot_name (str): Used to store the name of the robot that the worker is
            controlling.
        robot_id (int): Used to represent the ID of the robot that the worker is
            associated with.
        state (str): Used to store the current state of the worker (either "idle",
            "running", or "crossed").
        affordance_node_active_id (int): Used to store the ID of the node representing
            the affordance of the robot, which indicates whether the robot is
            active or not.
        room_exit_door_id (int): 4 by default, representing the ID of the door
            through which the agent exits a room. It's used to identify the current
            room in the agent's internal graph when no other information is available.
        exit_room_node_id (int): 4 by default, representing the ID of the room
            node that the agent exits when it reaches the end of a room.
        enter_room_node_id (int): Used to store the ID of the room node that the
            worker enters when it completes its task.
        exit_door_id (int): 0 by default, which represents the door that leads out
            of the current room in the agent's internal graph. It is used to
            determine when a new room should be created and the old one deleted.
        graph (Graph): Used to represent the graph of a specific worker's environment,
            containing nodes and edges that represent the worker's location and
            movements within the environment.
        vertex_size (int): 1 by default, which represents the size of each vertex
            in the graph. It can be adjusted to control the number of vertices in
            the graph.
        not_required_attrs (list): Used to specify a list of attributes that are
            not required for the worker's functionality, but can still be useful
            for debugging or other purposes.
        fig (int): 0 by default, indicating that the worker does not have a figure.
        ax (bt_state): Used to store the affordance node's BT state, which indicates
            whether the node has completed its task or not.
        insert_current_edge (Edge): Used to insert a new edge into the graph
            connecting the current room node with itself, indicating that the agent
            is currently in this room.
        timer (QTimer): Used to schedule a call to the `startup_check` method after
            200 milliseconds of starting the worker.
        compute (instance): Used to compute the similarity between a room and the
            gists of other rooms in the agent's internal graph, adjusting door
            connections accordingly.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Sets up various components and connections for an instance of the
        `SpecificWorker` class, including creating a graph, setting properties and
        timers, and connecting to signals for updates and removals.

        Args:
            proxy_map (igraphGraph): Used to map the original graph to a proxy
                graph, which allows for efficient computation of graph properties
                without affecting the original graph.
            startup_check (bool): Used to check if the agent has already started
                up, avoiding unnecessary startup checks when the agent has already
                run.

        """
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
            self.room_initialized = False
            self.robot_name = "Shadow"
            self.robot_id = self.g.get_node(self.robot_name).id


            # self.states = ["idle", "crossing", "crossed", "initializing_room", "new_room", "initializing_doors", "removing"]
            self.state = "idle"
            print("IDLE")
            # self.state = "removing"
            self.affordance_node_active_id = None
            self.room_exit_door_id = -1
            # self.room_exit_door_id = 183581510069125123
            self.exit_room_node_id = None
            self.enter_room_node_id = None
            self.exit_door_id = None

            self.graph = ig.Graph()
            self.vertex_size = 0
            self.not_required_attrs = ["parent", "timestamp_alivetime", "timestamp_creation", "rt", "valid", "obj_checked", "name", "id"]
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Check if there is a current room node in the graph
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
        """
        Sets parameters for a specific worker instance, removes a self-edge from
        a room, stores ID of an exit door, and adds attributes to both doors.

        Args:
            params (object): Used to set parameters for the function's execution.

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
        # rt_edges = self.g.get_edges_by_type("RT")
        # if len(rt_edges) > 0:
        #     print("RT edges")
        #     for edge in rt_edges:
        #         # get node of edge.origin and edge.destination
        #         origin_node = self.g.get_node(edge.origin)
        #         destination_node = self.g.get_node(edge.destination)
        #         #check if the origin node is a room and the destination node is not none
        #         if origin_node is not None and destination_node is not None:
        #             print(self.g.get_node(edge.origin).name, self.g.get_node(edge.destination).name)
        #             #get level of the origin node and the destination node
        #             origin_level = origin_node.attrs["level"].value
        #             destination_level = origin_node.attrs["level"].value
        #             #check if the origin level is not none and the destination level is not none
        #             if origin_level is not None and destination_level is not None:
        #                 print(origin_level, destination_level)

        """
        Determines the current state of a worker based on its graph and updates
        the state accordingly.

        """
        match self.state:
            case "idle":
                self.idle()
            case "crossing":
                self.crossing()
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
        """
        Determines if there is a room exit door id and sets the state to "crossing"
        if there is only one current room edge.

        """
        aff_cross_nodes = [node for node in self.g.get_nodes_by_type("affordance") if node.attrs["active"].value == True]
        # Check if not empty
        if len(aff_cross_nodes) == 0 or len(aff_cross_nodes) > 1:
            # print("No aff_cross nodes with valid attribute or more than one valid affordance")
            return
        else:
            current_edges = [edge for edge in self.g.get_edges_by_type("current") if self.g.get_node(edge.destination).type == "room" and self.g.get_node(edge.origin).type == "room"]
            if len(current_edges) == 1:
                self.room_exit_door_id = current_edges[0].origin
                print("Room exit door id", self.room_exit_door_id)
                # self.generate_room_picture(self.room_exit_door_id)
                self.affordance_node_active_id = aff_cross_nodes[0].id
                self.state = "crossing"
                print("CROSSING")
            else:
                print("No current room")
                return

    def crossing(self):
        """
        Performs actions when the worker reaches a certain state, including printing
        messages and setting the worker's state to "crossed".

        """
        pass
        # if self.g.get_edge(self.room_exit_door_id, self.room_exit_door_id, "current") is not None:
        #     print("Removing current edge from room")
        #     print(self.room_exit_door_id)
        #     self.g.delete_edge(self.room_exit_door_id, self.room_exit_door_id, "current")
        # Check if affordance_node has status attribute completed and is not active
        # affordance_node = self.g.get_node(self.affordance_node_active_id)
        # if affordance_node.attrs["bt_state"].value == "completed" and affordance_node.attrs["active"].value == False:
        #     print("Affordance node is completed and not active. Go to crossed state")
        #     # Remove "current" self-edge from the room
        #     self.state = "crossed"
        #     print("CROSSED")

    def crossed(self):
        # Get parent node of affordance node
        """
        Within the `SpecificWorker` class updates the state of the worker based
        on the value of an edge attribute and performs actions depending on the
        current state.

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

            # # Check if ""exit_door has an attribute value called "other_side_door_name"
            # print(self.g.get_node(self.exit_door_id).attrs["other_side_door_name"])
            # if not self.g.get_node(self.exit_door_id).attrs["other_side_door_name"].value:
            #     self.state = "initializing_room"
            #     print("INITIALIZING ROOM")
            # else:
            #     self.state = "known_room"
            #     print("INSERTING KNOWN ROOM")
            # Print attributes
            print(exit_door_id_node.attrs)
            if exit_door_id_node:
                try:
                    if exit_door_id_node.attrs["other_side_door_name"].value:
                        self.state = "known_room"
                        print("INSERTING KNOWN ROOM")

                except:
                    self.state = "initializing_room"
                    print("INITIALIZING ROOM")

    def initializing_room(self):

        # Get room nodes
        """
        Searches for nodes with certain names and IDs within the graph, then sets
        its enter room node ID to the first matching node's ID and inserts a current
        edge to it. Finally, it updates its state to "initializing doors."

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
        In the `SpecificWorker` class:
        1/ Finds the other side door node in the graph.
        2/ Creates a new edge and vertex in the graph to represent the robot's
        movement through the door.
        3/ Updates the robot's parent attribute with the room ID of the other side
        door node.

        """
        other_side_door_node = self.g.get_node(self.exit_door_id)
        other_side_door_name = other_side_door_node.attrs["other_side_door_name"].value # TODO: Get directly the connected_room_name

        # Search in self.graph for the node with the name of the other side door
        try:
            new_door_node = self.graph.vs.find(name=other_side_door_name)
            # Get room_id attribute of the other side door node
            new_door_room_id = new_door_node["room_id"]
            print("new_door_room_id", new_door_room_id)
            try:
                # Search in self.graph for the node with the room_id of the other side door
                other_side_door_room_node = self.graph.vs.find(name="room_"+str(new_door_room_id))
                other_side_door_room_node_index = other_side_door_room_node.index
                print("other_side_room_graph_name", other_side_door_room_node["name"])

                # Insert the room node in the DSR graph
                self.insert_dsr_vertex("root", other_side_door_room_node)
                self.insert_dsr_edge(None, other_side_door_room_node)

                self.traverse_igraph(other_side_door_room_node)

                # Delete RT edge from room node t oShadow
                self.g.delete_edge(self.room_exit_door_id, self.robot_id, "RT")
                new_room_id = self.g.get_node(other_side_door_room_node["name"]).id
                new_edge = Edge(self.robot_id, new_room_id, "RT", self.agent_id)

                rt_robot = self.inner_api.transform(other_side_door_room_node["name"], np.array([0. , -1000., 0.], dtype=np.float64), new_door_node["name"])
                print("sale por la puta puerta", new_door_node["name"])
                door_node = self.g.get_node(new_door_node["name"])
                door_parent_id = door_node.attrs["parent"].value
                door_parent_node = self.g.get_node(door_parent_id)
                print("Door parent name ", door_parent_node.name)
                # get door parent node
                # get rt from room node to door parent node
                rt_room_wall = self.rt_api.get_edge_RT(self.g.get_node(other_side_door_room_node["name"]), door_parent_id)
                # get rt_rotation_euler_xyz from rt_room_wall
                door_rotation = rt_room_wall.attrs["rt_rotation_euler_xyz"].value
                print("WALL ROTATION", door_rotation)
                new_edge.attrs["rt_translation"] = Attribute(np.array(rt_robot, dtype=np.float32), self.agent_id)
                # Get z rotation value and substract 180 degrees. then, keep the value between -pi and pi
                new_z_value = (door_rotation[2] - math.pi)
                if new_z_value > math.pi:
                    new_z_value = new_z_value - 2 * math.pi
                elif new_z_value < -math.pi:
                    new_z_value = new_z_value + 2 * math.pi



                new_edge.attrs["rt_rotation_euler_xyz"] = Attribute(np.array([door_rotation[0], door_rotation[1], new_z_value], dtype=np.float32),
                                                                    self.agent_id)
                print("FIRST ROBOT RT", rt_robot, [door_rotation[0], door_rotation[1], new_z_value])
                self.g.insert_or_assign_edge(new_edge)
                robot_node = self.g.get_node(self.robot_name)
                # Modify parent attribute of robot node
                robot_node.attrs["parent"] = Attribute(new_room_id, self.agent_id)
                self.g.update_node(robot_node)

                # Insert current edge
                self.insert_current_edge(new_room_id)
                # vertex_successors = self.graph.neighbors(other_side_door_room_node_index, mode="out")
                # for i in vertex_successors:
                #     sucessor = self.graph.vs[i]
                #     print("SUCESSOR:", sucessor["name"], sucessor["id"], sucessor["type"])
                #     self.insert_dsr_vertex(sucessor)
                #
                #     # Check if node with id i room_id attribute is the same as the room_id attribute of the room node
                #     if sucessor["room_id"] == other_side_door_room_node["room_id"]:
                #         self.insert_dsr_edge(other_side_door_room_node, sucessor)
                #         self.traverse_igraph(sucessor)
                #     else:
                #         continue

            except Exception as e:
                print("No other side door room node found")
                print(e)
                return
        except Exception as e:
            print("No other side door node found")
            print(e)
            return
        self.state = "store_graph"

    def initializing_doors(self):

        # Check if node called "room_entry" of type room exists
        """
        Determines the connected rooms of an agent, based on its exit door ID, and
        sets the other side door ID, room name, and connected room name attributes
        of the exit door node and its adjacent nodes in the graph.

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

                exit_door_node.attrs["other_side_door_name"] = Attribute(other_side_door_node.name, self.agent_id)
                # Insert the last number in the name of the room to the connected_room_id attribute
                exit_door_node.attrs["connected_room_name"] = Attribute(self.g.get_node(self.g.get_node(other_side_door_node.attrs["parent"].value).attrs["parent"].value).name, self.agent_id)

                # Read entrance door node and add an attribute other_side_door with the name of the exit door in the new room

                other_side_door_node.attrs["other_side_door_name"] = Attribute(exit_door_node.name, self.agent_id)
                other_side_door_node.attrs["connected_room_name"] = Attribute(self.g.get_node(self.room_exit_door_id).name, self.agent_id)
                self.g.update_node(exit_door_node)
                self.g.update_node(other_side_door_node)
                self.state = "store_graph"


    # def initializing_doors(self):
    #     print("INITIALIZING DOORS")
    #     # Check if node called "room_entry" of type room exists
    #     door_entry_node = self.g.get_node("door_entry")
    #     if door_entry_node and door_entry_node.type == "door":
    #         # Check if edge of type "same" exists between door_entry and enter_room_node
    #         same_edges = self.g.get_edges_by_type("same")
    #         if len(same_edges) == 0:
    #             print("No same edges found")
    #             return
    #         else:
    #             # Get the other side door id TODO: the edge comes from door_entry to nominal door (set in door_detector)
    #             other_side_door_id = same_edges[0].to
    #             # Read exit door node and add an attribute other_side_door with the name of the entrance door in the new room
    #             exit_door_node = self.g.get_node(self.exit_door_id)
    #             exit_door_node.attrs["other_side_door"] = other_side_door_id
    #             # Read entrance door node and add an attribute other_side_door with the name of the exit door in the new room
    #             enter_door_node = self.g.get_node(self.enter_room_node_id)
    #             enter_door_node.attrs["other_side_door"] = self.exit_door_id
    #             self.g.update_node(exit_door_node)
    #             self.g.update_node(enter_door_node)
    #             self.state = "removing"

    def store_graph(self):
        """
        Is part of the `SpecificWorker` class, which inherits from `GenericWorker`.
        The function stores the graph represented by an igraph object in the
        `self.g` variable, and then tries to find a node in the graph with the
        specified `room_exit_door_id` using `vs.find()`. If the node is found, the
        function prints a message and returns. If the node is not found, the
        function inserts a new room node into the graph using `traverse_graph()`
        and then prints a message before drawing the updated graph using `draw_graph()`.
        Finally, the function sets its state to "removing".

        """
        actual_room_node = self.g.get_node(self.room_exit_door_id)
        # Check if node in igraph with the same name exists
        try:
            room_node = self.graph.vs.find(name=actual_room_node.name)
            print("Room node found in igraph")
        except Exception as e:
            print("No room node found in igraph. Inserting room")
            self.traverse_graph(self.room_exit_door_id)
        self.draw_graph()

        self.state = "removing"
        print("REMOVING")

    def removing(self):
        # # Get last number in the name of the room
        """
        Removes RT and has edges that lead to a specific room (room number determined
        by `room_exit_door_id`) and deletes the corresponding nodes in the graph.
        It also updates a dictionary of old room numbers for each node in the graph.

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
            # self.g.delete_edge(item.origin, item.destination, "RT")
        self.state = "idle"

    def traverse_graph(self, node_id):
        # Mark the current node as visited and print it
        """
        Within `SpecificWorker` class recursively traverses a directed graph
        represented by an IGraph object, starting from a given node ID. It inserts
        the given node and its RT children into the graph, then traverses the
        children's destinations.

        Args:
            node_id (int): A unique identifier of a node in the graph.

        """
        node = self.g.get_node(node_id)
        rt_children = [edge for edge in self.g.get_edges_by_type("RT") if edge.origin == node_id]
        self.insert_igraph_vertex(node)
        # Recur for all the vertices adjacent to this vertex
        for i in rt_children:
            self.traverse_graph(i.destination)
            self.insert_igraph_edge(i)

    def traverse_igraph(self, node):
        """
        Recursively traverses the graph, inserting vertices and edges into a data
        structure called DSR for each vertex with a higher level than the current
        node.

        Args:
            node (igraphVertex): Passed as a reference to a vertex object representing
                a specific node in the graph.

        """
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
        """
        Adds a new vertex to an igraph object, passing it the necessary attributes
        and actions based on their types.

        Args:
            node (igraphNode): Passed as an instance of this class.

        """
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
        """
        Within the SpecificWorker class modifies an input node and adds it to the
        graph. It sets the new node's parent attribute to the id of the specified
        parent node, and then iterates through the node's attributes, adding each
        non-optional attribute with its value from the input node. Finally, the
        method calls `insert_node` on the graph with the new node.

        Args:
            parent_name (str): Used to specify the name of the parent node for the
                new vertex being inserted.
            node (Node): Passed with the function call, representing an existing
                vertex in the graph to be inserted as a child of another existing
                vertex.

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

        # Search for the origin and destination nodes in the graph
        """
        Adds an edge to an igraph representation of a graph, providing the origin
        and destination nodes and their attribute values for rotation and translation.

        Args:
            edge (igraphEdge): Used to insert an edge into an igraph graph.

        """
        origin_node = self.graph.vs.find(id=edge.origin)
        destination_node = self.graph.vs.find(id=edge.destination)
        # Add the edge to the graph
        self.graph.add_edge(origin_node, destination_node, rt=edge.attrs["rt_translation"].value, rotation=edge.attrs["rt_rotation_euler_xyz"].value)
        print("Inserting igraph edge", origin_node["name"], destination_node["name"])
        print("RT", edge.attrs["rt_translation"].value)
        print("Rotation", edge.attrs["rt_rotation_euler_xyz"].value)
        # Print origin and destination nodes

    def insert_dsr_edge(self, org, dest):
        # print("ORG::", org)
        # self.insert_dsr_vertex(dest)
        """
        Adds an edge to the graph with RT and rotation attributes, based on input
        org and dest nodes.

        Args:
            org (Agent): Used to represent the origin node of the DSR edge being
                inserted.
            dest (Agent): Represented by an object containing name, id, and other
                attributes.

        """
        if org is None:
            root_node = self.g.get_node("root")
            org_id = root_node.id
            rt_value = [0, 0, 0]
            orientation = [0, 0, 0]
        else:
            print("Inserting DSR edge", org["name"], dest["name"])
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


        print("RT", rt_value)
        print("Rotation", orientation)
        self.g.insert_or_assign_edge(new_edge)

    def draw_graph(self):
        """
        Generates and plots a graph based on a Kamada-Kawai layout, with nodes
        labeled and edges highlighted with arrows.

        """
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
        plt.show()

    def check_element_room_number(self, node_id):
        """
        Determines the room number associated with a given node ID in a graph,
        retrieving the value from the node's attribute "room_id" or returning -1
        if an exception occurs during the attempt.

        Args:
            node_id (str): Used to retrieve a specific node from the graph.

        Returns:
            int: The room ID associated with a given node ID.

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
        Within the SpecificWorker class checks for an element level attribute in
        a given node and returns its value if found, or -1 otherwise.

        Args:
            node_id (int): Represented by an identifier for a specific node in a
                graph.

        Returns:
            element: Level attribute of node.

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
        Generates an image of a room by traversing the room's nodes and edges in
        a graph, and retrieving attributes such as the room ID and translation
        information. It then draws the room polygon and doors, and saves the image
        to a file.

        Args:
            room_node_id (str): Used to retrieve the node object associated with
                a specific room id.

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
        Within the SpecificWorker class updates the current edge for the agent by
        inserting or assigning an edge in the graph based on the given room ID.

        Args:
            room_id (str): Passed as the ID of the current room where the agent
                is located.

        """
        current_edge = Edge(room_id, room_id, "current", self.agent_id)
        self.g.insert_or_assign_edge(current_edge)

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        """
        Updates the state of an affordance node based on its ID and type. If the
        active ID matches the node ID, it prints out information about the node's
        state. If the state is completed and not active, it sets the state to "crossed".

        Args:
            id (int): A unique identifier for each node in the scene graph, used
                to identify the node being updated.
            type (str): Used to identify the type of affordance node being updated,
                which helps determine the appropriate actions to take based on the
                node's state.

        """
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
        Updates an edge in the graph when a specific condition is met, involving
        the node's robot ID, type, and the presence of an edge.

        Args:
            fr (int): Representing the source node of the edge being updated.
            to (int): Used to represent the target node ID for the update operation.
            type (str): Used to indicate the type of edge being updated, specifically
                "RT".

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
