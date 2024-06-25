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
    Is a Python class that acts as an agent for a specific robot in a graph,
    navigating it through a environment and updating its state based on node and
    edge attributes. It also handles inserting new edges and deleting existing ones.

    Attributes:
        Period (str): Used to specify the period of time between the worker's
            actions, such as moving or turning.
        agent_id (int): Used to identify the agent's internal graph. It is used
            as a key for accessing the graph's nodes, edges, and attributes related
            to the specific agent.
        g (Graph): Used to represent the graph of the environment, which stores
            nodes and edges as objects, and provides methods for adding, removing,
            and querying nodes and edges.
        update_node (int): Used to update the state of a node based on its ID. It
            takes one argument, `type`, which can be either "crossed" or "completed".
            When `type` is "crossed", the state of the node is updated to "crossed",
            indicating that the node has been crossed. When `type` is "completed",
            the state of the node is updated to "completed", indicating that the
            node has been completed.
        update_edge (edge): Used to update an edge's attributes based on the current
            node state.
        startup_check (QTimersingleShot): Used to call QApplication.instance().quit
            after a delay of 200 milliseconds after the worker's construction,
            indicating that the worker has finished its startup tasks.
        rt_api (str): Used to store the RT API that is used for navigation.
        inner_api (Python): Used to access the internal API of the agent's
            environment, allowing the worker to interact with the agent and its
            environment in a more direct way.
        room_initialized (int): 0 by default, indicating that the room has not
            been initialized yet. When a new room is created or an existing room's
            state changes to "crossed", this attribute value is updated to 1,
            indicating that the room has been initialized.
        robot_name (str): Used to store the name of the robot for which the worker
            is designed to navigate.
        robot_id (int): Used to represent the ID of the robot in the agent's
            internal graph. It is used to identify the robot in the graph and to
            determine the appropriate actions for the robot to take based on its
            current location and goals.
        state (str): Used to keep track of the current state of the worker, such
            as "idle", "running", or "completed".
        affordance_node_active_id (int): Used to store the ID of the node representing
            the affordance of the robot's current action, which is actived when
            the robot performs the action.
        room_exit_door_id (int): 17 by default. It represents the ID of the door
            through which the robot exits a room, used for connecting doors in the
            internal graph of the agent.
        exit_room_node_id (int): Used to store the ID of the room node that the
            robot exits when it leaves a room. It is used in conjunction with the
            `insert_current_edge` method to set the current edge when no other
            edges are present in the graph.
        enter_room_node_id (int): Used to store the ID of the room node that the
            worker enters when it moves from one room to another.
        exit_door_id (int): Used to store the index of the door that leads out of
            a room. It is used for insertion of edges in the graph during navigation.
        graph (nxGraph): Used to represent the graph of nodes and edges that
            represent the environment for the specific worker. It provides access
            to methods for adding, removing, and querying nodes and edges in the
            graph.
        vertex_size (int): 0 by default, indicating that the worker has no vertex.
            It is used to keep track of the number of vertices added to the graph
            during the course of the algorithm.
        not_required_attrs (list): Used to store a list of attributes that are not
            required for the worker's functionality. It serves as a way to prioritize
            the processing of attributes based on their importance.
        fig (instance): Used to store the figure object for visualizing the graph
            during the worker's execution.
        ax (Matplotlib): Used to interact with the graphical user interface (GUI)
            to visualize the graph, draw edges, and display room images.
        insert_current_edge (Edge): Used to insert a new edge into the graph with
            a specific type (current) and source and destination nodes, which are
            the room node and the robot node.
        timer (QTimer): Used to schedule a call to the `QApplication.instance().quit()`
            function after 200 milliseconds, effectively shutting down the application.
        compute (instance): Used to hold the computation result of the worker. The
            `compute` method computes the result
            of a worker by calling its `work` method and returns the result as an
            instance of the
            `Result` class.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes an instance of the SpecificWorker class, defining its graph,
        node ID, and other properties. It also sets up event listeners for updates
        and deletes on the graph's nodes and edges, and defines a timer to call
        the `compute` method at regular intervals.

        Args:
            proxy_map (igraphGraph): Used to store the graph representing the environment.
            startup_check (bool): Optional. It is used to run a check during
                initialization, which can be useful for setting up
                specific worker parameters or initializing other parts of the program.

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
            self.not_required_attrs = ["parent", "timestamp_alivetime", "timestamp_creation"]
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
        Sets the parameters for a specific worker, removes self-edges from a room,
        stores ID of exit door, names other side doors, and reads entrance door
        node to set attributes.

        Args:
            params (object): Used to set parameters for the function's execution.

        Returns:
            Boolean: True.

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
        Determines the state of a worker based on the current state and edges in
        its graph. It updates the worker's state and performs actions accordingly.

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
            case "new_room":
                self.new_room()
            case "initializing_doors":
                self.initializing_doors()
            case "removing":
                self.removing()
            case "draw_graph":


                self.state = "idle"


    def idle(self):

        # Check if there is a node of type aff_cross and it has it's valid attribute to true using comprehension list
        """
        Determines if the worker can cross a room based on its affordance and
        current room information. If there is only one current room, it sets the
        `room_exit_door_id` and `affordance_node_active_id` attributes to their
        respective values and sets the `state` attribute to "crossing". Otherwise,
        it prints "No current room".

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
        Performs actions related to crossing a room, including printing messages
        and changing the state of the `SpecificWorker` object to "crossed".

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
        Determines if a door can be crossed based on its parent node's value,
        deleting an edge and updating the state to "initializing room" if possible.

        """
        affordance_node = self.g.get_node(self.affordance_node_active_id)
        if not affordance_node.attrs["parent"].value:
            # print("Affordance node has no parent")
            return
        else:
            self.exit_door_id = affordance_node.attrs["parent"].value
            # Remove "current" self-edge from the room
            self.g.delete_edge(self.room_exit_door_id, self.room_exit_door_id, "current")
            self.state = "initializing_room"
            print("INITIALIZING ROOM")

    def initializing_room(self):

        # Get room nodes
        """
        Determines and sets the worker's current room ID, adds an edge to the graph
        representing entry into that room, and changes the worker's state to
        "initializing doors".

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
    # def new_room(self):
    #     pass

    def initializing_doors(self):

        # Check if node called "room_entry" of type room exists
        """
        Identifies the doors connected to the agent's room and updates their
        attributes with information about the other side door.

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
                self.state = "removing"
                print("REMOVING")



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

    def removing(self):
        """
        Removes edges from a graph based on room numbers, deleting nodes corresponding
        to the removed edges and updating the graph's state.

        """
        print("ENTER")
        self.traverse_graph(self.room_exit_door_id)
        print("EXIT")
        print(self.graph)
        self.draw_graph()
        # Draw graph
        # layout = self.graph.layout("kk")
        #
        # ig.plot(self.graph, layout=layout)
        # time.sleep(10)
        self.state = "draw_graph"
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
            self.g.delete_node(item.destination)
            # self.g.delete_edge(item.origin, item.destination, "RT")
        self.state = "idle"

    def traverse_graph(self, node_id):
        # Mark the current node as visited and print it
        """
        Traverses the graph, starting from a specified node, and inserts vertices
        and edges according to a specific rule defined by the class `SpecificWorker`.

        Args:
            node_id (str): Used to identify a specific node in the graph.

        """
        node = self.g.get_node(node_id)
        rt_children = [edge for edge in self.g.get_edges_by_type("RT") if edge.origin == node_id]
        self.insert_vertex(node)
        # Recur for all the vertices adjacent to this vertex
        for i in rt_children:
            self.traverse_graph(i.destination)
            self.insert_edge_rt(i)

    def insert_vertex(self, node):
        """
        Adds a new vertex to a graph and associates it with node attributes. If
        the vertex has an "other_side_door_name" attribute, it tries to find a
        matching vertex in the graph with the same name and adds an edge between
        them.

        Args:
            node (Node): Passed as an instance of the Node class, representing a
                vertex to be inserted into the graph.

        """
        self.graph.add_vertex(name=node.name, id=node.id)
        print("Inserting vertex", node.name, node.id)
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
            #         connected_room_node = self.graph.vs.find(name=node.attrs[attr].value)
            #         if connected_room_node:
            #             self.graph.add_edge(self.vertex_size, connected_room_node.id)
            #     except:
            #         print("No connected_room_name attribute found")
        self.vertex_size += 1

    def insert_edge_rt(self, edge):
        # Search for the origin and destination nodes in the graph
        """
        Adds an edge to a graph with a specific translation attribute set to the
        value provided in the `attrs` dictionary of the edge object.

        Args:
            edge (Edge): Passed as an instance of the Edge class, containing
                information about an edge in the graph, including its origin and
                destination nodes and any relevant attribute values.

        """
        origin_node = self.graph.vs.find(id=edge.origin)
        destination_node = self.graph.vs.find(id=edge.destination)
        # Add the edge to the graph
        self.graph.add_edge(origin_node, destination_node, rt=edge.attrs["rt_translation"].value)
        # Print origin and destination nodes

    def draw_graph(self):
        """
        Clears the axis, layouts the graph using the "kamada_kawai" layout algorithm,
        and plots the vertices and edges using different colors and annotations.

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
        Within the SpecificWorker class takes a node ID as input and retrieves the
        room ID associated with that node using the `attrs` attribute. If the
        attribute is not found, it prints an error message and returns -1.

        Args:
            node_id (str): Passed to the function for checking the room number of
                an element in the graph.

        Returns:
            int: The room id of the element with the given node ID, or -1 if no
            such attribute is found.

        """
        node = self.g.get_node(node_id)
        try:
            room_id = node.attrs["room_id"].value
            return room_id
        except:
            print("No room_id attribute found")
            return -1

    def check_element_level(self, node_id):
        """
        Within the SpecificWorker class, determines the element level of a node
        and returns its value if found, or -1 if not. It also checks the robot
        position and adjusts door connections based on similarities with other
        rooms' internal graphs.

        Args:
            node_id (str): Used as the identifier of the node to check the element
                level.

        Returns:
            integerints: Level of an element in a graph represented as a node with
            attributes `level`.

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
        Retrieves the node attributes and edges of a room, and then draws the room
        polygon and doors using OpenCV. It also saves an image of the room as "room_image.jpg".

        Args:
            room_node_id (str): The ID of the room node for which a picture should
                be generated.

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
        Adds an edge to the graph representing the current location of the agent,
        with the specified room ID as the tail and head vertices, and the
        `self.agent_id` as the label.

        Args:
            room_id (str): Passed as an argument to the function, representing the
                identifier of the current room being navigated through.

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
        Updates an affordance node's state based on its ID and type, printing
        various values related to the node's state.

        Args:
            id (int): Used to identify the node for which affordance state is being
                checked.
            type (str): Used to represent the type of node being updated.

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
        Updates the current edge of a room node based on certain conditions,
        including the room ID, the from and to nodes, and the edge type.

        Args:
            fr (int): Referred to as the "from" index of an edge.
            to (int): Representing an integer value of a node ID in the graph.
            type (str): Set to "RT". This indicates that the update is related to
                room transitions.

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
