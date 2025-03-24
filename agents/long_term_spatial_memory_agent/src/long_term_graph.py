""" Wrapper class for the iGraph library to handle the long-term spatial memory"""

import igraph as ig
import pickle
import itertools
import matplotlib.pyplot as plt
import spatialmath as sm
import numpy as np
from PySide6.QtGui import QPolygon
from PySide6.QtCore import QPoint, QPointF, QLineF
from PySide6.QtCore import Qt
import matplotlib.patches as patches


class LongTermGraph:
    def __init__(self, file_name):
        # try:
        #     self.g = self.read_graph(file_name, directed=True)
        #     print("Graph read from", file_name, self.g.summary())
        # except FileNotFoundError:
        #     print("File not found")
        #     self.g = ig.Graph(directed=True)
        self.g = ig.Graph(directed=True)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.set_title('Metric reconstruction')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')

        self.fig_2, self.ax_2 = plt.subplots()
        self.fig_2.canvas.draw()
        self.fig_2.canvas.flush_events()
        self.ax_2.set_title('LTSM graph')
        self.ax_2.set_xlabel('X-axis')
        self.ax_2.set_ylabel('Y-axis')

    def read_graph(self, file_name, directed=True):
        """ Reads a graph from a file and returns it as an iGraph object"""
        with open(file_name, 'rb') as f:
            gi = pickle.load(f)     # TODO: comes as undirected from the file
        g_directed = ig.Graph(directed=True)
        g_directed.add_vertices(gi.vcount())
        g_directed.add_edges(gi.get_edgelist())

        # Copy vertex attributes
        for attribute in gi.vs.attributes():
            g_directed.vs[attribute] = gi.vs[attribute]

        # Copy edge attributes
        for attribute in gi.es.attributes():
            g_directed.es[attribute] = gi.es[attribute]

        if directed:
            return g_directed
        else:
            return gi

    def get_room_objects_by_type_recursive(self, node, object_type):
        """ recursive part of get_room_corners """

        if node["type"] == object_type:
            return [node]

        objects = []
        for succ in self.g.successors(node):
            if self.g.vs[succ]["room_id"] == node["room_id"]:
                objects += self.get_room_objects_by_type_recursive(self.g.vs[succ], object_type)
        return objects

    def get_room_objects_transform_matrices(self, room_name, object_type) -> list:
        """ Computes the transformation matrices of the objects of object_type in room_name"""

        room = self.g.vs.find(name_eq=room_name)
        objects = self.get_room_objects_by_type_recursive(room, object_type)

        rts = []
        for object in objects:
            path = self.g.get_shortest_path(room, object, weights=None, mode='out', output='epath', algorithm='auto')
            path = self.check_path(path)
            rt = sm.SE3()
            for edge_id in path:
                edge = self.g.es[edge_id]
                rotation_matrix = sm.SO3.RPY(edge["rotation"], unit='rad')
                rt *= sm.SE3.Rt(rotation_matrix, edge["rt"])
            rts.append(rt)
        return rts

    def get_room_objects_transform_matrices_with_name(self, room_name, object_type) -> list:
        """ Computes the transformation matrices of the objects of object_type in room_name"""

        room = self.g.vs.find(name_eq=room_name)
        objects = self.get_room_objects_by_type_recursive(room, object_type)

        rts = []
        for object in objects:
            path = self.g.get_shortest_path(room, object, weights=None, mode='out', output='epath', algorithm='auto')
            path = self.check_path(path)
            rt = sm.SE3()
            for edge_id in path:
                edge = self.g.es[edge_id]
                rotation_matrix = sm.SO3.RPY(edge["rotation"], unit='rad')
                rt *= sm.SE3.Rt(rotation_matrix, edge["rt"])
            rts.append((object["name"], rt))
        return rts

    def compute_room_map(self, target_room_name, origin_room_name):
        """ Computes the corners of origin room wrt a target room. Returns a QPolygon object"""

        transform = self.transform_room(target_room_name, origin_room_name)
        # corners = self.get_room_objects_coordinates(origin_room_name, "corner")
        corners_transf = self.get_room_objects_transform_matrices(origin_room_name, "corner")
        corners_in_room = [transform.A @ np.array(c_transf.A[:, -1]) for c_transf in corners_transf]
        # corners_in_room = [transform.A @ np.array(corner) for corner in corners]
        x_coords = [corner[0] for corner in corners_in_room]
        y_coords = [corner[1] for corner in corners_in_room]
        points = [QPoint(x, y) for x, y in zip(x_coords, y_coords)]
        polygon = QPolygon(points)
        return polygon

    def compute_door_map(self, target_room_name, origin_room_name):
        """ Computes the doors of origin room wrt a target room. Returns a QLine object"""

        lines = []
        transform = self.transform_room(target_room_name, origin_room_name)
        doors_transf = self.get_room_objects_transform_matrices(origin_room_name, "door")
        for d_transf in doors_transf:
            # transform left door points to local room frame
            left_point = d_transf.A @ np.array([-300, 0, 0, 1])
            right_point = d_transf.A @ np.array([300, 0, 0, 1])

            # transform points to target room frame
            left_point_at_target = transform.A @ np.array(left_point)
            right_point_at_target = transform.A @ np.array(right_point)

            line = QLineF(left_point_at_target[0], left_point_at_target[1],
                          right_point_at_target[0], right_point_at_target[1])
            lines.append(line)
        return lines

    def compute_element_pose(self, robot_pose, target_room_name, origin_room_name):
        """ Computes the robot pose from origin room wrt a target room. Returns a QPoint object"""

        """ Computes the robot pose from origin room wrt a target room. Returns a QPoint object"""
        pose = sm.SE3(robot_pose[0], robot_pose[1], 0) * sm.SE3.Rz(robot_pose[2], unit='rad')
        transform = self.transform_room(target_room_name, origin_room_name)
        element_pose_transformed = transform * pose
        rpy = element_pose_transformed.rpy()
        xyz = element_pose_transformed.A[:, -1]

        # print("Element pose transformed", xyz, rpy)

        # return QPoint object
        return np.array([xyz[0], xyz[1], rpy[2]])

    def transform_room(self, target_room_name, origin_room_name) -> sm.SE3:
        """ Computes the transformation matrix to express the origin room in the target room frame"""

        source_room = self.g.vs.find(name_eq=origin_room_name)
        target_room = self.g.vs.find(name_eq=target_room_name)

        # get the transformation chain from the current room to the connected room
        path = self.g.get_shortest_path(target_room, source_room, weights=None, mode='all', output='vpath',
                                        algorithm='auto')
        path = self.check_path(path)

        # compute the transformation matrices from the edges contents
        inverse = False
        door = False
        final = sm.SE3()
        for a, b in zip(path, itertools.islice(path, 1, None)):  # first from path, second from path  (islice)
            edge_id = self.g.get_eid(a, b, directed=False)
            edge = self.g.es(edge_id)

            tr = edge["rt"][0]
            rot = edge["rotation"][0]
            if tr is not None:
                rot = sm.SO3.RPY(rot, unit='rad')
                if door:
                    # print(self.g.vs(a)["name"], self.g.vs(b)["name"], "inverse")
                    rot *= sm.SO3.Rz(np.pi)
                    door = False
                transf = sm.SE3.Rt(rot, tr)
                if inverse:
                    final *= transf.inv()
                    if "room" in ''.join(
                            self.g.vs(b)["name"]):  # reached a room. Time to change to direct transformation
                        inverse = False
                else:
                    final *= transf
            else:  # change to inverse
                inverse = True
                door = True
        return final

    # Create a function that, given a path, check if sequence of (door, wall, door) exists
    def check_path(self, path):
        """ Check if a path contains a sequence of (door, wall, door) """
        # Copy path
        n_insertions = 0
        path_copy = path.copy()
        if len(path) < 3:
            return path

        for i in range(len(path) - 2):
            if self.g.vs[path[i]]["type"] == "door" and self.g.vs[path[i + 1]]["type"] == "wall" and self.g.vs[path[i + 2]]["type"] == "door":
                # get wall_node parent id
                wall_node = self.g.vs[path[i + 1]]
                # Get room_id attr
                room_id = wall_node["room_id"]
                # Search for the room node
                room_node = self.g.vs.find("room_"+str(room_id))
                # Get the wall node id
                room_id = room_node.index
                # insert room_id in the copied path after the wall node
                path_copy.insert(i + 2 + n_insertions, room_id)
                # insert another wall_id after the room_id
                path_copy.insert(i + 3 + n_insertions, path[i + 1])
                n_insertions += 2
        #         print("######################## JUMP DETECTED #########################")
        #         print("Room node id", room_id, "name", room_node["name"], "inserted", "wall id", path[i + 1])
        # print("Path after check", path_copy)
        return path_copy

    def compute_metric_map(self, base_room_name):
        """ Computes the metric map of the environment. Returns a dictionary with the rooms and doors.
            The structure of the dictionary is as follows:
            map
                base
                rooms
                    room_name
                        poly: QPolygon
                        doors: list of QLineF """

        m_map = {"base": base_room_name, "rooms": {}}
        rooms = self.g.vs.select(type_eq="room")
        for room in rooms:
            m_map["rooms"][room["name"]] = {}
            m_map["rooms"][room["name"]]["poly"] = self.compute_room_map(base_room_name, room["name"])
            m_map["rooms"][room["name"]]["doors"] = self.compute_door_map(base_room_name, room["name"])
        return m_map

    def check_point_in_map(self, rooms_map: dict, point: QPoint):
        """ Check if a point is inside a room polygon. Returns the room name if inside, None otherwise"""

        for room, val in rooms_map["rooms"].items():
            if val["poly"].containsPoint(point, Qt.OddEvenFill):
                return room
        return None

    def draw_graph(self, only_rooms=True):
        self.ax_2.clear()
        self.ax_2.set_title('LTSM graph')
        self.ax_2.set_xlabel('X-axis')
        self.ax_2.set_ylabel('Y-axis')
        if self.g != None:
            if only_rooms:
                admitted = ("room", "door", "wall")
                subgraph = self.g.subgraph(self.g.vs.select(lambda v: v["type"] in admitted))
                subgraph = subgraph.subgraph(subgraph.vs.select(
                lambda v: True if v["type"] != "wall" or
                          (v["type"] == "wall" and any(subgraph.vs[nei]["type"] == "door" for nei in subgraph.neighbors(v))) else False))
                layout = subgraph.layout("rt_circular")  # Utiliza el layout Kamada-Kawais
            else:
                layout = self.g.layout("kk")  # Utiliza el layout Kamada-Kawais
                subgraph = self.g

            # Create a list of colors
            colors = ["red" if node["type"] == "room" else "blue" if node["type"]
                      == "door" else "green" for node in subgraph.vs]

            # draw nodes
            x, y = zip(*layout)
            self.ax_2.scatter(x, y, s=100, color=colors)  # Ajustar el tamaño de los vértices con el parámetro 's'

            # draw edges
            for edge in subgraph.get_edgelist():
                edge_data = subgraph.get_eid(edge[0], edge[1])
                self.ax_2.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color="grey")
                self.ax_2.annotate(edge_data, xy=((x[edge[1]] + x[edge[0]]) / 2, (y[edge[1]] + y[edge[0]]) / 2),
                             textcoords="offset points",
                             xytext=(x[edge[0]], y[edge[0]]), ha='center', color='green')

            # draw names
            for i, txt in enumerate([f"Node {i}" for i in range(subgraph.vcount())]):
                # Get name attribute
                name = subgraph.vs[i]["name"]
                # name = str(i)
                self.ax_2.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

            # Adapt ax to the graph
            self.ax_2.set_xlim([min(x) - 2, max(x) + 2])
            self.ax_2.set_ylim([min(y) - 2, max(y) + 2])

    def draw_metric_map(self, rooms_map: dict, current_room_name=None):
        """ Draws the metric map including rooms and doors"""
        # Clear the previous plot
        self.ax.clear()
        self.ax.set_title('Metric reconstruction')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        for room, val in rooms_map["rooms"].items():
            if current_room_name == room:
                self.draw_room(room, val["poly"], current=True)
            else:
                self.draw_room(room, val["poly"])
            self.draw_doors(val["doors"])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_point(self, point: QPoint, point_in_direction: QPoint= None ):
        """ Draws a point in the map """

        circle = patches.Circle((float(point.x()), float(point.y())), 50, edgecolor='g', facecolor='none')
        self.ax.add_patch(circle)

        if point_in_direction is not None:
            print("Drawing arrow")
            print(point.x(), point.y(), point_in_direction.x(), point_in_direction.y())
            self.ax.arrow(point.x(), point.y(), point_in_direction.x() - point.x(), point_in_direction.y() - point.y(), head_width=20, head_length=15, fc='r', ec='r')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def draw_room(self, room_name, room_polygon, current=False):
        """ Draws the room polygon """

        # Get corner points from the polygon
        x_coords = [point.x() for point in room_polygon]
        y_coords = [point.y() for point in room_polygon]

        # Set line color depending on if it's the current room
        color = 'r-' if current else 'g-'

        # Plot the rectangle as a polyline
        self.ax.plot(x_coords, y_coords, color, linewidth=2)
        self.ax.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color, linewidth=2)

        # Calculate the center point of the rectangle
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        # Add the text at the center point
        self.ax.text(center_x, center_y, room_name, ha='center', va='center')

    def draw_doors(self, doors: list):  # list of QLineF
        """ Draws the door  """

        for line in doors:
            x_coords = [line.x1(), line.x2()]
            y_coords = [line.y1(), line.y2()]
            self.ax.plot(x_coords, y_coords, 'b-', linewidth=2)

#################################################
#
# def get_room_corners_recursive(self, node):
#     """ recursive part of get_room_corners """
#
#     if node["type"] == "corner":
#         return [node]
#
#     corners = []
#     for succ in self.g.successors(node):
#         corners += self.get_room_corners_recursive(self.g.vs[succ])
#     return corners
#
# def get_room_corners(self, room_name):
#     """ Returns the corners of a room node by type attribute"""
#
#     room = self.g.vs.find(name_eq=room_name)
#     return self.get_room_corners_recursive(room)
#
# def get_room_corners_coordinates(self, room_name) -> list:
#     """ Computes the projective coordinates (4x1) of the corners of room_name in the room frame """
#
#     corners = self.get_room_corners(room_name)
#     room = self.g.vs.find(name_eq=room_name)
#     rts = []
#     for corner in corners:
#         path = self.g.get_shortest_path(room, corner, weights=None, mode='out', output='epath', algorithm='auto')
#         rt = sm.SE3()
#         for edge_id in path:
#             edge = self.g.es[edge_id]
#             rotation_matrix = sm.SO3.RPY(edge["rotation"], unit='rad')
#             rt *= sm.SE3.Rt(rotation_matrix, edge["rt"])
#         rts.append(rt.A[:, -1])
#     return rts
# def get_room_objects_coordinates(self, room_name, object_type) -> list:
#     """ Computes the projective coordinates (4x1) of the corners of room_name in the room frame """
#
#     room = self.g.vs.find(name_eq=room_name)
#     objects = self.get_room_objects_by_type_recursive(room, object_type)
#
#     rts = []
#     for object in objects:
#         path = self.g.get_shortest_path(room, object, weights=None, mode='out', output='epath', algorithm='auto')
#         rt = sm.SE3()
#         for edge_id in path:
#             edge = self.g.es[edge_id]
#             rotation_matrix = sm.SO3.RPY(edge["rotation"], unit='rad')
#             rt *= sm.SE3.Rt(rotation_matrix, edge["rt"])
#         rts.append(rt.A[:, -1])
#     return rts
