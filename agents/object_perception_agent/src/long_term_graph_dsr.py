""" Wrapper class for the DSR library to handle the long-term spatial memory"""

import itertools
import matplotlib.pyplot as plt
import spatialmath as sm
import numpy as np
np.set_printoptions(precision=3, suppress=True) # Makes matrix printing cleaner
from PySide6.QtGui import QPolygon
from PySide6.QtCore import QPoint, QPointF, QLineF
from PySide6.QtCore import Qt
import matplotlib.patches as patches
from pydsr import *
import time

class LongTermGraphDSR:
    def __init__(self, dsr_graph, agent_id):
        self.g = dsr_graph
        self.agent_id = agent_id
        self.rt_api = rt_api(self.g)
        self.inner_api = inner_api(self.g)

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
        print("[DEBUG] LongTermGraphDSR initialized.")

        self.metric_map = None
        self.compute_metric_map("hab0")

    def get_room_objects_by_type_recursive(self, node, object_type):
        """
        Recursive part of get_room_objects.
        Finds all children (and children's children, etc.) of 'node'
        that match 'object_type' AND have the same 'room_id'.
        """

        if not node:
            return []

        objects = []
        node_edges = []
        try:
            # --- BULLETPROOFING ---
            node_edges = node.get_edges()
        except Exception as e:
            print(f"[DEBUG]   CRITICAL: node.get_edges() for {node.name} raised an exception: {e}")
            return [] # Can't get edges, return empty

        for edge in node_edges:
            succ_node = None
            try:
                # --- BULLETPROOFING ---
                # This is the call that is likely crashing
                succ_node = self.g.get_node(edge.destination)
            except Exception as e:
                # This block will catch the C++ crash
                print(f"[DEBUG]   CRITICAL: self.g.get_node({edge.destination}) raised an exception: {e}")
                continue # Skip this edge and try the next one

            if succ_node is None:
                print(f"[DEBUG]   Failed to get successor node for ID: {edge.destination} (dangling edge)")
                continue

            # Check attribute using DSR syntax
            if "room_id" in succ_node.attrs and "room_id" in node.attrs:
                if succ_node.attrs["room_id"].value == node.attrs["room_id"].value:
                    # This successor is part of the same room.
                    if succ_node.type == object_type:
                        print(f"[DEBUG]   Found matching object: {succ_node.name}")
                        objects.append(succ_node)

                    # Recurse to check this node's children
                    objects.extend(self.get_room_objects_by_type_recursive(succ_node, object_type))
            # else:
                # print(f"[DEBUG]   Skipping successor {succ_node.name}: missing 'room_id' attribute.")

        return objects


    def get_room_objects_transform_matrices(self, room_name, object_type) -> list:
        """ Computes the transformation matrices of the objects of object_type in room_name"""

        print(f"[DEBUG] Getting transform matrices for objects of type '{object_type}' in room '{room_name}'")
        # Use DSR API to get node by name
        room_node = self.g.get_node(room_name)
        if not room_node:
            print(f"[DEBUG] ERROR: Room node '{room_name}' not found.")
            return []

        if "room_id" not in room_node.attrs:
             print(f"[DEBUG] ERROR: Room node '{room_name}' has no 'room_id' attribute.")
             return []

        # This is the correct, fixed recursive call
        objects = self.get_room_objects_by_type_recursive(room_node, object_type)

        # The above function does NOT check the room_node itself.
        # We must check if the room_node matches (e.g., if object_type is "room")
        if room_node.type == object_type:
            print(f"[DEBUG]   Found matching object: {room_node.name} (is the room itself)")
            objects.append(room_node)

        print(f"[DEBUG]   Found {len(objects)} objects of type '{object_type}'.")

        rts = []
        for object_node in objects:
            print(f"[DEBUG]     Calculating transform: Dest='{room_name}', Orig='{object_node.name}'")
            # Get transform from object frame (orig) to room frame (dest)
            transform_matrix = None
            try:
                # --- BULLETPROOFING ---
                transform_matrix = self.inner_api.get_transformation_matrix(
                    room_node.name,      # destination_frame
                    object_node.name,    # origin_frame
                    np.uint64(time.time() * 1000)
                )
            except Exception as e:
                print(f"[DEBUG]   CRITICAL: get_transformation_matrix({room_node.name}, {object_node.name}) failed: {e}")

            if transform_matrix is not None:
                print(f"[DEBUG]     Transform SUCCESS for {object_node.name} {transform_matrix}")
                rt = sm.SE3(transform_matrix)
                rts.append(rt)
            else:
                print(f"[DEBUG]     Transform FAILED for Dest='{room_name}', Orig='{object_node.name}'.")
        return rts

    def get_room_objects_transform_matrices_with_name(self, room_name, object_type) -> list:
        """ Computes the transformation matrices of the objects of object_type in room_name"""

        print(f"[DEBUG] Getting transform matrices (with name) for '{object_type}' in '{room_name}'")
        # Use DSR API to get node by name
        room_node = self.g.get_node(room_name)
        if not room_node:
            print(f"[DEBUG] ERROR: Room node '{room_name}' not found.")
            return []

        if "room_id" not in room_node.attrs:
             print(f"[DEBUG] ERROR: Room node '{room_name}' has no 'room_id' attribute.")
             return []

        # This is the correct, fixed recursive call
        objects = self.get_room_objects_by_type_recursive(room_node, object_type)

        if room_node.type == object_type:
            print(f"[DEBUG]   Found matching object: {room_node.name} (is the room itself)")
            objects.append(room_node)

        print(f"[DEBUG]   Found {len(objects)} objects of type '{object_type}'.")

        rts = []
        for object_node in objects:
            print(f"[DEBUG]     Calculating transform: Dest='{room_name}', Orig='{object_node.name}'")
            # Get transform from object frame (orig) to room frame (dest)
            transform_matrix = None
            try:
                # --- BULLETPROOFING ---
                transform_matrix = self.inner_api.get_transformation_matrix(
                    room_node.name,      # destination_frame
                    object_node.name,    # origin_frame
                    np.uint64(time.time() * 1000)
                )
            except Exception as e:
                print(f"[DEBUG]   CRITICAL: get_transformation_matrix({room_node.name}, {object_node.name}) failed: {e}")

            if transform_matrix is not None:
                # print(f"[DEBUG]     Transform SUCCESS for {object_node.name}")
                rt = sm.SE3(transform_matrix)
                # Return tuple with node name
                rts.append((object_node.name, rt))
            else:
                print(f"[DEBUG]     Transform FAILED for Dest='{room_name}', Orig='{object_node.name}'.")
        return rts

    def compute_room_map(self, target_room_name, origin_room_name):
        """ Computes the corners of origin room wrt a target room. Returns a QPolygon object"""
        print(f"[DEBUG] Computing room map for '{origin_room_name}' relative to '{target_room_name}'")

        # We want the pose of origin_room_name IN target_room_name's frame.
        # This is T_target <- origin
        transform = self.transform_room(target_room_name, origin_room_name)

        if transform is None:
             print(f"[DEBUG] ERROR: Could not get transform for compute_room_map from {origin_room_name} to {target_room_name}. Returning empty polygon.")
             return QPolygon() # Return empty polygon

        # These are T_corner <- origin_room
        corners_transf = self.get_room_objects_transform_matrices(origin_room_name, "corner")
        if not corners_transf:
            print(f"[DEBUG] WARNING: No 'corner' objects found for room '{origin_room_name}'.")

        # P_target = T_target_<-_origin * P_origin
        # P_corner_in_target_frame = (T_target_<-_origin) * (P_corner_in_origin_frame)
        # c_transf.A[:, -1] gives P_corner_in_origin_frame
        corners_in_room = [transform.A @ np.array(c_transf.A[:, -1]) for c_transf in corners_transf]

        print(f"--- DEBUG POSES for {origin_room_name} in {target_room_name} frame ---")
        for i, corner in enumerate(corners_in_room):
            print(f"  Corner {i}: [x={corner[0]}, y={corner[1]}, z={corner[2]}]")
        print("--------------------------------------------------")

        x_coords = [corner[0] for corner in corners_in_room]
        y_coords = [corner[1] for corner in corners_in_room]
        points = [QPoint(x, y) for x, y in zip(x_coords, y_coords)]
        print(f"[DEBUG]   Computed {len(points)} polygon points.")
        polygon = QPolygon(points)
        return polygon

    def compute_door_map(self, target_room_name, origin_room_name):
        """ Computes the doors of origin room wrt a target room. Returns a QLine object"""
        print(f"[DEBUG] Computing door map for '{origin_room_name}' relative to '{target_room_name}'")
        lines = []

        # T_target <- origin
        transform = self.transform_room(target_room_name, origin_room_name)

        if transform is None:
            print(f"[DEBUG] ERROR: Could not get transform for compute_door_map from {origin_room_name} to {target_room_name}. Returning empty list.")
            return [] # Return empty list

        print(f"[DEBUG]   compute_door_map: Transform OK. Getting door objects...")

        # T_door <- origin_room
        doors_transf = self.get_room_objects_transform_matrices(origin_room_name, "door")

        print(f"[DEBUG]   compute_door_map: Found {len(doors_transf)} doors.")

        if not doors_transf:
            print(f"[DEBUG] WARNING: No 'door' objects found for room '{origin_room_name}'.")

        print(f"--- DEBUG DOOR POSES for {origin_room_name} in {target_room_name} frame ---")

        for i, d_transf in enumerate(doors_transf):
            # d_transf is T_door_<-_origin_room
            # left_point_in_door_frame = [-300, 0, 0, 1]
            # right_point_in_door_frame = [300, 0, 0, 1]

            # P_origin_frame = T_door_<-_origin_room * P_door_frame
            left_point = d_transf.A @ np.array([-300, 0, 0, 1])
            right_point = d_transf.A @ np.array([300, 0, 0, 1])

            # P_target_frame = T_target_<-_origin * P_origin_frame
            left_point_at_target = transform.A @ np.array(left_point)
            right_point_at_target = transform.A @ np.array(right_point)
            print(f"  Corner {i}: [left={left_point}, right={right_point}]")
            line = QLineF(left_point_at_target[0], left_point_at_target[1],
                          right_point_at_target[0], right_point_at_target[1])
            lines.append(line)
        print(f"[DEBUG]   Computed {len(lines)} door lines.")
        return lines

    def get_local_transform(self, target_node_name, origin_node_name) -> sm.SE3:
        """
        Computes the SE3 transform of an origin node relative to a target node
        (e.g., T_target_from_origin).
        This function assumes they are in the same room (no "same" edges).
        It is just a standard PRE-multiplication chain.
        """
        path = self.get_shortest_path(origin_node_name, target_node_name)
        if not path:
            print(f"[DEBUG] get_local_transform: No path from {origin_node_name} to {target_node_name}")
            return None

        T_final = sm.SE3()  # Identity

        # Path is [origin, ..., target]
        for i, (a_node, b_node) in enumerate(zip(path, path[1:])):
            T_step = None  # This will be T_b_from_a
            edge_found = False

            # 1. Check for a BACKWARD edge (b -> a)
            try:
                for edge in b_node.get_edges():
                    if edge.destination == a_node.id and edge.type == "RT":
                        edge_found = True
                        [rt_timestamp, rt_matrix] = self.rt_api.get_edge_RT_as_stamped_rtmat(edge,
                                                                                             int(time.time() * 1000))
                        if rt_matrix is not None:
                            T_step = sm.SE3(rt_matrix)  # T_step = T_b_from_a
                        break
            except Exception as e:
                print(f"[DEBUG]   CRITICAL: {b_node.name}.get_edges() failed: {e}")

            # 2. If no backward edge, check for a FORWARD edge (a -> b)
            if not edge_found:
                try:
                    for edge in a_node.get_edges():
                        if edge.destination == b_node.id and edge.type == "RT":
                            edge_found = True
                            [rt_timestamp, rt_matrix] = self.rt_api.get_edge_RT_as_stamped_rtmat(edge,
                                                                                                 int(time.time() * 1000))
                            if rt_matrix is not None:
                                T_a_from_b = sm.SE3(rt_matrix)
                                T_step = T_a_from_b.inv()  # We need T_b_from_a
                            break
                except Exception as e:
                    print(f"[DEBUG]   CRITICAL: {a_node.name}.get_edges() failed: {e}")

            # 3. Accumulate the transform
            if T_step is not None:
                T_final = T_step * T_final  # PRE-multiplication
            else:
                print(f"[DEBUG] ERROR: No 'RT' edge found between {a_node.name} and {b_node.name}")
                return None
        return T_final

    def compute_element_pose(self, local_pose_xytheta, target_frame_name, origin_frame_name, verbose: bool = False) -> list:
        """
        Transforms a local (x, y, theta) pose from an origin_frame to a target_frame.
        This uses the main transform_room function.
        """
        print(f"[DEBUG] compute_element_pose: Transforming pose from {origin_frame_name} to {target_frame_name}") if verbose else None

        # 1. Get the transform from the origin frame to the target frame
        T_target_from_origin = self.transform_room(target_frame_name, origin_frame_name)
        if T_target_from_origin is None:
            print(f"[DEBUG]   Failed to get transform.") if verbose else None
            return None

        # 2. Convert local (x, y, theta) into an SE3 matrix
        x, y, theta = local_pose_xytheta
        T_local_pose = sm.SE3.Trans(x, y, 0) * sm.SE3.Rz(theta)

        # 3. Chain the transforms: T_global_pose = T_global_from_origin * T_local_pose
        T_global_pose = T_target_from_origin * T_local_pose

        # 4. Extract the (x, y, theta) from the final matrix
        global_x = T_global_pose.t[0]
        global_y = T_global_pose.t[1]
        global_theta = T_global_pose.rpy()[2]  # rpy() returns (roll, pitch, yaw)

        return [global_x, global_y, global_theta]

    ###################################################################
    # GET_SHORTEST_PATH
    ###################################################################
    def get_shortest_path(self, origin_name: str, target_name: str, verbose: bool = False) -> list:
        """
        Computes the shortest path (list of nodes) from origin to target.
        """
        print(f"[DEBUG] Path: Finding shortest path from '{origin_name}' to '{target_name}'") if verbose else None
        origin_node = self.g.get_node(origin_name)
        target_node = self.g.get_node(target_name)

        if not origin_node or not target_node:
            print(f"[DEBUG] Path: Origin or target node not found.") if verbose else None
            return []

        if origin_name == target_name:
            return [origin_node]

        queue = [(origin_node.id, [origin_node])]
        visited = {origin_node.id}  # Store visited node IDs

        all_nodes_map = {}
        try:
            # --- BULLETPROOFING ---
            all_nodes_map = {node.id: node for node in self.g.get_nodes()} # Cache all nodes
        except Exception as e:
            print(f"[DEBUG] Path: CRITICAL g.get_nodes() failed: {e}") if verbose else None
            return [] # Cannot search

        while queue:
            current_id, current_path = queue.pop(0)
            current_node = current_path[-1]

            if current_id == target_node.id:
                print(f"[DEBUG] Path: Found path!") if verbose else None
                path_names = [node.name for node in current_path]
                print(f"[DEBUG] Path: { ' -> '.join(path_names) }") if verbose else None
                return current_path  # We found it!

            # --- 1. Check Successors (Children) ---
            node_edges = []
            try:
                # --- BULLETPROOFING ---
                node_edges = current_node.get_edges()
            except Exception as e:
                print(f"[DEBUG] Path: CRITICAL {current_node.name}.get_edges() failed: {e}") if verbose else None
                continue # Skip this node

            for edge in node_edges:
                dest_id = edge.destination
                if dest_id not in visited:
                    dest_node = None
                    try:
                        # --- BULLETPROOFING ---
                        dest_node = self.g.get_node(dest_id)
                    except Exception as e:
                        print(f"[DEBUG] Path: CRITICAL get_node({dest_id}) failed: {e}") if verbose else None

                    if not dest_node:
                        print(f"[DEBUG] Path: Can't get node for successor ID {dest_id}") if verbose else None
                        continue

                    if dest_node.type == "root" or dest_node.name == "Shadow":
                        print(f"[DEBUG] Path: Skipping 'root' successor {dest_node.name}") if verbose else None
                        continue

                    visited.add(dest_id)
                    new_path = current_path + [dest_node]
                    queue.append((dest_id, new_path))

            # --- 2. Check Predecessors (Parents) ---
            for node_id, node in all_nodes_map.items():
                if node_id in visited:
                    continue

                if node.type == "root" or node.name == "Shadow":
                    continue

                predecessor_edges = []
                try:
                    # --- BULLETPROOFING ---
                    predecessor_edges = node.get_edges()
                except Exception as e:
                    print(f"[DEBUG] Path: CRITICAL {node.name}.get_edges() failed: {e}") if verbose else None
                    continue # Skip this node

                for edge in predecessor_edges:
                    if edge.destination == current_id:
                        visited.add(node.id)
                        new_path = current_path + [node]
                        queue.append((node.id, new_path))
                        break

        print(f"[DEBUG] Path: No path found from {origin_name} to {target_name}") if verbose else None
        return []

    ###################################################################
    # TRANSFORM_ROOM (MODIFIED)
    ###################################################################
    def transform_room(self, target_room_name, origin_room_name, verbose=False) -> sm.SE3:
        """
        Computes the transformation matrix to express the origin_room_name in the target_room_name frame
        by accumulating transforms along the shortest path.

        This version uses standard PRE-multiplication and is the correct way
        to chain transforms from an origin to a target.
        """
        print(f"\n[DEBUG] ##################################################################") if verbose else None
        print(f"[DEBUG] get_transform: Calculating transform from '{origin_room_name}' to '{target_room_name}'") if verbose else None
        print(f"[DEBUG] Using standard PRE-multiplication logic.") if verbose else None

        path = self.get_shortest_path(origin_room_name, target_room_name)

        if not path:
            print(f"[DEBUG] get_transform: No path found.") if verbose else None
            return None

        print(f"[DEBUG] Path found: {[node.name for node in path]}") if verbose else None

        if len(path) == 1 and origin_room_name == target_room_name:
            print("[DEBUG]   Origin and target are the same. Returning Identity.") if verbose else None
            return sm.SE3()

        # This is the final transform, T_target_from_origin (e.g., T_room0_from_room1)
        T_final = sm.SE3()  # Start with Identity
        T_door_flip = sm.SE3.Rz(np.pi)

        print(f"[DEBUG] Initial T_final = \n{T_final.A}") if verbose else None
        print(f"[DEBUG] ##################################################################\n") if verbose else None

        # The path is from origin to target: [room_1, wall_1, door_1, door_0, wall_0, room_0]
        # We need to find T_step = T_b_from_a for each step.
        for i, (a_node, b_node) in enumerate(zip(path, path[1:])):
            print(f"--- STEP {i}: {a_node.name} -> {b_node.name} ---") if verbose else None

            T_step = None  # This will be T_b_from_a
            edge_found = False

            # 1. Check for a BACKWARD edge (b -> a)
            # This is the most common case in DSR: node 'b' has an RT edge pointing to its parent 'a'
            try:
                for edge in b_node.get_edges():
                    if edge.destination == a_node.id:
                        edge_found = True
                        if edge.type == "RT":
                            print(f"[DEBUG]   Found backward 'RT' edge {b_node.name} -> {a_node.name}") if verbose else None
                            [rt_timestamp, rt_matrix] = self.rt_api.get_edge_RT_as_stamped_rtmat(edge,
                                                                                                 int(time.time() * 1000))
                            if rt_matrix is not None:
                                T_step = sm.SE3(rt_matrix)  # T_step = T_b_from_a
                                print(f"[DEBUG]   T_step (T_{b_node.name}_from_{a_node.name}) = \n{T_step.A}") if verbose else None
                        elif edge.type == "same":
                            print(f"[DEBUG]   Found backward 'same' edge. Applying 180 flip.") if verbose else None
                            T_step = T_door_flip.inv()  # Inverse of flip is a flip
                        break
            except Exception as e:
                print(f"[DEBUG]   CRITICAL: {b_node.name}.get_edges() failed: {e}") if verbose else None
                return None

            # 2. If no backward edge, check for a FORWARD edge (a -> b)
            if not edge_found:
                try:
                    for edge in a_node.get_edges():
                        if edge.destination == b_node.id:
                            edge_found = True
                            if edge.type == "RT":
                                print(f"[DEBUG]   Found forward 'RT' edge {a_node.name} -> {b_node.name}") if verbose else None
                                [rt_timestamp, rt_matrix] = self.rt_api.get_edge_RT_as_stamped_rtmat(edge,
                                                                                                     int(time.time() * 1000))
                                if rt_matrix is not None:
                                    T_a_from_b = sm.SE3(rt_matrix)  # This is T_a_from_b
                                    T_step = T_a_from_b.inv()  # We need T_b_from_a
                                    print(f"[DEBUG]   Local T_a_from_b = \n{T_a_from_b.A}") if verbose else None
                                    print(f"[DEBUG]   T_step (T_b_from_a) = \n{T_step.A}") if verbose else None
                            elif edge.type == "same":
                                print(f"[DEBUG]   Found forward 'same' edge. Applying 180 flip.") if verbose else None
                                T_step = T_door_flip
                            break
                except Exception as e:
                    print(f"[DEBUG]   CRITICAL: {a_node.name}.get_edges() failed: {e}") if verbose else None
                    return None

            # 3. Accumulate the transform
            if T_step is not None:
                # This is PRE-multiplication. T_final = T_step_N * ... * T_step_2 * T_step_1
                T_final = T_step * T_final
                print(f"[DEBUG]   Accumulated T_final = \n{T_final.A}") if verbose else None
            else:
                print(f"[DEBUG] ERROR: No 'RT' or 'same' edge found between path nodes {a_node.name} and {b_node.name}") if verbose else None
                return None
            print(f"--- END STEP {i} ---\n") if verbose else None

        print(f"[DEBUG] ##################################################################") if verbose else None
        print(f"[DEBUG] get_transform: Successfully calculated final transform.") if verbose else None
        print(f"[DEBUG] Final T_target_from_origin = \n{T_final.A}") if verbose else None
        print(f"[DEBUG] ##################################################################") if verbose else None
        return T_final

    def compute_metric_map(self, base_room_name, verbose=False):
        """ Computes the metric map of the environment. Returns a dictionary with the rooms and doors."""

        print(f"[DEBUG] Computing metric map with base room '{base_room_name}'") if verbose else None
        m_map = {"base": base_room_name, "rooms": {}}

        # --- NEW ROBUST METHOD ---
        all_nodes = []
        try:
            # --- BULLETPROOFING ---
            all_nodes = self.g.get_nodes()
        except Exception as e:
            print(f"[DEBUG] CRITICAL: g.get_nodes() failed in compute_metric_map: {e}") if verbose else None
            return m_map # Return empty map

        rooms = []
        for node in all_nodes:
            if node.type == "room":
                rooms.append(node)
            # Fallback check in case 'type' is wrong but 'name' is right
            elif node.name.startswith("room_") and node not in rooms:
                print(f"[DEBUG]     Found node '{node.name}' by name, adding to room list.") if verbose else None
                rooms.append(node)
        # --- END NEW METHOD ---

        print(f"[DEBUG]   Found {len(rooms)} nodes with type 'room'.") if verbose else None

        for room_node in rooms:
            room_name = room_node.name
            print(f"[DEBUG]     Processing room: {room_name}") if verbose else None
            m_map["rooms"][room_name] = {}
            m_map["rooms"][room_name]["poly"] = self.compute_room_map(base_room_name, room_name)
            m_map["rooms"][room_name]["doors"] = self.compute_door_map(base_room_name, room_name)

        print(f"[DEBUG] Metric map computation finished.") if verbose else None
        self.update_metric_layout_attributes()
        self.metric_map = m_map

    def check_point_in_map(self, point: QPoint, rooms_map: dict = None, verbose=False):
        """ Check if a point is inside a room polygon. Returns the room name if inside, None otherwise"""
        if rooms_map:
            self.metric_map = rooms_map
        if self.metric_map:
            # print(f"[DEBUG] Checking if point ({point.x()}, {point.y()}) is in map.")
            for room, val in self.metric_map["rooms"].items():
                if val["poly"].containsPoint(point, Qt.OddEvenFill):
                    print(f"[DEBUG]   Point is in room: {room}") if verbose else None
                    return room
            # print(f"[DEBUG]   Point is not in any room.")
            return None
        else:
            print("Metric map not exists")
            return None

    def draw_graph(self, only_rooms=True):
        """ Draws the LTSM graph """
        print("[DEBUG] Skipping draw_graph(): Not implemented for DSR (requires igraph layout).")
        pass

    def draw_metric_map(self, rooms_map: dict, current_room_name=None):
        """ Draws the metric map including rooms and doors"""
        print("[DEBUG] Drawing metric map...")
        self.ax.clear()
        self.ax.set_title('Metric reconstruction')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        if not rooms_map or "rooms" not in rooms_map:
             print("[DEBUG] ERROR: Invalid rooms_map to draw.")
             return

        for room, val in rooms_map["rooms"].items():
            if val and "poly" in val:
                if current_room_name == room:
                    print("Drawing room", room, val["poly"])
                    self.draw_room(room, val["poly"], current=True)
                else:
                    self.draw_room(room, val["poly"])
            if val and "doors" in val:
                print("Drawing doors", room, val["doors"])
                self.draw_doors(val["doors"])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        print("[DEBUG] Metric map drawn.")

    def draw_point(self, point: QPoint, point_in_direction: QPoint= None ):
        """ Draws a point in the map """
        print(f"[DEBUG] Drawing point at ({point.x()}, {point.y()})")
        circle = patches.Circle((float(point.x()), float(point.y())), 50, edgecolor='g', facecolor='none')
        self.ax.add_patch(circle)

        if point_in_direction is not None:
            print(f"[DEBUG]   Drawing direction arrow to ({point_in_direction.x()}, {point_in_direction.y()})")
            self.ax.arrow(point.x(), point.y(), point_in_direction.x() - point.x(), point_in_direction.y() - point.y(), head_width=20, head_length=15, fc='r', ec='r')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def draw_room(self, room_name, room_polygon, current=False):
        """ Draws the room polygon """
        # print(f"[DEBUG] Drawing room: {room_name}")

        # Get corner points from the polygon
        x_coords = [point.x() for point in room_polygon]
        y_coords = [point.y() for point in room_polygon]

        # Set line color depending on if it's the current room
        color = 'r-' if current else 'g-'

        # Plot the rectangle as a polyline
        if x_coords and y_coords: # Only draw if polygon is not empty
            self.ax.plot(x_coords, y_coords, color, linewidth=2)
            self.ax.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color, linewidth=2)

            # Calculate the center point of the rectangle
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)

            # Add the text at the center point
            self.ax.text(center_x, center_y, room_name, ha='center', va='center')
        # else:
            # print(f"[DEBUG]   Skipping draw for {room_name}, polygon is empty.")


    def draw_doors(self, doors: list):  # list of QLineF
        """ Draws the door  """
        # print(f"[DEBUG] Drawing {len(doors)} doors.")
        for line in doors:
            x_coords = [line.x1(), line.x2()]
            y_coords = [line.y1(), line.y2()]
            self.ax.plot(x_coords, y_coords, 'b-', linewidth=2)

    def update_metric_layout_attributes(self, base_room_name="room_0", scale=0.1):
        """
        Calculates the global metric pose for EVERY node and saves it to
        the node's 'pos_x' and 'pos_y' attributes.
        """
        print(f"[DEBUG] Updating metric layout attributes relative to '{base_room_name}' with scale {scale}")

        all_nodes = []
        try:
            all_nodes = self.g.get_nodes()
        except Exception as e:
            print(f"[DEBUG] CRITICAL: g.get_nodes() failed: {e}")
            return

        for node in all_nodes:
            if node.type == "root":
                continue

            # Use our robust transform function to get the node's pose
            # relative to the base room (e.g., T_room0_from_node)
            T_base_from_node = self.transform_room(base_room_name, node.name)

            if T_base_from_node is not None:
                # Extract the translation (x, y, z)
                pos = T_base_from_node.t

                # Apply scale
                metric_x = pos[0] * scale
                metric_y = pos[1] * scale

                # Set attributes, inverting Y for a "+Y down" coordinate system
                node.attrs["pos_x"] = Attribute(float(metric_x), self.agent_id)
                node.attrs["pos_y"] = Attribute(float(-metric_y), self.agent_id)  # Invert Y axis
            else:
                # Node is disconnected or in a broken part of the graph
                if "pos_x" not in node.attrs:
                    node.attrs["pos_x"] = Attribute(float(0.0), self.agent_id)
                if "pos_y" not in node.attrs:
                    node.attrs["pos_y"] =Attribute(float(0.0), self.agent_id)
            self.g.update_node(node)

        print("[DEBUG] Metric layout attributes updated.")

    def clean_LTSM(self):
        all_nodes = self.g.get_nodes()
        for node in all_nodes:
            node_edges = node.get_edges()
            for edge in node_edges:
                n = edge.destination
                if not self.g.get_node(n):
                    print(f"[DEBUG]   Removing dangling edge from {node.name} to {n}")
                    self.g.delete_edge(edge.origin, n, edge.type)
                    if self.g.get_edge(edge.origin, n, edge.type):
                        print(f"[DEBUG]   ERROR: Edge from {node.name} to {n} still exists after deletion.")
            self.g.update_node(node)