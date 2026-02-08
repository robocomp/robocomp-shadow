#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
DSR Bridge - Handles all DSR graph operations for the object concept agent.

This module provides methods to:
- Synchronize detected objects to DSR graph
- Create/update nodes for tables and chairs
- Handle RT edges with pose and covariance
- Clean up stale nodes
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from rich.console import Console

console = Console(highlight=False)

# Import pydsr types
import sys
sys.path.append('/opt/robocomp/lib')
from pydsr import Node, Edge, Attribute


class DSRBridge:
    """Bridge between object detection and DSR graph."""

    def __init__(self, g, agent_id: int):
        """
        Initialize DSR bridge.

        Args:
            g: DSR graph instance
            agent_id: Agent ID for creating nodes/edges
        """
        self.g = g
        self.agent_id = agent_id
        self.dsr_viewer = None  # Will be set externally if needed

    def set_viewer(self, dsr_viewer):
        """Set the DSR viewer for edge drawing notifications."""
        self.dsr_viewer = dsr_viewer

    def cleanup_objects(self, object_types: List[str] = None):
        """
        Remove all nodes of specified types from DSR at startup.

        Args:
            object_types: List of node types to remove. Default: ['table', 'chair']
        """
        if object_types is None:
            object_types = ['table', 'chair']

        console.print(f"[yellow]Cleaning up existing {object_types} nodes from DSR...")

        for obj_type in object_types:
            nodes = self.g.get_nodes_by_type(obj_type)
            for node in nodes:
                console.print(f"[yellow]Removing {obj_type} node: {node.name}")
                self.g.delete_node(node.id)

        console.print("[green]DSR cleanup completed")

    def get_room_node(self):
        """Get the room node from DSR graph."""
        room_nodes = self.g.get_nodes_by_type("room")
        if not room_nodes:
            return None
        return room_nodes[0]

    def get_room_dimensions(self) -> Optional[Tuple[float, float]]:
        """
        Get room dimensions from DSR graph.

        Returns:
            Tuple (width, length) in meters, or None if room node not found
        """
        room_nodes = self.g.get_nodes_by_type("room")
        if not room_nodes:
            console.print("[yellow]Room node not found in G")
            return None

        room_node = room_nodes[0]
        if room_node and room_node.name == "room":
            try:
                width_mm = room_node.attrs["room_width"].value
                length_mm = room_node.attrs["room_length"].value
                width_m = width_mm / 1000.0
                depth_m = length_mm / 1000.0
                return (width_m, depth_m)
            except KeyError as e:
                console.print(f"[red]Room attributes not found: {e}")
                return None
        return None

    def get_robot_pose_and_cov(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get robot pose and covariance from the shadow node RT edge to room.

        Returns:
            Tuple (pose [x, y, theta], covariance 3x3 matrix) or (None, None) if not found
        """
        shadow_nodes = self.g.get_nodes_by_type("robot")
        room_nodes = self.g.get_nodes_by_type("room")

        if not shadow_nodes:
            console.print("[yellow]Shadow node not found in G")
            return None, None

        if not room_nodes:
            return None, None

        shadow_node = shadow_nodes[0]
        room_node = room_nodes[0]

        # Get RT edge from room to shadow (parent -> child)
        rt_edge = self.g.get_edge(room_node.id, shadow_node.id, "RT")
        if rt_edge is None:
            console.print("[yellow]RT edge from room to Shadow not found")
            return None, None

        try:
            translation = rt_edge.attrs["rt_translation"].value  # [x, y, z] in mm
            rotation = rt_edge.attrs["rt_rotation_euler_xyz"].value  # [rx, ry, rz] in rad

            # Convert to meters
            x_m = translation[0] / 1000.0
            y_m = translation[1] / 1000.0
            theta = rotation[2]  # rz is the heading angle

            pose = np.array([x_m, y_m, theta])

            # Get covariance if available
            if "rt_se2_covariance" in rt_edge.attrs:
                cov_flat = rt_edge.attrs["rt_se2_covariance"].value
                cov_matrix = np.array(cov_flat).reshape(3, 3)
            else:
                # Default covariance if not available
                cov_matrix = np.eye(3) * 0.01

            return pose, cov_matrix

        except KeyError as e:
            console.print(f"[red]RT edge attributes not found: {e}")
            return None, None

    def sync_objects(self, detected_objects: List[Any]):
        """
        Synchronize committed objects to DSR graph.

        Creates/updates nodes for tables and chairs that are committed (stabilized).
        Objects hang from the room node with RT edges containing pose and covariance.

        Args:
            detected_objects: List of MultiModelBelief objects
        """
        # Get room node
        room_node = self.get_room_node()
        if room_node is None:
            return

        # Get current table and chair nodes from DSR
        existing_tables = {node.name: node for node in self.g.get_nodes_by_type("table")}
        existing_chairs = {node.name: node for node in self.g.get_nodes_by_type("chair")}

        # Track which DSR nodes we've updated this frame
        updated_dsr_ids = set()

        for multi_belief in detected_objects:
            # Only sync committed objects
            model_sel = multi_belief.to_dict().get('model_selection', {})
            if model_sel.get('state') != 'committed':
                continue

            belief_dict = multi_belief.to_dict()
            obj_type = belief_dict.get('type')  # 'table' or 'chair'
            obj_id = belief_dict.get('id')

            # Create unique name for DSR node
            dsr_node_name = f"{obj_type}_{obj_id}"

            # Get pose (in room frame, meters)
            cx = belief_dict.get('cx', 0)
            cy = belief_dict.get('cy', 0)
            theta = belief_dict.get('angle', 0)

            # Get object dimensions based on type
            if obj_type == 'table':
                width = belief_dict.get('width', 0.7)
                depth = belief_dict.get('depth', 0.5)
                height = belief_dict.get('table_height', 0.75)
                existing_nodes = existing_tables
            elif obj_type == 'chair':
                width = belief_dict.get('seat_width', 0.45)
                depth = belief_dict.get('seat_depth', 0.45)
                height = belief_dict.get('seat_height', 0.45)
                existing_nodes = existing_chairs
            else:
                continue

            # Get covariance (default small covariance)
            cov = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]  # 3x3 flattened

            # Check if node already exists
            if dsr_node_name in existing_nodes:
                # Update existing node only if values changed
                self._update_object_node_if_changed(
                    room_node, existing_nodes[dsr_node_name],
                    cx, cy, theta, width, depth, height, cov
                )
            else:
                # Create new node
                self._create_object_node(
                    room_node, dsr_node_name, obj_type,
                    cx, cy, theta, width, depth, height, cov
                )

            updated_dsr_ids.add(dsr_node_name)

    def _find_free_canvas_position(self) -> Tuple[float, float]:
        """
        Find a free position in the DSR canvas for a new node.

        Checks all existing node positions and finds a spot that doesn't overlap.
        Uses different quadrants (negative and positive coordinates).

        Returns:
            (pos_x, pos_y) coordinates.
        """
        # Get all nodes and their positions
        occupied_positions = []
        all_nodes = self.g.get_nodes()

        for node in all_nodes:
            if "pos_x" in node.attrs and "pos_y" in node.attrs:
                px = node.attrs["pos_x"].value
                py = node.attrs["pos_y"].value
                occupied_positions.append((px, py))

        # Define positions in different quadrants
        quadrant_positions = [
            (-200, -200),   # Bottom-left quadrant
            (200, -200),    # Bottom-right quadrant
            (-200, 200),    # Top-left quadrant
            (200, 200),     # Top-right quadrant
            (-350, -100),   # Far left
            (350, -100),    # Far right
            (-100, -350),   # Far bottom
            (-100, 350),    # Far top
            (-350, 200),    # Top far left
            (350, 200),     # Top far right
            (-200, -350),   # Bottom far left
            (200, -350),    # Bottom far right
            (0, -300),      # Center bottom
            (0, 300),       # Center top
            (-300, 0),      # Center left
            (300, 0),       # Center right
        ]

        min_distance = 100  # Minimum distance from other nodes

        # Try predefined quadrant positions first
        for test_x, test_y in quadrant_positions:
            is_free = True
            for ox, oy in occupied_positions:
                distance = ((test_x - ox) ** 2 + (test_y - oy) ** 2) ** 0.5
                if distance < min_distance:
                    is_free = False
                    break

            if is_free:
                return (float(test_x), float(test_y))

        # Fallback: spiral out from center with alternating signs
        step = 150
        for i in range(50):
            sign_x = 1 if (i % 4) < 2 else -1
            sign_y = 1 if (i % 2) == 0 else -1
            offset = ((i // 4) + 1) * step
            test_x = sign_x * offset
            test_y = sign_y * offset

            is_free = True
            for ox, oy in occupied_positions:
                distance = ((test_x - ox) ** 2 + (test_y - oy) ** 2) ** 0.5
                if distance < min_distance:
                    is_free = False
                    break

            if is_free:
                return (float(test_x), float(test_y))

        # Final fallback
        return (float(-300 - len(occupied_positions) * 50), float(-300))

    def _create_object_node(self, room_node, node_name: str, obj_type: str,
                            cx: float, cy: float, theta: float,
                            width: float, depth: float, height: float,
                            cov: list):
        """Create a new object node in DSR hanging from room node."""
        try:
            # Create new node
            new_node = Node(agent_id=self.agent_id, type=obj_type, name=node_name)

            # Find a free position in the canvas for this new node
            pos_x, pos_y = self._find_free_canvas_position()
            new_node.attrs["pos_x"] = Attribute(pos_x, self.agent_id)
            new_node.attrs["pos_y"] = Attribute(pos_y, self.agent_id)

            # Add object-specific attributes (dimensions in mm as INT)
            new_node.attrs["obj_width"] = Attribute(int(width * 1000), self.agent_id)
            new_node.attrs["obj_depth"] = Attribute(int(depth * 1000), self.agent_id)
            new_node.attrs["obj_height"] = Attribute(int(height * 1000), self.agent_id)

            # Insert node first to get its ID
            node_id = self.g.insert_node(new_node)
            if node_id is None:
                console.print(f"[red]Failed to insert {obj_type} node: {node_name}")
                return

            # Create RT edge from room (parent) to object (child)
            # Edge constructor: Edge(to, from, type, agent_id)
            rt_edge = Edge(node_id, room_node.id, "RT", self.agent_id)

            # Set RT attributes (position in mm as vector<float>, rotation in rad)
            rt_edge.attrs["rt_translation"] = Attribute(
                [float(cx * 1000), float(cy * 1000), float(height * 500)],
                self.agent_id
            )
            rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(
                [0.0, 0.0, float(theta)],
                self.agent_id
            )
            rt_edge.attrs["rt_se2_covariance"] = Attribute(
                [float(c) for c in cov],
                self.agent_id
            )

            # Insert edge using insert_or_assign_edge
            result = self.g.insert_or_assign_edge(rt_edge)

            if result:
                console.print(f"[green]Created DSR node: {node_name} at ({cx:.2f}, {cy:.2f}) with RT edge")
                # Notify graph viewer to draw the edge
                self._notify_edge_created(room_node.id, node_id, "RT")
            else:
                console.print(f"[red]Failed to create RT edge for {node_name}")

        except Exception as e:
            console.print(f"[red]Error creating DSR node {node_name}: {e}")

    def _update_object_node_if_changed(self, room_node, obj_node,
                                        cx: float, cy: float, theta: float,
                                        width: float, depth: float, height: float,
                                        cov: list):
        """Update an existing object node in DSR only if values have changed."""
        try:
            # Thresholds for considering a value as changed
            pos_threshold = 0.01  # 1cm
            size_threshold = 0.01  # 1cm
            angle_threshold = 0.05  # ~3 degrees

            node_changed = False
            edge_changed = False

            # Check and update node attributes (dimensions)
            new_width_mm = int(width * 1000)
            new_depth_mm = int(depth * 1000)
            new_height_mm = int(height * 1000)

            if "obj_width" in obj_node.attrs:
                if abs(obj_node.attrs["obj_width"].value - new_width_mm) > size_threshold * 1000:
                    obj_node.attrs["obj_width"] = Attribute(new_width_mm, self.agent_id)
                    node_changed = True
            else:
                obj_node.attrs["obj_width"] = Attribute(new_width_mm, self.agent_id)
                node_changed = True

            if "obj_depth" in obj_node.attrs:
                if abs(obj_node.attrs["obj_depth"].value - new_depth_mm) > size_threshold * 1000:
                    obj_node.attrs["obj_depth"] = Attribute(new_depth_mm, self.agent_id)
                    node_changed = True
            else:
                obj_node.attrs["obj_depth"] = Attribute(new_depth_mm, self.agent_id)
                node_changed = True

            if "obj_height" in obj_node.attrs:
                if abs(obj_node.attrs["obj_height"].value - new_height_mm) > size_threshold * 1000:
                    obj_node.attrs["obj_height"] = Attribute(new_height_mm, self.agent_id)
                    node_changed = True
            else:
                obj_node.attrs["obj_height"] = Attribute(new_height_mm, self.agent_id)
                node_changed = True

            if node_changed:
                self.g.update_node(obj_node)

            # Check and update RT edge (from room to object: parent -> child)
            rt_edge = self.g.get_edge(room_node.id, obj_node.id, "RT")
            if rt_edge is None:
                # Edge doesn't exist, create it
                rt_edge = Edge(obj_node.id, room_node.id, "RT", self.agent_id)
                edge_changed = True

            # Check translation
            new_translation = [float(cx * 1000), float(cy * 1000), float(height * 500)]
            if "rt_translation" in rt_edge.attrs:
                old_trans = rt_edge.attrs["rt_translation"].value
                if (abs(old_trans[0] - new_translation[0]) > pos_threshold * 1000 or
                    abs(old_trans[1] - new_translation[1]) > pos_threshold * 1000 or
                    abs(old_trans[2] - new_translation[2]) > pos_threshold * 1000):
                    rt_edge.attrs["rt_translation"] = Attribute(new_translation, self.agent_id)
                    edge_changed = True
            else:
                rt_edge.attrs["rt_translation"] = Attribute(new_translation, self.agent_id)
                edge_changed = True

            # Check rotation
            new_rotation = [0.0, 0.0, float(theta)]
            if "rt_rotation_euler_xyz" in rt_edge.attrs:
                old_rot = rt_edge.attrs["rt_rotation_euler_xyz"].value
                if abs(old_rot[2] - new_rotation[2]) > angle_threshold:
                    rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(new_rotation, self.agent_id)
                    edge_changed = True
            else:
                rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(new_rotation, self.agent_id)
                edge_changed = True

            # Always update covariance (it may change with observations)
            rt_edge.attrs["rt_se2_covariance"] = Attribute(
                [float(c) for c in cov],
                self.agent_id
            )

            if edge_changed:
                self.g.insert_or_assign_edge(rt_edge)

        except Exception as e:
            console.print(f"[red]Error updating DSR node {obj_node.name}: {e}")

    def _notify_edge_created(self, fr: int, to: int, edge_type: str):
        """Notify the graph viewer that an edge was created."""
        if self.dsr_viewer:
            graph_viewer = self.dsr_viewer.get_graph_viewer()
            if graph_viewer:
                graph_viewer.add_or_assign_edge_slot(fr, to, edge_type)
