#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2026 by YOUR NAME HERE
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
import numpy as np
import time
import signal
from typing import Optional, Tuple

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose object model: 'box', 'table', 'chair', or 'multi' (table+chair with BMS)
OBJECT_MODEL = 'multi'

# Use DSRViewer with Qt3D tab (True) or standalone Open3D visualizer (False)
USE_DSR_VIEWER = True

# Ground truth for debug (adjust based on scene)
# Parameter names must match the debug_belief_vs_gt method signatures in each manager
GT_CONFIG = {
    'box': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_w': 0.5, 'gt_h': 0.5, 'gt_d': 0.25, 'gt_theta': 0.0},
    'table': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_w': 0.7, 'gt_h': 0.9, 'gt_table_height': 0.5, 'gt_theta': 0.0},
    'chair': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_seat_w': 0.45, 'gt_seat_d': 0.45, 'gt_seat_h': 0.45, 'gt_back_h': 0.40, 'gt_theta': 0.0},
    'multi': None,  # No single GT for multi-model
}

# =============================================================================
# IMPORTS based on model selection
# =============================================================================

if OBJECT_MODEL == 'box':
    from src.objects.box import BoxManager as ObjectManager
elif OBJECT_MODEL == 'table':
    from src.objects.table import TableManager as ObjectManager
elif OBJECT_MODEL == 'chair':
    from src.objects.chair import ChairManager as ObjectManager
elif OBJECT_MODEL == 'multi':
    # Multi-model: Bayesian Model Selection between table and chair
    from src.multi_model_manager import MultiModelManager
    from src.objects.table import TableBelief, TableBeliefConfig
    from src.objects.chair import ChairBelief, ChairBeliefConfig
    from src.model_selector import ModelSelectorConfig
    ObjectManager = None  # Will be created in __init__
else:
    raise ValueError(f"Unknown OBJECT_MODEL: {OBJECT_MODEL}. Choose 'box', 'table', 'chair', or 'multi'.")

# Visualizer imports based on configuration
if USE_DSR_VIEWER:
    from src.qt3d_viewer import Qt3DObjectVisualizerWidget
    from src.dsr_gui import DSRViewer, View
else:
    from src.visualizer_3d import BoxConceptVisualizer3D as BoxConceptVisualizer


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        # Initialize object manager based on OBJECT_MODEL
        if OBJECT_MODEL == 'multi':
            # Multi-model: Bayesian Model Selection between table and chair
            self.object_manager = MultiModelManager(
                model_classes={
                    'table': (TableBelief, TableBeliefConfig()),
                    'chair': (ChairBelief, ChairBeliefConfig())
                },
                selector_config=ModelSelectorConfig(
                    model_priors={'table': 0.5, 'chair': 0.5},
                    entropy_threshold=0.3,
                    concentration_threshold=0.85,
                    hysteresis_frames=10,
                    min_frames_before_commit=20,
                    debug_interval=20
                )
            )
            console.print(f"[green]Using MULTI-MODEL mode: table + chair with Bayesian Model Selection")

            # Clean up any existing table/chair nodes from previous runs
            self._cleanup_dsr_objects()
        else:
            self.object_manager = ObjectManager(self.g, self.g.get_agent_id())
            console.print(f"[green]Using single object model: {OBJECT_MODEL}")

        # Initialize visualizer
        if USE_DSR_VIEWER:
            # Use self (QMainWindow from GenericWorker) as the main window for DSRViewer
            self.setWindowTitle("Object Concept Agent - DSR Viewer")
            self.resize(1200, 800)

            # Create DSRViewer with graph view as main widget
            self.dsr_viewer = DSRViewer(self, self.g, View.graph, View.graph)

            # Create Qt3D visualizer widget (it will initialize in showEvent)
            self.visualizer = Qt3DObjectVisualizerWidget()
            self.dsr_viewer.add_custom_widget_to_dock("Objects 3D", self.visualizer.get_widget())

            console.print("[green]DSRViewer initialized with Qt3D Objects tab")
        else:
            # Standalone Open3D visualizer
            self.visualizer = BoxConceptVisualizer()
            self.visualizer.start_async()
            console.print("[green]Open3D visualizer started")

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        self.cleanup()

    def cleanup(self):
        """Clean up resources before exit."""
        print("[SpecificWorker] Cleaning up...")
        try:
            # Stop timer first
            if hasattr(self, 'timer'):
                self.timer.stop()

            # Clean up visualizer
            if hasattr(self, 'visualizer') and self.visualizer:
                if hasattr(self.visualizer, 'cleanup'):
                    self.visualizer.cleanup()

            # Save DSR viewer state
            if hasattr(self, 'dsr_viewer') and self.dsr_viewer:
                if hasattr(self.dsr_viewer, '_save_window_state'):
                    self.dsr_viewer._save_window_state()

            print("[SpecificWorker] Cleanup completed")
        except Exception as e:
            print(f"[SpecificWorker] Cleanup error: {e}")


    @QtCore.Slot()
    def compute(self):
        start_time = time.perf_counter()

        # Get lidar data
        t0 = time.perf_counter()
        lidar_points = self.get_lidar_points() # meters
        t_lidar = (time.perf_counter() - t0) * 1000

        # Check if room node exists in G
        room_dims = self.get_room_dimensions()
        if room_dims is None:
            return True

        # Get robot pose and covariance from shadow node
        robot_pose, robot_cov = self.get_robot_pose_and_cov()
        if robot_pose is None:
            return True

        # Call object manager to process the data
        t0 = time.perf_counter()
        detected_objects = self.object_manager.update(lidar_points, robot_pose, robot_cov, room_dims)
        t_manager = (time.perf_counter() - t0) * 1000

        # Debug: compare first belief against GT every N frames
        # NOTE: This is for DEVELOPMENT ONLY - we don't have GT in production
        # Set to 0 to disable
        debug_every_n_frames = 0  # DISABLED for production
        if debug_every_n_frames > 0 and len(detected_objects) > 0 and self.object_manager.frame_count % debug_every_n_frames == 0:
            if OBJECT_MODEL == 'multi':
                # For multi-model, print model selection status
                for obj in detected_objects:
                    obj.debug_print()
            else:
                gt = GT_CONFIG.get(OBJECT_MODEL, {})
                if gt:
                    ObjectManager.debug_belief_vs_gt(detected_objects[0].to_dict(), **gt)

        # Update visualizer
        t0 = time.perf_counter()
        self.visualizer.update(
            room_dims=room_dims,
            robot_pose=robot_pose,
            lidar_points_raw=self.object_manager.viz_data['lidar_points_raw'],
            lidar_points_filtered=self.object_manager.viz_data['lidar_points_filtered'],
            clusters=self.object_manager.viz_data['clusters'],
            beliefs=self.object_manager.get_beliefs_as_dicts(),
            historical_points=self.object_manager.get_historical_points_for_viz()
        )
        t_viz = (time.perf_counter() - t0) * 1000

        # Sync committed objects to DSR graph
        self.sync_objects_to_dsr(detected_objects)

        # Compute elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if self.object_manager.frame_count % 50 == 0:
            console.print(f"[dim]Compute: {elapsed_ms:.1f}ms (lidar:{t_lidar:.1f} manager:{t_manager:.1f} viz:{t_viz:.1f})[/dim]")

        return True

    ##############################################################
    def get_room_dimensions(self) -> Optional[Tuple[float, float]]:
        """Get room dimensions from DSR graph.

        Returns:
            Tuple (width, length) in meters, or None if room node not found
        """
        room_nodes = self.g.get_nodes_by_type("room")
        if not room_nodes:
            console.print("[yellow]Room node not found in G")
            return None

        if room_nodes[0] and room_nodes[0].name == "room":   # Take the first room node
            room_node = room_nodes[0]
            try:
                # get all node attributes
                #all_attrs = {name: attr.value for name, attr in room_node.attrs.items()}
                #console.print(f"[cyan]Room attributes: {all_attrs}")

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
        """Get robot pose and covariance from the shadow node RT edge to room.

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

    def get_lidar_points(self) -> np.ndarray:
        """Get LIDAR points in 3D.

        Coordinate convention in robot frame: x+ = right, y+ = forward, z+ = up
        Returns [N, 3] array with (x, y, z) coordinates in meters.
        """
        try:
            # Get 3D LIDAR data
            bpearl = self.lidar3d_proxy.getLidarDataWithThreshold2d("bpearl", 8000, 1)
            helios = self.lidar3d1_proxy.getLidarDataWithThreshold2d("helios", 8000, 1)
            # LIDAR points in robot frame: p.x = right, p.y = forward, p.z = up
            bpearl_np = np.array([[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in bpearl.points])
            helios_np = np.array([[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in helios.points if p.z < 1000])
            lidar_points = np.vstack((bpearl_np, helios_np))

        except Ice.Exception as e:
            print(f"Error reading lidar: {e}")
            lidar_points = np.array([]).reshape(0, 3)

        return lidar_points

    ##############################################################
    # DSR Object Synchronization
    ##############################################################

    def _find_free_canvas_position(self) -> Tuple[float, float]:
        """Find a free position in the DSR canvas for a new node.

        Checks all existing node positions and finds a spot that doesn't overlap.
        Uses different quadrants (negative and positive coordinates).
        Returns (pos_x, pos_y) coordinates.
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
        # Quadrant positions: mix of positive and negative coordinates
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

    def _cleanup_dsr_objects(self):
        """Remove all table and chair nodes from DSR at startup."""
        console.print("[yellow]Cleaning up existing table/chair nodes from DSR...")

        # Remove all table nodes
        table_nodes = self.g.get_nodes_by_type("table")
        for node in table_nodes:
            console.print(f"[yellow]Removing table node: {node.name}")
            self.g.delete_node(node.id)

        # Remove all chair nodes
        chair_nodes = self.g.get_nodes_by_type("chair")
        for node in chair_nodes:
            console.print(f"[yellow]Removing chair node: {node.name}")
            self.g.delete_node(node.id)

        console.print("[green]DSR cleanup completed")

    def sync_objects_to_dsr(self, detected_objects):
        """Synchronize committed objects to DSR graph.

        Creates/updates nodes for tables and chairs that are committed (stabilized).
        Objects hang from the room node with RT edges containing pose and covariance.
        """
        if OBJECT_MODEL != 'multi':
            return  # Only for multi-model mode

        # Get room node
        room_nodes = self.g.get_nodes_by_type("room")
        if not room_nodes:
            return
        room_node = room_nodes[0]

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

    def _create_object_node(self, room_node, node_name: str, obj_type: str,
                            cx: float, cy: float, theta: float,
                            width: float, depth: float, height: float,
                            cov: list):
        """Create a new object node in DSR hanging from room node."""
        try:
            # Create new node
            new_node = Node(agent_id=self.g.get_agent_id(), type=obj_type, name=node_name)

            # Find a free position in the canvas for this new node
            pos_x, pos_y = self._find_free_canvas_position()
            new_node.attrs["pos_x"] = Attribute(pos_x, self.g.get_agent_id())
            new_node.attrs["pos_y"] = Attribute(pos_y, self.g.get_agent_id())

            # Add object-specific attributes (dimensions in mm as INT)
            new_node.attrs["obj_width"] = Attribute(int(width * 1000), self.g.get_agent_id())
            new_node.attrs["obj_depth"] = Attribute(int(depth * 1000), self.g.get_agent_id())
            new_node.attrs["obj_height"] = Attribute(int(height * 1000), self.g.get_agent_id())

            # Insert node first to get its ID
            node_id = self.g.insert_node(new_node)
            if node_id is None:
                console.print(f"[red]Failed to insert {obj_type} node: {node_name}")
                return

            # Create RT edge from room (parent) to object (child)
            # Edge constructor: Edge(to, from, type, agent_id)
            # to = destination (child = new object), from = origin (parent = room)
            rt_edge = Edge(node_id, room_node.id, "RT", self.g.get_agent_id())
            console.print(f"[cyan]DEBUG: Creating RT edge from room({room_node.id}) to {obj_type}({node_id})")

            # Set RT attributes (position in mm as vector<float>, rotation in rad)
            rt_edge.attrs["rt_translation"] = Attribute(
                [float(cx * 1000), float(cy * 1000), float(height * 500)],
                self.g.get_agent_id()
            )
            rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(
                [0.0, 0.0, float(theta)],
                self.g.get_agent_id()
            )
            rt_edge.attrs["rt_se2_covariance"] = Attribute(
                [float(c) for c in cov],
                self.g.get_agent_id()
            )

            # Insert edge using insert_or_assign_edge
            result = self.g.insert_or_assign_edge(rt_edge)

            # Verify edge was created
            verify_edge = self.g.get_edge(room_node.id, node_id, "RT")
            if verify_edge:
                console.print(f"[green]DEBUG: Edge verified - exists in graph")
            else:
                console.print(f"[red]DEBUG: Edge NOT found after insert!")

            if result:
                console.print(f"[green]Created DSR node: {node_name} at ({cx:.2f}, {cy:.2f}) with RT edge")
                # Manually trigger update_edge since the signal doesn't seem to fire
                self.update_edge(room_node.id, node_id, "RT")
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
                    obj_node.attrs["obj_width"] = Attribute(new_width_mm, self.g.get_agent_id())
                    node_changed = True
            else:
                obj_node.attrs["obj_width"] = Attribute(new_width_mm, self.g.get_agent_id())
                node_changed = True

            if "obj_depth" in obj_node.attrs:
                if abs(obj_node.attrs["obj_depth"].value - new_depth_mm) > size_threshold * 1000:
                    obj_node.attrs["obj_depth"] = Attribute(new_depth_mm, self.g.get_agent_id())
                    node_changed = True
            else:
                obj_node.attrs["obj_depth"] = Attribute(new_depth_mm, self.g.get_agent_id())
                node_changed = True

            if "obj_height" in obj_node.attrs:
                if abs(obj_node.attrs["obj_height"].value - new_height_mm) > size_threshold * 1000:
                    obj_node.attrs["obj_height"] = Attribute(new_height_mm, self.g.get_agent_id())
                    node_changed = True
            else:
                obj_node.attrs["obj_height"] = Attribute(new_height_mm, self.g.get_agent_id())
                node_changed = True

            if node_changed:
                self.g.update_node(obj_node)

            # Check and update RT edge (from room to object: parent -> child)
            rt_edge = self.g.get_edge(room_node.id, obj_node.id, "RT")
            if rt_edge is None:
                # Edge doesn't exist, create it
                # Edge constructor: Edge(to, from, type, agent_id)
                rt_edge = Edge(obj_node.id, room_node.id, "RT", self.g.get_agent_id())
                edge_changed = True

            # Check translation
            new_translation = [float(cx * 1000), float(cy * 1000), float(height * 500)]
            if "rt_translation" in rt_edge.attrs:
                old_trans = rt_edge.attrs["rt_translation"].value
                if (abs(old_trans[0] - new_translation[0]) > pos_threshold * 1000 or
                    abs(old_trans[1] - new_translation[1]) > pos_threshold * 1000 or
                    abs(old_trans[2] - new_translation[2]) > pos_threshold * 1000):
                    rt_edge.attrs["rt_translation"] = Attribute(new_translation, self.g.get_agent_id())
                    edge_changed = True
            else:
                rt_edge.attrs["rt_translation"] = Attribute(new_translation, self.g.get_agent_id())
                edge_changed = True

            # Check rotation
            new_rotation = [0.0, 0.0, float(theta)]
            if "rt_rotation_euler_xyz" in rt_edge.attrs:
                old_rot = rt_edge.attrs["rt_rotation_euler_xyz"].value
                if abs(old_rot[2] - new_rotation[2]) > angle_threshold:
                    rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(new_rotation, self.g.get_agent_id())
                    edge_changed = True
            else:
                rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(new_rotation, self.g.get_agent_id())
                edge_changed = True

            # Always update covariance (it may change with observations)
            rt_edge.attrs["rt_se2_covariance"] = Attribute(
                [float(c) for c in cov],
                self.g.get_agent_id()
            )

            if edge_changed:
                self.g.insert_or_assign_edge(rt_edge)

        except Exception as e:
            console.print(f"[red]Error updating DSR node {obj_node.name}: {e}")


    def startup_check(self):
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        print(f"Testing RoboCompLidar3D.TDataCategory from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataCategory()
        print(f"Testing RoboCompLidar3D.TColorCloudData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TColorCloudData()
        QTimer.singleShot(200, QApplication.instance().quit)





    ######################
    # From the RoboCompLidar3D you can call this methods:
    # RoboCompLidar3D.TColorCloudData self.lidar3d_proxy.getColorCloudData()
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarData(str name, float start, float len, int decimationDegreeFactor)
    # RoboCompLidar3D.TDataImage self.lidar3d_proxy.getLidarDataArrayProyectedInImage(str name)
    # RoboCompLidar3D.TDataCategory self.lidar3d_proxy.getLidarDataByCategory(TCategories categories, long timestamp)
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarDataProyectedInImage(str name)
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarDataWithThreshold2d(str name, float distance, int decimationDegreeFactor)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # ifaces.RoboCompLidar3D.TPoint
    # ifaces.RoboCompLidar3D.TDataImage
    # ifaces.RoboCompLidar3D.TData
    # ifaces.RoboCompLidar3D.TDataCategory
    # ifaces.RoboCompLidar3D.TColorCloudData


    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        console.print(f"UPDATE EDGE: {fr} to {to} type={type}", style='green')
        # Notify graph viewer to draw the edge
        if hasattr(self, 'dsr_viewer') and self.dsr_viewer:
            graph_viewer = self.dsr_viewer.get_graph_viewer()
            if graph_viewer:
                graph_viewer.add_or_assign_edge_slot(fr, to, type)

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        #console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')
        pass

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
