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
    # Legacy methods (kept for non-multi mode compatibility)
    ##############################################################

    def get_room_dimensions(self) -> Optional[Tuple[float, float]]:
        """Get room dimensions from DSR graph (legacy - use dsr_bridge instead)."""
        room_nodes = self.g.get_nodes_by_type("room")
        if not room_nodes:
            console.print("[yellow]Room node not found in G")
            return None

        if room_nodes[0] and room_nodes[0].name == "room":
            room_node = room_nodes[0]
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
        """Get robot pose and covariance (legacy - use dsr_bridge instead)."""
        shadow_nodes = self.g.get_nodes_by_type("robot")
        room_nodes = self.g.get_nodes_by_type("room")

        if not shadow_nodes:
            console.print("[yellow]Shadow node not found in G")
            return None, None

        if not room_nodes:
            return None, None

        shadow_node = shadow_nodes[0]
        room_node = room_nodes[0]

        rt_edge = self.g.get_edge(room_node.id, shadow_node.id, "RT")
        if rt_edge is None:
            console.print("[yellow]RT edge from room to Shadow not found")
            return None, None

        try:
            translation = rt_edge.attrs["rt_translation"].value
            rotation = rt_edge.attrs["rt_rotation_euler_xyz"].value
            x_m = translation[0] / 1000.0
            y_m = translation[1] / 1000.0
            theta = rotation[2]
            pose = np.array([x_m, y_m, theta])

            if "rt_se2_covariance" in rt_edge.attrs:
                cov_flat = rt_edge.attrs["rt_se2_covariance"].value
                cov_matrix = np.array(cov_flat).reshape(3, 3)
            else:
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



    # =============== DSR BRIDGE METHODS ================
    # ===================================================

    def _cleanup_dsr_objects(self):
        """Remove any existing table/chair nodes from previous runs."""
        try:
            for node_type in ['table', 'chair']:
                nodes = self.g.get_nodes_by_type(node_type)
                for node in nodes:
                    console.print(f"[yellow]Removing old {node_type} node: {node.name}")
                    self.g.delete_node(node.id)
        except Exception as e:
            console.print(f"[red]Error cleaning up DSR objects: {e}")

    def sync_objects_to_dsr(self, detected_objects):
        """Sync committed objects to DSR graph."""
        if not hasattr(self, 'dsr_node_ids'):
            self.dsr_node_ids = {}  # Track created nodes: belief_id -> dsr_node_id

        for obj in detected_objects:
            # Check if object is committed
            # MultiModelBelief uses: state == ModelState.COMMITTED and committed_model
            # Single model beliefs use: committed (bool) and committed_type (str)
            is_committed = False
            obj_type = 'object'

            # Check MultiModelBelief style (state enum + committed_model)
            if hasattr(obj, 'state') and hasattr(obj, 'committed_model'):
                from src.model_selector import ModelState
                is_committed = (obj.state == ModelState.COMMITTED and obj.committed_model is not None)
                obj_type = obj.committed_model if is_committed else 'object'
            # Fallback: check older style (committed bool + committed_type)
            elif hasattr(obj, 'committed') and obj.committed:
                is_committed = obj.committed
                obj_type = getattr(obj, 'committed_type', 'object')

            if not is_committed:
                continue

            belief_id = obj.id if hasattr(obj, 'id') else 0

            # Check if already created
            if belief_id in self.dsr_node_ids:
                # Update existing node's RT edge
                self._update_dsr_node_pose(belief_id, obj)
            else:
                # Create new node
                self._create_dsr_node(belief_id, obj_type, obj)

    def _create_dsr_node(self, belief_id: int, obj_type: str, obj):
        """Create a new DSR node for a detected object."""
        try:
            # Get room node for RT edge
            room_nodes = self.g.get_nodes_by_type("room")
            if not room_nodes:
                console.print("[yellow]Cannot create object node: room not found")
                return

            room_node = room_nodes[0]
            node_name = f"{obj_type}_{belief_id}"

            # Check if node already exists
            existing = self.g.get_node(node_name)
            if existing:
                self.dsr_node_ids[belief_id] = existing.id
                return

            # Get object position
            obj_dict = obj.to_dict() if hasattr(obj, 'to_dict') else {}
            cx = obj_dict.get('cx', 0.0)
            cy = obj_dict.get('cy', 0.0)
            angle = obj_dict.get('angle', 0.0)

            # Convert to mm for DSR
            x_mm = float(cx * 1000.0)
            y_mm = float(cy * 1000.0)

            # Find free position for visualization in graph canvas
            pos_x, pos_y = self._find_free_canvas_position(belief_id, obj_type)

            # Create new node
            new_node = Node(self.g.get_agent_id(), obj_type, node_name)
            new_node.attrs["pos_x"] = Attribute(float(pos_x), self.g.get_agent_id())
            new_node.attrs["pos_y"] = Attribute(float(pos_y), self.g.get_agent_id())

            node_id = self.g.insert_node(new_node)

            if node_id:
                self.dsr_node_ids[belief_id] = node_id

                # Create RT edge from room to object
                rt_edge = Edge(room_node.id, node_id, "RT", self.g.get_agent_id())
                rt_edge.attrs["rt_translation"] = Attribute(
                    np.array([x_mm, y_mm, 0.0], dtype=np.float32),
                    self.g.get_agent_id()
                )
                rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(
                    np.array([0.0, 0.0, float(angle)], dtype=np.float32),
                    self.g.get_agent_id()
                )

                result = self.g.insert_or_assign_edge(rt_edge)
                if result:
                    console.print(f"[green]Created DSR node: {node_name} at ({cx:.2f}, {cy:.2f}) with RT edge")
                else:
                    console.print(f"[red]Failed to create RT edge for {node_name}")

        except Exception as e:
            console.print(f"[red]Error creating DSR node: {e}")
            import traceback
            traceback.print_exc()

    def _update_dsr_node_pose(self, belief_id: int, obj):
        """Update the RT edge pose for an existing DSR node."""
        try:
            if belief_id not in self.dsr_node_ids:
                return

            node_id = self.dsr_node_ids[belief_id]
            room_nodes = self.g.get_nodes_by_type("room")
            if not room_nodes:
                return

            room_node = room_nodes[0]

            # Get current pose
            obj_dict = obj.to_dict() if hasattr(obj, 'to_dict') else {}
            cx = obj_dict.get('cx', 0.0)
            cy = obj_dict.get('cy', 0.0)
            angle = obj_dict.get('angle', 0.0)

            x_mm = float(cx * 1000.0)
            y_mm = float(cy * 1000.0)

            # Get and update RT edge
            rt_edge = self.g.get_edge(room_node.id, node_id, "RT")
            if rt_edge:
                rt_edge.attrs["rt_translation"] = Attribute(
                    np.array([x_mm, y_mm, 0.0], dtype=np.float32),
                    self.g.get_agent_id()
                )
                rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(
                    np.array([0.0, 0.0, float(angle)], dtype=np.float32),
                    self.g.get_agent_id()
                )
                self.g.insert_or_assign_edge(rt_edge)

        except Exception as e:
            console.print(f"[red]Error updating DSR node pose: {e}")

    def _find_free_canvas_position(self, belief_id: int, obj_type: str) -> Tuple[float, float]:
        """Find a free position in the graph canvas for a new node.

        Places detected objects (chair, table) in quadrants different from robot.
        Robot is typically near center/room, so we place objects further away.
        """
        # Base offset - reduced to half for closer positioning
        base_offset = 200

        # Different positions based on object type to avoid overlap
        # Robot is typically near (0,0) or connected to room
        # Place chairs and tables in different quadrants
        type_offsets = {
            'chair': (base_offset, -base_offset),      # Q4: +x, -y (bottom-right)
            'table': (-base_offset, -base_offset),     # Q3: -x, -y (bottom-left)
        }

        # Get base position for this type, default to Q2 if unknown
        base_x, base_y = type_offsets.get(obj_type, (-base_offset, base_offset))

        # Add small offset based on belief_id to avoid stacking same-type objects
        id_offset = belief_id * 40

        return (base_x + id_offset, base_y)

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
        # console.print(f"UPDATE EDGE: {fr} to {to} type={type}", style='green')
        pass

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        #console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')
        pass

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
