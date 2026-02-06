#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qt3D-based 3D Viewer for Object Inference

This viewer integrates with DSRViewer as a custom tab and provides
3D visualization of:
- Room boundaries
- LIDAR point clouds
- Detected objects (tables, chairs, boxes)
- Robot pose and uncertainty
- Historical points

Replaces the Open3D-based visualizer_3d.py
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QUrl, Signal, Slot, QTimer
from PySide6.QtGui import QColor, QVector3D, QQuaternion, QMatrix4x4
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.Qt3DExtras import Qt3DExtras


@dataclass
class VisualizerConfig:
    """Configuration for the Qt3D visualizer."""
    # Window settings
    window_width: int = 800
    window_height: int = 600
    background_color: QColor = field(default_factory=lambda: QColor(50, 50, 60))

    # Point cloud settings
    point_size: float = 0.02
    point_color: QColor = field(default_factory=lambda: QColor(255, 100, 100))  # Red
    historical_point_color: QColor = field(default_factory=lambda: QColor(100, 255, 100))  # Green

    # Room settings
    room_color: QColor = field(default_factory=lambda: QColor(100, 100, 255))  # Blue
    room_line_width: float = 0.02

    # Robot settings
    robot_color: QColor = field(default_factory=lambda: QColor(0, 200, 0))  # Green
    robot_size: float = 0.3

    # Object colors
    table_color: QColor = field(default_factory=lambda: QColor(139, 90, 43, 180))  # Brown, semi-transparent
    chair_color: QColor = field(default_factory=lambda: QColor(70, 130, 180, 180))  # Steel blue, semi-transparent
    box_color: QColor = field(default_factory=lambda: QColor(255, 165, 0, 180))  # Orange, semi-transparent

    # Camera settings
    camera_distance: float = 10.0
    camera_elevation: float = 45.0  # degrees


class Qt3DObjectVisualizer(QWidget):
    """
    Qt3D-based 3D visualizer for object inference.

    Can be used standalone or integrated into DSRViewer as a custom tab.
    """

    def __init__(self, parent=None, config: VisualizerConfig = None):
        super().__init__(parent)
        self.config = config or VisualizerConfig()

        # Qt3D components
        self.view: Optional[Qt3DExtras.Qt3DWindow] = None
        self.root_entity: Optional[Qt3DCore.QEntity] = None
        self.scene_entity: Optional[Qt3DCore.QEntity] = None
        self.camera: Optional[Qt3DRender.QCamera] = None

        # Visualization entities
        self.point_cloud_entity: Optional[Qt3DCore.QEntity] = None
        self.historical_points_entity: Optional[Qt3DCore.QEntity] = None
        self.room_entity: Optional[Qt3DCore.QEntity] = None
        self.robot_entity: Optional[Qt3DCore.QEntity] = None
        self.object_entities: Dict[int, Qt3DCore.QEntity] = {}

        # State
        self.room_dims: Tuple[float, float] = (5.0, 5.0)
        self.robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Setup UI and 3D scene
        self._setup_ui()
        self._setup_3d_scene()

    def _setup_ui(self):
        """Setup the widget UI with controls and 3D view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # Checkboxes for visibility
        self.show_points_cb = QCheckBox("Points")
        self.show_points_cb.setChecked(True)
        self.show_points_cb.stateChanged.connect(self._on_visibility_changed)

        self.show_historical_cb = QCheckBox("Historical")
        self.show_historical_cb.setChecked(True)
        self.show_historical_cb.stateChanged.connect(self._on_visibility_changed)

        self.show_room_cb = QCheckBox("Room")
        self.show_room_cb.setChecked(True)
        self.show_room_cb.stateChanged.connect(self._on_visibility_changed)

        self.show_robot_cb = QCheckBox("Robot")
        self.show_robot_cb.setChecked(True)
        self.show_robot_cb.stateChanged.connect(self._on_visibility_changed)

        self.show_objects_cb = QCheckBox("Objects")
        self.show_objects_cb.setChecked(True)
        self.show_objects_cb.stateChanged.connect(self._on_visibility_changed)

        # Reset camera button
        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self._reset_camera)

        # Status label
        self.status_label = QLabel("Ready")

        control_layout.addWidget(self.show_points_cb)
        control_layout.addWidget(self.show_historical_cb)
        control_layout.addWidget(self.show_room_cb)
        control_layout.addWidget(self.show_robot_cb)
        control_layout.addWidget(self.show_objects_cb)
        control_layout.addWidget(reset_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.status_label)

        layout.addWidget(control_panel)

        # Create Qt3D window and embed it
        self.view = Qt3DExtras.Qt3DWindow()
        self.view.defaultFrameGraph().setClearColor(self.config.background_color)

        # Create container widget for the 3D window
        container = QWidget.createWindowContainer(self.view, self)
        container.setMinimumSize(400, 300)
        layout.addWidget(container, 1)  # stretch factor 1

    def _setup_3d_scene(self):
        """Setup the Qt3D scene with camera, lights, and base entities."""
        # Root entity
        self.root_entity = Qt3DCore.QEntity()
        self.scene_entity = Qt3DCore.QEntity(self.root_entity)

        # Setup camera
        self._setup_camera()

        # Setup lighting
        self._setup_lighting()

        # Create coordinate axes
        self._create_coordinate_axes()

        # Create ground grid
        self._create_ground_grid()

        # Set root entity
        self.view.setRootEntity(self.root_entity)

    def _setup_camera(self):
        """Setup the camera and camera controller."""
        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(
            45.0,  # FOV
            16.0 / 9.0,  # Aspect ratio
            0.1,  # Near plane
            1000.0  # Far plane
        )

        # Position camera
        distance = self.config.camera_distance
        elevation = np.radians(self.config.camera_elevation)
        self.camera.setPosition(QVector3D(
            distance * np.cos(elevation),
            distance * np.cos(elevation),
            distance * np.sin(elevation)
        ))
        self.camera.setViewCenter(QVector3D(0, 0, 0))
        self.camera.setUpVector(QVector3D(0, 0, 1))

        # Orbit camera controller
        self.cam_controller = Qt3DExtras.QOrbitCameraController(self.root_entity)
        self.cam_controller.setCamera(self.camera)
        self.cam_controller.setLinearSpeed(50.0)
        self.cam_controller.setLookSpeed(180.0)

    def _setup_lighting(self):
        """Setup scene lighting."""
        # Main light
        light_entity = Qt3DCore.QEntity(self.root_entity)
        light = Qt3DRender.QPointLight(light_entity)
        light.setColor(Qt.white)
        light.setIntensity(1.2)
        light_entity.addComponent(light)

        light_transform = Qt3DCore.QTransform(light_entity)
        light_transform.setTranslation(QVector3D(5, 5, 10))
        light_entity.addComponent(light_transform)

        # Fill light
        fill_light_entity = Qt3DCore.QEntity(self.root_entity)
        fill_light = Qt3DRender.QPointLight(fill_light_entity)
        fill_light.setColor(Qt.white)
        fill_light.setIntensity(0.6)
        fill_light_entity.addComponent(fill_light)

        fill_transform = Qt3DCore.QTransform(fill_light_entity)
        fill_transform.setTranslation(QVector3D(-5, -5, 5))
        fill_light_entity.addComponent(fill_transform)

    def _create_coordinate_axes(self):
        """Create XYZ coordinate axes at origin."""
        axes_entity = Qt3DCore.QEntity(self.scene_entity)

        axis_length = 0.5
        axis_radius = 0.01

        # X axis (Red)
        self._create_axis(axes_entity, QVector3D(1, 0, 0), Qt.red, axis_length, axis_radius)
        # Y axis (Green)
        self._create_axis(axes_entity, QVector3D(0, 1, 0), Qt.green, axis_length, axis_radius)
        # Z axis (Blue)
        self._create_axis(axes_entity, QVector3D(0, 0, 1), Qt.blue, axis_length, axis_radius)

    def _create_axis(self, parent: Qt3DCore.QEntity, direction: QVector3D,
                     color: QColor, length: float, radius: float):
        """Create a single axis as a cylinder."""
        entity = Qt3DCore.QEntity(parent)

        # Cylinder mesh
        mesh = Qt3DExtras.QCylinderMesh()
        mesh.setRadius(radius)
        mesh.setLength(length)

        # Material
        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(color)
        material.setAmbient(QColor(color).darker(150))

        # Transform
        transform = Qt3DCore.QTransform()
        transform.setTranslation(direction * (length / 2))

        # Rotate cylinder to align with direction
        if direction.x() != 0:
            transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 90))
        elif direction.z() != 0:
            transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 90))

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)

    def _create_ground_grid(self):
        """Create a ground grid at z=0."""
        grid_entity = Qt3DCore.QEntity(self.scene_entity)

        # Use a plane mesh for simplicity
        mesh = Qt3DExtras.QPlaneMesh()
        mesh.setWidth(20)
        mesh.setHeight(20)

        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(QColor(80, 80, 100, 50))
        material.setAmbient(QColor(60, 60, 80))

        transform = Qt3DCore.QTransform()
        transform.setRotationX(90)  # Rotate to XY plane

        grid_entity.addComponent(mesh)
        grid_entity.addComponent(material)
        grid_entity.addComponent(transform)

    def _reset_camera(self):
        """Reset camera to default position."""
        distance = self.config.camera_distance
        elevation = np.radians(self.config.camera_elevation)
        self.camera.setPosition(QVector3D(
            distance * np.cos(elevation),
            distance * np.cos(elevation),
            distance * np.sin(elevation)
        ))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

    def _on_visibility_changed(self):
        """Handle visibility checkbox changes."""
        if self.point_cloud_entity:
            self.point_cloud_entity.setEnabled(self.show_points_cb.isChecked())
        if self.historical_points_entity:
            self.historical_points_entity.setEnabled(self.show_historical_cb.isChecked())
        if self.room_entity:
            self.room_entity.setEnabled(self.show_room_cb.isChecked())
        if self.robot_entity:
            self.robot_entity.setEnabled(self.show_robot_cb.isChecked())
        for entity in self.object_entities.values():
            if entity:
                entity.setEnabled(self.show_objects_cb.isChecked())

    # =========================================================================
    # Public API - Update methods
    # =========================================================================

    def update_room(self, width: float, depth: float):
        """Update room dimensions."""
        self.room_dims = (width, depth)

        # Remove old room entity
        if self.room_entity:
            self.room_entity.setParent(None)
            self.room_entity.deleteLater()

        # Create new room wireframe
        self.room_entity = self._create_room_wireframe(width, depth)
        self.room_entity.setEnabled(self.show_room_cb.isChecked())

    def _create_room_wireframe(self, width: float, depth: float) -> Qt3DCore.QEntity:
        """Create a wireframe box representing the room."""
        entity = Qt3DCore.QEntity(self.scene_entity)

        hw, hd = width / 2, depth / 2
        height = 2.5  # Room height

        # Create edges as thin cylinders
        edges = [
            # Bottom edges
            ((-hw, -hd, 0), (hw, -hd, 0)),
            ((hw, -hd, 0), (hw, hd, 0)),
            ((hw, hd, 0), (-hw, hd, 0)),
            ((-hw, hd, 0), (-hw, -hd, 0)),
            # Top edges
            ((-hw, -hd, height), (hw, -hd, height)),
            ((hw, -hd, height), (hw, hd, height)),
            ((hw, hd, height), (-hw, hd, height)),
            ((-hw, hd, height), (-hw, -hd, height)),
            # Vertical edges
            ((-hw, -hd, 0), (-hw, -hd, height)),
            ((hw, -hd, 0), (hw, -hd, height)),
            ((hw, hd, 0), (hw, hd, height)),
            ((-hw, hd, 0), (-hw, hd, height)),
        ]

        for start, end in edges:
            self._create_line(entity, start, end, self.config.room_color, self.config.room_line_width)

        return entity

    def _create_line(self, parent: Qt3DCore.QEntity, start: tuple, end: tuple,
                     color: QColor, radius: float):
        """Create a line (cylinder) between two points."""
        entity = Qt3DCore.QEntity(parent)

        # Calculate line properties
        p1 = np.array(start)
        p2 = np.array(end)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        center = (p1 + p2) / 2

        # Cylinder mesh
        mesh = Qt3DExtras.QCylinderMesh()
        mesh.setRadius(radius)
        mesh.setLength(length)

        # Material
        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(color)

        # Transform
        transform = Qt3DCore.QTransform()
        transform.setTranslation(QVector3D(center[0], center[1], center[2]))

        # Rotate to align with direction
        if length > 0:
            direction_norm = direction / length
            # Default cylinder is Y-up, so we need to rotate from (0,1,0) to direction
            default_dir = np.array([0, 1, 0])
            rotation_axis = np.cross(default_dir, direction_norm)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.degrees(np.arccos(np.clip(np.dot(default_dir, direction_norm), -1, 1)))
                transform.setRotation(QQuaternion.fromAxisAndAngle(
                    QVector3D(rotation_axis[0], rotation_axis[1], rotation_axis[2]), angle))
            elif direction_norm[1] < 0:  # Pointing in -Y direction
                transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 180))

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)

    def update_robot(self, x: float, y: float, theta: float):
        """Update robot pose."""
        self.robot_pose = (x, y, theta)

        if not self.robot_entity:
            self._create_robot_entity()

        # Update transform
        for component in self.robot_entity.components():
            if isinstance(component, Qt3DCore.QTransform):
                component.setTranslation(QVector3D(x, y, 0.15))
                component.setRotationZ(np.degrees(theta))
                break

    def _create_robot_entity(self):
        """Create robot visualization entity."""
        self.robot_entity = Qt3DCore.QEntity(self.scene_entity)

        # Simple cone/arrow shape for robot
        mesh = Qt3DExtras.QConeMesh()
        mesh.setTopRadius(0)
        mesh.setBottomRadius(self.config.robot_size / 2)
        mesh.setLength(self.config.robot_size)

        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(self.config.robot_color)
        material.setAmbient(QColor(self.config.robot_color).darker(150))

        transform = Qt3DCore.QTransform()
        # Rotate so cone points forward (in Y direction)
        transform.setRotationX(-90)

        self.robot_entity.addComponent(mesh)
        self.robot_entity.addComponent(material)
        self.robot_entity.addComponent(transform)
        self.robot_entity.setEnabled(self.show_robot_cb.isChecked())

    def update_point_cloud(self, points: np.ndarray, color: QColor = None):
        """
        Update the LIDAR point cloud visualization.

        Args:
            points: [N, 3] numpy array of 3D points
            color: Optional color override
        """
        if points is None or len(points) == 0:
            if self.point_cloud_entity:
                self.point_cloud_entity.setEnabled(False)
            return

        color = color or self.config.point_color

        # Remove old entity
        if self.point_cloud_entity:
            self.point_cloud_entity.setParent(None)
            self.point_cloud_entity.deleteLater()

        # Create new point cloud entity
        self.point_cloud_entity = self._create_point_cloud(points, color)
        self.point_cloud_entity.setEnabled(self.show_points_cb.isChecked())

    def update_historical_points(self, points: np.ndarray, color: QColor = None):
        """
        Update historical points visualization.

        Args:
            points: [N, 3] numpy array of 3D points
            color: Optional color override
        """
        if points is None or len(points) == 0:
            if self.historical_points_entity:
                self.historical_points_entity.setEnabled(False)
            return

        color = color or self.config.historical_point_color

        # Remove old entity
        if self.historical_points_entity:
            self.historical_points_entity.setParent(None)
            self.historical_points_entity.deleteLater()

        # Create new point cloud entity
        self.historical_points_entity = self._create_point_cloud(points, color,
                                                                  point_size=self.config.point_size * 0.7)
        self.historical_points_entity.setEnabled(self.show_historical_cb.isChecked())

    def _create_point_cloud(self, points: np.ndarray, color: QColor,
                            point_size: float = None) -> Qt3DCore.QEntity:
        """Create a point cloud entity using instanced spheres."""
        entity = Qt3DCore.QEntity(self.scene_entity)

        point_size = point_size or self.config.point_size

        # For simplicity, create individual sphere entities
        # (For large point clouds, consider using instanced rendering)
        # Limit points for performance
        max_points = 2000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        # Shared mesh and material
        mesh = Qt3DExtras.QSphereMesh()
        mesh.setRadius(point_size)
        mesh.setRings(6)
        mesh.setSlices(6)

        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(color)
        material.setAmbient(QColor(color).darker(150))

        for p in points:
            point_entity = Qt3DCore.QEntity(entity)

            transform = Qt3DCore.QTransform()
            transform.setTranslation(QVector3D(float(p[0]), float(p[1]), float(p[2])))

            point_entity.addComponent(mesh)
            point_entity.addComponent(material)
            point_entity.addComponent(transform)

        return entity

    def update_object(self, obj_id: int, obj_type: str, params: dict):
        """
        Update or create an object visualization.

        Args:
            obj_id: Unique object identifier
            obj_type: Object type ('table', 'chair', 'box')
            params: Object parameters (cx, cy, width, depth, height, angle, etc.)
        """
        # Remove old entity if exists
        if obj_id in self.object_entities:
            old_entity = self.object_entities[obj_id]
            if old_entity:
                old_entity.setParent(None)
                old_entity.deleteLater()

        # Create new entity based on type
        if obj_type == 'table':
            entity = self._create_table(params)
        elif obj_type == 'chair':
            entity = self._create_chair(params)
        elif obj_type == 'box':
            entity = self._create_box(params)
        else:
            return

        entity.setEnabled(self.show_objects_cb.isChecked())
        self.object_entities[obj_id] = entity

    def remove_object(self, obj_id: int):
        """Remove an object from visualization."""
        if obj_id in self.object_entities:
            entity = self.object_entities[obj_id]
            if entity:
                entity.setParent(None)
                entity.deleteLater()
            del self.object_entities[obj_id]

    def clear_objects(self):
        """Remove all object visualizations."""
        for obj_id in list(self.object_entities.keys()):
            self.remove_object(obj_id)

    def _create_table(self, params: dict) -> Qt3DCore.QEntity:
        """Create a table entity (top + 4 legs)."""
        entity = Qt3DCore.QEntity(self.scene_entity)

        cx = params.get('cx', 0)
        cy = params.get('cy', 0)
        width = params.get('width', 1.0)
        depth = params.get('depth', 0.6)
        height = params.get('table_height', 0.75)
        angle = params.get('angle', 0)

        top_thickness = 0.03
        leg_radius = 0.025

        # Table top
        top_entity = Qt3DCore.QEntity(entity)
        top_mesh = Qt3DExtras.QCuboidMesh()
        top_mesh.setXExtent(width)
        top_mesh.setYExtent(depth)
        top_mesh.setZExtent(top_thickness)

        top_material = Qt3DExtras.QPhongAlphaMaterial()
        top_material.setDiffuse(self.config.table_color)
        top_material.setAlpha(0.7)

        top_transform = Qt3DCore.QTransform()
        top_transform.setTranslation(QVector3D(0, 0, height - top_thickness/2))

        top_entity.addComponent(top_mesh)
        top_entity.addComponent(top_material)
        top_entity.addComponent(top_transform)

        # Legs
        leg_height = height - top_thickness
        leg_positions = [
            (width/2 - leg_radius*2, depth/2 - leg_radius*2),
            (-width/2 + leg_radius*2, depth/2 - leg_radius*2),
            (-width/2 + leg_radius*2, -depth/2 + leg_radius*2),
            (width/2 - leg_radius*2, -depth/2 + leg_radius*2),
        ]

        for lx, ly in leg_positions:
            leg_entity = Qt3DCore.QEntity(entity)

            leg_mesh = Qt3DExtras.QCylinderMesh()
            leg_mesh.setRadius(leg_radius)
            leg_mesh.setLength(leg_height)

            leg_material = Qt3DExtras.QPhongAlphaMaterial()
            leg_material.setDiffuse(self.config.table_color)
            leg_material.setAlpha(0.7)

            leg_transform = Qt3DCore.QTransform()
            leg_transform.setTranslation(QVector3D(lx, ly, leg_height/2))
            leg_transform.setRotationX(90)

            leg_entity.addComponent(leg_mesh)
            leg_entity.addComponent(leg_material)
            leg_entity.addComponent(leg_transform)

        # Main transform for position and rotation
        main_transform = Qt3DCore.QTransform()
        main_transform.setTranslation(QVector3D(cx, cy, 0))
        main_transform.setRotationZ(np.degrees(angle))
        entity.addComponent(main_transform)

        return entity

    def _create_chair(self, params: dict) -> Qt3DCore.QEntity:
        """Create a chair entity (seat + backrest)."""
        entity = Qt3DCore.QEntity(self.scene_entity)

        cx = params.get('cx', 0)
        cy = params.get('cy', 0)
        seat_width = params.get('seat_width', 0.45)
        seat_depth = params.get('seat_depth', 0.45)
        seat_height = params.get('seat_height', 0.45)
        back_height = params.get('back_height', 0.4)
        angle = params.get('angle', 0)

        seat_thickness = 0.05
        back_thickness = 0.05

        # Seat
        seat_entity = Qt3DCore.QEntity(entity)
        seat_mesh = Qt3DExtras.QCuboidMesh()
        seat_mesh.setXExtent(seat_width)
        seat_mesh.setYExtent(seat_depth)
        seat_mesh.setZExtent(seat_thickness)

        seat_material = Qt3DExtras.QPhongAlphaMaterial()
        seat_material.setDiffuse(self.config.chair_color)
        seat_material.setAlpha(0.7)

        seat_transform = Qt3DCore.QTransform()
        seat_transform.setTranslation(QVector3D(0, 0, seat_height - seat_thickness/2))

        seat_entity.addComponent(seat_mesh)
        seat_entity.addComponent(seat_material)
        seat_entity.addComponent(seat_transform)

        # Backrest
        back_entity = Qt3DCore.QEntity(entity)
        back_mesh = Qt3DExtras.QCuboidMesh()
        back_mesh.setXExtent(seat_width)
        back_mesh.setYExtent(back_thickness)
        back_mesh.setZExtent(back_height)

        back_material = Qt3DExtras.QPhongAlphaMaterial()
        back_material.setDiffuse(self.config.chair_color)
        back_material.setAlpha(0.7)

        # Backrest at back of seat (+Y direction in local frame)
        back_y = seat_depth/2 - back_thickness/2
        back_z = seat_height + back_height/2

        back_transform = Qt3DCore.QTransform()
        back_transform.setTranslation(QVector3D(0, back_y, back_z))

        back_entity.addComponent(back_mesh)
        back_entity.addComponent(back_material)
        back_entity.addComponent(back_transform)

        # Main transform
        main_transform = Qt3DCore.QTransform()
        main_transform.setTranslation(QVector3D(cx, cy, 0))
        main_transform.setRotationZ(np.degrees(angle))
        entity.addComponent(main_transform)

        return entity

    def _create_box(self, params: dict) -> Qt3DCore.QEntity:
        """Create a box entity."""
        entity = Qt3DCore.QEntity(self.scene_entity)

        cx = params.get('cx', 0)
        cy = params.get('cy', 0)
        width = params.get('width', 0.5)
        height = params.get('height', 0.5)
        depth = params.get('depth', 0.5)
        angle = params.get('angle', 0)

        # Box mesh
        mesh = Qt3DExtras.QCuboidMesh()
        mesh.setXExtent(width)
        mesh.setYExtent(height)
        mesh.setZExtent(depth)

        material = Qt3DExtras.QPhongAlphaMaterial()
        material.setDiffuse(self.config.box_color)
        material.setAlpha(0.7)

        transform = Qt3DCore.QTransform()
        transform.setTranslation(QVector3D(cx, cy, depth/2))
        transform.setRotationZ(np.degrees(angle))

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)

        return entity

    def update_status(self, text: str):
        """Update the status label."""
        self.status_label.setText(text)

    # =========================================================================
    # Compatibility API - matches Open3D visualizer interface
    # =========================================================================

    def start_async(self):
        """Start the visualizer (no-op for Qt3D, already running in main thread)."""
        pass  # Qt3D runs in the main Qt event loop

    def update(self, room_dims: Tuple[float, float], robot_pose: Tuple[float, float, float],
               lidar_points_raw: np.ndarray, lidar_points_filtered: np.ndarray,
               clusters: List[np.ndarray], beliefs: List[dict],
               historical_points: Optional[Dict[int, np.ndarray]] = None):
        """
        Update all visualizations - compatible with Open3D visualizer interface.

        Args:
            room_dims: (width, depth) of the room in meters
            robot_pose: (x, y, theta) of the robot
            lidar_points_raw: Raw LIDAR points (unused, for compatibility)
            lidar_points_filtered: Filtered LIDAR points to display
            clusters: List of point clusters (unused, for compatibility)
            beliefs: List of belief dictionaries with object parameters
            historical_points: Dict mapping belief_id to historical points array
        """
        # Update room
        if room_dims:
            self.update_room(room_dims[0], room_dims[1])

        # Update robot pose
        if robot_pose is not None:
            self.update_robot(float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))

        # Update point cloud (use filtered points)
        if lidar_points_filtered is not None and len(lidar_points_filtered) > 0:
            self.update_point_cloud(lidar_points_filtered)
        else:
            self.update_point_cloud(None)

        # Update historical points (concatenate all belief historical points)
        if historical_points is not None and len(historical_points) > 0:
            all_hist_points = []
            for pts in historical_points.values():
                if pts is not None and len(pts) > 0:
                    all_hist_points.append(pts)
            if all_hist_points:
                combined_hist = np.vstack(all_hist_points)
                self.update_historical_points(combined_hist)
            else:
                self.update_historical_points(None)
        else:
            self.update_historical_points(None)

        # Update objects from beliefs
        current_ids = set()
        for belief_dict in beliefs:
            obj_id = belief_dict.get('id', 0)
            obj_type = belief_dict.get('type', 'box')
            current_ids.add(obj_id)

            # Map belief dict to visualization params based on type
            params = self._belief_to_viz_params(belief_dict, obj_type)
            self.update_object(obj_id, obj_type, params)

        # Remove objects that are no longer tracked
        old_ids = set(self.object_entities.keys()) - current_ids
        for obj_id in old_ids:
            self.remove_object(obj_id)

        # Update status
        self.update_status(f"Tracking {len(beliefs)} objects")

    def _belief_to_viz_params(self, belief_dict: dict, obj_type: str) -> dict:
        """Convert belief dictionary to visualization parameters."""
        if obj_type == 'table':
            return {
                'cx': belief_dict.get('cx', 0),
                'cy': belief_dict.get('cy', 0),
                'width': belief_dict.get('w', 1.0),
                'depth': belief_dict.get('h', 0.6),
                'table_height': belief_dict.get('table_height', 0.75),
                'angle': belief_dict.get('theta', 0)
            }
        elif obj_type == 'chair':
            return {
                'cx': belief_dict.get('cx', 0),
                'cy': belief_dict.get('cy', 0),
                'seat_width': belief_dict.get('seat_w', 0.45),
                'seat_depth': belief_dict.get('seat_d', 0.45),
                'seat_height': belief_dict.get('seat_h', 0.45),
                'back_height': belief_dict.get('back_h', 0.4),
                'angle': belief_dict.get('theta', 0)
            }
        elif obj_type == 'box':
            return {
                'cx': belief_dict.get('cx', 0),
                'cy': belief_dict.get('cy', 0),
                'width': belief_dict.get('w', 0.5),
                'height': belief_dict.get('h', 0.5),
                'depth': belief_dict.get('d', 0.5),
                'angle': belief_dict.get('theta', 0)
            }
        else:
            return belief_dict


# =============================================================================
# Integration with DSRViewer
# =============================================================================

def create_qt3d_viewer_for_dsr(dsr_viewer, name: str = "3D View") -> Qt3DObjectVisualizer:
    """
    Create a Qt3D visualizer and add it as a custom tab to DSRViewer.

    Args:
        dsr_viewer: DSRViewer instance
        name: Name for the tab

    Returns:
        Qt3DObjectVisualizer instance
    """
    visualizer = Qt3DObjectVisualizer()
    dsr_viewer.add_custom_widget_to_dock(name, visualizer)
    return visualizer


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Qt3D Object Visualizer Test")
    window.resize(1024, 768)

    # Create visualizer
    visualizer = Qt3DObjectVisualizer()
    window.setCentralWidget(visualizer)

    # Add some test data
    def update_test():
        # Room
        visualizer.update_room(6.0, 8.0)

        # Robot
        visualizer.update_robot(0.5, 0.5, 0.3)

        # Points
        points = np.random.randn(500, 3) * 0.5
        points[:, 2] = np.abs(points[:, 2])
        visualizer.update_point_cloud(points)

        # Table
        visualizer.update_object(0, 'table', {
            'cx': 1.0, 'cy': 0.0, 'width': 1.2, 'depth': 0.8,
            'table_height': 0.75, 'angle': 0.1
        })

        # Chair
        visualizer.update_object(1, 'chair', {
            'cx': -1.0, 'cy': 0.5, 'seat_width': 0.45, 'seat_depth': 0.45,
            'seat_height': 0.45, 'back_height': 0.4, 'angle': -0.5
        })

    # Timer to update after window shows
    QTimer.singleShot(100, update_test)

    window.show()
    sys.exit(app.exec_())
