#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qt3D Viewer - Simple approach: create everything once at startup
"""
import math
import os
import json
from PySide6.QtCore import Qt, QUrl, QObject, QEvent
from PySide6.QtGui import QColor, QVector3D, QQuaternion, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.Qt3DExtras import Qt3DExtras

# Path to robot mesh
MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes")
ROBOT_MESH_PATH = os.path.join(MESH_DIR, "shadow.obj")

# Path to save viewer settings
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".qt3d_viewer_settings.json")


class WebotsCameraController(QObject):
    """
    Custom camera controller that behaves like Webots:
    - Left button drag: Rotate around target
    - Right button drag: Pan
    - Wheel: Zoom in/out
    """
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.target = QVector3D(0, 0, 0)

        # Speeds
        self.rotation_speed = 0.005
        self.pan_speed = 0.01
        self.zoom_speed = 0.15

        # State
        self.last_pos = None
        self.left_pressed = False
        self.right_pressed = False

    def setTarget(self, target: QVector3D):
        self.target = target

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            self.last_pos = event.position()
            if event.button() == Qt.MouseButton.LeftButton:
                self.left_pressed = True
            elif event.button() == Qt.MouseButton.RightButton:
                self.right_pressed = True
            return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                self.left_pressed = False
            elif event.button() == Qt.MouseButton.RightButton:
                self.right_pressed = False
            return True

        elif event.type() == QEvent.Type.MouseMove:
            if self.last_pos is None:
                self.last_pos = event.position()
                return True

            dx = event.position().x() - self.last_pos.x()
            dy = event.position().y() - self.last_pos.y()
            self.last_pos = event.position()

            if self.left_pressed:
                # Rotate around target
                self._rotate(dx, dy)
            elif self.right_pressed:
                # Pan
                self._pan(dx, dy)
            return True

        elif event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            self._zoom(delta)
            return True

        return False

    def _rotate(self, dx, dy):
        """Rotate camera around target (orbit)."""
        # Get current position relative to target
        pos = self.camera.position() - self.target

        # Convert to spherical coordinates
        r = pos.length()
        if r < 0.001:
            return

        theta = math.atan2(pos.y(), pos.x())  # Azimuth
        phi = math.acos(max(-1, min(1, pos.z() / r)))  # Elevation

        # Update angles (inverted dy for natural feel - mouse up rotates view up)
        theta -= dx * self.rotation_speed
        phi -= dy * self.rotation_speed  # Inverted sign

        # Clamp elevation to avoid gimbal lock
        phi = max(0.1, min(math.pi - 0.1, phi))

        # Convert back to Cartesian
        new_pos = QVector3D(
            r * math.sin(phi) * math.cos(theta),
            r * math.sin(phi) * math.sin(theta),
            r * math.cos(phi)
        )

        self.camera.setPosition(self.target + new_pos)
        self.camera.setViewCenter(self.target)
        self.camera.setUpVector(QVector3D(0, 0, 1))

    def _pan(self, dx, dy):
        """Pan camera (move target and camera together)."""
        # Get camera's right and up vectors
        view_vec = (self.target - self.camera.position()).normalized()
        up = QVector3D(0, 0, 1)
        right = QVector3D.crossProduct(view_vec, up).normalized()
        actual_up = QVector3D.crossProduct(right, view_vec).normalized()

        # Calculate pan offset
        offset = right * (-dx * self.pan_speed) + actual_up * (dy * self.pan_speed)

        # Move both target and camera
        self.target += offset
        self.camera.setPosition(self.camera.position() + offset)
        self.camera.setViewCenter(self.target)

    def _zoom(self, delta):
        """Zoom in/out by moving camera closer/farther from target."""
        pos = self.camera.position() - self.target
        r = pos.length()

        # Zoom factor
        factor = 1.0 - delta * self.zoom_speed * 0.001
        factor = max(0.1, min(10.0, factor))

        new_r = r * factor
        new_r = max(0.5, min(100.0, new_r))  # Clamp distance

        new_pos = pos.normalized() * new_r
        self.camera.setPosition(self.target + new_pos)


class Qt3DObjectVisualizerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self._initialized = False

        # Qt3D core objects
        self.view = None
        self.root_entity = None
        self.container = None
        self.camera = None
        self.cam_controller = None

        # Scene objects - keep references
        self.scene_objects = {}

        # Room state
        self.room_dims = (6.0, 6.0)

        # Settings save counter (save every N updates)
        self._update_counter = 0
        self._save_interval = 100  # Save every 100 updates

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        print("[Qt3D] Widget created")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initialized:
            self._initialize()

    def _initialize(self):
        if self._initialized:
            return
        print("[Qt3D] Initializing...")

        try:
            # 1. Create Qt3DWindow
            self.view = Qt3DExtras.Qt3DWindow()
            self.view.defaultFrameGraph().setClearColor(QColor(50, 50, 60))

            # 2. Create container
            self.container = QWidget.createWindowContainer(self.view, self)
            self._layout.addWidget(self.container)

            # 3. Create root entity
            self.root_entity = Qt3DCore.QEntity()

            # 4. Setup camera
            self.camera = self.view.camera()
            self.camera.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 1000.0)
            self.camera.setPosition(QVector3D(8, 8, 6))
            self.camera.setViewCenter(QVector3D(0, 0, 0))
            self.camera.setUpVector(QVector3D(0, 0, 1))

            # 5. Camera controller - Webots style
            self.cam_controller = WebotsCameraController(self.camera)
            self.cam_controller.setTarget(QVector3D(0, 0, 0))
            self.cam_controller.rotation_speed = 0.005
            self.cam_controller.pan_speed = 0.01
            self.cam_controller.zoom_speed = 0.15
            self.view.installEventFilter(self.cam_controller)

            # 6. Lighting
            self._create_lights()


            # 7. Create floor
            self._create_floor()

            # 8. Create walls
            self._create_walls()

            # 9. Create robot
            self._create_robot()

            # 10. Set root entity LAST
            self.view.setRootEntity(self.root_entity)

            self._initialized = True

            # 11. Load saved camera settings
            self.load_settings()

            print("[Qt3D] Initialized successfully")

        except Exception as e:
            print(f"[Qt3D] Error: {e}")
            import traceback
            traceback.print_exc()

    def _create_lights(self):
        """Create scene lighting with directional light."""
        # Directional light pointing UP to illuminate floor from below (normals issue)
        light_entity = Qt3DCore.QEntity(self.root_entity)
        light = Qt3DRender.QDirectionalLight()
        light.setColor(QColor(255, 255, 255))
        light.setIntensity(1.0)
        light.setWorldDirection(QVector3D(0, 0, 1))  # Pointing UP
        light_entity.addComponent(light)
        self.scene_objects['light1'] = light_entity

        # Second directional light from side for wall illumination
        light2_entity = Qt3DCore.QEntity(self.root_entity)
        light2 = Qt3DRender.QDirectionalLight()
        light2.setColor(QColor(255, 255, 255))
        light2.setIntensity(0.7)
        light2.setWorldDirection(QVector3D(-1, -1, 0.5).normalized())
        light2_entity.addComponent(light2)
        self.scene_objects['light2'] = light2_entity

        # Third light from opposite side
        light3_entity = Qt3DCore.QEntity(self.root_entity)
        light3 = Qt3DRender.QDirectionalLight()
        light3.setColor(QColor(255, 255, 255))
        light3.setIntensity(0.7)
        light3.setWorldDirection(QVector3D(1, 1, 0.5).normalized())
        light3_entity.addComponent(light3)
        self.scene_objects['light3'] = light3_entity

    def _create_floor(self):
        """Create floor plane."""
        entity = Qt3DCore.QEntity(self.root_entity)

        mesh = Qt3DExtras.QCuboidMesh()
        mesh.setXExtent(self.room_dims[0])
        mesh.setYExtent(self.room_dims[1])
        mesh.setZExtent(0.1)

        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(QColor(180, 160, 140))  # Light beige
        material.setAmbient(QColor(90, 80, 70))

        transform = Qt3DCore.QTransform()
        transform.setTranslation(QVector3D(0, 0, -0.1))  # Move down so top face is at z=-0.05

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)

        self.scene_objects['floor'] = {
            'entity': entity,
            'mesh': mesh,
            'material': material,
            'transform': transform
        }
        print(f"[Qt3D] Floor created: {self.room_dims[0]}x{self.room_dims[1]}m")

    def _create_walls(self):
        """Create room walls with solid colors."""
        half_w = self.room_dims[0] / 2
        half_d = self.room_dims[1] / 2
        wall_height = 1.5
        wall_thickness = 0.08

        # Each wall with distinct color like the test cube
        wall_configs = [
            ('wall_back', 0, -half_d - wall_thickness/2, self.room_dims[0], wall_thickness, wall_height/2, QColor(0, 150, 0), QColor(0, 75, 0)),
            ('wall_front', 0, half_d + wall_thickness/2, self.room_dims[0], wall_thickness, wall_height/2, QColor(0, 0, 150), QColor(0, 0, 75)),
            ('wall_left', -half_w - wall_thickness/2, 0, wall_thickness, self.room_dims[1], wall_height/2, QColor(150, 150, 0), QColor(75, 75, 0)),
            ('wall_right', half_w + wall_thickness/2, 0, wall_thickness, self.room_dims[1], wall_height/2, QColor(150, 0, 150), QColor(75, 0, 75)),
        ]

        for name, x, y, sx, sy, z, diffuse, ambient in wall_configs:
            entity = Qt3DCore.QEntity(self.root_entity)

            mesh = Qt3DExtras.QCuboidMesh()
            mesh.setXExtent(sx)
            mesh.setYExtent(sy)
            mesh.setZExtent(wall_height)

            material = Qt3DExtras.QPhongMaterial()
            material.setDiffuse(diffuse)
            material.setAmbient(ambient)

            transform = Qt3DCore.QTransform()
            transform.setTranslation(QVector3D(x, y, z))

            entity.addComponent(mesh)
            entity.addComponent(material)
            entity.addComponent(transform)

            self.scene_objects[name] = {
                'entity': entity,
                'mesh': mesh,
                'material': material,
                'transform': transform
            }

        print(f"[Qt3D] Walls created")

    def _create_robot(self):
        """Create robot entity with mesh from .obj file."""
        entity = Qt3DCore.QEntity(self.root_entity)

        mesh = Qt3DRender.QMesh()
        mesh.setSource(QUrl.fromLocalFile(ROBOT_MESH_PATH))
        print(f"[Qt3D] Loading robot mesh from: {ROBOT_MESH_PATH}")

        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(QColor(0, 150, 255))
        material.setAmbient(QColor(0, 80, 150))
        material.setSpecular(QColor(255, 255, 255))
        material.setShininess(50.0)

        transform = Qt3DCore.QTransform()
        transform.setScale(1.0)
        transform.setTranslation(QVector3D(0, 0, 0))

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)

        self.scene_objects['robot'] = {
            'entity': entity,
            'mesh': mesh,
            'material': material,
            'transform': transform
        }

    def _update_floor_size(self, width, depth):
        """Update floor dimensions."""
        if 'floor' in self.scene_objects:
            floor = self.scene_objects['floor']
            floor['mesh'].setXExtent(width)
            floor['mesh'].setYExtent(depth)

    def _update_walls(self, width, depth):
        """Update wall positions and sizes."""
        half_w = width / 2
        half_d = depth / 2
        wall_thickness = 0.08

        if 'wall_back' in self.scene_objects:
            w = self.scene_objects['wall_back']
            w['mesh'].setXExtent(width)
            w['transform'].setTranslation(QVector3D(0, -half_d - wall_thickness/2, 0.75))

        if 'wall_front' in self.scene_objects:
            w = self.scene_objects['wall_front']
            w['mesh'].setXExtent(width)
            w['transform'].setTranslation(QVector3D(0, half_d + wall_thickness/2, 0.75))

        if 'wall_left' in self.scene_objects:
            w = self.scene_objects['wall_left']
            w['mesh'].setYExtent(depth)
            w['transform'].setTranslation(QVector3D(-half_w - wall_thickness/2, 0, 0.75))

        if 'wall_right' in self.scene_objects:
            w = self.scene_objects['wall_right']
            w['mesh'].setYExtent(depth)
            w['transform'].setTranslation(QVector3D(half_w + wall_thickness/2, 0, 0.75))

    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update robot position and orientation."""
        if 'robot' not in self.scene_objects:
            return
        transform = self.scene_objects['robot']['transform']
        transform.setTranslation(QVector3D(x, y, 0))
        rotation = QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), math.degrees(theta))
        transform.setRotation(rotation)

    def _create_table(self, table_id: int, cx: float, cy: float, width: float, depth: float,
                      table_height: float, leg_length: float, theta: float):
        """Create a table entity with top and 4 legs - following C++ pattern."""
        # Table parameters (matching C++ code)
        leg_thickness = 0.05
        top_thickness = 0.03
        leg_height = table_height  # Legs from floor to table top

        # Create parent entity for the whole table
        table_entity = Qt3DCore.QEntity(self.root_entity)

        # Global transform for the whole table
        table_transform = Qt3DCore.QTransform()
        table_transform.setTranslation(QVector3D(cx, cy, 0))
        table_transform.setRotationZ(math.degrees(theta))
        table_entity.addComponent(table_transform)

        # Materials
        top_material = Qt3DExtras.QPhongMaterial()
        top_material.setDiffuse(QColor(139, 90, 43))
        top_material.setAmbient(QColor(80, 50, 25))

        leg_material = Qt3DExtras.QPhongMaterial()
        leg_material.setDiffuse(QColor(100, 70, 40))
        leg_material.setAmbient(QColor(60, 40, 20))

        # ==================== TABLE TOP ====================
        top_mesh = Qt3DExtras.QCuboidMesh()
        top_mesh.setXExtent(width)
        top_mesh.setYExtent(depth)
        top_mesh.setZExtent(top_thickness)

        top_entity = Qt3DCore.QEntity(table_entity)
        top_transform = Qt3DCore.QTransform()
        top_transform.setTranslation(QVector3D(0, 0, table_height))  # Table top at height
        top_entity.addComponent(top_mesh)
        top_entity.addComponent(top_transform)
        top_entity.addComponent(top_material)

        # ==================== TABLE LEGS ====================
        leg_offset_x = width / 2.0 - leg_thickness
        leg_offset_y = depth / 2.0 - leg_thickness

        leg_entities = []
        leg_positions = [
            (-leg_offset_x, -leg_offset_y),
            (leg_offset_x, -leg_offset_y),
            (-leg_offset_x, leg_offset_y),
            (leg_offset_x, leg_offset_y),
        ]

        for lx, ly in leg_positions:
            leg_mesh = Qt3DExtras.QCuboidMesh()
            leg_mesh.setXExtent(leg_thickness)
            leg_mesh.setYExtent(leg_thickness)
            leg_mesh.setZExtent(leg_height)

            leg_entity = Qt3DCore.QEntity(table_entity)
            leg_transform = Qt3DCore.QTransform()
            leg_transform.setTranslation(QVector3D(lx, ly, leg_height / 2.0))
            leg_entity.addComponent(leg_mesh)
            leg_entity.addComponent(leg_transform)
            leg_entity.addComponent(leg_material)
            leg_entities.append({'entity': leg_entity, 'mesh': leg_mesh, 'transform': leg_transform})

        # Store references
        self.scene_objects[f'table_{table_id}'] = {
            'entity': table_entity,
            'transform': table_transform,
            'top_entity': top_entity,
            'top_mesh': top_mesh,
            'top_transform': top_transform,
            'top_material': top_material,
            'leg_material': leg_material,
            'leg_entities': leg_entities,
        }

        return table_entity

    def _update_table(self, table_id: int, cx: float, cy: float, width: float, depth: float,
                      table_height: float, leg_length: float, theta: float):
        """Update an existing table's position and size."""
        key = f'table_{table_id}'
        if key not in self.scene_objects:
            self._create_table(table_id, cx, cy, width, depth, table_height, leg_length, theta)
            return

        table = self.scene_objects[key]
        leg_thickness = 0.05
        top_thickness = 0.03
        leg_height = table_height

        # Update main transform
        table['transform'].setTranslation(QVector3D(cx, cy, 0))
        table['transform'].setRotationZ(math.degrees(theta))

        # Update top
        table['top_mesh'].setXExtent(width)
        table['top_mesh'].setYExtent(depth)
        table['top_transform'].setTranslation(QVector3D(0, 0, table_height))

        # Update leg positions and sizes
        leg_offset_x = width / 2.0 - leg_thickness
        leg_offset_y = depth / 2.0 - leg_thickness

        leg_positions = [
            (-leg_offset_x, -leg_offset_y),
            (leg_offset_x, -leg_offset_y),
            (-leg_offset_x, leg_offset_y),
            (leg_offset_x, leg_offset_y),
        ]

        for leg_data, (lx, ly) in zip(table['leg_entities'], leg_positions):
            leg_data['mesh'].setZExtent(leg_height)
            leg_data['transform'].setTranslation(QVector3D(lx, ly, leg_height / 2.0))

    def _remove_unused_tables(self, active_ids: set):
        """Remove table entities that are no longer tracked."""
        to_remove = []
        for key in self.scene_objects:
            if key.startswith('table_'):
                table_id = int(key.split('_')[1])
                if table_id not in active_ids:
                    to_remove.append(key)

        for key in to_remove:
            entity = self.scene_objects[key]['entity']
            entity.setParent(None)
            del self.scene_objects[key]

    # ==================== UNCERTAIN OBJECT METHODS ====================

    def _create_uncertain_object(self, obj_id: int, cx: float, cy: float,
                                  width: float, depth: float, height: float):
        """Create an uncertain object as wireframe - using cuboids (pattern that works)."""
        print(f"[Qt3D] CREATING UNCERTAIN {obj_id} at ({cx:.2f}, {cy:.2f})")

        # Create parent entity
        uncertain_entity = Qt3DCore.QEntity(self.root_entity)

        # Global transform
        uncertain_transform = Qt3DCore.QTransform()
        uncertain_transform.setTranslation(QVector3D(cx, cy, 0))
        uncertain_entity.addComponent(uncertain_transform)

        line_thickness = 0.02
        color = QColor(255, 180, 0)  # Orange
        hw = width / 2.0
        hd = depth / 2.0

        # Define all 12 edges as thin cuboids
        edges_def = []

        # 4 vertical edges (along Z)
        for x, y in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]:
            edges_def.append((x, y, height/2, line_thickness, line_thickness, height))

        # 4 bottom horizontal edges along X (at z~0)
        for y in [-hd, hd]:
            edges_def.append((0, y, line_thickness/2, width, line_thickness, line_thickness))

        # 4 bottom horizontal edges along Y (at z~0)
        for x in [-hw, hw]:
            edges_def.append((x, 0, line_thickness/2, line_thickness, depth, line_thickness))

        # 4 top horizontal edges along X (at z~height)
        for y in [-hd, hd]:
            edges_def.append((0, y, height - line_thickness/2, width, line_thickness, line_thickness))

        # 4 top horizontal edges along Y (at z~height)
        for x in [-hw, hw]:
            edges_def.append((x, 0, height - line_thickness/2, line_thickness, depth, line_thickness))

        # Store ALL references to prevent garbage collection
        edge_refs = []

        for ex, ey, ez, sx, sy, sz in edges_def:
            edge_entity = Qt3DCore.QEntity(uncertain_entity)

            mesh = Qt3DExtras.QCuboidMesh()
            mesh.setXExtent(sx)
            mesh.setYExtent(sy)
            mesh.setZExtent(sz)

            # NEW material for each edge
            material = Qt3DExtras.QPhongMaterial()
            material.setDiffuse(color)
            material.setAmbient(color.darker(120))

            transform = Qt3DCore.QTransform()
            transform.setTranslation(QVector3D(ex, ey, ez))

            edge_entity.addComponent(mesh)
            edge_entity.addComponent(material)
            edge_entity.addComponent(transform)

            # Store ALL references
            edge_refs.append({
                'entity': edge_entity,
                'mesh': mesh,
                'material': material,
                'transform': transform
            })

        # Store everything
        self.scene_objects[f'uncertain_{obj_id}'] = {
            'entity': uncertain_entity,
            'transform': uncertain_transform,
            'edges': edge_refs,
            'width': width,
            'depth': depth,
            'height': height,
        }

        print(f"[Qt3D] UNCERTAIN {obj_id} created successfully")
        return uncertain_entity

    def _update_uncertain_object(self, obj_id: int, cx: float, cy: float,
                                  width: float, depth: float, height: float):
        """Update an uncertain object's position and size."""
        key = f'uncertain_{obj_id}'
        if key not in self.scene_objects:
            self._create_uncertain_object(obj_id, cx, cy, width, depth, height)
            return

        obj = self.scene_objects[key]

        # If size changed significantly, recreate
        if (abs(obj['width'] - width) > 0.05 or
            abs(obj['depth'] - depth) > 0.05 or
            abs(obj['height'] - height) > 0.05):
            obj['entity'].setParent(None)
            del self.scene_objects[key]
            self._create_uncertain_object(obj_id, cx, cy, width, depth, height)
            return

        # Just update position
        obj['transform'].setTranslation(QVector3D(cx, cy, 0))

    def _remove_unused_uncertain(self, active_ids: set):
        """Remove uncertain objects that are no longer tracked."""
        to_remove = []
        for key in self.scene_objects:
            if key.startswith('uncertain_'):
                obj_id = int(key.split('_')[1])
                if obj_id not in active_ids:
                    to_remove.append(key)

        for key in to_remove:
            self.scene_objects[key]['entity'].setParent(None)
            del self.scene_objects[key]

    def update_uncertain_objects(self, beliefs: list):
        """Update uncertain object visualizations."""
        active_ids = set()

        for belief_dict in beliefs:
            model_sel = belief_dict.get('model_selection', {})
            state = model_sel.get('state', 'committed')

            if state != 'uncertain':
                continue

            obj_id = belief_dict.get('id', 0)
            active_ids.add(obj_id)

            # Get position from the active belief
            cx = belief_dict.get('cx', 0)
            cy = belief_dict.get('cy', 0)

            # Estimate size - use average of available dimensions
            width = belief_dict.get('width', belief_dict.get('seat_width', 0.5))
            depth = belief_dict.get('depth', belief_dict.get('seat_depth', 0.5))
            height = belief_dict.get('table_height', belief_dict.get('seat_height', 0.5))

            self._update_uncertain_object(obj_id, cx, cy, width, depth, height)

        self._remove_unused_uncertain(active_ids)

    def update_tables(self, beliefs: list):
        """Update table visualizations from belief list."""
        active_ids = set()

        for belief_dict in beliefs:
            # Only process committed tables
            model_sel = belief_dict.get('model_selection', {})
            state = model_sel.get('state', 'committed')
            if state != 'committed':
                continue

            if belief_dict.get('type') != 'table':
                continue

            table_id = belief_dict.get('id', 0)
            active_ids.add(table_id)

            cx = belief_dict.get('cx', 0)
            cy = belief_dict.get('cy', 0)
            width = belief_dict.get('width', 1.0)
            depth = belief_dict.get('depth', 0.6)
            table_height = belief_dict.get('table_height', 0.75)
            leg_length = belief_dict.get('leg_length', 0.7)
            theta = belief_dict.get('angle', 0)

            self._update_table(table_id, cx, cy, width, depth, table_height, leg_length, theta)

        self._remove_unused_tables(active_ids)

    # ==================== CHAIR METHODS ====================

    def _create_chair(self, chair_id: int, cx: float, cy: float, seat_width: float, seat_depth: float,
                      seat_height: float, back_height: float, theta: float):
        """Create a chair entity - USING SAME PATTERN AS TABLE."""
        print(f"[Qt3D] CREATING CHAIR {chair_id} at ({cx:.2f}, {cy:.2f})")

        # Parameters matching table creation pattern
        top_thickness = 0.05  # seat thickness
        leg_thickness = 0.04
        leg_height = seat_height  # legs go from floor to seat

        # Create parent entity for the whole chair
        chair_entity = Qt3DCore.QEntity(self.root_entity)

        # Global transform for the whole chair
        chair_transform = Qt3DCore.QTransform()
        chair_transform.setTranslation(QVector3D(cx, cy, 0))
        chair_transform.setRotationZ(math.degrees(theta))
        chair_entity.addComponent(chair_transform)

        # Materials - SAME AS TABLE
        top_material = Qt3DExtras.QPhongMaterial()
        top_material.setDiffuse(QColor(50, 100, 150))  # Blue for chair
        top_material.setAmbient(QColor(25, 50, 75))

        leg_material = Qt3DExtras.QPhongMaterial()
        leg_material.setDiffuse(QColor(100, 70, 40))
        leg_material.setAmbient(QColor(60, 40, 20))

        # ==================== SEAT (like table top) ====================
        top_mesh = Qt3DExtras.QCuboidMesh()
        top_mesh.setXExtent(seat_width)
        top_mesh.setYExtent(seat_depth)
        top_mesh.setZExtent(top_thickness)

        top_entity = Qt3DCore.QEntity(chair_entity)
        top_transform = Qt3DCore.QTransform()
        top_transform.setTranslation(QVector3D(0, 0, seat_height))
        top_entity.addComponent(top_mesh)
        top_entity.addComponent(top_transform)
        top_entity.addComponent(top_material)

        # ==================== BACKREST ====================
        back_mesh = Qt3DExtras.QCuboidMesh()
        back_mesh.setXExtent(seat_width)
        back_mesh.setYExtent(0.05)
        back_mesh.setZExtent(back_height)

        back_entity = Qt3DCore.QEntity(chair_entity)
        back_transform = Qt3DCore.QTransform()
        back_transform.setTranslation(QVector3D(0, -seat_depth/2, seat_height + back_height/2))
        back_entity.addComponent(back_mesh)
        back_entity.addComponent(back_transform)
        back_entity.addComponent(top_material)  # Same material as seat

        # ==================== LEGS (same as table) ====================
        leg_offset_x = seat_width / 2.0 - leg_thickness
        leg_offset_y = seat_depth / 2.0 - leg_thickness

        leg_entities = []
        leg_positions = [
            (-leg_offset_x, -leg_offset_y),
            (leg_offset_x, -leg_offset_y),
            (-leg_offset_x, leg_offset_y),
            (leg_offset_x, leg_offset_y),
        ]

        for lx, ly in leg_positions:
            leg_mesh = Qt3DExtras.QCuboidMesh()
            leg_mesh.setXExtent(leg_thickness)
            leg_mesh.setYExtent(leg_thickness)
            leg_mesh.setZExtent(leg_height)

            leg_entity = Qt3DCore.QEntity(chair_entity)
            leg_transform = Qt3DCore.QTransform()
            leg_transform.setTranslation(QVector3D(lx, ly, leg_height / 2.0))
            leg_entity.addComponent(leg_mesh)
            leg_entity.addComponent(leg_transform)
            leg_entity.addComponent(leg_material)
            leg_entities.append({'entity': leg_entity, 'mesh': leg_mesh, 'transform': leg_transform})

        # Store references - with seat_mesh for validation
        self.scene_objects[f'chair_{chair_id}'] = {
            'entity': chair_entity,
            'transform': chair_transform,
            'seat_entity': top_entity,
            'seat_mesh': top_mesh,
            'seat_transform': top_transform,
            'back_entity': back_entity,
            'back_mesh': back_mesh,
            'back_transform': back_transform,
            'top_material': top_material,
            'leg_material': leg_material,
            'leg_entities': leg_entities,
        }

        print(f"[Qt3D] CHAIR {chair_id} created successfully")
        return chair_entity

    def _update_chair(self, chair_id: int, cx: float, cy: float, seat_width: float, seat_depth: float,
                      seat_height: float, back_height: float, theta: float):
        """Update an existing chair's position and size."""
        key = f'chair_{chair_id}'

        # Check if chair exists AND has correct structure (seat_mesh)
        if key in self.scene_objects:
            if 'seat_mesh' not in self.scene_objects[key]:
                # Corrupted entry - remove it
                print(f"[VIZ] Removing corrupted chair entry {key}")
                self.scene_objects[key]['entity'].setParent(None)
                del self.scene_objects[key]

        if key not in self.scene_objects:
            self._create_chair(chair_id, cx, cy, seat_width, seat_depth, seat_height, back_height, theta)
            return

        chair = self.scene_objects[key]
        seat_thickness = 0.05
        back_thickness = 0.05
        leg_thickness = 0.04
        leg_height = seat_height

        # Update main transform
        chair['transform'].setTranslation(QVector3D(cx, cy, 0))
        chair['transform'].setRotationZ(math.degrees(theta))

        # Update seat
        chair['seat_mesh'].setXExtent(seat_width)
        chair['seat_mesh'].setYExtent(seat_depth)
        chair['seat_transform'].setTranslation(QVector3D(0, 0, seat_height))

        # Update backrest
        chair['back_mesh'].setXExtent(seat_width)
        chair['back_mesh'].setZExtent(back_height)
        chair['back_transform'].setTranslation(QVector3D(0, -seat_depth/2 + back_thickness/2,
                                                          seat_height + seat_thickness/2 + back_height/2))

        # Update leg positions and sizes
        leg_offset_x = seat_width / 2.0 - leg_thickness
        leg_offset_y = seat_depth / 2.0 - leg_thickness

        leg_positions = [
            (-leg_offset_x, -leg_offset_y),
            (leg_offset_x, -leg_offset_y),
            (-leg_offset_x, leg_offset_y),
            (leg_offset_x, leg_offset_y),
        ]

        for leg_data, (lx, ly) in zip(chair['leg_entities'], leg_positions):
            leg_data['mesh'].setZExtent(leg_height)
            leg_data['transform'].setTranslation(QVector3D(lx, ly, leg_height / 2.0))

    def _remove_unused_chairs(self, active_ids: set):
        """Remove chair entities that are no longer tracked."""
        to_remove = []
        for key in self.scene_objects:
            if key.startswith('chair_'):
                chair_id = int(key.split('_')[1])
                if chair_id not in active_ids:
                    to_remove.append(key)

        for key in to_remove:
            entity = self.scene_objects[key]['entity']
            entity.setParent(None)
            del self.scene_objects[key]

    def _cleanup_type_changes(self, active_objects: dict):
        """Remove objects of the OPPOSITE type when type changes.

        active_objects: dict of id -> (type, state)

        If current type is 'chair', remove any 'table' or 'uncertain' with same ID.
        If current type is 'table', remove any 'chair' or 'uncertain' with same ID.
        If state is 'uncertain', remove any 'table' or 'chair' with same ID.
        """
        for obj_id, (obj_type, state) in active_objects.items():
            table_key = f'table_{obj_id}'
            chair_key = f'chair_{obj_id}'
            uncertain_key = f'uncertain_{obj_id}'

            if state == 'uncertain':
                # Remove committed objects, keep uncertain
                if table_key in self.scene_objects:
                    self.scene_objects[table_key]['entity'].setParent(None)
                    del self.scene_objects[table_key]
                if chair_key in self.scene_objects:
                    self.scene_objects[chair_key]['entity'].setParent(None)
                    del self.scene_objects[chair_key]
            else:
                # state == 'committed' - remove uncertain and opposite type
                if uncertain_key in self.scene_objects:
                    self.scene_objects[uncertain_key]['entity'].setParent(None)
                    del self.scene_objects[uncertain_key]

                if obj_type == 'chair' and table_key in self.scene_objects:
                    self.scene_objects[table_key]['entity'].setParent(None)
                    del self.scene_objects[table_key]
                elif obj_type == 'table' and chair_key in self.scene_objects:
                    self.scene_objects[chair_key]['entity'].setParent(None)
                    del self.scene_objects[chair_key]

    def update_chairs(self, beliefs: list):
        """Update chair visualizations from belief list."""
        active_ids = set()

        for belief_dict in beliefs:
            # Only process committed chairs
            model_sel = belief_dict.get('model_selection', {})
            state = model_sel.get('state', 'committed')
            if state != 'committed':
                continue

            if belief_dict.get('type') != 'chair':
                continue

            chair_id = belief_dict.get('id', 0)
            active_ids.add(chair_id)

            cx = belief_dict.get('cx', 0)
            cy = belief_dict.get('cy', 0)
            seat_width = belief_dict.get('seat_width', 0.45)
            seat_depth = belief_dict.get('seat_depth', 0.45)
            seat_height = belief_dict.get('seat_height', 0.45)
            back_height = belief_dict.get('back_height', 0.40)
            theta = belief_dict.get('angle', 0)

            self._update_chair(chair_id, cx, cy, seat_width, seat_depth, seat_height, back_height, theta)

        self._remove_unused_chairs(active_ids)

    def get_widget(self):
        return self

    def update(self, robot_pose=None, room_dims=None, beliefs=None, **kwargs):
        """Update visualization with new data."""
        if robot_pose is not None:
            self.update_robot_pose(float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))

        if room_dims is not None and self._initialized:
            new_dims = (float(room_dims[0]), float(room_dims[1]))
            if new_dims != self.room_dims:
                self.room_dims = new_dims
                self._update_floor_size(new_dims[0], new_dims[1])
                self._update_walls(new_dims[0], new_dims[1])

        if beliefs is not None and self._initialized:
            # Track which IDs are active and their types/states
            active_objects = {}  # id -> (type, state)
            for b in beliefs:
                obj_id = b.get('id', 0)
                obj_type = b.get('type', 'unknown')
                model_sel = b.get('model_selection', {})
                state = model_sel.get('state', 'committed')
                active_objects[obj_id] = (obj_type, state)

            # Remove objects whose type has changed
            self._cleanup_type_changes(active_objects)

            # Update uncertain objects, tables and chairs
            self.update_uncertain_objects(beliefs)
            self.update_tables(beliefs)
            self.update_chairs(beliefs)

        # Periodic save of camera settings
        self._update_counter += 1
        if self._update_counter >= self._save_interval:
            self._update_counter = 0
            self.save_settings()

    def start_async(self):
        pass

    def save_settings(self):
        """Save camera position and target to file."""
        if not self._initialized or self.camera is None:
            return

        try:
            settings = {
                'camera_position': [
                    self.camera.position().x(),
                    self.camera.position().y(),
                    self.camera.position().z()
                ],
                'camera_target': [
                    self.cam_controller.target.x(),
                    self.cam_controller.target.y(),
                    self.cam_controller.target.z()
                ]
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"[Qt3D] Failed to save settings: {e}")

    def load_settings(self):
        """Load camera position and target from file."""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)

                # Restore camera position
                if 'camera_position' in settings:
                    pos = settings['camera_position']
                    self.camera.setPosition(QVector3D(pos[0], pos[1], pos[2]))

                # Restore camera target
                if 'camera_target' in settings:
                    target = settings['camera_target']
                    self.cam_controller.target = QVector3D(target[0], target[1], target[2])
                    self.camera.setViewCenter(self.cam_controller.target)

                print("[Qt3D] Loaded saved camera settings")
        except Exception as e:
            print(f"[Qt3D] Failed to load settings: {e}")

    def closeEvent(self, event):
        """Save settings when widget is closed."""
        self.save_settings()
        self.cleanup()
        super().closeEvent(event)

    def cleanup(self):
        """Clean up Qt3D resources in correct order to avoid segfault."""
        if not self._initialized:
            return

        try:
            # Save settings first
            self.save_settings()

            # Remove event filter
            if self.view and self.cam_controller:
                self.view.removeEventFilter(self.cam_controller)

            # Clear scene objects (remove from scene graph)
            for key in list(self.scene_objects.keys()):
                obj = self.scene_objects[key]
                if isinstance(obj, dict) and 'entity' in obj:
                    entity = obj['entity']
                    if entity:
                        entity.setParent(None)
                elif hasattr(obj, 'setParent'):
                    obj.setParent(None)
            self.scene_objects.clear()

            # Disconnect root entity from view
            if self.view and self.root_entity:
                self.view.setRootEntity(None)

            self._initialized = False
            print("[Qt3D] Cleanup completed")

        except Exception as e:
            print(f"[Qt3D] Cleanup error: {e}")


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Qt3D Test")
    window.resize(800, 600)
    viewer = Qt3DObjectVisualizerWidget()
    window.setCentralWidget(viewer)
    window.show()
    sys.exit(app.exec())
