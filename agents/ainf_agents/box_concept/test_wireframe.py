#!/usr/bin/env python3
"""Test script to verify wireframe box rendering in Qt3D."""

import sys
import math
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QVector3D, QQuaternion
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.Qt3DExtras import Qt3DExtras


def create_wireframe_box(root_entity, cx, cy, cz, width, depth, height, color):
    """Create a wireframe box - edges directly as children of root_entity."""

    line_thickness = 0.03
    hw = width / 2.0
    hd = depth / 2.0

    # IMPORTANT: Store ALL references to prevent garbage collection
    created_objects = {
        'entities': [],
        'meshes': [],
        'materials': [],
        'transforms': []
    }

    # Use thin cuboids - create each edge directly as child of root_entity
    edges = []

    # 4 vertical edges (along Z)
    for x, y in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]:
        edges.append((cx + x, cy + y, cz + height/2, line_thickness, line_thickness, height))

    # 4 bottom horizontal edges along X (at z=0)
    for y in [-hd, hd]:
        edges.append((cx, cy + y, cz + line_thickness/2, width, line_thickness, line_thickness))

    # 4 bottom horizontal edges along Y (at z=0)
    for x in [-hw, hw]:
        edges.append((cx + x, cy, cz + line_thickness/2, line_thickness, depth, line_thickness))

    # 4 top horizontal edges along X (at z=height)
    for y in [-hd, hd]:
        edges.append((cx, cy + y, cz + height - line_thickness/2, width, line_thickness, line_thickness))

    # 4 top horizontal edges along Y (at z=height)
    for x in [-hw, hw]:
        edges.append((cx + x, cy, cz + height - line_thickness/2, line_thickness, depth, line_thickness))

    print(f"Creating {len(edges)} edges...")

    for i, (ex, ey, ez, sx, sy, sz) in enumerate(edges):
        # Create edge entity DIRECTLY as child of root_entity
        edge_entity = Qt3DCore.QEntity(root_entity)

        # Cuboid mesh
        mesh = Qt3DExtras.QCuboidMesh()
        mesh.setXExtent(sx)
        mesh.setYExtent(sy)
        mesh.setZExtent(sz)

        # Material - create NEW for each edge
        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(color)
        material.setAmbient(color.darker(120))

        # Transform
        transform = Qt3DCore.QTransform()
        transform.setTranslation(QVector3D(ex, ey, ez))

        # Add components in same order as working cube
        edge_entity.addComponent(mesh)
        edge_entity.addComponent(material)
        edge_entity.addComponent(transform)

        # Store ALL references
        created_objects['entities'].append(edge_entity)
        created_objects['meshes'].append(mesh)
        created_objects['materials'].append(material)
        created_objects['transforms'].append(transform)

        print(f"  Edge {i}: pos=({ex:.2f}, {ey:.2f}, {ez:.2f}), size=({sx:.2f}, {sy:.2f}, {sz:.2f})")

    return created_objects


# Global variable to prevent garbage collection
_wireframe_objects = None

def main():
    global _wireframe_objects
    app = QApplication(sys.argv)

    # Create Qt3D window
    view = Qt3DExtras.Qt3DWindow()
    view.defaultFrameGraph().setClearColor(QColor(50, 50, 60))
    view.setTitle("Wireframe Box Test")
    view.resize(800, 600)

    # Create root entity
    root_entity = Qt3DCore.QEntity()

    # Setup camera
    camera = view.camera()
    camera.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 1000.0)
    camera.setPosition(QVector3D(5, 5, 5))
    camera.setViewCenter(QVector3D(0, 0, 0.5))
    camera.setUpVector(QVector3D(0, 0, 1))

    # Camera controller
    cam_controller = Qt3DExtras.QOrbitCameraController(root_entity)
    cam_controller.setCamera(camera)
    cam_controller.setLinearSpeed(5.0)
    cam_controller.setLookSpeed(180.0)

    # Lighting
    light_entity = Qt3DCore.QEntity(root_entity)
    light = Qt3DRender.QPointLight()
    light.setColor(QColor(255, 255, 255))
    light.setIntensity(1.5)
    light_entity.addComponent(light)
    light_transform = Qt3DCore.QTransform()
    light_transform.setTranslation(QVector3D(5, 5, 10))
    light_entity.addComponent(light_transform)

    # Create a simple floor for reference
    floor_entity = Qt3DCore.QEntity(root_entity)
    floor_mesh = Qt3DExtras.QPlaneMesh()
    floor_mesh.setWidth(4)
    floor_mesh.setHeight(4)
    floor_material = Qt3DExtras.QPhongMaterial()
    floor_material.setDiffuse(QColor(100, 100, 100))
    floor_transform = Qt3DCore.QTransform()
    floor_transform.setRotationX(90)  # Rotate to be horizontal
    floor_entity.addComponent(floor_mesh)
    floor_entity.addComponent(floor_material)
    floor_entity.addComponent(floor_transform)

    # Create a solid cube for comparison
    print("Creating solid cube at (-1, 0, 0.25)...")
    cube_entity = Qt3DCore.QEntity(root_entity)
    cube_mesh = Qt3DExtras.QCuboidMesh()
    cube_mesh.setXExtent(0.5)
    cube_mesh.setYExtent(0.5)
    cube_mesh.setZExtent(0.5)
    cube_material = Qt3DExtras.QPhongMaterial()
    cube_material.setDiffuse(QColor(0, 150, 200))
    cube_transform = Qt3DCore.QTransform()
    cube_transform.setTranslation(QVector3D(-1, 0, 0.25))
    cube_entity.addComponent(cube_mesh)
    cube_entity.addComponent(cube_material)
    cube_entity.addComponent(cube_transform)

    # Create a SECOND orange cube - same code pattern as blue cube
    print("Creating orange cube at (0, -1, 0.25)...")
    cube2_entity = Qt3DCore.QEntity(root_entity)
    cube2_mesh = Qt3DExtras.QCuboidMesh()
    cube2_mesh.setXExtent(0.5)
    cube2_mesh.setYExtent(0.5)
    cube2_mesh.setZExtent(0.5)
    cube2_material = Qt3DExtras.QPhongMaterial()
    cube2_material.setDiffuse(QColor(255, 180, 0))  # Orange
    cube2_material.setAmbient(QColor(255, 180, 0).darker(120))
    cube2_transform = Qt3DCore.QTransform()
    cube2_transform.setTranslation(QVector3D(0, -1, 0.25))
    cube2_entity.addComponent(cube2_mesh)
    cube2_entity.addComponent(cube2_material)
    cube2_entity.addComponent(cube2_transform)

    # Create wireframe box - ENABLED
    print("Creating wireframe box at (1, 0, 0)...")
    _wireframe_objects = create_wireframe_box(
        root_entity,
        1, 0, 0,  # position
        0.5, 0.5, 0.5,  # size
        QColor(255, 180, 0)  # orange color
    )

    # Set root entity
    view.setRootEntity(root_entity)
    view.show()

    print("Window opened. You should see:")
    print("  - Gray floor")
    print("  - Blue solid cube on the left")
    print("  - Orange wireframe box on the right")
    print("\nUse mouse to rotate the camera.")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
