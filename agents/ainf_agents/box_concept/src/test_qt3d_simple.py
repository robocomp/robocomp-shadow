#!/usr/bin/env python3
"""
Simple Qt3D test - just show a cube to verify Qt3D works
"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.Qt3DExtras import Qt3DExtras


def main():
    app = QApplication(sys.argv)

    # Create window
    view = Qt3DExtras.Qt3DWindow()
    view.defaultFrameGraph().setClearColor(QColor(50, 50, 60))
    view.setTitle("Qt3D Simple Test")
    view.resize(800, 600)

    # Root entity
    root_entity = Qt3DCore.QEntity()

    # Camera
    camera = view.camera()
    camera.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 1000.0)
    camera.setPosition(QVector3D(0, 0, 10))
    camera.setViewCenter(QVector3D(0, 0, 0))
    camera.setUpVector(QVector3D(0, 1, 0))

    # Camera controller
    cam_controller = Qt3DExtras.QOrbitCameraController(root_entity)
    cam_controller.setCamera(camera)
    cam_controller.setLinearSpeed(50.0)
    cam_controller.setLookSpeed(180.0)

    # Light
    light_entity = Qt3DCore.QEntity(root_entity)
    light = Qt3DRender.QPointLight(light_entity)
    light.setColor(QColor(255, 255, 255))
    light.setIntensity(1.0)
    light_entity.addComponent(light)

    light_transform = Qt3DCore.QTransform(light_entity)
    light_transform.setTranslation(QVector3D(5, 5, 5))
    light_entity.addComponent(light_transform)

    # Create a simple cube
    cube_entity = Qt3DCore.QEntity(root_entity)

    cube_mesh = Qt3DExtras.QCuboidMesh()
    cube_mesh.setXExtent(2.0)
    cube_mesh.setYExtent(2.0)
    cube_mesh.setZExtent(2.0)

    cube_material = Qt3DExtras.QPhongMaterial()
    cube_material.setDiffuse(QColor(255, 0, 0))  # Red
    cube_material.setAmbient(QColor(100, 0, 0))

    cube_transform = Qt3DCore.QTransform()
    cube_transform.setTranslation(QVector3D(0, 0, 0))

    cube_entity.addComponent(cube_mesh)
    cube_entity.addComponent(cube_material)
    cube_entity.addComponent(cube_transform)

    print("Created cube entity")
    print(f"  Mesh: {cube_mesh}")
    print(f"  Material: {cube_material}")
    print(f"  Transform: {cube_transform}")

    # Set root entity
    view.setRootEntity(root_entity)

    # Show
    view.show()

    print("Window shown - you should see a red cube")
    print("Use mouse to orbit camera")

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
