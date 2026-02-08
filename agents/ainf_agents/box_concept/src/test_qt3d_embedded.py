#!/usr/bin/env python3
"""
Minimal Qt3D test embedded in a QWidget
"""
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtGui import QColor, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.Qt3DExtras import Qt3DExtras


def main():
    app = QApplication(sys.argv)

    # Main window
    main_window = QMainWindow()
    main_window.setWindowTitle("Qt3D Embedded Test")
    main_window.resize(800, 600)

    # Create Qt3D window
    view = Qt3DExtras.Qt3DWindow()
    view.defaultFrameGraph().setClearColor(QColor(50, 50, 60))

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

    # Light
    light_entity = Qt3DCore.QEntity(root_entity)
    light = Qt3DRender.QPointLight(light_entity)
    light.setColor(QColor(255, 255, 255))
    light.setIntensity(1.0)
    light_entity.addComponent(light)

    light_transform = Qt3DCore.QTransform(light_entity)
    light_transform.setTranslation(QVector3D(5, 5, 5))
    light_entity.addComponent(light_transform)

    # Red cube
    cube_entity = Qt3DCore.QEntity(root_entity)

    cube_mesh = Qt3DExtras.QCuboidMesh()
    cube_mesh.setXExtent(2.0)
    cube_mesh.setYExtent(2.0)
    cube_mesh.setZExtent(2.0)

    cube_material = Qt3DExtras.QPhongMaterial()
    cube_material.setDiffuse(QColor(255, 0, 0))
    cube_material.setAmbient(QColor(100, 0, 0))

    cube_entity.addComponent(cube_mesh)
    cube_entity.addComponent(cube_material)

    # Set root
    view.setRootEntity(root_entity)

    # Create container widget and embed Qt3D window
    container = QWidget.createWindowContainer(view)
    container.setMinimumSize(400, 300)

    # Central widget with layout
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(container)

    main_window.setCentralWidget(central_widget)
    main_window.show()

    print("Main window shown with embedded Qt3D")
    print("You should see a red cube")

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
