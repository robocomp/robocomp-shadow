from PySide6.Qt3DExtras import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender

class GLViewer:
    def __init__(self, parent, points):
        #super().__init__()
        self.points = points
        self.view = Qt3DExtras.Qt3DWindow()
        self.container = parent.createWindowContainer(self.view)
        self.container.setParent(parent)
        self.container.resize(parent.width(), parent.height())
        self.root = Qt3DCore.QEntity()
        self.view.setRootEntity(self.root)

        # Camera
        self.view.camera().lens().setPerspectiveProjection(45, 16 / 9, 0.1, 1000)
        self.view.camera().setPosition(QVector3D(0, 0, 40))
        self.view.camera().setViewCenter(QVector3D(0, 0, 0))

        # Material
        self.material = Qt3DExtras.QPhongMaterial(self.root)

        # Create a grid geometry
        self.grid_geometry = Qt3DExtras.QPlaneMesh()
        self.grid_geometry.setWidth(20)
        self.grid_geometry.setHeight(20)
        self.grid_geometry.setMeshResolution(QSize(20, 20))

        # for camera control
        self.camController = Qt3DExtras.QOrbitCameraController(self.root)
        self.camController.setCamera(self.view.camera())

        # Create a grid plane
        self.grid_plane = self.create_grid_plane(20, 20)

        # load mesh
        self.load_stl_model()

        self.view.show()
        self.container.show()

    def create_grid_plane(self, size, resolution):

        # Create a grid geometry
        self.grid_geometry = Qt3DExtras.QPlaneMesh()
        self.grid_geometry.setWidth(size)
        self.grid_geometry.setHeight(size)
        self.grid_geometry.setMeshResolution(QSize(resolution, resolution))

        # Create a grid material
        self.grid_material = Qt3DExtras.QPhongMaterial()
        self.grid_material.setAmbient(QColor(Qt.darkGray))

        self.planeTransform = Qt3DCore.QTransform()
        self.planeTransform.setScale(2.0)
        self.planeTransform.setRotationX(45)

        # Create a grid entity
        self.grid_entity = Qt3DCore.QEntity(self.root)
        self.grid_entity.addComponent(self.grid_geometry)
        self.grid_entity.addComponent(self.planeTransform)
        self.grid_entity.addComponent(self.grid_material)

    def load_stl_model(self):
        self.modelEntity = Qt3DCore.QEntity(self.grid_entity)
        self.modelMesh = Qt3DRender.QMesh()
        self.modelMesh.setSource(QUrl.fromLocalFile('shadowV2.stl'))

        self.modelTransform = Qt3DCore.QTransform()
        self.modelTransform.setScale(5)
        self.modelTransform.setRotationX(-90)

        self.modelEntity.addComponent(self.modelMesh)
        self.modelEntity.addComponent(self.modelTransform)
        self.modelEntity.addComponent(self.material)

