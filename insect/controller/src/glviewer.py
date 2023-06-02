from PySide6.Qt3DExtras import *
from PySide6.QtGui import *
from PySide6.QtCore import *
<<<<<<< HEAD

# class InfiniteGridEntity(Qt3DCore.QEntity):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#
#         self.geometry = Qt3DRender.QGeometry(self)
#         self.vertexBuffer = Qt3DRender.QBuffer(self.geometry)
#         self.indexBuffer = Qt3DRender.QBuffer(self.geometry)
#
#         self.buildGeometry()
#
#         self.geometryRenderer = Qt3DRender.QGeometryRenderer(self.geometry)
#         self.geometryRenderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.Lines)
#         self.geometryRenderer.setVertexCount(0)
#         self.geometryRenderer.setIndexOffset(0)
#
#         self.geometryRenderer.geometry().addAttribute(self.vertexBuffer)
#         self.geometryRenderer.geometry().addAttribute(self.indexBuffer)
#
#         self.addComponent(self.geometryRenderer)
#
#     def buildGeometry(self):
#         vertices = []
#         indices = []
#
#         size = 10  # Grid size
#         step = 1  # Grid spacing
#
#         for i in range(-size, size + 1):
#             vertices.append(QVector3D(i * step, 0, -size * step))
#             vertices.append(QVector3D(i * step, 0, size * step))
#
#             vertices.append(QVector3D(-size * step, 0, i * step))
#             vertices.append(QVector3D(size * step, 0, i * step))
#
#             indices.append(2 * i)
#             indices.append(2 * i + 1)
#
#             indices.append(2 * (i + size + 1))
#             indices.append(2 * (i + size + 1) + 1)
#
#         verticesData = QByteArray()
#         verticesData.resize(len(vertices) * 3 * 4)
#         verticesDataPointer = QDataStream(verticesData, QIODevice.WriteOnly)
#         verticesDataPointer.setVersion(QDataStream.Qt_5_15)
#
#         for vertex in vertices:
#             verticesDataPointer << vertex.x()
#             verticesDataPointer << vertex.y()
#             verticesDataPointer << vertex.z()
#
#         indicesData = QByteArray()
#         indicesData.resize(len(indices) * 4)
#         indicesDataPointer = QDataStream(indicesData, QIODevice.WriteOnly)
#         indicesDataPointer.setVersion(QDataStream.Qt_5_15)
#
#         for index in indices:
#             indicesDataPointer << index
#
#         self.vertexBuffer.setData(verticesData)
#         self.indexBuffer.setData(indicesData)
#         self.geometryRenderer.setVertexCount(len(vertices))
#
=======
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
import copy
import numpy as np
>>>>>>> 4415da9637839d3c674406fcb566756c0b59eb19

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

<<<<<<< HEAD
        # Infinite plane
        #self.gridEntity = InfiniteGridEntity(self.root)
=======
        # load mesh
        self.load_stl_model()
>>>>>>> 4415da9637839d3c674406fcb566756c0b59eb19

        self.view.show()
        self.container.show()

        # Dict for inferenced elements models
        self.elements_model = {}

        self.meshes_dict = {
            "0" : "meshes/man.stl"
        }

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
        self.modelMesh.setSource(QUrl.fromLocalFile('meshes/shadow20k.stl'))

        self.modelTransform = Qt3DCore.QTransform()
        self.modelTransform.setScale(1)
        self.modelTransform.setRotationX(-90)
        self.modelTransform.setTranslation(QVector3D(0,0,0))

        self.modelEntity.addComponent(self.modelMesh)
        self.modelEntity.addComponent(self.modelTransform)
        self.modelEntity.addComponent(self.material)

    def process_elements(self, elements):
        new_elements = []
        elements_dict_keys = list(self.elements_model.keys())
        # print("element list size:", len(elements))
        for element in elements:
            # Update element
            if element.id in elements_dict_keys:
                element_center = ((element.right-element.left)/2)+element.left
                if abs(element.right-element.left) > 1920/2:
                    element_center = element.right/2 if element.right < element.left else element.left/2

                element_angle = element_center * 359 / 1919
                # print("ANGLE", element_angle)
                # print(QPointF(3*np.cos(element_angle), 3*np.sin(element_angle)))
                print(element.id, element.left, element.right, element_center)
                globals()["modelTransform"+"_"+str(element.id)].setTranslation(QVector3D(-3*np.sin(np.deg2rad(element_angle)), 0, 3*np.cos(np.deg2rad(element_angle))))
                elements_dict_keys.remove(element.id)
            else:
                new_elements.append(element)
        # Insert new elements
        self.insert_new_element(new_elements)
        # print("elements_dict_keys", elements_dict_keys)
        # print("KEYS To REMOVE:", elements_dict_keys)
        # Remove lost elements
        self.remove_element(elements_dict_keys)

    def insert_new_element(self, elements):
        for element in elements:
            globals()["modelEntity"+"_"+str(element.id)] = Qt3DCore.QEntity(self.grid_entity)
            globals()["modelMesh"+"_"+str(element.id)] = Qt3DRender.QMesh()
            globals()["modelMesh"+"_"+str(element.id)].setSource(QUrl.fromLocalFile('meshes/man.stl'))

            globals()["modelTransform"+"_"+str(element.id)] = Qt3DCore.QTransform()
            globals()["modelTransform"+"_"+str(element.id)].setScale(1)
            globals()["modelTransform"+"_"+str(element.id)].setRotationX(-90)
            element_center = ((element.right - element.left) / 2) + element.left
            if element_center > 1919:
                element_center = 1920 - element_center
            element_angle = element_center * 359 / 1919
            globals()["modelTransform"+"_"+str(element.id)].setTranslation(QVector3D(-3*np.sin(np.deg2rad(element_angle)), 0, 3*np.cos(np.deg2rad(element_angle))))

            globals()["modelEntity"+"_"+str(element.id)].addComponent(globals()["modelMesh"+"_"+str(element.id)])
            globals()["modelEntity"+"_"+str(element.id)].addComponent(globals()["modelTransform"+"_"+str(element.id)])
            globals()["modelEntity"+"_"+str(element.id)].addComponent(self.material)

            # print("MODELENTITY", modelEntity)
            self.elements_model[element.id] = globals()["modelEntity"+"_"+str(element.id)]

        # print(globals())
    def remove_element(self, elements):
        for element in elements:
            globals()["modelEntity"+"_"+str(element)].setEnabled(False)
            # del globals()["modelMesh" + "_" + str(element)]
            # # del globals()["modelTransform" + "_" + str(element)]
            self.elements_model.pop(element)