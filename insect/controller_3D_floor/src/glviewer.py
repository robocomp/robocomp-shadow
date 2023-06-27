import cv2
from PySide6.Qt3DExtras import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
import copy
import numpy as np

class Image(Qt3DRender.QPaintedTextureImage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None

    def paint(self, painter):
        rect = QRect(0, 0, self._image.width(), self._image.height())
        painter.drawImage(rect, self._image, rect)

    def set_image(self, image):
        self._image = image
        self.setSize(image.size())
        painter = QPainter()
        self.paint(painter)

class GLViewer:
    def __init__(self, parent, points):
        #super().__init__()
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

        # Create a light
        self.light_entity = Qt3DCore.QEntity(self.root)
        self.light = Qt3DRender.QDirectionalLight(self.light_entity)
        self.light.setColor(Qt.white)
        self.light.setIntensity(0.3)
        self.light_transform = Qt3DCore.QTransform()
        self.light_transform.setTranslation(QVector3D(0, 0, 50))
        self.light_entity.addComponent(self.light)
        self.light_entity.addComponent(self.light_transform)

        # Create a grid geometry
        self.grid_geometry = Qt3DExtras.QPlaneMesh()
        self.grid_geometry.setWidth(20)
        self.grid_geometry.setHeight(20)
        self.grid_geometry.setMeshResolution(QSize(20, 20))

        # Create a floor geometry
        self.floor_geometry = Qt3DExtras.QPlaneMesh()
        self.floor_geometry.setWidth(10)
        self.floor_geometry.setHeight(10)
        self.floor_geometry.setMeshResolution(QSize(10, 10))

        # for camera control
        self.camController = Qt3DExtras.QOrbitCameraController(self.root)
        self.camController.setCamera(self.view.camera())

        # Create a grid plane
        self.grid_plane = self.create_grid_plane(20, 20)
        self.floor_plane = self.create_floor_plane(10, 10)

        # load mesh
        self.load_stl_model()

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
        self.grid_material.setAmbient(QColor(Qt.lightGray))


        self.planeTransform = Qt3DCore.QTransform()
        self.planeTransform.setScale(2.0)
        self.planeTransform.setRotationX(45)

        # Create a grid entity
        self.grid_entity = Qt3DCore.QEntity(self.root)
        self.grid_entity.addComponent(self.grid_geometry)
        self.grid_entity.addComponent(self.planeTransform)
        self.grid_entity.addComponent(self.grid_material)

    def create_floor_plane(self, size, resolution):
        # Create a grid geometry
        self.floor_geometry = Qt3DExtras.QPlaneMesh()
        self.floor_geometry.setWidth(size)
        self.floor_geometry.setHeight(size)
        self.floor_geometry.setMeshResolution(QSize(resolution, resolution))

        # Create a grid material
        self.floor_material = Qt3DExtras.QNormalDiffuseMapMaterial()
        self.image = Image()
        array_created = np.full((500, 500, 3),
                                np.random.choice(range(256), size=3), dtype=np.uint8)
        qimage = QImage(array_created, array_created.shape[1], array_created.shape[0],
                        QImage.Format_RGB888)
        self.image.set_image(qimage)  # type(image): QImage
        self.floor_material.normal().addTextureImage(self.image)
        self.floor_material.diffuse().addTextureImage(self.image)
        self.floor_material.setTextureScale(0.2)
        self.floor_material.setAmbient(QColor.fromRgbF(1, 1, 1, 1))

        self.floorTransform = Qt3DCore.QTransform()
        self.floorTransform.setScale(1)
        self.floorTransform.setRotationX(45)
        self.floorTransform.setTranslation(QVector3D(0, 0, 0.005))

        # Create a grid entity
        self.floor_entity = Qt3DCore.QEntity(self.root)
        self.floor_entity.addComponent(self.floor_geometry)
        self.floor_entity.addComponent(self.floorTransform)
        self.floor_entity.addComponent(self.floor_material)

    def update_floor_plane(self, mask):

        mask_img = np.frombuffer(mask.image, dtype=np.uint8).reshape((mask.height, mask.width, 1))
        array_created = np.full((500, 500, 3),
                                np.random.choice(range(256), size=3), dtype=np.uint8)
        print(mask_img.shape)
        qimage = QImage(mask_img, mask_img.shape[1], mask_img.shape[0],
                        QImage.Format_RGB888)
        self.image = Image()
        self.image.set_image(qimage)  # type(image): QImage
        self.floor_material.normal().addTextureImage(self.image)
        self.floor_material.diffuse().addTextureImage(self.image)
        self.floor_material.setTextureScale(0.3)
        self.floor_material.setAmbient(QColor.fromRgbF(1, 1, 1, 1))

    def load_stl_model(self):
        self.modelEntity = Qt3DCore.QEntity(self.grid_entity)
        self.modelMesh = Qt3DRender.QMesh()
        self.modelMesh.setSource(QUrl.fromLocalFile('meshes/shadow20k.stl'))

        self.modelTransform = Qt3DCore.QTransform()
        self.modelTransform.setScale(1)
        self.modelTransform.setRotationX(-90)
        self.modelTransform.setTranslation(QVector3D(0, 0, 0))

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
                #print(element.id, element.left, element.right, element_center)
                # globals()["modelTransform"+"_"+str(element.id)].setTranslation(QVector3D(-3*np.sin(np.deg2rad(element_angle)), 0, 3*np.cos(np.deg2rad(element_angle))))
                globals()["modelTransform"+"_"+str(element.id)].setTranslation(QVector3D(element.x/1000, 0, -element.y/1000))
                #globals()["modelTransform" + "_" + str(element.id)].setTranslation(QVector3D(element.x, 0, element.y))
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

            globals()["modelTransform" + "_" + str(element.id)].setTranslation(QVector3D(element.x/1000, 0, -element.y/1000))

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
