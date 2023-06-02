from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from PySide6.Qt3DCore import *
from PySide6.Qt3DExtras import *
from PySide6.Qt3DRender import *
from PySide6.QtGui import *
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import *

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

        # for camera control
        self.camController = Qt3DExtras.QOrbitCameraController(self.root)
        self.camController.setCamera(self.view.camera())

        # Material
        self.material = Qt3DExtras.QPhongMaterial(self.root)

        # Infinite plane
        #self.gridEntity = InfiniteGridEntity(self.root)

        self.container.show()

# class GLViewer(QOpenGLWidget):
#     def __init__(self, parent, points):
#         super().__init__(parent)
#         self.resize(parent.width(), parent.height())
#         self.points = points
#
#     def initializeGL(self):
#         glDisable(GL_LIGHTING)
#         glPointSize(3.0)
#         #camera().setSceneRadius(5)
#
#     def resizeGL(self, width, height):
#         glViewport(0, 0, width, height)
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         gluPerspective(45, width / height, 0.1, 100.0)
#         glMatrixMode(GL_MODELVIEW)
#
#     def paintGL(self):
#         #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         #glLoadIdentity()
#         #gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
#
#         glColor3f(0.7, 0.7, 0.7)
#         self.drawGrid(5.0, 10)
#         glBegin(GL_POINTS)
#         for vertex in self.points:
#             glVertex3f(*vertex)
#         glEnd()
#
#     def drawGrid(self, size, nbSubdivisions):
#         lighting = glGetBooleanv(GL_LIGHTING)
#         glDisable(GL_LIGHTING)
#
#         glBegin(GL_LINES)
#         for i in range(nbSubdivisions):
#             pos = size * (2.0 * i/nbSubdivisions - 1.0)
#             glVertex2d(pos, -size)
#             glVertex2d(pos, +size)
#             glVertex2d(-size, pos)
#             glVertex2d(size, pos)
#         glEnd()
#
#         if lighting:
#             glEnable(GL_LIGHTING)
#
