import math
import sys
import random
from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt
from viewers._abstract_graphic_view import AbstractGraphicViewer
from .graph_edge import GraphicsEdge
from .graph_node import GraphicsNode
from pydsr import signals

class GraphViewer(AbstractGraphicViewer):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.gmap = {}  # node_id: QGraphicsEllipseItem
        self.gmap_edges = {}  # node_id: QGraphicsPathItem
        self.type_id_map = {}
        self._internal_update = False
        self.create_graph()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio )
        self.contextMenu = QMenu()
        self.showMenu = self.contextMenu.addMenu("&Show:")
        self.connect_graph_signals()

    def create_graph(self):
        self.blockSignals(True)
        self.gmap = {}
        self.gmap_edges = {}
        self.type_id_map = {}
        self.scene.clear()
        for node in self.g.get_nodes():
            self.add_or_assign_node_slot(node.id, node.type)
            self.type_id_map[node.type] = node.id
        for node in self.g.get_nodes():
            for edge in node.edges:
                self.add_or_assign_edge_slot(node.id, edge[0], edge[1])
        self.blockSignals(False)

    def connect_graph_signals(self):
        signals.connect(self.g, signals.UPDATE_NODE, self.add_or_assign_node_slot)
        signals.connect(self.g, signals.UPDATE_EDGE, self.add_or_assign_edge_slot)
        signals.connect(self.g, signals.DELETE_NODE, self.del_node_slot)
        signals.connect(self.g, signals.DELETE_EDGE, self.del_edge_slot)

    ########################################################################
    ###  DSR SIGNALS HANDLER
    ########################################################################
    @QtCore.Slot()
    def add_or_assign_node_slot(self, node_id: int, mtype: str):
        node = self.g.get_node(node_id)
        gnode = None
        if node:
            if node.id not in self.gmap:
                print("Graph viewer: Adding node", node.name)
                gnode = GraphicsNode(self)
                gnode.id_in_graph = node.id
                gnode.set_type(mtype)
                gnode.set_tag(node.name)
                self.scene.addItem(gnode)
                ## connect
                self.gmap[node.id] = gnode
                color = "coral"
                if node.attrs.__contains__("color"):
                    color = node.attrs["color"].value
                gnode.set_node_color(color)
                gnode.set_type(mtype)
            else:
                gnode = self.gmap[node.id]
        if node.attrs.__contains__("pos_x"):
            px = node.attrs["pos_x"].value
        else:
            px = random.uniform(-300, 300)
        if node.attrs.__contains__("pos_y"):
           py = node.attrs["pos_y"].value
        else:
           py = random.uniform(-300, 300)
        gnode.setPos(float(px), float(py))

        for edge in node.edges:
            key = (node.id, edge[0], edge[1])
            if key in self.gmap_edges and self.gmap_edges[key] is None:
                self.add_or_assign_edge_slot(node.id, edge[0], edge[1])

    def add_or_assign_edge_slot(self, fr: int, to: int, mtype: str):
        # Skip if already present
        key = (fr, to, mtype)
        if key in self.gmap_edges.keys():
            return
        try:
            if self.g.get_edge(fr, to, mtype):
                if key not in self.gmap_edges.keys():
                    #print("Graph viewer: edge accessed OK", fr, to, mtype)
                    source_node = self.gmap[fr]
                    #print("Graph viewer: edge accessed 2 OK", fr, to, mtype)
                    destination_node = self.gmap[to]
                    #print("Graph viewer: edge accessed 3 OK", fr, to, mtype)
                    item = GraphicsEdge(source_node, destination_node, mtype)
                    self.gmap_edges[key] = item
                    #print("Graph viewer: edge accessed 4 OK", fr, to, mtype)
                    self.scene.addItem(item)
        except Exception as e:
            print("Graph viewer: Exception in add_or_assign_edge", fr, to, mtype)
            #QMessageBox.critical("Error", f"Graph viewer: Exception in add_or_assign_edge {e}")  # TODO: Fix this

    def del_node_slot(self, id: int):
        try:
            # print(f"[SLOT] Delete node: {id}")
            while id in self.gmap:
                item = self.gmap[id]
                self.scene.removeItem(item)
                del item
                del self.gmap[id]
        except Exception as e:
            print(f"{e} Error {__name__}")

    def del_edge_slot(self, fr: int, to: int, mtype: str):
        key = (fr, to, mtype)
        if key not in self.gmap_edges.keys():
            print("Graph viewer: In delete_edge_slot -> Edge not found", key)
            return
        try:
            while key in self.gmap_edges:
                edge = self.gmap_edges.pop(key)
                if fr in self.gmap:
                    self.gmap[fr].delete_edge(edge)
                if to in self.gmap:
                    self.gmap[to].delete_edge(edge)
                if edge:
                    self.scene.removeItem(edge)
                    del edge
        except Exception as e:
            print("Graph viewer: Exception in del_edge_slot", e)
            # QMessageBox.critical(self, "Error", f"Graph viewer: Exception in del_edge_slot {e}")
            return

    ################## EVENTS ########################
    def mousePressEvent(self, event):
        item = self.scene.itemAt(self.mapToScene(event.pos()), QtGui.QTransform())
        if item:
            super().mousePressEvent(event)
        elif event.button() == QtCore.Qt.RightButton:
            self.showContextMenu(event)
        else:
            super().mousePressEvent(event)

    def showContextMenu(self, event):
      self.contextMenu.exec()
