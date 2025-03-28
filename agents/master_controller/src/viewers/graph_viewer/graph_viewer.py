import math

from PySide6 import QtCore
from PySide6.QtCore import QMutex
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QMenu, QGraphicsPathItem
from PySide6.QtGui import QColor, QPen, QBrush, QAction, QTransform, QPainterPath
from viewers._abstract_graphic_view import AbstractGraphicViewer
from .graph_edge import GraphicsEdge
from .graph_node import GraphicsNode


class GraphViewer(AbstractGraphicViewer):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.node_items = {}  # node_id: QGraphicsEllipseItem
        self.draw_entire_graph()

    def draw_entire_graph(self):
        for node in self.g.get_nodes():
            self._add_node(node)

    def _add_node(self, node):
        circle = GraphicsNode(node, self.g)
        self.scene.addItem(circle)
        self.node_items[node.id] = circle
        for edge in node.edges:
            edge_item = self._add_edge(node.id, edge[0], edge[1])
            #self.node_items[node.id].add_edge(edge[0], edge_item)
            #self.node_items[edge[0]].add_edge(edge[0], edge_item)

    def _add_edge(self, source_id, destination_id, edge_type):
        #Count how many parallel edges exist between these two nodes
        source = self.g.get_node(source_id)
        for index, edge in enumerate(source.edges):
            if edge == (destination_id, edge_type):
                break
        graphics_edge = GraphicsEdge(source_id, destination_id, edge_type, self.g, offset_index=index)
        self.scene.addItem(graphics_edge)
        return graphics_edge

    def _node_context_menu(self, event, node):
        menu = QMenu()
        for attr_name, attr in node.attrs.items():
            action = QAction(f"{attr_name}: {attr.value}", menu)
            action.setEnabled(False)
            menu.addAction(action)
        menu.exec(event.screenPos())

    ########################################################################
    @QtCore.Slot()
    def update_node_slot(self, id: int, type: str):
        if not id in self.node_items.keys():
            self._add_node(self.g.get_node(id))

    def update_node_attrs_slot(self, id: int, attribute_names: [str]):
        if id in self.node_items.keys():
            gnode = self.node_items[id]
            node = self.g.get_node(id)
            if("pos_x" in attribute_names or "pos_y" in attribute_names):
                x = float(node.attrs["pos_x"].value)
                y = float(node.attrs["pos_y"].value)
                gnode.setPos(x,y)
                arriving_edges = self.g.get_edges_to_id(id)
                for edge in arriving_edges:
                    key = (edge.origin, edge.destination, edge.type)
                    item = self.node_items[edge.origin].remove_edge(key, None)
                    if item:
                        self.scene.removeItem(item)
                        self._add_edge(edge.origin, edge.destination, edge.type)
                        #self.update_edge_slot(edge.origin, edge.destination, edge.type

    def update_edge_slot(self, fr: int, to: int, type: str):
        key = (fr, to, type)
        if not key in self.g.get_node(fr).edges.keys():
            self._add_edge(fr, to, type)

    def update_edge_attrs_slot(self, fr: int, to: int, type: str, attribute_names: [str]):
        if fr in self.node_items.keys():
            self.node_items[fr].update_edge(to, type, attribute_names)

    def delete_node_slot(self, id: int):
        item = self.node_items.pop(id, None)
        if item:
            self.scene.removeItem(item)

    def delete_edge_slot(self, fr: int, to: int, type: str):
        print("delete_edge_slot in graph_viewer")
        key = (fr, to, type)
        if fr in self.node_items.keys():
            item = self.node_items[fr].remove_edge(key, None)
            if item:
                self.scene.removeItem(item)

