import math
from PySide6 import QtCore
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
        # add nodes first
        for node in self.g.get_nodes():
            self._add_node(node)
        for node in self.g.get_nodes():
            for edge in node.edges:
                self._add_edge(node.id, edge[0], edge[1])

    def _add_node(self, node):
        circle = GraphicsNode(node, self.g)
        self.scene.addItem(circle)
        self.node_items[node.id] = circle

    def _add_edge(self, fr, to, mtype):
        existing_edges = [
            key for key in self.node_items[fr].connected_edges.keys()
            if key[0] == fr and key[1] == to
        ]
        offset_index = len(existing_edges)

        graphics_edge = GraphicsEdge(fr, to, mtype, self.g, offset_index)
        self.scene.addItem(graphics_edge)

        self.node_items[fr].connected_edges[graphics_edge.key] = graphics_edge
        self.node_items[to].connected_edges[graphics_edge.key] = graphics_edge

    def _node_context_menu(self, event, node):
        menu = QMenu()
        for attr_name, attr in node.attrs.items():
            action = QAction(f"{attr_name}: {attr.value}", menu)
            action.setEnabled(False)
            menu.addAction(action)
        menu.exec(event.screenPos())

    ########################################################################
    @QtCore.Slot()
    def update_node_slot(self, id: int, mtype: str):
        if not id in self.node_items.keys():
            self._add_node(self.g.get_node(id))


    def update_node_attrs_slot(self, id: int, attribute_names: [str]):
        if id in self.node_items.keys():
            gnode = self.node_items[id]
            node = self.g.get_node(id)
            if "pos_x" in attribute_names or "pos_y" in attribute_names:
                x = float(node.attrs["pos_x"].value)
                y = float(node.attrs["pos_y"].value)
                gnode.setPos(x, y)
                for edge in gnode.connected_edges.values():
                    edge.update_position()

    def update_edge_slot(self, fr: int, to: int, mtype: str):
        if not fr in self.node_items.keys():
            print("Adding node in update_edge_slot")
            self._add_edge(fr, to, mtype)

    def update_edge_attrs_slot(self, fr: int, to: int, mtype: str, attribute_names: [str]):
        if fr in self.node_items.keys():
            self.node_items[fr].update_edge(to, mtype, attribute_names)

    def delete_node_slot(self, id: int):
        item = self.node_items.pop(id, None)
        if item:
            self.scene.removeItem(item)

    def delete_edge_slot(self, fr: int, to: int, mtype: str):
        print("delete_edge_slot in graph_viewer")
        key = (fr, to, mtype)
        if fr in self.node_items:
            item = self.node_items[fr].remove_edge(key, None)
            if item:
                self.scene.removeItem(item)
        if to in self.node_items:
            self.node_items[to].remove_edge(key, None)


