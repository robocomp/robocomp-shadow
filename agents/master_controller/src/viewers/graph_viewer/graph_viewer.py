from PySide6 import QtCore
from PySide6.QtCore import QMutex
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QMenu
from PySide6.QtGui import QColor, QPen, QBrush, QAction, QTransform
from viewers._abstract_graphic_view import AbstractGraphicViewer
from .graph_edge import GraphicsEdge
from .graph_node import GraphicsNode


class GraphViewer(AbstractGraphicViewer):
    def __init__(self, G):
        super().__init__()
        self.G = G
        self.node_items = {}  # node_id: QGraphicsEllipseItem
        #self.edge_items = {}  # (from, to, type): QGraphicsLineItem
        #self.scale(1, -1)
        self.mutex = QMutex()
        self.draw_entire_graph()



    def draw_entire_graph(self):
        for node in self.G.get_nodes():
            self._add_node(node)

    def _add_node(self, node):
        circle = GraphicsNode(node, self.G)
        self.scene.addItem(circle)
        self.node_items[node.id] = circle

        for edge in node.edges:
            self._add_edge(node.id, edge[0], edge[1])

    def _add_edge(self, source_id, destination_id, edge_type):
        line = GraphicsEdge(source_id, destination_id, edge_type, self.G)
        self.scene.addItem(line)
        self.node_items[source_id].add_edge((source_id, destination_id, edge_type), line)
        #self.edge_items[(source_id, destination_id, edge_type)] = line

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
            self._add_node(self.G.get_node(id))

    def update_node_attrs_slot(self, id: int, attribute_names: [str]):
        if id in self.node_items.keys():
            gnode = self.node_items[id]
            node = self.G.get_node(id)
            if("pos_x" in attribute_names or "pos_y" in attribute_names):
                x = float(node.attrs["pos_x"].value)
                y = float(node.attrs["pos_y"].value)
                gnode.setPos(x,y)
                arriving_edges = self.G.get_edges_to_id(id)
                for edge in arriving_edges:
                    key = (edge.origin, edge.destination, edge.type)
                    item = self.node_items[edge.origin].remove_edge(key, None)
                    if item:
                        self.scene.removeItem(item)
                        self._add_edge(edge.origin, edge.destination, edge.type)
                        #self.update_edge_slot(edge.origin, edge.destination, edge.type

    def update_edge_slot(self, fr: int, to: int, type: str):
        key = (fr, to, type)
        if not key in self.node[fr].edges.keys():
            self._add_edge(fr, to, type)

    def update_edge_attrs_slot(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass

    def delete_node_slot(self, id: int):
        item = self.node_items.pop(id, None)
        if item:
            self.scene.removeItem(item)

    def delete_edge_slot(self, fr: int, to: int, type: str):
        key = (fr, to, type)
        item = self.node_items[fr].remove_edge(key, None)
        if item:
            self.scene.removeItem(item)
