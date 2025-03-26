from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsSceneContextMenuEvent, QMenu
from PySide6.QtGui import QPen, QBrush, QColor, QAction
from PySide6.QtCore import Qt
import math


class DSRGraphicsScene:
    def __init__(self, viewer, G):
        self.viewer = viewer
        self.G = G
        self.node_items = {}  # node_id: QGraphicsEllipseItem
        self.edge_items = {}  # (from, to): QGraphicsLineItem

        # Connect to signals
        from pydsr import signals
        signals.connect(self.G, signals.UPDATE_NODE, self.update_node)
        signals.connect(self.G, signals.UPDATE_EDGE, self.update_edge)
        signals.connect(self.G, signals.DELETE_NODE, self.delete_node)
        signals.connect(self.G, signals.DELETE_EDGE, self.delete_edge)

        self.draw_entire_graph()

    def draw_entire_graph(self):
        for node in self.G.get_nodes():
            self._add_node(node)
        for edge in self.G.get_edges():
            self._add_edge(edge)

    def _add_node(self, node):
        x = float(node.attrs.get("pos_x", 0).value)
        y = float(node.attrs.get("pos_y", 0).value)
        r = 30
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        ellipse.setBrush(QBrush(QColor("lightblue")))
        ellipse.setPen(QPen(QColor("black"), 2))
        ellipse.setFlag(ellipse.ItemIsSelectable)
        ellipse.setZValue(10)
        self.viewer.scene.addItem(ellipse)
        ellipse.setToolTip(f"{node.type} ({node.id})")
        ellipse.contextMenuEvent = lambda e, n=node: self.show_node_context(e, n)
        self.node_items[node.id] = ellipse

        label = QGraphicsTextItem(node.type)
        label.setDefaultTextColor(Qt.black)
        label.setPos(x - r, y - r - 20)
        self.viewer.scene.addItem(label)

    def _add_edge(self, edge):
        try:
            source = self.G.get_node(edge.src)
            target = self.G.get_node(edge.dst)
            if source and target:
                x1 = float(source.attrs.get("pos_x", 0).value)
                y1 = float(source.attrs.get("pos_y", 0).value)
                x2 = float(target.attrs.get("pos_x", 0).value)
                y2 = float(target.attrs.get("pos_y", 0).value)
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(QPen(Qt.darkGray, 2))
                line.setZValue(5)
                line.setToolTip(f"{edge.type} ({edge.src}->{edge.dst})")
                line.contextMenuEvent = lambda e, ed=edge: self.show_edge_context(e, ed)
                self.viewer.scene.addItem(line)
                self.edge_items[(edge.src, edge.dst)] = line
        except Exception as e:
            print(f"Error adding edge: {e}")

    # ==== SIGNAL SLOT METHODS ====

    def update_node(self, id, type):
        node = self.G.get_node(id)
        if node:
            self.delete_node(id)
            self._add_node(node)

    def update_edge(self, fr, to, type):
        edge = self.G.get_edge(fr, to, type)
        if edge:
            self.delete_edge(fr, to, type)
            self._add_edge(edge)

    def delete_node(self, id):
        if id in self.node_items:
            self.viewer.scene.removeItem(self.node_items[id])
            del self.node_items[id]

    def delete_edge(self, fr, to, type):
        if (fr, to) in self.edge_items:
            self.viewer.scene.removeItem(self.edge_items[(fr, to)])
            del self.edge_items[(fr, to)]

    # ==== CONTEXT MENU POPUPS ====

    def show_node_context(self, event: QGraphicsSceneContextMenuEvent, node):
        menu = QMenu()
        for attr, val in node.attrs.items():
            act = QAction(f"{attr}: {val.value}")
            act.setEnabled(False)
            menu.addAction(act)
        menu.exec(event.screenPos())

    def show_edge_context(self, event: QGraphicsSceneContextMenuEvent, edge):
        menu = QMenu()
        for attr, val in edge.attrs.items():
            act = QAction(f"{attr}: {val.value}")
            act.setEnabled(False)
            menu.addAction(act)
        menu.exec(event.screenPos())
