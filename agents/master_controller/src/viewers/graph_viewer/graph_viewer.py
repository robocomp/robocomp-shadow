from PySide6.QtCore import Signal, QPointF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QMenu
from PySide6.QtGui import QColor, QPen, QBrush, QAction
from viewers._abstract_graphic_view import AbstractGraphicViewer

class GraphViewer(AbstractGraphicViewer):
    def __init__(self, G):
        super().__init__()
        self.G = G
        self.node_items = {}  # node_id: QGraphicsEllipseItem
        self.edge_items = {}  # (from, to, type): QGraphicsLineItem
        self.scale(1, -1)
        self.draw_entire_graph()

    def draw_entire_graph(self):
        for node in self.G.get_node:
            self._add_node(node)
        for edge in self.G.get_edges():
            self._add_edge(edge)

    def _add_node(self, node):
        x = float(node.attrs.get("pos_x", 0).value)
        y = float(node.attrs.get("pos_y", 0).value)
        r = 30
        circle = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        circle.setBrush(QBrush(QColor("skyblue")))
        circle.setPen(QPen(QColor("black"), 2))
        circle.setZValue(10)
        circle.setToolTip(f"{node.type} ({node.id})")
        self.scene.addItem(circle)
        self.node_items[node.id] = circle

        label = QGraphicsTextItem(node.type)
        label.setDefaultTextColor(QColor("black"))
        label.setPos(x - r, y - r - 25)
        self.scene.addItem(label)

        # Attach right-click context menu
        circle.setFlag(circle.ItemIsSelectable)
        circle.setAcceptedMouseButtons(Qt.RightButton)
        circle.contextMenuEvent = lambda e, n=node: self._node_context_menu(e, n)

    def _add_edge(self, edge):
        source = self.G.get_node(edge.src)
        target = self.G.get_node(edge.dst)
        if not source or not target:
            return
        x1 = float(source.attrs.get("pos_x", 0).value)
        y1 = float(source.attrs.get("pos_y", 0).value)
        x2 = float(target.attrs.get("pos_x", 0).value)
        y2 = float(target.attrs.get("pos_y", 0).value)
        line = QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(QPen(QColor("gray"), 2))
        line.setZValue(5)
        line.setToolTip(f"{edge.type} ({edge.src} → {edge.dst})")
        self.scene.addItem(line)
        self.edge_items[(edge.src, edge.dst, edge.type)] = line

    def _node_context_menu(self, event, node):
        menu = QMenu()
        for attr_name, attr in node.attrs.items():
            action = QAction(f"{attr_name}: {attr.value}", menu)
            action.setEnabled(False)
            menu.addAction(action)
        menu.exec(event.screenPos())

    def add_or_assign_node_slot(self, node_id, node):
        self.del_node_slot(node_id)
        self._add_node(node)

    def add_or_assign_edge_slot(self, from_node, to_node, edge_type):
        edge = self.G.get_edge(from_node, to_node, edge_type)
        if edge:
            self.del_edge_slot(from_node, to_node, edge_type)
            self._add_edge(edge)

    def del_node_slot(self, node_id):
        item = self.node_items.pop(node_id, None)
        if item:
            self.scene.removeItem(item)

    def del_edge_slot(self, from_node, to_node, edge_tag):
        key = (from_node, to_node, edge_tag)
        item = self.edge_items.pop(key, None)
        if item:
            self.scene.removeItem(item)

    def node_change_slot(self, value, node_id, node_type, parent=None):
        node = self.G.get_node(node_id)
        if node:
            self.add_or_assign_node_slot(node_id, node)

    def category_change_slot(self, value, parent=None):
        pass  # Optional