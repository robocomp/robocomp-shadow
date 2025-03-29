from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QMenu, QGraphicsItem
from PySide6.QtGui import QColor, QBrush, QPen, QAction
from PySide6.QtCore import Qt

from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QMenu
from PySide6.QtGui import QColor, QBrush, QPen, QAction
from PySide6.QtCore import Qt
from taichi.examples.simulation.vortex_rings import new_pos


class GraphicsNode(QGraphicsEllipseItem):
    def __init__(self, node, graph_ref, radius=10):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.node = node
        self.graph = graph_ref
        self.radius = radius
        self.connected_edges = {}  # 🔗 Keep references to connected edges
        self._being_dragged = False

        self.setZValue(10)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)

        # Set position
        x = float(node.attrs["pos_x"].value)
        y = float(node.attrs["pos_y"].value)
        self.setPos(x, y)

        # Color
        if node.attrs.__contains__("color"):
            color = node.attrs["color"].value
        else:
            color = "lightblue"
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("black"), 2))

        # Label
        self.label = QGraphicsTextItem(node.name, self)
        self.label.setDefaultTextColor(Qt.black)
        self.label.setPos(radius, -radius)

        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

    def remove_edge(self, key, default=None):
        if key in self.connected_edges:
            cp = self.connected_edges[key]
            del self.connected_edges[key]
            return cp
        return default

    def update_edge(self, to: int, mtype: str, attribute_names: [str]):
        key = (self.node.id, to, mtype)
        if key in self.connected_edges:
            self.connected_edges[key].update_edge(attribute_names)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # User is dragging → update visuals
            for edge in self.connected_edges.values():
                edge.update_position()
            self._being_dragged = True  # we're being dragged
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        menu = QMenu()
        for attr_name, attr in self.node.attrs.items():
            action = QAction(f"{attr_name}: {attr.value}", menu)
            action.setEnabled(False)
            menu.addAction(action)
        action = QAction(f"Edges: {len(self.node.edges)}", menu)
        action.setEnabled(False)
        menu.addAction(action)
        menu.exec(event.screenPos())

    def mouseReleaseEvent(self, event):
        if self._being_dragged:
            self._being_dragged = False
            new_pos = self.pos()
            self.node.attrs["pos_x"].value = new_pos.x()
            self.node.attrs["pos_y"].value = new_pos.y()
            self.graph.update_node(self.node)
            # redraw edges
            for edge in self.connected_edges.values():
                edge.update_position()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        # Ensure edges are updated during dragging
        for edge in self.connected_edges.values():
            edge.update_position()
        self.scene().update()  # force scene redraw during dragging
        super().mouseMoveEvent(event)
