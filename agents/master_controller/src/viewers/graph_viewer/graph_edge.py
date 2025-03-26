from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsTextItem, QMenu, QGraphicsItem
from PySide6.QtGui import QPen, QColor, QAction
from PySide6.QtCore import Qt

class GraphicsEdge(QGraphicsLineItem):
    def __init__(self, source_node_id: int, target_node_id: int, type: str, graph_ref: object):
        super().__init__()
        self.G = graph_ref
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.source_node = self.G.get_node(source_node_id)
        self.target_node = self.G.get_node(target_node_id)
        self.type = type
        self.key = (source_node_id, target_node_id, type)

        self.setZValue(5)
        self.setPen(QPen(QColor("gray"), 2))

        # Text label
        self.label = QGraphicsTextItem(type, self)
        self.label.setDefaultTextColor(QColor("black"))
        self.update_label_position()
        self.update_position()

    def update_position(self, target_item=None):
        x1 = self.G.get_node(self.source_node_id).attrs["pos_x"].value
        y1 = self.G.get_node(self.source_node_id).attrs["pos_y"].value
        x2 = self.G.get_node(self.target_node_id).attrs["pos_x"].value
        y2 = self.G.get_node(self.target_node_id).attrs["pos_y"].value
        self.setLine(x1, y1, x2, y2)
        self.update_label_position()

    def update_label_position(self):
        mid_x = (self.line().x1() + self.line().x2()) / 2
        mid_y = (self.line().y1() + self.line().y2()) / 2
        self.label.setPos(mid_x, mid_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            menu = QMenu()
            edge = self.G.get_edge(self.source_node_id, self.target_node_id, self.type)
            if edge:
                for attr_name, attr in edge.attrs.items():
                    action = QAction(f"{attr_name}: {attr.value}", menu)
                    action.setEnabled(False)
                    menu.addAction(action)
            menu.exec(event.screenPos())
