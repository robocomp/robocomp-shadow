from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QMenu, QGraphicsItem, QStyle, QStyleOptionGraphicsItem, QMessageBox, QGraphicsSceneMouseEvent, \
    QGraphicsSimpleTextItem
from PySide6.QtGui import QColor, QBrush, QPen, QAction, QPainterPath, QRadialGradient, QPainter, QMouseEvent
from PySide6.QtCore import Qt, QObject, QRectF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QMenu
from PySide6.QtGui import QColor, QBrush, QPen, QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsItem
from pydsr import signals
if TYPE_CHECKING:
    # Import node_colorts only for type checking to avoid import cycles.
    from src.viewers.graph_viewer.graph_edge import GraphicsEdge
from .node_colors import node_colors
from .graph_node_widget import GraphNodeWidget

class GraphicsNode(QObject, QGraphicsEllipseItem):
    def __init__(self, graph_viewer: 'GraphViewer'):
        super().__init__()
        QGraphicsEllipseItem.__init__(self, 0, 0, 20, 20)
        self.node_widget = None
        self.graph_viewer = graph_viewer
        self.default_diameter = 20
        self.default_radius = int(self.default_diameter / 2)
        self.sunken_color = Qt.darkGray
        self.edge_list = [] # GraphEdge*
        self.id_in_graph = -1
        flags = (QGraphicsItem.ItemIsMovable |
                 QGraphicsItem.ItemIsSelectable |
                 QGraphicsItem.ItemSendsGeometryChanges |
                 QGraphicsItem.ItemUsesExtendedStyleOption |
                 QGraphicsItem.ItemIsFocusable)
        self.setFlags(flags)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setAcceptHoverEvents(True)
        self.setZValue(-1)
        self.node_brush = QBrush()
        self.node_brush.setStyle(Qt.SolidPattern)

        # context menu
        self.contextMenu = QMenu()
        table_action = QAction("View table", self.contextMenu)
        self.contextMenu.addAction(table_action)
        table_action.triggered.connect(lambda: self.show_node_widget("table"))

    def set_tag(self, tag: str):
        tag = QGraphicsSimpleTextItem(tag, self)
        tag.setX(10)
        tag.setY(-10)

    def set_type(self, mtype: str):
        if mtype in node_colors.keys():
            color_name = node_colors[mtype]
            self.set_node_color(QColor(color_name))
        else:
            self.set_node_color(QColor("coral"))

    def set_node_color(self, color: QColor):
        self.node_brush.setColor(color)
        self.setBrush(self.node_brush)

    ######################################################################
    def add_edge(self, edge: GraphicsEdge):
        same_count = 0
        bend_factor = 0
        for old_edge in self.edge_list:
            if old_edge == edge:
                raise ValueError("Trying to add an already existing edge " + str(edge.source_node.id_in_graph + "--" + edge.destination_node.id_in_graph))
            if edge.source_node.id_in_graph == old_edge.source_node.id_in_graph or \
               edge.source_node.id_in_graph == old_edge.destination_node.id_in_graph and \
               edge.destination_node.id_in_graph == old_edge.source_node.id_in_graph or \
               edge.destination_node.id_in_graph == old_edge.destination_node.id_in_graph:
                same_count += 1

        bend_factor += (pow(-1,same_count)*(-1 + pow(-1,same_count) - 2*same_count))/4
        edge.set_bend_factor(bend_factor)
        self.edge_list.append(edge)
        edge.adjust()

    def delete_edge(self, edge: GraphicsEdge):
        try:
            self.edge_list.remove(edge)
        except ValueError:
            print("Trying to delete an edge that does not exist " + str(edge.source_node.id_in_graph + "--" + edge.destination_node.id_in_graph))

    def edges(self):
        return self.edge_list

    def boundingRect(self):
        adjust = 2.0
        return QRectF(-self.default_radius - adjust, -self.default_radius - adjust, self.default_diameter + 3 + adjust, self.default_diameter + 3 + adjust)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(-self.default_radius, -self.default_radius, self.default_diameter, self.default_diameter)
        return path

    ############# EVENTS ################################################
    def paint(self, painter : QPainter, option: QStyleOptionGraphicsItem, widget=None):
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.sunken_color)
        gradient = QRadialGradient(-3, -3, 10)
        if option.state & QStyle.State_Sunken:
            gradient.setColorAt(0, QColor(Qt.darkGray).lighter())
            gradient.setColorAt(1, QColor(Qt.darkGray))
        else:
            gradient.setColorAt(0, self.node_brush.color())
            gradient.setColorAt(1, QColor(self.node_brush.color().darker()))
        painter.setBrush(gradient)
        if self.isSelected():
            painter.setPen(QPen(Qt.green, 0, Qt.DashLine))
        else:
            painter.setPen(QPen(Qt.black, 0, Qt.SolidLine))
        painter.drawEllipse(-self.default_radius, -self.default_radius, self.default_diameter, self.default_diameter)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.edge_list:
                edge.adjust(self, value)
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.RightButton:
            self.contextMenu.exec(event.screenPos())
        super().mouseDoubleClickEvent(event)

    def show_node_widget(self, show_type: str):
        self.node_widget = GraphNodeWidget(self.graph_viewer.g, self.id_in_graph)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.LeftButton:
            g = self.graph_viewer.g
            print(f"{__file__}:{__name__} node id in graphnode: {self.id_in_graph}")
            n = g.get_node(self.id_in_graph)
            if n:
                n.attrs["pos_x"].value = float(self.pos().x())
                n.attrs["pos_y"].value = float(self.pos().y())
                g.update_node(n)
        QGraphicsEllipseItem.mouseReleaseEvent(self, event)

    ######################################################################
    ### SLOTS from G
    ######################################################################
    def delete_node(self):
        # print(f"GraphNode::Delete node {self.id_in_graph}")
        # # show confirmation dialog
        # msgBox = QMessageBox()
        # msgBox.setText("Are you sure you want to delete node?")
        # msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # msgBox.setDefaultButton(QMessageBox.No)
        # reply = msgBox.exec()
        # if reply == QMessageBox.Yes:
        #     self.del_node_signal.emit(self.id_in_graph)
        print("GraphNode::delete_node not implemented")



