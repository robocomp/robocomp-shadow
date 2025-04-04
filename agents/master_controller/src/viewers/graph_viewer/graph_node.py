from PyQt5.QtWidgets import QGraphicsSimpleTextItem, QGraphicsSceneMouseEvent
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QMenu, QGraphicsItem, QStyle, QStyleOptionGraphicsItem
from PySide6.QtGui import QColor, QBrush, QPen, QAction, QPainterPath, QRadialGradient, QPainter, QMouseEvent
from PySide6.QtCore import Qt, QObject, QRectF

from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QMenu
from PySide6.QtGui import QColor, QBrush, QPen, QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsItem
from .graph_edge import GraphicsEdge

class GraphicsNode(QObject, QGraphicsEllipseItem):
    def __init__(self, graph_viewer: 'GraphViewer'):
        super().__init__()
        QGraphicsEllipseItem.__init__(self, 0, 0, 20, 20)
        self.graph_viewer = graph_viewer
        self.default_diameter = 20
        self.default_radius = self.default_diameter / 2
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
        # contextMenu = QMenu()
        # table_action = QAction("View table")
        # contextMenu.addAction(table_action)
        # table_action.triggered.connect(lambda: self.show_node_widget("table"))

        # self.node = node
        # #self.graph = graph_ref
        # self.radius = radius
        # self.connected_edges = {}  # 🔗 Keep references to connected edges
        # self._being_dragged = False
        #
        # self.setZValue(10)
        # self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        # self.setAcceptHoverEvents(True)
        #
        # # Set position
        # x = float(node.attrs["pos_x"].value)
        # y = float(node.attrs["pos_y"].value)
        # self.setPos(x, y)
        #
        # # Color
        # if node.attrs.__contains__("color"):
        #     color = node.attrs["color"].value
        # else:
        #     color = "lightblue"
        # self.setBrush(QBrush(QColor(color)))
        # self.setPen(QPen(QColor("black"), 2))
        #
        # # Label
        # self.label = QGraphicsTextItem(node.name, self)
        # self.label.setDefaultTextColor(Qt.black)
        # self.label.setPos(radius, -radius)
        #
        # self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

    def set_tag(self, tag: str):
        tag = QGraphicsSimpleTextItem(tag, self)
        tag.setX(10)
        tag.setY(-10)

    def set_type(self, mtype: str):
        #color_name = GraphColors()[type]
        color_name = "green" # TODO: get color from GraphColors
        self.set_node_color(QColor(color_name))
        # TODO: connect to show_node_widget

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

    def delete_edge(self, edge: GraphicsEdge):
        try:
            self.edge_list.remove(edge)
            edge.delete()
        except ValueError:
            print("Trying to delete an edge that does not exist " + str(edge.source_node.id_in_graph + "--" + edge.destination_node.id_in_graph))

    def edges(self):
        return self.edge_list

    ######################################################################
    def boundingRect(self):
        adjust = 2.0
        return QRectF(-self.default_radius - adjust, -self.default_radius - adjust, self.default_diameter + 3 + adjust, self.default_diameter + 3 + adjust)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(-self.default_radius, -self.default_radius, self.default_diameter, self.default_diameter)
        return path

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

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        # if event.button() == Qt.RightButton:
        #     self.contextMenu.exec(event.screenPos())
        QGraphicsEllipseItem.mouseDoubleClickEvent(event)

    # def show_node_widget(self, show_type: str):     # FOR type selection
    #     node_widget = GraphNodeWidget(self.graph_viewer.g, self.id_in_graph)

    ######################################################################

    # def remove_edge(self, key, default=None):
    #     if key in self.connected_edges:
    #         cp = self.connected_edges[key]
    #         del self.connected_edges[key]
    #         return cp
    #     return default
    #
    # def update_edge(self, to: int, mtype: str, attribute_names: [str]):
    #     key = (self.node.id, to, mtype)
    #     if key in self.connected_edges:
    #         self.connected_edges[key].update_edge(attribute_names)
    #
    # def itemChange(self, change, value):
    #     if change == QGraphicsItem.ItemPositionChange:
    #         # User is dragging → update visuals
    #         for edge in self.connected_edges.values():
    #             edge.update_position()
    #         self._being_dragged = True  # we're being dragged
    #     return super().itemChange(change, value)
    #
    # def mouseDoubleClickEvent(self, event):
    #     menu = QMenu()
    #     for attr_name, attr in self.node.attrs.items():
    #         action = QAction(f"{attr_name}: {attr.value}", menu)
    #         action.setEnabled(False)
    #         menu.addAction(action)
    #     action = QAction(f"Edges: {len(self.node.edges)}", menu)
    #     action.setEnabled(False)
    #     menu.addAction(action)
    #     menu.exec(event.screenPos())
    #
    # def mouseReleaseEvent(self, event):
    #     if self._being_dragged:
    #         self._being_dragged = False
    #         new_pos = self.pos()
    #         self.node.attrs["pos_x"].value = new_pos.x()
    #         self.node.attrs["pos_y"].value = new_pos.y()
    #         self.graph.update_node(self.node)
    #         # redraw edges
    #         for edge in self.connected_edges.values():
    #             edge.update_position()
    #     super().mouseReleaseEvent(event)
    #
    # def mouseMoveEvent(self, event):
    #     # Ensure edges are updated during dragging
    #     for edge in self.connected_edges.values():
    #         edge.update_position()
    #     self.scene().update()  # force scene redraw during dragging
    #     super().mouseMoveEvent(event)
