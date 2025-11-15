from __future__ import annotations
from typing import TYPE_CHECKING
import math
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsTextItem, QMenu, QGraphicsItem, QGraphicsPolygonItem, QWidget, QHBoxLayout, QLabel, QComboBox, QWidgetAction, \
    QGraphicsLineItem, QGraphicsSceneMouseEvent
from PySide6.QtGui import QPen, QColor, QAction, QPainterPath, QPolygonF, QTransform, QCursor, QPainter, QKeyEvent
from PySide6.QtCore import Qt, QPointF, Signal, QObject, QLineF, QRectF, QSizeF, QEvent
from PySide6.QtWidgets import QGraphicsTextItem, QMenu, QApplication
from pydsr import signals
if TYPE_CHECKING:
    # Import GraphicsNode only for type checking to avoid import cycles.
    from src.viewers.graph_viewer.graph_node import GraphicsNode
from viewers.graph_viewer.edge_colors import edge_colors
from .graph_edge_widget import GraphEdgeWidget

class GraphicsEdge(QObject, QGraphicsLineItem):
    def __init__(self, source_node: GraphicsNode, destination_node: GraphicsNode, edge_name: str):
        super().__init__()
        QGraphicsLineItem.__init__(self)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)  # TODO: NOT WORKING!!!
        self.arrow_size = 6
        self.color = edge_colors[edge_name] if edge_colors[edge_name] else "coral"
        self.label = None
        self.m_bend_factor = 0
        self.line_width = 2
        self.setZValue(-1)
        flags = (QGraphicsItem.ItemIsSelectable |
                 QGraphicsItem.ItemSendsGeometryChanges |
                 QGraphicsItem.ItemUsesExtendedStyleOption)
        self.setFlags(flags)
        self.tag = QGraphicsTextItem(edge_name, self)
        self.tag.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.tag.installEventFilter(self)
        self.setAcceptHoverEvents(True)
        self.source_node = source_node
        self.destination_node = destination_node
        source_node.add_edge(self)
        if source_node.id_in_graph != destination_node.id_in_graph:
            destination_node.add_edge(self)
        self.adjust()
        # Connect update signal to update node attributes
        #signals.connect(self.source_node.graph_viewer.g, signals.UPDATE_EDGE_ATTR, self.update_edge_attr_slot)

    def adjust(self, node: GraphicsEdge= None, pos: QPointF = None):
        if self.source_node is None or self.destination_node is None:
            return

        line = QLineF(self.mapFromItem(self.source_node, 0, 0),
                      self.mapFromItem(self.destination_node, 0, 0))

        length = line.length()
        self.prepareGeometryChange()

        node_radius = 10  # Matches C++ radius (20-diameter node implies radius=10)
        if length > 2 * node_radius:
            edge_offset = QPointF((line.dx() * node_radius) / length,
                                  (line.dy() * node_radius) / length)
            self.source_point = line.p1() + edge_offset
            self.dest_point = line.p2() - edge_offset
        else:
            self.source_point = self.dest_point = line.p1()

        self.setLine(QLineF(self.source_point, self.dest_point))

    def boundingRect(self) -> QRectF:
        if not self.source_node or not self.destination_node:
            return QRectF()

        extra = (1 + self.arrow_size) / 2.0
        if self.source_node != self.destination_node:
            return QRectF(self.line().p1(), QSizeF(self.line().p2().x() - self.line().p1().x(),
                                                   self.line().p2().y() - self.line().p1().y())).normalized().adjusted(-extra, -extra, extra, extra)
        else:
            default_diameter = 20
            default_radius = default_diameter * 2
            return QRectF(self.line().p1().x() - default_radius * 2, self.line().p1().y(), default_radius * 2, default_radius * 2)

    def paint(self, painter: QPainter, option: Qt.QStyleOptionsGraphicItem, widget: QWidget = None):
        painter.save()
        if self.source_node is None or self.destination_node is None:
            print("GraphicsEdge::paint: source or destination is None")
            return

        self.draw_arrows(painter)
        self.draw_arc(painter)
        painter.restore()
        if self.source_node == self.destination_node:
            self.setZValue(-10)

    def draw_arrows(self, painter: QPainter):
        angle = math.atan2(-self.line().dy(), self.line().dx())
        destArrowP1 = self.line().p2() + QPointF(math.sin(angle - math.pi / 3) * self.arrow_size,
                                                 math.cos(angle - math.pi / 3) * self.arrow_size)
        destArrowP2 = self.line().p2() + QPointF(math.sin(angle - math.pi + math.pi / 3) * self.arrow_size,
                                                 math.cos(angle - math.pi + math.pi / 3) * self.arrow_size)
        painter.setBrush(QColor(self.color))
        painter.setPen(self.color)
        painter.drawPolygon(QPolygonF([self.line().p2(), destArrowP1, destArrowP2]))

    def draw_arc(self, painter: QPainter):
        if self.source_node != self.destination_node:
            m_control_pos = QPointF((self.line().p1() + self.line().p2()) / 2.0)
            t1 = QPointF(m_control_pos)
            pos_factor = abs(self.m_bend_factor)

            bend_direction = True
            if self.m_bend_factor < 0:
                bend_direction = not bend_direction

            f1 = QLineF(t1, self.line().p2())
            f1.setAngle(f1.angle() + 90 if bend_direction else f1.angle() - 90)
            f1.setLength(f1.length() * 0.2 * pos_factor)
            m_control_pos = f1.p2()
            m_control_point = m_control_pos - (t1 - m_control_pos) * 0.33
            path = QPainterPath()
            path.moveTo(self.line().p1())
            path.cubicTo(m_control_point, m_control_point, self.line().p2())
            r = self.tag.boundingRect()
            w = r.width()
            h = r.height()
            self.tag.setDefaultTextColor(self.color)
            self.tag.setPos(m_control_point.x() - w / 2, m_control_point.y() - h / 2)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(self.color)
            painter.drawPath(path)
        else:   # Draw a circle
            painter.setBrush(Qt.NoBrush)
            default_diameter = 20
            default_radius = default_diameter * 2
            painter.drawEllipse(int(self.line().p1().x() - default_radius * 2), int(self.line().p1().y()), default_radius * 2, default_radius * 2)

    def set_bend_factor(self, factor: float):
        """Set the bend factor for the edge. Positive values curve up, negative values curve down."""
        self.m_bend_factor = factor

    ######### EVENTS ################################################
    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        print("GraphicsEdge::mouseDoubleClickEvent")
        if event.button() == Qt.RightButton:
            print(f"{__file__} {__name__} Edge from {self.source_node.id_in_graph} to {self.destination_node.id_in_graph} tag: {self.tag.toPlainText()}")
            self.do_stuff = None
            graph = self.source_node.graph_viewer.g
            if self.tag.toPlainText() in ["RT", "looking-at"]:
                #self.do_stuff = GraphEdgeRTWidget(graph, self.source_node.id_in_graph, self.destination_node.id_in_graph, self.tag.toPlainText())
                pass
            else:
                self.do_stuff = GraphEdgeWidget(graph, self.source_node.id_in_graph, self.destination_node.id_in_graph, self.tag.toPlainText())
            self.update()
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            if self.label is not None:
                self.label.close()
                del self.label
                self.label = None

    # def eventFilter(self, obj: QObject, event: QEvent) -> bool:
    #     if obj == self.tag:
    #         if event.type() == QEvent.GraphicsSceneMouseDoubleClick:
    #             mouse_event = event  # type: QGraphicsSceneMouseEvent
    #             if mouse_event.button() == Qt.RightButton:
    #                 self.mouse_double_clicked()
    #             return True
    #     return False

    ################### SLOTS ################################
    def update_edge_attr_slot(self, from_id: int, to_id: int, edge_type: str, att_name: list[str]):
        if from_id != self.source_node.id_in_graph or to_id != self.destination_node.id_in_graph:
            return
        if "color" in att_name:
            edge = self.source_node.graph_viewer.g.get_edge(from_id, to_id, self.tag.toPlainText())
            if edge:
                if edge.attrs.__contains__("color"):
                    self.color = QColor(edge.attrs["color"].value)

