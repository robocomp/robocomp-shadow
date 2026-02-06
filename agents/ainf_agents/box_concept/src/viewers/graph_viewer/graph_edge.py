#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphics Edge for DSR graph visualization."""

import math
from PySide6.QtCore import Qt, QObject, QPointF, QRectF, QSizeF
from PySide6.QtGui import QPainter, QColor, QPainterPath, QPolygonF, QKeyEvent
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsTextItem, QGraphicsItem, QWidget

from .edge_colors import edge_colors


class GraphicsEdge(QObject, QGraphicsLineItem):
    def __init__(self, source_node, destination_node, edge_name: str):
        super().__init__()
        QGraphicsLineItem.__init__(self)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.arrow_size = 6
        self.color = QColor(edge_colors.get(edge_name, "coral"))
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
        self.setAcceptHoverEvents(True)
        self.source_node = source_node
        self.destination_node = destination_node
        self.source_point = QPointF()
        self.dest_point = QPointF()
        source_node.add_edge(self)
        if source_node.id_in_graph != destination_node.id_in_graph:
            destination_node.add_edge(self)
        self.adjust()

    def adjust(self, node=None, pos: QPointF = None):
        if self.source_node is None or self.destination_node is None:
            return

        from PySide6.QtCore import QLineF
        line = QLineF(self.mapFromItem(self.source_node, 0, 0),
                      self.mapFromItem(self.destination_node, 0, 0))

        length = line.length()
        self.prepareGeometryChange()

        node_radius = 10
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

    def paint(self, painter: QPainter, option, widget: QWidget = None):
        painter.save()
        if self.source_node is None or self.destination_node is None:
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
        painter.setBrush(self.color)
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

            from PySide6.QtCore import QLineF
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
        else:
            painter.setBrush(Qt.NoBrush)
            default_diameter = 20
            default_radius = default_diameter * 2
            painter.drawEllipse(int(self.line().p1().x() - default_radius * 2), int(self.line().p1().y()), default_radius * 2, default_radius * 2)

    def set_bend_factor(self, factor: float):
        self.m_bend_factor = factor
