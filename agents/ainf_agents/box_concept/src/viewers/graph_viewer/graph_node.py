#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphics Node for DSR graph visualization."""

import random
from PySide6.QtCore import Qt, QObject, QRectF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QPainterPath, QAction
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsSimpleTextItem, QGraphicsItem,
                                QGraphicsSceneMouseEvent, QMenu, QStyle)

from .node_colors import node_colors
from .graph_edge import GraphicsEdge


class GraphicsNode(QObject, QGraphicsEllipseItem):
    def __init__(self, graph_viewer):
        super().__init__()
        QGraphicsEllipseItem.__init__(self, 0, 0, 40, 40)
        self.node_widget = None
        self.graph_viewer = graph_viewer
        self.default_diameter = 40
        self.default_radius = int(self.default_diameter / 2)
        self.sunken_color = Qt.darkGray
        self.edge_list = []
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

        self.contextMenu = QMenu()
        table_action = QAction("View table", self.contextMenu)
        self.contextMenu.addAction(table_action)

    def set_tag(self, tag: str):
        text_item = QGraphicsSimpleTextItem(tag, self)
        # Set font size to triple (default is ~10pt, now 30pt)
        font = text_item.font()
        font.setPointSize(30)
        text_item.setFont(font)
        text_item.setX(25)
        text_item.setY(-15)

    def set_type(self, mtype: str):
        if mtype in node_colors.keys():
            color_name = node_colors[mtype]
            self.set_node_color(QColor(color_name))
        else:
            self.set_node_color(QColor("coral"))

    def set_node_color(self, color):
        if isinstance(color, str):
            color = QColor(color)
        self.node_brush.setColor(color)
        self.setBrush(self.node_brush)

    def add_edge(self, edge: GraphicsEdge):
        same_count = 0
        bend_factor = 0
        for old_edge in self.edge_list:
            if old_edge == edge:
                return
            if (edge.source_node.id_in_graph == old_edge.source_node.id_in_graph or
                edge.source_node.id_in_graph == old_edge.destination_node.id_in_graph) and \
               (edge.destination_node.id_in_graph == old_edge.source_node.id_in_graph or
                edge.destination_node.id_in_graph == old_edge.destination_node.id_in_graph):
                same_count += 1

        bend_factor += (pow(-1, same_count) * (-1 + pow(-1, same_count) - 2 * same_count)) / 4
        edge.set_bend_factor(bend_factor)
        self.edge_list.append(edge)
        edge.adjust()

    def delete_edge(self, edge: GraphicsEdge):
        try:
            self.edge_list.remove(edge)
        except ValueError:
            pass

    def edges(self):
        return self.edge_list

    def boundingRect(self):
        adjust = 2.0
        return QRectF(-self.default_radius - adjust, -self.default_radius - adjust,
                      self.default_diameter + 3 + adjust, self.default_diameter + 3 + adjust)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(-self.default_radius, -self.default_radius, self.default_diameter, self.default_diameter)
        return path

    def paint(self, painter: QPainter, option, widget=None):
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

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.LeftButton:
            g = self.graph_viewer.g
            n = g.get_node(self.id_in_graph)
            if n:
                n.attrs["pos_x"].value = float(self.pos().x())
                n.attrs["pos_y"].value = float(self.pos().y())
                g.update_node(n)
        QGraphicsEllipseItem.mouseReleaseEvent(self, event)
