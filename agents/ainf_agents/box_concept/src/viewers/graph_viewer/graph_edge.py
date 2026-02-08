#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphics Edge for DSR graph visualization."""

import math
from PySide6.QtCore import Qt, QObject, QPointF, QRectF, QSizeF, QTimer
from PySide6.QtGui import QPainter, QColor, QPainterPath, QPolygonF, QKeyEvent
from PySide6.QtWidgets import (QGraphicsLineItem, QGraphicsTextItem, QGraphicsItem, QWidget,
                                QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                                QHeaderView, QLabel, QPushButton)

from .edge_colors import edge_colors


class EdgeAttributesDialog(QDialog):
    """Dialog to display edge attributes with live refresh."""

    def __init__(self, edge_type: str, from_id: int, to_id: int,
                 from_name: str, to_name: str, graph, parent=None):
        super().__init__(parent)
        self.edge_type = edge_type
        self.from_id = from_id
        self.to_id = to_id
        self.graph = graph
        self.setWindowTitle(f"Edge: {edge_type}")
        self.setMinimumSize(550, 450)
        self.setup_ui(edge_type, from_name, to_name)

        # Timer for live refresh (every 500ms)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_attributes)
        self.refresh_timer.start(500)

    def setup_ui(self, edge_type: str, from_name: str, to_name: str):
        layout = QVBoxLayout(self)

        # Header info
        header_layout = QVBoxLayout()
        type_label = QLabel(f"<b>Edge Type:</b> {edge_type}")
        type_label.setStyleSheet("font-size: 12pt; padding: 3px;")
        connection_label = QLabel(f"<b>From:</b> {from_name} (id={self.from_id})  →  <b>To:</b> {to_name} (id={self.to_id})")
        connection_label.setStyleSheet("font-size: 11pt; padding: 3px;")
        header_layout.addWidget(type_label)
        header_layout.addWidget(connection_label)
        layout.addLayout(header_layout)

        # Status label for live updates
        self.status_label = QLabel("🔄 Live refresh active")
        self.status_label.setStyleSheet("color: green; font-size: 10pt; padding: 2px;")
        layout.addWidget(self.status_label)

        # Attributes table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Attribute", "Type", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                font-size: 11pt;
            }
            QHeaderView::section {
                background-color: #d94a4a;
                color: white;
                padding: 5px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.table)

        # Initial population
        self.refresh_attributes()

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

    def refresh_attributes(self):
        """Refresh attribute values from the graph."""
        try:
            edge = self.graph.get_edge(self.from_id, self.to_id, self.edge_type)
            if edge is None:
                self.status_label.setText("⚠️ Edge no longer exists")
                self.status_label.setStyleSheet("color: red; font-size: 10pt; padding: 2px;")
                self.refresh_timer.stop()
                return

            attributes = edge.attrs

            # Remember scroll position
            scroll_pos = self.table.verticalScrollBar().value()

            # Update table
            self.table.setRowCount(len(attributes))
            for row, (attr_name, attr) in enumerate(sorted(attributes.items())):
                # Attribute name
                name_item = self.table.item(row, 0)
                if name_item is None:
                    name_item = QTableWidgetItem(attr_name)
                    name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 0, name_item)
                else:
                    name_item.setText(attr_name)

                # Attribute type
                try:
                    attr_type = type(attr.value).__name__
                except:
                    attr_type = "unknown"
                type_item = self.table.item(row, 1)
                if type_item is None:
                    type_item = QTableWidgetItem(attr_type)
                    type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 1, type_item)
                else:
                    type_item.setText(attr_type)

                # Attribute value
                try:
                    value = attr.value
                    if hasattr(value, '__iter__') and not isinstance(value, str):
                        if hasattr(value, 'tolist'):
                            value = value.tolist()
                        # Format floats nicely
                        if isinstance(value, (list, tuple)) and len(value) > 0:
                            if isinstance(value[0], float):
                                value_str = "[" + ", ".join(f"{v:.4f}" for v in value) + "]"
                            else:
                                value_str = str(value)
                        else:
                            value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                    elif isinstance(value, float):
                        value_str = f"{value:.6f}"
                    else:
                        value_str = str(value)
                except Exception as e:
                    value_str = f"<error: {e}>"

                value_item = self.table.item(row, 2)
                if value_item is None:
                    value_item = QTableWidgetItem(value_str)
                    value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 2, value_item)
                else:
                    # Highlight changed values
                    if value_item.text() != value_str:
                        value_item.setBackground(QColor(255, 255, 200))  # Light yellow
                    else:
                        value_item.setBackground(QColor(255, 255, 255))  # White
                    value_item.setText(value_str)

            # Restore scroll position
            self.table.verticalScrollBar().setValue(scroll_pos)

        except Exception as e:
            self.status_label.setText(f"⚠️ Error: {e}")
            self.status_label.setStyleSheet("color: red; font-size: 10pt; padding: 2px;")

    def closeEvent(self, event):
        """Stop timer when dialog closes."""
        self.refresh_timer.stop()
        super().closeEvent(event)


class EdgeTextItem(QGraphicsTextItem):
    """Custom text item for edge labels that handles right-click to show attributes."""

    def __init__(self, text: str, parent_edge):
        super().__init__(text, parent_edge)
        self.parent_edge = parent_edge
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

    def mousePressEvent(self, event):
        """Handle mouse press - right click shows edge attributes dialog."""
        if event.button() == Qt.RightButton:
            self.parent_edge.show_attributes_dialog()
            event.accept()
        else:
            super().mousePressEvent(event)


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
        self.edge_name = edge_name  # Store edge type name
        self.setZValue(-1)
        flags = (QGraphicsItem.ItemIsSelectable |
                 QGraphicsItem.ItemSendsGeometryChanges |
                 QGraphicsItem.ItemUsesExtendedStyleOption)
        self.setFlags(flags)

        # Use custom EdgeTextItem that handles right-click
        self.tag = EdgeTextItem(edge_name, self)
        # Set font size to triple (default is ~10pt, now 30pt)
        font = self.tag.font()
        font.setPointSize(30)
        self.tag.setFont(font)
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
            default_diameter = 40
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
            default_diameter = 40
            default_radius = default_diameter * 2
            painter.drawEllipse(int(self.line().p1().x() - default_radius * 2), int(self.line().p1().y()), default_radius * 2, default_radius * 2)

    def set_bend_factor(self, factor: float):
        self.m_bend_factor = factor

    def mousePressEvent(self, event):
        """Handle mouse press - right click shows attributes dialog."""
        if event.button() == Qt.RightButton:
            self.show_attributes_dialog()
            event.accept()
        else:
            super().mousePressEvent(event)

    def show_attributes_dialog(self):
        """Show a dialog with all edge attributes that refreshes live."""
        # Get graph from source node's graph_viewer
        if self.source_node is None or not hasattr(self.source_node, 'graph_viewer'):
            return

        graph = self.source_node.graph_viewer.g
        from_id = self.source_node.id_in_graph
        to_id = self.destination_node.id_in_graph

        # Get node names for display
        from_node = graph.get_node(from_id)
        to_node = graph.get_node(to_id)
        from_name = from_node.name if from_node else str(from_id)
        to_name = to_node.name if to_node else str(to_id)

        # Create and show dialog
        dialog = EdgeAttributesDialog(
            edge_type=self.edge_name,
            from_id=from_id,
            to_id=to_id,
            from_name=from_name,
            to_name=to_name,
            graph=graph
        )
        dialog.exec()

