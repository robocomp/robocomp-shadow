#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSR Graph Widget - Qt-based graph visualization for DSR

Displays the DSR graph as an interactive node-edge diagram using QGraphicsView.
"""

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPen, QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
                                QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem,
                                QGraphicsItem)
import math
from typing import Dict, Optional, Tuple


class NodeItem(QGraphicsEllipseItem):
    """A node in the graph visualization."""

    def __init__(self, node_id: int, node_name: str, node_type: str, x: float, y: float):
        self.node_radius = 25
        super().__init__(-self.node_radius, -self.node_radius,
                         self.node_radius * 2, self.node_radius * 2)

        self.node_id = node_id
        self.node_name = node_name
        self.node_type = node_type

        # Set position
        self.setPos(x, y)

        # Set appearance based on node type
        self.setColor(node_type)

        # Enable selection and movement
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        # Add label
        self.label = QGraphicsTextItem(node_name, self)
        self.label.setDefaultTextColor(Qt.white)
        font = QFont("Arial", 8)
        self.label.setFont(font)
        # Center the label
        label_rect = self.label.boundingRect()
        self.label.setPos(-label_rect.width() / 2, -label_rect.height() / 2)

    def setColor(self, node_type: str):
        """Set node color based on type."""
        colors = {
            'root': QColor(255, 100, 100),
            'world': QColor(200, 200, 100),
            'room': QColor(100, 200, 100),
            'robot': QColor(100, 100, 255),
            'shadow': QColor(100, 150, 255),
            'person': QColor(255, 200, 100),
            'object': QColor(200, 150, 255),
            'table': QColor(139, 90, 43),
            'chair': QColor(70, 130, 180),
        }
        color = colors.get(node_type, QColor(180, 180, 180))

        self.setBrush(QBrush(color))
        self.setPen(QPen(color.darker(150), 2))


class EdgeItem(QGraphicsLineItem):
    """An edge in the graph visualization."""

    def __init__(self, source: NodeItem, target: NodeItem, edge_type: str):
        super().__init__()

        self.source = source
        self.target = target
        self.edge_type = edge_type

        # Set appearance based on edge type
        colors = {
            'RT': QColor(100, 200, 100),
            'has': QColor(200, 200, 100),
            'in': QColor(100, 100, 200),
        }
        color = colors.get(edge_type, QColor(150, 150, 150))

        self.setPen(QPen(color, 2))
        self.setZValue(-1)  # Draw behind nodes

        self.updatePosition()

    def updatePosition(self):
        """Update line position based on node positions."""
        if self.source and self.target:
            self.setLine(
                self.source.pos().x(), self.source.pos().y(),
                self.target.pos().x(), self.target.pos().y()
            )


class DSRGraphWidget(QWidget):
    """
    Qt widget for visualizing DSR graph.

    Uses QGraphicsView for smooth panning and zooming.
    """

    def __init__(self, g, parent=None, update_interval_ms: int = 1000):
        super().__init__(parent)
        self.g = g

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create graphics view and scene
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setBackgroundBrush(QBrush(QColor(40, 40, 50)))

        layout.addWidget(self.view)

        # Node and edge tracking
        self.node_items: Dict[int, NodeItem] = {}
        self.edge_items: Dict[Tuple[int, int, str], EdgeItem] = {}

        # Layout parameters
        self.layout_radius = 150
        self.center_x = 0
        self.center_y = 0

        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.updateGraph)
        self.update_timer.start(update_interval_ms)

        # Initial update
        self.updateGraph()

    def updateGraph(self):
        """Update the graph visualization from DSR."""
        if self.g is None:
            return

        try:
            # Get all nodes
            nodes = self.g.get_nodes()
            current_node_ids = set()

            # Calculate layout positions
            node_positions = self._calculateLayout(nodes)

            # Update/create nodes
            for node in nodes:
                node_id = node.id
                current_node_ids.add(node_id)

                pos = node_positions.get(node_id, (0, 0))

                if node_id in self.node_items:
                    # Update existing node position (if not being dragged)
                    item = self.node_items[node_id]
                    if not item.isSelected():
                        # Smooth movement
                        current_pos = item.pos()
                        new_x = current_pos.x() * 0.8 + pos[0] * 0.2
                        new_y = current_pos.y() * 0.8 + pos[1] * 0.2
                        item.setPos(new_x, new_y)
                else:
                    # Create new node
                    item = NodeItem(node_id, node.name, node.type, pos[0], pos[1])
                    self.scene.addItem(item)
                    self.node_items[node_id] = item

            # Remove nodes that no longer exist
            for node_id in list(self.node_items.keys()):
                if node_id not in current_node_ids:
                    item = self.node_items.pop(node_id)
                    self.scene.removeItem(item)

            # Update edges
            current_edges = set()
            for node in nodes:
                edges = self.g.get_edges_by_id(node.id)
                if edges:
                    for edge in edges:
                        edge_key = (edge.origin, edge.destination, edge.type)
                        current_edges.add(edge_key)

                        if edge_key not in self.edge_items:
                            # Create new edge
                            if edge.origin in self.node_items and edge.destination in self.node_items:
                                source = self.node_items[edge.origin]
                                target = self.node_items[edge.destination]
                                edge_item = EdgeItem(source, target, edge.type)
                                self.scene.addItem(edge_item)
                                self.edge_items[edge_key] = edge_item
                        else:
                            # Update existing edge
                            self.edge_items[edge_key].updatePosition()

            # Remove edges that no longer exist
            for edge_key in list(self.edge_items.keys()):
                if edge_key not in current_edges:
                    item = self.edge_items.pop(edge_key)
                    self.scene.removeItem(item)

        except Exception as e:
            print(f"[DSRGraphWidget] Error updating graph: {e}")

    def _calculateLayout(self, nodes) -> Dict[int, Tuple[float, float]]:
        """Calculate node positions using a simple radial layout."""
        positions = {}

        if not nodes:
            return positions

        # Find root node (or first node)
        root_node = None
        for node in nodes:
            if node.type == 'root' or node.name == 'root':
                root_node = node
                break

        if root_node is None and nodes:
            root_node = nodes[0]

        # Place root at center
        if root_node:
            positions[root_node.id] = (self.center_x, self.center_y)

        # Group nodes by type for better visualization
        nodes_by_level = {}
        for node in nodes:
            if node.id == (root_node.id if root_node else -1):
                continue
            level = self._getNodeLevel(node.type)
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        # Place nodes in concentric circles
        for level, level_nodes in sorted(nodes_by_level.items()):
            radius = self.layout_radius * (level + 1)
            n_nodes = len(level_nodes)

            for i, node in enumerate(level_nodes):
                if n_nodes == 1:
                    angle = -math.pi / 2  # Top
                else:
                    angle = 2 * math.pi * i / n_nodes - math.pi / 2

                x = self.center_x + radius * math.cos(angle)
                y = self.center_y + radius * math.sin(angle)
                positions[node.id] = (x, y)

        return positions

    def _getNodeLevel(self, node_type: str) -> int:
        """Get the hierarchical level for a node type."""
        levels = {
            'root': 0,
            'world': 0,
            'room': 1,
            'robot': 2,
            'shadow': 2,
            'person': 3,
            'object': 3,
            'table': 3,
            'chair': 3,
        }
        return levels.get(node_type, 4)

    def fitInView(self):
        """Fit the entire graph in the view."""
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def zoomIn(self):
        """Zoom in."""
        self.view.scale(1.2, 1.2)

    def zoomOut(self):
        """Zoom out."""
        self.view.scale(1/1.2, 1/1.2)


# Test
if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # Mock graph for testing
    class MockNode:
        def __init__(self, id, name, type):
            self.id = id
            self.name = name
            self.type = type

    class MockEdge:
        def __init__(self, origin, destination, type):
            self.origin = origin
            self.destination = destination
            self.type = type

    class MockGraph:
        def __init__(self):
            self.nodes = [
                MockNode(0, "root", "root"),
                MockNode(1, "room", "room"),
                MockNode(2, "shadow", "robot"),
                MockNode(3, "table_1", "table"),
                MockNode(4, "chair_1", "chair"),
            ]
            self.edges = [
                MockEdge(0, 1, "RT"),
                MockEdge(1, 2, "RT"),
                MockEdge(1, 3, "has"),
                MockEdge(1, 4, "has"),
            ]

        def get_nodes(self):
            return self.nodes

        def get_edges_by_id(self, node_id):
            return [e for e in self.edges if e.origin == node_id]

    window = QMainWindow()
    window.setWindowTitle("DSR Graph Widget Test")
    window.resize(800, 600)

    g = MockGraph()
    widget = DSRGraphWidget(g)
    window.setCentralWidget(widget)

    window.show()
    sys.exit(app.exec())
