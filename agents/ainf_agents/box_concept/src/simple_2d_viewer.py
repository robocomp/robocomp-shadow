#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple 2D Viewer using QPainter for testing.
This replaces Qt3D temporarily to verify the dock widget works.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget
import math


class Simple2DViewer(QWidget):
    """
    Simple 2D viewer using QPainter.
    Draws a red square and some circles to verify rendering works.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setAutoFillBackground(True)

        # Set background color
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(50, 50, 60))
        self.setPalette(palette)

        # Animation angle
        self.angle = 0

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)  # 20 FPS

        print("[2D Viewer] Simple 2D viewer created")

    def _animate(self):
        """Update animation angle."""
        self.angle = (self.angle + 2) % 360
        self.update()

    def paintEvent(self, event):
        """Draw content."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        cx = width // 2
        cy = height // 2

        # Draw background
        painter.fillRect(0, 0, width, height, QColor(50, 50, 60))

        # Draw grid
        painter.setPen(QPen(QColor(80, 80, 100), 1))
        for x in range(0, width, 50):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, 50):
            painter.drawLine(0, y, width, y)

        # Draw axes
        painter.setPen(QPen(QColor(255, 0, 0), 2))  # X axis - Red
        painter.drawLine(cx, cy, cx + 100, cy)
        painter.setPen(QPen(QColor(0, 255, 0), 2))  # Y axis - Green
        painter.drawLine(cx, cy, cx, cy - 100)

        # Draw a red square (static)
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0, 128)))
        painter.drawRect(cx - 50, cy - 50, 100, 100)

        # Draw a rotating circle
        radius = 80
        angle_rad = math.radians(self.angle)
        px = int(cx + radius * math.cos(angle_rad))
        py = int(cy - radius * math.sin(angle_rad))

        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.setBrush(QBrush(QColor(0, 255, 255, 180)))
        painter.drawEllipse(px - 15, py - 15, 30, 30)

        # Draw text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(10, 20, f"2D Viewer Test - Angle: {self.angle}°")
        painter.drawText(10, 40, f"Size: {width}x{height}")

        painter.end()

    def update(self, *args, **kwargs):
        """Trigger repaint and handle update calls from specificworker."""
        super().update()

    def get_widget(self):
        """Return self for compatibility with Qt3D viewer pattern."""
        return self

    def show(self):
        """Show widget."""
        super().show()

    def start_async(self):
        """Compatibility method."""
        pass


# Alias for compatibility
Qt3DObjectVisualizerWidget = Simple2DViewer


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("Simple 2D Viewer Test")
    window.resize(600, 500)

    viewer = Simple2DViewer()
    window.setCentralWidget(viewer)

    window.show()

    print("Window shown - you should see a red square with a rotating cyan circle")

    sys.exit(app.exec())
