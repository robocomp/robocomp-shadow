#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract Graphic Viewer - Base class for graph visualization widgets.
"""

from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QApplication


class TcWidget:
    """Mixin for tree-connectable widgets."""
    pass


class AbstractGraphicViewer(QGraphicsView, TcWidget):
    def __init__(self, parent=None):
        super(AbstractGraphicViewer, self).__init__(parent)
        self._panning = False
        self._last_mouse_pos = None

        self.scene = QGraphicsScene()
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.scene.setSceneRect(-2000, -2000, 4000, 4000)
        self.setScene(self.scene)

        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(400, 400)
        self.adjustSize()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        view_pos = event.position().toPoint()
        scene_pos = self.mapToScene(view_pos)
        self.centerOn(scene_pos)
        self.scale(factor, factor)
        delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
        self.centerOn(scene_pos - delta)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is None:
                self._panning = True
                self._last_mouse_pos = event.position().toPoint()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        super(AbstractGraphicViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_mouse_pos is not None:
            delta = event.position().toPoint() - self._last_mouse_pos
            self._last_mouse_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super(AbstractGraphicViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super(AbstractGraphicViewer, self).mouseReleaseEvent(event)

    def showEvent(self, event):
        super(AbstractGraphicViewer, self).showEvent(event)
        adjusted = self.scene.itemsBoundingRect().adjusted(-100, -100, 100, 100)
        self.scene.setSceneRect(adjusted)
        update_state = self.updatesEnabled()
        self.setUpdatesEnabled(False)
        self.fitInView(adjusted, Qt.KeepAspectRatio)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        self.fitInView(adjusted, Qt.KeepAspectRatio)
        self.setUpdatesEnabled(update_state)
