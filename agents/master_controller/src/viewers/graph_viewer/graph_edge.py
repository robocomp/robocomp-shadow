import math
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsTextItem, QMenu, QGraphicsItem, QGraphicsPolygonItem, QWidget, QHBoxLayout, QLabel, QComboBox, QWidgetAction
from PySide6.QtGui import QPen, QColor, QAction, QPainterPath, QPolygonF, QTransform, QCursor
from PySide6.QtCore import Qt, QPointF, Signal, QObject
from PySide6.QtWidgets import QGraphicsTextItem, QMenu, QApplication

class HoverableLabel(QGraphicsTextItem):
    def __init__(self, gedge: 'GraphicsEdge'):
        super().__init__(gedge)
        self.gedge = gedge
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

    def on_edge_updated(self):
        # Update label properties on edge update as needed.
        # For example, update the displayed text:
        self.setPlainText(f"{self.gedge.mtype} updated")
        
    def hoverEnterEvent(self, event):
        if self.gedge.mtype == "has_intention":
            self.gedge.showHasIntentionMenu(event.screenPos())
        super().hoverEnterEvent(event)

class GraphicsEdge(QObject, QGraphicsPathItem):
    edgeUpdated = Signal()  # Define the signal as a class attribute

    def __init__(self, source_node_id: int, target_node_id: int, mtype: str, graph_ref: object, offset_index: int = 0):
        #initialize QObject and QGraphicsPathItem
        super().__init__()
        QGraphicsPathItem.__init__(self)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.g = graph_ref
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.mtype = mtype
        self.key = (source_node_id, target_node_id, mtype)
        self.offset_index = offset_index

        self.setZValue(5)
        self.setPen(QPen(self.edge_color(self.mtype), 2))

        # Label
        self.label = HoverableLabel(self)  # Use HoverableLabel instead of QGraphicsTextItem
        self.label.setPlainText(self.mtype)
        self.label.setDefaultTextColor(self.edge_color(self.mtype))
        self.label.setAcceptHoverEvents(True)
        self.label.installSceneEventFilter(self)
        self.update_label_position()

        # Connect signal so the label can update when the edge updates
        #self.edgeUpdated.connect(self.label.on_edge_updated)

        # arrow
        self.arrow = QGraphicsPolygonItem(self)
        self.arrow.setBrush(self.edge_color(self.mtype))
        self.arrow.setZValue(6)  # above the edge path

        self.update_position()

    def update_position(self):
        source = self.g.get_node(self.source_node_id)
        target = self.g.get_node(self.target_node_id)
        if not source or not target:
            return

        x1 = source.attrs["pos_x"].value
        y1 = source.attrs["pos_y"].value
        x2 = target.attrs["pos_x"].value
        y2 = target.attrs["pos_y"].value

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length == 0:
            return

        # Perpendicular offset
        nx = -dy / length
        ny = dx / length
        spacing = 40  # curve separation
        offset = (self.offset_index - 0.5) * spacing  # center the curves if 2+

        # Control point for quadratic Bézier
        ctrl_x = (x1 + x2) / 2 + nx * offset
        ctrl_y = (y1 + y2) / 2 + ny * offset

        # Create the path
        path = QPainterPath()
        path.moveTo(x1, y1)
        path.quadTo(ctrl_x, ctrl_y, x2, y2)
        self.setPath(path)

        # Arrow placement at the end
        # Constants
        node_radius = 20  # approximate visual radius of your node circle
        arrow_size = 10

        # Adjust target point to stop early
        path = QPainterPath()
        path.moveTo(x1, y1)

        # Curve control point (as before)
        ctrl_x = (x1 + x2) / 2 + nx * offset
        ctrl_y = (y1 + y2) / 2 + ny * offset

        # Compute new target point (shrink the vector)
        vec_x = x2 - x1
        vec_y = y2 - y1
        vec_len = math.hypot(vec_x, vec_y)
        if vec_len == 0:
            vec_len = 1  # avoid division by zero
        shrink_distance = node_radius * 0.5  # or tweak to 15, 10, etc.
        shrink_factor = (vec_len - shrink_distance) / vec_len
        x2_adj = x1 + vec_x * shrink_factor
        y2_adj = y1 + vec_y * shrink_factor

        # Rebuild path with adjusted endpoint
        path.quadTo(ctrl_x, ctrl_y, x2_adj, y2_adj)
        self.setPath(path)

        # Update color if this is a has_intention edge
        if self.mtype == "has_intention":
            edge = self.g.get_edge(self.source_node_id, self.target_node_id, self.mtype)
            active = edge.attrs["active"].value if "active" in edge.attrs else False
            color = QColor("green") if active else QColor("blue")
            self.setPen(QPen(color, 2))
            self.label.setDefaultTextColor(color)
            self.arrow.setBrush(color)

        # Arrow placement at adjusted end
        angle = math.radians(-path.angleAtPercent(1.0))
        end_point = path.pointAtPercent(1.0)

        arrow_poly = QPolygonF([
            QPointF(0, 0),
            QPointF(-arrow_size, arrow_size / 2),
            QPointF(-arrow_size, -arrow_size / 2),
        ])

        transform = QTransform()
        transform.translate(end_point.x(), end_point.y())
        transform.rotateRadians(angle)
        arrow_poly = transform.map(arrow_poly)
        self.arrow.setPolygon(arrow_poly)
        self.update_label_position()

    def update_label_position(self):
        path = self.path()
        percent = 0.5  # middle of the curve

        pt = path.pointAtPercent(percent)
        tangent = path.angleAtPercent(percent)
        radians = math.radians(tangent + 90)  # perpendicular to tangent

        offset_distance = 12  # pixels away from the curve
        dx = math.cos(radians) * offset_distance
        dy = -math.sin(radians) * offset_distance  # Qt y-axis is inverted

        self.label.setPos(pt.x() + dx, pt.y() + dy)

    def edge_color(self, edge_type: str) -> QColor:
        # You can customize this as needed
        color_map = {
            "RT": QColor("purple"),
            "has_intention": QColor("blue"),
            "has_affordance": QColor("green"),
            "current": QColor("orange"),
            "parent": QColor("darkRed"),
        }
        return color_map.get(edge_type, QColor("gray"))  # default fallback

    def update_edge(self, attribute_names: [str]):
        # TODO: is a persistent popup menu, update values
        # ... update logic here if needed ...
        self.update_position()
        #self.edgeUpdated.emit()  # emit the signal to the HoverableLabel

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            menu = QMenu()
            edge = self.g.get_edge(self.source_node_id, self.target_node_id, self.mtype)
            if edge:
                for attr_name, attr in edge.attrs.items():
                    action = QAction(f"{attr_name}: {attr.value}", menu)
                    action.setEnabled(False)
                    menu.addAction(action)
            menu.exec(event.screenPos())

    def showHasIntentionMenu(self, screen_pos):
        """Open a QMenu for 'has_intention' edges to toggle attributes, etc."""
        menu = QMenu()
        edge = self.g.get_edge(self.source_node_id, self.target_node_id, self.mtype)
        if edge:
            for attr_name, attr in edge.attrs.items():
                if attr_name == "active":
                    # Create widget with label + combo box
                    widget = QWidget()
                    layout = QHBoxLayout()
                    layout.setContentsMargins(4, 2, 4, 2)

                    label = QLabel("active:")
                    combo = QComboBox()
                    combo.addItems(["true", "false"])
                    combo.setCurrentText("true" if attr.value else "false")

                    def handler(edge=edge, combo=combo):
                        value = combo.currentText()
                        edge.attrs["active"].value = True if value == "true" else False
                        self.g.insert_or_assign_edge(edge)

                    combo.currentTextChanged.connect(lambda: handler(edge=edge, combo=combo))

                    layout.addWidget(label)
                    layout.addWidget(combo)
                    widget.setLayout(layout)

                    action = QWidgetAction(menu)
                    action.setDefaultWidget(widget)
                    menu.addAction(action)

                else:
                    # Read-only fallback for other attributes
                    action = QAction(f"{attr_name}: {attr.value}", menu)
                    action.setEnabled(False)
                    menu.addAction(action)

        menu.exec(screen_pos)

    def sceneEventFilter(self, watched: QGraphicsItem, event):
        print("event", event.type())
        if watched == self.label and event.type() == event.GraphicsSceneHoverEnter:
            if self.mtype == "has_intention":
                self.showHasIntentionMenu(event.screenPos())
            return True
        return super().sceneEventFilter(watched, event)
