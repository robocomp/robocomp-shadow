import sys
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import Slot, Qt
from pydsr import Attribute, signals

class GraphEdgeWidget(QTableWidget):
    def __init__(self, graph, from_id, to_id, edge_type):
        super().__init__()
        self.graph = graph
        self.from_id = from_id
        self.to_id = to_id
        self.edge_type = edge_type
        self.widget_map = {}
        edge = self.graph.get_edge(from_id, to_id, edge_type)
        if edge is not None:
            self.setWindowTitle(f"Edge {edge.type} from [{from_id}] to [{to_id}]")
            self.setColumnCount(2)
            attrs = edge.attrs
            attrs["type"] = Attribute(edge.type, 66)
            self.setHorizontalHeaderLabels(["Key", "Value"])
            for key, attr in attrs.items():
                self.insert_attribute(key, attr)
            self.horizontalHeader().setStretchLastSection(True)
            self.resize_widget()
            signals.connect(self.graph, signals.UPDATE_EDGE_ATTR, self.update_edge_attr_slot)
            self.show()

    @Slot(int, int, str, list)
    def update_edge_attr_slot(self, from_id, to_id, edge_type, attr_names):
        if from_id != self.from_id or to_id != self.to_id or edge_type != self.edge_type:
            return
        edge = self.graph.get_edge(from_id, to_id, edge_type)
        if edge is not None:
            for attrib_name in attr_names:
                if attrib_name in edge.attrs:
                    attr = edge.attrs[attrib_name]
                    if attrib_name in self.widget_map:
                        self.update_attribute_value(attrib_name, attr)
                    else:
                        self.insert_attribute(attrib_name, attr)

    def resizeEvent(self, event):
        cols = self.columnCount()
        for col in range(cols):
            self.setColumnWidth(col, (self.width() - self.verticalHeader().width() - 4) // cols)
        super().resizeEvent(event)

    def closeEvent(self, event):
        try:
            self.graph.signals.UPDATE_EDGE_ATTR.disconnect(self.update_edge_attr_slot)
        except Exception as e:
            print("GraphEdgeWidget: Error disconnecting signal:", e)
        super().closeEvent(event)

    def update_attribute_value(self, key, attr):
        widget = self.widget_map.get(key)
        if not widget:
            return
        widget.blockSignals(True)
        value = attr.value
        if isinstance(value, str):
            widget.setText(str(attr.value))
        elif isinstance(value, int):
            widget.setValue(int(attr.value))
        elif isinstance(value, float):
            widget.setValue(round(float(attr.value), 6))
        elif isinstance(value, bool):
            widget.setCurrentText("true" if attr.value else "false")
        widget.blockSignals(False)

    def insert_attribute(self, key, attr):
        inserted = True
        current_row = self.rowCount()
        self.insertRow(current_row)
        value = attr.value
        if isinstance(value, str):
            ledit = QLineEdit(str(value))
            self.setCellWidget(current_row, 1, ledit)
            self.widget_map[key] = ledit
            ledit.textChanged.connect(lambda text, mkey=key: self._on_text_changed(mkey, text))
        elif isinstance(value, int) and int(value) < 10000000 and int(value) > -10000000:
            spin = QSpinBox()
            spin.setMinimum(-10000000)
            spin.setMaximum(10000000)
            spin.setValue(int(value))
            self.setCellWidget(current_row, 1, spin)
            self.widget_map[key] = spin
            spin.valueChanged.connect(lambda val, mkey=key: self._on_value_changed(mkey, val))
        elif isinstance(value, float):
            dspin = QDoubleSpinBox()
            dspin.setMinimum(sys.float_info.min)
            dspin.setMaximum(sys.float_info.max)
            dspin.setValue(round(float(value), 6))
            self.setCellWidget(current_row, 1, dspin)
            self.widget_map[key] = dspin
            dspin.valueChanged.connect(lambda val, mkey=key: self._on_double_changed(mkey, val))
        elif isinstance(value, bool):
            combo = QComboBox()
            combo.addItems(["true", "false"])
            combo.setCurrentText("true" if value else "false")
            self.setCellWidget(current_row, 1, combo)
            self.widget_map[key] = combo
            combo.currentTextChanged.connect(lambda text, mkey=key: self._on_combo_changed(mkey, text))
        else:
            inserted = False

        if inserted:
            item = QTableWidgetItem(key)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.setItem(current_row, 0, item)
        else:
            self.removeRow(current_row)

    def _on_text_changed(self, key, text):
        edge = self.graph.get_edge(self.from_id, self.to_id, self.edge_type)
        if edge:
            edge.attrs[key] = Attribute(text, 66)
            self.graph.insert_or_assign_edge(edge)

    def _on_value_changed(self, key, value):
        edge = self.graph.get_edge(self.from_id, self.to_id, self.edge_type)
        if edge:
            edge.attrs[key] = Attribute(int(value), 66)
            self.graph.insert_or_assign_edge(edge)

    def _on_double_changed(self, key, value):
        edge = self.graph.get_edge(self.from_id, self.to_id, self.edge_type)
        if edge:
            edge.attrs[key] = Attribute(round(float(value), 6), 66)
            self.graph.insert_or_assign_edge(edge)

    def _on_combo_changed(self, key, text):
        edge = self.graph.get_edge(self.from_id, self.to_id, self.edge_type)
        if edge:
            edge.attrs[key] = Attribute(1, 66) if text == "true" else Attribute(0, 66)
            self.graph.insert_or_assign_edge(edge)

    def resize_widget(self):
        self.resizeRowsToContents()
        self.resizeColumnsToContents()
        width = self.model().columnCount() - 1 + self.verticalHeader().width() + 4
        height = self.model().rowCount() - 1 + self.horizontalHeader().height()
        for col in range(self.model().columnCount()):
            width += self.columnWidth(col)
        for row in range(self.model().rowCount()):
            height += self.rowHeight(row)
        self.setMinimumWidth(width)
        self.setMinimumHeight(height)
