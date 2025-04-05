# Language: python
import sys

from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import Slot, Qt
from pydsr import Attribute, signals

class GraphNodeWidget(QTableWidget):
    def __init__(self, graph, node_id):
        super().__init__()
        self.graph = graph
        self.node_id = node_id
        self.widget_map = {}
        node = self.graph.get_node(node_id)
        if node is not None:
            self.setWindowTitle("Node " + str(node.type) + " [" + str(node_id) + "]")
            self.setColumnCount(2)
            # Add system attributes (ID, type, name)
            attrs = node.attrs
            attrs["ID"] = Attribute(node_id, 66) #TODO: get value from g
            attrs["type"] = Attribute(node.type, 66)
            attrs["name"] = Attribute(node.name, 66)
            self.setHorizontalHeaderLabels(["Key", "Value"])
            for key, attr in attrs.items():
                self.insert_attribute(key, attr)
            self.horizontalHeader().setStretchLastSection(True)
            self.resize_widget()
            # Connect update signal to update node attributes
            signals.connect(self.graph, signals.UPDATE_NODE_ATTR, self.update_node_attr_slot)
            self.show()

    @Slot(int, list)
    def update_node_attr_slot(self, node_id: int, attr_names: [str]):
        if node_id != self.node_id:
            return
        node = self.graph.get_node(node_id)
        if node is not None:
            for attrib_name in attr_names:
                if node.attrs.__contains__(attrib_name):
                    attr = node.attrs[attrib_name]
                    if attrib_name in self.widget_map:
                        self.update_attribute_value(attrib_name, attr)
                    else:
                        self.insert_attribute(attrib_name, attr)

    @Slot(int, str)
    def update_node_slot(self, node_id, node_type):
        if node_id != self.node_id:
            return
        node = self.graph.get_node(node_id)
        if node is not None:
            for key, attr in node.attrs:
                if key in self.widget_map:
                    self.update_attribute_value(key, attr)
                else:
                    self.insert_attribute(key, attr)

    def resizeEvent(self, event):
        cols = self.columnCount()
        for col in range(cols):
            self.setColumnWidth(col, (self.width() - self.verticalHeader().width() - 4) // cols)
        super().resizeEvent(event)

    def closeEvent(self, event):
        try:
            self.graph.signals.UPDATE_NODE_ATTR.disconnect(self.update_node_attr_slot)
        except Exception as e:
            print("Graph_Node_Widget: Error disconnecting signal:", e)
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
        elif isinstance(value,  float):
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
        node = self.graph.get_node(self.node_id)
        if node:
            node.attrs[key] = Attribute(text, 66)
            self.graph.update_node(node)

    def _on_value_changed(self, key, value):
        node = self.graph.get_node(self.node_id)
        if node:
            node.attrs[key] = Attribute(int(value), 66)
            self.graph.update_node(node)

    def _on_double_changed(self, key, value):
        node = self.graph.get_node(self.node_id)
        if node:
            node.attrs[key] = Attribute(round(float(value), 6), 66)
            self.graph.update_node(node)

    def _on_combo_changed(self, key, text):
        node = self.graph.get_node(self.node_id)
        if node:
            node.attrs[key] = Attribute(1, 66) if text == "true" else Attribute(0, 66)
            self.graph.update_node(node)

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

