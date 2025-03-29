from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from pydsr import signals, Attribute
from ui_masterUI import Ui_master
from pydsr import signals

class Affordances(QWidget, Ui_master):
    def __init__(self, g):
        super(Affordances, self).__init__()
        ui = Ui_master()
        ui.setupUi(self)
        self.tree = ui.treeView
        self.g = g
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Concepts and Affordances"])
        self.tree.setModel(self.model)
        self.tree.setHeaderHidden(False)
        self.tree.expandAll()

        self.model.itemChanged.connect(self.on_item_changed)

        self.relevant_node_ids = set()
        self.relevant_edges = set()
        self._internal_update = False

        # Connect to DSR graph signals
        try:
            # signals.connect(self.g, signals.UPDATE_NODE, self.on_graph_update_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.on_graph_update_edge)
            # signals.connect(self.g, signals.DELETE_NODE, self.on_graph_update_node)
            # signals.connect(self.g, signals.DELETE_EDGE, self.on_graph_update_edge)
            # signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.on_graph_update_node)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.on_graph_update_edge_attrs)
        except Exception as e:
            print(e)

    def populate(self):
        print("Affordances::populating")
        self.relevant_node_ids.clear()
        self.relevant_edges.clear()

        self.model.blockSignals(True)
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Concepts and Affordances"])

        root = self.g.get_node("root")
        if root is None:
            self.model.blockSignals(False)
            return

        for edge in root.edges:
            key = (root.id, edge[0], edge[1])
            e = self.g.get_edge(*key)
            if e.type == "RT":
                child_node = self.g.get_node(e.destination)
                if child_node:
                    self.relevant_node_ids.add(child_node.id)
                    root_item = self.create_node_item(child_node)
                    self.model.appendRow(root_item)
                    self.recursive_add_children(child_node, root_item)

        self.model.blockSignals(False)
        self.tree.expandAll()

    def recursive_add_children(self, parent_node, parent_item):
        has_special = False
        special_parent = QStandardItem("Affordances / Intentions")
        special_parent.setEditable(False)

        for edge_tuple in parent_node.edges:
            key = (parent_node.id, edge_tuple[0], edge_tuple[1])
            e = self.g.get_edge(*key)
            if e and e.type in ["has_affordance", "has_intention"]:
                target_node = self.g.get_node(e.destination)
                if target_node is None:
                    continue
                self.relevant_edges.add((e.origin, e.destination, e.type))
                self.relevant_node_ids.add(target_node.id)
                active = e.attrs["active"].value if "active" in e.attrs else False
                aff_item = QStandardItem(f"→ {e.type} ➝ {e.attrs['state'].value}")
                aff_item.setCheckable(True)
                aff_item.setCheckState(Qt.Checked if active else Qt.Unchecked)
                aff_item.setData((e.origin, e.destination, e.type), Qt.UserRole + 1)
                aff_item.setEditable(False)
                special_parent.appendRow(aff_item)
                has_special = True

        if has_special:
            parent_item.appendRow(special_parent)

        # Recurse RT children
        for edge in parent_node.edges:
            key = (parent_node.id, edge[0], edge[1])
            e = self.g.get_edge(*key)
            if e and e.type == "RT":
                child_node = self.g.get_node(e.destination)
                if child_node:
                    self.relevant_node_ids.add(child_node.id)
                    self.relevant_edges.add((e.origin, e.destination, e.type))
                    child_item = self.create_node_item(child_node)
                    parent_item.appendRow(child_item)
                    self.recursive_add_children(child_node, child_item)

    def create_node_item(self, node):
        item = QStandardItem(node.name)
        item.setCheckable(False)
        active = node.attrs["active"].value if "active" in node.attrs else False
        item.setCheckState(Qt.Checked if active else Qt.Unchecked)
        item.setData(node.id, Qt.UserRole + 1)
        item.setEditable(False)
        return item

    def on_item_changed(self, item):
        if self._internal_update:
            return

        if item.parent() is None:
            node_id = item.data(Qt.UserRole + 1)
            active = item.checkState() == Qt.Checked
            node = self.g.get_node(node_id)
            if node:
                self.g.insert_or_assign_node(node_id)
        else:
            edge_key = item.data(Qt.UserRole + 1)
            if edge_key:
                fr, to, etype = edge_key
                edge = self.g.get_edge(fr, to, etype)
                if edge:
                    active = item.checkState() == Qt.Checked
                    if edge.attrs.__contains__("active"):
                        edge.attrs["active"].value = active
                    self.g.insert_or_assign_edge(edge)

    def on_graph_update_node(self, id: int, type: str):
        # if id not in self.relevant_node_ids:
        #     self.populate() # only new node creations
        pass

    def on_graph_update_edge_attrs(self, fr: int, to: int, type: str, attribute_names: [str]):
        # We have to update only the "has_intention" and "has_affordance" edges' attributes
        if type not in ["has_affordance", "has_intention"]:
             return
        edge = self.g.get_edge(fr, to, type)
        edge_item = self.find_edge_item(fr, to, type)
        if edge_item:
            self.model.blockSignals(True)
            self._internal_update = True
            for attribute_name in attribute_names:
                if attribute_name == "active":
                    if edge.attrs.__contains__("active"):
                        edge_item.setCheckState(Qt.Checked if edge.attrs["active"].value else Qt.Unchecked)
                elif attribute_name == "state":
                    if edge.attrs.__contains__("state"):
                        state = edge_item.text().split("➝")[1].strip()
                        edge_item.setText(f"→ {type} ➝ {to} ➝ {state}")
            self.model.blockSignals(False)
            # Force a dataChanged signal on edge_item to refresh the UI
            idx = edge_item.index()
            self.model.dataChanged.emit(idx, idx)
            self._internal_update = False

    def find_edge_item(self, fr: int, to: int, mtype: str):
        for i in range(self.model.rowCount()):
            parent_item = self.model.item(i)
            for j in range(parent_item.rowCount()):
                child = parent_item.child(j)
                if child.text() == "Affordances / Intentions":
                    for k in range(child.rowCount()):
                        edge_item = child.child(k)
                        if edge_item.data(Qt.UserRole + 1) == (fr, to, mtype):
                            return edge_item
        return None
