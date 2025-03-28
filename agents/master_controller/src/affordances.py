from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from pydsr import signals, Attribute
from ui_masterUI import Ui_master

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

        # Connect to DSR graph signals
        signals.connect(self.g, signals.UPDATE_NODE, self.on_graph_update_node)
        # signals.connect(self.g, signals.UPDATE_EDGE, self.on_graph_update_edge)
        # signals.connect(self.g, signals.DELETE_NODE, self.on_graph_update_node)
        # signals.connect(self.g, signals.DELETE_EDGE, self.on_graph_update_edge)

        self.populate()

    def populate(self):
        print("populating")
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
                aff_item = QStandardItem(f"→ {e.type} ➝ {target_node.name} ➝ {e.attrs['state'].value}")
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
        item.setCheckable(True)
        active = node.attrs["active"].value if "active" in node.attrs else False
        item.setCheckState(Qt.Checked if active else Qt.Unchecked)
        item.setData(node.id, Qt.UserRole + 1)
        item.setEditable(False)
        return item

    def on_item_changed(self, item):
        if item.parent() is None:
            node_id = item.data(Qt.UserRole + 1)
            active = item.checkState() == Qt.Checked
            node = self.g.get_node(node_id)
            if node:
                self.g.add_or_modify_attrib_local_node(node_id, Attribute("active", active))
                self.g.update_node(node)
        else:
            edge_key = item.data(Qt.UserRole + 1)
            if edge_key:
                fr, to, etype = edge_key
                edge = self.g.get_edge(fr, to, etype)
                if edge:
                    active = item.checkState() == Qt.Checked
                    self.g.add_or_modify_attrib_local_edge(fr, to, etype, Attribute("active", active))
                    self.g.update_edge(fr, to, etype)

    def on_graph_update_node(self, id: int, type: str):
        if id not in self.relevant_node_ids:
            self.populate() # only new node creations

    def on_graph_update_edge_attrs(self, fr: int, to: int, type: str, attribute_names: [str]):
        key = (fr, to, type)
        if key not in self.relevant_edges:
            return

        edge = self.g.get_edge(fr, to, type)
        if edge is None or "active" not in edge.attrs:
            return

        active = edge.attrs["active"].value
        state = edge.attrs["state"].value if "state" in edge.attrs else ""

        target_node = self.g.get_node(to)
        target_name = target_node.name if target_node else "unknown"

        # Walk through the model to find and update the item
        for i in range(self.model.rowCount()):
            parent_item = self.model.item(i)
            for j in range(parent_item.rowCount()):
                child = parent_item.child(j)
                if child.text() == "Affordances / Intentions":
                    for k in range(child.rowCount()):
                        edge_item = child.child(k)
                        if edge_item.data(Qt.UserRole + 1) == key:
                            self.model.blockSignals(True)
                            edge_item.setCheckState(Qt.Checked if active else Qt.Unchecked)
                            edge_item.setText(f"→ {type} ➝ {target_name} ➝ {state}")
                            self.model.blockSignals(False)
                            return

