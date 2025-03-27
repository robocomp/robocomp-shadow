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

        # Connect to DSR graph signals
        # signals.connect(self.g, signals.UPDATE_NODE, self.on_graph_update)
        # signals.connect(self.g, signals.UPDATE_EDGE, self.on_graph_update)
        # signals.connect(self.g, signals.DELETE_NODE, self.on_graph_update)
        # signals.connect(self.g, signals.DELETE_EDGE, self.on_graph_update)

        self.populate()

    def populate(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Concepts and Affordances"])
        for node in self.g.get_nodes_by_type("omnirobot"):
            concept_item = QStandardItem(node.name)
            concept_item.setCheckable(True)
            #concept_item.setCheckState(Qt.Checked if node.attrs.get("active", False) else Qt.Unchecked)
            concept_item.setData(node.id, Qt.UserRole + 1)
            concept_item.setEditable(False)
            self.model.appendRow(concept_item)

            # for edge in self.g.get_edges_by_type("has"):
            #     if edge.src == node.id:
            #         affordance = self.g.get_node(edge.dst)
            #         if affordance and affordance.type == "affordance":
            #             aff_item = QStandardItem(affordance.attrs.node.attrs["name"].value, node.attrs["name"].value)
            #             aff_item.setFlags(Qt.ItemIsEnabled)
            #             concept_item.appendRow(aff_item)

        self.tree.expandAll()

    def on_item_changed(self, item):
        if item.parent() is None:  # Only top-level items are concepts
            node_id = item.data(Qt.UserRole + 1)
            active = item.checkState() == Qt.Checked
            node = self.g.get_node(node_id)
            if node:
                self.g.add_or_modify_attrib_local_node(node_id, Attribute("active", active))
                self.g.update_node(node)

    def on_graph_update(self, *args, **kwargs):
        self.populate()
