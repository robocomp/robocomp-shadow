from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QTreeWidget, QTreeWidgetItem
from pydsr import signals, Attribute
from ui_masterUI import Ui_master
from pydsr import signals, Edge

class Affordances(QWidget, Ui_master):
    def __init__(self, g):
        super(Affordances, self).__init__()
        ui = Ui_master()
        ui.setupUi(self)
        self.g = g
        self.tree = ui.treeWidget  # Ensure it's changed to a QTreeWidget in the .ui file
        self.tree.setColumnCount(1)
        self.tree.setHeaderLabel("Concepts and Affordances")
        self.tree.itemChanged.connect(self.on_item_changed)

        self.relevant_node_ids = set()
        self.relevant_edges = set()
        self._internal_update = False
        self.node_item_map = {}
        self.edge_item_map = {}

    def connect_signals(self):
        # Connect to DSR graph signals
        try:
            signals.connect(self.g, signals.UPDATE_NODE, self.on_graph_update_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.on_graph_update_edge)
            signals.connect(self.g, signals.DELETE_NODE, self.on_graph_delete_node)
            signals.connect(self.g, signals.DELETE_EDGE, self.on_graph_delete_edge)
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.on_graph_update_node)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.on_graph_update_edge_attrs)
        except Exception as e:
            print(e)

    def populate(self):
        self._internal_update = True
        self.relevant_node_ids.clear()
        self.relevant_edges.clear()
        self.tree.clear()

        root = self.g.get_node("root")
        if root:
            root_item = self.create_node_item(root)
            self.tree.addTopLevelItem(root_item)
            self.relevant_node_ids.add(root.id)
            self.add_subtree(root, root_item)

        self.tree.expandAll()
        self._internal_update = False

    def add_subtree(self, parent_node, parent_item):
        # Traverse edges of type "RT" to recurse on child nodes
        for edge_tuple in parent_node.edges:
            fr, to, etype = parent_node.id, edge_tuple[0], edge_tuple[1]
            if etype == "RT":
                child_node = self.g.get_node(to)
                if child_node:
                    self.relevant_node_ids.add(child_node.id)
                    child_item = self.create_node_item(child_node)
                    parent_item.addChild(child_item)
                    self.add_subtree(child_node, child_item)

        # Handle "has_affordance" and "has_intention" at this level
        special_parent = None
        for edge_tuple in parent_node.edges:
            fr, to, etype = parent_node.id, edge_tuple[0], edge_tuple[1]
            if etype in ["has_affordance", "has_intention"]:
                e = self.g.get_edge(fr, to, etype)
                if e:
                    if not special_parent:
                        special_parent = QTreeWidgetItem(["Affordances / Intentions"])
                        parent_item.addChild(special_parent)
                    edge_item = self.create_edge_item(e)
                    special_parent.addChild(edge_item)

    def create_edge_item(self, e):
        """Build a QTreeWidgetItem for 'has_affordance' or 'has_intention' edges."""
        active = e.attrs["active"].value if "active" in e.attrs else False
        state_value = e.attrs["state"].value if "state" in e.attrs else "undefined"
        text = f"→ {e.type} ➝ {self.g.get_node(e.origin).name} ➝ {state_value}"
        item = QTreeWidgetItem([text])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked if active else Qt.Unchecked)
        item.setData(0, Qt.UserRole + 1, (e.origin, e.destination, e.type))
        self.relevant_edges.add((e.origin, e.destination, e.type))
        edge_key = (e.origin, e.destination, e.type)
        self.edge_item_map[edge_key] = item
        return item

    def create_node_item(self, node):
        item = QTreeWidgetItem([node.name])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        active = node.attrs["active"].value if "active" in node.attrs else False
        item.setCheckState(0, Qt.Checked if active else Qt.Unchecked)
        item.setData(0, Qt.UserRole + 1, node.id)
        self.node_item_map[node.id] = item
        return item

    def on_item_changed(self, item, column):
        if self._internal_update:
            return

        data = item.data(0, Qt.UserRole + 1)
        if isinstance(data, int):
            node_id = data
            active = item.checkState(0) == Qt.Checked
            if active: # insert a "current" type edge from node_id to node_id
                if not self.g.get_edge(node_id, node_id, "current"):
                    edge = Edge(node_id, node_id, "current", 66)  # TODO: hard
                    self.g.insert_or_assign_edge(edge)
            elif self.g.get_edge(node_id, node_id, "current"): # remove the "current" type edge from node_id to node_id
                self.g.delete_edge(node_id, node_id, "current")
        else:
            edge_key = data
            if edge_key:
                fr, to, etype = edge_key
                edge = self.g.get_edge(fr, to, etype)
                if edge:
                    active = item.checkState(0) == Qt.Checked
                    if edge.attrs.__contains__("active"):
                        edge.attrs["active"].value = active
                    self.g.insert_or_assign_edge(edge)

    ################################################################################
    ### SLOTS from G
    ################################################################################
    def on_graph_update_node(self, id: int, mtype: str):
        if id not in self.relevant_node_ids:
            print("Affordances::on_graph_update_node", self.g.get_node(id).name, mtype)
            self.populate() # only new node creations
        pass

    def on_graph_update_node_attrs(self, id: int, attribute_names: [str]):
        if self._internal_update:
            return
        node = self.g.get_node(id)
        if not node or id not in self.relevant_node_ids:
            return

        self._internal_update = True
        node_item = self.find_node_item(id)
        # if node_item:
        #     # Update "active" in QTreeWidget if it's changed
        #     if "active" in attribute_names and "active" in node.attrs:
        #         node_item.setCheckState(0, Qt.Checked if node.attrs["active"].value else Qt.Unchecked )
        #     # ...existing code for other attributes if needed...
        self._internal_update = False

    def find_node_item(self, node_id: int):
        """Locate the QTreeWidgetItem corresponding to this node ID."""
        return self.node_item_map.get(node_id, None)

    def on_graph_update_edge_attrs(self, fr: int, to: int, mtype: str, attribute_names: [str]):
        # We have to update only the "has_intention" and "has_affordance" edges' attributes
        if mtype not in ["has_affordance", "has_intention"]:
             return
        edge = self.g.get_edge(fr, to, mtype)
        edge_item = self.edge_item_map.get((fr, to, mtype), None)
        print("Affordances::on_graph_update_edge_attrs", fr, to, mtype, attribute_names, edge_item)
        if edge_item:
            self._internal_update = True    # prevent recursive calls
            for attribute_name in attribute_names:
                if attribute_name == "active":
                    if edge.attrs.__contains__("active"):
                        edge_item.setCheckState(0, Qt.Checked if edge.attrs["active"].value else Qt.Unchecked)
                elif attribute_name == "state":
                    if edge.attrs.__contains__("state"):
                        print("State changed", edge.attrs["state"].value)
                        state_value = edge.attrs["state"].value if "state" in edge.attrs else "undefined"
                        edge_item.setText(0, f"→ {mtype} ➝ {self.g.get_node(edge.destination).name} ➝ {state_value}")
            # Force a dataChanged signal on edge_item to refresh the UI
            self._internal_update = False

    def on_graph_update_edge(self, fr: int, to: int, mtype: str):
        if mtype in ["has_intention", "has_affordance"]:
            print("Affordances::on_graph_update_edge", self.g.get_node(fr).name, self.g.get_node(to).name, mtype)
        if (fr, to, mtype) not in self.relevant_edges:
            self.create_edge_item(self.g.get_edge(fr, to, mtype))

    def on_graph_delete_node(self, id: int):
        print("Affordances::on_graph_delete_node", id)
        for i in range(self.tree.topLevelItemCount()):
            parent_item = self.tree.topLevelItem(i)
            for j in range(parent_item.childCount()):
                child = parent_item.child(j)
                if child.data(0, Qt.UserRole + 1) == id:
                    parent_item.removeChild(child)
                    break
        # remove the node from the relevant node ids
        self.relevant_node_ids.discard(id)
        if id in self.node_item_map:
            del self.node_item_map[id]
        self.populate()

    def on_graph_delete_edge(self, fr: int, to: int, mtype: str):
        print("Affordances::on_graph_delete_edge", fr, to, mtype)
        # Remove the edge from the model
        self.relevant_edges.discard((fr, to, mtype))
        if (fr, to, mtype) in self.edge_item_map:
            del self.edge_item_map[(fr, to, mtype)]

