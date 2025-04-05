from PyQt5.QtGui import QTextItem
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QTreeWidget, QTreeWidgetItem, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QLabel
from pydsr import signals, Attribute
from pydsr import signals, Edge

class Affordances(QTreeWidget):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.tree_map = {}
        self.types_map = {}
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.create_tree()
        self.setColumnCount(3)
        self.setHeaderLabels(["Concepts", "Affordances", "State"])
        self.header().setDefaultSectionSize(200)
        self.expandAll()
        signals.connect(self.g, signals.UPDATE_NODE, self.add_or_assign_node_SLOT)
        signals.connect(self.g, signals.DELETE_NODE, self.del_node_SLOT)
        signals.connect(self.g, signals.DELETE_EDGE, self.del_edge_SLOT)
        signals.connect(self.g,signals.UPDATE_EDGE_ATTR, self.update_edge_att_SLOT)

    def create_tree(self):
        self.clear()
        self.tree_map.clear()
        self.types_map.clear()
        try:
            for node in self.g.get_nodes():
                if node.type not in ["root"]:
                    self.add_or_assign_node_SLOT(node.id, node.type)
        except Exception as e:
            print("Affordances: Error creating tree:", e)

    #########################################################################
    ###  DSR SIGNALS HANDLER
    #########################################################################
    def add_or_assign_node_SLOT(self, mid: int, mtype: str):
        try:
            node = self.g.get_node(mid)
            if mid not in self.tree_map:
                symbol_widget = self.types_map.get(mtype)
                if not symbol_widget:
                    # Create a new QTreeWidgetItem for the type if it doesn't exist
                    symbol_widget = QTreeWidgetItem(self.invisibleRootItem())
                    symbol_widget.setText(0, f"Type: {mtype}")
                    self.types_map[mtype] = symbol_widget
                # Create a new QTreeWidgetItem for the node
                item = QTreeWidgetItem(symbol_widget)
                #item.setText(0, f"{node.name}")
                checkbox = QCheckBox(f"{node.name}")
                self.setItemWidget(item, 0, checkbox)
                checkbox.stateChanged.connect(lambda value, m_node=node: self.node_current_change_SLOT(value, m_node))
                self.tree_map[mid] = item
                self.create_columns_for_affordances(item, node)

        except Exception as e:
            print(f"Affordances: Error {e} in method {self.add_or_assign_node_SLOT.__name__} when adding node {mid} of type {mtype}")

    def del_node_SLOT(self, id: int):
        while id in self.tree_map:
            item = self.tree_map[id]
            self.invisibleRootItem().removeChild(item)
            del item
            del self.tree_map[id]

    def update_edge_att_SLOT(self, fr: int, to: int, mtype: str, attribute_names: [str]):
        if mtype != "RT":
            print("Affordances: update_edge_att_SLOT", fr, to, mtype, attribute_names)
        edge = self.g.get_edge(fr, to, mtype)
        if edge is not None and mtype in ["has_affordance", "has_intention"]:
           if fr in self.tree_map.keys():
               parent = self.tree_map[fr]
               if "state" in attribute_names:
                   if edge.attrs.__contains__("state"):
                       new_state = edge.attrs["state"].value
                       self.itemWidget(parent,2).setText(f"{new_state}")
                       if new_state == "waiting":
                          self.itemWidget(parent, 2).setStyleSheet("color: blue;")
                       elif new_state == "in_progress":
                            self.itemWidget(parent, 2).setStyleSheet("color: orange;")
                       elif new_state == "completed":
                            self.itemWidget(parent, 2).setStyleSheet("color: green;")
               if "active" in attribute_names:
                      if edge.attrs.__contains__("active"):
                        new_active = edge.attrs["active"].value
                        self.itemWidget(parent,1).setChecked(new_active)
                        self.itemWidget(parent, 1).setStyleSheet("color: green;") if new_active else self.itemWidget(parent,1).setStyleSheet("color: blue;")

    def del_edge_SLOT(self, fr: int, to: int, mtype: str):
        if mtype in ["has_affordance", "has_intention"]:
            if fr in self.tree_map.keys():
                parent = self.tree_map[fr]
                self.removeItemWidget(parent, 1)  # Clear widget in column 1
                self.removeItemWidget(parent, 2)  # Clear widget in column 2

    ############# LOCAL SLOTS ######################
    def node_change_SLOT(self, value, id, type, parent):
        # sender = self.sender()
        # if isinstance(sender, QCheckBox):
        #     self.node_check_state_changed.emit(value, id, type, parent)
        pass

    def create_columns_for_affordances(self, parent, node):
        for edge in node.get_edges():
            n = self.g.get_node(edge.destination)
            if n is None:
                self.g.delete_edge(edge.origin, edge.destination, edge.type)
                print("Affordances: Dangling edge removed", edge.origin, edge.destination, edge.type)
                continue
            if edge.type == "has_affordance" or edge.type == "has_intention":
                print("no deberaía de estar aqui", edge.type, edge.origin, self.g.get_node(edge.destination).name)
                if  edge.attrs.__contains__("active"):
                    val = edge.attrs['active'].value
                    act = QCheckBox(f"{edge.type}")
                    act.setChecked(val)
                    act.setStyleSheet("color: green;") if val else act.setStyleSheet("color: blue;")
                    self.setItemWidget(parent, 1, act)
                    act.stateChanged.connect(lambda value, m_node=node, m_edge=edge: self.edge_active_state_change_SLOT(value, m_node, m_edge))
                # add column 2 for state
                if edge.attrs.__contains__("state"):
                    state = edge.attrs['state'].value
                    st = QLabel(str(state))
                    if state == "waiting":
                       st.setStyleSheet("color: blue;")
                    elif state == "in_progress":
                       st.setStyleSheet("color: orange;")
                    elif state == "completed":
                        st.setStyleSheet("color: green;")
                    self.setItemWidget(parent, 2, st)

    def edge_active_state_change_SLOT(self, value, node, edge):
        e = self.g.get_edge(edge.origin, edge.destination, edge.type)
        if e is not None:
            v = True if value >0 else False
            print(f"Active state changed to {v} for node {node.name} and edge {edge.type}")
            e.attrs["active"] = Attribute(v)
            res = self.g.insert_or_assign_edge(e)
            if not res:
                print(f"Affordances: Error updating edge {node.name} with edge {e.type}")

    def node_current_change_SLOT(self, value, node):
        v = True if value >0 else False
        print(f" {node.name} changed to {v} for current")
        edge = Edge(node.id, node.id, "current", self.g.get_agent_id())
        res = self.g.insert_or_assign_edge(edge)
        if not res:
            print(f"Affordances: Error updating edge {node.name} with edge {e.type}")
