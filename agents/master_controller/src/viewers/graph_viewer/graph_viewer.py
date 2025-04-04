import math

from LanguageSelector.gtk.GtkLanguageSelector import blockSignals
from PySide6 import QtCore
from PySide6.QtCore import Qt
from viewers._abstract_graphic_view import AbstractGraphicViewer
from .graph_edge import GraphicsEdge
from .graph_node import GraphicsNode

class GraphViewer(AbstractGraphicViewer):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.gmap = {}  # node_id: QGraphicsEllipseItem
        self.gmap_edges = {}  # node_id: QGraphicsPathItem
        self.type_id_map = {}
        self._internal_update = False
        self.create_graph()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio )
        #self.connectGraphSignals()

    def create_graph(self):
        self.blockSignals(True)
        self.gmap = {}
        self.gmap_edges = {}
        self.type_id_map = {}
        self.scene.clear()
        for node in self.g.get_nodes():
            self.add_or_assign_node_slot(node.id, node.type)
            self.type_id_map[node.type] = node.id
        for node in self.g.get_nodes():
            for edge in node.edges:
                self.add_or_assign_edge_slot(node.id, edge[0], edge[1])
        self.blockSignals(False)

    #################################################################################
    @QtCore.Slot()
    def add_or_assign_node_slot(self, node_id: int, mtype: str):
        node = self.g.get_node(node_id)
        gnode = None
        if node:
            if node.id not in self.gmap:
                print("Graph viewer: Adding node", node.name)
                gnode = GraphicsNode(self)
                gnode.id_in_graph = node.id
                gnode.set_type(mtype)
                self.scene.addItem(gnode)
                ## connect
                self.gmap[node.id] = gnode
                color = "coral"
                if node.attrs.__contains__("color"):
                    color = node.attrs["color"].value
                gnode.set_node_color(color)
                gnode.set_type(mtype)
            else:
                gnode = self.gmap[node.id]
        if node.attrs.__contains__("pos_x") and node.attrs.__contains__("pos_y"):
            px = node.attrs["pos_x"]
            py = node.attrs["pos_y"]
            if px is not None and py is not None:
                gnode.setPos(float(px.value), float(py.value))

        for edge in node.edges:
            key = (node.id, edge[0], edge[1])
            if key in self.gmap_edges and self.gmap_edges[key] is None:
                self.add_or_assign_edge_slot(node.id, edge[0], edge[1])

    def add_or_assign_edge_slot(self, fr: int, to: int, mtype: str):
        # Skip if already present
        if fr is None or to is None:
            print("Graph viewer: Error adding edge!", fr, to, mtype)
            return

        key = (fr, to, mtype)
        try:
            if self.g.get_edge(fr, to, mtype):
                if key not in self.gmap_edges.keys():
                    print("Graph viewer: Adding edge", fr, to, mtype)
                    source_node = self.gmap[fr]
                    destination_node = self.gmap[to]
                    item = GraphicsEdge(source_node, destination_node, mtype)
                    self.gmap_edges[key] = item
                    self.scene.addItem(item)
        except Exception as e:
            print("Graph viewer: Exception in add_or_assign_edge", e)
            # QMessageBox.critical(self, "Error", f"Graph viewer: Exception in add_or_assign_edge {e}")
            return

        #if (fr, to, mtype) in self.gmap[fr].connected_edges:
        #     return
        # count only outgoing edges
        # existing_edges = [
        #     key for key in self.gmap[fr].connected_edges.keys()
        #     if key[0] == fr and key[1] == to
        # ]
        #print("Graph_viewer: add_edge", self.node_items, existing_edges, len(existing_edges), self.g.get_node(fr).name, self.g.get_node(to).name, mtype)
        # offset_index = len(existing_edges)
        #
        # graphics_edge = GraphicsEdge(fr, to, mtype, self.g, offset_index)
        # self.scene.addItem(graphics_edge)
        #
        # self.gmap[fr].connected_edges[graphics_edge.key] = graphics_edge
        # self.gmap[to].connected_edges[graphics_edge.key] = graphics_edge

    ########################################################################
    ###  DSR SIGNALS HANDLER
    ########################################################################

    # @QtCore.Slot()
    # def update_node_slot(self, id: int, mtype: str):
    #     if not id in self.gmap.keys():
    #         print("Graph viewer: Adding node", self.g.get_node(id).name)
    #         self.add_or_assign_node_SLOT(self.g.get_node(id))
    #
    # def update_node_attrs_slot(self, id: int, attribute_names: [str]):
    #     if id in self.gmap.keys():
    #         gnode = self.gmap[id]
    #         node = self.g.get_node(id)
    #         if "pos_x" in attribute_names or "pos_y" in attribute_names:
    #             x = float(node.attrs["pos_x"].value)
    #             y = float(node.attrs["pos_y"].value)
    #             gnode.setPos(x, y)
    #             for edge in gnode.connected_edges.values():
    #                 edge.update_position()
    #
    # def update_edge_slot(self, fr: int, to: int, mtype: str):
    #     if mtype != "RT":
    #         print("Graph viewer: update_edge", fr, to, mtype)
    #     if fr in self.gmap.keys() and to in self.gmap.keys():
    #         self.add_or_assign_edge_SLOT(fr, to, mtype)
    #
    # def update_edge_attrs_slot(self, fr: int, to: int, mtype: str, attribute_names: [str]):
    #     if fr in self.gmap.keys():
    #         self.gmap[fr].update_edge(to, mtype, attribute_names)

    def delete_node_slot(self, nid: int):
        print("Graph viewer: delete_node", nid)
        if nid in self.gmap.keys():
            item = self.gmap[nid]
            if item:
                for key in item.connected_edges.keys():
                    print("Graph viewer: delete_edge", nid, key, item.connected_edges[key].mtype  )
                    edge_item = item.connected_edges[key]
                    self.scene.removeItem(edge_item)
                for key in list(item.connected_edges.keys()):
                    self.gmap[key[1]].connected_edges.pop(key, None)
                self.scene.removeItem(item)
                del self.gmap[nid]

    def del_edge_slot(self, fr: int, to: int, mtype: str):
        key = (fr, to, mtype)
        if key not in self.gmap_edges.keys():
            print("Graph viewer: In delete_edge_slot -> Edge not found", key)
            return
        try:
            while len(self.gmap_edges[key]) > 0:
                edge = self.gmap_edges[key].pop(0)
                if fr in self.gmap:
                    self.gmap[fr].deleteEdge(edge)
                if to in self.gmap:
                    self.gmap[to].deleteEdge(edge)
                if edge:
                    self.scene.removeItem(edge)
                    del edge
        except Exception as e:
            print("Graph viewer: Exception in del_edge_slot", e)
            # QMessageBox.critical(self, "Error", f"Graph viewer: Exception in del_edge_slot {e}")
            return

        # if fr in self.gmap:
        #     item = self.gmap[fr].remove_edge(key, None)
        #     if item:
        #         self.scene.removeItem(item)
        # if to in self.gmap:
        #     self.gmap[to].remove_edge(key, None)


