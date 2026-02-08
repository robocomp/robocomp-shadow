#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph Viewer for DSR graph visualization."""
import random
from PySide6.QtCore import Qt
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import QMenu
from pydsr import signals
from src.viewers._abstract_graphic_view import AbstractGraphicViewer
from .graph_node import GraphicsNode
from .graph_edge import GraphicsEdge
class GraphViewer(AbstractGraphicViewer):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.gmap = {}  # node_id: GraphicsNode
        self.gmap_edges = {}  # (fr, to, type): GraphicsEdge
        self.pending_edges = []  # Edges waiting for nodes to be created
        self.type_id_map = {}
        self._internal_update = False
        self.create_graph()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.contextMenu = QMenu()
        self.showMenu = self.contextMenu.addMenu("&Show:")
        self.connect_graph_signals()
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
    def connect_graph_signals(self):
        signals.connect(self.g, signals.UPDATE_NODE, self.add_or_assign_node_slot)
        signals.connect(self.g, signals.UPDATE_EDGE, self.add_or_assign_edge_slot)
        signals.connect(self.g, signals.DELETE_NODE, self.del_node_slot)
        signals.connect(self.g, signals.DELETE_EDGE, self.del_edge_slot)
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
                gnode.set_tag(node.name)
                self.scene.addItem(gnode)
                self.gmap[node.id] = gnode
                color = "coral"
                if node.attrs.__contains__("color"):
                    color = node.attrs["color"].value
                gnode.set_node_color(color)
                gnode.set_type(mtype)
            else:
                gnode = self.gmap[node.id]
        if gnode is None:
            return
        if node.attrs.__contains__("pos_x"):
            px = node.attrs["pos_x"].value
        else:
            px = random.uniform(-300, 300)
        if node.attrs.__contains__("pos_y"):
            py = node.attrs["pos_y"].value
        else:
            py = random.uniform(-300, 300)
        gnode.setPos(float(px), float(py))
        # Try to add any edges from this node that aren't already in the viewer
        print(f"Graph viewer: Node {node.name} has {len(node.edges)} edges")
        for edge in node.edges:
            key = (node.id, edge[0], edge[1])
            if key not in self.gmap_edges:
                self.add_or_assign_edge_slot(node.id, edge[0], edge[1])
        # Process pending edges that may now be drawable
        self._process_pending_edges()
    def add_or_assign_edge_slot(self, fr: int, to: int, mtype: str):
        key = (fr, to, mtype)
        if key in self.gmap_edges.keys():
            return
        try:
            edge = self.g.get_edge(fr, to, mtype)
            if edge:
                if key not in self.gmap_edges.keys():
                    fr_in_gmap = fr in self.gmap
                    to_in_gmap = to in self.gmap
                    if not fr_in_gmap or not to_in_gmap:
                        # Save as pending edge to be added when nodes are available
                        if key not in self.pending_edges:
                            self.pending_edges.append(key)
                            print(f"Graph viewer: Pending edge {mtype} added from {fr} to {to} (fr_in_gmap={fr_in_gmap}, to_in_gmap={to_in_gmap})")
                        return
                    source_node = self.gmap[fr]
                    destination_node = self.gmap[to]
                    item = GraphicsEdge(source_node, destination_node, mtype)
                    self.gmap_edges[key] = item
                    self.scene.addItem(item)
                    print(f"Graph viewer: Edge {mtype} added successfully from {fr} to {to}")
            else:
                print(f"Graph viewer: Edge {mtype} from {fr} to {to} not found in DSR")
        except Exception as e:
            print("Graph viewer: Exception in add_or_assign_edge", fr, to, mtype, e)

    def _process_pending_edges(self):
        """Try to add pending edges that were waiting for nodes to be created."""
        if self.pending_edges:
            print(f"Graph viewer: Processing {len(self.pending_edges)} pending edges")
        still_pending = []
        for key in self.pending_edges:
            fr, to, mtype = key
            fr_in_gmap = fr in self.gmap
            to_in_gmap = to in self.gmap
            if fr_in_gmap and to_in_gmap:
                # Both nodes exist now, try to add the edge
                if key not in self.gmap_edges:
                    try:
                        edge = self.g.get_edge(fr, to, mtype)
                        if edge:
                            source_node = self.gmap[fr]
                            destination_node = self.gmap[to]
                            item = GraphicsEdge(source_node, destination_node, mtype)
                            self.gmap_edges[key] = item
                            self.scene.addItem(item)
                            print(f"Graph viewer: *** RESOLVED pending edge {mtype} from {fr} to {to} ***")
                        else:
                            print(f"Graph viewer: Pending edge {mtype} from {fr} to {to} - edge not found in DSR anymore")
                            # Don't keep it pending if it's gone from DSR
                    except Exception as e:
                        print(f"Graph viewer: Exception adding pending edge: {e}")
            else:
                # Still waiting for nodes
                print(f"Graph viewer: Edge {mtype} still pending (fr={fr} in_gmap={fr_in_gmap}, to={to} in_gmap={to_in_gmap})")
                still_pending.append(key)
        self.pending_edges = still_pending
    def del_node_slot(self, id: int):
        try:
            while id in self.gmap:
                item = self.gmap[id]
                self.scene.removeItem(item)
                del item
                del self.gmap[id]
        except Exception as e:
            print(f"Error deleting node: {e}")
    def del_edge_slot(self, fr: int, to: int, mtype: str):
        key = (fr, to, mtype)
        if key not in self.gmap_edges.keys():
            return
        try:
            while key in self.gmap_edges:
                edge = self.gmap_edges.pop(key)
                if fr in self.gmap:
                    self.gmap[fr].delete_edge(edge)
                if to in self.gmap:
                    self.gmap[to].delete_edge(edge)
                if edge:
                    self.scene.removeItem(edge)
                    del edge
        except Exception as e:
            print("Graph viewer: Exception in del_edge_slot", e)
    def mousePressEvent(self, event):
        item = self.scene.itemAt(self.mapToScene(event.pos()), QtGui.QTransform())
        if item:
            super().mousePressEvent(event)
        elif event.button() == QtCore.Qt.RightButton:
            self.showContextMenu(event)
        else:
            super().mousePressEvent(event)
    def showContextMenu(self, event):
        self.contextMenu.exec()
    def toggle_animation(self, state):
        pass
