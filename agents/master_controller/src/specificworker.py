#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem
from rich.console import Console
from genericworker import *
import interfaces as ifaces
from pydsr import *
from dsr_gui import DSRViewer, View
from ui_masterUI import Ui_master
from affordances import Affordances
from PySide6 import QtCore, QtWidgets
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.lidar_draw_items = []    # to store the items to be drawn
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 66
        self.g = DSRGraph(0, "master_controller", self.agent_id)
        self.dsr_viewer = DSRViewer(self, self.g, View.graph + View.scene, View.graph)
        self.dsr_viewer.window.resize(1000, 600)

        # connect signals to slots in graph viewer
        graph_viewer = self.dsr_viewer.widgets_by_type[View.graph].widget
        try:
            signals.connect(self.g, signals.UPDATE_NODE, graph_viewer.add_or_assign_node_slot)
            signals.connect(self.g, signals.UPDATE_EDGE, graph_viewer.add_or_assign_edge_slot)
            signals.connect(self.g, signals.DELETE_NODE, graph_viewer.delete_node_slot)
            signals.connect(self.g, signals.DELETE_EDGE, graph_viewer.del_edge_slot)
            #signals.connect(self.g, signals.UPDATE_NODE_ATTR, graph_viewer.update_node_attrs_slot)
            #signals.connect(self.g, signals.UPDATE_EDGE_ATTR, graph_viewer.update_edge_attrs_slot)
        except Exception as e:
            print(e)

        self.viewer_2d = self.dsr_viewer.widgets_by_type[View.scene].widget
        self.dsr_viewer.docks["2D"].setWindowTitle("Residuals")

        # custom widget
        self.affordance_viewer = Affordances(self.g)
        self.master = self.dsr_viewer.add_custom_widget_to_dock("Master", self.affordance_viewer)
        self.dsr_viewer.window.tabifyDockWidget(self.dsr_viewer.docks["Master"], self.dsr_viewer.docks["2D"])
        self.dsr_viewer.docks["Master"].raise_()  # Forzar que el dock "Master" tenga el foco
        self.affordance_viewer.setFocus()
        self.affordance_viewer.populate()
        self.affordance_viewer.tree.expandAll()
        # connect signals to slots in affordance viewer
        signals.connect(self.g, signals.UPDATE_NODE, self.affordance_viewer.on_graph_update_node)
        signals.connect(self.g, signals.UPDATE_EDGE, self.affordance_viewer.on_graph_update_edge)
        signals.connect(self.g, signals.DELETE_NODE, self.affordance_viewer.on_graph_delete_node)
        signals.connect(self.g, signals.DELETE_EDGE, self.affordance_viewer.on_graph_delete_edge)
        signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.affordance_viewer.on_graph_update_node_attrs)
        signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.affordance_viewer.on_graph_update_edge_attrs)


        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        try:
            lidar_data = self.lidar3d_proxy.getLidarDataWithThreshold2d("helios", 5000, 10)
        except Exception as e:
            print(e)
        residuals = self.room_residuals(lidar_data.points)
        #residuals = self.fridge_residuals(lidar_data.points)

        self.draw_lidar_data_local(self.viewer_2d, residuals)

    # =============== WORK  ================
    def room_residuals(self, points: list):
        # if room in Graph, get room
        room = self.g.get_nodes_by_type("room")
        if room:
            # draw room

            # move points to room frame using Pytorch3D

            # filter points close to the walls
            # return residuals
            pass
        return points

    def fridge_residuals(self, points):
        # if fridge in Graph, get fridge
        fridge = self.g.get_nodes_by_type("fridge")
        if fridge:
            # move points to fridge frame

            # filter points close to the walls
            # return residuals
            pass
        return points

    def draw_lidar_data_local(self, viewer, points):
        # Clear previous items
        for item in self.lidar_draw_items:
            viewer.scene.removeItem(item)
            del item
        self.lidar_draw_items.clear()
        # Draw current Lidar points
        color = QColor(0, 255, 0)
        pen = QPen(color, 20)
        for i in range(0, len(points)):
            ellipse = QGraphicsEllipseItem(-20, -20, 40, 40)
            ellipse.setPen(pen)
            ellipse.setBrush(color)
            ellipse.setPos(points[i].x, points[i].y)
            viewer.scene.addItem(ellipse)
            self.lidar_draw_items.append(ellipse)



    # =============== AUX  ================
    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # # =============== DSR SLOTS  ================
    # # =============================================
    #
    # def update_node_att(self, id: int, attribute_names: [str]):
    #     #console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')
    #     pass
    #
    # def update_node(self, id: int, mtype: str):
    #     #console.print(f"UPDATE NODE: {id} {mtype}", style='green')
    #     pass
    #
    # def delete_node(self, id: int):
    #     #console.print(f"DELETE NODE:: {id} ", style='green')
    #     pass
    #
    # def update_edge(self, fr: int, to: int, mtype: str):
    #     #console.print(f"UPDATE EDGE: {fr} to {mtype}", mtype, style='green')
    #     pass
    #
    # def update_edge_att(self, fr: int, to: int, mtype: str, attribute_names: [str]):
    #     # if "state" in attribute_names:
    #     #     console.print(f"UPDATE EDGE ATT: {fr} to {mtype} {attribute_names}", style='green')
    #     pass
    #
    # def delete_edge(self, fr: int, to: int, mtype: str):
    #     #console.print(f"DELETE EDGE: {fr} to {mtype}", style='green')
    #     pass
