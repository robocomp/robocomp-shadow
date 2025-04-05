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
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, Transform3d, euler_angles_to_matrix
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

        self.viewer_2d = self.dsr_viewer.widgets_by_type[View.scene].widget
        self.dsr_viewer.docks["2D"].setWindowTitle("Residuals")

        # custom widget
        self.affordance_viewer = Affordances(self.g)
        self.master = self.dsr_viewer.add_custom_widget_to_dock("Master", self.affordance_viewer)
        self.dsr_viewer.window.tabifyDockWidget(self.dsr_viewer.docks["Master"], self.dsr_viewer.docks["2D"])
        self.dsr_viewer.docks["Master"].raise_()  # Forzar que el dock "Master" tenga el foco

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'dsr_viewer'):
            del self.dsr_viewer

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        lidar_data = []
        try:
            lidar_data = self.lidar3d_proxy.getLidarDataWithThreshold2d("helios", 5000, 10)
        except Exception as e:
            print(e)

        if self.there_is_room():
            residuals = self.room_residuals(lidar_data.points)
            #residuals = self.fridge_residuals(lidar_data.points)
            self.draw_residuals(self.viewer_2d, residuals)
        else:
            self.draw_lidar_data_local(self.viewer_2d, lidar_data.points)

    # =============== WORK  ================
    def there_is_room(self):
        edges = self.g.get_edges_by_type("current")
        for e in edges:
            n = self.g.get_node(e.origin)
            if n is not None and e.origin == e.destination and n.type == "room":  # self current edge
                return True
        return False

    def room_residuals(self, points: list):
        # draw room
        self.draw_room()

        # move points to room frame using Pytorch3D
        points_list = [[p.x, p.y, p.z] for p in points]
        # Convert the list to a PyTorch tensor
        points_tensor = torch.tensor(points_list)

        # Transform the corners to the current model scale, position and orientation
        t = (Transform3d(device="cuda").scale(self.w, self.d, self.h)
             .rotate(euler_angles_to_matrix(torch.stack([self.a, self.b, self.c]), "ZYX"))
             .translate(self.x, self.y, self.z))
        points_in_room = t.transform_points(points_list)

        # filter points close to the walls
        residuals = self.remove_explained_points(points_in_room)

        return residuals

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

    def remove_explained_points(self, points: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """
        Removes points whose distance to any face in the mesh is less than the threshold.
        Returns the points not explained by the mesh (distance > threshold).
        """
        mesh = self.forward()
        verts = mesh.verts_packed()  # (V, 3)
        faces = mesh.faces_packed()  # (F, 3)
        tris = verts[faces]  # (F, 3, 3)

        # Expand points to (P, F, 3) and tris to (P, F, 3, 3)
        P = points.shape[0]
        F = tris.shape[0]
        points_exp = points[:, None, :].expand(P, F, 3)
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]

        # Compute vectors
        v0v1 = v1 - v0  # (F, 3)
        v0v2 = v2 - v0  # (F, 3)
        pvec = points_exp - v0  # (P, F, 3)

        # Compute dot products
        d00 = (v0v1 * v0v1).sum(-1)  # (F,)
        d01 = (v0v1 * v0v2).sum(-1)
        d11 = (v0v2 * v0v2).sum(-1)
        d20 = (pvec * v0v1[None, :, :]).sum(-1)  # (P, F)
        d21 = (pvec * v0v2[None, :, :]).sum(-1)  # (P, F)

        denom = d00 * d11 - d01 * d01 + 1e-8  # (F,)
        v = (d11 * d20 - d01 * d21) / denom  # (P, F)
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        inside = (u >= 0) & (v >= 0) & (w >= 0)  # (P, F)

        # Distance to plane of triangle
        n = torch.cross(v0v1, v0v2)  # (F, 3)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
        plane_dists = torch.abs((pvec * n[None, :, :]).sum(-1))  # (P, F)

        # Squared Euclidean distance to closest vertex if outside triangle
        dist_v0 = ((points[:, None, :] - v0[None, :, :]) ** 2).sum(-1)
        dist_v1 = ((points[:, None, :] - v1[None, :, :]) ** 2).sum(-1)
        dist_v2 = ((points[:, None, :] - v2[None, :, :]) ** 2).sum(-1)
        corner_dists = torch.min(torch.stack([dist_v0, dist_v1, dist_v2], dim=-1), dim=-1).values  # (P, F)

        # Use plane distance if inside triangle, corner distance otherwise
        total_dists = torch.where(inside, plane_dists, torch.sqrt(corner_dists))  # (P, F)

        # Take min across triangles
        min_dists, _ = total_dists.min(dim=1)  # (P,)

        # Keep only unexplained points
        keep_mask = min_dists > threshold
        return points[keep_mask]

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

