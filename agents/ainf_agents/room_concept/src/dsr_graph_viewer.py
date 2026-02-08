#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSR Graph Viewer using DearPyGui

Visualizes the DSR graph showing nodes and edges.
Updates periodically by querying the G object.
Right-click on a node or edge to inspect its attributes.
"""

import dearpygui.dearpygui as dpg
import time
import math
from typing import Dict, List, Tuple, Optional


class DSRGraphViewerDPG:
    """
    DearPyGui-based DSR graph viewer.
    Periodically reads G and displays nodes and edges.
    """

    def __init__(self, g, window_width=380, window_height=380, update_period_ms=500, canvas_tag="dsr_canvas"):
        self.g = g
        self.canvas_width = window_width
        self.canvas_height = window_height
        self.update_period = update_period_ms / 1000.0
        self.canvas_tag = canvas_tag

        self.last_update_time = 0

        # Node colors (RGB, 0-255)
        self.node_colors = {
            'root': (255, 100, 100),
            'room': (100, 200, 100),
            'robot': (100, 100, 255),
            'shadow': (100, 150, 255),
            'default': (200, 200, 200)
        }

        # Edge colors
        self.edge_colors = {
            'RT': (220, 220, 220),
            'default': (200, 200, 200)
        }

        # Node positions (computed layout)
        self.node_positions = {}

        # Hit-test regions updated each frame (plain Python data only — no pydsr objects)
        # List of (cx, cy, radius, node_data_dict) for nodes
        self._node_hit_regions: List[Tuple[float, float, float, dict]] = []
        # List of (mx, my, edge_data_dict) for edges
        self._edge_hit_regions: List[Tuple[float, float, dict]] = []

        # Popup management
        self._popup_tag = "dsr_inspector_popup"
        self._popup_content_tag = "dsr_popup_content"
        # What the popup is tracking: None, ('node', node_name), or ('edge', origin_name, dest_name, edge_type)
        self._popup_tracking = None
        # Debounce: was RMB down last frame?
        self._rmb_was_down = False

    # ---- public ---------------------------------------------------------

    def update(self):
        """Called from room viewer's render loop."""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_period:
            # Still check for clicks every frame even if we don't redraw the graph
            self._poll_right_click()
            return
        self.last_update_time = current_time

        self._poll_right_click()
        self._update_graph()

    # ---- mouse handling (polled, no callbacks) --------------------------

    def _poll_right_click(self):
        """Check for right-click by polling mouse state each frame."""
        rmb_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Right)

        # Detect rising edge (press, not hold)
        if rmb_down and not self._rmb_was_down:
            self._handle_click()
        self._rmb_was_down = rmb_down

    def _handle_click(self):
        """Process a single right-click: hit-test and open popup."""
        try:
            if not dpg.does_item_exist(self.canvas_tag):
                return
            mouse_pos = dpg.get_mouse_pos(local=False)
            canvas_min = dpg.get_item_rect_min(self.canvas_tag)
        except Exception:
            return

        local_x = mouse_pos[0] - canvas_min[0]
        local_y = mouse_pos[1] - canvas_min[1]

        if local_x < 0 or local_y < 0 or local_x > self.canvas_width or local_y > self.canvas_height:
            return

        # Hit-test nodes first (priority over edges)
        for cx, cy, radius, node_data in self._node_hit_regions:
            if math.hypot(local_x - cx, local_y - cy) <= radius:
                self._show_node_popup(node_data, mouse_pos)
                return

        # Hit-test edges (within 18px of midpoint)
        for mx, my, edge_data in self._edge_hit_regions:
            if math.hypot(local_x - mx, local_y - my) <= 18:
                self._show_edge_popup(edge_data, mouse_pos)
                return

    # ---- popups ---------------------------------------------------------

    def _close_popup(self):
        self._popup_tracking = None
        if dpg.does_item_exist(self._popup_tag):
            dpg.delete_item(self._popup_tag)

    def _show_node_popup(self, node_data: dict, screen_pos):
        """Show node attributes from pre-cached plain Python data, auto-refreshed."""
        self._close_popup()
        self._popup_tracking = ('node', node_data['name'])

        with dpg.window(label=f"Node: {node_data['name']}", tag=self._popup_tag,
                        width=300, height=280, pos=screen_pos,
                        no_resize=True, no_collapse=True,
                        on_close=lambda: self._close_popup()):
            dpg.add_group(tag=self._popup_content_tag)

        self._fill_node_popup(node_data)

    def _show_edge_popup(self, edge_data: dict, screen_pos):
        """Show edge attributes from pre-cached plain Python data, auto-refreshed."""
        self._close_popup()
        self._popup_tracking = ('edge', edge_data['origin_name'], edge_data['dest_name'], edge_data['type'])

        with dpg.window(label=f"Edge: {edge_data['type']}", tag=self._popup_tag,
                        width=320, height=250, pos=screen_pos,
                        no_resize=True, no_collapse=True,
                        on_close=lambda: self._close_popup()):
            dpg.add_group(tag=self._popup_content_tag)

        self._fill_edge_popup(edge_data)

    def _fill_node_popup(self, node_data: dict):
        """Fill (or refill) the popup content group with node data."""
        tag = self._popup_content_tag
        if not dpg.does_item_exist(tag):
            return
        dpg.delete_item(tag, children_only=True)
        dpg.add_text(f"id: {node_data['id']}", color=(180, 180, 180), parent=tag)
        dpg.add_text(f"type: {node_data['type']}", color=(180, 180, 180), parent=tag)
        dpg.add_separator(parent=tag)
        dpg.add_text("Attributes", color=(255, 255, 0), parent=tag)
        if node_data['attrs']:
            for attr_name, attr_str in node_data['attrs'].items():
                dpg.add_text(f"  {attr_name} = {attr_str}", color=(200, 200, 200), parent=tag)
        else:
            dpg.add_text("  (none)", color=(120, 120, 120), parent=tag)

    def _fill_edge_popup(self, edge_data: dict):
        """Fill (or refill) the popup content group with edge data."""
        tag = self._popup_content_tag
        if not dpg.does_item_exist(tag):
            return
        dpg.delete_item(tag, children_only=True)
        dpg.add_text(f"{edge_data['origin_name']} ({edge_data['origin_id']})  -->  "
                     f"{edge_data['dest_name']} ({edge_data['dest_id']})",
                     color=(180, 180, 180), parent=tag)
        dpg.add_text(f"type: {edge_data['type']}", color=(180, 180, 180), parent=tag)
        dpg.add_separator(parent=tag)
        dpg.add_text("Attributes", color=(255, 255, 0), parent=tag)
        if edge_data['attrs']:
            for attr_name, attr_val in edge_data['attrs'].items():
                if isinstance(attr_val, list):
                    # Multi-line attribute (e.g. 3x3 matrix)
                    dpg.add_text(f"  {attr_name}:", color=(200, 200, 200), parent=tag)
                    for row in attr_val:
                        dpg.add_text(row, color=(200, 200, 200), parent=tag)
                else:
                    dpg.add_text(f"  {attr_name} = {attr_val}", color=(200, 200, 200), parent=tag)
        else:
            dpg.add_text("  (none)", color=(120, 120, 120), parent=tag)

    def _refresh_popup(self):
        """Refresh the open popup with the latest cached data (all plain Python, no G calls)."""
        if self._popup_tracking is None:
            return
        if not dpg.does_item_exist(self._popup_tag):
            self._popup_tracking = None
            return

        kind = self._popup_tracking[0]
        if kind == 'node':
            node_name = self._popup_tracking[1]
            for _, _, _, node_data in self._node_hit_regions:
                if node_data['name'] == node_name:
                    self._fill_node_popup(node_data)
                    return
        elif kind == 'edge':
            _, origin_name, dest_name, edge_type = self._popup_tracking
            for _, _, edge_data in self._edge_hit_regions:
                if (edge_data['origin_name'] == origin_name and
                        edge_data['dest_name'] == dest_name and
                        edge_data['type'] == edge_type):
                    self._fill_edge_popup(edge_data)
                    return

    @staticmethod
    def _fmt(val) -> str:
        """Format an attribute value for display."""
        if isinstance(val, float):
            return f"{val:.3f}"
        if hasattr(val, '__iter__') and not isinstance(val, str):
            try:
                return "[" + ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                       for v in val) + "]"
            except Exception:
                return str(val)
        return str(val)

    @staticmethod
    def _fmt_matrix3x3(vals) -> list:
        """Format a flat 9-element list as 3 row strings for 3x3 matrix display."""
        try:
            v = [float(x) for x in vals]
            if len(v) == 9:
                return [
                    f"  [{v[0]:10.5f} {v[1]:10.5f} {v[2]:10.5f}]",
                    f"  [{v[3]:10.5f} {v[4]:10.5f} {v[5]:10.5f}]",
                    f"  [{v[6]:10.5f} {v[7]:10.5f} {v[8]:10.5f}]",
                ]
        except Exception:
            pass
        return [str(vals)]

    # ---- graph drawing --------------------------------------------------

    def _update_graph(self):
        """Read G and update the visualization"""
        try:
            if not dpg.does_item_exist(self.canvas_tag):
                return

            dpg.delete_item(self.canvas_tag, children_only=True)

            # Clear hit regions
            self._node_hit_regions.clear()
            self._edge_hit_regions.clear()

            nodes = self._get_all_nodes()
            if not nodes:
                dpg.draw_text((10, 10), "No nodes in graph yet",
                              parent=self.canvas_tag, color=(255, 255, 0))
                return

            node_map = {}
            for node in nodes:
                nid = self._get_node_id(node)
                if nid is not None:
                    node_map[nid] = node

            self._compute_layout(nodes)
            edges = self._get_all_edges(nodes)

            for edge_tuple in edges:
                self._draw_edge(edge_tuple, node_map)
            for node in nodes:
                self._draw_node(node)

            # Refresh open popup with latest cached data
            self._refresh_popup()

        except Exception as e:
            print(f"[DSRGraphViewer] Error updating graph: {e}")

    def _get_all_nodes(self):
        try:
            nodes = self.g.get_nodes()
            return nodes if nodes else []
        except Exception as e:
            print(f"[DSRGraphViewer] Error getting nodes: {e}")
            return []

    def _get_all_edges(self, nodes):
        try:
            all_edges = []
            for node in nodes:
                node_id = self._get_node_id(node)
                if node_id is None:
                    continue
                if node.edges:
                    for edge_key, edge_obj in node.edges.items():
                        all_edges.append((edge_obj.origin, edge_obj.destination,
                                          edge_obj.type, edge_obj))
            return all_edges
        except Exception as e:
            print(f"[DSRGraphViewer] Error getting edges: {e}")
            return []

    def _compute_layout(self, nodes):
        center_x = self.canvas_width / 2
        center_y = self.canvas_height / 2
        radius = min(self.canvas_width, self.canvas_height) * 0.3

        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / num_nodes
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.node_positions[self._get_node_name(node)] = (x, y)

    def _draw_node(self, node):
        try:
            node_name = self._get_node_name(node)
            node_type = self._get_node_type(node)
            if node_name not in self.node_positions:
                return

            x, y = self.node_positions[node_name]
            color = self.node_colors.get(node_type, self.node_colors['default'])
            radius = 15

            dpg.draw_circle((x, y), radius, color=color, fill=color,
                            parent=self.canvas_tag)
            dpg.draw_circle((x, y), radius, color=(255, 255, 255), thickness=2,
                            parent=self.canvas_tag)
            label = f"{node_name}\n({node_type})"
            dpg.draw_text((x - 20, y - 8), label,
                          parent=self.canvas_tag, color=(255, 255, 0), size=12)

            # Cache node data as plain Python dict (no pydsr references)
            node_data = self._extract_node_data(node)
            self._node_hit_regions.append((x, y, radius, node_data))

        except Exception as e:
            print(f"[DSRGraphViewer] Error drawing node: {e}")

    def _draw_edge(self, edge_tuple, node_map):
        try:
            origin_id, dest_id, edge_type, edge_obj = edge_tuple

            origin_node = node_map.get(origin_id)
            dest_node = node_map.get(dest_id)
            if not origin_node or not dest_node:
                return

            origin_name = self._get_node_name(origin_node)
            dest_name = self._get_node_name(dest_node)
            if origin_name not in self.node_positions or dest_name not in self.node_positions:
                return

            x1, y1 = self.node_positions[origin_name]
            x2, y2 = self.node_positions[dest_name]
            color = self.edge_colors.get(edge_type, self.edge_colors['default'])

            dpg.draw_line((x1, y1), (x2, y2), color=color, thickness=4,
                          parent=self.canvas_tag)

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dpg.draw_text((mid_x, mid_y), edge_type,
                          parent=self.canvas_tag, color=(0, 255, 255), size=12)

            # Cache edge data as plain Python dict (no pydsr references)
            edge_data = self._extract_edge_data(edge_obj, origin_name, dest_name)
            self._edge_hit_regions.append((mid_x, mid_y, edge_data))

        except Exception as e:
            print(f"[DSRGraphViewer] Error drawing edge: {e}")

    # ---- data extraction (pydsr → plain Python) ---------------------------

    # Short display names for common DSR attribute keys
    _SHORT_NAMES = {
        'rt_translation': 'trans',
        'rt_rotation_euler_xyz': 'rot',
        'rt_se2_covariance': 'cov',
        'robot_ref_adv_speed': 'adv_speed',
        'robot_ref_rot_speed': 'rot_speed',
        'room_width': 'width',
        'room_length': 'length',
    }

    @classmethod
    def _short(cls, attr_name: str) -> str:
        return cls._SHORT_NAMES.get(attr_name, attr_name)

    def _extract_node_data(self, node) -> dict:
        """Extract all node info into a plain Python dict during the G-safe draw pass."""
        data = {
            'name': self._get_node_name(node),
            'type': self._get_node_type(node),
            'id': self._get_node_id(node),
            'attrs': {}
        }
        try:
            if node.attrs:
                for attr_name, attr in node.attrs.items():
                    data['attrs'][self._short(str(attr_name))] = self._fmt(attr.value)
        except Exception:
            pass
        return data

    def _extract_edge_data(self, edge_obj, origin_name: str, dest_name: str) -> dict:
        """Extract all edge info into a plain Python dict during the G-safe draw pass."""
        data = {
            'type': str(edge_obj.type),
            'origin_id': edge_obj.origin,
            'dest_id': edge_obj.destination,
            'origin_name': origin_name,
            'dest_name': dest_name,
            'attrs': {}
        }
        try:
            if edge_obj.attrs:
                for attr_name, attr in edge_obj.attrs.items():
                    key = self._short(str(attr_name))
                    if str(attr_name) == 'rt_se2_covariance':
                        data['attrs'][key] = self._fmt_matrix3x3(attr.value)
                    else:
                        data['attrs'][key] = self._fmt(attr.value)
        except Exception:
            pass
        return data

    # Helper methods
    def _get_node_name(self, node):
        try:
            return node.name
        except:
            return "unknown"

    def _get_node_type(self, node):
        try:
            return node.type
        except:
            return "default"

    def _get_node_id(self, node):
        try:
            return node.id
        except:
            return None
