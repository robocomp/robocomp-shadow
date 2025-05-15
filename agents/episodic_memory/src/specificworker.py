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

from PySide6.QtCore import QTimer, QMutex, QMutexLocker
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import deque
import numpy as np
import os


console = Console(highlight=False)
from pydsr import *

DICT2GRAPH = np.array([['pose_x', 'pose_y', 'pose_z'],
                ['rot_x', 'rot_y',  'rot_z'],
                ['lin_vel_x', 'lin_vel_y', 'lin_vel_z'],
                ['rot_vel_x', 'rot_vel_y', 'rot_vel_z'],
                ['lin_acc_x', 'lin_acc_y', 'lin_acc_z'],
                ['rot_acc_x', 'rot_acc_y', 'rot_acc_z']])

# Configuración
BUFFER_SIZE = 1000  # Guardar cada 1000 registros


def process_RT_data(edge_type, edge):
    attrs = edge.attrs
    get = lambda name: np.array(attrs[name].value, dtype=np.float32) if name in attrs else [None, None, None]
    
    pose = get("rt_translation")
    rot = get("rt_rotation_euler_xyz")
    lin_vel = get("rt_translation_velocity")
    rot_vel = get("rt_rotation_euler_xyz_velocity")
    lin_acc = get("rt_translation_acceleration")
    rot_acc = get("rt_rotation_euler_xyz_acceleration")
    ts = get("rt_timestamps")

    return {
        'type': edge_type,
        'timestamp': ts[0],
        'pose_x': pose[0],
        'pose_y': pose[1],
        'pose_z': pose[2],
        'rot_x': rot[0],
        'rot_y': rot[1],
        'rot_z': rot[2],
        'lin_vel_x': lin_vel[0],
        'lin_vel_y': lin_vel[1],
        'lin_vel_z': lin_vel[2],
        'rot_vel_x': rot_vel[0],
        'rot_vel_y': rot_vel[1],
        'rot_vel_z': rot_vel[2],
        'lin_acc_x': lin_acc[0],
        'lin_acc_y': lin_acc[1],
        'lin_acc_z': lin_acc[2],
        'rot_acc_x': rot_acc[0],
        'rot_acc_y': rot_acc[1],
        'rot_acc_z': rot_acc[2],
    }




############################################## SPECIFICWORKER ############################################
class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        # Configuración de tiempos (en milisegundos)
        self.plot_interval = 250  # Actualizar gráfico 
        self.save_interval = 60000  # Guardar datos cada 1 minuto
        
        # Timers independientes
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot_from_buffer)
        self.plot_timer.start(self.plot_interval)
        
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.save_data_from_buffer)
        self.save_timer.start(self.save_interval)

        # Estructuras de datos (buffer circular)
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = QMutex()  # Para acceso thread-safe

        rows, cols = DICT2GRAPH.shape  # rows=6, cols=3

        
        # Configuración correcta: nrows=6, ncols=3
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
        self.ax = np.atleast_2d(self.ax)  # Asegurar que siempre es 2D
        
        # Configuración común
        plt.ion()
        plt.tight_layout()
        for i in range(DICT2GRAPH.shape[0]):
            for j in range(DICT2GRAPH.shape[1]):
                self.ax[i,j].set_ylim(bottom=-5, top=5)
                self.ax[i,j].set_autoscale_on(False) 
        
        

        self.agent_id = 409
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        try:
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        self.save_data_from_buffer()
        self.plot_timer.stop()
        self.save_timer.stop()

    @QtCore.Slot()
    def compute(self):
        print('SpecificWorker.compute...')

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def update_plot(self, df):
        """Actualiza la matriz de gráficos"""
        for i in range(DICT2GRAPH.shape[0]):
            for j in range(DICT2GRAPH.shape[1]):

                self.ax[i,j].clear()
                metric = DICT2GRAPH[i,j]
                
                # Filtra y plotea datos
                for tipo, color in [('VRT', 'r'), ('RT', 'b')]:
                    subset = df[df['type'] == tipo]
                    if not subset.empty:
                        self.ax[i,j].plot(subset['timestamp'], subset[metric], color, label=tipo)
                
                self.ax[i,j].set_title(metric)
                self.ax[i,j].legend()
                self.ax[i,j].grid(True)
        
        plt.pause(0.01)

    def save_to_json(self, df, filename='datos_tiempo_real.json'):
        """Guarda de forma segura con bloqueo"""
        try:
            if not df.empty:
                # Modo append seguro con bloqueo de archivo
                with open(filename, 'a') as f:
                    for record in df.to_dict('records'):
                        f.write(json.dumps(record) + '\n')
                print(f"Datos guardados: {filename}")
        except Exception as e:
            print(f"Error al guardar: {e}")

    def update_plot_from_buffer(self):
        """Actualiza gráficos con los datos del buffer"""
        try:
            # Copiar datos del buffer de forma segura
            with QMutexLocker(self.buffer_lock):
                if not self.data_buffer:
                    return
                df = pd.DataFrame(self.data_buffer)
            
            # Actualizar gráficos
            self.update_plot(df)
            
        except Exception as e:
            print(f"Error actualizando gráficos: {e}")

    @QtCore.Slot()
    def save_data_from_buffer(self):
        """Guarda los datos del buffer y lo limpia"""
        try:
            # Copiar y limpiar buffer de forma segura
            with QMutexLocker(self.buffer_lock):
                if not self.data_buffer:
                    return
                    
                df = pd.DataFrame(self.data_buffer)
                self.data_buffer.clear()  # Limpiar después de copiar
            name = "save/" + datetime.now().strftime('%H:%M:%S') + ".json"
            # Guardar datos
            self.save_to_json(df, name)
            print(f"Datos guardados: {name}")
            
        except Exception as e:
            print(f"Error guardando datos: {e}")

    # =============== DSR SLOTS  ================
    # =============================================
    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        """Procesamiento de datos entrantes - Solo añade al buffer"""
        if fr == 100 and to == 200 and type in ["RT", "VRT"]:
            try:
                edge = self.g.get_edge(fr, to, type)
                if edge is not None:
                    data = process_RT_data(type, edge)
                    
                    # Añadir al buffer de forma thread-safe
                    with QMutexLocker(self.buffer_lock):
                        self.data_buffer.append(data)
                    
            except Exception as e:
                print(f"Error procesando edge: {e}")



