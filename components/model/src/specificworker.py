#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 by YOUR NAME HERE
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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import pybullet as p
import time
import pybullet_data
import math
from PersonManager import PersonManager
import numpy as np
import lap
sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)






class TRoi:
    def __init__(self, xcenter, ycenter, xsize, ysize, finalxsize, finalysize):
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.xsize = xsize
        self.ysize = ysize
        self.finalxsize = finalxsize
        self.finalysize = finalysize

class TObject:
    def __init__(self, id, object_type, left, top, right, bot, score, depth, x, y, z, metrics):
        self.id = id
        self.type = object_type
        self.left = left
        self.top = top
        self.right = right
        self.bot = bot
        self.score = score
        self.depth = depth
        self.x = x
        self.y = y
        self.z = z
        self.metrics = metrics
        # Image and person attributes omitted for simplicity


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):

        # BULLET
        self.objects = []
        self.physics_client = p.connect(p.GUI)

        p.setTimeStep(1/50,0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF('plane.urdf')
        p.setGravity(0, 0, -9.81)
        self.person_manager = PersonManager(self.physics_client, p)

        self.VisualObjectsSynth = []
        self.height = 1.6

        return True

    @QtCore.Slot()
    def compute(self):
        t1 = time.time()
        try:
            t2 = time.time()
            self.visual_element = self.visualelements_proxy.getVisualObjects()

            print(time.time() * 1000 - self.visual_element.timestampgenerated)
        except:
            print("Proxy error")

        self.update_simulation(self.visual_element.objects, 1)

        #self.process_errors(self.visual_element.objects, self.VisualObjectsSynth)

        t3 = time.time()
        p.stepSimulation()
        t4 = time.time()
        # print("TIME",t2-t1,t3-t2,t4-t3,t4-t1)

        return True

    ######################### UTILS #################################
    def update_simulation(self, received_objects, k):

        received_ids = set(obj.id for obj in received_objects)

        self.person_manager.remove_disappeared_persons()

        for received_obj in received_objects:
            # Encuentra el objeto correspondiente en VisualObjectsSynth
            simulated_obj = next((obj for obj in self.VisualObjectsSynth if obj.id == received_obj.id), None)

            if simulated_obj:
                # Obtener la posición actual del objeto simulado en pybullet
                current_position = self.person_manager.get_person_position(simulated_obj.id)

                if current_position:
                    # Calcular la diferencia de posición (vector de error)
                    dx = received_obj.x / 1000 - current_position[0]
                    dy = received_obj.y / 1000 - current_position[1]
                    dz = received_obj.z / 1000 - current_position[2]
                    dz = 0

                    # print("DX/DY/DZ: ", dx, dy, dz)
                    # Calcular el vector de velocidad basado en el error y el factor k
                    k = 33
                    velocity = [k * dx, k * dy, k * dz]

                    # pose = [received_obj.x / 1000, received_obj.y / 1000, received_obj.z / 1000]
                    quaternions = self.rotation_z_to_quaternion(received_obj.person.orientation)

                    # Actualizar la posición y la velocidad del objeto simulado en pybullet
                    self.person_manager.update_person(simulated_obj.id, current_position, quaternions,
                                                 velocity)
                # Actualizar la posición del objeto simulado
                simulated_obj.x, simulated_obj.y, simulated_obj.z = received_obj.x, received_obj.y, received_obj.z
            else: #No insertado en el sistema
                person_name = received_obj.id

                # Incrementa el conteo de observaciones
                if person_name in self.person_manager.observation_count:
                    self.person_manager.observation_count[person_name] += 1
                else:
                    self.person_manager.observation_count[person_name] = 1

                # Si el objeto es una persona, no hay ninguna en pybullet y observaciones > M crea una nueva
                if not any(obj.type == "person" for obj in self.VisualObjectsSynth) and self.person_manager.observation_count[person_name] >= self.person_manager.M:
                    orientation = self.rotation_z_to_quaternion(received_obj.person.orientation)
                    person_id = self.person_manager.add_person(received_obj.id,
                                                          [received_obj.x / 1000, received_obj.y / 1000, self.height], orientation, (0.3, 0.15, self.height))
                    self.VisualObjectsSynth.append(received_obj)

        # Luego, para cualquier persona en la simulación que no esté en received_objects, aplicar la última velocidad
        for simulated_obj in self.VisualObjectsSynth:
            if simulated_obj.id not in received_ids:
                self.person_manager.apply_last_velocity(simulated_obj.id)

    ######################### UPDATE 2 #################################
    # We assume that real_objects do not have usable ids
    # there are phantom objects in the pred_synth_objects lists, marked with a flag,
    # that are not yet accepted to the model
    def process_errors(self, real_objects, pred_synth_objects):
        # compute differences between real_objects and pred_synth_objects

        # start comparing real_objects and pred_synth_objects using the Hungarian algorithm
        # using the simplest metric: 3d distance  ||(x,y,z) - (x',y',z')||
        rows = len(real_objects)
        cols = len(pred_synth_objects)
        cost_mat = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                cost_mat[i, j] = np.linalg.norm(real_objects[i].x - pred_synth_objects[j].x, real_objects[i].y - pred_synth_objects[j].y, real_objects[i].z - pred_synth_objects[j].z)
        cost, assigned_rows, assigned_cols = lap.lapjv(cost_mat, extend_cost=True, cost_limit=500) # mm away from each other
        # y is a size-cols array specifying to which real_object each synth_object is assigned or -1 if it is not assigned
        # cost_limit: double An upper limit for a cost of a single assignment.Default: `np.inf`.
        # details of lap in https: // github.com / gatagat / lap / blob / master / lap / _lapjv.pyx

        # MATCH stage
        # create a new list with the matched matched_pred_synth_objects and their corresponding real object
        # create a new list of unmatched_pred_synth_objects to be processed later
        matched_pred_synth_objects = []
        unmatched_pred_synth_objects = []
        for j in range(cols):
            if assigned_cols[j] != -1:
                matched_pred_synth_objects.append([pred_synth_objects[j], real_objects[assigned_cols[j]]])
            else:
                unmatched_pred_synth_objects.append(pred_synth_objects[j])
        # update model with the computed errors
        # if the matched object is a phantom object, add 1 to the hit counter. If the hit counter is > M, add the object to the model_insertion_list
        # create a new list of real_objects with the remaining UNMATCHED: unmatched_real_objects
        unmatched_real_objects = []
        for i in range(rows):
            if assigned_rows[i] == -1:
                unmatched_real_objects.append(real_objects[assigned_rows[i]])

        # ADD NEW OBJECTS stage
        # for each unmatched_real_object, add a new phantom object to the phantom_insertion_list
        # and initialize its hit counter to 1

        # REMOVE OLD OBJECTS stage
        # for each unmatched_pred_synth_object, if it is not a phantom object, add 1 to its miss counter.
            # If the miss counter is > M, add to the removal list
        # for each unmatched_pred_synth_object, if it is a phantom object, add object to the removal list

        # UPDATE PyBullet model processing the matched_pred_synth_objects, phantomn_insertion_list and removal_list
        pass

    def calculate_velocity(self, simulated_obj, received_obj, k):
        # Calcula la diferencia de posición en cada eje
        dx = received_obj.x - simulated_obj.x
        dy = received_obj.y - simulated_obj.y
        dz = received_obj.z - simulated_obj.z

        # Escala la diferencia por k para obtener la velocidad
        velocity_x = k * dx
        velocity_y = k * dy
        velocity_z = k * dz

        return [velocity_x, velocity_y, velocity_z]

    def rotation_z_to_quaternion(self, theta):
        # Calcular los componentes del cuaternión
        w = math.cos(theta / 2)
        z = math.sin(theta / 2)

        # El cuaternión para una rotación alrededor del eje Z
        return [0, 0, z, w]

    ######################
    # From the RoboCompVisualElements you can call this methods:
    # self.visualelements_proxy.getVisualObjects(...)
    # self.visualelements_proxy.setVisualObjects(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject
    # RoboCompVisualElements.TObjects

    ######################
    def startup_check(self):
            print(f"Testing RoboCompVisualElements.TRoi from ifaces.RoboCompVisualElements")
            test = ifaces.RoboCompVisualElements.TRoi()
            print(f"Testing RoboCompVisualElements.TObject from ifaces.RoboCompVisualElements")
            test = ifaces.RoboCompVisualElements.TObject()
            QTimer.singleShot(200, QApplication.instance().quit)