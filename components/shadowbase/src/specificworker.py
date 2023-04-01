#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 by Alejandro Torrejon Harto
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

#from lib2to3.pgen2 import driver
#from logging import exception
#from time import sleep
#from tkinter import image_names


from lib2to3.pgen2 import driver
from multiprocessing.resource_sharer import stop
from time import sleep
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

import threading

import numpy as np
import traceback

import SVD48V

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel



class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        print("Iniciando shadow")
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 10
        
        #variables de clase
        self.targetSpeed = np.array([[0.0], [0.0], [0.0]])
        self. oldTargetSpeed = np.array([[0.0], [0.0], [0.0]])
        self.driver=None
        self.joystickControl = False

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        print("Shadow iniciado correctamente")

    def __del__(self):
        print("Finalizando shadow")
        """Destructor"""
        if self.driver is not None:
            self.driver.__del__()
        print("Shadow destruido")

    def setParams(self, params):
        print("Cargando parametros:")
        try:
            self.distAxes =  float(params["distAxes"])
            self.axesLength =  float(params["axesLength"])
            port = params["port"]
            self.maxLinSpeed = float(params["maxLinSpeed"])
            self.maxRotSpeed = float(params["maxRotSpeed"])
            maxWheelSpeed = float(params["maxWheelSpeed"])
            maxAcceleration = float(params["maxAcceleration"])
            maxDeceleration = float(params["maxDeceleration"])
            idDrivers = [int(params["idDriver1"]), int(params["idDriver2"])]
            wheelRadius = float(params["wheelRadius"])

            self.driver = SVD48V.SVD48V(port, idDrivers, wheelRadius, maxWheelSpeed,
                                        maxAcceleration, maxDeceleration)  

            if not self.driver.still_alive():
                print("NO se conecto al driver o fallo uno de ellos , cerrando programa")
                exit(-1)


            print("creando matriz de conversion")
            ll = 0.5*(self.distAxes + self.axesLength)

            ''''MATRIZ DE CONVESION'''
            self.m_wheels = np.array([  [-1.0,  -1.0, -ll],
                                        [1.0,  -1.0, ll], 
                                        [1.0,  -1.0, -ll],
                                        [ -1.0,  -1.0, ll]])
            #self.m_wheels_inv = np.linalg.inv(self.m_wheels)

            print(self.m_wheels)
           #print(self.m_wheels_inv)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.driver.show_params)
            #self.timer.start(0.5)

        except Exception as e:
            print("Error reading config params or start motor")
            print(e)
        return True

    def setAdvx(self, val):
        if val>self.maxLinSpeed:  
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA",self.targetSpeed[0],"CUNADO MAXIMA ES", self.maxLinSpeed)
            self.targetSpeed[0] = self.maxLinSpeed
        else:  self.targetSpeed[0] = val

    def setAdvz(self, val):
        if val>self.maxLinSpeed: 
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA",self.targetSpeed[1],"CUNADO MAXIMA ES", self.maxLinSpeed) 
            self.targetSpeed[1] = self.maxLinSpeed
        else:  self.targetSpeed[1] = val

    def setRot(self, val):
        if val>self.maxRotSpeed:  
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA", self.targetSpeed[2],"CUNADO MAXIMA ES", self.maxRotSpeed)
            self.targetSpeed[2] = self.maxRotSpeed     
        else:  self.targetSpeed[2] = val

    #######################################COMPUTE###########################################
    @QtCore.Slot()
    def compute(self):
        if self.driver != None:
            if  not np.array_equal(self.targetSpeed, self.oldTargetSpeed):
                speeds = self.m_wheels@self.targetSpeed
                #print("mm/s",speeds)
                self.driver.set_speed(speeds)
                self.oldTargetSpeed = np.copy(self.targetSpeed)
                #print("Modificamos velocidades: ", self.oldTargetSpeed)

            #self.driver.show_params(False)
        return True

    def startup_check(self):
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompJoystickAdapter.AxisParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.AxisParams()
        print(f"Testing RoboCompJoystickAdapter.ButtonParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.ButtonParams()
        print(f"Testing RoboCompJoystickAdapter.TData from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.TData()
        QTimer.singleShot(200, QApplication.instance().quit)



    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
   
    # IMPLEMENTATION of correctOdometer method from OmniRobot interface
    #
    def OmniRobot_correctOdometer(self, x, z, alpha):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of getBasePose method from OmniRobot interface
    #
    def OmniRobot_getBasePose(self):
    
        x=0
        z=0
        alpha=0

        return [x, z, alpha]
    #
    # IMPLEMENTATION of getBaseState method from OmniRobot interface
    #
    def OmniRobot_getBaseState(self):
    
        #
        # write your CODE here
        #
        state = RoboCompGenericBase.TBaseState()
        return state
    #
    # IMPLEMENTATION of resetOdometer method from OmniRobot interface
    #
    def OmniRobot_resetOdometer(self):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setOdometer method from OmniRobot interface
    #
    def OmniRobot_setOdometer(self, state):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setOdometerPose method from OmniRobot interface
    #
    def OmniRobot_setOdometerPose(self, x, z, alpha):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setSpeedBase method from OmniRobot interface
    #
    def OmniRobot_setSpeedBase(self, advx, advz, rot):
        self.setAdvx(advx)
        self.setAdvz(advz)
        self.setRot(rot)



    #
    # IMPLEMENTATION of stopBase method from OmniRobot interface
    #
    def OmniRobot_stopBase(self):
        print("///////////////////////PARADA DE EMERGENCIA////////////////////")
        self.setAdvx(0.0)
        self.setAdvz(0.0)
        self.setRot(0.0)


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompJoystickAdapter you can use this types:
    # RoboCompJoystickAdapter.AxisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to sendData method from JoystickAdapter interface
    #
    def JoystickAdapter_sendData(self, data):
    
        #print(data)
        for b in data.buttons:
            if b.name == "stop":
                if b.step == 1:
                    self.OmniRobot_stopBase()
                    self.joystickControl = False
            elif  b.name == "block":
                if b.step == 1:
                    self.driver.stop_driver()
                    self.joystickControl = False
            elif b.name == "joystick_control":
                if b.step == 1:
                    self.joystickControl = not self.joystickControl
            else:
                print(b.name, "PULASDOR NO AJUSTADO")
            
            if self.joystickControl:
                for a in  data.axes:
                    if a.name == "rotate":
                        self.setRot(a.value)
                    elif  a.name == "advance":
                        self.setAdvz(a.value)
                    elif a.name == "side":
                        self.setAdvx(a.value)
                    else:
                        print(a.name, "VELODIDAD NO AJUSTADA")

        

        #self.targetSpeed = [data.axes[0], data.axes[1], data.axes[2]]
        


