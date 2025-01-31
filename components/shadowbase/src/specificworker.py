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

from multiprocessing.resource_sharer import stop
import time
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import SVD48V

console = Console(highlight=False)

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        print("Iniciando shadow")
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 5
        
        #variables de clase
        self.targetSpeed = np.array([[0.0], [0.0], [0.0]])
        self. oldTargetSpeed = np.array([[0.0], [0.0], [0.0]])
        self.driver=None
        self.joystickControl = False
        self.time_disble = time.time()
        self.time_emergency = time.time()
        self.time_move = time.time()

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        print("Shadow iniciado correctamente")

    ####################################################### IMPORTANT ################################################################################
    #please in the shadowbase.py file the handler will call the __del__ 
    #SIGNALS handler
    #def sigint_handler(*args):
    #    QtCore.QCoreApplication.quit()
    #    worker.__del__()
    ####################################################### IMPORTANT ###########################################################
    def __del__(self):
        print("Finalizando shadow")
        """Destructor"""
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
            maxCurrent = int(params["maxCurrent"])
            maxAcceleration = int(params["maxAcceleration"])
            maxDeceleration = int(params["maxDeceleration"])
            idDrivers = [int(params["idDriver1"]), int(params["idDriver2"])]
            wheelRadius = float(params["wheelRadius"])


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

            maxWheelSpeed = max(abs(self.m_wheels@np.array([self.maxLinSpeed, self.maxLinSpeed, self.maxRotSpeed])))

            self.driver = SVD48V.SVD48V(port, idDrivers, wheelRadius, maxWheelSpeed,
                                        maxAcceleration, maxDeceleration, maxCurrent)  

            if not self.driver.get_enable():
                print("NO se conecto al driver o fallo uno de ellos , cerrando programa")
                exit(-1)


            self.timer = QTimer(self)
            self.timer.timeout.connect(self.driver.show_params)
            #self.timer.start(1000)

        except Exception as e:
            print("Error reading config params or start motor")
            print(e)
        return True

    def setAdvx(self, val):
        if abs(val)>self.maxLinSpeed:  
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA",val,"CUANDO MAXIMA ES", self.maxLinSpeed)
        self.targetSpeed[0] = np.clip(val, -self.maxLinSpeed,  self.maxLinSpeed)
        
    def setAdvz(self, val):
        if abs(val)>self.maxLinSpeed: 
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA",val,"CUANDO MAXIMA ES", self.maxLinSpeed) 
        self.targetSpeed[1] = np.clip(val, -self.maxLinSpeed,  self.maxLinSpeed)
        
    def setRot(self, val):
        if abs(val)>self.maxRotSpeed:  
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA", val,"CUANDO MAXIMA ES", self.maxRotSpeed)
        self.targetSpeed[2] = np.clip(val, -self.maxRotSpeed,  self.maxRotSpeed)
        
    #######################################COMPUTE###########################################
    @QtCore.Slot()
    def compute(self):
        if self.driver != None:
            if self.driver.get_enable() and self.driver.get_safety():
                if  not np.array_equal(self.targetSpeed, self.oldTargetSpeed):
                    print(np.allclose(self.targetSpeed, 0, rtol=0.05))
                    print(
                        f"Modificamos velocidades: {np.round(self.oldTargetSpeed, 5).tolist()} a {np.round(self.targetSpeed, 5).tolist()} ")
                    speeds = self.m_wheels@self.targetSpeed
                    print(f"Velocidades de ruedas: {np.round(speeds, 5).tolist()}")
                    self.driver.set_speed(speeds)

                    self.oldTargetSpeed = np.copy(self.targetSpeed)
                    self.time_move = time.time()
                #si en un segundo no hay nuevo target se detiene
                elif time.time() - self.time_move > 5:
                    print("No comand, Stoping ")
                    self.OmniRobot_setSpeedBase(0.0,0.0,0.0)
                    self.time_move = float("inf")
                    #self.driver.disable_driver()
                    #self.driver.enable_driver()
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
        self.setAdvx(-advx)
        self.setAdvz(advz)
        self.setRot(rot)



    #
    # IMPLEMENTATION of stopBase method from OmniRobot interface
    #
    def OmniRobot_stopBase(self):
        self.time_emergency =time.time()
        self.OmniRobot_setSpeedBase(0, 0, 0)
        self.driver.emergency_stop()
    
    def reset_emergency_stop(self):
        if time.time()-self.time_emergency>1:
            self.OmniRobot_setSpeedBase(0, 0, 0)
            self.driver.reset_emergency_stop()


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
            if b.name == "block":
                if b.step == 1:
                    if self.driver.get_safety():
                        self.OmniRobot_stopBase()
                    else:
                        print("Unlock")
                        self.reset_emergency_stop()
                    self.joystickControl = False
            elif  b.name == "stop":
                if b.step == 1:
                    if self.driver.get_enable():
                        self.time_disble = time.time()
                        self.driver.disable_driver()
                    elif time.time()-self.time_disble > 1:
                        self.driver.enable_driver()
                    self.joystickControl = False
            elif b.name == "joystick_control":
                if b.step == 1:
                    self.joystickControl = not self.joystickControl
                    if not self.joystickControl:
                        self.OmniRobot_setSpeedBase(0, 0, 0)
                    print("Joystick control: ", self.joystickControl)
            else:
                pass#print(b.name, "PULASDOR NO AJUSTADO")
            
            if self.joystickControl:
                for a in  data.axes:
                    print(a.name, a.value)
                    if a.name == "rotate":
                        self.setRot(a.value)
                    elif  a.name == "advance":
                        self.setAdvz(a.value)
                    elif a.name == "side":
                        self.setAdvx(a.value)
                    else:
                        pass#print(a.name, "JOYSTICK NO AJUSTADO")

        

        #self.targetSpeed = [data.axes[0], data.axes[1], data.axes[2]]
        


