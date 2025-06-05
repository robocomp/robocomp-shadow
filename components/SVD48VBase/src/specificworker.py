#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by Alejandro Torrejon Harto
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
from PySide6.QtWidgets import QApplication
from time import time
from rich.console import Console
from rich.text import Text
from genericworker import *
import interfaces as ifaces
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import SVD48V

console = Console(highlight=False)

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        print("Iniciando base")
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = 10
        
        #variables de clase
        self.targetSpeed = np.array([[0.0], [0.0], [0.0]])
        self.oldTargetSpeed = np.array([[0.0], [0.0], [0.0]])
        self.driver = None
        self.joystickControl = False
        self.time_disble = time()
        self.time_emergency = time()
        self.time_move = time()



        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)


    ####################################################### IMPORTANT ################################################################################
    #please in the SVD48VBase.py file the handler will call the __del__ 
    #SIGNALS handler
    #def sigint_handler(*args):
    #    QtCore.QCoreApplication.quit()
    #    worker.__del__()
    ####################################################### IMPORTANT ###########################################################
    def __del__(self):
        print("Finalizando base")
        """Destructor"""
        self.timer.stop()
        self.driver.__del__()
        print("Base destruida")        

    def setParams(self, params):
        print("Cargando parametros:")
        try:

            # Common params
            baseType = params["baseType"]
            assert baseType in ["Differential", "Omnidirectional"], "You must specificate the base type Diferential/Omnidirectional"

            axesLength =  float(params["axesLength"])
            port = params["port"]
            self.maxLinSpeed = float(params["maxLinSpeed"])
            self.maxRotSpeed = float(params["maxRotSpeed"])
            maxCurrent = int(params["maxCurrent"])
            maxAcceleration = int(params["maxAcceleration"])
            maxDeceleration = int(params["maxDeceleration"])
            idDrivers = [int(params["idDriver1"])]
            wheelRadius = int(params["wheelRadius"])
            polePairs = int(params["polePairs"])

            if baseType == "Omnidirectional":
                # Omnidireccional params
                self.isOmni=True
                distAxes =  float(params["distAxes"])
                idDrivers.append(int(params["idDriver2"]))
                
                ''''MATRIZ DE CONVESION'''
                ll = 0.5*(distAxes + axesLength)
                self.m_wheels = np.array([  [-1.0,  1.0, ll],
                                            [ 1.0,  1.0, -ll], 
                                            [ 1.0,  1.0, ll],
                                            [-1.0,  1.0, -ll]])
                
                maxWheelSpeed = max(abs(self.m_wheels@np.array([self.maxLinSpeed, self.maxLinSpeed, self.maxRotSpeed])))
                self.inv_m_wheels = np.linalg.pinv(self.m_wheels)
                
            else:
                # Diferential params
                self.isOmni=False
                self.m_wheels = np.array([[-1, axesLength/2], [-1, -axesLength/2]])
                # self.m_wheels = np.array([[1, -axesLength/2], [1, axesLength/2]]) TODO CHANGE WHEELS ORIENTATION
                maxWheelSpeed = max(abs(self.m_wheels@np.array([self.maxLinSpeed, self.maxRotSpeed])))
                self.inv_m_wheels = np.linalg.inv(self.m_wheels)
                
            print(self.m_wheels)

            self.driver = SVD48V.SVD48V(port=port, IDs=idDrivers, wheelRadius=wheelRadius, maxSpeed=maxWheelSpeed,
                                        maxAcceleration=maxAcceleration, maxDeceleration=maxDeceleration, maxCurrent=maxCurrent, polePairs=polePairs)  

            assert self.driver.get_enable(), "NO se conecto al driver o fallo uno de ellos , cerrando programa"
            self.oldOdometry = self.driver.get_position().flatten()
            #                      num rot * rot2mm
            self.maxOdometryDiff = 13 * (2 * np.pi * wheelRadius)/60



            self.showParams = QTimer(self)
            self.showParams.timeout.connect(self.driver.show_params)
            #self.showParams.start(1000)
            self.timer.start(self.Period)
            
            print("Base iniciada correctamente")

        except Exception as e:
            print("Error reading config params or start motor")
            print(e)
            exit(-1)
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
        if self.driver.get_enable() and self.driver.get_safety():
            if  not np.array_equal(self.targetSpeed, self.oldTargetSpeed):
                print(f"\033[32mModificamos velocidades: {np.round(self.oldTargetSpeed, 5).tolist()} a {np.round(self.targetSpeed, 5).tolist()} \033[0m")
                if self.isOmni:
                    speeds = self.m_wheels@self.targetSpeed
                else:
                    speeds = self.m_wheels@self.targetSpeed[1:]
                    
                #print(f"Modificamos velocidadespos: {np.round(self.oldTargetSpeed, 5).tolist()} a {np.round(self.targetSpeed, 5).tolist()} ")
                self.oldTargetSpeed = np.copy(self.targetSpeed)
                #print(f"oldTargetSpeed: {np.round(self.oldTargetSpeed, 5).tolist()}")

                self.driver.set_speed(speeds)
                #print(f"post speed: {np.round(self.oldTargetSpeed, 5).tolist()}\n")
                    

                self.time_move = time()
            #si en un segundo no hay nuevo target se detiene
            elif time() - self.time_move > 5:
                print("No comand, Stoping ")
                self.OmniRobot_setSpeedBase(0, 0, 0) if self.isOmni else self.DifferentialRobot_setSpeedBase(0, 0)
                self.driver.set_speed([0]*4 if self.isOmni else [0]*2)
                print("rpm",self.driver.get_rpm())
                if np.all(np.isclose(a=self.driver.get_rpm(),b=0, atol=0.5)): 
                    print("No comand, Stoped ")
                    self.time_move = float("inf")
                    self.driver.disable_driver()
                    self.driver.enable_driver()
                    
                    
            ##############PUB ODOMETRY####################
            try:
                odometry = ifaces.RoboCompFullPoseEstimation.FullPoseEuler()
                odometry.timestamp = np.longlong(time()*1000)
                
                #Millimeters odometry wheels
                newOdometry = self.driver.get_position().flatten()
                velocity = self.driver.get_speed().flatten()
                diffOdometry = newOdometry-self.oldOdometry
                if diffOdometry > self.maxOdometryDiff:
                    console.print(Text(f"Pose singularity with {diffOdometry}, maximum {self.maxOdometryDiff}", "bright_yellow"))
                    if diffOdometry < 0:
                        diffOdometry += self.maxOdometryDiff
                    else:
                        diffOdometry -= self.maxOdometryDiff
                    console.print(Text(f"Pose chaged to {self.maxOdometryDiff}", "bright_yellow"))

                self.oldOdometry = newOdometry

                #Convert wheel odometry to base odomery
                velocity = self.inv_m_wheels@velocity
                diffOdometry = self.inv_m_wheels@diffOdometry
                
                #Fill publish
                odometry.x = diffOdometry[0]
                odometry.y = diffOdometry[1] if diffOdometry.shape==3 else 0
                odometry.z = 0
                odometry.rx = 0
                odometry.ry = 0
                odometry.rz = diffOdometry[2] if diffOdometry.shape==3 else diffOdometry[1]
                odometry.vx = velocity[0]
                odometry.vy = velocity[1] if velocity.shape==3 else 0
                odometry.vz = 0
                odometry.vrx = 0
                odometry.vry = 0
                odometry.vrz = velocity[2] if velocity.shape==3 else velocity[1]
                odometry.ax = 0
                odometry.ay = 0
                odometry.az = 0
                odometry.arx = 0
                odometry.ary = 0
                odometry.arz = 0
                odometry.adv = 0
                odometry.side = 0
                odometry.rot = 0
                odometry.confidence = 0
                # print(odometry)
                self.fullposeestimationpub_proxy.newFullPose(odometry)
            except Exception as e:
                #console.print_exception(e)
                console.print(Text("Fault reading odometry", style="yellow"))
        return True

    def startup_check(self):
        print(f"Testing RoboCompDifferentialRobot.TMechParams from ifaces.RoboCompDifferentialRobot")
        test = ifaces.RoboCompDifferentialRobot.TMechParams()
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
    # IMPLEMENTATION of correctOdometer method from DifferentialRobot interface
    #
    def DifferentialRobot_correctOdometer(self, x, z, alpha):
        if not self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of getBasePose method from DifferentialRobot interface
    #
    def DifferentialRobot_getBasePose(self):
        x=0
        z=0
        alpha=0
        if not self.isOmni:
            #
            # write your CODE here
            #
            return [x, z, alpha]
    #
    # IMPLEMENTATION of getBaseState method from DifferentialRobot interface
    #
    def DifferentialRobot_getBaseState(self):
        #
        # write your CODE here
        #
        state = ifaces.RoboCompGenericBase.TBaseState()
        if not self.isOmni:
            pass
        return state
    #
    # IMPLEMENTATION of resetOdometer method from DifferentialRobot interface
    #
    def DifferentialRobot_resetOdometer(self):
        if not self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of setOdometer method from DifferentialRobot interface
    #
    def DifferentialRobot_setOdometer(self, state):
        if not self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of setOdometerPose method from DifferentialRobot interface
    #
    def DifferentialRobot_setOdometerPose(self, x, z, alpha):
        if not self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of setSpeedBase method from DifferentialRobot interface
    #
    def DifferentialRobot_setSpeedBase(self, adv, rot):
        if not self.isOmni and not self.joystickControl:
            self.setAdvz(adv)
            self.setRot(rot)

    #
    # IMPLEMENTATION of stopBase method from DifferentialRobot interface
    #
    def DifferentialRobot_stopBase(self):
        if not self.isOmni:
            self.time_emergency =time()
            self.DifferentialRobot_setSpeedBase(0, 0)
            self.driver.emergency_stop()

    #
    # IMPLEMENTATION of correctOdometer method from OmniRobot interface
    #
    def OmniRobot_correctOdometer(self, x, z, alpha):
        if self.isOmni:
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

        if self.isOmni:
            pass
        return [x, z, alpha]
    #
    # IMPLEMENTATION of getBaseState method from OmniRobot interface
    #
    def OmniRobot_getBaseState(self):
        #
        # write your CODE here
        #
        state = RoboCompGenericBase.TBaseState()
        if self.isOmni:
            pass
        return state
    #
    # IMPLEMENTATION of resetOdometer method from OmniRobot interface
    #
    def OmniRobot_resetOdometer(self):
        if self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of setOdometer method from OmniRobot interface
    #
    def OmniRobot_setOdometer(self, state):
        if self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of setOdometerPose method from OmniRobot interface
    #
    def OmniRobot_setOdometerPose(self, x, z, alpha):
        if self.isOmni:
            #
            # write your CODE here
            #
            pass


    #
    # IMPLEMENTATION of setSpeedBase method from OmniRobot interface
    #
    def OmniRobot_setSpeedBase(self, advx, advz, rot):
        if self.isOmni and not self.joystickControl:
            self.setAdvx(-advx)
            self.setAdvz(advz)
            self.setRot(rot)
        if not self.isOmni and not self.joystickControl:
            self.setAdvz(advz)
            self.setRot(rot)



    #
    # IMPLEMENTATION of stopBase method from OmniRobot interface
    #
    def OmniRobot_stopBase(self):
         if self.isOmni:
            self.time_emergency =time()
            self.OmniRobot_setSpeedBase(0, 0, 0)
            self.driver.emergency_stop()
    
    def reset_emergency_stop(self):
        if time()-self.time_emergency>1:
            self.OmniRobot_setSpeedBase(0, 0, 0) if self.isOmni else self.DifferentialRobot_setSpeedBase(0, 0)
            self.driver.reset_emergency_stop()



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
                        self.OmniRobot_stopBase() if self.isOmni else self.DifferentialRobot_stopBase()
                    else:
                        self.reset_emergency_stop()
                    self.joystickControl = False
            elif  b.name == "stop":
                if b.step == 1:
                    if self.driver.get_enable():
                        self.time_disble = time()
                        self.driver.disable_driver()
                    elif time()-self.time_disble > 1:
                        self.driver.enable_driver()
                    self.joystickControl = False
            elif b.name == "joystick_control":
                if b.step == 1:
                    self.joystickControl = not self.joystickControl
                    if not self.joystickControl:
                        self.OmniRobot_setSpeedBase(0, 0, 0) if self.isOmni else self.DifferentialRobot_setSpeedBase(0, 0)

                    print("Joystick control: ", self.joystickControl)
            else:
                pass#print(b.name, "PULASDOR NO AJUSTADO")
            
        if self.joystickControl:
            for a in  data.axes:
                #print(a.name, a.value)
                if a.name == "rotate":
                    self.setRot(a.value)
                elif  a.name == "advance":
                    self.setAdvz(a.value)
                elif a.name == "side":
                    self.setAdvx(a.value)
                else:
                    pass#print(a.name, "JOYSTICK NO AJUSTADO")

        


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompFullPoseEstimationPub you can publish calling this methods:
    # RoboCompFullPoseEstimationPub.void self.fullposeestimationpub_proxy.newFullPose(RoboCompFullPoseEstimation.FullPoseEuler pose)

    ######################
    # From the RoboCompDifferentialRobot you can use this types:
    # RoboCompDifferentialRobot.TMechParams

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompJoystickAdapter you can use this types:
    # RoboCompJoystickAdapter.AxisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData



