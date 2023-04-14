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
import time, pandas, datetime

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 30
        self.enable =False
        self.columns=["tiempo", "x", "y", "z"]
        self.data = pandas.DataFrame(columns=self.columns)
        self.tstart = time.time()
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        if self.enable:
            self.data.to_csv("test/" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_vibrator.csv")

    def setParams(self, params):
        return True


    @QtCore.Slot()
    def compute(self):
        if self.enable:
            print('Scan...')
            try:
                imu = self.imu_proxy.getAcceleration()
                self.data = pandas.concat([self.data, pandas.DataFrame({"tiempo":time.time()-self.tstart, "x":imu.XAcc, "y":imu.YAcc, "z":imu.ZAcc}, index=[0])], ignore_index=True)
            except Exception as e :
                print(e)
            
        return True

    def startup_check(self):
        print(f"Testing RoboCompIMU.Acceleration from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Acceleration()
        print(f"Testing RoboCompIMU.Gyroscope from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Gyroscope()
        print(f"Testing RoboCompIMU.Magnetic from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Magnetic()
        print(f"Testing RoboCompIMU.Orientation from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Orientation()
        print(f"Testing RoboCompIMU.DataImu from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.DataImu()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompJoystickAdapter.AxisParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.AxisParams()
        print(f"Testing RoboCompJoystickAdapter.ButtonParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.ButtonParams()
        print(f"Testing RoboCompJoystickAdapter.TData from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.TData()
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================
    def vibration_test(self):
        print("START VIBRATOR")
        try:
            #avanza  lento
            self.omnirobot_proxy.setSpeedBase(0,300,0)
            time.sleep(10)
            #gira 2 vueltas y media
            self.omnirobot_proxy.setSpeedBase(0,0,1)
            time.sleep(6.25)
            #movimiento lateral lento
            self.omnirobot_proxy.setSpeedBase(300,0,0)
            time.sleep(10)
            #ponemos recto
            self.omnirobot_proxy.setSpeedBase(0,0,1)
            time.sleep(1.25)
            self.omnirobot_proxy.setSpeedBase(0,0,0)
            time.sleep(0.25)
            #avanzamos rapido
            self.omnirobot_proxy.setSpeedBase(0,-800,0)
            time.sleep(3.75)
            self.omnirobot_proxy.setSpeedBase(0,0,0)
            time.sleep(0.25)
            #gira 2 vueltaws y media
            self.omnirobot_proxy.setSpeedBase(0,0,1)
            time.sleep(6.25)
            self.omnirobot_proxy.setSpeedBase(0,0,0)
            time.sleep(0.25)
            #movimiento lateral rapido
            self.omnirobot_proxy.setSpeedBase(-800,0,0)
            time.sleep(3.75)
            self.omnirobot_proxy.setSpeedBase(0,0,0)
            time.sleep(0.25)
            #Giramos para diagonal
            self.omnirobot_proxy.setSpeedBase(0,0,0.5)
            time.sleep(1)
            #avanzamos lento
            self.omnirobot_proxy.setSpeedBase(300,300,0)
            time.sleep(5)
            self.omnirobot_proxy.setSpeedBase(0,0,0)
            time.sleep(0.25)
            #avanzamos rapido
            self.omnirobot_proxy.setSpeedBase(-800,-800,0)
            time.sleep(3.75)
            #Giramos para ponermos rectos
            self.omnirobot_proxy.setSpeedBase(0,0,0.5)
            time.sleep(1.25)
            #paramos
            self.omnirobot_proxy.setSpeedBase(0,0,0)
        except Exception as e :
            print(e)


    #
    # SUBSCRIPTION to sendData method from JoystickAdapter interface
    #
    def JoystickAdapter_sendData(self, data):
    
        for b in data.buttons:
            if b.name == "test" and b.step == 1 and self.enable ==False:
                self.data = pandas.DataFrame(columns=["tiempo", "x", "y", "z"])
                self.tstart = time.time()
                self.enable = True
                self.vibration_test()
                print("Result:\n", self.data)
                self.data.to_csv("test/" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_vibrator.csv")
                self.enable = False
            



    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompIMU you can call this methods:
    # self.imu_proxy.getAcceleration(...)
    # self.imu_proxy.getAngularVel(...)
    # self.imu_proxy.getDataImu(...)
    # self.imu_proxy.getMagneticFields(...)
    # self.imu_proxy.getOrientation(...)
    # self.imu_proxy.resetImu(...)

    ######################
    # From the RoboCompIMU you can use this types:
    # RoboCompIMU.Acceleration
    # RoboCompIMU.Gyroscope
    # RoboCompIMU.Magnetic
    # RoboCompIMU.Orientation
    # RoboCompIMU.DataImu

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # self.omnirobot_proxy.correctOdometer(...)
    # self.omnirobot_proxy.getBasePose(...)
    # self.omnirobot_proxy.getBaseState(...)
    # self.omnirobot_proxy.resetOdometer(...)
    # self.omnirobot_proxy.setOdometer(...)
    # self.omnirobot_proxy.setOdometerPose(...)
    # self.omnirobot_proxy.setSpeedBase(...)
    # self.omnirobot_proxy.stopBase(...)

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompJoystickAdapter you can use this types:
    # RoboCompJoystickAdapter.AxisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData


