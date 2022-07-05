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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

import serial
import numpy as np

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel




''''ESTRUCTURA DE TELEGRAMAS/DATAGRAMA'''
#___________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   |   8    |
#|ID|CODE|ADD_REG|NUM_REG|CRC_HIGH|CRC_HIGH|
TELEGRAM_READ = bytearray([0xee,0x03])

#______________________________________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   | 8 - 9| 10-11|...|  N-1   |   N    |
#|ID|CODE|ADD_REG|NUM_REG|NUM_BYTE|DATA_1|DATA_2|...|CRC_HIGH|CRC_HIGH|
TELEGRAM_WRITE = bytearray([0xee,0x10])


''''REGISTROS DEL DRIVER'''
R_SET_STATUS = bytearray([0x53, 0x00])          #uint16//SET ESTADO DEL DRIVER 0=STOP; 1=START; 2=CLEAN ALARMS
R_GET_STATUS = bytearray([0x54, 0x00])          #ESTADO DEL DRIVER 0=STOP; 1=RUN
R_SET_SPEED = bytearray([0x53, 0x04])           #int16//VELOCIDAD TARGET
R_GET_SPEED = bytearray([0x54, 0x10])           #int16//VELOCIDAD ACTUAL
R_GET_TEMPERATURE = bytearray([0x54, 0x04])     #int16//TEMPERATURA DEL MOTOR
R_MAX_SPEED = bytearray([0x50, 0x1C])           #uint16//MAXIMA VELOCIDAD
R_DIRECTION = bytearray([0x50, 0x28])           #uint16//DIRECCION DEL MOTOR 0=NORMAL; 1=INVERT
R_ACCELERATION_MAX = bytearray([0x51, 0x08])    #uint16//MAXIMA ACELERACION
R_DECELATION_MAX = bytearray([0x51, 0x0C])      #uint16//MAXIMA DECELERACION   
R_CURVE_ACCELERATION = bytearray([0x51, 0x10])  #uint16//CURVA EN S DE ACELERACION "Speed smoothing time S-type acceleration time"

''''MATRIX DE CONVESION'''
M_WHEELS = np.array[[1, 1, 1, 1], 
                    [-1, 1, 1, -1], 
                    [1, -1, 1, -1]]

def start_diver(port=""):
    print('Abriendo puerto serie con el driver')

    self.driver = serial.Serial(
        port=port,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0)
    if self.driver.isOpen():
        print("puerto abierto")
        #imprimimos el arranque del driver
        print_info = True
        flag = False
        while print_info:
            text = self.driver.readline()
            if text == '' and flag:
                print_info =False
            elif text != '':
                print(text)
                flag = True

        print("Encendemos motores")
        #arrancamos los driver con velocidad 0
        write_register(R_SET_SPEED, False, [0,0,0,0])
        write_register(R_SET_STATUS, False, [0,1,0,1])

def stop_driver():
    #paramos los driver con velocidad 0
    write_register(R_SET_SPEED, False, [0,0,0,0])
    write_register(R_SET_STATUS, False, [0,0,0,0])
    #Confirmamos el estado
    status = read_register(R_GET_STATUS, False)
    if 1 in status:
        print("Error al parar los motores")
    else:
        print("Motores parados, cerrando serie")
        self.driver.close()
        if not self.driver.isOpen():
            print("Puerto cerrado correctamente")

def shortto2bytes(short):
    low = short & 0x00FF
    high = short & 0xFF00
    high = int(high/255)
    return high, low
    

def read_register(add_register, single):
    data = []
    if self.driver.isOpen():
        telegram = TELEGRAM_READ.copy()
        telegram.extend(add_register)
        if single:
            telegram.extend([0,1])
        else:
            telegram.extend([0,2])
        telegram.extend(shortto2bytes(Calc_Crc(telegram)))   
        self.driver.flushInput()
        read_data = True
        while read_data:
            text = self.driver.readline()
            if text != '':
                print(text)
                telegram =bytearray (text)
                if telegram[1] != 0x03:
                    continue
                crc_low = telegram.pop()
                crc_high = telegram.pop()
                tel_crc_high, tel_crc_low = shortto2bytes(Calc_Crc(telegram))
                if crc_high != tel_crc_high or crc_low !=tel_crc_low:
                    print("FALLO EN EL CRC")
                    continue
                for i in range(telegram[3]):
                    data.append(telegram[i+3] * 255) + telegram[i+4]
                    print(data)
                read_data = True
    else:
        print("PRUERTO NO ABIERTO")
    return data
            
    

def write_register(add_register, single, tupla_data):
    if self.driver.isOpen():
        telegram = TELEGRAM_WRITE.copy()
        telegram.extend(add_register)
        if single:
            telegram.extend([0,1])
        else:
            telegram.extend([0,2])
        for data in tupla_data:
            telegram.extend(shortto2bytes(data))
        telegram.extend(shortto2bytes(Calc_Crc(telegram)))   
        self.driver.write(telegram)
    else:
        print("PRUERTO NO ABIERTO")


'''
Function name: Calc_Crc(uint8_t\*pack_buff,uint8_tpack_len) Description:
Modbus protocol CRC check incoming value: pack_buff, packet data,
pack_len refers to the length of the data to be checked. Outgoing value: returns a twobyte CRC check code ***/ 
'''
def Calc_Crc(telegram):
    num = len(telegram)
    crc_result = 0xffff
    crc_num = 0
    xor_flag = 0
    for i in range(num):
        crc_result = crc_result ^ int(telegram[i])
        crc_num = (crc_result & 0x0001); 
        for m in range(8):
            if (crc_num == 1):
                xor_flag = 1
            else:
                xor_flag = 0
            crc_result >>= 1
            if(xor_flag):
                crc_result = crc_result ^ 0xa001
            crc_num =(crc_result & 0x0001)

    return crc_result


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 10
        
        #variables de clase
        self.target = {'Speed' : [np.array[0, 0, 0], R_SET_SPEED], 'Status' : [0, R_SET_STATUS]}
        self. newtarget = self.target.copy()
        self.actual = {'Speed' : [0, R_GET_SPEED], 'Status' : [0, R_GET_STATUS], 'Temperature' : [0, R_GET_TEMPERATURE]}
        self.driver=None


        start_diver()

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        stop_driver()

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):
        print('SpecificWorker.compute...')
        # computeCODE
        # try:
        #   self.differentialrobot_proxy.setSpeedBase(100, 0)
        # except Ice.Exception as e:
        #   traceback.print_exc()
        #   print(e)

        # The API of python-innermodel is not exactly the same as the C++ version
        # self.innermodel.updateTransformValues('head_rot_tilt_pose', 0, 0, 0, 1.3, 0, 0)
        # z = librobocomp_qmat.QVec(3,0)
        # r = self.innermodel.transform('rgbd', z, 'laser')
        # r.printvector('d')
        # print(r[0], r[1], r[2])
        for key, val in self.target:
            if val[0] != self.newtarget[key][0]:
                speeds = M_WHEELS@val[0]
                m1 = speeds[0]
                m2 = speeds[1]
                m3 = speeds[2]
                m4 = speeds[3]#mx se puede eliminar
                write_register(val[1], False, [m1, m2])
                write_register(val[1], False, [m3, m4]) ##########ver lo de la direccion del segundo driver##########################################################################
        for key, val in self.actual:
            data = read_register(val[1],False)
            data.extend(read_register(val[1],False))##########ver lo de la direccion del segundo driver######################################################################
            print(key, ": M1 ", data[0], ": M2 ", data[1], ": M3 ", data[2], ": M4 ", data[3])
        return True

    def startup_check(self):
        print(f"Testing RoboCompDifferentialRobot.TMechParams from ifaces.RoboCompDifferentialRobot")
        test = ifaces.RoboCompDifferentialRobot.TMechParams()
        print(f"Testing RoboCompGenericBase.TBaseState from ifaces.RoboCompGenericBase")
        test = ifaces.RoboCompGenericBase.TBaseState()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        QTimer.singleShot(200, QApplication.instance().quit)



    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of correctOdometer method from DifferentialRobot interface
    #
    def DifferentialRobot_correctOdometer(self, x, z, alpha):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of getBasePose method from DifferentialRobot interface
    #
    def DifferentialRobot_getBasePose(self):
    
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
        state = RoboCompGenericBase.TBaseState()
    return state
    #
    # IMPLEMENTATION of resetOdometer method from DifferentialRobot interface
    #
    def DifferentialRobot_resetOdometer(self):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setOdometer method from DifferentialRobot interface
    #
    def DifferentialRobot_setOdometer(self, state):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setOdometerPose method from DifferentialRobot interface
    #
    def DifferentialRobot_setOdometerPose(self, x, z, alpha):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setSpeedBase method from DifferentialRobot interface
    #
    def DifferentialRobot_setSpeedBase(self, adv, rot):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of stopBase method from DifferentialRobot interface
    #
    def DifferentialRobot_stopBase(self):
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of getBasePose method from GenericBase interface
    #
    def GenericBase_getBasePose(self):
    
        #
        # write your CODE here
        #
            return [x, z, alpha]
    #
    # IMPLEMENTATION of getBaseState method from GenericBase interface
    #
    def GenericBase_getBaseState(self):
    
        #
        # write your CODE here
        #
        state = RoboCompGenericBase.TBaseState()
    return state
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
    
        #
        # write your CODE here
        #
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
    
        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of stopBase method from OmniRobot interface
    #
    def OmniRobot_stopBase(self):
    
        #
        # write your CODE here
        #
        pass


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompDifferentialRobot you can use this types:
    # RoboCompDifferentialRobot.TMechParams

    ######################
    # From the RoboCompGenericBase you can use this types:
    # RoboCompGenericBase.TBaseState

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams


