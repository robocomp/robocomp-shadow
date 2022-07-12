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

from time import sleep


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
#|ID|CODE|ADD_REG|NUM_REG|CRC_HIGH|CRC_LOW|
CODE_TELEGRAM_READ = bytearray([0x03])

#______________________________________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   | 8 - 9| 10-11|...|  N-1   |   N    |
#|ID|CODE|ADD_REG|NUM_REG|NUM_BYTE|DATA_1|DATA_2|...|CRC_HIGH|CRC_LOW|
CODE_TELEGRAM_WRITE = bytearray([0x10])


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
R_ID = bytearray([0x30, 0x01])                  #

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        print("Iniciando shadow")
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 10
        
        #variables de clase
        self.targetSpeed = np.array([0, 0, 0])
        self. newTargetSpeed = np.array([0, 0, 0])
        self.actual = {'Speed' : [0, R_GET_SPEED], 'Status' : [0, R_GET_STATUS], 'Temperature' : [0, R_GET_TEMPERATURE]}
        self.driver=None


        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        print("Shadow iniciado correctamente")

    def __del__(self):
        print("Finalizando shadow")
        """Destructor"""
        if self.driver.isOpen():
            self.stop_driver(self.driver, 0xee)
        print("Shadow destruido")

    def setParams(self, params):
        print("Cargando parametros:")
        try:
            self.port = params["port"]
            self.wheelRadius = float(params["wheelRadius"])
            self.distAxes =  float(params["distAxes"])
            self.axesLength =  float(params["axesLength"])

            print("Parametros cargados:")
            print("port: ", self.port)
            print("wheelRadius: ", self.wheelRadius)
            print("distAxes: ", self.distAxes)
            print("axesLength: ", self.axesLength)

            self.driver = self.start_diver(self.port,0xee)  
            if self.driver == None:
                print("NO se conecto, cerrando preograma")
                exit(-1)

        except:
            print("Error reading config params")

        print("creando matriz de conversion")
        ll = 0.5*(self.distAxes + self.axesLength)

        ''''MATRIZ DE CONVESION'''
        self.m_wheels = np.array([[1, -1, 1, ll], 
                                    [1, 1, -1, -ll], 
                                    [1, 1, 1, ll],
                                    [1, -1, -1, -ll]])
        self.m_wheels = self.m_wheels * (1/(2 * np.pi * self.wheelRadius / 60)) # mm/s to rpm
        print(self.m_wheels)

        return True


    ##############################FUNCIONES DEL DRIVER######################################
    '''
    PRE:-
    POST:-
    DESC: Establece la conexión con el driver asigando al puerto "port", 
        habiliando al variable self.driver y poniendo el motor a 0 y
        el registro de funcionamiento a ON
    '''
    def start_diver(self, port="", id=0xee):
        if port == "":
            print('No hay puerto asignado')
        else:
            print('Abriendo puerto serie con el driver')

            driver = serial.Serial(
                port=port,
                baudrate=115200,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0)

            if driver.isOpen():
                print("Puerto abierto")
                #imprimimos el arranque del driver
                sleep(1)
                #imprimimos el arranque del driver si se estaria arancando
                print_info = True
                flag = False
                timeout=0
                while print_info and timeout<25:
                    text = driver.readline()
                    if text == "b''" and flag:
                        print_info =False
                    elif text != "b''":
                        print(text)
                        flag = True
                    timeout=timeout+1

                print("Encendemos motores")
                #arrancamos los driver con velocidad 0
                self.write_register(driver, id, R_SET_SPEED, False, [0,0,0,0])
                self.write_register(driver, id, R_SET_STATUS, False, [0,1,0,1])
        return driver

    '''
    PRE: self.driver abierto
    POST:-
    DESC: Cierra la conexión con el driver (variable self.driver)
        además pondra el motor a 0 y el registro de funcionamiento a OFF
    '''
    def stop_driver(self, driver, id=0xee):
        #paramos los driver con velocidad 0
        self.write_register(driver, id, R_SET_SPEED, False, [0,0,0,0])
        self.write_register(driver, id, R_SET_STATUS, False, [0,0,0,0])
        #Confirmamos el estado
        status = self.read_register(driver, id, R_GET_STATUS, False)
        if 1 in status:
            print("Error al parar los motores")
        else:
            print("Motores parados, cerrando serie")
            self.driver.close()
            if not self.driver.isOpen():
                print("Puerto cerrado correctamente")

    '''
    PRE:-
    POST: Devuelve dosr variables enteras fragmentos del short
    DESC: Fragmenta los dos brtes del short en dos bytes independientes
    '''
    def shortto2bytes(short):
        low = short & 0x00FF
        high = short & 0xFF00
        high = int(high/2**8)
        return high, low
        

    '''
    PRE: self.driver abierto
    POST: Devuelve una lista de los datos leidos
    DESC: Lee los registros del driver, "add_register"=dirección de comienzo
        "single"=true un solo registro, =false dos registros contiguos(M1 y M2)
    '''
    def read_register(self, driver, id=0xee, add_register=R_GET_STATUS, single=False):
        data = []
        if driver.isOpen():
            telegram = bytearray(id)
            telegram.extend(CODE_TELEGRAM_READ)
            telegram.extend(add_register)
            if single:
                telegram.extend([0,1])
            else:
                telegram.extend([0,2])
            telegram.extend(self.shortto2bytes(self.Calc_Crc(telegram)))   
            # driver.flushInput()
            # driver.reset_input_buffer()
            # driver.reset_output_buffer()
            driver.write(telegram)

            read_data = True
            while read_data:
                sleep(0.5)
                text = driver.readline()
                telegram = bytearray (driver.readline())
                if len(telegram) > 0:
                    print("respuesta recivida: ", telegram)
                    if telegram[1] != CODE_TELEGRAM_READ:
                        continue
                    crc_low = telegram.pop()
                    crc_high = telegram.pop()
                    tel_crc_high, tel_crc_low = self.shortto2bytes(self.Calc_Crc(telegram))
                    if crc_high != tel_crc_high or crc_low !=tel_crc_low:
                        print("FALLO EN EL CRC")
                        continue
                    for i in range(int(telegram[2]/2)):
                        data.append(int(telegram[i+3] * 2**8) + telegram[i+4])
                        print(data)
                    read_data = False
        else:
            print("PRUERTO NO ABIERTO")
        return data
                
        
    '''
    PRE: self.driver abierto & len("tupla_data")<=2
    POST: -
    DESC: Escribe los registros del driver, "add_register"=dirección de comienzo
        "single"=true un solo registro, =false dos registros contiguos(M1 y M2), 
        "tupla_data" datos a escribir
    '''
    def write_register(self, driver, id=0xee, add_register=R_SET_STATUS, single=False, tupla_data=[0,0]):
        if driver.isOpen():
            telegram = bytearray(id)
            telegram.extend(CODE_TELEGRAM_WRITE)
            telegram.extend(add_register)
            if single:
                telegram.extend([0,1,2])
            else:
                telegram.extend([0,2,4])
            for data in tupla_data:
                telegram.extend(self.shortto2bytes(data))
            telegram.extend(self.shortto2bytes(self.Calc_Crc(telegram)))   
            driver.write(telegram)
            print("envio, escritura: ", telegram)
            sleep(0.5)
            print("respuesta, escritura: ", driver.readline())
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
        

    def test_function(self):
        for i in range(256):
            leer_id = bytearray([i,0x03,0x30,0x01,0x00,0x01])
            leer_id.extend(self.Calc_Crc(leer_id))
            self.driver.write(leer_id)
            print(leer_id)
            print("leer id: ", self.driver.readline())
            sleep(0.1)


        print("ID", self.read_register(R_ID,True))
        print("velocidad max", self.read_register(R_MAX_SPEED,True))
        '''
        for x in range(10):
            print("ponemos velocidad")
            write_register(R_SET_SPEED,True,[0,10])
            for i in range(100):
                print("velocidad", read_register(R_GET_SPEED,True))

                print("aumentamos velocidad")
                write_register(R_SET_SPEED,True,[0,10])
                sleep(2)
                print("velocidad", read_register(R_GET_SPEED,True))
                sleep(2)
                print("quitamos velocidad")
                write_register(R_SET_SPEED,True,[0,0])
                sleep(2)
        '''


    #######################################COMPUTE###########################################
    @QtCore.Slot()
    def compute(self):
   
        self.test_function()
        '''
        if self.targetSpeed != self.newTargetSpeed:
            speeds = self.m_wheels@self.targetSpeed
            print("RPM",speeds)
            m1 = speeds[0]
            m2 = speeds[1]
            m3 = speeds[2]
            m4 = speeds[3]#mx se puede eliminar
            self.write_register(self.driver, 0xee, R_SET_SPEED, False, [m1, m2])
            self.write_register(self.driver, 0xee, R_SET_SPEED, False, [m3, m4]) ##########ver lo de la direccion del segundo driver##########################################################################
            print("Modificamos velocidades: ", self.targetSpeed)
        for key, val in self.actual:
            data = self.read_register(self.driver, 0xee, val[1],False)
            data.extend(self.read_register(self.driver, 0xee, val[1],False))##########ver lo de la direccion del segundo driver######################################################################
            print(key, ": M1 ", data[0], ": M2 ", data[1], ": M3 ", data[2], ": M4 ", data[3])
        '''
        return True

    def startup_check(self):
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
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
        self.targetSpeed[0] = advx
        self.targetSpeed[1] = advz
        self.targetSpeed[2] = rot


    #
    # IMPLEMENTATION of stopBase method from OmniRobot interface
    #
    def OmniRobot_stopBase(self):
    
        self.targetSpeed[0] = 0
        self.targetSpeed[1] = 0
        self.targetSpeed[2] = 0


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams


