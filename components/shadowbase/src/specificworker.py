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

from logging import exception
from time import sleep


from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

import serial
import numpy as np
import traceback

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
R_MODE = bytearray([0x30, 0x08]) 

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        print("Iniciando shadow")
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 10
        
        #variables de clase
        self.targetSpeed = np.array([[0.0], [0.0], [0.0]])
        self. newTargetSpeed = np.array([[0.0], [0.0], [0.0]])
        self.actual = {'Speed' : [ np.array([[0.0], [0.0], [0.0], [0.0]]), R_GET_SPEED], 'Status' : [[0,0], R_GET_STATUS], 'Temperature' : [[0,0,0,0], R_GET_TEMPERATURE]}
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
            self.stop_driver(self.driver, self.idDrivers)
        print("Shadow destruido")

    def setParams(self, params):
        print("Cargando parametros:")
        try:
            self.port = params["port"]
            self.maxSpeed = int(params["maxSpeed"])
            self.idDrivers = [int(params["idDriver1"]), int(params["idDriver2"])]
            self.wheelRadius = float(params["wheelRadius"])
            self.distAxes =  float(params["distAxes"])
            self.axesLength =  float(params["axesLength"])

            self.show_params(True)

            print("creando matriz de conversion")
            ll = 0.5*(self.distAxes + self.axesLength)

            ''''MATRIZ DE CONVESION'''
            self.m_wheels = np.array([[1.0, -1.0, ll], 
                                        [1.0, 1.0, -ll], 
                                        [1.0, 1.0, ll],
                                        [1.0, -1.0, -ll]])
            #self.m_wheels_inv = np.linalg.inv(self.m_wheels)
            self.m_wheels = self.m_wheels * (1/(2 * np.pi * self.wheelRadius / 60)) # mm/s to rpm
            #self.m_wheels_inv = self.m_wheels_inv * ((2 * np.pi * self.wheelRadius / 60)) # rpm to mm/s
            print(self.m_wheels)
           #print(self.m_wheels_inv)

            self.driver = self.start_diver(self.port,self.idDrivers)  
            if self.driver == None:
                print("NO se conecto, cerrando programa")
                exit(-1)

        except exception as e:
            print("Error reading config params or start motor")
            print(e)

        

        return True


    ##############################FUNCIONES DEL DRIVER######################################
    '''
    PRE:-
    POST:-
    DESC: Establece la conexión con el driver asigando al puerto "port", 
        habiliando al variable self.driver y poniendo el motor a 0 y
        el registro de funcionamiento a ON
    '''
    def start_diver(self, port="", drivers=[0xee]):
        driver = None
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
                    if len(text) == 0 and flag:
                        print_info =False
                    elif len(text) > 0:
                        print(text)
                        flag = True
                    timeout=timeout+1

                print("Encendemos motores")
                #arrancamos los driver con velocidad 0
                for id in drivers:
                    self.write_register(driver, id, R_ACCELERATION_MAX, [self.maxSpeed,self.maxSpeed])
                    self.write_register(driver, id, R_SET_SPEED, [0,0])
                    self.write_register(driver, id, R_SET_STATUS, [1,1])
            else:
                print("No se pudo habrir el puerto")
        return driver

    '''
    PRE: self.driver abierto
    POST:-
    DESC: Cierra la conexión con el driver (variable self.driver)
        además pondra el motor a 0 y el registro de funcionamiento a OFF
    '''
    def stop_driver(self, driver, drivers=[0xee]):
        #paramos los driver con velocidad 0
        for id in drivers:
            self.write_register(driver, id, R_SET_SPEED, [0,0])
            self.write_register(driver, id, R_SET_STATUS, [0,0])
            #Confirmamos el estado
            status = self.read_register(driver, id, R_GET_STATUS, False)
            if 1 in status:
                print("Error al parar los motores")
            else:
                print("Motores parado")
        print("Cerramos serie")
        self.driver.close()

    '''
    PRE:-
    POST: Devuelve dosr variables enteras fragmentos del short
    DESC: Fragmenta los dos brtes del short en dos bytes independientes
    '''
    def shortto2bytes(self, short):
        low = int(short & 0x00FF)
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
            telegram = bytearray([id])
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
            print("Telegrama de peticion: ", telegram)
            driver.write(telegram)

            read_data = True
            while read_data:
                sleep(0.5)
                telegram = bytearray (driver.readline())
                if len(telegram) > 0:
                    print("respuesta recivida: ", telegram)
                    #print(telegram[1], " - ", CODE_TELEGRAM_READ)
                    if telegram[1] != CODE_TELEGRAM_READ[0] :
                        print("temegrama no apto")
                        continue
                    crc_low = telegram.pop()
                    crc_high = telegram.pop()
                    tel_crc_high, tel_crc_low = self.shortto2bytes(self.Calc_Crc(telegram))
                    if crc_high != tel_crc_high or crc_low !=tel_crc_low:
                        print("FALLO EN EL CRC")
                        continue
                    for i in range(0, telegram[2], 2):
                        #print(telegram[i+3], " - ", telegram[i+4] )
                        data.append(np.int16(int(telegram[i+3] * 2**8) + telegram[i+4]))
                        #print(data)
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
    def write_register(self, driver, id=0xee, add_register=R_SET_STATUS, tupla_data=[0,0]):
        if driver.isOpen():
            telegram = bytearray([id])
            telegram.extend(CODE_TELEGRAM_WRITE)
            telegram.extend(add_register)
            if len(tupla_data)==1:
                telegram.extend([0,1,2])
            elif(len(tupla_data)==2):
                telegram.extend([0,2,4])
            else:
                print("Faltan o sobran datos de registro")
                return -2
            for data in tupla_data:
                telegram.extend(self.shortto2bytes(data))
            telegram.extend(self.shortto2bytes(self.Calc_Crc(telegram)))   
            driver.write(telegram)
            print("envio, escritura: ", telegram)
            sleep(0.5)
            print("respuesta, escritura: ", driver.readline())
            return 0
        else:
            print("PRUERTO NO ABIERTO")
            return-1


    '''
    Function name: Calc_Crc(uint8_t\*pack_buff,uint8_tpack_len) Description:
    Modbus protocol CRC check incoming value: pack_buff, packet data,
    pack_len refers to the length of the data to be checked. Outgoing value: returns a twobyte CRC check code ***/ 
    '''
    def Calc_Crc(self, telegram):
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
        for i in self.idDrivers:
            print("ponemos velocidad")
            self.write_register(self.driver, i, R_SET_SPEED,[10,10])
            sleep(2)
            self.write_register(self.driver, i, R_SET_SPEED,[-25,5])
            sleep(2)
            print("ID", self.read_register(self.driver,i, R_ID,True))
            print("vel", self.read_register(self.driver,i, R_GET_SPEED,False))
            self.write_register(self.driver, i, R_SET_SPEED,[0,0])


    
        sleep(5) 
        for i in self.idDrivers:
            self.write_register(self.driver, i, R_SET_SPEED,[0,0])
        sleep(2) 

    def show_params(self, advanced=False):
        print("------------------------------")
        print("Lista de parametros:")
        if advanced:
            print("port: ", self.port)
            print("maxSpeed: ", self.maxSpeed)
            print("Drivers ID: ", self.idDrivers)
            print("wheelRadius: ", self.wheelRadius)
            print("distAxes: ", self.distAxes)
            print("axesLength: ", self.axesLength)
        print("Estados drivers: ", self.actual["Status"][0])   
        print("Velocidad objetivo (advx, advz, rot): ", self.targetSpeed)
        print("Velocidad (advx, advz, rot): ", self.actual["Speed"][0])
        print("Temperaruta motores: ", self.actual["Temperature"][0])
        print("------------------------------")
    
    def set_actualSpeed(self,data):
        print("M1 ", data[0], "M2 ", data[1], "M3 ", data[2], "M4 ", data[3])

        self.actual["Speed"][0]



    #######################################COMPUTE###########################################
    @QtCore.Slot()
    def compute(self):
        if self.driver != None:
            #self.test_function()
        
            if  not np.array_equal(self.targetSpeed, self.newTargetSpeed):
                
                speeds = self.m_wheels@self.targetSpeed
                
                print("RPM",speeds)
                for i in range(len(self.idDrivers)):
                    self.write_register(self.driver,  self.idDrivers[i], R_SET_SPEED, [int(speeds[i*2][0]), int(speeds[(i*2)+1][0])])
                
                print("Modificamos velocidades: ", self.targetSpeed)
            for key, val in self.actual.items():
                val[0] = []
                data = [[]]
                print("Actualizamos: ", key)
                for id in self.idDrivers:
                    data.extend(self.read_register(self.driver, id, val[1], len(val[0])))
                   
                if (key=="Speed"):
                    #data = self.m_wheels_inv@data
                    pass
                
                val[0].extend(data)
            self.show_params(False)
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


