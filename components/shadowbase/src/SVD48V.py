import time
import threading
import numpy as np
import threading
import serial


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
MSL = 0.01                                      #Maximum Segment Lifetime             
NUM_ATTEMPT = 3

class SVD48V:

    SLEEP_TEGRAM = 0.01
    def __init__(self, port, IDs, wheelRadius, maxSpeed):
        self.port = port
        self.IDs = IDs
        self.wheelRadius = wheelRadius
        self.mms2rpm = (60/(2*np.pi*self.wheelRadius))
        self.maxSpeed = int(maxSpeed * self.mms2rpm)
        self.mutex = threading.Lock()

        print('Abriendo puerto serie con el driver')
        self.driver = serial.Serial(
            port=self.port,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0)

        self.start_diver()

        self.data = { }
        self.thereadsLive = True
        self.thereads = []
        self.thereads.append(threading.Thread(target=self.update_parameter, args=(0.05, R_GET_STATUS, "Status")))
        self.thereads.append(threading.Thread(target=self.update_parameter, args=(0.05, R_GET_SPEED, "Speed")))
        self.thereads.append(threading.Thread(target=self.update_parameter, args=(5, R_GET_TEMPERATURE, "Temperature")))
        for th in self.thereads:
            th.daemon = True
            th.start()
        
    
    def __del__(self):
        self.thereadsLive = False
        self.stop_driver()
        for th in self.thereads:
            th.join()

        print("Cerramos serie")
        self.driver.close()

    '''
    PRE:-
    POST:-
    DESC: Establece la conexión con el driver asigando al puerto "port", 
        habiliando al variable self.driver y poniendo el motor a 0 y
        el registro de funcionamiento a ON
    '''
    def start_diver(self):
        if self.driver.isOpen():
            print("Puerto abierto")
            #imprimimos el arranque del driver
            time.sleep(1)
            #imprimimos el arranque del driver si se estaria arancando
            print_info = True
            flag = False
            timeout=0
            while print_info and timeout<25:
                text = self.driver.readline()
                if len(text) == 0 and flag:
                    print_info =False
                elif len(text) > 0:
                    print(text)
                    flag = True
                timeout=timeout+1

            print("Encendemos motores")
            #arrancamos los driver con velocidad 0
            err=0
            for id in self.IDs:
                err-=self.write_register(id, R_ACCELERATION_MAX, [self.maxSpeed,self.maxSpeed])
                err-=self.write_register(id, R_SET_SPEED, [0,0])
                err-=self.write_register(id, R_SET_STATUS, [1,1])
            return err
        else:
            print("No se pudo habrir el puerto")
            return -50

    '''
    PRE:-
    POST:-
    DESC: Establece la conexión con el driver asigando al puerto "port", 
        habiliando al variable self.driver y poniendo el motor a 0 y
        el registro de funcionamiento a ON
    '''
    def reset_diver(self):
        if self.driver.isOpen():
            print("Reiniciando driver")
            #imprimimos el arranque del driver
            time.sleep(1)
            #imprimimos el arranque del driver si se estaria arancando
            print_info = True
            flag = False
            timeout=0
            while print_info and timeout<25:
                text = self.driver.readline()
                if len(text) == 0 and flag:
                    print_info =False
                elif len(text) > 0:
                    print(text)
                    flag = True
                timeout=timeout+1

            print("Encendemos motores")
            #arrancamos los driver con velocidad 0
            err=0
            for id in self.IDs:
                err-=self.write_register(id, R_SET_SPEED, [0,0])
                err-=self.write_register(id, R_SET_STATUS, [1,1])
            return err
        else:
            print("No se pudo habrir el puerto")
            return -50
    
    '''
    PRE: self.driver abierto
    POST:-
    DESC: Cierra la conexión con el driver (variable self.driver)
        además pondra el motor a 0 y el registro de funcionamiento a OFF
    '''
    def stop_driver(self):
        #paramos los driver con velocidad 0
        for id in self.IDs:
            self.write_register(id, R_SET_SPEED, [0,0])
            self.write_register(id, R_SET_STATUS, [0,0])
            #Confirmamos el estado
            status = self.read_register(id, R_GET_STATUS, False)
            if 1 in status:
                print("Error al parar los motores")
            else:
                print("Motores parado")


    '''
    PRE: self.driver abierto
    POST: Devuelve una lista de los datos leidos
    DESC: Lee los registros del driver, "add_register"=dirección de comienzo
        "single"=true un solo registro, =false dos registros contiguos(M1 y M2)
    '''
    def read_register(self, id=0xee, add_register=R_GET_STATUS, single=False):
        data = []
        if self.driver.isOpen():
            #generamos el telegrama de peticion
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
        
            #procedemos al envio y escucha de los telegramas
            read_data = True
            attempt = NUM_ATTEMPT #numeros de intentos
            self.mutex.acquire()  #bloqueamos el serial para evitar colisiones

            while read_data:
                if attempt == 0:
                    print("Fallo por intentos en lectua")
                    break
                #print("Telegrama de peticion: ", telegram)
                self.driver.write(telegram) #enviamos telegrama
                t1 = time.time()        #obtememos tiempo 
                attempt-=1
                while read_data:
                    time.sleep(0)
                    telegram = bytearray (self.driver.readline())
                    if len(telegram) > 0:
                        #print("respuesta recivida: ", telegram)
                        if telegram[1] != CODE_TELEGRAM_READ[0] :
                            print("telegrama no apto reintentando")
                            break
                        crc_low = telegram.pop()
                        crc_high = telegram.pop()
                        tel_crc_high, tel_crc_low = self.shortto2bytes(self.Calc_Crc(telegram))
                        if crc_high != tel_crc_high or crc_low !=tel_crc_low:
                            print("FALLO EN EL CRC REINTENTANDO")
                            break
                        for i in range(0, telegram[2], 2):
                            #print(telegram[i+3], " - ", telegram[i+4] )
                            data.append(np.int16(int(telegram[i+3] * 2**8) + telegram[i+4]))
                            #print(data)
                        read_data = False
                    t2 = time.time()
                    if t2 - t1 > MSL:
                        break

            self.mutex.release()
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
    def write_register(self, id=0xee, add_register=R_SET_STATUS, tupla_data=[0,0]):
        if self.driver.isOpen():
            #generamos el telegrama de escritura
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

            #procedemos al envio y escucha de los telegramas
            read_data = True
            attempt = NUM_ATTEMPT #numeros de intentos
            self.mutex.acquire()  #bloqueamos el serial para evitar colisiones

            while read_data:
                if attempt == 0:
                    print("Fallo por intentos en lectua")
                    break
                #print("Telegrama de peticion: ", telegram)
                self.driver.write(telegram) #enviamos telegrama
                t1 = time.time()        #obtememos tiempo 
                attempt-=1
                while read_data:
                    time.sleep(0)
                    telegram = bytearray (self.driver.readline())
                    if len(telegram) > 0:
                        #print("respuesta recivida: ", telegram)
                        if telegram[1] != CODE_TELEGRAM_WRITE[0] :
                            print("telegrama no apto reintentando")
                            break
                        crc_low = telegram.pop()
                        crc_high = telegram.pop()
                        tel_crc_high, tel_crc_low = self.shortto2bytes(self.Calc_Crc(telegram))
                        if crc_high != tel_crc_high or crc_low !=tel_crc_low:
                            print("FALLO EN EL CRC REINTENTANDO")
                            break
                        read_data = False
                    t2 = time.time()
                    if t2 - t1 > MSL:
                        break
            self.mutex.release()
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
        

    def show_params(self, advanced=False):
        print("------------------------------")
        print("Lista de parametros del driver:")
        if advanced:
            print("port: ", self.port)
            print("maxSpeed: ", self.maxSpeed)
            print("Drivers ID: ", self.IDs)
            print("wheelRadius: ", self.wheelRadius)
        print("Estados drivers: ", self.data["Status"])   
        print("Velocidad (m1, m2, m3, m4): ", self.data["Speed"])
        print("Temperaruta motores: ", self.data["Temperature"])
        print("------------------------------")


    def test_function(self):
        for i in self.IDs:
            print("ID", self.read_register(i, R_ID,True))
            print("vel", self.read_register(i, R_GET_SPEED,False))
            self.write_register(i, R_SET_SPEED,[0,0])

        time.sleep(5) 
        for i in self.IDs:
            self.write_register(i, R_SET_SPEED,[0,0])
        time.sleep(2) 



    #funcion para hilos de lectura
    def update_parameter(self, times, register, tag):
        while self.thereadsLive:
            data = []
            for id in self.IDs:
                data.extend(self.read_register(id, register, False))
            self.data[tag] = data
            time.sleep(times)

    def still_alive(self):
        if not self.driver.isOpen():
            return False
        for id in self.IDs:
            #Confirmamos el estado
            status = self.read_register(id, R_GET_STATUS, False)
            if len(status) == 0:
                return False
        return True

    def get_status(self):
        return self.data["Status"]

    def get_temperature(self):
        return self.data["Temperature"]

    def get_speed(self):
        return(self.data["Speed"]/self.mms2rpm)
                  
    
    def set_speed(self, motor_speed):
        print("M1 ", motor_speed[0], "M2 ", motor_speed[1], "M3 ", motor_speed[2], "M4 ", motor_speed[3])
        rpm = motor_speed * self.mms2rpm
        
        for i in range(0, len(motor_speed),2):
            self.write_register(self.IDs[i], R_SET_SPEED,[rpm[i],rpm[i+1]])
        
        if rpm.max() > self.maxSpeed:
            print("AVISO SUPERADA LA VELOCIDAD MAXIMA")
            return -1
        else:
            return 0

