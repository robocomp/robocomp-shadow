import time, sys
import threading
import numpy as np
import serial


''''ESTRUCTURA DE TELEGRAMAS/DATAGRAMA'''
CODE_TELEGRAM ={
#___________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   |   8    |
#|ID|CODE|ADD_REG|NUM_REG|CRC_HIGH|CRC_LOW|
"READ": bytearray([0x03]),
#______________________________________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   | 8 - 9| 10-11|...|  N-1   |   N    |
#|ID|CODE|ADD_REG|NUM_REG|NUM_BYTE|DATA_1|DATA_2|...|CRC_HIGH|CRC_LOW|
"WRITE": bytearray([0x10])
}


DRIVER_REGISTERS = {
    "SET_STATUS": bytearray([0x53, 0x00]),          # uint16//SET DRIVER STATUS 0=STOP; 1=START; 2=CLEAN ALARMS
    "GET_STATUS": bytearray([0x54, 0x00]),          # DRIVER STATUS 0=STOP; 1=RUN
    "SET_SPEED": bytearray([0x53, 0x04]),           # int16//TARGET SPEED
    "GET_SPEED": bytearray([0x54, 0x10]),           # int16//CURRENT SPEED
    "GET_MOTOR_TEMPERATURE": bytearray([0x54, 0x04]),# int16//MOTOR TEMPERATURE
    "GET_DRIVE_TEMPERATURE": bytearray([0x21, 0x35]),# int16//DRIVER TEMPERATURE
    "MAX_SPEED": bytearray([0x50, 0x1C]),           # uint16//MAXIMUM SPEED
    "DIRECTION": bytearray([0x50, 0x28]),           # uint16//MOTOR DIRECTION 0=NORMAL; 1=INVERT
    "MAX_ACCELERATION": bytearray([0x51, 0x08]),    # uint16//MAXIMUM ACCELERATION
    "MAX_DECELATION": bytearray([0x51, 0x0C]),      # uint16//MAXIMUM DECELERATION   
    "CURVE_ACCELERATION": bytearray([0x51, 0x10]),  # uint16//S-CURVE ACCELERATION "Speed smoothing time S-type acceleration time"
    "ID": bytearray([0x30, 0x01]),                  # ID
    "MODE": bytearray([0x30, 0x08])                 # MODE
}             #
R_MODE = bytearray([0x30, 0x08]) 
''''VARIABLES DE COMUNICACCIÓN'''
MSL = 0.015 * 2                                  #Maximum Segment Lifetime             
NUM_ATTEMPT = 3                                 #Número de intentos de la conexión con el driver

class SVD48V:
    """
    Driver class to handle interaction with a motor controller device.

    Attributes:
        port: The serial port the driver is connected to.
        IDs: The identifiers for the motors driver.
        wheelRadius: The radius of the wheels for the motor.
        rpmMaxSpeed: The maximum speed in RPM.
    """

    def __init__(self, port="/dev/ttyUSB0", IDs=[1,2], wheelRadius=6, maxSpeed=800, maxAcceleration=1000, maxDeceleration=1500):
        """
        Initializes an instance of the class.

        Args:
            port (str): The port to be used. Defaults to "/dev/ttyUSB0".
            IDs (list): A list containing the IDs. Defaults to [1,2].
            wheelRadius (int): The radius of the wheel. Defaults to 6.
            maxSpeed (int): The maximum speed mm/s. Defaults to 800.
            maxAcceleration (int): The maximum acceleration mm/s2. Defaults to 1000.
            maxDeceleration (int): The maximum deceleration mm/s2. Defaults to 1500.
        """
        print("Initializing SVD48V")
        self.port = port
        self.ids = IDs
        self.wheel_radius = wheelRadius
        self.mms2rpm = 60 / (2 * np.pi * self.wheel_radius)
        self.rpm_max_speed = int(maxSpeed * self.mms2rpm)
        self.rpm_max_acceleration = int(maxAcceleration * self.mms2rpm)
        self.rpm_max_deceleration = int(maxDeceleration * self.mms2rpm)
        self.mutex = threading.Lock()

        self.accuracy_com = {"MSL" : 0.0, "CRC" : 0.0, "CODE" : 0.0, "OK" : 0.0}
        self.time_com = []
        self.data = {'Speed' : [0.0, 0.0, 0.0], 'Status' : [0,0], 'Temperature' : [0,0,0,0]}

        print('Opening serial port with the SVD48V')
        self.driver = None
        self.thSelfResetting = None
        self.self_resetting = False
        try:
            self.driver = serial.Serial(
                port=self.port,
                baudrate=115200,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0
            )
        except Exception:
            sys.exit("Failed to open serial port "+ self.port +" with the SVD48V")

        print("Starting SVD48V")
        if self.driver.isOpen():
            for id in self.ids:
                self.write_register(id, DRIVER_REGISTERS['MAX_ACCELERATION'], [self.rpm_max_acceleration, self.rpm_max_acceleration])
                self.write_register(id, DRIVER_REGISTERS['MAX_DECELATION'], [self.rpm_max_deceleration, self.rpm_max_deceleration])
                #self.write_register(id, DRIVER_REGISTERS['MAX_SPEED'], [self.rpm_max_speed, self.rpm_max_speed])
                self.start_diver()
        else:
            sys.exit("Failed to open serial port "+ self.port +" with the SVD48V")

        print("Creating reading threads")
        self.threads_alive = True
        self.threads = [
        threading.Thread(daemon=True, target=self.update_parameter, args=(0.5, DRIVER_REGISTERS["GET_STATUS"], "Status")),
        threading.Thread(daemon=True, target=self.update_parameter, args=(0.5, DRIVER_REGISTERS["GET_SPEED"], "Speed")),
        threading.Thread(daemon=True, target=self.update_parameter, args=(10, DRIVER_REGISTERS["GET_MOTOR_TEMPERATURE"], "Temperature"))
        ]
        for thread in self.threads:
            thread.start()
        
        self.show_params(True)
        time.sleep(0.5)
        print("SVD48V started successfully")
        
    
    def __del__(self):
        self.threads_alive = False
        if self.driver is not None:
            print("Turning off SVD48V")
            self.stop_driver()

            print("Closing serial connection")
            self.driver.close()
            if not self.driver.isOpen():
                self.driver = None
            print("-------------------------------COMMUNICATION STATISTICS--------------------------------")

            print("COMMUNICATION ERRORS:", self.accuracy_com, "ACCURACY:",self.accuracy_com["OK"]*100/sum(self.accuracy_com.values()) )
            print("AVERAGE COMMUNICATION TIME: ", np.mean(self.time_com))
            for thread in self.threads:
                thread.join()

    def start_diver(self):
        """
        Start the motor driver.
        """
        if self.driver.isOpen():
            time.sleep(1)

            #Print initial message
            print_info = True
            flag = False
            timeout=0
            while print_info and timeout<25:
                text = self.driver.readline()
                if len(text) == 0 and flag:
                    print_info = False
                elif len(text) > 0:
                    print(text)
                    flag = True
                timeout += 1

            print("Starting motors")
            for id in self.ids:
                self.write_register(id, DRIVER_REGISTERS['SET_SPEED'], [0,0])
                self.write_register(id, DRIVER_REGISTERS['SET_STATUS'], [1,1])
            if self.thSelfResetting is None:
                self.self_resetting = True
                self.thSelfResetting = threading.Thread(target=self.still_alive,daemon=True)
                self.thSelfResetting.start()

        else:
            sys.exit("PORT NOT OPEN. Has the SVD48V possibly been disconnected?")

    def stop_driver(self):
        """
        Stop the motor driver.
        """
        if self.thSelfResetting is not None:
            self.self_resetting = False
            self.thSelfResetting.join()

        # Stop the drivers with speed 0
        for id in self.ids:
            self.write_register(id, DRIVER_REGISTERS['SET_SPEED'], [0,0])
            self.write_register(id, DRIVER_REGISTERS['SET_STATUS'], [0,0])
            # Confirm status
            status = self.read_register(id, DRIVER_REGISTERS['GET_STATUS'], False)
            if 1 in status:
                print("Error when stopping the motors")
            else:
                print("Motors stopped")

    def generate_telegram(self, id=0xee, action_code="READ", add_register=DRIVER_REGISTERS["GET_STATUS"],single=False, data=None):
        """
        Generate a telegram to be sent to the motor driver.
        
        Args:
            id (int): The id of the motor driver. Defaults to 0xee.
            action_code (str): The action to be performed ("READ" or "WRITE"). Defaults to "READ".
            add_register (Register): The register to read from or write to. Defaults to DRIVER_REGISTERS["GET_STATUS"].
            single (bool): Whether to read/write only a single register. Defaults to False.
            data (list): The data to write to the register. Required if action_code is "WRITE".
            
        Returns:
            bytearray: The generated telegram.
        """
        telegram = bytearray([id])                             #Direction
        telegram.extend(CODE_TELEGRAM[action_code])            #Action
        telegram.extend(add_register)                          #Registers

        if action_code == "WRITE":                             #telegram write
            if len(data) in [1, 2]:
                telegram.extend([0, len(data), len(data) * 2]) #Length Registers
            else:
                print("Missing or extra register data")
                return None
            for item in data:
                telegram.extend(self.short_to_2bytes(int(item)))#Set Data
        else:                                                   #Telegram Read
            telegram.extend([0, 1] if single else [0, 2])       #Length registers
        telegram.extend(self.short_to_2bytes(self.calculate_crc(telegram))) 

        return telegram
    

    def process_telegram(self, telegram, action_code="READ"):
        """
        Process/Send a telegram from the motor driver.
        
        Args:
            telegram (bytearray): The telegram to process/send.
            action_code (str): The action performed ("READ" or "WRITE"). Defaults to "READ".
            
        Returns:
            list: A list of data read from the register. [] if the action_code is "WRITE". None in case of failure.
        """
        data = []
        if self.driver.isOpen():
            self.mutex.acquire()      
            #numbers attempt
            for _ in range(NUM_ATTEMPT):
                self.driver.flushInput()
                self.driver.write(telegram) #Send telegram
                self.driver.flush()
                t1 = time.time()            #get time for MSL
                #Listening
                while True:
                    time.sleep(0)
                    reply = bytearray (self.driver.readline())#get telegram of reply
                    #Have telegram of reply?
                    if len(reply) > 1:
                        #Have same number of action?
                        if reply[1] != CODE_TELEGRAM[action_code][0] :
                            print(f"UNSUITABLE TELEGRAM. RE-{action_code}")
                            self.accuracy_com["CODE"] += 1
                            break

                        #Check CRC for telegram
                        crc_low, crc_high = reply.pop(), reply.pop()
                        tel_crc_high, tel_crc_low = self.short_to_2bytes(self.calculate_crc(reply))
                        if crc_high != tel_crc_high or crc_low != tel_crc_low:
                            print(f"CRC FAILURE. RE-{action_code}")
                            self.accuracy_com["CRC"] += 1
                            break

                        #Get data from telegram response
                        if action_code == "READ":
                            for i in range(0, reply[2], 2):
                                data.append(np.int16(int(reply[i+3] * 2**8) + reply[i+4]))
                        
                        #Telegram OK
                        self.mutex.release()
                        self.accuracy_com["OK"] += 1
                        self.time_com.insert(0, time.time() - t1)
                        self.time_com = self.time_com[:50]
                        return data
                    
                    if time.time() - t1 > MSL:
                        print(f"MSL FAILURE. RE-{action_code}")
                        self.accuracy_com["MSL"] += 1
                        break  
            self.mutex.release()
            print(f"{action_code} ATTEMPTS EXHAUSTED ")
        return None

    def read_register(self, id=0xee, add_register=DRIVER_REGISTERS["GET_STATUS"], single=False):
        """
        Read register from the motor driver.

        Args:
            id (int): The id of the motor driver. Defaults to 0xee.
            add_register (str): The register to read from. Defaults to DRIVER_REGISTERS["GET_STATUS"].
            single (bool): Whether to read only a single register. Defaults to False.

        Returns:
            list: A list of the data from the register.
        """
        
        telegram = self.generate_telegram(id, "READ",add_register, False)
        while (data := self.process_telegram(telegram, "READ")) is None:
            print("Restarting Reading")
            time.sleep(0.0001)
        return data
        
        
    def write_register(self, id=0xee, add_register=DRIVER_REGISTERS["SET_STATUS"], data_tuple=[0,0]):
        """
        Write to a register on the motor driver.

        Args:
            id (int): The id of the motor driver. Defaults to 0xee.
            add_register (str): The register to write to. Defaults to DRIVER_REGISTERS["SET_STATUS"].
            data_tuple (list): The data to write to the register. Defaults to [0,0].

        Returns:
            int: -2 if the wrong number of data items are provided. 0 otherwise.
        """
        telegram = self.generate_telegram(id, "WRITE", add_register,False, data_tuple)
        if telegram is None:
            print("Wrong number of data items")
            return -2
        while (data := self.process_telegram(telegram, "WRITE")) is None:
            print("Restarting Writing")
            time.sleep(0.0001)
        return 0


    def calculate_crc(self, telegram):
        """
        Calculate the Cyclic Redundancy Check (CRC) of a telegram.

        Args:
            telegram (bytearray): The telegram to calculate the CRC of.

        Returns:
            int: The calculated CRC of the telegram.
        """
        crc_result = 0xffff
        for i in range(len(telegram)):
            crc_result = crc_result ^ int(telegram[i])
            for m in range(8):
                crc_num = (crc_result & 0x0001)
                xor_flag = 1 if crc_num == 1 else 0
                crc_result >>= 1
                if xor_flag:
                    crc_result = crc_result ^ 0xa001
        return crc_result

    def short_to_2bytes(self, short):
        """
        Convert a short integer to a list of 2 bytes.

        Args:
            short (int): The short integer to convert.

        Returns:
            list: The short integer represented as a list of 2 bytes.
        """
        low = int(short & 0x00FF)
        high = int((short & 0xFF00) >> 8)
        return high, low
        

    def show_params(self, advanced=False):
        """
        Display the parameters of the current object.

    Args:
        advanced (bool, optional): If True, display advanced parameters. Defaults to False.
        """

        print("------------------------------")
        print("Driver parameters list:")
        if advanced:
            print("port: ", self.port)
            print("rpmMaxSpeed: ", self.rpm_max_speed)
            print("Drivers ID: ", self.ids)
            print("wheelRadius: ", self.wheel_radius)
        print("Driver states: ", self.get_status())   
        print("Speed (m1, m2, m3, m4): ", self.get_speed())
        print("Motor temperature: ", self.get_temperature())
        print("------------------------------")

    def update_parameter(self, period, register, tag):
        """
        Update a specific parameter in a given register.

    Args:
        period (int): The period to updating the parameter.
        register (dict): The register that contains the parameter to update.
        tag (str): The tag of the parameter to update in the register.
        """
        while self.threads_alive:
            data = []
            for id in self.ids:
                data.extend(self.read_register(id, register, False))
            self.data[tag] = data
            time.sleep(period)

    def still_alive(self):
        """
        Check whether the motor driver is still responsive and try to reset it if not.
        """
        time.sleep(0.5)
        while self.self_resetting:
            if self.driver.isOpen():
                for id in self.ids:
                    #Confirmamos el estado
                    status = self.read_register(id, DRIVER_REGISTERS["GET_STATUS"], False)
                    if 0 in status:
                        print("THE DRIVER HAS STOPPED. TRYING TO RESTART")
                        self.write_register(id, DRIVER_REGISTERS["SET_STATUS"], [2,2])
                        self.start_diver()
            else:
                sys.exit("PORT NOT OPEN. Has the SVD48V possibly been disconnected?")
            time.sleep(0.5)

    def get_status(self):
        """
        Get the status of the motor driver.

        Returns:
            The status of the motor driver.
        """
        return self.data["Status"]

    def get_temperature(self):
        """
        Get the temperature of the motors.

        Returns:
            The temperature of the motors.
        """
        return np.array(self.data["Temperature"])/-10

    def get_speed(self):
        """
        Get the speed of the motors.

        Returns:
            The speed of the motors.
        """
        return np.array(self.data["Speed"])/(self.mms2rpm*10)
    
    
    def set_speed(self, motor_speed):
        """
        Set the speed of the motors. If the speed exceeds the maximum, 
        all speeds are reduced proportionally.

        Args:
            motor_speed: The speed to set for the four motors.

        Returns:
            -1 if the speed had to be adjusted due to exceeding the maximum RPM, 0 otherwise.
        """

        ret = 0
        rpm = np.array(motor_speed, dtype=float) * self.mms2rpm
        max_rpm = np.max(np.abs(rpm))

        if max_rpm > self.rpm_max_speed:
            print(f"WARNING: WHEEL SPEED LIMIT EXCEEDED {max_rpm} WHEN MAXIMUM IS {self.rpm_max_speed}")
            rpm = (rpm / max_rpm) * self.rpm_max_speed
            ret = -1

        print(f"MM/S {motor_speed}")
        print(f"RPM {rpm}")
        for i in range(len(motor_speed) // 2):
            self.write_register(self.ids[i], DRIVER_REGISTERS["SET_SPEED"],[rpm[i * 2], rpm[i * 2 + 1]])
        return ret
