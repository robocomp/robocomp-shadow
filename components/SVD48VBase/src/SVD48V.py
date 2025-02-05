from tabnanny import check
import time, sys
import threading
import numpy as np
import serial


''''ESTRUCTURA DE TELEGRAMAS/DATAGRAMA'''
CODE_TELEGRAM ={
#___________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   |   8   |
#|ID|CODE|ADD_REG|NUM_REG|CRC_HIGH|CRC_LOW|
"READ": bytearray([0x03]),
#______________________________________________________________________
#|1 |  2 | 3 - 4 | 5 - 6 |    7   | 8 - 9| 10-11|...|  N-1   |   N   |
#|ID|CODE|ADD_REG|NUM_REG|NUM_BYTE|DATA_1|DATA_2|...|CRC_HIGH|CRC_LOW|
"WRITE": bytearray([0x10]),
#___________________________________________
#|1 |  2 |    3    |    5   |    7  |
#|ID|CODE|EXCEPTION|CRC_HIGH|CRC_LOW|
"ERROR_READ": bytearray([0x86]),
#____________________________________
#|1 |  2 |    3    |    5   |    7  |
#|ID|CODE|EXCEPTION|CRC_HIGH|CRC_LOW|
"ERROR_WRITE": bytearray([0x90])
}

class EXCEPTION_TELEGRAM(Exception):
    pass

error_messages=["", "Invalid function code", "Invalid register address", "Invalid data value"]


DRIVER_REGISTERS = {
    #"SAVE": bytearray([0x31, 0x00]),                # uint16//0=Do not save parameters to FLASH; 1: Save to FLASH
    "SET_STATUS": bytearray([0x53, 0x00]),          # uint16//SET DRIVER STATUS 0=STOP; 1=START; 2=CLEAN ALARMS
    "GET_STATUS": bytearray([0x54, 0x00]),          # DRIVER STATUS 0=STOP; 1=RUN
    "SET_SPEED": bytearray([0x53, 0x04]),           # int16//TARGET SPEED
    "GET_SPEED": bytearray([0x54, 0x10]),           # int16//CURRENT SPEED
    "GET_CURRENT": bytearray({0x54, 0x14}),         # int16//CURRENT CURRENT
    "GET_MOTOR_TEMPERATURE": bytearray([0x54, 0x04]),# int16//MOTOR TEMPERATURE
    "GET_DRIVE_TEMPERATURE": bytearray([0x21, 0x35]),# int16//DRIVER TEMPERATURE
    #dont work "MAX_SPEED": bytearray([0x50, 0x1C]), # uint16//MAXIMUM SPEED
    "POLE_PAIRS": bytearray([0x50, 0x18]),          # uint16//NUMNER OF POLE PAIRS
    "MAX_CURRENT": bytearray([0x50, 0x20]),         # uint16//MAXIMUM CURRENT
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

    def __init__(self, port="/dev/ttyUSB0", IDs=[1,2], polePairs=10, wheelRadius=6, maxSpeed=800, maxAcceleration=1000, maxDeceleration=1500, maxCurrent=6):
        """
        Initializes an instance of the class.

        Args:
            port (str): The port to be used. Defaults to "/dev/ttyUSB0".
            IDs (list): A list containing the IDs. Defaults to [1,2].
            polePairs(int): The number of motor pole pairs 
            wheelRadius (int): The radius of the wheel. Defaults to 6.
            maxSpeed (int): The maximum speed mm/s. Defaults to 800.
            maxAcceleration (int): The maximum acceleration mm/s2. Defaults to 1000.
            maxDeceleration (int): The maximum deceleration mm/s2. Defaults to 1500.
            maxCurrent (int): The maximum curren motor in Amperes. Defaults to 6.
        """
        print("Initializing SVD48V")
        self.port = port
        self.ids = IDs
        self.wheel_radius = wheelRadius
        self.mms2rpm = 60 / (2 * np.pi * self.wheel_radius)
        self.set_max_speed(maxSpeed)
        self.set_acceleration(maxAcceleration)
        self.set_deceleration(maxDeceleration)
        self.mutex = threading.Lock()

        self.accuracy_com = {"MSL" : 0.0, "CRC" : 0.0, "CODE" : 0.0, "OK" : 0.0}
        self.time_com = []
        self.data = {'Speed' : [0.0, 0.0, 0.0], 'Status' : [0,0], 'Temperature' : [0,0,0,0], "Current": [0,0,0,0]}

        self.driver = None
        self.thSelfResetting = None
        self.self_resetting = False
        self.safety = False

        while self.driver is None:
            print('Opening serial port with the SVD48V')
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
                        #print(text)
                        flag = True
                    timeout += 1
            else:
                sys.exit("Failed to open serial port "+ self.port +" with the SVD48V")
            if not self.check_connect():
                self.driver.close()
                self.driver = None
                print("Failed to connect motor drivers, restarting...")

        print("Starting SVD48V")
        for id in self.ids:
            self.write_register(id, DRIVER_REGISTERS['MAX_ACCELERATION'], [self.rpm_max_acceleration, self.rpm_max_acceleration])
            self.write_register(id, DRIVER_REGISTERS['MAX_DECELATION'], [self.rpm_max_deceleration, self.rpm_max_deceleration])
            self.write_register(id, DRIVER_REGISTERS['MAX_CURRENT'], [maxCurrent, maxCurrent])
            self.write_register(id,DRIVER_REGISTERS['POLE_PAIRS'],[polePairs, polePairs])
            
            
            self.enable_driver()
        

        print("Creating reading threads")
        self.threads_alive = True
        self.threads = [
        threading.Thread(daemon=True, target=self.update_parameter, args=(0.5, DRIVER_REGISTERS["GET_STATUS"], "Status")),
        threading.Thread(daemon=True, target=self.update_parameter, args=(0.5, DRIVER_REGISTERS["GET_SPEED"], "Speed")),
        threading.Thread(daemon=True, target=self.update_parameter, args=(10, DRIVER_REGISTERS["GET_MOTOR_TEMPERATURE"], "Temperature")),
        threading.Thread(daemon=True, target=self.update_parameter, args=(0.5, DRIVER_REGISTERS["GET_CURRENT"], "Current"))
        ]
        for thread in self.threads:
            thread.start()

        self.show_params(True)
        time.sleep(0.5)
        self.safety = True
        print("SVD48V started successfully")
        
    
    def __del__(self):
        
        if self.driver is not None and self.threads_alive:
            self.threads_alive = False
            print("Turning off SVD48V")
            self.disable_driver()

            print("Closing serial connection")
            self.driver.close()
            print("-------------------------------COMMUNICATION STATISTICS--------------------------------")

            print("COMMUNICATION ERRORS:", self.accuracy_com, "ACCURACY:",self.accuracy_com["OK"]*100/sum(self.accuracy_com.values()) )
            print("AVERAGE COMMUNICATION TIME: ", np.mean(self.time_com))
            for thread in self.threads:
                thread.join(timeout=2)
            print("SVD48V Deleted")

    def check_connect(self):
        """
        Check de conection motor driver.
        """
        for id in self.ids:
            status = self.read_register(id, DRIVER_REGISTERS['GET_STATUS'], False)
            if status is None:
                return False
        return True


    def enable_driver(self):
        """
        Enable the motor driver.
        """
        if self.driver.isOpen():
            print("Starting motors")
            for id in self.ids:
                if 0 in self.read_register(id, DRIVER_REGISTERS["GET_STATUS"], False):
                    print("start motor", id)
                    self.write_register(id, DRIVER_REGISTERS["SET_STATUS"], [2,2])
                    self.write_register(id, DRIVER_REGISTERS['SET_SPEED'], [0,0])
                    self.write_register(id, DRIVER_REGISTERS['SET_STATUS'], [1,1])
            if not self.self_resetting:
                self.self_resetting = True
                self.thSelfResetting = threading.Thread(target=self.still_alive,daemon=True)
                self.thSelfResetting.start()
            print("Motors started ")

        else:
            sys.exit("PORT NOT OPEN. Has the SVD48V possibly been disconnected?")

    def disable_driver(self):
        """
        Disable the motor driver.
        """
        print("Stopping motors")
        if self.thSelfResetting is not None:
            self.self_resetting = False
            self.thSelfResetting.join(timeout=2)

        # Stop the drivers with speed 0
        for id in self.ids:
            self.write_register(id, DRIVER_REGISTERS['SET_SPEED'], [0,0])
            self.write_register(id, DRIVER_REGISTERS['SET_STATUS'], [0,0])
            # Confirm status
            status = self.read_register(id, DRIVER_REGISTERS['GET_STATUS'], False)
            if 1 in status:
                print("Error when stopping the motors")
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
        try:
            data = []
            if self.driver.isOpen():
                self.mutex.acquire()      
                #numbers attempt
                for _ in range(NUM_ATTEMPT):
                    self.driver.flushInput()
                    # print(f"send telegram {list(telegram)}")
                    self.driver.write(telegram) #Send telegram
                    self.driver.flush()
                    t1 = time.time()            #get time for MSL
                    #Listening
                    while True:
                        time.sleep(0)
                        reply = bytearray (self.driver.readline())#get telegram of reply
                        #Have telegram of reply?
                        if len(reply) > 1:
                            #print(f"Reply telegram {list(reply)}")
                            #Have same number of action?
                            if reply[1] != CODE_TELEGRAM[action_code][0] :
                                if reply[1] != CODE_TELEGRAM["ERROR_" + action_code][0]:
                                    self.mutex.release()
                                    raise EXCEPTION_TELEGRAM(error_messages[reply[2]])
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
        except serial.SerialException:
            sys.exit("FAUL IN PORT. Has the SVD48V possibly been disconnected?")
        except EXCEPTION_TELEGRAM:
            raise
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
        attempt = 0
        telegram = self.generate_telegram(id, "READ",add_register, single)
        try:
            while (data := self.process_telegram(telegram, "READ")) is None and attempt < 2:
                print("Restarting Reading")
                time.sleep(0.005)
                attempt+=1
            return data
        except Exception as e:
            print("######ERROR###### ", e)
            return None
    
        
        
    def write_register(self, id=0xee, add_register=DRIVER_REGISTERS["SET_STATUS"], data_tuple=[0,0]):
        """
        Write to a register on the motor driver.

        Args:
            id (int): The id of the motor driver. Defaults to 0xee.
            add_register (str): The register to write to. Defaults to DRIVER_REGISTERS["SET_STATUS"].
            data_tuple (list): The data to write to the register. Defaults to [0,0].

        Returns:
            int: -2 if the wrong number of data items are provided, 0 OK and None fault communication .
        """

        attempt = 0
        telegram = self.generate_telegram(id, "WRITE", add_register,False, data_tuple)
        if telegram is None:
            print("Wrong number of data items")
            return -2
        try:
            while (data := self.process_telegram(telegram, "WRITE")) is None and attempt < 2:
                print("Restarting Writing")
                time.sleep(0.005)
                attempt+=1
            return data
        except Exception as e:
            print("######ERROR###### ", e)
            return None

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
            for _ in range(8):
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
        print("Current (m1, m2, m3, m4): ", self.get_current())
        print("Motor temperature: ", self.get_temperature())
        print("COMMUNICATION ERRORS:", self.accuracy_com, "ACCURACY:",self.accuracy_com["OK"]*100/sum(self.accuracy_com.values()) )
        print("AVERAGE COMMUNICATION TIME: ", np.mean(self.time_com))
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
                data.extend(self.read_register(id, register, False) or [None, None])
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
                        time.sleep(1)
                        print("THE DRIVER HAS STOPPED. TRYING TO RESTART")
                        self.enable_driver()
            else:
                sys.exit("PORT NOT OPEN. Has the SVD48V possibly been disconnected?")
            time.sleep(0.5)
        return 

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
    
    def get_rpm(self):
        """
        Get the rmp of the motors.

        Returns:
            The rpm of the motors.
        """
        return np.array(self.data["Speed"])/10

    def get_speed(self):
        """
        Get the speed of the motors.

        Returns:
            The speed of the motors.
        """
        return self.get_rpm()/self.mms2rpm
    

    
    def get_current(self):
        """
        Get the current of the motors.

        Returns:
            The current of the motors.
        """
        return np.array(self.data["Current"])/10
    
    def get_safety(self):
        """
        Returns the current safety status of the motor drivers.

        Returns:
            bool: The safety status. True if it's safe to operate, False it is not safe to operate.
        """
        return self.safety
    
    def get_enable(self):
        """
        Returns the current status of the motor drivers.

        Returns:
            bool: The status. True if it's  enable, False it's disable.
        """
        return self.self_resetting
    
    
    def set_acceleration(self, max_acceleration):
        """
        Set the maximum acceleration.

        Args:
            max_acceleration (int): The maximum acceleration to set in mm/s^2.
        """
        self.rpm_max_acceleration = abs(int(max_acceleration * self.mms2rpm))

    def set_deceleration(self, max_deceleration):
        """
        Set the maximum deceleration.

        Args:
            max_deceleration (int): The maximum deceleration to set in mm/s^2.
        """
        self.rpm_max_deceleration = abs(int(max_deceleration * self.mms2rpm))

    def set_max_speed(self, max_speed):
        """
        Set the maximum speed.

        Args:
            max_speed: The maximum speed to set in mm/s.
        """
        self.rpm_max_speed = abs(int(max_speed * self.mms2rpm))
        
    def set_rpm(self, motor_rpm:np.array):
        """
        Set the rpm of the motors. If the rpm exceeds the maximum, 
        all rpms are reduced proportionally.

        Args:
            motor_rpm[(int), (int), (int), (int)]: The rpm to set for the four motors.

        Returns:
            -2 if emergency stop is activated, -1 if the rpm had to be adjusted due to exceeding the maximum RPM, 0 otherwise.
        """
        if self.safety:
            ret = 0
            max_rpm = np.max(np.abs(motor_rpm))

            if max_rpm > self.rpm_max_speed:
                print(f"WARNING: WHEEL SPEED LIMIT EXCEEDED {max_rpm} WHEN MAXIMUM IS {self.rpm_max_speed}")
                motor_rpm = (motor_rpm / max_rpm) * self.rpm_max_speed
                ret = -1
            motor_rpm = motor_rpm.astype(np.int8)
            # print(f"RPM {rpm.tolist()} || MM/S {motor_rpm.tolist()}")

            for i in range(len(motor_rpm) // 2):
                self.write_register(self.ids[i], DRIVER_REGISTERS["SET_SPEED"],[motor_rpm[i * 2], motor_rpm[i * 2 + 1]])
        else:
            ret = -2
            print("Emergency stop is activated")
        return ret
        
    
    def set_speed(self, motor_speed:np.array):
        """
        Set the speed of the motors. If the speed exceeds the maximum, 
        all speeds are reduced proportionally.

        Args:
            motor_speed[(int), (int), (int), (int)]: The speed to set for the four motors.

        Returns:
            -2 if emergency stop is activated, -1 if the speed had to be adjusted due to exceeding the maximum RPM, 0 otherwise.
        """
        return self.set_rpm(np.array(motor_speed, dtype=float) * self.mms2rpm)
    
    def reset_emergency_stop(self):
        """
        Resets the emergency stop for the motor drivers.
        """
        print("RESET EMERGENCY STOP")
        for id in self.ids:
                self.write_register(id, DRIVER_REGISTERS['MAX_DECELATION'],[self.rpm_max_acceleration, self.rpm_max_acceleration])
        self.safety = True

    def emergency_stop(self):
        """
        Triggers an emergency stop for the motor drivers.

        Note:
            The acceleration is increased abruptly to induce a quick stop for each motor driver using their IDs.
            The speed is set to zero and the safety flag is also set to False, indicating an unsafe condition.
        """
        print("EMERGENCY STOP")
        self.safety = False
        acc_stop = self.mms2rpm*2000
        for id in self.ids:
            self.write_register(id, DRIVER_REGISTERS['MAX_DECELATION'], [acc_stop, acc_stop])
            self.write_register(id, DRIVER_REGISTERS["SET_SPEED"], [0, 0])   
                
