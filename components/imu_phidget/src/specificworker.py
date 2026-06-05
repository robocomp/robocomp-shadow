#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2026 by YOUR NAME HERE
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
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import os
console = Console(highlight=False)

import threading
import logging
import copy

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

from src.phidget import PhidgetIMU, IMUSample


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

        cfg_imu    = self.configData.get("IMU", {})

        # ---- Parametros del driver IMU ----
        self._imu_interval_ms  = int(cfg_imu.get("interval_ms", 4))
        self._static_collect_s = float(cfg_imu.get("static_collect_s", 2.0))
        # rad/s ? umbral de deteccion de reposo para la inicializacion estatica
        self._static_omega_thr = float(cfg_imu.get("static_omega_thr", 0.01))
        self._imu_timeout_s    = float(cfg_imu.get("timeout_s", 0.2))

        self._imu = PhidgetIMU(data_interval_ms=self._imu_interval_ms)
        if not self._imu.start(timeout_s=5.0):
            log.error("No se pudo conectar el IMU Phidget")
            return False
        
        self._imu_lock = threading.Lock()
        self._last_imu_measurement = self._imu.get_latest()


    def __del__(self):
        """Destructor"""


    @QtCore.Slot()
    def compute(self):
        try:
            new = self._imu.get_latest()          
            if new is not None:
                fresh = copy.deepcopy(new)
                with self._imu_lock:              
                    self._last_imu_measurement = fresh
        except Exception as e:
            console.print_exception(e)
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
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== Methods for Component Implements ==================
    # ===================================================================

    def _snapshot(self):
        """Devuelve una referencia atómica; el lock se libera enseguida."""
        with self._imu_lock:
            return copy.deepcopy(self._last_imu_measurement)

    #
    # IMPLEMENTATION of getAcceleration method from IMU interface
    #
    def IMU_getAcceleration(self):
        m = self._snapshot()
        ret = ifaces.RoboCompIMU.Acceleration()
        ret.XAcc = m.accel[0]
        ret.YAcc = m.accel[1]
        ret.ZAcc = m.accel[2]
        ret.timestamp = m.timestamp_us

        return ret
    #
    # IMPLEMENTATION of getAngularVel method from IMU interface
    #
    def IMU_getAngularVel(self):
        m = self._snapshot()
        ret = ifaces.RoboCompIMU.Gyroscope()
        ret.XGyr = m.gyro[0]
        ret.YGyr = m.gyro[1]
        ret.ZGyr = m.gyro[2]
        ret.timestamp = m.timestamp_us

        return ret
    #
    # IMPLEMENTATION of getDataImu method from IMU interface
    #
    def IMU_getDataImu(self):
        m = self._snapshot()
        ret = ifaces.RoboCompIMU.DataImu()

        ret.acc.XAcc = m.accel[0]
        ret.acc.YAcc = m.accel[1]
        ret.acc.ZAcc = m.accel[2]
        ret.acc.timestamp = m.timestamp_us

        ret.gyro.XGyr = m.gyro[0]
        ret.gyro.YGyr = m.gyro[1]
        ret.gyro.ZGyr = m.gyro[2]
        ret.gyro.timestamp = m.timestamp_us

        ret.mag.XMag = m.mag[0]
        ret.mag.YMag = m.mag[1]
        ret.mag.ZMag = m.mag[2]
        ret.mag.timestamp = m.timestamp_us

        ret.temperature = m.temperature

        return ret
    #
    # IMPLEMENTATION of getMagneticFields method from IMU interface
    #
    def IMU_getMagneticFields(self):
        m = self._snapshot()
        ret = ifaces.RoboCompIMU.Magnetic()
        ret.XMag = m.mag[0]
        ret.YMag = m.mag[1]
        ret.ZMag = m.mag[2]
        ret.timestamp = m.timestamp_us

        return ret
    #
    # IMPLEMENTATION of getOrientation method from IMU interface
    #
    def IMU_getOrientation(self):
        log.error("No implementado")
        return 
    #
    # IMPLEMENTATION of resetImu method from IMU interface
    #
    def IMU_resetImu(self):
        self._imu.stop()
        if not self._imu.start(timeout_s=5.0):
            log.error("No se pudo conectar el IMU Phidget")
        
        log.info("IMU Phidget correctamente reseteado")


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompIMU you can use this types:
    # ifaces.RoboCompIMU.Acceleration
    # ifaces.RoboCompIMU.Gyroscope
    # ifaces.RoboCompIMU.Magnetic
    # ifaces.RoboCompIMU.Orientation
    # ifaces.RoboCompIMU.DataImu



