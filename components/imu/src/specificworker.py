#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2026 by NoeZC
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
"""
Componente imu.

Dos modos de operación, seleccionados por el flag `simulation` del config:

  - simulation=false (por defecto, robot real): lee el Phidget real
    (src/phidget.py) y publica la muestra cruda (accel/gyro/mag, temperatura;
    SIN orientación -- no hay filtro que la estime) tal cual llega, a la tasa
    configurada en IMU.interval_ms (100 Hz por defecto). Sin covarianza
    conocida (el driver no la reporta) -> Cov3 "unknown" (m00=-1), igual que
    hace Webots2Robocomp para lo que tampoco puede medir.

  - simulation=true: NO toca el Phidget. Actúa de proxy hacia webots-bridge
    (Proxies.IMU, `requires IMU`) -- reenvía la MISMA RoboCompIMU.DataImu que
    devuelve el bridge (incluyendo su Cov3, su simTimestamp y su
    simulated=true) por los mismos tres planos que en modo real. Sirve para
    ejercitar el plano DDS zero-copy y el push IceStorm contra la simulación,
    ya que Webots2Robocomp solo expone el pull ICE (implements IMU), sin DDS.

En ambos modos, tres planos de exposición independientes, alimentados por la
misma muestra de cada ciclo de compute() (ambos push pueden estar activos a
la vez; el pull siempre lo está):
  - ICE pull  (implements IMU, getAcceleration/getAngularVel/getDataImu/...) --
    MISMO formato que expone webots-bridge (Webots2Robocomp) en simulación:
    un consumidor con `requires IMU` que ya funciona contra webots-bridge
    (Proxies.IMU apuntando a Webots2Robocomp) funciona igual sin tocar código
    apuntando aquí (Proxies.IMU -> este componente) al pasar a robot real.
  - ICE push  (ice_pub, IMUPub.publish, RoboCompIMU.DataImu)  -- siempre disponible.
  - DDS push  (dds_pub, imu_dds_native, zero-copy FastDDS)    -- opcional, requiere
    el módulo nativo compilado (ver src/CMakeLists.txt); si no está compilado
    el componente sigue funcionando igual, solo sin ese plano (aviso en el log).

Sin DSR: MediaPlaneDDS_getMediaDescriptor() expone el descriptor del plano DDS
para que un agente con grafo (p.ej. robot_concept) lo relaye, igual que hacía
imu_fusion_dds con su plano de pose.
"""

import logging
import math
import os
import re
import sys
import threading
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("Ice", "PySide6"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

console = Console(highlight=False)
log = logging.getLogger(__name__)

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except Exception:
    pass

from src.phidget import PhidgetIMU
# Publicador zero-copy FastDDS (ver src/dds_publisher.h/.cpp, src/imu_dds_bindings.cpp)
# -- opcional en tiempo de import: si el .so no está compilado (o dds_pub=false, ver
# initialize()) el componente sigue funcionando igual, solo sin ese plano adicional.
try:
    from src import imu_dds_native
except ImportError:
    imu_dds_native = None


def _endpoint_port(s: str) -> str | None:
    """Extract the '-p N' port out of an Ice endpoint or proxy string ('tcp -p 10007',
    'imu:tcp -h localhost -p 10007'), or None if not found."""
    m = re.search(r"-p\s+(\d+)", s)
    return m.group(1) if m else None


def _unknown_cov() -> "ifaces.RoboCompIMU.Cov3":
    """Cov3 'unknown' sentinel (m00=-1) -- ver comentario en IMU.idsl. Un cero se
    leería aguas abajo como confianza infinita, así que se marca explícitamente."""
    c = ifaces.RoboCompIMU.Cov3()
    c.m00 = -1.0
    return c


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        if startup_check:
            self.startup_check()
        else:
            if self.initialize():
                self.timer.timeout.connect(self.compute)
                self.timer.start(self.Period)
            else:
                log.error("initialize() falló — componente no arrancado, saliendo")
                sys.exit(1)

    def __del__(self):
        """Destructor"""
        imu = getattr(self, "_imu", None)
        if imu is not None:
            imu.stop()

    def _warn_if_proxy_targets_own_port(self) -> None:
        """simulation=true means Proxies.IMU (this proxy's OWN 'requires IMU') must reach an
        EXTERNAL webots-bridge -- but by convention both webots-bridge and this component's
        own 'implements IMU' (Endpoints.IMU) bind the same port (10007), for interchangeability
        in the non-proxy case. If both land on localhost with the same port here, this
        component silently talks to itself: its own servant answers with whatever is cached
        in self._imu_data (all-zero until a real sample arrives, which in proxy mode never
        happens), so `getDataImu()` "succeeds" with garbage instead of failing loudly. Caught
        this exact silent loop while testing simulation=true without webots-bridge running.
        """
        proxy_str = str(self.configData.get("Proxies", {}).get("IMU", ""))
        endpoint_str = str(self.configData.get("Endpoints", {}).get("IMU", ""))
        same_port = _endpoint_port(proxy_str) is not None and _endpoint_port(proxy_str) == _endpoint_port(endpoint_str)
        localhost = any(h in proxy_str for h in ("localhost", "127.0.0.1"))
        if same_port and localhost:
            log.warning(
                "simulation=true: Proxies.IMU ('%s') apunta al MISMO puerto que este "
                "componente sirve en Endpoints.IMU ('%s') -- si webots-bridge no está "
                "escuchando ahí, este componente se conectará a SÍ MISMO y publicará ceros "
                "sin avisar. Verifica que webots-bridge esté arriba en ese puerto, o usa un "
                "puerto distinto para uno de los dos.",
                proxy_str, endpoint_str)

    # ================================================================ initialize
    def initialize(self) -> bool:
        self.simulation = bool(self.configData.get("simulation", False))
        log.info("=== imu: initialize (simulation=%s) ===", self.simulation)

        self._t_last_sample_recv = 0.0   # reloj de pared local; 0.0 = aún ninguna muestra
        self._fault_logged = False

        self._imu = None
        if self.simulation:
            # Modo proxy: sin driver local, self.imu_proxy (requires IMU, ver .cdsl)
            # apunta a webots-bridge (Proxies.IMU en el config).
            cfg_imu = self.configData.get("IMU", {})
            self._imu_timeout_s = float(cfg_imu.get("timeout_s", 0.2))
            self._warn_if_proxy_targets_own_port()
        else:
            cfg_imu = self.configData.get("IMU", {})
            self._imu_interval_ms = int(cfg_imu.get("interval_ms", 10))   # 100Hz por defecto
            self._imu_timeout_s   = float(cfg_imu.get("timeout_s", 0.2))

            self._imu = PhidgetIMU(data_interval_ms=self._imu_interval_ms)
            if not self._imu.start(timeout_s=5.0):
                log.error("No se pudo conectar el IMU Phidget")
                return False

        # ---- Caché para el plano ICE pull (implements IMU) ----
        # Los servants IMU_get*() corren en el pool de hilos de Ice, en paralelo a
        # compute() (hilo Qt) -- de ahí el lock. Mismo patrón que update_imu_data()/
        # IMU_get*() en Webots2Robocomp (webots-bridge/src/specificworker.cpp).
        self._data_lock = threading.Lock()
        self._imu_data = ifaces.RoboCompIMU.DataImu()   # todo-cero hasta la primera muestra

        # ---- Diagnóstico (rate-limit) ----
        self._diag_interval_s = 2.0
        self._t_last_diag = time.monotonic()
        self._diag_n = 0

        # ---- Plano ICE (IMUPub, siempre disponible) ----
        self.ice_pub = bool(self.configData.get("ice_pub", True))

        # ---- Plano DDS zero-copy (opcional) ----
        self.dds_pub = bool(self.configData.get("dds_pub", False))
        self._dds_pub_handle = None
        if self.dds_pub:
            if imu_dds_native is None:
                log.warning("dds_pub=true pero imu_dds_native no está compilado (ver src/CMakeLists.txt) — desactivando")
                self.dds_pub = False
            else:
                cfg_dds = self.configData.get("DDS", {})
                dds_cfg = imu_dds_native.Config()
                if "Domain" in cfg_dds:            dds_cfg.domain_id = int(cfg_dds["Domain"])
                if "Topic" in cfg_dds:              dds_cfg.topic = str(cfg_dds["Topic"])
                if "HistoryDepth" in cfg_dds:       dds_cfg.history_depth = int(cfg_dds["HistoryDepth"])
                if "SharedMemoryOnly" in cfg_dds:   dds_cfg.shared_memory_only = bool(cfg_dds["SharedMemoryOnly"])
                if "DataSharing" in cfg_dds:        dds_cfg.data_sharing = bool(cfg_dds["DataSharing"])
                self._dds_pub_handle = imu_dds_native.ImuDDSPublisher()
                if not self._dds_pub_handle.init(dds_cfg):
                    log.warning("DDS imu media plane init FAILED — desactivando dds_pub")
                    self._dds_pub_handle = None
                    self.dds_pub = False

        if self.simulation:
            log.info("initialize OK — proxy de webots-bridge (Proxies.IMU) | ice_pub=%s dds_pub=%s",
                      self.ice_pub, self.dds_pub)
        else:
            log.info("initialize OK — Phidget a %.0fHz | ice_pub=%s dds_pub=%s",
                      1000.0 / self._imu_interval_ms, self.ice_pub, self.dds_pub)
        return True

    # ================================================================= compute
    @QtCore.Slot()
    def compute(self) -> None:
        try:
            if self.simulation:
                self._compute_impl_simulation()
            else:
                self._compute_impl_real()
        except Exception:
            log.exception("compute(): excepción no controlada, se descarta este ciclo")

    def _compute_impl_real(self) -> None:
        """Toma la última muestra cruda del Phidget (sin drenar: solo interesa la más
        reciente, no hay integración que perder muestras intermedias rompa) y la
        publica tal cual por los planos activos."""
        sample = self._imu.get_latest()
        if sample is None:
            now = time.monotonic()
            if self._t_last_sample_recv and now - self._t_last_sample_recv > self._imu_timeout_s:
                if not self._fault_logged:
                    log.error("Sin datos del IMU desde hace %.1fs", now - self._t_last_sample_recv)
                    self._fault_logged = True
            return

        self._t_last_sample_recv = time.monotonic()
        self._fault_logged = False

        stamp_ms = int(sample.wall_time_s * 1000.0)   # origen real (ver phidget.py), no de publicación
        cov = _unknown_cov()   # el driver Phidget no reporta varianza por muestra

        data = ifaces.RoboCompIMU.DataImu()
        data.acc = ifaces.RoboCompIMU.Acceleration(
            sample.accel[0], sample.accel[1], sample.accel[2], cov, stamp_ms, 0, False)
        data.gyro = ifaces.RoboCompIMU.Gyroscope(
            sample.gyro[0], sample.gyro[1], sample.gyro[2], cov, stamp_ms, 0, False)
        if any(math.isnan(v) for v in sample.mag):
            data.mag = ifaces.RoboCompIMU.Magnetic(0.0, 0.0, 0.0, cov, stamp_ms, 0, False)
        else:
            data.mag = ifaces.RoboCompIMU.Magnetic(
                sample.mag[0], sample.mag[1], sample.mag[2], cov, stamp_ms, 0, False)
        # Sin ESKF -> sin actitud fiable que publicar (ver docstring del módulo).
        data.rot = ifaces.RoboCompIMU.Orientation(0.0, 0.0, 0.0, cov, stamp_ms, 0, False)
        data.temperature = sample.temperature

        self._publish(data, stamp_ms=stamp_ms, sim_stamp_ms=0, gyro_var=-1.0,
                       diag_extra=lambda: (sample.accel, sample.gyro))

    def _compute_impl_simulation(self) -> None:
        """Modo proxy: pide la muestra a webots-bridge (self.imu_proxy, requires IMU) y
        reenvía exactamente lo que devuelve -- Cov3, simTimestamp y simulated incluidos --
        por los mismos planos que en modo real. Un fallo de conexión (bridge aún no
        levantado, o caído) se trata igual que la falta de datos del Phidget: FAULT
        logueado una vez, sin abortar el ciclo de compute()."""
        try:
            data = self.imu_proxy.getDataImu()
        except Ice.Exception as e:
            now = time.monotonic()
            if self._t_last_sample_recv and now - self._t_last_sample_recv > self._imu_timeout_s:
                if not self._fault_logged:
                    log.error("Sin datos de webots-bridge (Proxies.IMU) desde hace %.1fs (%s)",
                              now - self._t_last_sample_recv, e)
                    self._fault_logged = True
            return

        self._t_last_sample_recv = time.monotonic()
        self._fault_logged = False

        self._publish(data, stamp_ms=data.acc.timestamp, sim_stamp_ms=data.acc.simTimestamp,
                       gyro_var=data.gyro.cov.m00,
                       diag_extra=lambda: ((data.acc.XAcc, data.acc.YAcc, data.acc.ZAcc),
                                            (data.gyro.XGyr, data.gyro.YGyr, data.gyro.ZGyr)))

    def _publish(self, data, *, stamp_ms: int, sim_stamp_ms: int, gyro_var: float, diag_extra) -> None:
        with self._data_lock:
            self._imu_data = data

        if self.ice_pub:
            try:
                self.imupub_proxy.publish(data)
            except Ice.Exception as e:
                log.warning("ice_pub: fallo al publicar (%s)", e)

        if self.dds_pub and self._dds_pub_handle is not None:
            try:
                self._dds_pub_handle.publish(
                    stamp_ms, sim_stamp_ms,
                    data.acc.XAcc, data.acc.YAcc, data.acc.ZAcc,
                    data.gyro.XGyr, data.gyro.YGyr, data.gyro.ZGyr,
                    data.mag.XMag, data.mag.YMag, data.mag.ZMag,
                    data.rot.Roll, data.rot.Pitch, data.rot.Yaw,
                    data.temperature, gyro_var)
            except Exception as e:
                log.warning("dds_pub: fallo al publicar (%s)", e)

        self._diag_n += 1
        now = time.monotonic()
        if now - self._t_last_diag > self._diag_interval_s:
            hz = self._diag_n / (now - self._t_last_diag)
            accel, gyro = diag_extra()
            log.info("IMU: %.1f Hz | ax=%.3f ay=%.3f az=%.3f wx=%.4f wy=%.4f wz=%.4f",
                      hz, accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2])
            self._diag_n = 0
            self._t_last_diag = now

    def startup_check(self):
        print(f"Testing RoboCompIMU.Cov3 from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Cov3()
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

    #
    # IMPLEMENTATION of getMediaDescriptor method from MediaPlaneDDS interface
    #
    def MediaPlaneDDS_getMediaDescriptor(self) -> str:
        """Descriptor JSON del plano DDS, para que un agente con grafo lo relaye. "" si dds_pub=false."""
        if self._dds_pub_handle is not None:
            return self._dds_pub_handle.descriptor_json()
        return ""

    #
    # IMU (pull, ICE) -- mismo formato que Webots2Robocomp.IMU_get*() en webots-bridge,
    # para que Proxies.IMU pueda apuntar indistintamente a la simulación o a este
    # componente. Todos devuelven una copia de la última muestra cacheada en compute()
    # (la del Phidget en modo real, o la reenviada de webots-bridge en modo simulación).
    #
    def IMU_getAcceleration(self):
        with self._data_lock:
            return self._imu_data.acc

    def IMU_getAngularVel(self):
        with self._data_lock:
            return self._imu_data.gyro

    def IMU_getDataImu(self):
        with self._data_lock:
            return self._imu_data

    def IMU_getMagneticFields(self):
        with self._data_lock:
            return self._imu_data.mag

    def IMU_getOrientation(self):
        with self._data_lock:
            return self._imu_data.rot

    def IMU_resetImu(self):
        # Sin filtro que resetear (sin ESKF) -- no-op documentado, ver docstring del módulo.
        log.info("IMU_resetImu(): no-op (componente sin filtro/estado que resetear)")

    # ===================================================================
    # ===================================================================

    ######################
    # From the RoboCompIMUPub you can publish calling this methods:
    # RoboCompIMUPub.void self.imupub_proxy.publish(RoboCompIMU.DataImu imu)
