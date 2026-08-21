# imu

Componente RoboComp puro (sin grafo DSR) que publica la muestra **cruda** del
IMU (aceleración, giroscopio, campo magnético, temperatura), sin fusión ni
filtro alguno. Copia depurada de `imu_fusion_dds` (ver ese componente para la
versión con ESKF + fusión de odometría): aquí se ha eliminado el ESKF y la
suscripción a la odometría de la base — solo queda la adquisición y la
publicación del dato en bruto, en dos modos seleccionados por `simulation`
en `etc/config`:

- **`simulation = false`** (por defecto, robot real): lee el IMU Phidget
  Spatial Precision 3/3/3 conectado por USB.
- **`simulation = true`**: no toca el Phidget; actúa de **proxy** hacia
  `webots-bridge` (`Webots2Robocomp`) vía `requires IMU` (`Proxies.IMU`) y
  reenvía tal cual la `RoboCompIMU.DataImu` que este devuelve (incluida su
  `Cov3`, `simTimestamp` y `simulated=true`) por los mismos planos de abajo.
  Sirve para ejercitar el plano DDS zero-copy y el push IceStorm contra la
  simulación, ya que `Webots2Robocomp` solo expone el pull ICE
  (`implements IMU`), sin DDS.

No hay orientación (`Roll`/`Pitch`/`Yaw`) fiable que publicar sin un filtro
de actitud: se sirve siempre a 0. Un consumidor que necesite pose debe
fusionar esto él mismo (ver `imu_fusion_dds`, que apunta su `IMU.interval_ms`
a la misma tasa por la misma razón — la integración de orientación por mapa
exponencial SO(3) del ESKF es exacta independientemente del tamaño de paso;
ver `room_concept_rotation.md` en el repo de notas del robot).

## Publicación: dos planos independientes

Cada uno se activa con su propio flag booleano en `etc/config`, ambos pueden
estar activos a la vez:

- **`ice_pub`** — `RoboCompIMUPub` (IceStorm, topic `Proxies.IMUPubPrefix`):
  `RoboCompIMU.DataImu` completo, un mensaje por ciclo de `compute()`.
  Siempre disponible, sin dependencias adicionales.
- **`dds_pub`** — plano zero-copy FastDDS (`imu_dds_native`, ver
  `src/dds_publisher.h/.cpp`, `src/imu_dds_bindings.cpp`, `src/CMakeLists.txt`),
  reutilizando `ImuFrame`/`rc::media::ImuPublisher` de
  `active_inference/common/media_transport` (el mismo tipo que ya consumen
  `lidar3d_dds`/`zed_camera`/`ricoh_omni_dds` para sus planos de medios). Si
  el módulo nativo no está compilado, el componente sigue funcionando igual
  (aviso en el log, `dds_pub` se desactiva solo en tiempo de ejecución).
  `MediaPlaneDDS_getMediaDescriptor()` expone el descriptor JSON del plano
  para que un agente con grafo (p.ej. `robot_concept`) lo relaye a DSR.

## Dependencias

- Paquete Python `Phidget22` (`pip install Phidget22`) + drivers nativos de
  Phidgets instalados en el sistema (ver
  [docs de Phidgets para Linux](https://www.phidgets.com/docs/OS_-_Linux)).
- El IMU Phidget Spatial Precision 3/3/3 físicamente conectado.
- Opcional (`dds_pub=true`): `imu_dds_native` compilado — requiere FastDDS/
  FastCDR instalados (`/usr/local/lib`, ver `ldconfig -p`) y `pybind11`
  (paquete `python3-pybind11` o `pip install pybind11`, con el `Config.cmake`
  visible a CMake).

## Timestamps

`RoboCompIMU.Acceleration/Gyroscope/Magnetic/Orientation.timestamp` y
`ImuFrame.stamp_ms` llevan el reloj de pared tomado en el propio callback del
driver Phidget (`phidget.py::_on_data`, campo `IMUSample.wall_time_s`), NO el
instante en que `compute()` lo lee/publica — evita reintroducir el problema
de "processing time vs. origin time" ya depurado en `imu_fusion` (ver
`room_concept_rotation.md`).

## Lanzamiento

```
bin/imu etc/config
```
