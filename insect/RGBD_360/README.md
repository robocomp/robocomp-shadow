# RGBD_360

## Descripción

RGBD_360 es un componente de fusión sensorial que combina datos de una **cámara RGB 360°** con datos de un **LiDAR 3D** para generar imágenes RGB-D (RGB + Depth) y nubes de puntos coloreadas. El componente sincroniza temporalmente ambos flujos de datos usando timestamps para obtener una fusión precisa.

### Funcionalidades principales

- **Fusión RGB-LiDAR**: Combina datos de profundidad del LiDAR con imágenes RGB de la cámara 360°
- **Sincronización temporal**: Utiliza buffers circulares para sincronizar datos de sensores con diferentes frecuencias de muestreo
- **Extracción de ROI**: Permite extraer regiones de interés (ROI) con manejo especial para imágenes panorámicas (wrapping horizontal)
- **Nube de puntos coloreada**: Genera nubes de puntos 3D con información de color RGB
- **Thread-safe**: Implementa acceso concurrente seguro mediante `shared_mutex`

### Arquitectura

```
┌─────────────────┐     ┌──────────────────┐
│  Camera360RGB   │────▶│                  │
│    (Proxy)      │     │     RGBD_360     │────▶ Camera360RGBD (Interface)
└─────────────────┘     │   (Fusion Node)  │
                        │                  │────▶ Lidar3D (Interface)
┌─────────────────┐     │                  │      - getColorCloudData()
│    Lidar3D      │────▶│                  │
│    (Proxy)      │     └──────────────────┘
└─────────────────┘
```

## Dependencies

Las siguientes dependencias son necesarias para compilar y ejecutar RGBD_360:

### Sistema
```bash
# Qt6
sudo apt install qt6-base-dev qt6-declarative-dev qt6-scxml-dev libqt6statemachineqml6 libqt6statemachine6

# OpenCV
sudo apt install libopencv-dev

# Boost
sudo apt install libboost-all-dev

# Ice (ZeroC Ice)
sudo apt install zeroc-ice-all-dev
```

### Bibliotecas adicionales
```bash
# toml++ (para archivos de configuración)
mkdir ~/software 2> /dev/null
git clone https://github.com/marzer/tomlplusplus.git ~/software/tomlplusplus
cd ~/software/tomlplusplus && cmake -B build && sudo make install -C build -j12

# libQGLViewer (opcional, para visualización)
git clone https://github.com/GillesDebunne/libQGLViewer.git ~/software/libQGLViewer
cd ~/software/libQGLViewer && qmake6 *.pro && make -j12 && sudo make install && sudo ldconfig
```

### RoboComp
- RoboComp framework instalado y configurado
- Interfaces: `Camera360RGB`, `Camera360RGBD`, `Lidar3D`

## Configuration parameters

El componente requiere un archivo de configuración para iniciar. En `etc/config` o `etc/config.toml` encontrarás ejemplos de configuración.

### Parámetros principales

| Parámetro | Tipo | Descripción | Ejemplo |
|-----------|------|-------------|---------|
| `Endpoints.Camera360RGBD` | string | Endpoint del servidor Camera360RGBD | `"tcp -p 10100"` |
| `Endpoints.Lidar3D` | string | Endpoint del servidor Lidar3D | `"tcp -h localhost -p 12001"` |
| `Proxies.Camera360RGB` | string | Proxy para conectar a la cámara RGB 360° | `"camera360rgb:tcp -h localhost -p 10097"` |
| `Proxies.Lidar3D` | string | Proxy para conectar al LiDAR | `"lidar3d:tcp -h localhost -p 11990"` |
| `display` | bool | Mostrar visualización OpenCV | `False` |
| `lidar` | string | Nombre del sensor LiDAR | `"helios"` |
| `Period.Compute` | int | Periodo del ciclo de cómputo en ms | `30` |
| `Period.Emergency` | int | Periodo del estado de emergencia en ms | `500` |

### Configuración de Ice (Importante)

Para mensajes grandes (imágenes y nubes de puntos), es necesario configurar correctamente el tamaño máximo de mensaje:

```ini
Ice.MessageSizeMax = 20004800    # ~20MB - Requerido para imágenes de alta resolución
Ice.Warn.Connections = 0         # Desactivar warnings de conexión
Ice.Trace.Network = 0            # Desactivar trazas de red
Ice.Trace.Protocol = 0           # Desactivar trazas de protocolo
```

> ⚠️ **Error común**: Si ves el error `Ice::MemoryLimitException: protocol error: memory limit exceeded`, incrementa el valor de `Ice.MessageSizeMax` en el archivo de configuración.

### Ejemplo de archivo de configuración completo

```ini
# Endpoints para interfaces implementadas
Endpoints.Camera360RGBD = "tcp -p 10100"
Endpoints.Lidar3D = "tcp -h localhost -p 12001"

# Proxies para interfaces requeridas
Proxies.Camera360RGB = "camera360rgb:tcp -h localhost -p 10097"
Proxies.Lidar3D = "lidar3d:tcp -h localhost -p 11990"

# Parámetros del componente
display = False
lidar = "helios"

# Configuración de Ice
Ice.Warn.Connections = 0
Ice.Trace.Network = 0
Ice.Trace.Protocol = 0
Ice.MessageSizeMax = 20004800

# Periodos de estado
Period.Compute = 30
Period.Emergency = 500
```

## Starting the component
To avoid modifying the config file directly in the repository, you can copy it to the component's home directory. This prevents changes from being overridden by future `git pull` commands:

```bash
cd <RGBD_360's path> 
cp etc/config etc/yourConfig
```

After editing the new config file we can run the component:

```bash
cmake -B build && make -C build -j12 # Compile the component
bin/RGBD_360 etc/yourConfig # Execute the component
```

## API Reference

### Interfaces Implementadas

#### Camera360RGBD

Proporciona imágenes RGB-D fusionadas con capacidad de extracción de ROI.

```cpp
TRGBD getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)
```

**Parámetros:**
| Parámetro | Descripción |
|-----------|-------------|
| `cx` | Centro X de la ROI (-1 para centrar automáticamente) |
| `cy` | Centro Y de la ROI (-1 para centrar automáticamente) |
| `sx` | Ancho de la ROI a extraer (-1 para imagen completa) |
| `sy` | Alto de la ROI a extraer (-1 para imagen completa) |
| `roiwidth` | Ancho final de la imagen de salida (-1 para mantener) |
| `roiheight` | Alto final de la imagen de salida (-1 para mantener) |

**Retorna:** `TRGBD` con:
- `rgb`: Imagen RGB (bytes)
- `depth`: Imagen de profundidad (3 canales float: X, Y, Z por pixel)
- `width`, `height`: Dimensiones
- `alivetime`: Timestamp de captura
- `period`: Periodo actual de fusión

**Manejo especial de imágenes panorámicas:**
El componente maneja automáticamente el "wrap-around" horizontal cuando la ROI cruza los límites de la imagen 360°.

#### Lidar3D (Subconjunto implementado)

```cpp
TColorCloudData getColorCloudData()
```

**Retorna:** Nube de puntos coloreada con:
- `X`, `Y`, `Z`: Coordenadas 3D (int16_t)
- `R`, `G`, `B`: Color RGB por punto
- `timestamp`: Marca temporal
- `numberPoints`: Número de puntos

### Interfaces Requeridas

El componente necesita conectarse a:

- **Camera360RGB**: Cámara panorámica RGB
  - `getROI()`: Obtener imagen RGB
  
- **Lidar3D**: Sensor LiDAR 3D
  - `getLidarDataArrayProyectedInImage()`: Datos LiDAR proyectados en coordenadas de imagen

## Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RGBD_360 Component                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────┐         ┌───────────────┐                      │
│   │ Camera360RGB  │         │    Lidar3D    │                      │
│   │    Proxy      │         │     Proxy     │                      │
│   └───────┬───────┘         └───────┬───────┘                      │
│           │                         │                               │
│           ▼                         ▼                               │
│   ┌───────────────┐         ┌───────────────┐                      │
│   │ Circular Buf. │         │ Circular Buf. │                      │
│   │  (3 frames)   │         │  (1 frame)    │                      │
│   └───────┬───────┘         └───────┬───────┘                      │
│           │                         │                               │
│           └────────────┬────────────┘                               │
│                        ▼                                            │
│              ┌─────────────────┐                                    │
│              │   Timestamp     │                                    │
│              │   Matching      │   MAX_DIFF: 500µs                  │
│              └────────┬────────┘                                    │
│                       ▼                                             │
│              ┌─────────────────┐                                    │
│              │  Data Fusion    │                                    │
│              │  - RGB Image    │                                    │
│              │  - Depth Image  │                                    │
│              │  - Color Cloud  │                                    │
│              └────────┬────────┘                                    │
│                       ▼                                             │
│   ┌────────────────────────────────────────┐                       │
│   │          shared_mutex Buffer           │                       │
│   │  - rgb_frame_write                     │                       │
│   │  - depth_frame_write                   │                       │
│   │  - pointCloud                          │                       │
│   └────────────────────────────────────────┘                       │
│                       │                                             │
│           ┌───────────┴───────────┐                                │
│           ▼                       ▼                                │
│   ┌───────────────┐       ┌───────────────┐                        │
│   │Camera360RGBD  │       │    Lidar3D    │                        │
│   │  Interface    │       │   Interface   │                        │
│   │  getROI()     │       │getColorCloud()│                        │
│   └───────────────┘       └───────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Sincronización Temporal

El componente implementa un algoritmo de sincronización por timestamp:

1. **Buffers circulares**: 
   - Cámara: 3 frames
   - LiDAR: 1 frame

2. **Matching**: Busca el par de frames con menor diferencia de timestamp
   - Umbral máximo: 500 microsegundos
   - Búsqueda bidireccional en los buffers

3. **Fusión**: Solo se fusionan datos con timestamps suficientemente cercanos
-----
-----
# Developer Notes
This section explains how to work with the generated code of RGBD_360, including what can be modified and how to use key features.
## Editable Files
You can freely edit the following files:
- etc/* – Configuration files
- src/* – Component logic and implementation
- README.md – Documentation

The `generated` folder contains autogenerated files. **Do not edit these files directly**, as they will be overwritten every time the component is regenerated with RoboComp.

## ConfigLoader
The `ConfigLoader` simplifies fetching configuration parameters. Use the `get<>()` method to retrieve parameters from the configuration file.
```C++
// Syntax
type variable = this->configLoader.get<type>("ParameterName");

// Example
int computePeriod = this->configLoader.get<int>("Period.Compute");
```

## StateMachine
RoboComp components utilize a state machine to manage the main execution flow. The default states are:

1. **Initialize**:
    - Executes once after the constructor.
    - May use for parameter initialization, opening devices, and calculating constants.
2. **Compute**:
    - Executes cyclically after Initialize.
    - Place your functional logic here. If an emergency is detected, call goToEmergency() to transition to the Emergency state.
3. **Emergency**:
    - Executes cyclically during emergencies.
    - Once resolved, call goToRestore() to transition to the Restore state.
4. **Restore**:
    - Executes once to restore the component after an emergency.
    - Transitions automatically back to the Compute state.

### Setting and Getting State Periods
You can get the period of some state with de function `getPeriod` and set with `setPeriod`
```C++
int currentPeriod = getPeriod("Compute");   // Get the current Compute period
setPeriod("Compute", currentPeriod * 0.5); // Set Compute period to half
```

### Creating Custom States
To add a custom state, follow these steps in the constructor:
1. **Define Your State** Use `GRAFCETStep` to create your state. If any function is not required, use `nullptr`.

```C++
states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
                                                      std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
                                                      std::bind(&SpecificWorker::customEnter, this), // On-enter function
                                                      std::bind(&SpecificWorker::customExit, this)); // On-exit function

```
2. **Define Transitions** Add transitions between states using `addTransition`. You can trigger transitions using Qt signals such as `entered()` and `exited()` or custom signals in .h.
```C++
// Syntax
states[srcState]->addTransition(originOfSignal, signal, dstState)

// Example
states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get());

```
3. **Add State to the StateMachine** Include your state in the state machine:
```C++
statemachine.addState(states["CustomState"].get());

```

## Hibernation Flag
The `#define HIBERNATION_ENABLED` flag in `specificworker.h` activates hibernation mode. When enabled, the component reduces its state execution frequency to 500ms if no method calls are received within 5 seconds. Once a method call is received, the period is restored to its original value.

Default hibernation monitoring runs every 500ms.

## Changes Introduced in the New Code Generator
If you’re regenerating or adapting old components, here’s what has changed:

- Deprecated classes removed: `CommonBehavior`, `InnerModel`, `AGM`, `Monitors`, and `src/config.h`.
- Configuration parsing replaced with the new `ConfigLoader`, supporting both .`toml` and legacy configuration formats.
- Skeleton code split: `generated` (non-editable) and `src` (editable).
- Component period is now configurable in the configuration file.
- State machine integrated with predefined states: `Initialize`, `Compute`, `Emergency`, and `Restore`.
- With the `dsr` option, you generate `G` in the GenericWorker, making the viewer independent. If you want to use the `dsrviewer`, you will need the `Qt GUI (QMainWindow)` and the `dsr` option enabled in the **CDSL**.
- Strings in the legacy config now need to be enclosed in quotes (`""`).

## Adapting Old Components
To adapt older components to the new structure:

1. **Add** `Period.Compute` and `Period.Emergency` and swap Endpoints and Proxies with their names in the `etc/config` file.
2. **Merge** the new `src/CMakeLists.txt` and the old `CMakeListsSpecific` files.
3. **Modify** `specificworker.h`:
    - Add the `HIBERNATION_ENABLED` flag.
    - Update the constructor signature.
    - Replace `setParams` with state definitions (`Initialize`, `Compute`, etc.).
4. **Modify** `specificworker.cpp`:
    - Refactor the constructor entirely.
    - Move `setParams` logic to the `initialize` state using `ConfigLoader.get<>()`.
    - Remove the old timer and period logic and replace it with `getPeriod()` and `setPeriod()`.
    - Add the new function state `Emergency`, and `Restore`.
    - Add the following code to the implements and publish functions:
        ```C++
        #ifdef HIBERNATION_ENABLED
            hibernation = true;
        #endif
        ```
5. **Update Configuration Strings**, ensure all strings in the `config` under legacy are enclosed in quotes (`""`), as required by the new structure.
6. **Using DSR**, if you use the DSR option, note that `G` is generated in `GenericWorker`, making the viewer independent. However, to use the `dsrviewer`, you must integrate a `Qt GUI (QMainWindow)` and enable the `dsr` option in the **CDSL**. 
7. **Installing toml++**, to use the new .toml configuration format, install the toml++ library:
```bash
mkdir ~/software 2> /dev/null; git clone https://github.com/marzer/tomlplusplus.git ~/software/tomlplusplus
cd ~/software/tomlplusplus && cmake -B build && sudo make install -C build -j12 && cd -
```
8. **Installing qt6 Dependencies**
```bash
sudo apt install qt6-base-dev qt6-declarative-dev qt6-scxml-dev libqt6statemachineqml6 libqt6statemachine6

mkdir ~/software 2> /dev/null; git clone https://github.com/GillesDebunne/libQGLViewer.git ~/software/libQGLViewer
cd ~/software/libQGLViewer && qmake6 *.pro && make -j12 && sudo make install && sudo ldconfig && cd -
```
9. **Generated Code**, When the component is generated, a `generated` folder is created containing non-editable files. You can delete everything in the `src` directory except for:
- `src/specificworker.h`
- `src/specificworker.cpp`
- `src/CMakeLists.txt`
- `src/mainUI.ui`
- `README.md`
- `etc/config`
- `etc/config.toml`
- Your Clases...

-----
-----
# Troubleshooting

## Errores comunes

### Ice::MemoryLimitException

```
Ice::MemoryLimitException: protocol error: memory limit exceeded:
requested 5529691 bytes, maximum allowed is 1048576 bytes (see Ice.MessageSizeMax)
```

**Causa**: El tamaño del mensaje Ice (imágenes, nubes de puntos) excede el límite predeterminado.

**Solución**: Añadir o incrementar `Ice.MessageSizeMax` en el archivo de configuración:

```ini
Ice.MessageSizeMax = 20004800
```

> 💡 El valor debe ser suficiente para la resolución de imagen más grande que se espere. Para imágenes 4K panorámicas, valores de 20-50MB son recomendados.

### Warning: numeric property Ice.Warn.Connections set to non-numeric value

**Causa**: En archivos de configuración legacy, los valores numéricos en Ice deben ser strings.

**Solución**: Asegúrate de usar comillas:

```ini
Ice.Warn.Connections = "0"
```

### Sensor no disponible (waiting loop infinito)

**Causa**: El componente espera indefinidamente a que los sensores estén disponibles.

**Solución**: 
1. Verificar que los componentes de cámara y LiDAR están ejecutándose
2. Verificar los endpoints en la configuración
3. Comprobar conectividad de red

### No hay fusión de datos

**Causa**: Los timestamps de cámara y LiDAR difieren más de 500µs.

**Solución**:
1. Verificar sincronización de relojes del sistema
2. Verificar que ambos sensores están publicando datos
3. Aumentar el tamaño de los buffers circulares si es necesario

## Depuración

Activar la visualización para depuración:

```ini
display = True
```

Esto mostrará ventanas OpenCV con las imágenes RGB y de profundidad fusionadas.

-----
-----
# Ejemplos de Uso

## Cliente Python

```python
import RoboCompCamera360RGBD
import Ice
import numpy as np

# Conexión al componente
ic = Ice.initialize()
proxy = ic.stringToProxy("camera360rgbd:tcp -h localhost -p 10100")
camera = RoboCompCamera360RGBD.Camera360RGBDPrx.checkedCast(proxy)

# Obtener imagen completa
result = camera.getROI(-1, -1, -1, -1, -1, -1)

# Convertir a numpy arrays
rgb = np.frombuffer(result.rgb, dtype=np.uint8).reshape(result.height, result.width, 3)
depth = np.frombuffer(result.depth, dtype=np.float32).reshape(result.height, result.width, 3)

print(f"RGB shape: {rgb.shape}")
print(f"Depth shape: {depth.shape}")  # Canales: X, Y, Z
```

## Cliente C++

```cpp
#include <Ice/Ice.h>
#include "Camera360RGBD.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    Ice::CommunicatorPtr ic = Ice::initialize(argc, argv);
    
    auto proxy = ic->stringToProxy("camera360rgbd:tcp -h localhost -p 10100");
    auto camera = RoboCompCamera360RGBD::Camera360RGBDPrx::checkedCast(proxy);
    
    // Obtener ROI centrada de 640x480
    auto result = camera->getROI(-1, -1, 640, 480, 640, 480);
    
    // Convertir a cv::Mat
    cv::Mat rgb(result.height, result.width, CV_8UC3, 
                const_cast<unsigned char*>(result.rgb.data()));
    cv::Mat depth(result.height, result.width, CV_32FC3, 
                  const_cast<unsigned char*>(result.depth.data()));
    
    cv::imshow("RGB", rgb);
    cv::waitKey(0);
    
    ic->destroy();
    return 0;
}
```

## Obtener Nube de Puntos Coloreada

```cpp
#include "Lidar3D.h"

// Conectar al endpoint Lidar3D del componente RGBD_360
auto lidarProxy = RoboCompLidar3D::Lidar3DPrx::checkedCast(
    ic->stringToProxy("lidar3d:tcp -h localhost -p 12001"));

auto cloud = lidarProxy->getColorCloudData();

// Iterar sobre los puntos
for (size_t i = 0; i < cloud.numberPoints; ++i) {
    float x = cloud.X[i];
    float y = cloud.Y[i];
    float z = cloud.Z[i];
    uint8_t r = cloud.R[i];
    uint8_t g = cloud.G[i];
    uint8_t b = cloud.B[i];
    
    // Procesar punto coloreado...
}
```

-----
-----
# Estructura de Datos

## TRGBD (Camera360RGBD)

```cpp
struct TRGBD {
    bool rgbcompressed;       // Si RGB está comprimido
    bool depthcompressed;     // Si profundidad está comprimida
    int cameraID;             // ID de cámara
    int width, height;        // Dimensiones de imagen
    int rgbchannels;          // Canales RGB (típicamente 3)
    int depthchannels;        // Canales profundidad (3: X,Y,Z)
    int focalx, focaly;       // Parámetros focales
    long alivetime;           // Timestamp de captura
    float period;             // Periodo de fusión actual
    ImgType rgb;              // Datos RGB (bytes)
    ImgType depth;            // Datos profundidad (float32 x 3)
    TRoi roi;                 // Información de ROI
};
```

## TColorCloudData (Lidar3D)

```cpp
struct TColorCloudData {
    sequence<int16_t> X, Y, Z;  // Coordenadas 3D
    sequence<uint8_t> R, G, B;  // Colores RGB
    long timestamp;             // Marca temporal
    int numberPoints;           // Número de puntos
    bool compressed;            // Si está comprimido
};
```

-----
-----
# Licencia

Este componente es parte de RoboComp y está licenciado bajo la GNU General Public License v3.0.

```
RoboComp is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

