
find_package(Qt6 COMPONENTS 3DCore 3DRender 3DExtras REQUIRED)



# Set LibTorch path
set(CMAKE_PREFIX_PATH "/home/robocomp/software/libtorch")

# Set the CUDA architecture list for PyTorch
set(CMAKE_CUDA_ARCHITECTURES 89)
set(TORCH_CUDA_ARCH_LIST "8.9" CACHE STRING "CUDA architectures for PyTorch")

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Colors for messages
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColorReset "${Esc}[m")
    set(ColorBold  "${Esc}[1m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
endif()

message(STATUS "${Green}Configuring YOLOv11 ONNX Detector${ColorReset}")

# Set ONNX Runtime path
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "Path to ONNX Runtime installation")

# Check if ONNX Runtime exists
if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "${Red}ONNX Runtime not found at: ${ONNXRUNTIME_ROOT}${ColorReset}\n"
            "Please install it with: sudo ./install_onnxruntime.sh cpu")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(FATAL_ERROR "${Red}ONNX Runtime libraries not found in: ${ONNXRUNTIME_ROOT}/lib${ColorReset}")
endif()

message(STATUS "${Green}Found ONNX Runtime at: ${ONNXRUNTIME_ROOT}${ColorReset}")


# Include directories
include_directories(
        ${ONNXRUNTIME_ROOT}/include
)
# Link directories
link_directories(
        ${ONNXRUNTIME_ROOT}/lib
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
        ../src/room_model.cpp
        ../src/time_series_plotter.cpp
        ../src/qcustomplot.cpp
        ../src/door_detector.cpp
        ../src/room_freezing_manager.cpp
        ../src/qt3d_visualizer.cpp
        ../src/room_optimizer.cpp
        ../src/uncertainty_manager.cpp
        ../src/room_loss.cpp
        ../src/uncertainty_estimator.cpp
        ../src/yolo_detector_onnx.cpp
        ../src/model_based_filter.cpp
        ../src/pointcloud_center_estimator.cpp
        ../src/door_concept.cpp
        ../src/door_model.cpp
        ../src/equirectangular_projection.cpp
        ../src/door_projection.cpp
)
# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
        ../src/room_model.h
        ../src/time_series_plotter.h
        ../src/door_detector.h
        ../src/room_freezing_manager.h
        ../src/qt3d_visualizer.h
        ../src/room_optimizer.h
        ../src/uncertainty_manager.h
        ../src/room_loss.h
        ../src/uncertainty_estimator.h
        ../src/yolo_detector_onnx.h
        ../src/model_based_filter.h
        ../src/pointcloud_center_estimator.h
        ../src/door_concept.h
        ../src/door_model.h
        ../src/equirectangular_projection.h
        ../src/door_projection.h
)

# Include OpenCV
INCLUDE($ENV{ROBOCOMP}/cmake/modules/opencv4.cmake)

LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}" tbb Qt63DCore Qt63DRender Qt63DExtras Qt6PrintSupport onnxruntime )

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-g -O4 -fmax-errors=1 -fno-char8_t)

