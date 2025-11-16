
find_package(Qt6 COMPONENTS 3DCore 3DRender 3DExtras REQUIRED)



# Set LibTorch path
set(CMAKE_PREFIX_PATH "/absolute/path/to/libtorch")

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
)


LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}" tbb Qt63DCore Qt63DRender Qt63DExtras Qt6PrintSupport)

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O4 -fmax-errors=1 -fno-char8_t)

