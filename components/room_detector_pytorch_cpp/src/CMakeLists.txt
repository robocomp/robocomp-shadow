

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
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
)


LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}" tbb Qt6PrintSupport)

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O4 -fmax-errors=1 -fno-char8_t)

