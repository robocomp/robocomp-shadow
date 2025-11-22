cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(test_room_model)

#set(CMAKE_CXX_STANDARD 20)

# Set LibTorch path
set(CMAKE_PREFIX_PATH "/home/pbustos/software/libtorch")

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create test executable
add_executable(test_room_model 
    test_detector.cpp
    room_model.cpp
)

# Link against LibTorch
target_link_libraries(test_room_model "${TORCH_LIBRARIES}")

message(STATUS "LibTorch found: ${TORCH_LIBRARIES}")
