cmake_minimum_required(VERSION 2.6)
PROJECT(  pybullet_dump )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase IMU JoystickAdapter OmniRobot )


# Set C++ standard
set(CMAKE_CXX_STANDARD 23)

# Add compiler definitions
add_definitions(-O0 -g -fmax-errors=1 -fno-char8_t -fopenmp -Wno-unused-function -Wno-cpp)