cmake_minimum_required(VERSION 2.6)
PROJECT(  vibrator_test )

INCLUDE( /opt/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase IMU JoystickAdapter OmniRobot )
