cmake_minimum_required(VERSION 2.6)
PROJECT(  component_test )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior FullPoseEstimation GenericBase IMU JoystickAdapter OmniRobot )
