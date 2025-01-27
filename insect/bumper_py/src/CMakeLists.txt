cmake_minimum_required(VERSION 2.6)
PROJECT(  bumper_py )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase GridPlanner JoystickAdapter Lidar3D OmniRobot OmniRobot SegmentatorTrackingPub VisualElementsPub )
