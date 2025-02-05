cmake_minimum_required(VERSION 2.6)
PROJECT(  SVD48VBase )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior DifferentialRobot GenericBase GenericBase JoystickAdapter OmniRobot )
