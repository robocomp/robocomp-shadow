cmake_minimum_required(VERSION 2.6)
PROJECT(  internal_representation )

IF(NOT EXISTS $ENV{ROBOCOMP}/cmake)
  MESSAGE(FATAL_ERROR "Couldn't find RoboComp in $ENV{ROBOCOMP}! Exiting...")
ENDIF(NOT EXISTS $ENV{ROBOCOMP}/cmake)

INCLUDE( $ENV{ROBOCOMP}/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Lidar3D )
