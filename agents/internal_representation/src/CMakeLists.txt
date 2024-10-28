cmake_minimum_required(VERSION 2.6)
PROJECT(  internal_representation )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Lidar3D )
