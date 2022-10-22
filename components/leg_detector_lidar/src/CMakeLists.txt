cmake_minimum_required(VERSION 2.6)
PROJECT(  leg_detector_lidad )

INCLUDE( /opt/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase Laser LegDetector2DLidar )
