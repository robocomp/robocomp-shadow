cmake_minimum_required(VERSION 2.6)
PROJECT(  SemanticSegmentation360 )

INCLUDE( /home/robolab/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB CameraRGBDSimple CameraSimple Person SemanticSegmentation VisualElements )
