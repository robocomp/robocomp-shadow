cmake_minimum_required(VERSION 2.6)
PROJECT(  semantic_segmentation )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB CameraRGBDSimple CameraSimple Person SemanticSegmentation VisualElements )
