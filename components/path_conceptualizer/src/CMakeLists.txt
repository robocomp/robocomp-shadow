cmake_minimum_required(VERSION 2.6)
PROJECT( controller )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior CameraRGBDSimple CameraSimple GenericBase MPC OmniRobot SemanticSegmentation YoloObjects )
