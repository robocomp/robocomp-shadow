cmake_minimum_required(VERSION 2.6)
PROJECT(  yolov8_360 )

INCLUDE( /home/robolab/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior ByteTrack Camera360RGB CameraRGBDSimple CameraRGBDSimple CameraRGBDSimple YoloObjects )
