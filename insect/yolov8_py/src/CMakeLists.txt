cmake_minimum_required(VERSION 2.6)
PROJECT(  yolov8_py )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior ByteTrack CameraRGBDSimple CameraRGBDSimple CameraRGBDSimple YoloObjects )
