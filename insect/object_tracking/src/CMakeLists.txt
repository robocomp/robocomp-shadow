cmake_minimum_required(VERSION 2.6)
PROJECT(  object_tracking )

INCLUDE( /opt/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB Camera360RGB Camera360RGBD Person SegmentatorTrackingPub VisualElements VisualElements )
