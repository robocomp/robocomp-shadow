cmake_minimum_required(VERSION 2.6)
PROJECT(  controller_orin )

INCLUDE( /opt/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB MaskElements SegmentatorTrackingPub SegmentatorTrackingPub VisualElements VisualElements )
