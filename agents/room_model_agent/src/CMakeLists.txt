cmake_minimum_required(VERSION 2.6)
PROJECT(  room_model_agent )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior JoystickAdapter )
