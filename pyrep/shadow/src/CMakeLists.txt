cmake_minimum_required(VERSION 2.6)
PROJECT(  shadowPyrep )

INCLUDE( $ENV{ROBOCOMP}/cmake/robocomp.cmake )
SET(IFACES "/ifaces")
ROBOCOMP_IDSL_TO_ICE( CommonBehavior CameraRGBDSimple GenericBase JoystickAdapter Laser OmniRobot HumanToDSRPub CoppeliaUtils FullPoseEstimation BatteryStatus RSSIStatus JointMotorSimple CameraSimple BillCoppelia Camera360RGB)

