cmake_minimum_required(VERSION 2.6)
PROJECT(  shadowPyrep )

INCLUDE( $ENV{ROBOCOMP}/cmake/robocomp.cmake )
SET(IFACES "/ifaces")
ROBOCOMP_IDSL_TO_ICE( ${IFACES}/CommonBehavior ${IFACES}/CameraRGBDSimple ${IFACES}/GenericBase ${IFACES}/JoystickAdapter ${IFACES}/Laser ${IFACES}/DifferentialRobot ${IFACES}/HumanToDSRPub ${IFACES}/CoppeliaUtils ${IFACES}/FullPoseEstimation ${IFACES}/BatteryStatus ${IFACES}/RSSIStatus ${IFACES}/JointMotorSimple ${IFACES}/CameraSimple ${IFACES}/BillCoppelia)

