cmake_minimum_required(VERSION 3.10)
PROJECT( imu )

# Zero-copy DDS media plane (ImuFrame) — shared source in active_inference/common/media_transport.
# This component is a Python RoboComp component, so FastDDS is exposed to it via a thin
# pybind11 extension (imu_dds_native), not linked directly into the Python interpreter.
# Mirror of imu_fusion_dds's pose_dds_native (that sibling component) for the raw-IMU plane.
#
# NOTA: robocompdsl SOBREESCRIBE este fichero sin avisar en cada regeneración (a diferencia
# de specificworker.py/README.md/etc/config, que sí respeta con un .new) -- si se vuelve a
# tocar el .cdsl, restaurar este bloque a mano después.
set(MEDIA_TRANSPORT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../active_inference/common/media_transport)
set(MEDIA_TRANSPORT_IDL ${MEDIA_TRANSPORT_DIR}/generated/idl)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(WARNING "pybind11 CMake package not found — imu_dds_native (DDS raw-IMU plane) will not be built. dds_pub stays available in config but degrades to a no-op at runtime (see specificworker.py).")
else()
  # media_transport.cpp's Core dispatches over ALL of FrameKind{Image,Image360,Lidar,Imu,Pose}
  # in one switch, so the linker needs every frame type's PubSubType/TypeObjectSupport even
  # though this component only ever uses the Imu branch at runtime (confirmed: linking only
  # imu_frame*.cxx left `undefined symbol: LidarFramePubSubType::LidarFramePubSubType()`).
  pybind11_add_module(imu_dds_native
    dds_publisher.cpp
    imu_dds_bindings.cpp
    ${MEDIA_TRANSPORT_DIR}/media_transport.cpp
    ${MEDIA_TRANSPORT_IDL}/image_framePubSubTypes.cxx
    ${MEDIA_TRANSPORT_IDL}/image_frameTypeObjectSupport.cxx
    ${MEDIA_TRANSPORT_IDL}/image360_framePubSubTypes.cxx
    ${MEDIA_TRANSPORT_IDL}/image360_frameTypeObjectSupport.cxx
    ${MEDIA_TRANSPORT_IDL}/lidar_framePubSubTypes.cxx
    ${MEDIA_TRANSPORT_IDL}/lidar_frameTypeObjectSupport.cxx
    ${MEDIA_TRANSPORT_IDL}/imu_framePubSubTypes.cxx
    ${MEDIA_TRANSPORT_IDL}/imu_frameTypeObjectSupport.cxx
    ${MEDIA_TRANSPORT_IDL}/pose_framePubSubTypes.cxx
    ${MEDIA_TRANSPORT_IDL}/pose_frameTypeObjectSupport.cxx
  )
  target_include_directories(imu_dds_native PRIVATE ${MEDIA_TRANSPORT_DIR})
  target_compile_features(imu_dds_native PRIVATE cxx_std_23)
  # fastdds/fastcdr are installed under /usr/local (see ldconfig -p) with no exported
  # Config.cmake on this machine — link by plain name like the other media_transport
  # consumers (e.g. active_inference/room_concept) do, instead of find_package(fastdds).
  target_link_libraries(imu_dds_native PRIVATE fastdds fastcdr)
  # Build artifact lands directly in src/ next to specificworker.py, which imports it as
  # `from src import imu_dds_native` — same layout imu_fusion_dds already uses.
  set_target_properties(imu_dds_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
endif()
