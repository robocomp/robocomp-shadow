// pybind11 binding: exposes ImuDDSPublisher to this component's Python worker so it can
// publish raw IMU samples on the zero-copy FastDDS media plane (rc::media::ImuPublisher)
// without the Python worker itself linking FastDDS directly — the Phidget/ICE code stays
// pure Python, only this thin publish leg is native. Mirror of imu_fusion_dds's
// pose_dds_bindings.cpp (see that sibling component) for the raw-IMU plane instead of
// the fused pose.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dds_publisher.h"

namespace py = pybind11;

PYBIND11_MODULE(imu_dds_native, m)
{
    m.doc() = "Zero-copy FastDDS publisher for this component's raw IMU samples (ImuFrame)";

    py::class_<ImuDDSPublisher::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("domain_id", &ImuDDSPublisher::Config::domain_id)
        .def_readwrite("topic", &ImuDDSPublisher::Config::topic)
        .def_readwrite("history_depth", &ImuDDSPublisher::Config::history_depth)
        .def_readwrite("shared_memory_only", &ImuDDSPublisher::Config::shared_memory_only)
        .def_readwrite("data_sharing", &ImuDDSPublisher::Config::data_sharing);

    py::class_<ImuDDSPublisher>(m, "ImuDDSPublisher")
        .def(py::init<>())
        .def("init", &ImuDDSPublisher::init, py::arg("cfg"))
        .def("ready", &ImuDDSPublisher::ready)
        .def("descriptor_json", &ImuDDSPublisher::descriptor_json)
        .def("publish", &ImuDDSPublisher::publish,
             py::arg("stamp_ms"), py::arg("sim_stamp_ms"),
             py::arg("acc_x"), py::arg("acc_y"), py::arg("acc_z"),
             py::arg("gyro_x"), py::arg("gyro_y"), py::arg("gyro_z"),
             py::arg("mag_x"), py::arg("mag_y"), py::arg("mag_z"),
             py::arg("roll"), py::arg("pitch"), py::arg("yaw"),
             py::arg("temperature"), py::arg("gyro_var"));
}
