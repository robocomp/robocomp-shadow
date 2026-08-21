/*
 *    Copyright (C) 2026 by NoeZC
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// ImuDDSPublisher — publishes this component's raw IMU samples on a dedicated zero-copy
// DDS "media plane" (FastDDS), reusing active_inference/common/media_transport
// (rc::media::ImuPublisher, ImuFrame). Same role/shape as imu_fusion_dds's
// PoseDDSPublisher (see the sibling component) — this is the raw-sensor twin: no
// fusion, so roll/pitch/yaw are carried through as 0 (see specificworker.py).
//
// It does NOT touch DSR: the discovery descriptor is relayed onto the graph by an
// agent with a graph (e.g. robot_concept), same as lidar3d_dds/zed_camera. This class
// only moves the raw-IMU sample, gated by config (dds_pub, see etc/config).
// FastDDS headers stay behind a PIMPL so this header stays includable from the
// pybind11 binding without pulling FastDDS into the Python extension's ABI surface.
// Not thread-safe: drive publish() from one thread (this component's compute() loop).

#include <cstdint>
#include <memory>
#include <string>

class ImuDDSPublisher
{
public:
    struct Config
    {
        std::uint32_t domain_id = 7;               // CORTEX media domain (shared with lidar/pose/image planes)
        std::string   topic     = "rc/imu/raw";     // "imu" stream topic
        // --- QoS (carried in the descriptor; both ends must agree) ---
        int           history_depth      = 8;
        bool          shared_memory_only = true;
        bool          data_sharing       = false;   // OFF = churn-safe (see media_transport.h)
    };

    ImuDDSPublisher();
    ~ImuDDSPublisher();
    ImuDDSPublisher(const ImuDDSPublisher&) = delete;
    ImuDDSPublisher& operator=(const ImuDDSPublisher&) = delete;

    bool init(const Config& cfg);
    [[nodiscard]] bool ready() const { return ready_; }

    // rc::media::MediaDescriptor JSON (domain, topic, type tag, QoS) for an agent with a
    // graph to relay onto DSR. Returns "" if not ready.
    [[nodiscard]] std::string descriptor_json() const;

    // Publish one raw IMU sample. SI units (m/s², rad/s, Gauss, °C). roll/pitch/yaw are
    // whatever the caller passes (0 when there is no attitude filter — see
    // specificworker.py). sim_stamp_ms is 0 for a real-hardware sample, or the source
    // simulator's own clock when this component is proxying webots-bridge (simulation=true
    // in etc/config) — same "0 = not simulated" contract as RoboCompIMU's simTimestamp.
    // gyro_var is the per-sample angular-rate variance if the source provides one
    // (webots-bridge does; the Phidget driver does not), or -1 for "unknown". Returns false
    // (dropped) on: not ready or loan unavailable/publish failure.
    bool publish(std::uint64_t stamp_ms, std::uint64_t sim_stamp_ms,
                 float acc_x, float acc_y, float acc_z,
                 float gyro_x, float gyro_y, float gyro_z,
                 float mag_x, float mag_y, float mag_z,
                 float roll, float pitch, float yaw,
                 float temperature, float gyro_var);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    Config cfg_;
    bool ready_ = false;
};
