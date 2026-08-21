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

#include "dds_publisher.h"

#include <chrono>
#include <print>

#include "media_transport.h"   // active_inference/common/media_transport (added to the include path in CMake)

struct ImuDDSPublisher::Impl
{
    rc::media::ImuPublisher pub;
    std::uint64_t frame_id = 0;
    // Diagnostic publish stats (reported every ~5 s from publish()).
    std::uint64_t stat_ok = 0;
    std::uint64_t stat_drop = 0;
    std::chrono::steady_clock::time_point stat_t0{};
};

ImuDDSPublisher::ImuDDSPublisher() : pimpl_(std::make_unique<Impl>()) {}
ImuDDSPublisher::~ImuDDSPublisher() = default;

bool ImuDDSPublisher::init(const Config& cfg)
{
    cfg_ = cfg;

    rc::media::PublisherConfig pc;
    pc.domain_id          = cfg.domain_id;
    pc.topic_name         = cfg.topic;
    pc.history_depth      = cfg.history_depth;
    pc.shared_memory_only = cfg.shared_memory_only;
    pc.data_sharing       = cfg.data_sharing;

    pimpl_->stat_t0 = std::chrono::steady_clock::now();
    ready_ = pimpl_->pub.init(pc);

    if (ready_)
        std::print("[imu] DDS imu media plane ready domain={} topic='{}'\n", cfg.domain_id, cfg.topic);
    else
        std::print(stderr, "[imu] DDS imu media plane init FAILED (topic='{}')\n", cfg.topic);
    return ready_;
}

std::string ImuDDSPublisher::descriptor_json() const
{
    if (!ready_)
        return {};

    rc::media::MediaDescriptor d;
    d.version                = 1;
    d.domain_id              = cfg_.domain_id;
    d.type_name              = "ImuFrame";
    d.type_tag               = rc::media::IMU_FRAME_TYPE_TAG;
    d.history_depth          = cfg_.history_depth;
    d.shared_memory_only     = cfg_.shared_memory_only;
    d.data_sharing           = cfg_.data_sharing;
    d.ready                  = ready_;
    d.streams["imu"]         = cfg_.topic;
    d.stream_types["imu"]    = "ImuFrame";
    return d.to_json();
}

bool ImuDDSPublisher::publish(std::uint64_t stamp_ms, std::uint64_t sim_stamp_ms,
                               float acc_x, float acc_y, float acc_z,
                               float gyro_x, float gyro_y, float gyro_z,
                               float mag_x, float mag_y, float mag_z,
                               float roll, float pitch, float yaw,
                               float temperature, float gyro_var)
{
    bool ok = false;
    if (ready_)
    {
        if (rc::media::ImuFrame* s = pimpl_->pub.loan(); s != nullptr)
        {
            // loan() hands back a reused SHM pool slot, not a zero-initialized one -- every
            // field must be set explicitly or it carries whatever the previous sample left
            // there.
            s->stream_id(rc::media::STREAM_IMU);
            s->frame_id(pimpl_->frame_id++);
            s->stamp_ms(stamp_ms);
            s->sim_stamp_ms(sim_stamp_ms);
            s->acc_x(acc_x);   s->acc_y(acc_y);   s->acc_z(acc_z);
            s->gyro_x(gyro_x); s->gyro_y(gyro_y); s->gyro_z(gyro_z);
            s->mag_x(mag_x);   s->mag_y(mag_y);   s->mag_z(mag_z);
            s->roll(roll);     s->pitch(pitch);   s->yaw(yaw);
            s->temperature(temperature);
            s->gyro_var(gyro_var);
            ok = pimpl_->pub.publish(s);
        }
        // else: SHM pool exhausted -> counted as a drop below
    }

    // Diagnostic: report published/dropped every ~5 s.
    ok ? ++pimpl_->stat_ok : ++pimpl_->stat_drop;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - pimpl_->stat_t0).count() >= 5.0)
    {
        std::print("[Imu] published {} / dropped {}\n", pimpl_->stat_ok, pimpl_->stat_drop);
        pimpl_->stat_ok = 0;
        pimpl_->stat_drop = 0;
        pimpl_->stat_t0 = now;
    }
    return ok;
}
