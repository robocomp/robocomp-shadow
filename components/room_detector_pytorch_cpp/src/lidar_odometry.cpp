/*
 * Copyright (C) 2025
 */

#include "lidar_odometry.h"
#include <QDebug>

LidarOdometry::LidarOdometry()
    : config_()                // 1. Initialize config with defaults
    , pipeline_(config_)       // 2. Initialize pipeline with those defaults (satisfies compiler)
{
    // 3. Set your specific configuration
    config_.max_range = 40.0;
    config_.min_range = 0.5;
    config_.voxel_size = 0.5;
    config_.deskew = false;
    config_.max_points_per_voxel = 20;
    // th parms
    config_.min_motion_th = 0.001;
    config_.initial_threshold = 2.0;

    // 4. Re-initialize pipeline with the CORRECT config
    pipeline_ = kiss_icp::pipeline::KissICP(config_);
}

void LidarOdometry::reset()
{
    pipeline_ = kiss_icp::pipeline::KissICP(config_);
    last_pose_ = Eigen::Matrix4d::Identity();
    initialized_ = false;
    qInfo() << "LidarOdometry: Reset";
}

LidarOdometry::Result LidarOdometry::update(const RoboCompLidar3D::TPoints &points,
                                            const std::chrono::time_point<std::chrono::system_clock> &timestamp)
{
    Result res;
    res.success = false;
    res.delta_pose = Eigen::Vector3f::Zero();
    res.covariance = Eigen::Matrix3f::Identity();

    if (points.empty()) return res;

    std::vector<Eigen::Vector3d> eigen_points; eigen_points.reserve(points.size());
    for (const auto& p : points)
        eigen_points.emplace_back(p.x , p.y , p.z );

    // std::vector<double> timestamps(eigen_points.size());
    // std::fill(timestamps.begin(), timestamps.end(),
    //           std::chrono::duration<double>(timestamp.time_since_epoch()).count());

    if (!initialized_)
    {
        // Primer frame: inicializar pose anterior
        pipeline_.RegisterFrame(eigen_points, {static_cast<double>(timestamp.time_since_epoch().count())});
        last_pose_ = pipeline_.pose().matrix();  // Guardar pose absoluta
        initialized_ = true;
        res.success = false;
        qInfo() << "LidarOdometry: Initialized with" << points.size() << "points";
        return res;
    }

    const Eigen::Matrix4d prev_pose = last_pose_;

    // Registrar frame actual
    pipeline_.RegisterFrame(eigen_points, {static_cast<double>(timestamp.time_since_epoch().count())});
    const Eigen::Matrix4d current_pose = pipeline_.pose().matrix();  // Pose absoluta actual (4x4)

    // Calcular delta relativo: T_delta = T_last^(-1) * T_current
    const Eigen::Matrix4d delta_transform = prev_pose.inverse() * current_pose;

    // Extract rotation and translation
    const Eigen::Vector3d translation = delta_transform.block<3,1>(0,3);
    const Eigen::Matrix3d rotation = delta_transform.block<3,3>(0,0);
    // Yaw (rotación alrededor de Z) usando convención ZYX
    const double yaw = std::atan2(rotation(1,0), rotation(0,0));

    res.delta_pose = Eigen::Vector3f{ static_cast<const float &>(translation.x()),
                                      static_cast<const float &>(translation.y()),
                                    static_cast<float>(yaw) };

    // Actualizar pose anterior
    last_pose_ = current_pose;

    res.covariance(0,0) = 1e-5;
    res.covariance(1,1) = 1e-5;
    res.covariance(2,2) = 1e-4;
    res.success = true;
    return res;
}
