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
    config_.max_range = 10.0;
    config_.min_range = 0.5;
    config_.voxel_size = 0.05;
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
    //const auto delta_pose = pipeline_.delta().matrix();

     // Compute relative motion: T_delta = T_last^(-1) * T_current
     const Eigen::Matrix4d delta_transform = prev_pose.inverse() * current_pose;

     // Extract rotation and translation
     const Eigen::Vector3d translation = delta_transform.block<3,1>(0,3);
     const Eigen::Matrix3d rotation = delta_transform.block<3,3>(0,0);
     double dx = translation(0);
     double dy = translation(1);

     // === ROBUST YAW EXTRACTION ===
     // Instead of manual atan2, let Eigen decompose the 3D rotation.
     // We assume the standard sequence: Z (Yaw) -> Y (Pitch) -> X (Roll)
     // The first angle (index 0) is Yaw.
     //Eigen::Vector3d euler = rotation.eulerAngles(2, 1, 0); // Z, Y, X

     // Handle Euler angle multiplicity (Eigen can return angles outside +/- PI)
     //double dtheta = euler[0];
    double dtheta = std::atan2(rotation(1,0), rotation(0,0));

    //const auto rotation = delta_pose.block<3,3>(0,0);
    //double dtheta = atan2(rotation(1,0), rotation(0,0));
    // Normalize to [-PI, PI] for safety
    while (dtheta > M_PI) dtheta -= 2 * M_PI;
    while (dtheta < -M_PI) dtheta += 2 * M_PI;
    //double dx = delta_pose(0,3);
    //double dy = delta_pose(1,3);

    // === PHYSICS FILTER / CLAMPING ===
    // A. Estimate velocity (assuming ~10Hz or use actual dt)
    // You might want to pass 'dt' to update() for better accuracy
    float assumed_dt = 0.1f;
    const float dt = std::chrono::duration<float>(timestamp - last_timestamp).count();
    if (dt > 0.0f)
        assumed_dt = dt;
    const float linear_speed = std::sqrt(dx*dx + dy*dy) / assumed_dt;
    const float angular_speed = std::abs(dtheta) / assumed_dt;

    // B. Sanity Check (Max speed of your robot is ~1 m/s)
    // If ICP says we moved 10 m/s, it failed. Ignore this update.
    if (linear_speed > 2.0f)
    {
        qWarning() << "LidarOdometry: REJECTED Linear Jump (" << linear_speed << "m/s)";
        res.success = false; return res;
    }
    if (angular_speed > 2.0f)
    {
        qWarning() << "LidarOdometry: REJECTED Angular Jump (" << angular_speed << "rad/s)";
        res.success = false; return res;
    }

    // C. Zero-Velocity Update (ZUPT) / Deadband
    // If we moved less than 1mm or 0.1 deg, it's just sensor noise. Clamp to 0.
    // This prevents "Drifting while standing still".
    if (std::abs(dx) < 0.001f) dx = 0.0f;
    if (std::abs(dy) < 0.001f) dy = 0.0f;
    if (std::abs(dtheta) < 0.002f) dtheta = 0.0f;

    res.delta_pose = Eigen::Vector3f(dx, dy, dtheta);
    last_pose_ = current_pose;
    last_timestamp = timestamp;

    res.covariance(0,0) = 1e-5;
    res.covariance(1,1) = 1e-5;
    res.covariance(2,2) = 1e-4;
    res.success = true;

    return res;
}
