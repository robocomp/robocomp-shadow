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
    config_.voxel_size = 0.05;
    config_.deskew = false;

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

LidarOdometry::Result LidarOdometry::update(const RoboCompLidar3D::TPoints &points)
{
    Result res;
    res.success = false;
    res.delta_pose = Eigen::Vector3f::Zero();
    res.covariance = Eigen::Matrix3f::Identity(); // High uncertainty fallback

    if (points.empty()) return res;

    // 1. Convert Points
    // KISS-ICP expects std::vector<Eigen::Vector3d>
    const auto eigen_points = convert_points(points);

    // 2. Register Scan
    // The pipeline returns the global pose of the robot in the odometry frame
    // NOTE: kiss_icp::pipeline::KissICP::RegisterFrame returns the POSE, not delta.
    pipeline_.RegisterFrame(eigen_points, {});
    const auto current_odom = pipeline_.delta(); // Get latest pose

    // return values
    res.delta_pose = Eigen::Vector3f(current_odom.translation().x(), current_odom.translation().y(), current_odom.angleZ());
    
    // 4. Estimate Covariance (KISS-ICP provides robust registration but not explicit covariance)
    // We synthesize a covariance based on the fitness/points matched, or use a tight constant
    // because Lidar Odometry is extremely accurate locally.
    res.covariance = Eigen::Matrix3f::Identity();
    res.covariance(0,0) = 1e-5; // Very precise X
    res.covariance(1,1) = 1e-5; // Very precise Y
    res.covariance(2,2) = 1e-4; // Very precise Theta
    res.success = true;
    return res;
}

std::vector<Eigen::Vector3d> LidarOdometry::convert_points(const RoboCompLidar3D::TPoints &points)
{
    std::vector<Eigen::Vector3d> eigen_pts;
    eigen_pts.reserve(points.size());
    
    for (const auto& p : points)
    {
        // Convert mm to meters
        eigen_pts.emplace_back(p.x / 1000.0, p.y / 1000.0, p.z / 1000.0);
    }
    return eigen_pts;
}
