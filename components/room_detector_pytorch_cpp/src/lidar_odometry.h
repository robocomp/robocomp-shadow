/*
 * Copyright (C) 2025
 * LidarOdometry wrapper using KISS-ICP
 */

#ifndef LIDAR_ODOMETRY_H
#define LIDAR_ODOMETRY_H

#include <vector>
#include <Eigen/Dense>
#include <Lidar3D.h>
#include <kiss_icp/pipeline/KissICP.hpp>

/**
 * @brief High-precision Lidar Odometry using KISS-ICP
 * * Replaces velocity commands for the "Slow Path" (Graph Optimization).
 * It maintains a local map to provide robust scan-to-map matching.
 */
class LidarOdometry
{
public:
    struct Result {
        bool success = false;
        Eigen::Vector3f delta_pose;  // [dx, dy, dtheta] relative motion since last frame
        Eigen::Matrix3f covariance;  // Estimated uncertainty
    };

    LidarOdometry();
    ~LidarOdometry() = default;

    /**
     * @brief Process a new point cloud
     * @param points Raw LiDAR points
     * @return Result containing relative motion since the PREVIOUS scan
     */
    Result update(const RoboCompLidar3D::TPoints &points, const std::chrono::time_point<std::chrono::system_clock> &timestamp);

    /**
     * @brief Reset the odometry (e.g., on room change)
     */
    void reset();

private:
    // KISS-ICP Pipeline
    kiss_icp::pipeline::KISSConfig config_;
    kiss_icp::pipeline::KissICP pipeline_;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_timestamp = std::chrono::time_point<std::chrono::system_clock>{};
    // State tracking
    bool initialized_ = false;
    Eigen::Matrix4d last_pose_ = Eigen::Matrix4d::Identity();

    
    // Helper to convert RoboComp points to Eigen
    std::vector<Eigen::Vector3d> convert_points(const RoboCompLidar3D::TPoints &points);
};

#endif // LIDAR_ODOMETRY_H
