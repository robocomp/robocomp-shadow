/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 */

#ifndef TABLE_MODEL_H
#define TABLE_MODEL_H

// Undefine Qt macros that conflict with PyTorch
#ifdef slots
#undef slots
#endif
#ifdef signals
#undef signals
#endif
#ifdef emit
#undef emit
#endif

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include <Lidar3D.h>
#include <opencv2/opencv.hpp>

/**
 * @brief 3D table model using Signed Distance Function (SDF)
 *
 * Models a table with:
 * - Tabletop: rectangular box
 * - Four legs: cylindrical supports at corners
 *
 * The table is positioned and oriented in 3D space. We optimize:
 * - Table pose: (x, y, z, theta) - position and yaw orientation
 * - Table geometry: (width, depth, height, top_thickness)
 * - Fixed parameters: leg_radius
 *
 * COORDINATE SYSTEM (aligned with robot and Qt3D):
 *   X+ = right
 *   Y+ = forward (depth direction)
 *   Z+ = up (height direction)
 *
 * Table local frame (when theta=0):
 *   - Table width spans along X-axis
 *   - Table depth spans along Y-axis
 *   - Table height extends along Z-axis
 *   - Floor level at z = 0
 *   - Legs positioned at corners: (±width/2, ±depth/2)
 */
class TableModel final : public torch::nn::Module
{
public:
    /**
     * @brief Default constructor
     */
    TableModel() = default;

    std::vector<Eigen::Vector3f> roi_points;
    cv::Rect roi;
    unsigned int id;
    std::string label;  // debug
    float score;

    /**
     * @brief Initialize table model from YOLO detection ROI
     *
     * @param roi_points LiDAR points within YOLO bounding box
     * @param roi Bounding box from YOLO
     * @param id Unique table identifier
     * @param label Detection label
     * @param initial_width Initial guess for table width (meters)
     * @param initial_depth Initial guess for table depth (meters)
     * @param initial_height Initial guess for table height (meters)
     */
    void init(const std::vector<Eigen::Vector3f>& roi_points_,
              const cv::Rect &roi_,
              unsigned int id_,
              const std::string &label_,
              float initial_width = 1.0f,
              float initial_depth = 0.6f,
              float initial_height = 0.75f);

    /**
     * @brief Compute signed distance from points to table surface
     *
     * Points are in ROBOT frame. The table pose transforms them to TABLE local frame
     * where the SDF is computed.
     *
     * @param points_robot Tensor [N, 3] with (x, y, z) in robot frame
     * @return Tensor [N] with signed distances
     */
    torch::Tensor sdf(const torch::Tensor& points_robot) const;

    /**
     * @brief Get current table parameters
     * @return Vector with [x, y, z, theta, width, depth, height, top_thickness]
     */
    std::vector<float> get_table_parameters() const;

    /**
     * @brief Get table pose (position and orientation)
     * @return Vector with [x, y, z, theta]
     */
    std::vector<float> get_table_pose() const;

    /**
     * @brief Get table geometry
     * @return Vector with [width, depth, height, top_thickness]
     */
    std::vector<float> get_table_geometry() const;

    /**
     * @brief Get all trainable parameters
     */
    std::vector<torch::Tensor> parameters() const;

    /**
     * @brief Get only pose parameters (for selective optimization)
     */
    std::vector<torch::Tensor> get_pose_parameters() const;

    /**
     * @brief Get only geometry parameters (for selective optimization)
     */
    std::vector<torch::Tensor> get_geometry_parameters() const;

    /**
     * @brief Freeze geometry parameters (optimize only pose)
     */
    void freeze_geometry();

    /**
     * @brief Unfreeze geometry parameters
     */
    void unfreeze_geometry();

    /**
     * @brief Check if geometry is frozen
     */
    bool is_geometry_frozen() const;

    void set_table_position(float x, float y, float z);

    void set_theta(float theta);

    void set_pose(float x, float y, float z, float theta);

    /**
     * @brief Print current table configuration
     */
    void print_info() const;

    // Fixed table parameters (not optimized)
    float leg_radius_ = 0.025f;      // 2.5cm leg radius (standard)

    // Table pose in robot frame (trainable)
    torch::Tensor table_position_;   // [x, y, z] - table center position (at tabletop)
    torch::Tensor table_theta_;      // [theta] - yaw orientation around Z-axis

    // Table geometry (trainable)
    torch::Tensor table_width_;      // [width] - table width (along X)
    torch::Tensor table_depth_;      // [depth] - table depth (along Y)
    torch::Tensor table_height_;     // [height] - total height from floor to tabletop
    torch::Tensor top_thickness_;    // [thickness] - tabletop thickness

private:

    /**
     * @brief Transform points from robot frame to table local frame
     * @param points_robot Points in robot's frame [N, 3]
     * @return Points in table local frame [N, 3]
     */
    torch::Tensor transform_to_table_frame(const torch::Tensor& points_robot) const;

    /**
     * @brief SDF for axis-aligned box (batched)
     * @param p Query points [N, 3]
     * @param b Box half-extents [3]
     * @return Signed distances [N]
     */
    torch::Tensor sdBox(const torch::Tensor& p, const torch::Tensor& b) const;

    /**
     * @brief SDF for cylinder (batched)
     * @param p Query points [N, 3]
     * @param h Cylinder height
     * @param r Cylinder radius
     * @return Signed distances [N]
     */
    torch::Tensor sdCylinder(const torch::Tensor& p, float h, float r) const;

    /**
     * @brief Compute SDF for tabletop (box)
     * @param points_table Points in table local frame [N, 3]
     * @return Signed distances to tabletop [N]
     */
    torch::Tensor sdf_tabletop(const torch::Tensor& points_table) const;

    /**
     * @brief Compute SDF for all four legs (cylinders)
     * @param points_table Points in table local frame [N, 3]
     * @return Signed distances to legs [N]
     */
    torch::Tensor sdf_legs(const torch::Tensor& points_table) const;

    /**
     * @brief Union operation for SDF (minimum distance)
     * @param d1 First SDF values [N]
     * @param d2 Second SDF values [N]
     * @return Union SDF [N]
     */
    torch::Tensor sdf_union(const torch::Tensor& d1, const torch::Tensor& d2) const;
};

#endif // TABLE_MODEL_H
