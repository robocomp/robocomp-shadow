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

#ifndef DOOR_MODEL_H
#define DOOR_MODEL_H

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
 * @brief 3D door model using Signed Distance Function (SDF)
 * 
 * Models a door with:
 * - Frame: three boxes (top lintel, left jamb, right jamb)
 * - Articulated leaf: rotating box hinged on the left side
 * 
 * The door is positioned and oriented in 3D space. We optimize:
 * - Door pose: (x, y, z, theta) - position and yaw orientation
 * - Door geometry: (width, height) 
 * - Door state: (opening_angle) - articulation angle
 * - Fixed parameters: frame_thickness, frame_depth, leaf_thickness
 * 
 * COORDINATE SYSTEM:
 * - Door frame is axis-aligned in its local frame
 * - Y+ points up (height direction)
 * - Opening angle rotates around Y-axis at hinge position
 * - Hinge is at left edge: x = -width/2
 */
class DoorModel final : public torch::nn::Module
{
    public:
        /**
         * @brief Default constructor
         */
        DoorModel() = default;

        cv::Rect roi;
        int classId;  // debug
        std::string label;  // debug
        float score;

        /**
         * @brief Initialize door model from YOLO detection ROI
         *
         * @param roi_points LiDAR points within YOLO bounding box
         * @param initial_width Initial guess for door width (meters)
         * @param initial_height Initial guess for door height (meters)
         * @param initial_angle Initial opening angle (radians, 0 = closed)
         */
        void init(const std::vector<Eigen::Vector3f>& roi_points,
                  const cv::Rect &roi_,
                  int classId_,
                  const std::string &label_,
                  float initial_width = 1.0f,
                  float initial_height = 2.0f,
                  float initial_angle = 0.0f);

        /**
         * @brief Compute signed distance from points to door surface
         *
         * Points are in ROBOT frame. The door pose transforms them to DOOR local frame
         * where the SDF is computed.
         *
         * @param points_robot Tensor [N, 3] with (x, y, z) in robot frame
         * @return Tensor [N] with signed distances
         */
        torch::Tensor sdf(const torch::Tensor& points_robot) const;

        /**
         * @brief Get current door parameters
         * @return Vector with [x, y, z, theta, width, height, angle]
         */
        std::vector<float> get_door_parameters() const;

        /**
         * @brief Get door pose (position and orientation)
         * @return Vector with [x, y, z, theta]
         */
        std::vector<float> get_door_pose() const;

        /**
         * @brief Get door geometry
         * @return Vector with [width, height]
         */
        std::vector<float> get_door_geometry() const;

        /**
         * @brief Get door opening angle
         * @return Opening angle in radians
         */
        float get_opening_angle() const;

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
         * @brief Freeze geometry parameters (optimize only pose and angle)
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

        void set_door_position(float x, float y, float z);

        void set_theta(float theta);

        void set_pose(float x, float y, float z, float theta);

        /**
         * @brief Print current door configuration
         */
        void print_info() const;

        // Fixed door frame parameters (not optimized)
        float frame_thickness_ = 0.10f;  // 10cm frame thickness
        float frame_depth_ = 0.15f;      // 15cm frame depth
        float leaf_thickness_ = 0.04f;   // 4cm door leaf thickness

        // Door pose in robot frame (trainable)
        torch::Tensor door_position_;  // [x, y, z] - door center position
        torch::Tensor door_theta_;     // [theta] - yaw orientation around Y-axis

        // Door geometry (trainable)
        torch::Tensor door_width_;     // [width] - opening width
        torch::Tensor door_height_;    // [height] - door height

        // Door articulation state (trainable)
        torch::Tensor opening_angle_;  // [angle] - rotation of leaf (0 = closed)

    private:

        /**
         * @brief Transform points from robot frame to door local frame
         * @param points_robot Points in robot's frame [N, 3]
         * @return Points in door local frame [N, 3]
         */
        torch::Tensor transform_to_door_frame(const torch::Tensor& points_robot) const;

        /**
         * @brief SDF for axis-aligned box (batched)
         * @param p Query points [N, 3]
         * @param b Box half-extents [3]
         * @return Signed distances [N]
         */
        torch::Tensor sdBox(const torch::Tensor& p, const torch::Tensor& b) const;

        /**
         * @brief Compute SDF for door frame (three boxes)
         * @param points_door Points in door local frame [N, 3]
         * @return Signed distances to frame [N]
         */
        torch::Tensor sdf_frame(const torch::Tensor& points_door) const;

        /**
         * @brief Compute SDF for articulated door leaf
         * @param points_door Points in door local frame [N, 3]
         * @return Signed distances to leaf [N]
         */
        torch::Tensor sdf_leaf(const torch::Tensor& points_door) const;
};

#endif // DOOR_MODEL_H
