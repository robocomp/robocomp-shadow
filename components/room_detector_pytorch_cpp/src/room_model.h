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

#ifndef ROOM_MODEL_H
#define ROOM_MODEL_H

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
#include "common_types.h"


/**
 * @brief Axis-aligned box room model using Signed Distance Function (SDF)
 *
 * IMPORTANT: The room is FIXED at origin (0,0) and axis-aligned.
 * This defines our coordinate system. Only 5 parameters are optimized:
 * - Room size: (half_width, half_height)
 * - Robot pose relative to room: (robot_x, robot_y, robot_theta)
 *
 * This removes 2 degrees of freedom (room translation) which are
 * unobservable from LiDAR data alone (gauge freedom).
 */
class RoomModel : public torch::nn::Module
{
    public:
        /**
         * @brief Constructor with initial room size and robot pose
         *
         * Room is always centered at (0,0) and axis-aligned.
         * Robot pose is relative to room center.
         *
         * @param half_width Initial half-width (distance from center to wall in X)
         * @param half_height Initial half-height (distance from center to wall in Y)
         * @param robot_x Initial robot X position (relative to room center)
         * @param robot_y Initial robot Y position (relative to room center)
         * @param robot_theta Initial robot orientation (radians)
         */
        RoomModel()= default;
        void init(float half_width, float half_height,
                  float robot_x = 0.0f, float robot_y = 0.0f, float robot_theta = 0.0f);

        /**
         * @brief Compute the signed distance from points to the box surface
         *
         * Points are assumed to be in ROBOT frame and are transformed to
         * ROOM frame (which is also the global frame) using the robot pose.
         *
         * Room is always at origin (0,0), so:
         * - Negative SDF: inside the room
         * - Zero SDF: on the walls
         * - Positive SDF: outside the room
         *
         * @param points_robot Tensor of shape [N, 2] with (x, y) in robot frame
         * @return Tensor of shape [N] with signed distances
         */
        torch::Tensor sdf(const torch::Tensor& points_robot);

        /**
         * @brief Get current room parameters (always centered at origin)
         * @return Vector with [half_width, half_height]
         */
        std::vector<float> get_room_parameters() const;

        /**
         * @brief Get current robot pose (relative to room center at origin)
         * @return Vector with [robot_x, robot_y, robot_theta]
         */
        std::vector<float> get_robot_pose() const;

        /**
         * @brief Get all trainable parameters for optimization (5 total)
         */
        std::vector<torch::Tensor> parameters() const;

        /**
         * @brief Print current room and robot configuration
         */
        void print_info() const;

        /**
         * @brief Freeze room parameters (stop optimizing room shape)
         * Sets requires_grad = false for half_extents
         */
        void freeze_room_parameters();

        /**
         * @brief Unfreeze room parameters (resume optimizing room shape)
         * Sets requires_grad = true for half_extents
         */
        void unfreeze_room_parameters();

        /**
         * @brief Check if room parameters are frozen
         */
        bool are_room_parameters_frozen() const;

        /**
         * @brief Get only robot parameters (for selective optimization)
         */
        std::vector<torch::Tensor> get_robot_parameters() const;

        // Odometry calibration
        void init_odometry_calibration(float k_trans = 1.0f, float k_rot = 1.0f);
        std::vector<float> get_odometry_calibration() const;
        void freeze_odometry_calibration();
        void unfreeze_odometry_calibration();
        // Apply calibration to velocity command
        Eigen::Vector3f calibrate_velocity(const VelocityCommand& cmd, float dt) const;
            // Robot pose (relative to room at origin)
        torch::Tensor robot_pos_;     // [robot_x, robot_y]
        torch::Tensor robot_theta_;   // [theta] in radians

    private:
        // Room parameters (FIXED at origin)
        // No trainable center - it's always (0, 0)
        torch::Tensor half_extents_;  // [half_width, half_height]

        // Odometry calibration parameters
        torch::Tensor k_translation_;  // Scale factor for translation (adv_x, adv_z)
        torch::Tensor k_rotation_;     // Scale factor for rotation

        /**
         * @brief Transform points from robot frame to room frame (at origin)
         * @param points_robot Points in robot's local frame [N, 2]
         * @return Points in room frame [N, 2]
         */
        torch::Tensor transform_to_room_frame(const torch::Tensor& points_robot);
};

#endif // ROOM_MODEL_H