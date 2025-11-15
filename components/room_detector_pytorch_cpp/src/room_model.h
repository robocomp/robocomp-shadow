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

private:
    // Room parameters (FIXED at origin)
    // No trainable center - it's always (0, 0)
    torch::Tensor half_extents_;  // [half_width, half_height]

    // Robot pose (relative to room at origin)
    torch::Tensor robot_pos_;     // [robot_x, robot_y]
    torch::Tensor robot_theta_;   // [theta] in radians

    /**
     * @brief Transform points from robot frame to room frame (at origin)
     * @param points_robot Points in robot's local frame [N, 2]
     * @return Points in room frame [N, 2]
     */
    torch::Tensor transform_to_room_frame(const torch::Tensor& points_robot);
};

/**
 * @brief Loss function for room fitting
 *
 * Measures how well the LiDAR points fit the room model.
 * Points should be close to the box surface (SDF ≈ 0)
 */
class RoomLoss
{
public:
    /**
     * @brief Compute loss between LiDAR points and room model
     *
     * @param points Tensor of shape [N, 2] with (x, y) LiDAR points in robot frame
     * @param room Room model to compare against
     * @param wall_thickness Expected thickness of walls for robust fitting
     * @return Scalar tensor with the loss value
     */
    static torch::Tensor compute_loss(const torch::Tensor& points,
                                  RoomModel& room,
                                  float wall_thickness = 0.1f);
};

/**
 * @brief Uncertainty estimation using Laplace approximation
 *
 * Computes covariance matrix of the MAP estimate.
 * Adapts to frozen/unfrozen parameters:
 * - MAPPING state: 5×5 matrix (room + robot)
 * - LOCALIZED state: 3×3 matrix (robot only)
 */
class UncertaintyEstimator
{
public:
    /**
     * @brief Compute covariance matrix using Laplace approximation (ADAPTIVE)
     *
     * Automatically detects which parameters require gradients and computes
     * covariance only for those parameters.
     *
     * When room is frozen (LOCALIZED):
     *   - Returns 3×3 covariance for [robot_x, robot_y, robot_theta]
     * When room is unfrozen (MAPPING):
     *   - Returns 5×5 covariance for [half_width, half_height, robot_x, robot_y, robot_theta]
     *
     * @param points LiDAR points tensor [N, 2] in robot frame
     * @param room Room model at MAP estimate
     * @param wall_thickness Same as used in loss computation
     * @return Covariance matrix [n, n] where n = number of unfrozen parameters
     */
    static torch::Tensor compute_covariance(const torch::Tensor& points,
                                           RoomModel& room,
                                           float wall_thickness = 0.1f);

    /**
     * @brief Extract standard deviations (sqrt of diagonal of covariance)
     *
     * @param covariance Covariance matrix [n, n]
     * @return Vector of std devs (size depends on frozen state)
     */
    static std::vector<float> get_std_devs(const torch::Tensor& covariance);

    /**
     * @brief Extract correlation matrix from covariance matrix
     *
     * @param covariance Covariance matrix [n, n]
     * @return Correlation matrix [n, n] with values in [-1, 1]
     */
    static torch::Tensor get_correlation_matrix(const torch::Tensor& covariance);

    /**
     * @brief Print uncertainty information in readable format (ADAPTIVE)
     *
     * Adapts output based on whether room is frozen or not.
     *
     * @param covariance Covariance matrix [n, n]
     * @param room Room model to label parameters
     */
    static void print_uncertainty(const torch::Tensor& covariance, const RoomModel& room);
};

#endif // ROOM_MODEL_H