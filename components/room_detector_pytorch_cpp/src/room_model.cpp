/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 */

// Undefine Qt macros before including PyTorch headers
#ifdef slots
#undef slots
#endif
#ifdef signals
#undef signals
#endif
#ifdef emit
#undef emit
#endif

#include "room_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void RoomModel::init(float half_width, float half_height,
                     float robot_x, float robot_y, float robot_theta)
{
    // Initialize room size as trainable tensors
    // Room center is FIXED at (0, 0) - not a parameter!
    half_extents_ = torch::tensor({half_width, half_height}, torch::requires_grad(true));

    // Initialize robot pose as trainable tensors (relative to room at origin)
    robot_pos_ = torch::tensor({robot_x, robot_y}, torch::requires_grad(true));
    robot_theta_ = torch::tensor({robot_theta}, torch::requires_grad(true));

    // Register all parameters for optimization (5 total, not 7!)
    register_parameter("half_extents", half_extents_);
    register_parameter("robot_pos", robot_pos_);
    register_parameter("robot_theta", robot_theta_);
}

torch::Tensor RoomModel::transform_to_room_frame(const torch::Tensor& points_robot) {
    // Transform points from robot frame to room frame (at origin)
    // p_room = R(theta) * p_robot + robot_pos

    // Get rotation angle
    torch::Tensor cos_theta = torch::cos(robot_theta_);
    torch::Tensor sin_theta = torch::sin(robot_theta_);

    // Create rotation matrix [2x2]
    torch::Tensor rotation = torch::stack({
        torch::stack({cos_theta.squeeze(), -sin_theta.squeeze()}),
        torch::stack({sin_theta.squeeze(), cos_theta.squeeze()})
    });

    // Apply rotation: [N, 2] @ [2, 2]^T = [N, 2]
    torch::Tensor rotated = torch::matmul(points_robot, rotation.transpose(0, 1));

    // Add translation: [N, 2] + [2] (broadcast)
    torch::Tensor points_room = rotated + robot_pos_;

    return points_room;
}

torch::Tensor RoomModel::sdf(const torch::Tensor& points_robot)
{
    // Transform points from robot frame to room frame
    torch::Tensor points_room = transform_to_room_frame(points_robot);

    // Compute absolute distances from origin in each dimension
    torch::Tensor abs_points = torch::abs(points_room);

    // Distance from each point to the box in each dimension
    torch::Tensor d = abs_points - half_extents_;

    // SDF for axis-aligned box at origin
    torch::Tensor outside_distance = torch::norm(
        torch::max(d, torch::zeros_like(d)),
        2,
        /*dim=*/1
    );

    torch::Tensor inside_distance = torch::clamp_max(
        torch::max(d.select(1, 0), d.select(1, 1)),
        0.0
    );

    return outside_distance + inside_distance;
}

std::vector<float> RoomModel::get_room_parameters() const
{
    auto extents_acc = half_extents_.accessor<float, 1>();
    return {extents_acc[0], extents_acc[1]};
}

std::vector<float> RoomModel::get_robot_pose() const
{
    auto pos_acc = robot_pos_.accessor<float, 1>();
    auto theta_acc = robot_theta_.accessor<float, 1>();
    return {pos_acc[0], pos_acc[1], theta_acc[0]};
}

std::vector<torch::Tensor> RoomModel::parameters() const
{
    return {half_extents_, robot_pos_, robot_theta_};
}

void RoomModel::freeze_room_parameters()
{
    if (half_extents_.defined())
        half_extents_.set_requires_grad(false);
}

void RoomModel::unfreeze_room_parameters()
{
    if (half_extents_.defined())
        half_extents_.set_requires_grad(true);
}

bool RoomModel::are_room_parameters_frozen() const {
    return half_extents_.defined() && !half_extents_.requires_grad();
}

std::vector<torch::Tensor> RoomModel::get_robot_parameters() const {
    return {robot_pos_, robot_theta_};
}

void RoomModel::print_info() const {
    auto room_params = get_room_parameters();
    auto robot_pose = get_robot_pose();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== ROOM MODEL ===\n";
    std::cout << "  Center: (0.000, 0.000) m [FIXED at origin]\n";
    std::cout << "  Half-extents: (" << room_params[0] << ", " << room_params[1] << ") m\n";
    std::cout << "  Full dimensions: " << 2*room_params[0] << " x " << 2*room_params[1] << " m\n";
    std::cout << "  Frozen: " << (are_room_parameters_frozen() ? "YES" : "NO") << "\n";
    std::cout << "\n=== ROBOT POSE (relative to room) ===\n";
    std::cout << "  Position: (" << robot_pose[0] << ", " << robot_pose[1] << ") m\n";
    std::cout << "  Orientation: " << robot_pose[2] << " rad ("
              << (robot_pose[2] * 180.0 / M_PI) << " deg)\n";
}

