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

torch::Tensor RoomModel::sdf(const torch::Tensor& points_robot) {
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

std::vector<float> RoomModel::get_room_parameters() const {
    auto extents_acc = half_extents_.accessor<float, 1>();
    return {extents_acc[0], extents_acc[1]};
}

std::vector<float> RoomModel::get_robot_pose() const {
    auto pos_acc = robot_pos_.accessor<float, 1>();
    auto theta_acc = robot_theta_.accessor<float, 1>();
    return {pos_acc[0], pos_acc[1], theta_acc[0]};
}

std::vector<torch::Tensor> RoomModel::parameters() const {
    return {half_extents_, robot_pos_, robot_theta_};
}

void RoomModel::freeze_room_parameters() {
    if (half_extents_.defined())
        half_extents_.set_requires_grad(false);
}

void RoomModel::unfreeze_room_parameters() {
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

torch::Tensor RoomLoss::compute_loss(const torch::Tensor& points,
                                 RoomModel& room,
                                 float wall_thickness)
{
    const torch::Tensor sdf_values = room.sdf(points);
    torch::Tensor loss = torch::mean(torch::square(sdf_values));
    return loss;
}

// ============================================================================
// Uncertainty Estimation using Laplace Approximation - ADAPTIVE VERSION
// ============================================================================

torch::Tensor UncertaintyEstimator::compute_covariance(const torch::Tensor& points,
                                                       RoomModel& room,
                                                       float wall_thickness) {
    // Get all parameters
    auto all_params = room.parameters();

    // Filter only parameters that require gradients
    std::vector<torch::Tensor> params;
    for (const auto& p : all_params) {
        if (p.requires_grad()) {
            params.push_back(p);
        }
    }

    if (params.empty()) {
        std::cerr << "ERROR: No parameters require gradients!\n";
        return torch::eye(5, torch::kFloat32);  // Return identity as fallback
    }

    // Count total number of parameters
    int total_params = 0;
    for (const auto& p : params) {
        total_params += p.numel();
    }

    // Compute loss (ensure requires_grad is true)
    torch::Tensor loss = RoomLoss::compute_loss(points, room, wall_thickness);

    // Compute gradients (first derivatives)
    auto gradients = torch::autograd::grad({loss}, params,
                                          {}, // grad_outputs
                                          true, // retain_graph
                                          true); // create_graph

    // Flatten all gradients into a single vector
    std::vector<torch::Tensor> grad_flat;
    for (const auto& g : gradients) {
        grad_flat.push_back(g.flatten());
    }
    torch::Tensor grad_vector = torch::cat(grad_flat);

    // Compute Hessian matrix
    torch::Tensor hessian = torch::zeros({total_params, total_params}, torch::kFloat32);

    // Compute each row of Hessian (second derivatives)
    for (int i = 0; i < total_params; ++i) {
        torch::Tensor grad_for_this_param = torch::zeros_like(grad_vector);
        grad_for_this_param[i] = 1.0;

        auto second_derivs = torch::autograd::grad({grad_vector}, params,
                                                   {grad_for_this_param},
                                                   true,  // retain_graph
                                                   false, // create_graph
                                                   true); // allow_unused

        std::vector<torch::Tensor> second_flat;
        for (const auto& sd : second_derivs) {
            if (sd.defined()) {
                second_flat.push_back(sd.flatten());
            } else {
                int size = 0;
                for (const auto& p : params) {
                    size += p.numel();
                }
                second_flat.push_back(torch::zeros({size}, torch::kFloat32));
            }
        }
        torch::Tensor second_vector = torch::cat(second_flat);
        hessian[i] = second_vector;
    }

    // // Add regularization
    // const float reg = 1e-6;
    // hessian = hessian + reg * torch::eye(total_params, torch::kFloat32);
    //
    // // Covariance is inverse of Hessian
    // // check for positive definiteness so it can be inverted
    //
    // torch::Tensor covariance = torch::inverse(hessian);

    // Ensure numerical symmetry
    hessian = 0.5 * (hessian + hessian.transpose(0, 1));

    // Robust inversion strategy
    torch::Tensor covariance;
    torch::Tensor I = torch::eye(total_params, hessian.options());
    double reg = 1e-6;
    const int max_attempts = 6;
    bool success = false;

    for (int attempt = 0; attempt < max_attempts; ++attempt)
    {
        torch::Tensor h_reg = hessian + reg * I;
        try {
            // Try Cholesky and build inverse from the factor
            auto L = torch::linalg_cholesky(h_reg, /*upper=*/false);
            covariance = torch::cholesky_inverse(L, /*upper=*/false);
            success = true;
            break;
        } catch (const c10::Error& e) { reg *= 10.0; }
    }

    if (!success)
    {
        // First fallback: pseudo-inverse (can handle singular/indefinite matrices)
        try {
            covariance = torch::pinverse(hessian + reg * I);
        } catch (const c10::Error& e) {
            // Last-resort fallback: safe diagonal inverse
            torch::Tensor diag = torch::diag(hessian).clone();
            diag = torch::clamp(diag, 1e-12, 1e12);
            covariance = torch::diag(1.0 / diag);
        }
    }

    covariance = covariance / static_cast<float>(points.size(0));

    return covariance.detach();
}

std::vector<float> UncertaintyEstimator::get_std_devs(const torch::Tensor& covariance) {
    torch::Tensor variances = torch::diagonal(covariance);
    torch::Tensor std_devs = torch::sqrt(torch::clamp(variances, 0.0, 1e10));

    std::vector<float> result;
    auto acc = std_devs.accessor<float, 1>();
    for (int i = 0; i < std_devs.size(0); ++i) {
        result.push_back(acc[i]);
    }

    return result;
}

torch::Tensor UncertaintyEstimator::get_correlation_matrix(const torch::Tensor& covariance) {
    torch::Tensor std_devs = torch::sqrt(torch::diagonal(covariance));
    std_devs = torch::clamp(std_devs, 1e-10, 1e10);

    int n = covariance.size(0);
    torch::Tensor correlation = torch::zeros({n, n}, torch::kFloat32);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            correlation[i][j] = covariance[i][j] / (std_devs[i] * std_devs[j]);
        }
    }

    return correlation;
}

void UncertaintyEstimator::print_uncertainty(const torch::Tensor& covariance, const RoomModel& room) {
    auto std_devs = get_std_devs(covariance);
    auto room_params = room.get_room_parameters();
    auto robot_pose = room.get_robot_pose();
    bool room_frozen = room.are_room_parameters_frozen();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== UNCERTAINTY ESTIMATES ===\n";
    std::cout << "(1-sigma / 68% confidence intervals)\n\n";

    if (room_frozen) {
        std::cout << "Room Parameters (FROZEN):\n";
        std::cout << "  Half-width:   " << room_params[0] << " m (frozen)\n";
        std::cout << "  Half-height:  " << room_params[1] << " m (frozen)\n";
        std::cout << "\nRobot Pose (OPTIMIZED):\n";
        std::cout << "  Position X:   " << robot_pose[0] << " ± " << std_devs[0] << " m\n";
        std::cout << "  Position Y:   " << robot_pose[1] << " ± " << std_devs[1] << " m\n";
        std::cout << "  Orientation:  " << robot_pose[2] << " ± " << std_devs[2]
                  << " rad (" << (std_devs[2] * 180.0 / M_PI) << " deg)\n";
    } else {
        std::cout << "Room Parameters (center FIXED at origin):\n";
        std::cout << "  Half-width:   " << room_params[0] << " ± " << std_devs[0] << " m\n";
        std::cout << "  Half-height:  " << room_params[1] << " ± " << std_devs[1] << " m\n";

        std::cout << "\nRobot Pose (relative to room):\n";
        std::cout << "  Position X:   " << robot_pose[0] << " ± " << std_devs[2] << " m\n";
        std::cout << "  Position Y:   " << robot_pose[1] << " ± " << std_devs[3] << " m\n";
        std::cout << "  Orientation:  " << robot_pose[2] << " ± " << std_devs[4]
                  << " rad (" << (std_devs[4] * 180.0 / M_PI) << " deg)\n";

        std::cout << "\nFull Room Dimensions:\n";
        float width = 2 * room_params[0];
        float height = 2 * room_params[1];
        float width_std = 2 * std_devs[0];
        float height_std = 2 * std_devs[1];
        std::cout << "  Width:        " << width << " ± " << width_std << " m\n";
        std::cout << "  Height:       " << height << " ± " << height_std << " m\n";
    }

    // Compute correlation matrix
    auto correlation = get_correlation_matrix(covariance);

    std::cout << "\n=== KEY CORRELATIONS ===\n";
    std::cout << std::setprecision(3);

    auto corr_acc = correlation.accessor<float, 2>();

    if (room_frozen) {
        // Only 3x3 covariance (robot pose only)
        std::cout << "Robot X vs Robot Y:         " << corr_acc[0][1] << "\n";
        std::cout << "Robot X vs Robot Theta:     " << corr_acc[0][2] << "\n";
        std::cout << "Robot Y vs Robot Theta:     " << corr_acc[1][2] << "\n";
    } else {
        // Full 5x5 covariance
        std::cout << "Robot X vs Robot Y:         " << corr_acc[2][3] << "\n";
        std::cout << "Robot X vs Robot Theta:     " << corr_acc[2][4] << "\n";
        std::cout << "Robot X vs Room Width:      " << corr_acc[2][0] << "\n";
        std::cout << "Robot Y vs Room Height:     " << corr_acc[3][1] << "\n";
        std::cout << "Room Width vs Height:       " << corr_acc[0][1] << "\n";
    }

    std::cout << "\n(Correlations close to ±1 indicate strong dependence)\n";
}