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
#include <QDebug>
#include "pointcloud_center_estimator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void RoomModel::init(const RoboCompLidar3D::TPoints &points)
{

    // Guard against double initialization
    if (half_extents_.defined()) {
        qWarning() << "RoomModel::init() called twice - ignoring";
        return;
    }

    // Compute initial guess for room SIZE from point cloud bounds
	// Convert LiDAR points to PyTorch tensor [N, 2]
	// Keep points in ROBOT FRAME - the model will transform them
	std::vector<float> points_data;
	points_data.reserve(points.size() * 2);
	for (const auto& p : points)
	{
		points_data.push_back(p.x / 1000.0f);  // Convert mm to meters
		points_data.push_back(p.y / 1000.0f);
	}
	const torch::Tensor points_tensor = torch::from_blob
	(
		points_data.data(),
		{static_cast<long>(points.size()), 2},
		torch::kFloat32
	).clone();
	const auto x_coords = points_tensor.index({torch::indexing::Slice(), 0});
	const auto y_coords = points_tensor.index({torch::indexing::Slice(), 1});

	const float x_min = x_coords.min().item<float>();
	const float x_max = x_coords.max().item<float>();
	const float y_min = y_coords.min().item<float>();
	const float y_max = y_coords.max().item<float>();

	// Initial room size (room will be at origin by definition)
	const float half_width = (x_max - x_min) / 2.0f;
	const float half_depth = (y_max - y_min) / 2.0f;

	// Initial robot pose (offset from room center to point cloud center)
    rc::PointcloudCenterEstimator center_estimator;
    Eigen::Vector2d room_center;
    if (const auto room_center_ = center_estimator.estimate(points); room_center_.has_value())
        room_center = room_center_.value();
    else
    {
        room_center = Eigen::Vector2d{(x_min + x_max) / 2.0, (y_min + y_max) / 2.0};
        qWarning() << "RoomModel::init() - Could not estimate room center from point cloud. Using (0,0).";
    }
	const float robot_x = -room_center.cast<float>().x();  // Negative because room is at origin
	const float robot_y = -room_center.cast<float>().y();
	const float robot_theta = 0.f; // TODO: Improve initial orientation estimate

	std::cout << "\n  RoomModel::init()  Initial guess:\n";
	std::cout << "  Point cloud bounds: X[" << x_min << ", " << x_max << "], Y[" << y_min << ", " << y_max << "]\n";
	std::cout << "  Room size (at origin): " << 2*half_width << " x " << 2*half_depth << " m\n";
	std::cout << "  Robot pose (relative to room): (" << robot_x << ", " << robot_y << ", " << robot_theta << ")\n";
	std::cout << "----------------------------------------\n";

    // Initialize room size as trainable tensors
    // Room center is FIXED at (0, 0) - not a parameter!
    half_extents_ = torch::tensor({half_width, half_depth}, torch::requires_grad(true));

    // Initialize robot pose as trainable tensors (relative to room at origin)
    robot_pos_ = torch::tensor({robot_x, robot_y}, torch::requires_grad(true));
    robot_theta_ = torch::tensor({robot_theta}, torch::requires_grad(true));

    // Register all parameters for optimization (5 total)
    register_parameter("half_extents", half_extents_);
    register_parameter("robot_pos", robot_pos_);
    register_parameter("robot_theta", robot_theta_);
}

torch::Tensor RoomModel::transform_to_room_frame(const torch::Tensor& points_robot) const
{
    // Transform points from robot frame to room frame (at origin)
    // p_room = R(theta) * p_robot + robot_pos

    // Get rotation angle
    const torch::Tensor cos_theta = torch::cos(robot_theta_);
    const torch::Tensor sin_theta = torch::sin(robot_theta_);

    // Create rotation matrix [2x2]
    const torch::Tensor rotation = torch::stack({
        torch::stack({cos_theta.squeeze(), -sin_theta.squeeze()}),
        torch::stack({sin_theta.squeeze(), cos_theta.squeeze()})
    });

    // Apply rotation: [N, 2] @ [2, 2]^T = [N, 2]
    const torch::Tensor rotated = torch::matmul(points_robot, rotation.transpose(0, 1));

    // Add translation: [N, 2] + [2] (broadcast)
    torch::Tensor points_room = rotated + robot_pos_;

    return points_room;
}

torch::Tensor RoomModel::sdf(const torch::Tensor& points_robot) const
{
    // Transform points from robot frame to room frame
    const torch::Tensor points_room = transform_to_room_frame(points_robot);

    // Compute absolute distances from origin in each dimension
    const torch::Tensor abs_points = torch::abs(points_room);

    // Distance from each point to the box in each dimension
    const torch::Tensor d = abs_points - half_extents_;

    // SDF for axis-aligned box at origin
    const torch::Tensor outside_distance = torch::norm(
        torch::max(d, torch::zeros_like(d)),
        2,
        /*dim=*/1
    );

    const torch::Tensor inside_distance = torch::clamp_max(
        torch::max(d.select(1, 0), d.select(1, 1)),
        0.0
    );

    return outside_distance + inside_distance;
}

// In room_model.cpp:

void RoomModel::init_odometry_calibration(float k_trans, float k_rot)
{
    k_translation_ = torch::tensor({k_trans}, torch::requires_grad(true)); // Start trainable
    k_rotation_ = torch::tensor({k_rot}, torch::requires_grad(true));

    register_parameter("k_translation", k_translation_);
    register_parameter("k_rotation", k_rotation_);

    qInfo() << "Odometry calibration initialized: k_trans=" << k_trans
            << "k_rot=" << k_rot << "(trainable)";
}

std::vector<float> RoomModel::get_odometry_calibration() const
{
    return {k_translation_.item<float>(), k_rotation_.item<float>()};
}

void RoomModel::freeze_odometry_calibration()
{
    if (k_translation_.defined()) k_translation_.set_requires_grad(false);
    if (k_rotation_.defined()) k_rotation_.set_requires_grad(false);
}

void RoomModel::unfreeze_odometry_calibration()
{
    if (k_translation_.defined()) k_translation_.set_requires_grad(true);
    if (k_rotation_.defined()) k_rotation_.set_requires_grad(true);
}

Eigen::Vector3f RoomModel::calibrate_velocity(const VelocityCommand& cmd, float dt) const
{
    float k_t = k_translation_.item<float>();
    float k_r = k_rotation_.item<float>();

    // Apply calibration
    float dx_local = (cmd.adv_x * k_t * dt) / 1000.0f;
    float dy_local = (cmd.adv_z * k_t * dt) / 1000.0f;
    float dtheta = -cmd.rot * k_r * dt;

    // Transform to room frame
    auto current_pose = get_robot_pose();
    float theta = current_pose[2];

    Eigen::Vector3f delta;
    delta[0] = dx_local * std::cos(theta) - dy_local * std::sin(theta);
    delta[1] = dx_local * std::sin(theta) + dy_local * std::cos(theta);
    delta[2] = dtheta;

    return delta;
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

Eigen::Vector3f RoomModel::get_robot_pose_as_eigen() const
{
    return Eigen::Vector3f{
        robot_pos_.accessor<float, 1>()[0],
        robot_pos_.accessor<float, 1>()[1],
        robot_theta_.accessor<float, 1>()[0]
    };
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

std::vector<torch::Tensor> RoomModel::get_calibration_parameters() const {
    std::vector<torch::Tensor> params;
    if (k_translation_.defined() && k_translation_.requires_grad())
        params.push_back(k_translation_);
    if (k_rotation_.defined() && k_rotation_.requires_grad())
        params.push_back(k_rotation_);
    return params;
}

torch::Tensor RoomModel::predict_pose_tensor(const torch::Tensor& current_pose_tensor,
                                             const VelocityCommand& cmd,
                                             float dt) const
{
    // Ensure calibration params are scalars (but keep gradients!)
    const auto k_t = k_translation_.squeeze();  // [1] -> scalar, keeps requires_grad
    const auto k_r = k_rotation_.squeeze();     // [1] -> scalar, keeps requires_grad

    // Convert velocity commands to tensors for autodiff
    const auto v_x_raw = torch::tensor(cmd.adv_x / 1000.0f, torch::kFloat32);  // mm/s -> m/s
    const auto v_z_raw = torch::tensor(cmd.adv_z / 1000.0f, torch::kFloat32);
    const auto w_raw = torch::tensor(-cmd.rot, torch::kFloat32);
    const auto dt_tensor = torch::tensor(dt, torch::kFloat32);

    // Apply calibration (maintains gradients)
    const auto v_x = v_x_raw * k_t * dt_tensor;
    const auto v_z = v_z_raw * k_t * dt_tensor;
    const auto w = w_raw * k_r * dt_tensor;

    // Extract current pose (scalars)
    const auto x = current_pose_tensor[0];
    const auto y = current_pose_tensor[1];
    const auto theta = current_pose_tensor[2];

    // Transform local velocities to global frame (all tensor ops)
    const auto cos_theta = torch::cos(theta);
    const auto sin_theta = torch::sin(theta);

    const auto dx_global = v_x * cos_theta - v_z * sin_theta;
    const auto dy_global = v_x * sin_theta + v_z * cos_theta;

    // Build predicted pose ensuring 1D shape [3]
    auto new_x = (x + dx_global).reshape({1});
    auto new_y = (y + dy_global).reshape({1});
    auto new_theta = (theta + w).reshape({1});

    // Concatenate to form [3] tensor
    return torch::cat({new_x, new_y, new_theta}, 0);
}

torch::Tensor RoomModel::predict_pose_from_delta(const torch::Tensor& current_pose_tensor,
                                                  const Eigen::Vector3f& delta_pose) const
{
    // This version uses the pre-integrated delta directly
    // More reliable than predict_pose_tensor when velocity_cmd might be stale

    // Extract current pose
    const auto x = current_pose_tensor[0];
    const auto y = current_pose_tensor[1];
    const auto theta = current_pose_tensor[2];

    // Convert delta to tensors (no gradients needed - delta is constant)
    const auto dx = torch::tensor(delta_pose[0], torch::kFloat32);
    const auto dy = torch::tensor(delta_pose[1], torch::kFloat32);
    const auto dtheta = torch::tensor(delta_pose[2], torch::kFloat32);

    // Add delta to current pose
    auto new_x = (x + dx).reshape({1});
    auto new_y = (y + dy).reshape({1});
    auto new_theta = (theta + dtheta).reshape({1});

    // Concatenate to form [3] tensor
    return torch::cat({new_x, new_y, new_theta}, 0);
}

void RoomModel::print_info() const
{
    const auto room_params = get_room_parameters();
    const auto robot_pose = get_robot_pose();

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

    if (k_translation_.defined() && k_rotation_.defined())
    {
        const auto calib = get_odometry_calibration();
        std::cout << "\n=== ODOMETRY CALIBRATION ===\n";
        std::cout << "  Translation scale: " << calib[0] << " "
                  << (k_translation_.requires_grad() ? "(trainable)" : "(frozen)") << "\n";
        std::cout << "  Rotation scale: " << calib[1] << " "
                  << (k_rotation_.requires_grad() ? "(trainable)" : "(frozen)") << "\n";
    }
}