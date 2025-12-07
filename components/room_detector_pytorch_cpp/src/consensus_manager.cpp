/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "consensus_manager.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ConsensusManager::ConsensusManager()
    : initialized_(false)
    , current_robot_pose_index_(0)
{
}

gtsam::Pose2 ConsensusManager::toPose2(const Eigen::Vector3f& pose)
{
    return gtsam::Pose2(pose.x(), pose.y(), pose.z());
}

Eigen::Vector3f ConsensusManager::fromPose2(const gtsam::Pose2& pose)
{
    return Eigen::Vector3f(pose.x(), pose.y(), pose.theta());
}

void ConsensusManager::initializeFromRoom(const std::shared_ptr<RoomModel>& room_model,
                                          const Eigen::Vector3f& robot_pose_in_room,
                                          const Eigen::Matrix3f& robot_pose_covariance)
{
    if (initialized_)
    {
        std::cerr << "ConsensusManager: Already initialized. Clear first to reinitialize.\n";
        return;
    }

    // Store room model reference
    room_model_ = room_model;

    // Get room dimensions from model
    const auto room_params = room_model->get_room_parameters();
    room_half_width_ = room_params[0];   // half_width
    room_half_depth_ = room_params[1];   // half_height (depth in Y direction)

    // Initialize the factor graph with room at origin
    Eigen::Vector3d room_uncertainty(0.1, 0.1, 0.05);
    graph_.initializeRoom(room_half_width_, room_half_depth_, room_uncertainty);

    // Add initial robot pose
    gtsam::Pose2 robot_pose = toPose2(robot_pose_in_room);

    // Extract sigmas from covariance diagonal (sqrt of variances)
    Eigen::Vector3d pose_sigmas(
        std::sqrt(std::max(1e-6f, robot_pose_covariance(0, 0))),
        std::sqrt(std::max(1e-6f, robot_pose_covariance(1, 1))),
        std::sqrt(std::max(1e-6f, robot_pose_covariance(2, 2)))
    );

    current_robot_pose_index_ = graph_.addRobotPose(robot_pose, pose_sigmas);

    initialized_ = true;

    std::cout << "ConsensusManager: Initialized with room "
              << (2 * room_half_width_) << " x " << (2 * room_half_depth_) << " m\n";
    std::cout << "  Initial robot pose: (" << robot_pose_in_room.x() << ", "
              << robot_pose_in_room.y() << ", " << robot_pose_in_room.z() << ")\n";
    std::cout << "  Pose sigmas: (" << pose_sigmas.x() << ", "
              << pose_sigmas.y() << ", " << pose_sigmas.z() << ")\n";
}

WallID ConsensusManager::determineWallAttachment(const Eigen::Vector3f& door_pose) const
{
    const float x = door_pose.x();
    const float y = door_pose.y();
    const float theta = door_pose.z();

    // Compute distances to each wall
    const float dist_north = std::abs(y - room_half_depth_);
    const float dist_south = std::abs(y + room_half_depth_);
    const float dist_east = std::abs(x - room_half_width_);
    const float dist_west = std::abs(x + room_half_width_);

    // Find minimum distance
    const float min_dist = std::min({dist_north, dist_south, dist_east, dist_west});

    // Also consider door orientation for disambiguation
    // Door facing into room: theta should match wall's inward normal
    // North wall (y+): door faces -Y, so theta ≈ -π/2
    // South wall (y-): door faces +Y, so theta ≈ +π/2
    // East wall (x+): door faces -X, so theta ≈ π
    // West wall (x-): door faces +X, so theta ≈ 0

    // Normalize theta to [-π, π]
    float norm_theta = std::fmod(theta + M_PI, 2 * M_PI) - M_PI;

    // Score each wall by distance + orientation match
    auto score = [&](WallID wall, float dist, float expected_theta) -> float {
        float theta_diff = std::abs(norm_theta - expected_theta);
        if (theta_diff > M_PI) theta_diff = 2 * M_PI - theta_diff;
        // Weight: distance matters more for close walls, orientation for ambiguous cases
        return dist + 0.5f * theta_diff;
    };

    float score_n = score(WallID::NORTH, dist_north, -M_PI / 2);
    float score_s = score(WallID::SOUTH, dist_south, M_PI / 2);
    float score_e = score(WallID::EAST, dist_east, M_PI);
    float score_w = score(WallID::WEST, dist_west, 0.0f);

    float min_score = std::min({score_n, score_s, score_e, score_w});

    if (min_score == score_n) return WallID::NORTH;
    if (min_score == score_s) return WallID::SOUTH;
    if (min_score == score_e) return WallID::EAST;
    return WallID::WEST;
}

double ConsensusManager::computeWallOffset(WallID wall, const Eigen::Vector3f& door_pose) const
{
    // Offset is the position along the wall (in wall's local X direction)
    switch (wall)
    {
        case WallID::NORTH:
        case WallID::SOUTH:
            // Horizontal walls: offset is X position
            return door_pose.x();

        case WallID::EAST:
        case WallID::WEST:
            // Vertical walls: offset is Y position
            return door_pose.y();

        default:
            return 0.0;
    }
}

size_t ConsensusManager::addDoor(const std::shared_ptr<DoorModel>& door_model,
                                  const Eigen::Matrix3f& detection_covariance)
{
    if (!initialized_)
    {
        throw std::runtime_error("ConsensusManager: Must initialize from room before adding doors");
    }

    // Get door pose from model
    const auto door_params = door_model->get_door_pose();
    Eigen::Vector3f door_pose(door_params[0], door_params[1], door_params[3]); // x, y, theta

    // Determine which wall the door is attached to
    WallID attached_wall = determineWallAttachment(door_pose);
    double wall_offset = computeWallOffset(attached_wall, door_pose);

    std::cout << "ConsensusManager: Adding door at (" << door_pose.x() << ", "
              << door_pose.y() << ", " << door_pose.z() << ")\n";
    std::cout << "  Attached to wall: ";
    switch (attached_wall)
    {
        case WallID::NORTH: std::cout << "NORTH"; break;
        case WallID::SOUTH: std::cout << "SOUTH"; break;
        case WallID::EAST: std::cout << "EAST"; break;
        case WallID::WEST: std::cout << "WEST"; break;
    }
    std::cout << ", offset: " << wall_offset << " m\n";

    // Add to graph
    gtsam::Pose2 pose = toPose2(door_pose);

    // Extract sigmas from covariance diagonal
    Eigen::Vector3d pose_sigmas(
        std::sqrt(std::max(1e-6f, detection_covariance(0, 0))),
        std::sqrt(std::max(1e-6f, detection_covariance(1, 1))),
        std::sqrt(std::max(1e-6f, detection_covariance(2, 2)))
    );

    std::cout << "  Detection sigmas: (" << pose_sigmas.x() << ", "
              << pose_sigmas.y() << ", " << pose_sigmas.z() << ")\n";

    size_t door_index = graph_.addObject(pose, attached_wall, wall_offset, pose_sigmas);

    // Store door model reference
    door_models_.push_back(door_model);

    return door_index;
}

size_t ConsensusManager::updateRobotPose(const Eigen::Vector3f& new_pose,
                                         const Eigen::Vector3f& pose_uncertainty,
                                         const std::optional<Eigen::Vector3f>& odometry,
                                         const Eigen::Vector3f& odom_uncertainty)
{
    if (!initialized_)
    {
        throw std::runtime_error("ConsensusManager: Must initialize before updating robot pose");
    }

    gtsam::Pose2 pose = toPose2(new_pose);
    Eigen::Vector3d pose_unc(pose_uncertainty.x(),
                             pose_uncertainty.y(),
                             pose_uncertainty.z());

    std::optional<gtsam::Pose2> odom_pose;
    if (odometry.has_value())
    {
        odom_pose = toPose2(odometry.value());
    }

    Eigen::Vector3d odom_unc(odom_uncertainty.x(),
                             odom_uncertainty.y(),
                             odom_uncertainty.z());

    current_robot_pose_index_ = graph_.addRobotPose(pose, pose_unc, odom_pose, odom_unc);

    return current_robot_pose_index_;
}

void ConsensusManager::addDoorObservation(size_t door_index,
                                          const Eigen::Vector3f& observed_pose,
                                          const Eigen::Vector3f& observation_uncertainty)
{
    if (!initialized_)
    {
        throw std::runtime_error("ConsensusManager: Must initialize before adding observations");
    }

    gtsam::Pose2 obs = toPose2(observed_pose);
    Eigen::Vector3d obs_unc(observation_uncertainty.x(),
                            observation_uncertainty.y(),
                            observation_uncertainty.z());

    graph_.addObjectObservation(current_robot_pose_index_, door_index, obs, obs_unc);
}

ConsensusResult ConsensusManager::optimize()
{
    if (!initialized_)
    {
        throw std::runtime_error("ConsensusManager: Must initialize before optimizing");
    }

    std::cout << "ConsensusManager: Running optimization...\n";
    graph_.print("  ");

    ConsensusResult result = graph_.optimize();

    std::cout << "  Initial error: " << result.initial_error << "\n";
    std::cout << "  Final error: " << result.final_error << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << "\n";

    return result;
}

void ConsensusManager::applyResults(const ConsensusResult& result)
{
    // Apply optimized poses back to the models

    // Note: Since room is fixed at origin in our formulation,
    // we don't update the room model itself. Instead, we update
    // the robot's pose in room frame.

    // Update door models with optimized poses
    for (size_t i = 0; i < door_models_.size(); ++i)
    {
        if (result.object_poses.count(i) > 0)
        {
            const auto& opt_pose = result.object_poses.at(i);

            // Get current door z position (height is separate from 2D pose)
            const auto current_params = door_models_[i]->get_door_pose();
            const float z = current_params[2];

            // Apply optimized x, y, theta
            door_models_[i]->set_pose(opt_pose.x(), opt_pose.y(), z, opt_pose.theta());

            std::cout << "ConsensusManager: Updated door " << i << " pose to ("
                      << opt_pose.x() << ", " << opt_pose.y() << ", " << opt_pose.theta() << ")\n";
        }
    }
}

void ConsensusManager::print() const
{
    std::cout << "=== ConsensusManager ===" << std::endl;
    std::cout << "Initialized: " << (initialized_ ? "yes" : "no") << std::endl;
    std::cout << "Room dimensions: " << (2 * room_half_width_) << " x "
              << (2 * room_half_depth_) << " m" << std::endl;
    std::cout << "Current robot pose index: " << current_robot_pose_index_ << std::endl;
    std::cout << "Tracked doors: " << door_models_.size() << std::endl;

    graph_.print("  ");
}

std::optional<std::pair<Eigen::Vector3f, Eigen::Matrix3f>>
ConsensusManager::getDoorConsensusPrior(size_t door_index) const
{
    if (!initialized_ || door_index >= door_models_.size())
    {
        return std::nullopt;
    }

    // Get optimized pose and covariance from last optimization result
    // Note: This requires storing the last result or recomputing marginals
    try
    {
        // Compute marginals to get current estimate and uncertainty
        gtsam::Marginals marginals(graph_.getGraph(), graph_.getValues());

        auto object_sym = ConsensusGraph::ObjectSymbol(door_index);

        // Check if the object exists in values
        if (!graph_.getValues().exists(object_sym))
        {
            return std::nullopt;
        }

        gtsam::Pose2 opt_pose = graph_.getValues().at<gtsam::Pose2>(object_sym);
        Eigen::Matrix3d cov = marginals.marginalCovariance(object_sym);

        Eigen::Vector3f pose(opt_pose.x(), opt_pose.y(), opt_pose.theta());
        Eigen::Matrix3f cov_f = cov.cast<float>();

        return std::make_pair(pose, cov_f);
    }
    catch (const std::exception& e)
    {
        std::cerr << "ConsensusManager::getDoorConsensusPrior() - Error: " << e.what() << "\n";
        return std::nullopt;
    }
}