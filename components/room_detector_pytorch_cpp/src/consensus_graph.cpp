/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "consensus_graph.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ConsensusGraph::ConsensusGraph()
{
    clear();
}

void ConsensusGraph::clear()
{
    graph_ = gtsam::NonlinearFactorGraph();
    values_ = gtsam::Values();
    room_initialized_ = false;
    room_half_width_ = 0.0;
    room_half_depth_ = 0.0;
    robot_pose_count_ = 0;
    object_count_ = 0;
    wall_relative_poses_.clear();
}

ConsensusGraph::SharedNoiseModel ConsensusGraph::createNoiseModel(const Eigen::Vector3d& sigmas) const
{
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::Pose2 ConsensusGraph::computeWallPose(WallID wall, double half_width, double half_depth) const
{
    // Wall poses are defined in room frame
    // Each wall's local frame has:
    //   - Origin at wall center
    //   - X axis along the wall (left to right when facing the wall from inside)
    //   - Y axis pointing into the room

    switch (wall)
    {
        case WallID::NORTH:
            // North wall at y = +half_depth, facing -Y (into room)
            return Pose2(0.0, half_depth, -M_PI / 2.0);

        case WallID::SOUTH:
            // South wall at y = -half_depth, facing +Y (into room)
            return Pose2(0.0, -half_depth, M_PI / 2.0);

        case WallID::EAST:
            // East wall at x = +half_width, facing -X (into room)
            return Pose2(half_width, 0.0, M_PI);

        case WallID::WEST:
            // West wall at x = -half_width, facing +X (into room)
            return Pose2(-half_width, 0.0, 0.0);

        default:
            throw std::runtime_error("Unknown wall ID");
    }
}

void ConsensusGraph::addRigidWallConstraint(WallID wall, const Pose2& relative_pose)
{
    // Very tight constraint - walls are rigidly attached to room
    // Use very small sigma (essentially zero DOF)
    const Eigen::Vector3d rigid_sigmas(0.001, 0.001, 0.001);
    auto noise = createNoiseModel(rigid_sigmas);

    // BetweenFactor: Room -> Wall with relative_pose
    graph_.add(gtsam::BetweenFactor<Pose2>(
        RoomSymbol(),
        WallSymbol(wall),
        relative_pose,
        noise
    ));

    // Store for later use
    wall_relative_poses_[wall] = relative_pose;
}

void ConsensusGraph::initializeRoom(double half_width,
                                    double half_depth,
                                    const Eigen::Vector3d& room_uncertainty)
{
    if (room_initialized_)
    {
        std::cerr << "Warning: Room already initialized. Call clear() first to reinitialize.\n";
        return;
    }

    room_half_width_ = half_width;
    room_half_depth_ = half_depth;

    // 1. Add room at origin (fixed - this eliminates gauge freedom)
    Pose2 room_pose(0.0, 0.0, 0.0);
    values_.insert(RoomSymbol(), room_pose);

    // Very tight prior to fix room at origin
    const Eigen::Vector3d fixed_sigmas(0.001, 0.001, 0.001);
    auto fixed_noise = createNoiseModel(fixed_sigmas);
    graph_.add(gtsam::PriorFactor<Pose2>(RoomSymbol(), room_pose, fixed_noise));

    // 2. Add four walls with rigid constraints
    for (WallID wall : {WallID::NORTH, WallID::SOUTH, WallID::EAST, WallID::WEST})
    {
        Pose2 wall_pose = computeWallPose(wall, half_width, half_depth);
        values_.insert(WallSymbol(wall), wall_pose);
        addRigidWallConstraint(wall, wall_pose);
    }

    room_initialized_ = true;

    std::cout << "ConsensusGraph: Room initialized at origin with dimensions "
              << (2 * half_width) << " x " << (2 * half_depth) << " m\n";
}

size_t ConsensusGraph::addRobotPose(const Pose2& pose,
                                    const Eigen::Vector3d& pose_uncertainty,
                                    const std::optional<Pose2>& odometry_from_previous,
                                    const Eigen::Vector3d& odometry_uncertainty)
{
    if (!room_initialized_)
    {
        throw std::runtime_error("Room must be initialized before adding robot poses");
    }

    size_t index = robot_pose_count_++;
    Symbol robot_sym = RobotSymbol(index);

    // Add robot pose to values
    values_.insert(robot_sym, pose);

    // Add prior factor for this pose (from detector estimate)
    auto pose_noise = createNoiseModel(pose_uncertainty);
    graph_.add(gtsam::PriorFactor<Pose2>(robot_sym, pose, pose_noise));

    // Add odometry factor if not the first pose
    if (index > 0 && odometry_from_previous.has_value())
    {
        Symbol prev_robot_sym = RobotSymbol(index - 1);
        auto odom_noise = createNoiseModel(odometry_uncertainty);

        graph_.add(gtsam::BetweenFactor<Pose2>(
            prev_robot_sym,
            robot_sym,
            odometry_from_previous.value(),
            odom_noise
        ));
    }

    return index;
}

void ConsensusGraph::addRoomObservation(size_t robot_index,
                                        const Pose2& observed_room_pose,
                                        const Eigen::Vector3d& observation_uncertainty)
{
    if (robot_index >= robot_pose_count_)
    {
        throw std::runtime_error("Invalid robot index for room observation");
    }

    // The observation is: from robot's perspective, where is the room?
    // This creates a BetweenFactor: robot -> room
    // The relative pose is the inverse of robot pose in room frame

    auto obs_noise = createNoiseModel(observation_uncertainty);

    graph_.add(gtsam::BetweenFactor<Pose2>(
        RobotSymbol(robot_index),
        RoomSymbol(),
        observed_room_pose,
        obs_noise
    ));
}

size_t ConsensusGraph::addObject(const Pose2& object_pose,
                                 WallID attached_wall,
                                 double wall_offset,
                                 const Eigen::Vector3d& pose_uncertainty)
{
    if (!room_initialized_)
    {
        throw std::runtime_error("Room must be initialized before adding objects");
    }

    size_t index = object_count_++;
    Symbol object_sym = ObjectSymbol(index);

    // Add object pose to values (initial estimate)
    values_.insert(object_sym, object_pose);

    // NOTE: We do NOT add a prior factor here anymore.
    // The door pose will be constrained by:
    // 1. Wall attachment factor (door is ON the wall)
    // 2. Robot observation factor (robot saw door at relative pose)

    // Add wall attachment constraint (pass door's theta in room frame)
    addWallAttachmentConstraint(index, attached_wall, wall_offset, object_pose.theta());

    return index;
}

size_t ConsensusGraph::addObjectWithObservation(
    size_t robot_index,
    const Pose2& object_pose_in_room,
    const Pose2& object_pose_in_robot,
    WallID attached_wall,
    double wall_offset,
    const Eigen::Vector3d& observation_uncertainty)
{
    if (!room_initialized_)
    {
        throw std::runtime_error("Room must be initialized before adding objects");
    }
    if (robot_index >= robot_pose_count_)
    {
        throw std::runtime_error("Invalid robot index for object observation");
    }

    size_t index = object_count_++;
    Symbol object_sym = ObjectSymbol(index);

    // Add object pose to values (initial estimate in room frame)
    values_.insert(object_sym, object_pose_in_room);

    // Add wall attachment constraint
    addWallAttachmentConstraint(index, attached_wall, wall_offset, object_pose_in_room.theta());

    // Add robot → object observation factor
    // This says: "From robot's current pose, I observed the object at this relative pose"
    auto obs_noise = createNoiseModel(observation_uncertainty);
    graph_.add(gtsam::BetweenFactor<Pose2>(
        RobotSymbol(robot_index),
        object_sym,
        object_pose_in_robot,
        obs_noise
    ));

    std::cout << "  Added observation factor: Robot" << robot_index << " -> Object" << index << std::endl;
    std::cout << "  Relative pose (in robot frame): (" << object_pose_in_robot.x()
              << ", " << object_pose_in_robot.y() << ", " << object_pose_in_robot.theta() << ")" << std::endl;

    return index;
}

void ConsensusGraph::addObjectObservation(size_t robot_index,
                                          size_t object_index,
                                          const Pose2& observed_pose,
                                          const Eigen::Vector3d& observation_uncertainty)
{
    if (robot_index >= robot_pose_count_)
    {
        throw std::runtime_error("Invalid robot index for object observation");
    }
    if (object_index >= object_count_)
    {
        throw std::runtime_error("Invalid object index for observation");
    }

    auto obs_noise = createNoiseModel(observation_uncertainty);

    // BetweenFactor: robot -> object (relative observation)
    graph_.add(gtsam::BetweenFactor<Pose2>(
        RobotSymbol(robot_index),
        ObjectSymbol(object_index),
        observed_pose,
        obs_noise
    ));
}

void ConsensusGraph::addWallAttachmentConstraint(size_t object_index,
                                                  WallID wall,
                                                  double offset_along_wall,
                                                  double door_theta_in_room,
                                                  const Eigen::Vector3d& constraint_uncertainty)
{
    if (object_index >= object_count_)
    {
        throw std::runtime_error("Invalid object index for wall constraint");
    }

    // Wall frame definition (from computeWallPose):
    //   - Origin at wall center
    //   - X axis along wall surface
    //   - Y axis pointing INTO the room
    //   - Wall theta = angle of wall's Y axis in room frame
    //
    // To preserve the detected door orientation, we compute door's theta in wall frame:
    //   theta_in_wall = theta_in_room - wall_theta
    //
    // This way the constraint preserves the detector's orientation, while
    // enforcing that the door is ON the wall surface (y = 0 in wall frame).

    Pose2 wall_pose = computeWallPose(wall, room_half_width_, room_half_depth_);
    double wall_theta = wall_pose.theta();

    // Compute door's theta relative to wall frame
    double theta_in_wall = door_theta_in_room - wall_theta;

    // Normalize to [-π, π]
    while (theta_in_wall > M_PI) theta_in_wall -= 2 * M_PI;
    while (theta_in_wall < -M_PI) theta_in_wall += 2 * M_PI;

    // Door position in wall frame: (offset along wall, 0 on wall surface, relative theta)
    Pose2 object_in_wall_frame(offset_along_wall, 0.0, theta_in_wall);

    auto constraint_noise = createNoiseModel(constraint_uncertainty);

    // BetweenFactor: wall -> object
    graph_.add(gtsam::BetweenFactor<Pose2>(
        WallSymbol(wall),
        ObjectSymbol(object_index),
        object_in_wall_frame,
        constraint_noise
    ));

    std::cout << "  Wall attachment: wall_theta=" << wall_theta
              << " door_theta_room=" << door_theta_in_room
              << " theta_in_wall=" << theta_in_wall << std::endl;
}

ConsensusResult ConsensusGraph::optimize(int max_iterations, double convergence_threshold)
{
    ConsensusResult result;

    if (!room_initialized_)
    {
        std::cerr << "Warning: Cannot optimize - room not initialized\n";
        result.converged = false;
        return result;
    }

    // Calculate initial error
    result.initial_error = graph_.error(values_);

    // Configure optimizer
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = max_iterations;
    params.relativeErrorTol = convergence_threshold;
    params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;

    // Run optimization
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, values_, params);
    gtsam::Values optimized_values = optimizer.optimize();

    result.final_error = graph_.error(optimized_values);
    result.iterations = optimizer.iterations();
    result.converged = (result.final_error < result.initial_error);

    // Extract room pose
    result.room_pose = optimized_values.at<Pose2>(RoomSymbol());

    // Extract wall poses
    for (WallID wall : {WallID::NORTH, WallID::SOUTH, WallID::EAST, WallID::WEST})
    {
        result.wall_poses[wall] = optimized_values.at<Pose2>(WallSymbol(wall));
    }

    // Extract robot poses
    for (size_t i = 0; i < robot_pose_count_; ++i)
    {
        result.robot_poses.push_back(optimized_values.at<Pose2>(RobotSymbol(i)));
    }

    // Extract object poses
    for (size_t i = 0; i < object_count_; ++i)
    {
        result.object_poses[i] = optimized_values.at<Pose2>(ObjectSymbol(i));
    }

    // Compute marginal covariances
    try
    {
        gtsam::Marginals marginals(graph_, optimized_values);

        result.room_covariance = marginals.marginalCovariance(RoomSymbol());

        for (WallID wall : {WallID::NORTH, WallID::SOUTH, WallID::EAST, WallID::WEST})
        {
            result.wall_covariances[wall] = marginals.marginalCovariance(WallSymbol(wall));
        }

        for (size_t i = 0; i < robot_pose_count_; ++i)
        {
            result.robot_covariances.push_back(marginals.marginalCovariance(RobotSymbol(i)));
        }

        for (size_t i = 0; i < object_count_; ++i)
        {
            result.object_covariances[i] = marginals.marginalCovariance(ObjectSymbol(i));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Warning: Could not compute marginal covariances: " << e.what() << "\n";
    }

    // Update internal values with optimized result
    values_ = optimized_values;

    return result;
}

void ConsensusGraph::print(const std::string& prefix) const
{
    std::cout << prefix << "=== ConsensusGraph ===" << std::endl;
    std::cout << prefix << "Room initialized: " << (room_initialized_ ? "yes" : "no") << std::endl;

    if (room_initialized_)
    {
        std::cout << prefix << "Room dimensions: " << (2 * room_half_width_) << " x "
                  << (2 * room_half_depth_) << " m" << std::endl;
    }

    std::cout << prefix << "Robot poses: " << robot_pose_count_ << std::endl;
    std::cout << prefix << "Objects: " << object_count_ << std::endl;
    std::cout << prefix << "Total factors: " << graph_.size() << std::endl;
    std::cout << prefix << "Total variables: " << values_.size() << std::endl;

    if (!values_.empty())
    {
        std::cout << prefix << "Current error: " << graph_.error(values_) << std::endl;
    }

    std::cout << prefix << "======================" << std::endl;
}