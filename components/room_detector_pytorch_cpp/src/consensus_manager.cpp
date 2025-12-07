/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "consensus_manager.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <QDebug>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ConsensusManager::ConsensusManager(QObject* parent)
    : QObject(parent)
    , initialized_(false)
    , has_room_(false)
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

// =============================================================================
// PUBLIC SLOTS
// =============================================================================

void ConsensusManager::onRoomUpdated(const std::shared_ptr<RoomModel>& room_model,
                                      const Eigen::Vector3f& robot_pose,
                                      const Eigen::Matrix3f& robot_covariance,
                                      const std::vector<float>& room_params)
{
    QMutexLocker lock(&mutex_);

    // Cache data
    cached_robot_pose_ = robot_pose;
    cached_robot_covariance_ = robot_covariance;
    cached_room_params_ = room_params;

    // Initialize on first room update
    if (!has_room_)
    {
        initializeFromRoom(room_model, robot_pose, robot_covariance, room_params);
        has_room_ = true;
    }

    // If we have doors, run optimization
    if (initialized_ && !door_models_.empty())
    {
        // Update robot pose in graph (could add odometry factor here)
        // For now, we just run optimization with current state
        // runOptimization();  // Uncomment if you want continuous updates
    }
}

void ConsensusManager::onDoorDetected(const std::shared_ptr<DoorModel>& door_model,
                                       const Eigen::Matrix3f& detection_covariance)
{
    QMutexLocker lock(&mutex_);

    if (!initialized_)
    {
        qWarning() << "ConsensusManager::onDoorDetected - Not initialized yet, ignoring";
        return;
    }

    // Check if this door is already tracked (by ID)
    for (const auto& existing : door_models_)
    {
        if (existing->id == door_model->id)
        {
            qInfo() << "ConsensusManager: Door" << door_model->id << "already tracked";
            return;
        }
    }

    // Add new door
    size_t door_index = addDoor(door_model, detection_covariance);
    qInfo() << "ConsensusManager: Added door" << door_model->id << "at index" << door_index;

    // Run initial optimization
    lock.unlock();  // Unlock before optimization (it will lock internally)
    runOptimization();
}

void ConsensusManager::onDoorUpdated(const Eigen::Vector3f& door_pose,
                                      const Eigen::Matrix3f& door_covariance)
{
    QMutexLocker lock(&mutex_);

    if (!initialized_ || door_models_.empty())
    {
        return;
    }

    // For now, just trigger optimization
    // In a more sophisticated version, we'd add an observation factor
    lock.unlock();
    runOptimization();
}

void ConsensusManager::onDoorTrackingLost()
{
    QMutexLocker lock(&mutex_);
    qInfo() << "ConsensusManager: Door tracking lost";
    // Could clear door models here, or mark as uncertain
}

void ConsensusManager::runOptimization()
{
    QMutexLocker lock(&mutex_);

    if (!initialized_)
    {
        qWarning() << "ConsensusManager::runOptimization - Not initialized";
        return;
    }

    if (door_models_.empty())
    {
        qDebug() << "ConsensusManager::runOptimization - No doors to optimize";
        return;
    }

    try
    {
        // Run optimization
        ConsensusResult result = optimize();

        // Emit full result
        Q_EMIT consensusReady(result);

        // Emit door priors and visualization poses
        for (size_t i = 0; i < door_models_.size(); ++i)
        {
            if (auto prior = getDoorConsensusPrior(i))
            {
                Q_EMIT doorPriorReady(i, prior->first, prior->second);

                // Emit door pose in room coordinates for 3D visualization
                // Get geometry from door model, pose from consensus
                const auto& door_model = door_models_[i];
                auto geom = door_model->get_door_geometry();

                // Opening angle is relative to the door frame, not room/robot frame
                // So we can use it directly from the detector
                float opening_angle = door_model->get_opening_angle();

                // Get z from original door model (consensus only handles 2D)
                auto original_pose = door_model->get_door_pose();
                float z = original_pose[2];  // z position

                Q_EMIT doorPoseInRoom(i,
                    prior->first.x(),      // x from consensus
                    prior->first.y(),      // y from consensus
                    z,                     // z from detector
                    prior->first.z(),      // theta from consensus
                    geom[0],               // width from detector
                    geom[1],               // height from detector
                    opening_angle          // opening angle from detector (relative to door frame)
                );

                qDebug() << "ConsensusManager: Emitted door" << i << "in room frame:"
                         << prior->first.x() << prior->first.y() << z << prior->first.z()
                         << "opening:" << opening_angle;
            }
        }
    }
    catch (const std::exception& e)
    {
        qWarning() << "ConsensusManager::runOptimization - Exception:" << e.what();
    }
}

void ConsensusManager::reset()
{
    QMutexLocker lock(&mutex_);

    graph_ = ConsensusGraph();  // Reset graph
    initialized_ = false;
    has_room_ = false;
    current_robot_pose_index_ = 0;
    room_model_.reset();
    door_models_.clear();
    room_half_width_ = 0.0;
    room_half_depth_ = 0.0;

    qInfo() << "ConsensusManager: Reset";
}

// =============================================================================
// PRIVATE METHODS
// =============================================================================

void ConsensusManager::initializeFromRoom(const std::shared_ptr<RoomModel>& room_model,
                                          const Eigen::Vector3f& robot_pose_in_room,
                                          const Eigen::Matrix3f& robot_pose_covariance,
                                          const std::vector<float>& room_params)
{
    if (initialized_)
    {
        qWarning() << "ConsensusManager: Already initialized";
        return;
    }

    // Store room model reference
    room_model_ = room_model;

    // Validate room params
    if (room_params.size() < 2)
    {
        qWarning() << "ConsensusManager: Invalid room_params size";
        return;
    }
    room_half_width_ = room_params[0];
    room_half_depth_ = room_params[1];

    // Initialize the factor graph with room at origin
    Eigen::Vector3d room_uncertainty(0.1, 0.1, 0.05);
    graph_.initializeRoom(room_half_width_, room_half_depth_, room_uncertainty);

    // Add initial robot pose
    gtsam::Pose2 robot_pose = toPose2(robot_pose_in_room);

    // Extract sigmas from covariance diagonal
    Eigen::Vector3d pose_sigmas(
        std::sqrt(std::max(1e-6f, robot_pose_covariance(0, 0))),
        std::sqrt(std::max(1e-6f, robot_pose_covariance(1, 1))),
        std::sqrt(std::max(1e-6f, robot_pose_covariance(2, 2)))
    );

    current_robot_pose_index_ = graph_.addRobotPose(robot_pose, pose_sigmas);

    initialized_ = true;
    graph_dirty_ = true;

    qInfo() << "ConsensusManager: Initialized with room"
            << (2 * room_half_width_) << "x" << (2 * room_half_depth_) << "m";
    qInfo() << "  Robot pose:" << robot_pose_in_room.x()
            << robot_pose_in_room.y() << robot_pose_in_room.z();

    Q_EMIT initialized();
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

    // Normalize theta to [-π, π]
    float norm_theta = std::fmod(theta + M_PI, 2 * M_PI) - M_PI;

    // Score each wall by distance + orientation match
    auto score = [&](WallID wall, float dist, float expected_theta) -> float {
        float theta_diff = std::abs(norm_theta - expected_theta);
        if (theta_diff > M_PI) theta_diff = 2 * M_PI - theta_diff;
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
    switch (wall)
    {
        case WallID::NORTH:
        case WallID::SOUTH:
            return door_pose.x();

        case WallID::EAST:
        case WallID::WEST:
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
        throw std::runtime_error("ConsensusManager: Must initialize before adding doors");
    }

    // Get door pose from model (in robot frame - this is how the detector sees it)
    const auto door_params = door_model->get_door_pose();
    Eigen::Vector3f door_pose_robot(door_params[0], door_params[1], door_params[3]); // x, y, theta in robot frame

    // Get current robot pose in room frame
    gtsam::Pose2 robot_pose_room = graph_.getValues().at<gtsam::Pose2>(
        ConsensusGraph::RobotSymbol(current_robot_pose_index_));

    // Transform door pose from robot frame to room frame
    // door_room = robot_room * door_robot
    gtsam::Pose2 door_pose_robot_gtsam(door_pose_robot.x(), door_pose_robot.y(), door_pose_robot.z());
    gtsam::Pose2 door_pose_room = robot_pose_room.compose(door_pose_robot_gtsam);

    Eigen::Vector3f door_pose_in_room(door_pose_room.x(), door_pose_room.y(), door_pose_room.theta());

    // Determine wall attachment (using room frame pose)
    WallID attached_wall = determineWallAttachment(door_pose_in_room);
    double wall_offset = computeWallOffset(attached_wall, door_pose_in_room);

    // NOTE: We do NOT flip the door orientation here.
    // The detector's theta is correct for visualization (opening angle is relative to it).
    // The wall attachment constraint will handle geometric consistency.

    const char* wall_names[] = {"NORTH", "SOUTH", "EAST", "WEST"};
    qInfo() << "ConsensusManager: Adding door";
    qInfo() << "  Room frame:" << door_pose_in_room.x() << door_pose_in_room.y() << door_pose_in_room.z();
    qInfo() << "  Robot frame:" << door_pose_robot.x() << door_pose_robot.y() << door_pose_robot.z();
    qInfo() << "  Attached to:" << wall_names[static_cast<int>(attached_wall)] << "offset:" << wall_offset;

    // Observation uncertainty from detection covariance
    Eigen::Vector3d obs_sigmas(
        std::sqrt(std::max(1e-6f, detection_covariance(0, 0))),
        std::sqrt(std::max(1e-6f, detection_covariance(1, 1))),
        std::sqrt(std::max(1e-6f, detection_covariance(2, 2)))
    );

    // Add door with observation factor (not prior factor)
    // Use original theta from detector (not flipped)
    gtsam::Pose2 door_room_gtsam = toPose2(door_pose_in_room);
    gtsam::Pose2 door_robot_gtsam(door_pose_robot.x(), door_pose_robot.y(), door_pose_robot.z());

    size_t door_index = graph_.addObjectWithObservation(
        current_robot_pose_index_,
        door_room_gtsam,
        door_robot_gtsam,
        attached_wall,
        wall_offset,
        obs_sigmas
    );

    // Store door model and mark graph as dirty
    door_models_.push_back(door_model);
    graph_dirty_ = true;

    return door_index;
}

ConsensusResult ConsensusManager::optimize()
{
    if (!initialized_)
    {
        throw std::runtime_error("ConsensusManager: Must initialize before optimizing");
    }

    // Skip if nothing changed since last optimization
    if (!graph_dirty_)
    {
        // Return cached result
        return last_result_;
    }

    qDebug() << "ConsensusManager: Running optimization...";

    ConsensusResult result = graph_.optimize();

    // Fix convergence check: converged if error is low OR no iterations needed (already optimal)
    result.converged = (result.iterations == 0) || (result.final_error < result.initial_error);

    qDebug() << "  Initial error:" << result.initial_error
             << "Final error:" << result.final_error
             << "Iterations:" << result.iterations
             << "Converged:" << result.converged;

    // Cache result and mark graph as clean
    last_result_ = result;
    graph_dirty_ = false;

    return result;
}

std::optional<std::pair<Eigen::Vector3f, Eigen::Matrix3f>>
ConsensusManager::getDoorConsensusPrior(size_t door_index) const
{
    if (!initialized_ || door_index >= door_models_.size())
    {
        return std::nullopt;
    }

    try
    {
        gtsam::Marginals marginals(graph_.getGraph(), graph_.getValues());

        auto object_sym = ConsensusGraph::ObjectSymbol(door_index);

        if (!graph_.getValues().exists(object_sym))
        {
            return std::nullopt;
        }

        gtsam::Pose2 opt_pose = graph_.getValues().at<gtsam::Pose2>(object_sym);
        Eigen::Matrix3d cov = marginals.marginalCovariance(object_sym);

        Eigen::Vector3f pose(opt_pose.x(), opt_pose.y(), opt_pose.theta());
        Eigen::Matrix3f cov_f = cov.cast<float>();

        // Set high uncertainty on theta to avoid fighting with detector's measurements
        // The consensus manager's theta might differ from detector's due to orientation correction
        // Let the detector's measurements determine orientation, consensus refines position
        cov_f(2, 2) = std::max(cov_f(2, 2), 1.0f);  // At least 1 radian^2 variance on theta
        cov_f(0, 2) = 0.0f;  // Remove correlations with theta
        cov_f(1, 2) = 0.0f;
        cov_f(2, 0) = 0.0f;
        cov_f(2, 1) = 0.0f;

        return std::make_pair(pose, cov_f);
    }
    catch (const std::exception& e)
    {
        qWarning() << "ConsensusManager::getDoorConsensusPrior - Error:" << e.what();
        return std::nullopt;
    }
}

void ConsensusManager::print() const
{
    QMutexLocker lock(&mutex_);

    qInfo() << "=== ConsensusManager ===";
    qInfo() << "Initialized:" << initialized_;
    qInfo() << "Room:" << (2 * room_half_width_) << "x" << (2 * room_half_depth_) << "m";
    qInfo() << "Robot pose index:" << current_robot_pose_index_;
    qInfo() << "Tracked doors:" << door_models_.size();

    graph_.print("  ");
}