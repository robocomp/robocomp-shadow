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

#include "room_freezing_manager.h"

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
        last_robot_pose_ = robot_pose;
        has_last_pose_ = true;
        return;
    }

    // If the room estimate has grown/shrunk by > 20cm, we need to rebuild the graph so the walls move to the correct place.
    if (std::abs(room_params[0] - room_half_width_) > 0.5f || std::abs(room_params[1] - room_half_depth_) > 0.5f)
    {
        qInfo() << "ConsensusManager: Room dimensions changed significantly. Resetting graph.";
        reset_internal();
        // Re-initialize immediately with new params
        initializeFromRoom(room_model, robot_pose, robot_covariance, room_params);
        has_room_ = true;
        // Reset tracking so we treat this as a fresh start
        last_robot_pose_ = robot_pose;
        has_last_pose_ = true;
        return;
    }

    // Add new robot pose if moved enough
    if (last_room_state_ == RoomState::LOCALIZED and has_last_pose_ and initialized_)
    {
        float dx = robot_pose.x() - last_robot_pose_.x();
        float dy = robot_pose.y() - last_robot_pose_.y();
        float dtheta = robot_pose.z() - last_robot_pose_.z();

        // Normalize dtheta to [-π, π]
        while (dtheta > M_PI) dtheta -= 2 * M_PI;
        while (dtheta < -M_PI) dtheta += 2 * M_PI;

        float translation = std::sqrt(dx * dx + dy * dy);
        float rotation = std::abs(dtheta);

        if (translation > MIN_TRANSLATION_FOR_NEW_NODE || rotation > MIN_ROTATION_FOR_NEW_NODE)
        {

            // If the robot "moved" > 1 meter or rotated > 1.5 rad (~90 deg) in a single frame,
            // it is a tracking glitch. IGNORE IT.
            if (translation > 1.0f || rotation > 1.5f)
            {
                qWarning() << "ConsensusManager: REJECTING TELEPORT (Dist:" << translation
                           << "m, Rot:" << rotation << "rad)";
                return; // Do NOT add node, do NOT update last_pose_
            }

            // Compute odometry (relative motion from last pose to current)
            // odom = last_pose^(-1) * current_pose
            float cos_last = std::cos(last_robot_pose_.z());
            float sin_last = std::sin(last_robot_pose_.z());

            // Transform delta to last pose's frame
            float odom_x = cos_last * dx + sin_last * dy;
            float odom_y = -sin_last * dx + cos_last * dy;
            float odom_theta = dtheta;

            gtsam::Pose2 odometry(odom_x, odom_y, odom_theta);

            // Odometry uncertainty (scales with motion)
            Eigen::Vector3d odom_uncertainty(
                0.1 * translation + 0.02,  // x uncertainty
                0.1 * translation + 0.02,  // y uncertainty
                0.1 * rotation + 0.01      // theta uncertainty
            );

            // Current pose uncertainty from room detector
            Eigen::Vector3d pose_sigmas(
                std::sqrt(std::max(1e-6f, robot_covariance(0, 0))),
                std::sqrt(std::max(1e-6f, robot_covariance(1, 1))),
                std::sqrt(std::max(1e-6f, robot_covariance(2, 2)))
            );

            // In LOCALIZED mode, we want the DOORS to correct the pose.
            // If we trust the RoomModel perfectly, the doors can't move the robot.
            // Multiply the uncertainty by a factor (e.g., 2.0 or 5.0) to loosen the grip.
            if (last_room_state_ == RoomState::LOCALIZED) {
                pose_sigmas *= 5.0;
            }

            // Add new robot pose with odometry factor
            const gtsam::Pose2 new_pose(robot_pose.x(), robot_pose.y(), robot_pose.z());
            current_robot_pose_index_ = graph_.addRobotPose(
                new_pose,
                pose_sigmas,
                odometry,
                odom_uncertainty
            );

            // New Keyframe -> Reset Observation Tracker ===
            doors_observed_at_current_pose_.clear();

            graph_dirty_ = true;
            last_robot_pose_ = robot_pose;

            qInfo() << "ConsensusManager: Added robot pose" << current_robot_pose_index_
                    << "with odometry (" << odom_x << "," << odom_y << "," << odom_theta << ")";

            // Run optimization if we have doors
            if (!door_models_.empty())
            {
                lock.unlock();
                runOptimization();
            }
        }
    }

    // If we have doors, run optimization
    // if (initialized_ && !door_models_.empty())
    // {
    //     // Update robot pose in graph (could add odometry factor here)
    //     // For now, we just run optimization with current state
    //     // runOptimization();  // Uncomment if you want continuous updates
    // }
}

void ConsensusManager::onRoomStateChanged(RoomState new_state)
{
    QMutexLocker lock(&mutex_);
    last_room_state_ = new_state;
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

    if (!initialized_ || door_models_.empty()) return;

    // Assumes single door (index 0)
    // const size_t door_index = 0;
    // gtsam::Symbol door_sym = ConsensusGraph::ObjectSymbol(door_index);
    //
    // // Check if door exists in graph
    // if (!graph_.getValues().exists(door_sym)) return;
    //
    // // If we have already constrained this door to the current robot node,
    // // DON'T add another factor. It just inflates the error and slows down optimization.
    // if (doors_observed_at_current_pose_.count(door_index) > 0)
    //     return;
    //
    // // 1. Prepare Measurement (Room Frame)
    // gtsam::Pose2 measurement(door_pose.x(), door_pose.y(), door_pose.z());
    //
    // // 2. Get Robot Pose
    // gtsam::Pose2 robot_pose = graph_.getValues().at<gtsam::Pose2>(
    //     ConsensusGraph::RobotSymbol(current_robot_pose_index_));
    //
    // // 3. Calculate Relative Measurement: Robot -> Door
    // // This creates the constraint "I see the door at X meters relative to me"
    // gtsam::Pose2 relative_measurement = robot_pose.between(measurement);
    //
    // // 4. Extract Uncertainty
    // // Use the inflated covariance logic we discussed (minimum 5cm/3deg)
    // double min_pos_var = 0.05 * 0.05;
    // double min_ang_var = 0.05 * 0.05;
    //
    // // Extract sigmas (taking max of calculated vs minimum)
    // Eigen::Vector3d obs_sigmas(
    //     std::sqrt(std::max((double)door_covariance(0,0), min_pos_var)),
    //     std::sqrt(std::max((double)door_covariance(1,1), min_pos_var)),
    //     std::sqrt(std::max((double)door_covariance(2,2), min_ang_var))
    // );
    //
    // // 5. === FIX: ADD OBSERVATION TO EXISTING NODE ===
    // // Do NOT call addObjectWithObservation here
    // graph_.addObservation(
    //     current_robot_pose_index_,
    //     door_index,
    //     relative_measurement,
    //     obs_sigmas
    // );
    // doors_observed_at_current_pose_.insert(door_index);
    // graph_dirty_ = true;
    //
    // // Unlock and Optimize
    // lock.unlock();
    // runOptimization();
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
                float opening_angle = door_model->get_opening_angle();
                float smooth_angle = opening_angle;

                // Check if we have a history for this door
                if (smoothed_angles_.find(i) != smoothed_angles_.end())
                {
                    const float prev_angle = smoothed_angles_[i];
                    // Handle angle wrapping (-PI to PI) if necessary
                    // (Assuming opening angle is usually 0 to PI/2, simple math works)
                    // Exponential Moving Average: New = Alpha * Raw + (1 - Alpha) * Old
                    smooth_angle = ALPHA_SMOOTHING * opening_angle + (1.0f - ALPHA_SMOOTHING) * prev_angle;
                }
                // Update history
                smoothed_angles_[i] = smooth_angle;

                // Get z from original door model (consensus only handles 2D)
                //auto original_pose = door_model->get_door_pose();
                //float z = original_pose[2];  // z position
                const float height = geom[1]; // Full height
                const float z = 0.f; // Use half height as z position

                Q_EMIT doorPoseInRoom(i,
                    prior->first.x(),      // x from consensus
                    prior->first.y(),      // y from consensus
                    z,                     // z from door model
                    prior->first.z(),      // theta from consensus
                    geom[0],               // width from detector
                    height,                     // height from detector
                    smooth_angle          // opening angle from detector (relative to door frame)
                );

                // qDebug() << "ConsensusManager: Emitted door" << i << "in room frame:"
                //          << prior->first.x() << prior->first.y() << z << prior->first.z()
                //          << "opening:" << opening_angle;
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
    reset_internal();
}

void ConsensusManager::reset_internal()
{
    graph_ = ConsensusGraph();
    initialized_ = false;
    has_room_ = false;
    current_robot_pose_index_ = 0;
    room_model_.reset();
    door_models_.clear();
    room_half_width_ = 0.0;
    room_half_depth_ = 0.0;

    // IMPORTANT: Also clear the observation tracker
    doors_observed_at_current_pose_.clear();

    // Reset state trackers
    last_robot_pose_ = Eigen::Vector3f::Zero();
    has_last_pose_ = false;

    qInfo() << "ConsensusManager: Reset (Internal)";
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

// In consensus_manager.cpp

double ConsensusManager::computeWallOffset(WallID wall, const Eigen::Vector3f& door_pose) const
{
    // Transforms the global door coordinate into the wall's local "Along-Wall" (X) coordinate.
    // Must match the frame definitions in ConsensusGraph::computeWallPose

    switch (wall)
    {
        case WallID::NORTH:
            // Frame is rotated 180 deg (X points West).
            // Positive Global X is "Left" in local frame (Negative Local X).
            return -door_pose.x();

        case WallID::SOUTH:
            // Frame is rotated 0 deg (X points East).
            // Positive Global X is "Right" in local frame (Positive Local X).
            return door_pose.x();

        case WallID::EAST:
            // Frame is rotated 90 deg (X points North).
            // Positive Global Y is "Right/Forward" in local frame (Positive Local X).
            return door_pose.y();

        case WallID::WEST:
            // Frame is rotated -90 deg (X points South).
            // Positive Global Y is "Back/Left" in local frame (Negative Local X).
            return -door_pose.y();

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

        const auto opt_pose = graph_.getValues().at<gtsam::Pose2>(object_sym);
        const Eigen::Matrix3d cov = marginals.marginalCovariance(object_sym);

        Eigen::Vector3f pose(opt_pose.x(), opt_pose.y(), opt_pose.theta());
        Eigen::Matrix3f cov_f = cov.cast<float>();

        // Set high uncertainty on theta to avoid fighting with detector's measurements
        // The consensus manager's theta might differ from detector's due to orientation correction
        // Let the detector's measurements determine orientation, consensus refines position
        // cov_f(2, 2) = std::max(cov_f(2, 2), 1.0f);  // At least 1 radian^2 variance on theta
        // cov_f(0, 2) = 0.0f;
        // cov_f(1, 2) = 0.0f;
        // cov_f(2, 0) = 0.0f;
        // cov_f(2, 1) = 0.0f;

        return std::make_pair(pose, cov_f);
    }
    catch (const std::exception& e)
    {
        qWarning() << "ConsensusManager::getDoorConsensusPrior - Error:" << e.what();
        return std::nullopt;
    }
}

// consensus_manager.cpp

bool ConsensusManager::is_statistically_consistent(const gtsam::Pose2& belief,
                                                   const gtsam::Matrix3& belief_cov,
                                                   const gtsam::Pose2& measurement,
                                                   const gtsam::Matrix3& meas_cov,
                                                   double sigma_threshold)
{
    // 1. Calculate Error Vector (Innovation)
    // Use GTSAM's localCoordinates to handle angle wrapping correctly
    gtsam::Vector3 error = belief.localCoordinates(measurement);

    // 2. Compute Combined Covariance (Innovation Covariance S)
    // S = Sigma_belief + Sigma_measurement
    gtsam::Matrix3 S = belief_cov + meas_cov;

    // 3. Compute Mahalanobis Distance Squared (D^2)
    // D^2 = e^T * S^-1 * e
    // LLT decomposition is faster/stable for positive definite matrices
    double mahalanobis_sq;
    try {
        mahalanobis_sq = error.transpose() * S.llt().solve(error);
    } catch (...) {
        // Fallback for ill-conditioned matrices (rare if noise is properly set)
        mahalanobis_sq = error.transpose() * S.inverse() * error;
    }

    // 4. Chi-Square Test
    // Degrees of Freedom (DOF) = 3 (x, y, theta)
    // Threshold depends on sigma_threshold (standard deviations)
    // 3.0 sigma ~= 99.7% confidence -> Threshold approx 9.0 - 11.0 for 3DOF
    // Let's compute threshold squared.
    // Actually, usually we check against specific Chi2 values.
    // For 3 DOF: 95% = 7.81, 99% = 11.34, 99.9% = 16.27

    // Using simple sigma squared for scaling:
    double threshold = sigma_threshold * sigma_threshold;

    // Better: Hardcoded Chi-Square value for p=0.01 (99% confidence)
    double chi_square_99 = 11.345;

    if (mahalanobis_sq > chi_square_99) {
        qWarning() << "REJECTED OUTLIER: Mahalanobis =" << mahalanobis_sq
                   << ">" << chi_square_99;
        return false;
    }

    return true;
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