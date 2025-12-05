/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    ConsensusManager: Bridge between existing detectors and the GTSAM factor graph
 */

#ifndef CONSENSUS_MANAGER_H
#define CONSENSUS_MANAGER_H

#include "consensus_graph.h"
#include "room_model.h"
#include "door_model.h"

#include <memory>
#include <optional>

/**
 * @brief Manages the consensus between room and door detectors using factor graph optimization
 * 
 * This class provides a high-level interface to:
 * 1. Initialize the factor graph from room detector
 * 2. Add doors detected by the door detector
 * 3. Update robot pose estimates
 * 4. Run optimization and propagate results back to detectors
 */
class ConsensusManager
{
public:
    ConsensusManager();

    /**
     * @brief Initialize from room detection
     * 
     * Takes the current room estimate and initializes the factor graph
     * with room at origin and walls as rigid children.
     * 
     * @param room_model Current room model from RoomOptimizer
     * @param robot_pose_in_room Robot's current pose in room frame
     * @param robot_pose_uncertainty Uncertainty (σ for x, y, θ)
     */
    void initializeFromRoom(const std::shared_ptr<RoomModel>& room_model,
                            const Eigen::Vector3f& robot_pose_in_room,
                            const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &robot_pose_uncertainty);

    /**
     * @brief Add a door detection to the graph
     * 
     * Automatically determines which wall the door is attached to
     * based on the door's position and orientation.
     * 
     * @param door_model Door model from DoorConcept
     * @param detection_uncertainty Detection uncertainty (σ for x, y, θ)
     * @return size_t Index of the added door in the graph
     */
    size_t addDoor(const std::shared_ptr<DoorModel>& door_model,
                   const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &detection_uncertainty);

    /**
     * @brief Update robot pose (e.g., from odometry)
     * 
     * @param new_pose New robot pose in room frame
     * @param pose_uncertainty Pose uncertainty
     * @param odometry Delta pose from previous (if available)
     * @param odom_uncertainty Odometry uncertainty
     * @return size_t Index of the new robot pose
     */
    size_t updateRobotPose(const Eigen::Vector3f& new_pose,
                           const Eigen::Vector3f& pose_uncertainty,
                           const std::optional<Eigen::Vector3f>& odometry = std::nullopt,
                           const Eigen::Vector3f& odom_uncertainty = Eigen::Vector3f(0.05f, 0.05f, 0.02f));

    /**
     * @brief Add an observation of a door from current robot pose
     * 
     * @param door_index Index of the door being observed
     * @param observed_pose Door pose as observed (in robot frame)
     * @param observation_uncertainty Observation uncertainty
     */
    void addDoorObservation(size_t door_index,
                            const Eigen::Vector3f& observed_pose,
                            const Eigen::Vector3f& observation_uncertainty);

    /**
     * @brief Run consensus optimization
     * 
     * @return ConsensusResult Optimized poses and covariances
     */
    ConsensusResult optimize();

    /**
     * @brief Apply consensus results back to the models
     * 
     * Updates the RoomModel and DoorModels with optimized poses.
     * 
     * @param result Optimization result
     */
    void applyResults(const ConsensusResult& result);

    /**
     * @brief Get the underlying factor graph (for debugging)
     */
    const ConsensusGraph& getGraph() const { return graph_; }

    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get current robot pose index
     */
    size_t getCurrentRobotPoseIndex() const { return current_robot_pose_index_; }

    bool has_doors() const { return !door_models_.empty(); }

    /**
     * @brief Print status
     */
    void print() const;

private:
    /**
     * @brief Determine which wall a door is attached to based on its pose
     * 
     * Uses the door's position and orientation to find the nearest wall.
     */
    WallID determineWallAttachment(const Eigen::Vector3f& door_pose) const;

    /**
     * @brief Compute offset along wall for a door
     */
    double computeWallOffset(WallID wall, const Eigen::Vector3f& door_pose) const;

    /**
     * @brief Convert Eigen::Vector3f pose to gtsam::Pose2
     */
    static gtsam::Pose2 toPose2(const Eigen::Vector3f& pose);

    /**
     * @brief Convert gtsam::Pose2 to Eigen::Vector3f
     */
    static Eigen::Vector3f fromPose2(const gtsam::Pose2& pose);

    ConsensusGraph graph_;
    
    bool initialized_ = false;
    size_t current_robot_pose_index_ = 0;
    
    // Store references to models for applying results
    std::shared_ptr<RoomModel> room_model_;
    std::vector<std::shared_ptr<DoorModel>> door_models_;
    
    // Room dimensions (needed for wall attachment computation)
    double room_half_width_ = 0.0;
    double room_half_depth_ = 0.0;
};

#endif // CONSENSUS_MANAGER_H
