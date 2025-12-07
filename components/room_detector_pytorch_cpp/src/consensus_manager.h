/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    ConsensusManager: Bridge between existing detectors and the GTSAM factor graph
 *    Now with Qt signals/slots for clean thread communication
 */

#ifndef CONSENSUS_MANAGER_H
#define CONSENSUS_MANAGER_H

#include <QObject>
#include <QMutex>

#include "consensus_graph.h"
#include "room_model.h"
#include "door_model.h"

#include <memory>
#include <optional>

/**
 * @brief Manages the consensus between room and door detectors using factor graph optimization
 *
 * This class provides a high-level interface using Qt signals/slots:
 *
 * Input slots:
 *   - onRoomUpdated: Receive room pose updates
 *   - onDoorDetected: First door detection
 *   - onDoorUpdated: Subsequent door updates
 *
 * Output signals:
 *   - doorPriorReady: Emitted after optimization with door prior for DoorThread
 *   - consensusReady: Emitted with full optimization results
 */
class ConsensusManager : public QObject
{
    Q_OBJECT

public:
    explicit ConsensusManager(QObject* parent = nullptr);

    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Check if any doors have been added
     */
    bool has_doors() const { return !door_models_.empty(); }

    /**
     * @brief Get the underlying factor graph (for debugging)
     */
    const ConsensusGraph& getGraph() const { return graph_; }

    /**
     * @brief Get current robot pose index
     */
    size_t getCurrentRobotPoseIndex() const { return current_robot_pose_index_; }

    /**
     * @brief Print status
     */
    void print() const;

Q_SIGNALS:
    /**
     * @brief Emitted when a door prior is ready after optimization
     * @param door_index Index of the door
     * @param pose Optimized door pose [x, y, theta]
     * @param covariance 3x3 covariance matrix
     */
    void doorPriorReady(size_t door_index, const Eigen::Vector3f& pose, const Eigen::Matrix3f& covariance);

    /**
     * @brief Emitted when consensus optimization completes
     * @param result Full optimization result
     */
    void consensusReady(const ConsensusResult& result);

    /**
     * @brief Emitted with door pose in room coordinates for visualization
     * @param door_index Index of the door
     * @param x, y, z, theta, width, height, opening_angle - door parameters in room frame
     */
    void doorPoseInRoom(size_t door_index, float x, float y, float z, float theta,
                        float width, float height, float opening_angle);

    /**
     * @brief Emitted when initialized
     */
    void initialized();

public Q_SLOTS:
    /**
     * @brief Initialize from room detection
     *
     * @param room_model Current room model from RoomOptimizer
     * @param robot_pose Robot's current pose in room frame [x, y, theta]
     * @param robot_covariance 3x3 covariance matrix for robot pose
     * @param room_params Room parameters [half_width, half_depth]
     */
    void onRoomUpdated(const std::shared_ptr<RoomModel>& room_model,
                       const Eigen::Vector3f& robot_pose,
                       const Eigen::Matrix3f& robot_covariance,
                       const std::vector<float>& room_params);

    /**
     * @brief Handle first door detection
     *
     * @param door_model Door model from DoorConcept
     * @param detection_covariance 3x3 covariance matrix for door pose
     */
    void onDoorDetected(const std::shared_ptr<DoorModel>& door_model,
                        const Eigen::Matrix3f& detection_covariance);

    /**
     * @brief Handle door tracking update
     *
     * Runs optimization and emits doorPriorReady
     *
     * @param door_pose Current door pose [x, y, theta]
     * @param door_covariance 3x3 covariance matrix
     */
    void onDoorUpdated(const Eigen::Vector3f& door_pose,
                       const Eigen::Matrix3f& door_covariance);

    /**
     * @brief Handle door tracking lost
     */
    void onDoorTrackingLost();

    /**
     * @brief Force optimization run
     */
    void runOptimization();

    /**
     * @brief Reset the consensus manager
     */
    void reset();

private:
    /**
     * @brief Initialize the factor graph from room
     */
    void initializeFromRoom(const std::shared_ptr<RoomModel>& room_model,
                            const Eigen::Vector3f& robot_pose_in_room,
                            const Eigen::Matrix3f& robot_pose_covariance,
                            const std::vector<float>& room_params);

    /**
     * @brief Add a door to the factor graph
     */
    size_t addDoor(const std::shared_ptr<DoorModel>& door_model,
                   const Eigen::Matrix3f& detection_covariance);

    /**
     * @brief Run optimization and emit results
     */
    ConsensusResult optimize();

    /**
     * @brief Get consensus prior for a door
     */
    std::optional<std::pair<Eigen::Vector3f, Eigen::Matrix3f>> getDoorConsensusPrior(size_t door_index) const;

    /**
     * @brief Determine which wall a door is attached to
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

    // Thread safety
    mutable QMutex mutex_;

    // Factor graph
    ConsensusGraph graph_;

    // State
    bool initialized_ = false;
    bool has_room_ = false;
    size_t current_robot_pose_index_ = 0;

    // Model references
    std::shared_ptr<RoomModel> room_model_;
    std::vector<std::shared_ptr<DoorModel>> door_models_;

    // Room dimensions
    double room_half_width_ = 0.0;
    double room_half_depth_ = 0.0;

    // Cached room data for initialization
    Eigen::Vector3f cached_robot_pose_;
    Eigen::Matrix3f cached_robot_covariance_;
    std::vector<float> cached_room_params_;

    // Optimization state
    bool graph_dirty_ = true;  // True when graph changed since last optimization
    ConsensusResult last_result_;  // Cached result from last optimization
};

#endif // CONSENSUS_MANAGER_H