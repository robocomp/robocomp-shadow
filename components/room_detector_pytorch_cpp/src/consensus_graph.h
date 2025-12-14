/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    Consensus Graph using GTSAM for combining room and door detections
 *
 *    Factor Graph Structure (Scenario 3: Room as Origin with Walls):
 *
 *         Room (fixed origin)
 *           |
 *      φ_prior (anchors room at origin)
 *           |
 *     ┌─────┼─────┬─────┐
 *     │     │     │     │
 *   φ_rigid_N  φ_rigid_S  φ_rigid_E  φ_rigid_W
 *     │     │     │     │
 *    W_N   W_S   W_E   W_W  (walls as rigid children)
 *     │
 *   φ_obj (object attached to wall)
 *     │
 *    L_0 (door/landmark)
 *     │
 *   φ_obj (observation from robot)
 *     │
 *    r_i (robot poses)
 *     │
 *   φ_odom (odometry between poses)
 *     │
 *    r_{i+1}
 *
 *    Variables:
 *      - Room: Fixed at origin (gauge freedom elimination)
 *      - W_N, W_S, W_E, W_W: Wall poses (derived from room via rigid constraints)
 *      - r_0, r_1, ..., r_n: Robot poses over time
 *      - L_0, L_1, ...: Object/landmark poses (doors, etc.)
 *
 *    Factors:
 *      - φ_prior: Room fixed at origin
 *      - φ_rigid_X: Rigid body constraints (walls relative to room)
 *      - φ_room_obs: Robot observations of room geometry
 *      - φ_obj: Robot observations of objects/landmarks
 *      - φ_odom: Odometry between consecutive robot poses
 */

#ifndef CONSENSUS_GRAPH_H
#define CONSENSUS_GRAPH_H

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>
#include <optional>

// Forward declarations
class RoomModel;
class DoorModel;

/**
 * @brief Wall identifier enum for the four walls of a rectangular room
 */
enum class WallID { NORTH, SOUTH, EAST, WEST };

/**
 * @brief Result structure containing optimized poses and covariances
 */
struct ConsensusResult
{
    // Room pose (fixed at origin in this formulation)
    gtsam::Pose2 room_pose;
    Eigen::Matrix3d room_covariance;

    // Wall poses (derived from room)
    std::map<WallID, gtsam::Pose2> wall_poses;
    std::map<WallID, Eigen::Matrix3d> wall_covariances;

    // Robot poses over time
    std::vector<gtsam::Pose2> robot_poses;
    std::vector<Eigen::Matrix3d> robot_covariances;

    // Object/landmark poses (doors, etc.)
    std::map<size_t, gtsam::Pose2> object_poses;
    std::map<size_t, Eigen::Matrix3d> object_covariances;

    // Optimization statistics
    double initial_error;
    double final_error;
    int iterations;
    bool converged;

    void print() const
    {
        auto printPose = [](const gtsam::Pose2& pose) {
            std::cout << "(" << pose.x() << ", " << pose.y() << ", " << pose.theta() << ")";
        };

        auto printCov = [](const Eigen::Matrix3d& cov) {
            std::cout << "[[" << cov(0,0) << ", " << cov(0,1) << ", " << cov(0,2) << "], "
                      << cov(1,0) << ", " << cov(1,1) << ", " << cov(1,2) << "], "
                      << cov(2,0) << ", " << cov(2,1) << ", " << cov(2,2) << "]]";
        };

        std::cout << "ConsensusResult\n";
        std::cout << "  Room pose: ";
        printPose(room_pose);
        std::cout << "\n  Room covariance: ";
        printCov(room_covariance);

        for (const auto& [wall, pose] : wall_poses) {
            std::cout << "\n  Wall " << static_cast<int>(wall) << " pose: ";
            printPose(pose);
            std::cout << "\n    Covariance: ";
            printCov(wall_covariances.at(wall));
        }

        for (size_t i = 0; i < robot_poses.size(); ++i) {
            std::cout << "\n  Robot pose[" << i << "]: ";
            printPose(robot_poses[i]);
            if (i < robot_covariances.size()) {
                std::cout << "\n    Covariance: ";
                printCov(robot_covariances[i]);
            }
        }

        for (const auto& [idx, pose] : object_poses) {
            std::cout << "\n  Object[" << idx << "] pose: ";
            printPose(pose);
            std::cout << "\n    Covariance: ";
            printCov(object_covariances.at(idx));
        }

        std::cout << "\n  Initial error: " << initial_error
                  << "\n  Final error: " << final_error
                  << "\n  Iterations: " << iterations
                  << "\n  Converged: " << std::boolalpha << converged << "\n";
    }

};

/**
 * @brief GTSAM-based factor graph for spatial consensus between room and object detectors
 *
 * This class implements the factor graph shown in the schema:
 * - Room is fixed at origin (eliminates gauge freedom)
 * - Walls are rigidly attached to room
 * - Objects (doors) are attached to walls with geometric constraints
 * - Robot poses are connected via odometry
 * - Observations create loop closures that improve estimates
 */
class ConsensusGraph
{
public:
    using Symbol = gtsam::Symbol;
    using Pose2 = gtsam::Pose2;
    using SharedNoiseModel = gtsam::SharedNoiseModel;

    /**
     * @brief Construct a new Consensus Graph
     */
    ConsensusGraph();

    /**
     * @brief Initialize the room at origin with given dimensions
     *
     * Creates:
     * - Room node fixed at origin
     * - Four wall nodes with rigid constraints
     *
     * @param half_width Room half-width (X direction)
     * @param half_depth Room half-depth (Y direction)
     * @param room_uncertainty Uncertainty in room dimensions (σ for x, y, θ)
     */
    void initializeRoom(double half_width,
                        double half_depth,
                        const Eigen::Vector3d& room_uncertainty = Eigen::Vector3d(0.1, 0.1, 0.05));

    /**
     * @brief Add a robot pose to the graph
     *
     * @param pose Robot pose in room frame
     * @param pose_uncertainty Uncertainty (σ for x, y, θ)
     * @param odometry_from_previous Odometry from previous pose (if not first pose)
     * @param odometry_uncertainty Odometry uncertainty
     * @return size_t Index of the added robot pose
     */
    size_t addRobotPose(const Pose2& pose,
                        const Eigen::Vector3d& pose_uncertainty,
                        const std::optional<Pose2>& odometry_from_previous = std::nullopt,
                        const Eigen::Vector3d& odometry_uncertainty = Eigen::Vector3d(0.05, 0.05, 0.02));

    /**
     * @brief Add room observation factor from a robot pose
     *
     * Creates a factor connecting robot pose to room based on
     * LiDAR room detection measurements.
     *
     * @param robot_index Index of the observing robot pose
     * @param observed_room_pose Room pose as observed from robot
     * @param observation_uncertainty Measurement uncertainty
     */
    void addRoomObservation(size_t robot_index,
                            const Pose2& observed_room_pose,
                            const Eigen::Vector3d& observation_uncertainty);

    /**
     * @brief Add an object (door/landmark) to the graph
     *
     * NOTE: This method does NOT add a prior factor. The object is constrained by:
     * - Wall attachment factor (geometric constraint)
     * - Robot observation factors (added separately)
     *
     * @param object_pose Initial pose estimate in room frame
     * @param attached_wall Which wall the object is attached to (for geometric constraint)
     * @param wall_offset Offset along the wall (in wall's local X direction)
     * @param pose_uncertainty Initial pose uncertainty (used for wall constraint)
     * @return size_t Index of the added object
     */
    size_t addObject(const Pose2& object_pose,
                     WallID attached_wall,
                     double wall_offset,
                     const Eigen::Vector3d& pose_uncertainty);

    /**
     * @brief Add an object with initial robot observation
     *
     * Combines addObject + addObjectObservation in one call.
     * The object is constrained by:
     * - Wall attachment factor (geometric constraint: door ON wall)
     * - Robot observation factor (measurement: robot saw door at relative pose)
     *
     * @param robot_index Index of the robot pose that observed this object
     * @param object_pose_in_room Object pose in room frame (for initial value)
     * @param object_pose_in_robot Object pose as observed from robot (relative pose)
     * @param attached_wall Which wall the object is attached to
     * @param wall_offset Offset along the wall
     * @param observation_uncertainty Measurement uncertainty
     * @return size_t Index of the added object
     */
    size_t addObjectWithObservation(size_t robot_index,
                                    const Pose2& object_pose_in_room,
                                    const Pose2& object_pose_in_robot,
                                    WallID attached_wall,
                                    double wall_offset,
                                    const Eigen::Vector3d& observation_uncertainty);

    /**
     * @brief Add object observation factor from a robot pose
     *
     * Creates a factor connecting robot pose to object based on
     * detection measurements (e.g., YOLO + LiDAR for doors).
     *
     * @param robot_index Index of the observing robot pose
     * @param object_index Index of the observed object
     * @param observed_pose Object pose as observed from robot
     * @param observation_uncertainty Measurement uncertainty
     */
    void addObjectObservation(size_t robot_index,
                              size_t object_index,
                              const Pose2& observed_pose,
                              const Eigen::Vector3d& observation_uncertainty);

    /**
     * @brief Add geometric constraint: object attached to wall
     *
     * Creates a BetweenFactor constraining the object to lie on the specified wall.
     *
     * @param object_index Index of the object
     * @param wall Wall the object is attached to
     * @param offset_along_wall Position along wall (wall's local X)
     * @param door_theta_in_room Door's theta in room frame (to compute relative theta)
     * @param constraint_uncertainty How tight the constraint is
     */
    void addWallAttachmentConstraint(size_t object_index,
                                     WallID wall,
                                     double offset_along_wall,
                                     double door_theta_in_room,
                                     const Eigen::Vector3d& constraint_uncertainty = Eigen::Vector3d(0.05, 0.0001, 0.00001));

    /**
     * @brief Run the optimization
     *
     * @param max_iterations Maximum LM iterations
     * @param convergence_threshold Relative error decrease threshold
     * @return ConsensusResult Optimized poses and covariances
     */
    ConsensusResult optimize(int max_iterations = 100,
                             double convergence_threshold = 1e-5);

    /**
     * @brief Add an observation factor to an EXISTING object
     * * @param robot_idx Index of the robot pose node
     * @param object_idx Index of the existing object node
     * @param measurement Relative pose (Robot -> Object)
     * @param sigmas Measurement uncertainty [x, y, theta]
     */
    void addObservation(size_t robot_idx,
                        size_t object_idx,
                        const gtsam::Pose2& measurement,
                        const Eigen::Vector3d& sigmas);

    /**
         * @brief Add a global prior to a specific robot node (anchors it to the room)
         */
    void addRobotPrior(size_t robot_idx, const gtsam::Pose2& pose, const Eigen::Vector3d& sigmas);

    /**
     * @brief Get the current factor graph (for debugging/visualization)
     */
    const gtsam::NonlinearFactorGraph& getGraph() const { return graph_; }

    /**
     * @brief Get current values (for debugging)
     */
    const gtsam::Values& getValues() const { return values_; }

    /**
     * @brief Clear the graph and reset
     */
    void clear();

    /**
     * @brief Print graph structure for debugging
     */
    void print(const std::string& prefix = "") const;

    // Symbol generators for consistent key naming
    static Symbol RoomSymbol() { return Symbol('R', 0); }
    static Symbol WallSymbol(WallID wall) { return Symbol('W', static_cast<size_t>(wall)); }
    static Symbol RobotSymbol(size_t index) { return Symbol('r', index); }
    static Symbol ObjectSymbol(size_t index) { return Symbol('L', index); }

private:
    /**
     * @brief Create noise model from uncertainty vector (σ values)
     */
    SharedNoiseModel createNoiseModel(const Eigen::Vector3d& sigmas) const;

    /**
     * @brief Compute wall pose relative to room center
     *
     * @param wall Which wall
     * @param half_width Room half-width
     * @param half_depth Room half-depth
     * @return Pose2 Wall pose in room frame
     */
    Pose2 computeWallPose(WallID wall, double half_width, double half_depth) const;

    /**
     * @brief Add rigid constraint between room and wall
     */
    void addRigidWallConstraint(WallID wall, const Pose2& relative_pose);

    // GTSAM components
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;

    // Graph state
    bool room_initialized_ = false;
    double room_half_width_ = 0.0;
    double room_half_depth_ = 0.0;

    size_t robot_pose_count_ = 0;
    size_t object_count_ = 0;

    // Stored wall poses for constraint computation
    std::map<WallID, Pose2> wall_relative_poses_;
};

#endif // CONSENSUS_GRAPH_H