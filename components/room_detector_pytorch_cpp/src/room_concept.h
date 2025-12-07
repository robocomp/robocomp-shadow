//
// RoomConcept - EKF-style predict-update cycle for SLAM
//
// OPTIMIZATION-BASED EKF APPROACH:
// ================================
// Unlike standard EKF which uses closed-form Kalman gain, this implementation
// uses iterative optimization to fuse odometry prior with LiDAR measurements.
//
// Standard EKF:
//   x_new = x_pred + K * (z - h(x_pred))     // Closed-form update
//   K = P * H^T * (H*P*H^T + R)^-1           // Kalman gain
//
// Our approach:
//   x_new = argmin { ||z - h(x)||² + ||x - x_pred||²_P }  // Optimization-based
//   - Measurement term: SDF-based LiDAR fit
//   - Prior term: Mahalanobis distance from prediction
//   - Balance between terms acts as implicit Kalman gain
//
// Advantages:
//   - Handles non-linear measurement function (SDF) naturally
//   - Can optimize calibration parameters alongside state
//   - More flexible than linearization (EKF) or sampling (particle filter)
//

#pragma once
#include <vector>
#include <torch/torch.h>
#include <Lidar3D.h>
#include "room_model.h"
#include "time_series_plotter.h"
#include "room_freezing_manager.h"
#include "uncertainty_manager.h"
#include "model_based_filter.h"

namespace rc
{
    class RoomConcept
    {
        public:
            struct OdometryPrior
            {
                bool valid = false;
                Eigen::Vector3f delta_pose;      // [dx, dy, dtheta] in meters & radians
                torch::Tensor covariance;        // 3x3 covariance matrix
                VelocityCommand velocity_cmd;    // ADD: The actual velocity command
                float dt;                        // ADD: Time delta
                float prior_weight = 1.0f;      // How much to trust this prior

                OdometryPrior()
                    : delta_pose(Eigen::Vector3f::Zero())
                    , covariance(torch::zeros({3,3}, torch::kFloat32))
                    , dt(0.0f)
                {}
            };

            struct PredictionState
            {
                torch::Tensor propagated_cov;  // Predicted covariance
                bool have_propagated = false;   // Whether prediction was performed
                std::vector<float> previous_pose;  // Robot pose BEFORE prediction (for prior loss)
                std::vector<float> predicted_pose; // Robot pose after prediction
            };

            struct Result
            {
                torch::Tensor covariance;            // 3x3 or 5x5
                torch::Tensor propagated_cov;        // 3x3 or 5x5 (after motion, before measurement)
                std::vector<float> std_devs;         // flat std dev vector
                float final_loss = 0.0f;             // Total loss after optimization
                float prior_loss = 0.0f;             // For calibration learning
                float measurement_loss = 0.0f;       // Final measurement loss
                bool uncertainty_valid = true;       // Whether covariance is valid
                bool used_fusion = false;            // Whether odometry prior was used
                OdometryPrior prior;                 // Odometry prior used
                PredictionState prediction_state;    // For diagnostics
                std::vector<float> optimized_pose;   // Final robot pose after optimization
                float innovation_norm = 0.0f;        // For diagnostics
                float motion_magnitude = 0.0f;       // For diagnostics
            };

            struct CalibrationConfig
            {
                float regularization_weight = 0.001f;  // Reduced from 0.1 - was too strong!
                float min_value = 0.1f;               // Expanded: allow 10%-300% (for units issues)
                float max_value = 3.0f;
                float uncertainty_inflation = 100.0f; // Inflate covariance (overconfidence correction)
                bool enable_odometry_optimization = true;  // Enable/disable odometry-based optimization
                bool enable_calibration_learning = true;   // Enable/disable learning k_trans, k_rot (requires odometry_optimization)

                // Long-term learning parameters
                float calibration_learning_rate = 0.001f;  // EMA weight for slow updates (0.1% per frame)
                float min_prior_loss_for_learning = 0.01f; // Only learn when prior loss > threshold
                int min_frames_before_learning = 50;        // Wait this many frames before starting to learn
                float max_calibration_change_per_frame = 0.001f; // Max 0.1% change per frame
            };

            struct PredictionParameters
            {
                float NOISE_TRANS = 0.02f;  // 2cm stddev per meter
                float NOISE_ROT = 0.01f;     // 0.1 rad
            };

            PredictionParameters prediction_params;

            RoomConcept() = default;

            /**
             * Main EKF cycle: predict-update
             * Runs adaptive optimization (MAPPING or LOCALIZED mode) and computes uncertainty
             */
            Result update( const TimePoints &points,
                           const VelocityHistory &velocity_history,
                           int num_iterations = 150,
                           float min_loss_threshold = 0.01f,
                           float learning_rate = 0.01f
            );

            [[nodiscard]] std::shared_ptr<RoomModel> get_room_model() const { return room; }

            // Public components
            RoomFreezingManager room_freezing_manager;
            UncertaintyManager uncertainty_manager;
            CalibrationConfig calib_config;                 // Exposed for tuning
            ModelBasedFilter model_based_filter;
            std::chrono::time_point<std::chrono::high_resolution_clock> last_lidar_timestamp = {};
            unsigned long int frame_number = 0;

        private:
            std::shared_ptr<RoomModel> room;
            float prior_weight = 1.0f; // Adaptive prior weight

            // ===== EKF PREDICT PHASE =====
            /**
             * Predict step: propagate state and covariance using motion model
             */
            PredictionState predict_step(std::shared_ptr<RoomModel> &room,
                                         const OdometryPrior &odometry_prior,
                                         bool is_localized);

            /**
             * Filter measurements based on predicted state (top-down prediction)
             */
            ModelBasedFilter::Result filter_measurements(const RoboCompLidar3D::TPoints &points,
                                                          std::shared_ptr<RoomModel> &room,
                                                          const PredictionState &prediction);

            // ===== EKF UPDATE PHASE =====
            /**
             * Update step: optimize state using measurements
             */
            Result update_step(const torch::Tensor &points_tensor,
                              std::shared_ptr<RoomModel> &room,
                              const OdometryPrior &odometry_prior,
                              const PredictionState &prediction,
                              bool is_localized,
                              int num_iterations,
                              float min_loss_threshold,
                              float learning_rate);

            /**
             * Select parameters for optimization based on mode
             */
            std::vector<torch::Tensor> select_optimization_parameters(std::shared_ptr<RoomModel> &room, bool is_localized);

            /**
             * Run gradient descent optimization loop
             */
            struct OptimizationResult
            {
                float total_loss = 0.0f;
                float prior_loss = 0.0f;
                float measurement_loss = 0.0f;
            };

            OptimizationResult run_optimization_loop(const torch::Tensor &points_tensor,
                                       std::shared_ptr<RoomModel> &room,
                                       const torch::Tensor &predicted_pose_tensor,
                                       const OdometryPrior &odometry_prior,
                                       torch::optim::Optimizer &optimizer,
                                       const PredictionState &prediction,
                                       bool use_odometry_prior,
                                       int num_iterations,
                                       float min_loss_threshold);

            /**
             * Compute odometry prior loss (Mahalanobis distance)
             */
            torch::Tensor compute_prior_loss(std::shared_ptr<RoomModel> &room,
                                             const torch::Tensor &predicted_pose,
                                             const OdometryPrior &odometry_prior,
                                             const PredictionState &prediction);

            /**
             * Compute calibration regularization loss
             */
            torch::Tensor compute_calibration_regularization(std::shared_ptr<RoomModel> &room);

            /**
             * Estimate uncertainty after optimization
             * @param propagated_cov The covariance after motion propagation (for LOCALIZED mode)
             */
            Result estimate_uncertainty(std::shared_ptr<RoomModel> &room,
                                       const torch::Tensor &points_tensor,
                                       bool is_localized,
                                       float final_loss,
                                       const torch::Tensor &propagated_cov = torch::Tensor());

            /**
             * Update state management (freezing, history)
             */
            void update_state_management(const Result &res,
                                        std::shared_ptr<RoomModel> &room,
                                        bool is_localized,
                                        float final_loss);

            // ===== HELPER METHODS =====
            /**
             * Convert point cloud to tensor
             */
            torch::Tensor convert_points_to_tensor(const RoboCompLidar3D::TPoints &points);

            /**
             * Integrate velocity commands over time window
             */
            Eigen::Vector3f integrate_velocity_over_window(const std::shared_ptr<RoomModel> &room,
                                                           const boost::circular_buffer<VelocityCommand> &velocity_history,
                                                           const std::chrono::time_point<std::chrono::high_resolution_clock> &t_start,
                                                           const std::chrono::time_point<std::chrono::high_resolution_clock> &t_end);

            /**
             * Compute odometry prior between LiDAR timestamps
             */
            OdometryPrior compute_odometry_prior(const std::shared_ptr<RoomModel> &room,
                                                 const boost::circular_buffer<VelocityCommand>& velocity_history,
                                                 const std::chrono::time_point<std::chrono::high_resolution_clock> &lidar_timestamp);

            /**
             * Compute motion-based covariance for odometry prior
             * Uses consistent formula: σ = base + k * distance
             */
            Eigen::Matrix3f compute_motion_covariance(const OdometryPrior &odometry_prior);

            float compute_adaptive_prior_weight_simple(float innovation_norm, float motion_magnitude, float current_weight);
            float compute_adaptive_prior_weight(float innovation_norm,
                                                float motion_magnitude,
                                                const Eigen::Matrix3f& prior_cov,
                                                float current_weight);

            float wall_thickness = 0.1f;  // Wall thickness for loss computation
        };
};