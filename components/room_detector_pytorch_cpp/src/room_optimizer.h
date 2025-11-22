//
// RoomOptimizer - EKF-style predict-update cycle for SLAM
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

class RoomOptimizer
{
    public:
        struct Result
        {
            torch::Tensor covariance;            // 3x3 or 5x5
            torch::Tensor propagated_cov;        // 3x3 or 5x5 (after motion, before measurement)
            std::vector<float> std_devs;         // flat std dev vector
            float final_loss = 0.0f;
            float prior_loss = 0.0f;             // For calibration learning
            bool uncertainty_valid = true;
            bool used_fusion = false;
            OdometryPrior prior;
            std::vector<float> optimized_pose;   // Final robot pose after optimization
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

        struct PredictionState
        {
            torch::Tensor propagated_cov;  // Predicted covariance
            bool have_propagated = false;   // Whether prediction was performed
            std::vector<float> previous_pose;  // Robot pose BEFORE prediction (for prior loss)
            std::vector<float> predicted_pose; // Robot pose after prediction
        };

        PredictionParameters prediction_params;

        RoomOptimizer() = default;

        /**
         * Main EKF cycle: predict-update
         * Runs adaptive optimization (MAPPING or LOCALIZED mode) and computes uncertainty
         */
        Result optimize( const TimePoints &points,
                         RoomModel &room,
                         const VelocityHistory &velocity_history,
                         std::shared_ptr<TimeSeriesPlotter> time_series_plotter = nullptr,
                         int num_iterations = 150,
                         float min_loss_threshold = 0.001f,
                         float learning_rate = 0.01f
        );

        // Public components
        RoomFreezingManager room_freezing_manager;
        UncertaintyManager uncertainty_manager;
        CalibrationConfig calib_config;                 // Exposed for tuning
        ModelBasedFilter model_based_filter;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_lidar_timestamp = {};
        unsigned long int frame_number = 0;

    private:
        // ===== EKF PREDICT PHASE =====
        /**
         * Predict step: propagate state and covariance using motion model
         */
        PredictionState predict_step(RoomModel &room,
                                     const OdometryPrior &odometry_prior,
                                     bool is_localized);

        /**
         * Filter measurements based on predicted state (top-down prediction)
         */
        ModelBasedFilter::Result filter_measurements(const RoboCompLidar3D::TPoints &points,
                                                      RoomModel &room,
                                                      const PredictionState &prediction);

        // ===== EKF UPDATE PHASE =====
        /**
         * Update step: optimize state using measurements
         */
        Result update_step(const torch::Tensor &points_tensor,
                          RoomModel &room,
                          const OdometryPrior &odometry_prior,
                          const PredictionState &prediction,
                          bool is_localized,
                          int num_iterations,
                          float min_loss_threshold,
                          float learning_rate,
                          std::shared_ptr<TimeSeriesPlotter> time_series_plotter);

        /**
         * Select parameters for optimization based on mode
         */
        std::vector<torch::Tensor> select_optimization_parameters(RoomModel &room, bool is_localized);

        /**
         * Run gradient descent optimization loop
         */
        struct OptimizationResult
        {
            float final_loss = 0.0f;
            float final_prior_loss = 0.0f;
        };

        OptimizationResult run_optimization_loop(const torch::Tensor &points_tensor,
                                   RoomModel &room,
                                   const torch::Tensor &predicted_pose_tensor,
                                   const OdometryPrior &odometry_prior,
                                   torch::optim::Optimizer &optimizer,
                                   bool use_odometry_prior,
                                   int num_iterations,
                                   float min_loss_threshold,
                                   std::shared_ptr<TimeSeriesPlotter> time_series_plotter);

        /**
         * Compute measurement loss (SDF-based)
         */
        torch::Tensor compute_measurement_loss(const torch::Tensor &points_tensor, RoomModel &room);

        /**
         * Compute odometry prior loss (Mahalanobis distance)
         */
        torch::Tensor compute_prior_loss(RoomModel &room,
                                         const torch::Tensor &predicted_pose,
                                         const OdometryPrior &odometry_prior);

        /**
         * Compute calibration regularization loss
         */
        torch::Tensor compute_calibration_regularization(RoomModel &room);

        /**
         * Estimate uncertainty after optimization
         * @param propagated_cov The covariance after motion propagation (for LOCALIZED mode)
         */
        Result estimate_uncertainty(RoomModel &room,
                                   const torch::Tensor &points_tensor,
                                   bool is_localized,
                                   float final_loss,
                                   const torch::Tensor &propagated_cov = torch::Tensor());

        /**
         * Update state management (freezing, history)
         */
        void update_state_management(const Result &res,
                                    RoomModel &room,
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
        Eigen::Vector3f integrate_velocity_over_window(const RoomModel& room,
                                                       const boost::circular_buffer<VelocityCommand> &velocity_history,
                                                       const std::chrono::time_point<std::chrono::high_resolution_clock> &t_start,
                                                       const std::chrono::time_point<std::chrono::high_resolution_clock> &t_end);

        /**
         * Compute odometry prior between LiDAR timestamps
         */
        OdometryPrior compute_odometry_prior(const RoomModel &room,
                                            const boost::circular_buffer<VelocityCommand>& velocity_history,
                                            const std::chrono::time_point<std::chrono::high_resolution_clock> &lidar_timestamp);

        /**
         * Compute motion-based covariance for odometry prior
         * Uses consistent formula: σ = base + k * distance
         */
        Eigen::Matrix3f compute_motion_covariance(const OdometryPrior &odometry_prior);

        /**
         * Update calibration parameters slowly using long-term learning
         */
        void update_calibration_slowly(RoomModel &room,
                                      const OdometryPrior &odometry_prior,
                                      const std::vector<float> &predicted_pose,
                                      const std::vector<float> &optimized_pose,
                                      float prior_loss);

        float wall_thickness = 0.05f;  // Wall thickness for loss computation

        // Adaptive prior weighting
        float prior_weight_ = 1.0f;  // Current weight for prior loss (updated each frame)

        // Long-term calibration learning state
        unsigned long int calibration_learning_frames = 0;  // Frames since learning started

};