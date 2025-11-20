//
// RoomOptimizer - EKF-style predict-update cycle for SLAM
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
            std::vector<float> std_devs;         // flat std dev vector
            float final_loss = 0.0f;
            bool uncertainty_valid = true;
            bool used_fusion = false;
            OdometryPrior prior;
        };

        struct CalibrationConfig
        {
            float regularization_weight = 0.1f;  // Weight for keeping calibration near 1.0
            float min_value = 0.8f;              // Minimum calibration value (80%)
            float max_value = 1.2f;              // Maximum calibration value (120%)
            float uncertainty_inflation = 100.0f; // Inflate covariance (overconfidence correction)
        };

        struct PredictionParameters
        {
           float NOISE_TRANS = 0.02f;  // 2cm stddev per meter
           float NOISE_ROT = 0.1f;     // 0.1 rad
        };

        struct PredictionState
        {
            torch::Tensor propagated_cov;  // Predicted covariance
            bool have_propagated = false;   // Whether prediction was performed
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
        float run_optimization_loop(const torch::Tensor &points_tensor,
                                   RoomModel &room,
                                   const torch::Tensor &predicted_pose,
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
         */
        Result estimate_uncertainty(RoomModel &room,
                                   const torch::Tensor &points_tensor,
                                   bool is_localized,
                                   float final_loss);

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

        float wall_thickness = 0.05f;  // Wall thickness for loss computation
};