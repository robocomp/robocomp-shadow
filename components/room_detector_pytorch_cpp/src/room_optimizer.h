//
// Refactored RoomOptimizer - minimal version matching your existing structure
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
        PredictionParameters prediction_params;

        RoomOptimizer() = default;

        /**
         * Main optimization function
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
        float wall_thickness = 0.05f;  // Wall thickness for loss computation
        ModelBasedFilter::Result top_down_prediction(const RoboCompLidar3D::TPoints &points,
                                                     RoomModel &room,
                                                     bool have_propagated,
                                                     const torch::Tensor &
                                                     propagated_cov);
        Eigen::Vector3f integrate_velocity_over_window( const RoomModel& room,
                                                        const boost::circular_buffer<VelocityCommand> &velocity_history,
                                                        const std::chrono::time_point<std::chrono::high_resolution_clock> &t_start,
                                                        const std::chrono::time_point<std::chrono::high_resolution_clock> &t_end);
        OdometryPrior compute_odometry_prior(const RoomModel &room,
                                             const boost::circular_buffer<VelocityCommand>& velocity_history,
                                             const std::chrono::time_point<std::chrono::high_resolution_clock> &lidar_timestamp);
};