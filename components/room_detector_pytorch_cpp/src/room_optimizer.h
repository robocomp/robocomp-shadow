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

class RoomOptimizer
{
    public:
        struct Result
        {
            torch::Tensor covariance;            // 3x3 or 5x5
            std::vector<float> std_devs;         // flat std dev vector
            float final_loss = 0.0f;
            bool uncertainty_valid = true;
        };

        RoomOptimizer() = default;

        /**
         * Main optimization function
         * Runs adaptive optimization (MAPPING or LOCALIZED mode) and computes uncertainty
         */
        Result optimize(const RoboCompLidar3D::TPoints& points,
                        RoomModel& room,
                        std::shared_ptr<TimeSeriesPlotter> time_series_plotter = nullptr,
                        int num_iterations = 150,
                        float min_loss_threshold = 0.001f,
                        float learning_rate = 0.01f,
                        const OdometryPrior& odometry_prior = {},
                        int frame_number = 0
        );

        // Public components
        RoomFreezingManager room_freezing_manager;
        UncertaintyManager uncertainty_manager;

    private:
        float wall_thickness = 0.05f;  // Wall thickness for loss computation
};