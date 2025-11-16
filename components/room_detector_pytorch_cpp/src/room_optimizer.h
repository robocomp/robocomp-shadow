//
// Created by pbustos on 16/11/25.
//

#pragma once
#include <vector>
#include <torch/torch.h>
#include <Lidar3D.h>
#include "room_model.h"
#include "time_series_plotter.h"
#include "room_freezing_manager.h"

class RoomOptimizer
{
    public:
        struct Result
        {
            torch::Tensor covariance;            // 3x3 or 5x5
            std::vector<float> std_devs;         // flat std dev vector
            float final_loss = 0.0f;
        };

        RoomOptimizer() = default;

        // Ejecuta la optimización y devuelve resultados de incertidumbre.
        // time_series_plotter puede ser nullptr si no se desea trazado.
        Result optimize(const RoboCompLidar3D::TPoints &points,
                        RoomModel &room,
                        std::shared_ptr<TimeSeriesPlotter> time_series_plotter = nullptr,
                        int num_iterations = 150,
                        float min_loss_threshold = 0.01f,
                        float learning_rate = 0.01f);

        // Room freezing manager
        RoomFreezingManager room_freezing_manager;
};
