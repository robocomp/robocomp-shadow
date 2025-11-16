//
// Created by pbustos on 16/11/25.
//


// cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "room_optimizer.h"

RoomOptimizer::Result RoomOptimizer::optimize(const RoboCompLidar3D::TPoints &points,
                                              RoomModel &room,
                                              std::shared_ptr<TimeSeriesPlotter> time_series_plotter,
                                              int num_iterations,
                                              float min_loss_threshold,
                                              float learning_rate)
{
    Result res;

    if (points.empty()) return res;

    // Convert points to flat float vector and then to tensor [N,2]
    std::vector<float> points_data;
    points_data.reserve(points.size() * 2);
    for (const auto &p : points) {
        points_data.push_back(p.x / 1000.0f);
        points_data.push_back(p.y / 1000.0f);
    }

    torch::Tensor points_tensor = torch::from_blob(
        points_data.data(),
        {static_cast<long>(points.size()), 2},
        torch::kFloat32
    ).clone();

    // ADAPTIVE PARAMETER SELECTION
    std::vector<torch::Tensor> params_to_optimize;
    bool is_localized = room_freezing_manager.should_freeze_room();
    if (room_freezing_manager.should_freeze_room())
    {
        // Only optimize robot pose
        params_to_optimize = room.get_robot_parameters();
        room.freeze_room_parameters();
        qInfo() << "🔒 LOCALIZED: Optimizing robot pose only";
    } else
    {
        // Optimize everything
        params_to_optimize = room.parameters();
        room.unfreeze_room_parameters();
        qInfo() << "🗺️  MAPPING: Optimizing room + robot pose";
    }

    // Adam optimizer
    torch::optim::Adam optimizer(params_to_optimize, torch::optim::AdamOptions(learning_rate));

    float final_loss = 0.0f;
    const int print_every = 30;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        optimizer.zero_grad();

        torch::Tensor loss = RoomLoss::compute_loss(points_tensor, room, 0.1f);

        loss.backward();
        optimizer.step();

        if (iter == num_iterations - 1)
            final_loss = loss.item<float>();

        if (time_series_plotter)
            time_series_plotter->addDataPoint(0, loss.item<double>());

        if (loss.item<float>() < min_loss_threshold)
        {
            if (iter % print_every != 0 && iter != num_iterations - 1)
            {
                const auto robot_pose = room.get_robot_pose();
                std::cout << "  Final State " << std::setw(3) << iter
                          << " | Loss: " << std::fixed << std::setprecision(6)
                          << loss.item<float>()
                          << " | Robot: (" << std::setprecision(2)
                          << robot_pose[0] << ", " << robot_pose[1] << ", "
                          << robot_pose[2] << ")\n";
            }
            final_loss = loss.item<float>();
            break;
        }
    }

    // Compute covariance and std devs via UncertaintyEstimator
    try
    {
        res.covariance = UncertaintyEstimator::compute_covariance(points_tensor, room, 0.1f);
        res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);

        std::vector<float> room_std_devs;
        std::vector<float> robot_std_devs;
        if (is_localized)
        {
            // LOCALIZED: covariance is 3x3 (robot only)
            // std_devs = [robot_x_std, robot_y_std, robot_theta_std]
            robot_std_devs = {res.std_devs[0], res.std_devs[1], res.std_devs[2]};

            // Room uncertainty is zero (frozen)
            room_std_devs = {0.0f, 0.0f};

            // qInfo() << "Robot uncertainty: X=" << robot_std_devs[0]
            // 		<< " Y=" << robot_std_devs[1]
            // 		<< " θ=" << robot_std_devs[2];
        }
        else
        {
            // MAPPING: covariance is 5x5 (room + robot)
            // std_devs = [half_width_std, half_height_std, robot_x_std, robot_y_std, robot_theta_std]
            room_std_devs = {res.std_devs[0], res.std_devs[1]};
            robot_std_devs = {res.std_devs[2], res.std_devs[3], res.std_devs[4]};

            qInfo() << "Room uncertainty: W=" << room_std_devs[0]
                    << " H=" << room_std_devs[1];
            qInfo() << "Robot uncertainty: X=" << robot_std_devs[0]
                    << " Y=" << robot_std_devs[1]
                    << " θ=" << robot_std_devs[2];
        }

        // =========================================================================
        // UPDATE FREEZING MANAGER
        // =========================================================================

        const auto room_params = room.get_room_parameters();
        const bool state_changed = room_freezing_manager.update(
            room_params,
            room_std_devs,
            robot_std_devs,
            room.get_robot_pose(),
            final_loss,
            num_iterations
        );

        if (state_changed) {
            room_freezing_manager.print_status();
            UncertaintyEstimator::print_uncertainty(res.covariance, room);
        }
    } catch (const std::exception &e)
    {
        std::cerr << "RoomOptimizer::optimize - uncertainty computation failed: " << e.what() << "\n";
        res.covariance = torch::empty({0});
        res.std_devs.clear();
    }
    res.final_loss = final_loss;


    return res;
}
