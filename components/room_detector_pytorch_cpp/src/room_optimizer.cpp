//
// Refactored RoomOptimizer - minimal version matching your existing structure
//

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "room_optimizer.h"
#include "room_loss.h"
#include "uncertainty_estimator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

RoomOptimizer::Result RoomOptimizer::optimize(
    const RoboCompLidar3D::TPoints& points,
    RoomModel& room,
    std::shared_ptr<TimeSeriesPlotter> time_series_plotter,
    int num_iterations,
    float min_loss_threshold,
    float learning_rate)
{
    Result res;

    if (points.empty()) {
        qWarning() << "No points to optimize";
        return res;
    }

    // ===== STEP 1: CONVERT POINTS TO TENSOR =====
    std::vector<float> points_data;
    points_data.reserve(points.size() * 2);
    for (const auto& p : points) {
        points_data.push_back(p.x / 1000.0f);
        points_data.push_back(p.y / 1000.0f);
    }

    torch::Tensor points_tensor = torch::from_blob(
        points_data.data(),
        {static_cast<long>(points.size()), 2},
        torch::kFloat32
    ).clone();

    // ===== STEP 2: SELECT PARAMETERS (MAPPING vs LOCALIZED) =====
    std::vector<torch::Tensor> params_to_optimize;
    bool is_localized = room_freezing_manager.should_freeze_room();

    if (is_localized) {
        // LOCALIZED: Only optimize robot pose
        params_to_optimize = room.get_robot_parameters();
        room.freeze_room_parameters();
        qInfo() << "🔒 LOCALIZED: Optimizing robot pose only";
    } else {
        // MAPPING: Optimize everything
        params_to_optimize = room.parameters();
        room.unfreeze_room_parameters();
        qInfo() << "🗺️  MAPPING: Optimizing room + robot pose";
    }

    // ===== STEP 3: RUN OPTIMIZATION LOOP =====
    torch::optim::Adam optimizer(
        params_to_optimize,
        torch::optim::AdamOptions(learning_rate)
    );

    float final_loss = 0.0f;
    const int print_every = 30;

    for (int iter = 0; iter < num_iterations; ++iter) {
        optimizer.zero_grad();

        // Using your existing RoomLoss class
        torch::Tensor loss = RoomLoss::compute_loss(points_tensor, room, 0.1f);

        loss.backward();
        optimizer.step();

        final_loss = loss.item<float>();

        // Plot loss
        if (time_series_plotter) {
            time_series_plotter->addDataPoint(0, loss.item<double>());
        }

        // Early stopping
        if (final_loss < min_loss_threshold) {
            if (iter % print_every != 0 && iter != num_iterations - 1) {
                const auto robot_pose = room.get_robot_pose();
                std::cout << "  Final State " << std::setw(3) << iter
                         << " | Loss: " << std::fixed << std::setprecision(6) << final_loss
                         << " | Robot: (" << std::setprecision(2)
                         << robot_pose[0] << ", " << robot_pose[1] << ", "
                         << robot_pose[2] << ")\n";
            }
            break;
        }
    }

    // ===== STEP 4: COMPUTE UNCERTAINTY (using new UncertaintyManager) =====
    try {
        auto uncertainty_result = uncertainty_manager.compute(
            points_tensor,
            room,
            0.1f,  // huber_delta
            is_localized
        );

        res.covariance = uncertainty_result.covariance;
        res.std_devs = uncertainty_result.std_devs;
        res.uncertainty_valid = uncertainty_result.is_valid;

        if (!uncertainty_result.is_valid) {
            qWarning() << "Uncertainty computation issue:"
                      << QString::fromStdString(uncertainty_result.error_message);
        }

        // Optional debug info
        if (uncertainty_result.used_propagation) {
            qDebug() << "Used motion propagation";
        }
        if (uncertainty_result.used_fusion) {
            qDebug() << "Used covariance fusion";
        }

    } catch (const std::exception& e) {
        qWarning() << "Uncertainty computation failed:" << e.what();
        res.covariance = torch::eye(is_localized ? 3 : 5, torch::kFloat32) * 0.01f;
        res.std_devs = std::vector<float>(is_localized ? 3 : 5, 0.1f);
        res.uncertainty_valid = false;
    }

    // ===== STEP 5: UPDATE FREEZING MANAGER =====
    std::vector<float> room_std_devs;
    std::vector<float> robot_std_devs;

    if (is_localized) {
        // LOCALIZED: std_devs = [robot_x, robot_y, robot_theta]
        robot_std_devs = {res.std_devs[0], res.std_devs[1], res.std_devs[2]};
        room_std_devs = {0.0f, 0.0f};  // Room frozen
    } else {
        // MAPPING: std_devs = [half_width, half_height, robot_x, robot_y, robot_theta]
        room_std_devs = {res.std_devs[0], res.std_devs[1]};
        robot_std_devs = {res.std_devs[2], res.std_devs[3], res.std_devs[4]};

        qInfo() << "Room uncertainty: W=" << room_std_devs[0]
                << " H=" << room_std_devs[1];
        qInfo() << "Robot uncertainty: X=" << robot_std_devs[0]
                << " Y=" << robot_std_devs[1]
                << " θ=" << robot_std_devs[2];
    }

    const auto room_params = room.get_room_parameters();
    bool state_changed = room_freezing_manager.update(
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

        // Reset uncertainty history on state transition
        uncertainty_manager.reset();
    }

    res.final_loss = final_loss;
    return res;
}