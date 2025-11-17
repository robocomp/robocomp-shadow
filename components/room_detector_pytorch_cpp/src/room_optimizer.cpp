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

RoomOptimizer::Result RoomOptimizer::optimize( const RoboCompLidar3D::TPoints& points,
                                               RoomModel& room,
                                               std::shared_ptr<TimeSeriesPlotter> time_series_plotter,
                                               int num_iterations,
                                               float min_loss_threshold,
                                               float learning_rate,
                                               const OdometryPrior& odometry_prior,
                                               int frame_number)
{
    Result res;

    if (points.empty()) {
        qWarning() << "No points to optimize";
        return res;
    }

    // ===== STEP 0: PROPAGATE COVARIANCE BEFORE OPTIMIZATION =====
    bool is_localized = room_freezing_manager.should_freeze_room();
    const int expected_dim = is_localized ? 3 : 5;

    torch::Tensor propagated_cov;
    bool have_propagated = false;

    if (odometry_prior.valid && uncertainty_manager.has_history()) {
        // Get previous covariance
        auto prev_cov = uncertainty_manager.get_previous_cov();

        // Propagate using velocity command
        propagated_cov = uncertainty_manager.propagate_with_velocity(
            odometry_prior.velocity_cmd,  // Pass velocity command
            odometry_prior.dt,             // Time delta
            prev_cov,
            is_localized
        );

        have_propagated = true;
        qDebug() << "Propagated covariance before optimization";
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
    if (is_localized)
    {
        // LOCALIZED: Optimize robot pose + odometry calibration
        params_to_optimize = room.get_robot_parameters();

        // Add calibration parameters
        auto calib_params = room.get_calibration_parameters();
        params_to_optimize.insert(params_to_optimize.end(),
                                  calib_params.begin(),
                                  calib_params.end());

        room.freeze_room_parameters();
        //qInfo() << "🔒 LOCALIZED: Optimizing robot pose + calibration";
    } else
    {
        // MAPPING: Optimize everything (room + robot)
        params_to_optimize = room.parameters();
        room.unfreeze_room_parameters();
        //qInfo() << "🗺️  MAPPING: Optimizing room + robot pose";
    }

    // ===== STEP 2.5: GET PRIOR FROM VELOCITY INTEGRATION =====
    bool use_odometry = odometry_prior.valid and is_localized;  // Only in LOCALIZED mode
    torch::Tensor prior_loss = torch::zeros(1, torch::kFloat32);
    torch::Tensor predicted_pose;

    if (use_odometry)
    {
        // Get previous pose as tensor (detached - not part of optimization yet)
        auto prev_pose = room.get_robot_pose();
        torch::Tensor prev_pose_tensor = torch::tensor(
            {prev_pose[0], prev_pose[1], prev_pose[2]},
            torch::kFloat32
        ).requires_grad_(false);

        // Predict using CALIBRATED motion model
        // This maintains gradients w.r.t. calibration parameters
        predicted_pose = room.predict_pose_tensor(
            prev_pose_tensor,
            odometry_prior.velocity_cmd,
            odometry_prior.dt
        );

        // NOTE: predicted_pose has gradients w.r.t. k_translation, k_rotation
        // but NOT w.r.t. prev_pose_tensor (which is detached)
        // We'll use retain_graph=True in backward() to avoid "backward twice" error

        // qDebug() << "Odometry prior active. Predicted pose:"
        //          << predicted_pose[0].item<float>()
        //          << predicted_pose[1].item<float>()
        //          << predicted_pose[2].item<float>();
        //
        // auto calib = room.get_odometry_calibration();
        // qDebug() << "Calibration: k_trans=" << calib[0] << ", k_rot=" << calib[1];
    }

    // ===== STEP 3: RUN OPTIMIZATION LOOP =====
    torch::optim::Adam optimizer(params_to_optimize, torch::optim::AdamOptions(learning_rate));
    float final_loss = 0.0f;
    const int print_every = 30;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        optimizer.zero_grad();

        // Measurement likelihood (SDF loss)
        torch::Tensor measurement_loss = RoomLoss::compute_loss(points_tensor, room, 0.1f);

        // Combined loss: likelihood + prior
        torch::Tensor total_loss = measurement_loss;

        if (use_odometry)   // compute prior loss as the Mahalanobis distance between predicted and current pose
        {
            // Build pose_diff from actual parameter tensors (connected to computation graph)
            torch::Tensor predicted_pos = predicted_pose.slice(0, 0, 2);     // [x, y]
            torch::Tensor predicted_theta = predicted_pose.slice(0, 2, 3);   // [theta]
            // Use actual parameters (these have gradients!)
            torch::Tensor pos_diff = room.robot_pos_ - predicted_pos;
            torch::Tensor theta_diff = room.robot_theta_ - predicted_theta;
            torch::Tensor pose_diff = torch::cat({pos_diff, theta_diff});

            // Mahalanobis distance
            torch::Tensor reg_cov = odometry_prior.covariance + 1e-6 * torch::eye(3);
            torch::Tensor info_matrix = torch::inverse(reg_cov);

            prior_loss = 0.5 * torch::matmul(
                pose_diff.unsqueeze(0),
                torch::matmul(info_matrix, pose_diff.unsqueeze(1))
            ).squeeze();

            const float prior_weight = 1.0f;
            total_loss = total_loss + prior_weight * prior_loss;
        }

        // Backward pass and optimization step
        // Retain graph if using odometry prior (predicted_pose has gradients w.r.t. calibration)
        if (use_odometry && iter < num_iterations - 1) {
            total_loss.backward({}, /*retain_graph=*/true);
        } else {
            total_loss.backward();
        }
        optimizer.step();
        for (auto& p : params_to_optimize)
            if (p.grad().defined())
                p.grad().clamp_(-10.0f, 10.0f);

        // Clamp calibration parameters to reasonable range
        if (is_localized) {
            torch::NoGradGuard no_grad;
            auto calib_params = room.get_calibration_parameters();
            for (auto& p : calib_params) {
                p.clamp_(0.5f, 2.0f);  // Allow 50%-200% scaling
            }
        }

        final_loss = total_loss.item<float>();

        // Plot loss
        if (time_series_plotter)
            time_series_plotter->addDataPoint(0, total_loss.item<double>());

        // Early stopping
        if (final_loss < min_loss_threshold)
        {
            if (iter % print_every != 30 && iter != num_iterations - 1)
            {
                // const auto robot_pose = room.get_robot_pose();
                // std::cout << "  Final State " << std::setw(3) << iter
                //          << " | Loss: " << std::fixed << std::setprecision(6) << final_loss
                //          << " | Robot: (" << std::setprecision(2)
                //          << robot_pose[0] << ", " << robot_pose[1] << ", "
                //          << robot_pose[2] << ")\n";
            }
            break;
        }

        // Debug calibration learning
        // if (use_odometry && iter % 20 == 0) {
        //     auto calib = room.get_odometry_calibration();
        //     qDebug() << "Iter" << iter
        //              << "Meas:" << measurement_loss.item<float>()
        //              << "Prior:" << prior_loss.item<float>()
        //              << "Total:" << total_loss.item<float>()
        //              << "k_trans:" << calib[0]
        //              << "k_rot:" << calib[1];
        // }
    }

    // Print learned calibration (if optimized)
    if (is_localized) {
        auto calib = room.get_odometry_calibration();
        static int print_counter = 0;
        if (print_counter++ % 50 == 0) {  // Print every 50 frames
            qInfo() << "🎯 Odometry calibration: k_trans=" << calib[0]
                    << ", k_rot=" << calib[1];
        }
    }

    // ===== STEP 4: COMPUTE MEASUREMENT COVARIANCE =====
    torch::Tensor measurement_cov;
    try {
        measurement_cov = UncertaintyEstimator::compute_covariance(
            points_tensor, room, 0.1f
        );

        // ===== STEP 5: FUSE WITH PROPAGATED COVARIANCE =====
        if (have_propagated) {
            // Information form fusion
            res.covariance = uncertainty_manager.fuse_covariances(
                propagated_cov,
                measurement_cov
            );
            res.used_fusion = true;
        } else {
            // First frame - use measurement only
            res.covariance = measurement_cov;
        }

        res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);
        res.uncertainty_valid = true;

    } catch (const std::exception& e) {
        qWarning() << "Uncertainty computation failed:" << e.what();
        res.covariance = torch::eye(expected_dim, torch::kFloat32) * 0.01f;
        res.std_devs = std::vector<float>(expected_dim, 0.1f);
        res.uncertainty_valid = false;
    }

    // ===== STEP 6: UPDATE HISTORY =====
    uncertainty_manager.set_previous_cov(res.covariance);
    uncertainty_manager.set_previous_pose(room.get_robot_pose());

    // ===== STEP 7: UPDATE FREEZING MANAGER =====
    std::vector<float> room_std_devs;
    std::vector<float> robot_std_devs;

    if (is_localized) {
        // LOCALIZED: std_devs may include calibration params
        // Always use first 3 for robot pose
        robot_std_devs = {res.std_devs[0], res.std_devs[1], res.std_devs[2]};
        room_std_devs = {0.0f, 0.0f};  // Room frozen

        // Log calibration uncertainty if present
        if (res.std_devs.size() >= 5) {
            static int calib_log_counter = 0;
            if (calib_log_counter++ % 100 == 0) {
                qInfo() << "Calibration uncertainty: k_trans ±" << res.std_devs[3]
                        << ", k_rot ±" << res.std_devs[4];
            }
        }
    } else {
        // MAPPING: std_devs = [half_width, half_height, robot_x, robot_y, robot_theta]
        room_std_devs = {res.std_devs[0], res.std_devs[1]};
        robot_std_devs = {res.std_devs[2], res.std_devs[3], res.std_devs[4]};

        // qInfo() << "Robot uncertainty: X=" << robot_std_devs[0]
        //         << " Y=" << robot_std_devs[1]
        //         << " θ=" << robot_std_devs[2];
    }

    const auto room_params = room.get_room_parameters();
    bool state_changed = room_freezing_manager.update(
        room_params,
        room_std_devs,
        robot_std_devs,
        room.get_robot_pose(),
        final_loss,
        frame_number
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