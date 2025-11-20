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

RoomOptimizer::Result RoomOptimizer::optimize(  const TimePoints &points_,
                                                RoomModel &room,
                                                const VelocityHistory &velocity_history,
                                                std::shared_ptr<TimeSeriesPlotter> time_series_plotter,
                                                int num_iterations,
                                                float min_loss_threshold,
                                                float learning_rate)
{
    Result res;
    const auto &[points_r, lidar_timestamp] = points_;
    if (points_r.empty()) { qWarning() << "No points to optimize";  return {};}

    // Compute odometry prior betweeen lidar timestamps
    auto odometry_prior = compute_odometry_prior(room, velocity_history, lidar_timestamp);

    bool is_localized = room_freezing_manager.should_freeze_room();

    // ===== STEP 0: PROPAGATE COVARIANCE BEFORE OPTIMIZATION =====
    // Covariance dimension: 3 for robot pose in LOCALIZED, 5 for room+pose in MAPPING
    const int expected_dim = is_localized ? 3 : 5;
    torch::Tensor propagated_cov;
    bool have_propagated = false;

    if (odometry_prior.valid and uncertainty_manager.has_history())
    {
        // Get previous covariance and check for dimension mismatch (state transition)
        if (auto prev_cov = uncertainty_manager.get_previous_cov(); prev_cov.size(0) != expected_dim)
            uncertainty_manager.reset();
        else
        {
            // Propagate using velocity command
            propagated_cov = uncertainty_manager.propagate_with_velocity(
                odometry_prior.velocity_cmd,  // Pass velocity command
                odometry_prior.dt,             // Time delta
                prev_cov,
                is_localized
            );
            have_propagated = true;
            //qDebug() << "Propagated covariance before optimization";
        }
    }

    // STEP 0.3 The model should be at predicted pose when filtering
    std::vector<float> original_pose;  // Save original for restoration if needed
    if (odometry_prior.valid && is_localized)
    {
        // Save current pose
        original_pose = room.get_robot_pose();

        // Update model to predicted pose
        auto predicted_pose = room.get_robot_pose();
        predicted_pose[0] += odometry_prior.delta_pose[0];
        predicted_pose[1] += odometry_prior.delta_pose[1];
        predicted_pose[2] += odometry_prior.delta_pose[2];

        // Update model to predicted pose using .data (bypasses autograd)
        room.robot_pos_.data()[0] = predicted_pose[0];
        room.robot_pos_.data()[1] = predicted_pose[1];
        room.robot_theta_.data()[0] = predicted_pose[2];

        //    qDebug() << "Updated model to predicted pose for filtering:"
        //             << predicted_pose[0] << predicted_pose[1] << predicted_pose[2];
    }

    // ===== STEP 0.5: TOP-DOWN PREDICTION: FILTER POINTS BASED ON MODEL FIT =====
    auto residual  = top_down_prediction(points_r, room, have_propagated, propagated_cov);
    //auto residual_points = residual.filtered_set;
    auto points = points_r;

    //if (residual.explained_ratio > 99) return res;  // Nothing to optimize

    // ===== STEP 1: CONVERT POINTS TO TENSOR =====
    std::vector<float> points_data;
    points_data.reserve(points.size() * 2);
    for (const auto& p : points)
    {
        points_data.push_back(p.x / 1000.0f);
        points_data.push_back(p.y / 1000.0f);
    }

    torch::Tensor points_tensor = torch::from_blob(
        points_data.data(),
        {static_cast<long>(points.size()), 2},
        torch::kFloat32
    ).clone();  // TODO: check memory management

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
    float final_measurement_loss = 0.0f;
    constexpr int print_every = 30;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        optimizer.zero_grad();

        // Measurement likelihood (SDF loss)
        torch::Tensor measurement_loss = RoomLoss::compute_loss(points_tensor, room, 0.1f);
        final_measurement_loss = measurement_loss.item<float>();

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

            constexpr float prior_weight = 1.0f;
            total_loss = total_loss + prior_weight * prior_loss;

            // Add calibration regularization to keep parameters near 1.0
            // This prevents wild oscillations and encourages conservative calibration
            if (auto calib_params = room.get_calibration_parameters(); !calib_params.empty())
            {
                torch::Tensor calib_reg_loss = torch::zeros(1, torch::kFloat32);
                for (const auto& p : calib_params) // Penalize deviation from 1.0
                    calib_reg_loss = calib_reg_loss + torch::square(p.squeeze() - 1.0f);
                total_loss = total_loss + calib_config.regularization_weight * calib_reg_loss;
            }
        }

        // Backward pass and optimization step
        // Retain graph if using odometry prior (predicted_pose has gradients w.r.t. calibration)
        if (use_odometry && iter < num_iterations - 1)
            total_loss.backward({}, /*retain_graph=*/true);
        else
            total_loss.backward();

        optimizer.step();
        for (auto& p : params_to_optimize)
            if (p.grad().defined())
                p.grad().clamp_(-10.0f, 10.0f);

        // Clamp calibration parameters to reasonable range
        if (is_localized)
        {
            torch::NoGradGuard no_grad;
            for (auto calib_params = room.get_calibration_parameters(); auto& p : calib_params)
                p.clamp_(calib_config.min_value, calib_config.max_value);
        }

        final_loss = total_loss.item<float>();

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

    // Plot loss
    if (time_series_plotter)
    {
        time_series_plotter->addDataPoint(0, final_loss);
        time_series_plotter->addDataPoint(2, prior_loss.item<float>()); // prior
        time_series_plotter->addDataPoint(1, final_measurement_loss); //likelihood
    }


    // ===== STEP 4: COMPUTE MEASUREMENT COVARIANCE =====
    // NOTE: We only compute uncertainty for robot pose, not calibration parameters.
    // Calibration parameters don't affect measurement likelihood (SDF), so their
    // uncertainty can't be estimated from Hessian of measurement loss.
    torch::Tensor measurement_cov;
    try {
        // Temporarily get only robot parameters for uncertainty computation
        auto robot_params = room.get_robot_parameters();

        // Freeze calibration during uncertainty computation (always)
        // Calibration doesn't affect measurement likelihood, so it can't have
        // uncertainty estimated from Hessian
        bool calib_was_trainable = false;
        auto calib_params = room.get_calibration_parameters();
        if (!calib_params.empty() && calib_params[0].requires_grad()) {
            calib_was_trainable = true;
            room.freeze_odometry_calibration();
        }

        measurement_cov = UncertaintyEstimator::compute_covariance(
            points_tensor, room, 0.1f
        );

        // Restore calibration state
        if (calib_was_trainable) {
            room.unfreeze_odometry_calibration();
        }

        // Check for NaN/Inf before inflation
        if (torch::any(torch::isnan(measurement_cov)).item<bool>() ||
            torch::any(torch::isinf(measurement_cov)).item<bool>()) {
            qWarning() << "Covariance contains NaN/Inf, using conservative fallback";
            // Use conservative estimate: larger uncertainty = safer
            float base_uncertainty = 0.1f;  // 10cm base uncertainty
            measurement_cov = torch::eye(expected_dim, torch::kFloat32) * (base_uncertainty * base_uncertainty);
        }

        // Inflate covariance to account for modeling errors and be more conservative
        // The Laplace approximation tends to be overconfident
        measurement_cov = measurement_cov * calib_config.uncertainty_inflation;

        // ===== STEP 5: FUSE WITH PROPAGATED COVARIANCE =====
        if (have_propagated) {
            // Both should have same dimension now
            if (propagated_cov.size(0) == measurement_cov.size(0)) {
                // Information form fusion
                res.covariance = uncertainty_manager.fuse_covariances(
                    propagated_cov,
                    measurement_cov
                );
                res.used_fusion = true;
            } else {
                qWarning() << "Dimension mismatch in fusion:"
                          << propagated_cov.size(0) << "vs" << measurement_cov.size(0);
                res.covariance = measurement_cov;
            }
        } else {
            // First frame or state transition - use measurement only
            res.covariance = measurement_cov;
        }

        res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);

        // Final validation: check for NaN in std_devs
        bool has_nan = false;
        for (const auto& val : res.std_devs) {
            if (std::isnan(val) || std::isinf(val)) {
                has_nan = true;
                break;
            }
        }

        if (has_nan)
        {
            qWarning() << "NaN detected in std_devs after computation, using safe fallback";
            float safe_pos_std = 0.05f;  // 5cm after inflation (0.5mm before)
            float safe_ang_std = 0.02f;  // ~1.1 degrees after inflation

            if (expected_dim == 3) {
                // LOCALIZED: [x, y, theta]
                res.std_devs = {safe_pos_std, safe_pos_std, safe_ang_std};
            } else {
                // MAPPING: [width, height, x, y, theta]
                res.std_devs = {safe_pos_std, safe_pos_std, safe_pos_std, safe_pos_std, safe_ang_std};
            }

            // Reconstruct covariance from std_devs
            res.covariance = torch::diag(torch::tensor(res.std_devs).square()) * calib_config.uncertainty_inflation;
            res.uncertainty_valid = false;
        } else {
            res.uncertainty_valid = true;
        }

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
        // LOCALIZED: Covariance is always 3x3 (robot pose only)
        // Calibration uncertainty not computed (doesn't affect measurement likelihood)
        robot_std_devs = res.std_devs;  // All 3 values are robot pose
        room_std_devs = {0.0f, 0.0f};   // Room frozen
    } else {
        // MAPPING: std_devs = [half_width, half_height, robot_x, robot_y, robot_theta]
        room_std_devs = {res.std_devs[0], res.std_devs[1]};
        robot_std_devs = {res.std_devs[2], res.std_devs[3], res.std_devs[4]};
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

    // Debugging: report prior loss
    // if (use_odometry)
    // {
    //     float final_prior_loss = prior_loss.item<float>();
    //
    //     qDebug() << "Prediction quality: prior_loss =" << final_prior_loss;
    //
    //     if (final_prior_loss > 0.1f) {
    //         qWarning() << "⚠️  Large prediction error! Consider:"
    //                    << "- Checking odometry calibration"
    //                    << "- Increasing process noise"
    //                    << "- Wheel slip or dynamic motion";
    //     }
    // }
    frame_number++;
    return res;
}

Eigen::Vector3f RoomOptimizer::integrate_velocity_over_window( const RoomModel& room,
                                                               const boost::circular_buffer<VelocityCommand> &velocity_history,
                                                               const std::chrono::time_point<std::chrono::high_resolution_clock> &t_start,
                                                               const std::chrono::time_point<std::chrono::high_resolution_clock> &t_end)
{
    Eigen::Vector3f total_delta = Eigen::Vector3f::Zero();

    const auto current_pose = room.get_robot_pose();
    float running_theta = current_pose[2];

    // Integrate over all velocity commands in [t_start, t_end]
    for (size_t i = 0; i < velocity_history.size(); ++i)
    {
        const auto&[adv_x, adv_z, rot, timestamp] = velocity_history[i];

        // Get time window for this command
        auto cmd_start = timestamp;
        auto cmd_end = (i + 1 < velocity_history.size())
                        ? velocity_history[i + 1].timestamp
                        : t_end;

        // Clip to [t_start, t_end]
        if (cmd_end < t_start) continue;
        if (cmd_start > t_end) break;

        auto effective_start = std::max(cmd_start, t_start);
        auto effective_end = std::min(cmd_end, t_end);

        const float dt = std::chrono::duration<float>(effective_end - effective_start).count();
        if (dt <= 0) continue;

        // Integrate this segment
        const float dx_local = (adv_x * dt) / 1000.0f;
        const float dy_local = (adv_z * dt) / 1000.0f;
        const float dtheta = -rot * dt;  // Negative for right-hand rule

        // Transform to global frame using RUNNING theta
        total_delta[0] += dx_local * std::cos(running_theta) - dy_local * std::sin(running_theta);
        total_delta[1] += dx_local * std::sin(running_theta) + dy_local * std::cos(running_theta);
        total_delta[2] += dtheta;

        // Update running theta for next segment
        running_theta += dtheta;
    }

    return total_delta;
}

OdometryPrior RoomOptimizer::compute_odometry_prior(const RoomModel &room,
                                                    const boost::circular_buffer<VelocityCommand>& velocity_history,
                                                    const std::chrono::time_point<std::chrono::high_resolution_clock> &lidar_timestamp)
{
    OdometryPrior prior;
    prior.valid = false;

    if (last_lidar_timestamp == std::chrono::time_point<std::chrono::high_resolution_clock>{})
        return prior;

    if (velocity_history.empty())
    {
        //qWarning() << "No velocity commands in buffer";
        return prior;
    }

    // Time delta BETWEEN LIDAR SCANS
    const float dt = std::chrono::duration<float>(lidar_timestamp - last_lidar_timestamp).count();

    if (dt <= 0 or dt > 0.5f)
    {
        qWarning() << "Invalid dt for odometry:" << dt << "s";
        return prior;
    }

    // Integrate velocity over the time window [t_start, t_end]
    prior.delta_pose = integrate_velocity_over_window(room, velocity_history, last_lidar_timestamp, lidar_timestamp);

    // For storing in prior structure (if needed by optimizer)
    if (not velocity_history.empty())
        prior.velocity_cmd = velocity_history.back();

    prior.dt = dt;

    // Compute process noise - proportional to motion magnitude
    const float trans_magnitude = std::sqrt(
        prior.delta_pose[0]*prior.delta_pose[0] +
        prior.delta_pose[1]*prior.delta_pose[1]
    );
    const float rot_magnitude = std::abs(prior.delta_pose[2]);
    float trans_noise_std = prediction_params.NOISE_TRANS * trans_magnitude;
    float rot_noise_std = prediction_params.NOISE_ROT * rot_magnitude;

    // Minimum uncertainty
    trans_noise_std = std::max(trans_noise_std, 0.1f);
    rot_noise_std = std::max(rot_noise_std, 0.15f);

    // Extra uncertainty for pure rotation
    if (trans_magnitude < 0.01f && rot_magnitude > 0.01f) {
        rot_noise_std *= 2.0f;
        trans_noise_std = 0.2f;
    }

    prior.covariance = torch::eye(3, torch::kFloat32);
    prior.covariance[0][0] = trans_noise_std * trans_noise_std;
    prior.covariance[1][1] = trans_noise_std * trans_noise_std;
    prior.covariance[2][2] = rot_noise_std * rot_noise_std;

    // qDebug() << "=== ODOMETRY PRIOR (LIDAR-TO-LIDAR) ===";
    // qDebug() << "dt:" << dt << "s";
    // qDebug() << "Delta pose:" << prior.delta_pose[0] << prior.delta_pose[1] << prior.delta_pose[2];
    // qDebug() << "Motion magnitude: trans=" << trans_magnitude << "rot=" << rot_magnitude;

    prior.valid = true;
    last_lidar_timestamp = lidar_timestamp; // Update last timestamp
    return prior;
}

ModelBasedFilter::Result RoomOptimizer::top_down_prediction(const RoboCompLidar3D::TPoints &points,
                                                            RoomModel &room,
                                                            bool have_propagated,
                                                            const torch::Tensor &propagated_cov)
{
    const bool is_localized = room_freezing_manager.should_freeze_room();
    RoboCompLidar3D::TPoints residual;
    if (is_localized)  // localized
    {
        // Get robot uncertainty from PREVIOUS timestep (not current res!)
        float robot_uncertainty = 0.0f;

        if (have_propagated)  // Use the propagated covariance we just computed
            if (propagated_cov.size(0) == 3) // This is the PREDICTED uncertainty at current time
                robot_uncertainty = std::sqrt(propagated_cov[0][0].item<float>() + propagated_cov[1][1].item<float>());

        // Filter points based on model fit
        auto filter_result = model_based_filter.filter(points, room, robot_uncertainty);
        model_based_filter.print_result( points.size(), filter_result);
        return filter_result;  // residual
    }
}



