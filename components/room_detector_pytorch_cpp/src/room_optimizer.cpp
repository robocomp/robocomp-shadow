//
// RoomOptimizer - EKF-style predict-update cycle implementation
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

// ============================================================================
// MAIN EKF CYCLE
// ============================================================================

RoomOptimizer::Result RoomOptimizer::optimize(
    const TimePoints &points_,
    RoomModel &room,
    const VelocityHistory &velocity_history,
    std::shared_ptr<TimeSeriesPlotter> time_series_plotter,
    int num_iterations,
    float min_loss_threshold,
    float learning_rate)
{
    const auto &[points_r, lidar_timestamp] = points_;
    if (points_r.empty()) {
        qWarning() << "No points to optimize";
        return {};
    }

    // Check if in LOCALIZED mode
    const bool is_localized = room_freezing_manager.should_freeze_room();

    // Compute odometry prior between lidar timestamps
    auto odometry_prior = compute_odometry_prior(room, velocity_history, lidar_timestamp);

    // ===== PREDICT STEP =====
    const PredictionState prediction = predict_step(room, odometry_prior, is_localized);

    // ===== FILTER MEASUREMENTS (TOP-DOWN PREDICTION) =====
    auto filter_result = filter_measurements(points_r, room, prediction);
    // Could use filtered points: auto points = filter_result.filtered_set;
    const auto points = points_r;  // For now, use all points

    // ===== CONVERT MEASUREMENTS TO TENSOR =====
    const torch::Tensor points_tensor = convert_points_to_tensor(points);

    // ===== UPDATE STEP =====
    Result res = update_step(points_tensor, room, odometry_prior, is_localized,
                             num_iterations, min_loss_threshold, learning_rate, time_series_plotter);

    // Store the prior for debugging
    res.prior = odometry_prior;

    // ===== UPDATE STATE MANAGEMENT =====
    update_state_management(res, room, is_localized, res.final_loss);

    frame_number++;
    return res;
}

// ============================================================================
// PREDICT PHASE
// ============================================================================

RoomOptimizer::PredictionState RoomOptimizer::predict_step(
    RoomModel &room,
    const OdometryPrior &odometry_prior,
    bool is_localized)
{
    PredictionState prediction;

    // Covariance dimension: 3 for robot pose in LOCALIZED, 5 for room+pose in MAPPING
    const int expected_dim = is_localized ? 3 : 5;

    // Check if we can propagate covariance
    if (!odometry_prior.valid || !uncertainty_manager.has_history()) {
        return prediction;  // No prediction available
    }

    // Get previous covariance and check for dimension mismatch (state transition)
    auto prev_cov = uncertainty_manager.get_previous_cov();
    if (prev_cov.size(0) != expected_dim) {
        uncertainty_manager.reset();
        return prediction;
    }

    // Propagate covariance using velocity command
    prediction.propagated_cov = uncertainty_manager.propagate_with_velocity(
        odometry_prior.velocity_cmd,
        odometry_prior.dt,
        prev_cov,
        is_localized
    );
    prediction.have_propagated = true;

    // Update model to predicted pose (for LOCALIZED mode)
    if (is_localized) {
        // Compute predicted pose
        auto current_pose = room.get_robot_pose();
        prediction.predicted_pose = {
            current_pose[0] + odometry_prior.delta_pose[0],
            current_pose[1] + odometry_prior.delta_pose[1],
            current_pose[2] + odometry_prior.delta_pose[2]
        };

        // Update model parameters (bypassing autograd with .data())
        room.robot_pos_.data()[0] = prediction.predicted_pose[0];
        room.robot_pos_.data()[1] = prediction.predicted_pose[1];
        room.robot_theta_.data()[0] = prediction.predicted_pose[2];
    }

    return prediction;
}

ModelBasedFilter::Result RoomOptimizer::filter_measurements(
    const RoboCompLidar3D::TPoints &points,
    RoomModel &room,
    const PredictionState &prediction)
{
    const bool is_localized = room_freezing_manager.should_freeze_room();

    if (!is_localized) {
        return {};  // No filtering in MAPPING mode
    }

    // Get robot uncertainty from predicted covariance
    float robot_uncertainty = 0.0f;
    if (prediction.have_propagated && prediction.propagated_cov.size(0) == 3) {
        robot_uncertainty = std::sqrt(
            prediction.propagated_cov[0][0].item<float>() +
            prediction.propagated_cov[1][1].item<float>()
        );
    }

    // Filter points based on model fit
    auto filter_result = model_based_filter.filter(points, room, robot_uncertainty);
    //model_based_filter.print_result(points.size(), filter_result);

    return filter_result;
}

// ============================================================================
// UPDATE PHASE
// ============================================================================

RoomOptimizer::Result RoomOptimizer::update_step(
    const torch::Tensor &points_tensor,
    RoomModel &room,
    const OdometryPrior &odometry_prior,
    bool is_localized,
    int num_iterations,
    float min_loss_threshold,
    float learning_rate,
    std::shared_ptr<TimeSeriesPlotter> time_series_plotter)
{
    Result res;
    res.prior = odometry_prior;

    // Select parameters to optimize based on mode
    auto params_to_optimize = select_optimization_parameters(room, is_localized);

    // Prepare odometry prior for optimization
    bool use_odometry_prior = odometry_prior.valid && is_localized;
    torch::Tensor prev_pose_tensor;

    // Check if there's meaningful motion for calibration learning
    bool has_meaningful_motion = false;
    if (use_odometry_prior)
    {
        const float trans_magnitude = std::sqrt(
            odometry_prior.delta_pose[0] * odometry_prior.delta_pose[0] +
            odometry_prior.delta_pose[1] * odometry_prior.delta_pose[1]
        );
        const float rot_magnitude = std::abs(odometry_prior.delta_pose[2]);

        // INCREASED THRESHOLDS for more stable calibration learning:
        // Require at least 20mm translation OR 2 degrees rotation
        // AND dt must be at least 100ms (avoid noisy short intervals)
        const float min_translation = 0.02f;  // 20mm
        const float min_rotation = 0.035f;     // ~2 degrees
        const float min_dt = 0.09f;             // 100ms

        has_meaningful_motion = (trans_magnitude > min_translation || rot_magnitude > min_rotation)
                              && odometry_prior.dt > min_dt;

        if (!has_meaningful_motion)
        {
            if (odometry_prior.dt < min_dt)
            {
                qDebug() << "⚠️ Time interval too small ("
                        << QString::number(odometry_prior.dt * 1000.0f, 'f', 1)
                        << "ms < " << (min_dt * 1000.0f) << "ms) - skipping calibration";
            } else {
                //qDebug() << "⚠️ Robot motion too small - skipping calibration optimization";
            }
            use_odometry_prior = false;  // Disable prior, optimize only from measurements
        }
    }

    use_odometry_prior = false;
    if (use_odometry_prior)
    {
        // Get previous pose (detached from computation graph)
        auto prev_pose = room.get_robot_pose();
        prev_pose_tensor = torch::tensor(
            {prev_pose[0], prev_pose[1], prev_pose[2]},
            torch::kFloat32
        ).requires_grad_(false);

        // Note: predicted_pose will be computed INSIDE the optimization loop
        // to allow gradients to flow through calibration parameters
    }

    // Run optimization loop
    torch::optim::Adam optimizer(params_to_optimize, torch::optim::AdamOptions(learning_rate));

    float final_loss = run_optimization_loop(
        points_tensor,
        room,
        prev_pose_tensor,
        odometry_prior,
        optimizer,
        use_odometry_prior,
        num_iterations,
        min_loss_threshold,
        time_series_plotter
    );

    // Estimate uncertainty after optimization
    res = estimate_uncertainty(room, points_tensor, is_localized, final_loss);

    return res;
}

std::vector<torch::Tensor> RoomOptimizer::select_optimization_parameters(
    RoomModel &room,
    bool is_localized)
{
    std::vector<torch::Tensor> params_to_optimize;

    if (is_localized) {
        // LOCALIZED: Optimize robot pose + odometry calibration
        params_to_optimize = room.get_robot_parameters();

        // Add calibration parameters
        auto calib_params = room.get_calibration_parameters();
        params_to_optimize.insert(params_to_optimize.end(),
                                 calib_params.begin(),
                                 calib_params.end());

        room.freeze_room_parameters();
    } else {
        // MAPPING: Optimize everything (room + robot)
        params_to_optimize = room.parameters();
        room.unfreeze_room_parameters();
    }

    return params_to_optimize;
}

float RoomOptimizer::run_optimization_loop(
    const torch::Tensor &points_tensor,
    RoomModel &room,
    const torch::Tensor &prev_pose_tensor,
    const OdometryPrior &odometry_prior,
    torch::optim::Optimizer &optimizer,
    bool use_odometry_prior,
    int num_iterations,
    float min_loss_threshold,
    std::shared_ptr<TimeSeriesPlotter> time_series_plotter)
{
    float final_loss = 0.0f;
    constexpr int print_every = 30;

    for (int iter = 0; iter < num_iterations; ++iter) {
        optimizer.zero_grad();

        // Measurement loss
        torch::Tensor measurement_loss = compute_measurement_loss(points_tensor, room);
        torch::Tensor total_loss = measurement_loss;

        // Add prior loss if using odometry
        torch::Tensor prior_loss;
        torch::Tensor calib_reg;
        if (use_odometry_prior) {
            // CRITICAL FIX: Recompute predicted_pose INSIDE the loop
            // This allows gradients to flow through calibration parameters on each iteration
            torch::Tensor predicted_pose = room.predict_pose_tensor(
                prev_pose_tensor,
                odometry_prior.velocity_cmd,
                odometry_prior.dt
            );

            prior_loss = compute_prior_loss(room, predicted_pose, odometry_prior);
            constexpr float prior_weight = 1.0f;
            total_loss = total_loss + prior_weight * prior_loss;

            // Add calibration regularization (with proper gradients!)
            calib_reg = compute_calibration_regularization(room);
            total_loss = total_loss + calib_reg;
        }

        // Backward pass
        total_loss.backward();

        // Clip gradients for stability (more lenient to allow calibration learning)
        torch::nn::utils::clip_grad_norm_(room.parameters(), 5.0);

        optimizer.step();

        // Enforce calibration bounds
        auto calib = room.get_odometry_calibration();
        bool hit_bounds = false;

        if (calib[0] < calib_config.min_value || calib[0] > calib_config.max_value ||
            calib[1] < calib_config.min_value || calib[1] > calib_config.max_value) {

            // Only warn on first iteration when hitting bounds
            if (iter == 0) {
                if (calib[0] <= calib_config.min_value && room.k_translation_.grad().item<float>() > 0.1f) {
                    qWarning() << "⚠️  k_translation at MIN bound (" << calib_config.min_value
                              << ") - likely units mismatch";
                    hit_bounds = true;
                }

                if (calib[0] >= calib_config.max_value && room.k_translation_.grad().item<float>() < -0.1f) {
                    qWarning() << "⚠️  k_translation at MAX bound (" << calib_config.max_value
                              << ") - likely units mismatch";
                    hit_bounds = true;
                }
            }

            room.k_translation_.data().clamp_(calib_config.min_value, calib_config.max_value);
            room.k_rotation_.data().clamp_(calib_config.min_value, calib_config.max_value);
        }

        if (hit_bounds && iter == 0) {
            // Show expected vs actual motion for diagnosis
            const float trans_mag = std::sqrt(
                odometry_prior.delta_pose[0] * odometry_prior.delta_pose[0] +
                odometry_prior.delta_pose[1] * odometry_prior.delta_pose[1]
            );

            const float cmd_x = odometry_prior.velocity_cmd.adv_x * odometry_prior.dt / 1000.0f;
            const float cmd_z = odometry_prior.velocity_cmd.adv_z * odometry_prior.dt / 1000.0f;
            const float cmd_mag = std::sqrt(cmd_x * cmd_x + cmd_z * cmd_z);

            if (cmd_mag > 0.001f) {
                float apparent_scale = trans_mag / cmd_mag;
                qWarning() << "  Commanded:" << QString::number(cmd_mag * 1000.0f, 'f', 1)
                          << "mm, Actual:" << QString::number(trans_mag * 1000.0f, 'f', 1)
                          << "mm, Scale:" << QString::number(apparent_scale, 'f', 3);
            }
        }

        final_loss = total_loss.item<float>();

        // Plot if requested
        if (time_series_plotter && iter % 5 == 0) {
            time_series_plotter->addDataPoint(0, measurement_loss.item<float>());
        }

        // Early stopping
        if (final_loss < min_loss_threshold) {
            break;
        }
    }

    return final_loss;
}

// ============================================================================
// LOSS COMPUTATION
// ============================================================================

torch::Tensor RoomOptimizer::compute_measurement_loss(
    const torch::Tensor &points_tensor,
    RoomModel &room)
{
    return RoomLoss::compute_loss(points_tensor, room, wall_thickness);
}

torch::Tensor RoomOptimizer::compute_prior_loss(
    RoomModel &room,
    const torch::Tensor &predicted_pose,
    const OdometryPrior &odometry_prior)
{
    // Build pose difference (connected to computation graph)
    torch::Tensor predicted_pos = predicted_pose.slice(0, 0, 2);     // [x, y]
    torch::Tensor predicted_theta = predicted_pose.slice(0, 2, 3);   // [theta]

    torch::Tensor pos_diff = room.robot_pos_ - predicted_pos;
    torch::Tensor theta_diff = room.robot_theta_ - predicted_theta;
    torch::Tensor pose_diff = torch::cat({pos_diff, theta_diff});

    // Mahalanobis distance
    torch::Tensor reg_cov = odometry_prior.covariance + 1e-6 * torch::eye(3);
    torch::Tensor info_matrix = torch::inverse(reg_cov);

    torch::Tensor prior_loss = 0.5 * torch::matmul(
        pose_diff.unsqueeze(0),
        torch::matmul(info_matrix, pose_diff.unsqueeze(1))
    ).squeeze();

    return prior_loss;
}

torch::Tensor RoomOptimizer::compute_calibration_regularization(RoomModel &room)
{
    // CRITICAL FIX: Use tensor operations to maintain gradients!
    // Previous version used float operations which broke the gradient flow

    // Get calibration parameters as tensors (maintain gradients)
    torch::Tensor k_trans = room.k_translation_;
    torch::Tensor k_rot = room.k_rotation_;

    // Target value (1.0)
    torch::Tensor target = torch::ones(1, torch::kFloat32);

    // Compute squared deviation from 1.0 (maintains gradients)
    torch::Tensor trans_dev = torch::pow(k_trans - target, 2);
    torch::Tensor rot_dev = torch::pow(k_rot - target, 2);

    // Sum and scale by regularization weight
    torch::Tensor reg_loss = calib_config.regularization_weight * (trans_dev + rot_dev);

    return reg_loss.squeeze();  // Return scalar tensor with gradients
}

// ============================================================================
// UNCERTAINTY ESTIMATION
// ============================================================================

RoomOptimizer::Result RoomOptimizer::estimate_uncertainty(
    RoomModel &room,
    const torch::Tensor &points_tensor,
    bool is_localized,
    float final_loss)
{
    Result res;

    // Use adaptive compute_covariance that handles both LOCALIZED and MAPPING modes
    torch::Tensor covariance;

    try {
        // The UncertaintyEstimator automatically detects which parameters require gradients
        // and returns the appropriate size covariance matrix
        covariance = UncertaintyEstimator::compute_covariance(
            points_tensor,
            room,
            wall_thickness
        );

        // Inflate covariance to correct for overconfidence
        res.covariance = covariance * calib_config.uncertainty_inflation;
        res.uncertainty_valid = true;

        // Extract std devs from diagonal
        res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);

        // Validate dimensions
        int expected_dim = is_localized ? 3 : 5;
        if (res.std_devs.size() != expected_dim) {
            qWarning() << "Unexpected covariance dimension:" << res.std_devs.size()
                      << "expected:" << expected_dim;
            throw std::runtime_error("Dimension mismatch in uncertainty estimation");
        }

    } catch (const std::exception& e) {
        qWarning() << "Uncertainty estimation failed:" << e.what();

        // Fallback based on mode
        if (is_localized) {
            // LOCALIZED: 3x3 for robot pose only
            res.covariance = torch::eye(3, torch::kFloat32) * 0.1f;
            res.std_devs = {0.3f, 0.3f, 0.1f};  // Conservative: 30cm position, 0.1 rad orientation
        } else {
            // MAPPING: 5x5 for room + robot
            res.covariance = torch::eye(5, torch::kFloat32) * 0.1f;
            res.std_devs = {0.1f, 0.1f, 0.3f, 0.3f, 0.1f};  // Conservative estimates
        }
        res.uncertainty_valid = false;
    }

    res.final_loss = final_loss;
    return res;
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

void RoomOptimizer::update_state_management(
    const Result &res,
    RoomModel &room,
    bool is_localized,
    float final_loss)
{
    // Store current covariance and pose for next prediction
    uncertainty_manager.set_previous_cov(res.covariance);
    uncertainty_manager.set_previous_pose(room.get_robot_pose());

    // Update freezing manager
    std::vector<float> room_std_devs;
    std::vector<float> robot_std_devs;

    if (is_localized) {
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
}

// ============================================================================
// HELPER METHODS
// ============================================================================

torch::Tensor RoomOptimizer::convert_points_to_tensor(const RoboCompLidar3D::TPoints &points)
{
    std::vector<float> points_data;
    points_data.reserve(points.size() * 2);

    for (const auto& p : points) {
        points_data.push_back(p.x / 1000.0f);  // Convert mm to meters
        points_data.push_back(p.y / 1000.0f);
    }

    return torch::from_blob(
        points_data.data(),
        {static_cast<long>(points.size()), 2},
        torch::kFloat32
    ).clone();
}

Eigen::Vector3f RoomOptimizer::integrate_velocity_over_window(
    const RoomModel& room,
    const boost::circular_buffer<VelocityCommand> &velocity_history,
    const std::chrono::time_point<std::chrono::high_resolution_clock> &t_start,
    const std::chrono::time_point<std::chrono::high_resolution_clock> &t_end)
{
    Eigen::Vector3f total_delta = Eigen::Vector3f::Zero();

    const auto current_pose = room.get_robot_pose();
    float running_theta = current_pose[2];

    // Integrate over all velocity commands in [t_start, t_end]
    for (size_t i = 0; i < velocity_history.size(); ++i) {
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

OdometryPrior RoomOptimizer::compute_odometry_prior(
    const RoomModel &room,
    const boost::circular_buffer<VelocityCommand>& velocity_history,
    const std::chrono::time_point<std::chrono::high_resolution_clock> &lidar_timestamp)
{
    OdometryPrior prior;
    prior.valid = false;

    // Check 1: First frame (no previous timestamp)
    if (last_lidar_timestamp == std::chrono::time_point<std::chrono::high_resolution_clock>{}) {
        last_lidar_timestamp = lidar_timestamp;
        return prior;
    }

    // Check 2: Empty velocity history
    if (velocity_history.empty()) {
        return prior;
    }

    // Check 3: Time delta between LiDAR scans
    const float dt = std::chrono::duration<float>(lidar_timestamp - last_lidar_timestamp).count();

    if (dt <= 0) {
        qWarning() << "Invalid dt <= 0 (" << dt << "s) - timestamp not advancing!";
        last_lidar_timestamp = lidar_timestamp;
        return prior;
    }

    if (dt > 0.5f) {
        qWarning() << "Large dt (" << dt << "s) - possible frame skip";
        last_lidar_timestamp = lidar_timestamp;
        return prior;
    }

    // Integrate velocity over the time window
    prior.delta_pose = integrate_velocity_over_window(room, velocity_history,
                                                      last_lidar_timestamp, lidar_timestamp);

    // Store last velocity command
    if (!velocity_history.empty()) {
        prior.velocity_cmd = velocity_history.back();
    }

    prior.dt = dt;

    // Compute process noise - proportional to motion magnitude
    const float trans_magnitude = std::sqrt(
        prior.delta_pose[0] * prior.delta_pose[0] +
        prior.delta_pose[1] * prior.delta_pose[1]
    );
    const float rot_magnitude = std::abs(prior.delta_pose[2]);

    float trans_noise_std = prediction_params.NOISE_TRANS * trans_magnitude;
    float rot_noise_std = prediction_params.NOISE_ROT * rot_magnitude;

    // Minimum uncertainty
    trans_noise_std = std::max(trans_noise_std, 0.01f);
    rot_noise_std = std::max(rot_noise_std, 0.02f);

    // Extra uncertainty for pure rotation
    if (trans_magnitude < 0.01f && rot_magnitude > 0.01f) {
        rot_noise_std *= 2.0f;
        trans_noise_std = 0.05f;
    }

    prior.covariance = torch::eye(3, torch::kFloat32);
    prior.covariance[0][0] = trans_noise_std * trans_noise_std;
    prior.covariance[1][1] = trans_noise_std * trans_noise_std;
    prior.covariance[2][2] = rot_noise_std * rot_noise_std;

    prior.valid = true;
    last_lidar_timestamp = lidar_timestamp;

    return prior;
}