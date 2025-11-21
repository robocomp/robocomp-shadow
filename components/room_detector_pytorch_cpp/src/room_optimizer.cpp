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

    // ===== EKF PREDICT STEP =====
    // Propagates state: x_pred = x_prev + f(u, dt)
    // Propagates covariance: P_pred = F*P*F^T + Q
    // Sets room.robot_pos_ = x_pred (initialization for optimization)
    const PredictionState prediction = predict_step(room, odometry_prior, is_localized);

    // ===== FILTER MEASUREMENTS (TOP-DOWN PREDICTION) =====
    // Optional: filter outliers based on model fit
    auto filter_result = filter_measurements(points_r, room, prediction);
    // Could use filtered points: auto points = filter_result.filtered_set;
    const auto points = points_r;  // For now, use all points

    // ===== CONVERT MEASUREMENTS TO TENSOR =====
    const torch::Tensor points_tensor = convert_points_to_tensor(points);

    auto pose_before_opt = room.get_robot_pose();

    // ===== EKF UPDATE STEP =====
    // Optimization-based update: x_new = argmin { L(z|x) + ||x - x_pred||²_P }
    // - Starts from x_pred (set in predict_step)
    // - Measurement term: L(z|x) pulls toward LiDAR fit
    // - Prior term: ||x - x_pred||² pulls toward prediction
    // - Result: x_new = x_pred + K*(innovation), where K is implicit in optimization
    Result res = update_step(points_tensor, room, odometry_prior, prediction, is_localized,
                             num_iterations, min_loss_threshold, learning_rate, time_series_plotter);

    // Store the prior for debugging
    res.prior = odometry_prior;

    // ===== COMPUTE EKF RESIDUAL (Innovation) =====
    // residual = x_optimized - x_predicted
    // This is the "innovation" in EKF terminology: where measurement pulled us
    // Used for long-term calibration learning
    res.optimized_pose = room.get_robot_pose();

    // ===== UPDATE STATE MANAGEMENT =====
    // Use measurement loss only for freezing decisions (prior loss is intentionally high when preventing drift)
    float measurement_loss_for_freezing = res.final_loss - res.prior_loss;
    update_state_management(res, room, is_localized, measurement_loss_for_freezing);

    // ===== ADAPTIVE PRIOR WEIGHTING =====
    // Compute innovation from this frame to adjust prior weight for next frame
    if (is_localized && prediction.have_propagated && odometry_prior.valid)
    {
        // Final innovation: how far did optimization move from prediction?
        float innovation_x = res.optimized_pose[0] - prediction.predicted_pose[0];
        float innovation_y = res.optimized_pose[1] - prediction.predicted_pose[1];
        float innovation_theta = res.optimized_pose[2] - prediction.predicted_pose[2];

        Eigen::Vector3f innovation(innovation_x, innovation_y, innovation_theta);

        // Use consistent motion covariance helper
        Eigen::Matrix3f cov_eigen = compute_motion_covariance(odometry_prior);

        Eigen::Matrix3f inv_cov = (cov_eigen + 1e-6f * Eigen::Matrix3f::Identity()).inverse();
        float innovation_norm = std::sqrt(innovation.transpose() * inv_cov * innovation);

        // Check if robot was stationary
        float motion_magnitude = std::sqrt(
            odometry_prior.delta_pose[0] * odometry_prior.delta_pose[0] +
            odometry_prior.delta_pose[1] * odometry_prior.delta_pose[1]
        );

        // Update adaptive weight for NEXT frame
        float new_weight;

        // Special case: when stationary, force strong prior immediately
        if (motion_magnitude < 0.01f) {
            // Stationary: Set weight to 1.0 directly (no EMA smoothing)
            new_weight = 1.0f;
            prior_weight_ = 1.0f;  // Force it directly, bypass EMA
        } else if (innovation_norm < 1.0f) {
            // Within 1-sigma: trust prior strongly
            new_weight = 1.0f;
            // Use EMA for smooth transition
            float alpha = 0.3f;
            prior_weight_ = alpha * new_weight + (1.0f - alpha) * prior_weight_;
        } else if (innovation_norm < 3.0f) {
            // 1-3 sigma: reduce weight linearly
            new_weight = 1.0f / (1.0f + (innovation_norm - 1.0f));
            float alpha = 0.3f;
            prior_weight_ = alpha * new_weight + (1.0f - alpha) * prior_weight_;
        } else {
            // > 3-sigma: very low weight (likely odometry error)
            new_weight = 0.1f;
            float alpha = 0.3f;
            prior_weight_ = alpha * new_weight + (1.0f - alpha) * prior_weight_;
        }

        float old_weight = prior_weight_;  // For diagnostics

    }

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

    // ===== EKF PREDICT: Initialize model to predicted pose =====
    // This sets the starting point for optimization
    // Update model to predicted pose (for LOCALIZED mode)
    if (is_localized)
    {
        // CRITICAL: Save previous pose BEFORE computing prediction
        const auto current_pose = room.get_robot_pose();
        prediction.previous_pose = current_pose;  // Store for prior loss computation

        // Predict using the integrated delta (more reliable than velocity commands)
        torch::Tensor current_pose_tensor = torch::tensor(
            {current_pose[0], current_pose[1], current_pose[2]},
            torch::kFloat32
        );

        torch::Tensor predicted_tensor = room.predict_pose_from_delta(
            current_pose_tensor,
            odometry_prior.delta_pose
        );

        // Extract predicted pose
        prediction.predicted_pose = {
            predicted_tensor[0].item<float>(),
            predicted_tensor[1].item<float>(),
            predicted_tensor[2].item<float>()
        };

        // Initialize model to prediction (starting point for optimization)
        // Update model parameters (bypassing autograd with .data())
        room.robot_pos_.data()[0] = prediction.predicted_pose[0];
        room.robot_pos_.data()[1] = prediction.predicted_pose[1];
        room.robot_theta_.data()[0] = prediction.predicted_pose[2];

        // ===== CHECK IF ROBOT IS STATIONARY =====
        // If robot hasn't moved, force strong prior to prevent drift
        float motion_magnitude = std::sqrt(
            odometry_prior.delta_pose[0] * odometry_prior.delta_pose[0] +
            odometry_prior.delta_pose[1] * odometry_prior.delta_pose[1]
        );

        // Force strong prior immediately when stationary (bypass EMA smoothing)
        if (motion_magnitude < 0.01f) {  // < 10mm = stationary
            prior_weight_ = 1.0f;  // Set directly, no smoothing
        }
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
    const PredictionState &prediction,
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

    // Save pose before optimization for diagnostics
    auto pose_before_opt = room.get_robot_pose();

    // Prepare odometry prior for optimization
    bool use_odometry_prior = odometry_prior.valid && is_localized && calib_config.enable_odometry_optimization;
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
        // AND dt must be at least 50ms (avoid noisy short intervals)
        const float min_translation = 0.02f;  // 20mm
        const float min_rotation = 0.035f;     // ~2 degrees
        const float min_dt = 0.05f;             // 50ms (20Hz) - typical LiDAR rate

        has_meaningful_motion = (trans_magnitude > min_translation || rot_magnitude > min_rotation)
                              && odometry_prior.dt > min_dt;

        // Enable odometry prior even for small motions to prevent drift
        // The prior will use the minimum floor covariance for small motions
        if (!has_meaningful_motion)
        {
            if (odometry_prior.dt < min_dt)
            {
                use_odometry_prior = false;  // Disable only if dt too small
            }
        }

        // Get TRUE previous pose (BEFORE prediction was applied)
        // CRITICAL FIX: room.get_robot_pose() returns PREDICTED pose at this point
        // Use prediction.previous_pose which was saved in predict_step()
        if (!prediction.previous_pose.empty()) {
            prev_pose_tensor = torch::tensor(
                {prediction.previous_pose[0], prediction.previous_pose[1], prediction.previous_pose[2]},
                torch::kFloat32
            ).requires_grad_(false);
        } else {
            // Fallback: use current (predicted) pose if previous not available
            auto prev_pose = room.get_robot_pose();
            prev_pose_tensor = torch::tensor(
                {prev_pose[0], prev_pose[1], prev_pose[2]},
                torch::kFloat32
            ).requires_grad_(false);
        }

        // Note: predicted_pose will be computed INSIDE the optimization loop
        // to allow gradients to flow through calibration parameters
    }

    // Run optimization loop
    torch::optim::Adam optimizer(params_to_optimize, torch::optim::AdamOptions(learning_rate));

    auto opt_result = run_optimization_loop(
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

    // === DIAGNOSTIC: Movement check ===

    // Estimate uncertainty after optimization
    // Pass the propagated covariance (after motion growth) for proper EKF update
    res = estimate_uncertainty(room, points_tensor, is_localized, opt_result.final_loss,
                               prediction.have_propagated ? prediction.propagated_cov : torch::Tensor());
    res.prior_loss = opt_result.final_prior_loss;  // Store for calibration learning

    return res;
}

std::vector<torch::Tensor> RoomOptimizer::select_optimization_parameters(
    RoomModel &room,
    bool is_localized)
{
    std::vector<torch::Tensor> params_to_optimize;

    if (is_localized) {
        // LOCALIZED: Optimize robot pose only
        params_to_optimize = room.get_robot_parameters();
        room.freeze_room_parameters();
    } else {
        // MAPPING: Optimize everything (room + robot)
        params_to_optimize = room.parameters();
        room.unfreeze_room_parameters();
    }

    return params_to_optimize;
}

RoomOptimizer::OptimizationResult RoomOptimizer::run_optimization_loop(
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
    OptimizationResult result;
    float final_prior_loss = 0.0f;

    torch::Tensor calib_reg;
    torch::Tensor prior_loss;
    for (int iter = 0; iter < num_iterations; ++iter)
    {
        optimizer.zero_grad();

        // ===== MEASUREMENT LIKELIHOOD: p(z|x) =====
        // SDF-based loss: how well robot pose explains LiDAR measurements
        torch::Tensor measurement_loss = compute_measurement_loss(points_tensor, room);
        torch::Tensor total_loss = measurement_loss;

        // ===== MOTION PRIOR: p(x|x_pred) =====
        // Add prior loss if using odometry
        if (use_odometry_prior)
        {
            // Compute predicted pose using the integrated delta approach
            // Convert prev_pose to tensor and add integrated delta
            auto prev_pose_vec = prev_pose_tensor.accessor<float, 1>();
            torch::Tensor predicted_pose = torch::tensor({
                prev_pose_vec[0] + odometry_prior.delta_pose[0],
                prev_pose_vec[1] + odometry_prior.delta_pose[1],
                prev_pose_vec[2] + odometry_prior.delta_pose[2]
            }, torch::kFloat32).requires_grad_(false);

            prior_loss = compute_prior_loss(room, predicted_pose, odometry_prior);
            total_loss = total_loss + prior_weight_ * prior_loss;
        }

        // ===== OPTIMIZATION STEP (Implicit Kalman Gain) =====
        // In standard EKF: x_new = x_pred + K*(z - h(x_pred))
        // Here: x_new = argmin { ||z - h(x)||² + ||x - x_pred||²_P }
        // The balance between terms acts like Kalman gain

        // Backward pass
        total_loss.backward();

        // Clip gradients for stability (more lenient to allow calibration learning)
        torch::nn::utils::clip_grad_norm_(room.parameters(), 5.0);

        optimizer.step();

        // Enforce calibration bounds
        auto calib = room.get_odometry_calibration();
        bool hit_bounds = false;

        if (calib[0] < calib_config.min_value || calib[0] > calib_config.max_value ||
            calib[1] < calib_config.min_value || calib[1] > calib_config.max_value)
        {

            // Only warn on first iteration when hitting bounds
            if (iter == 0) {
                if (calib[0] <= calib_config.min_value && room.k_translation_.grad().item<float>() > 0.1f)
                {
                    qWarning() << "⚠️  k_translation at MIN bound (" << calib_config.min_value
                              << ") - likely units mismatch";
                    hit_bounds = true;
                }

                if (calib[0] >= calib_config.max_value && room.k_translation_.grad().item<float>() < -0.1f)
                {
                    qWarning() << "⚠️  k_translation at MAX bound (" << calib_config.max_value
                              << ") - likely units mismatch";
                    hit_bounds = true;
                }
            }

            room.k_translation_.data().clamp_(calib_config.min_value, calib_config.max_value);
            room.k_rotation_.data().clamp_(calib_config.min_value, calib_config.max_value);
        }

        if (hit_bounds && iter == 0)
        {
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

        result.final_loss = total_loss.item<float>();
        if (prior_loss.defined() && prior_loss.numel() > 0)
        {
            final_prior_loss = prior_loss.item<float>();
        }

        // Plot if requested
        if (time_series_plotter && iter % 5 == 0)
        {
            time_series_plotter->addDataPoint(0, measurement_loss.item<float>());
            time_series_plotter->addDataPoint(2, result.final_loss);
            if (prior_loss.defined() && prior_loss.numel() > 0)
            {
                time_series_plotter->addDataPoint(1, prior_loss.item<float>());
            }
        }

        // Early stopping
        // Don't stop early when using odometry prior - need multiple iterations
        // for prior loss to constrain the pose (gradient is 0 at iter 0)
        if (!use_odometry_prior && result.final_loss < min_loss_threshold) {
            break;
        }
    }

    result.final_prior_loss = final_prior_loss;
    return result;
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

    // Use moderate fixed inflation
    // Lower value = tighter prior constraint = less drift
    float inflation = 1.0f;  // No inflation - trust the computed covariance

    // Compute motion-based covariance using consistent helper method
    Eigen::Matrix3f cov_eigen = compute_motion_covariance(odometry_prior);

    // Convert to torch tensor
    torch::Tensor reg_cov = torch::zeros({3, 3}, torch::kFloat32);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            reg_cov[i][j] = cov_eigen(i, j) * inflation;
        }
    }
    reg_cov = reg_cov + 1e-6 * torch::eye(3);

    torch::Tensor info_matrix = torch::inverse(reg_cov);

    torch::Tensor prior_loss = 0.5 * torch::matmul(
        pose_diff.unsqueeze(0),
        torch::matmul(info_matrix, pose_diff.unsqueeze(1))
    ).squeeze();

    return prior_loss;
}

torch::Tensor RoomOptimizer::compute_calibration_regularization(RoomModel &room)
{
    // CALIBRATION REGULARIZATION DISABLED
    return torch::zeros(1, torch::kFloat32);
}

// ============================================================================
// UNCERTAINTY ESTIMATION
// ============================================================================

RoomOptimizer::Result RoomOptimizer::estimate_uncertainty(
    RoomModel &room,
    const torch::Tensor &points_tensor,
    bool is_localized,
    float final_loss,
    const torch::Tensor &propagated_cov)
{
    Result res;

    // For LOCALIZED mode with odometry prior, use the propagated covariance
    // (which already includes motion growth) and apply measurement update

    if (is_localized && propagated_cov.defined() && propagated_cov.numel() > 0) {
        // Use the propagated covariance (AFTER motion growth, BEFORE measurement update)
        torch::Tensor P_predicted = propagated_cov;

        // Estimate measurement information from fit quality
        // final_loss ∈ [0.0005, 0.01] typically
        // Map to information gain factor
        float normalized_loss = std::clamp(final_loss / 0.01f, 0.0f, 1.0f);

        // Information gain: good fit (low loss) → high gain, poor fit → low gain
        float information_gain = 0.1f + 0.4f * (1.0f - normalized_loss);  // Range: [0.1, 0.5]

        // EKF update: P_new = (1 - information_gain) * P_predicted
        // This means measurements reduce uncertainty, but not to zero
        res.covariance = P_predicted * (1.0f - information_gain);

        // Enforce minimum uncertainty to prevent degeneracy
        // Even with perfect measurements, keep at least 1mm position std, 0.1° orientation std
        auto cov_acc = res.covariance.accessor<float, 2>();
        cov_acc[0][0] = std::max(cov_acc[0][0], 1e-6f);  // min 1mm std
        cov_acc[1][1] = std::max(cov_acc[1][1], 1e-6f);  // min 1mm std
        cov_acc[2][2] = std::max(cov_acc[2][2], 3e-6f);  // min ~0.1° std

        res.uncertainty_valid = true;
        res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);
        res.final_loss = final_loss;
        return res;
    }

    // For MAPPING mode, use Hessian-based uncertainty estimation
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
        qWarning() << "Uncertainty estimation failed:" << e.what() << "- using motion-based fallback";

        // Use motion-based uncertainty estimate instead of fixed values
        // This gives more realistic uncertainty that grows with motion
        if (is_localized) {
            // LOCALIZED: 3x3 for robot pose only
            // Base uncertainty on recent motion magnitude
            float base_pos_std = 0.05f;   // 5cm base
            float base_ang_std = 0.05f;   // ~3 degrees base

            res.covariance = torch::eye(3, torch::kFloat32);
            res.covariance[0][0] = base_pos_std * base_pos_std;
            res.covariance[1][1] = base_pos_std * base_pos_std;
            res.covariance[2][2] = base_ang_std * base_ang_std;

            res.std_devs = {base_pos_std, base_pos_std, base_ang_std};
        } else {
            // MAPPING: 5x5 for room + robot
            res.covariance = torch::eye(5, torch::kFloat32);
            res.covariance[0][0] = 0.01f;  // Room half-width: 10cm std
            res.covariance[1][1] = 0.01f;  // Room half-height: 10cm std
            res.covariance[2][2] = 0.0025f; // Robot x: 5cm std
            res.covariance[3][3] = 0.0025f; // Robot y: 5cm std
            res.covariance[4][4] = 0.0025f; // Robot theta: 5cm std (0.05 rad)

            res.std_devs = {0.1f, 0.1f, 0.05f, 0.05f, 0.05f};
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

    // Compute process noise using consistent helper method
    Eigen::Matrix3f cov_eigen = compute_motion_covariance(prior);

    // Convert to torch tensor
    prior.covariance = torch::eye(3, torch::kFloat32);
    prior.covariance[0][0] = cov_eigen(0, 0);
    prior.covariance[1][1] = cov_eigen(1, 1);
    prior.covariance[2][2] = cov_eigen(2, 2);

    prior.valid = true;
    last_lidar_timestamp = lidar_timestamp;

    return prior;
}

// ===== HELPER METHOD: Compute motion-based covariance =====
/**
 * Compute motion-based covariance consistently
 * σ = base + k * distance
 */
Eigen::Matrix3f RoomOptimizer::compute_motion_covariance(const OdometryPrior &odometry_prior)
{
    float motion_magnitude = std::sqrt(
        odometry_prior.delta_pose[0] * odometry_prior.delta_pose[0] +
        odometry_prior.delta_pose[1] * odometry_prior.delta_pose[1]
    );

    // Proper motion model: uncertainty grows with distance
    // BUT: when stationary, use much tighter uncertainty to prevent drift
    float base_uncertainty;
    if (motion_magnitude < 0.01f) {
        // Stationary: Very tight constraint (1mm)
        base_uncertainty = 0.001f;  // 1mm when not moving
    } else {
        // Moving: Normal base uncertainty
        base_uncertainty = 0.005f;  // 5mm base when moving
    }

    float noise_per_meter = prediction_params.NOISE_TRANS;  // Use configured value
    float position_std = base_uncertainty + noise_per_meter * motion_magnitude;

    float base_rot_std = 0.01f;  // 10 mrad base
    float noise_per_radian = prediction_params.NOISE_ROT;
    float rotation_std = base_rot_std + noise_per_radian * std::abs(odometry_prior.delta_pose[2]);

    Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();
    cov(0, 0) = position_std * position_std;
    cov(1, 1) = position_std * position_std;
    cov(2, 2) = rotation_std * rotation_std;

    return cov;
}

// ============================================================================
// LONG-TERM CALIBRATION LEARNING
// ============================================================================

void RoomOptimizer::update_calibration_slowly(
    RoomModel &room,
    const OdometryPrior &odometry_prior,
    const std::vector<float> &predicted_pose,
    const std::vector<float> &optimized_pose,
    float prior_loss)
{
    // CALIBRATION LEARNING DISABLED
    // Using fixed k_trans = 1.0, k_rot = 1.0
    return;
}