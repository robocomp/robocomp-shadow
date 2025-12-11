//
// RoomConcept - EKF-style predict-update cycle implementation
//

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "room_concept.h"
#include "room_loss.h"
#include "uncertainty_estimator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// MAIN EKF CYCLE
// ============================================================================

namespace rc
{
    RoomConcept::Result RoomConcept::update(
        const TimePoints &points_,
        const VelocityHistory &velocity_history,
        int num_iterations,
        float min_loss_threshold,
        float learning_rate)
    {
        const auto &[points_r, lidar_timestamp] = points_;
        if (points_r.empty()) {
            qWarning() << "No points to optimize";
            return {};
        }

        if (room == nullptr)
        {
            room = std::make_shared<RoomModel>();
            room->init(std::get<RoboCompLidar3D::TPoints>(points_));
            room->init_odometry_calibration(1.0f, 1.0f); // Start with no correction
            qInfo() << "RoomConcept::update() - RoomModel initialized";
        }

        // Check if in LOCALIZED mode
        const bool is_localized = room_freezing_manager.should_freeze_room();

        // Compute odometry prior between lidar timestamps
        auto odometry_prior = compute_odometry_prior(room, velocity_history, points_);

        // ===== PREDICT STEP =====
        // Propagates state: x_pred = x_prev + f(u, dt)
        // Propagates covariance: P_pred = F*P*F^T + Q where F is motion model Jacobian
        // Sets room->robot_pos_ = x_pred (initialization for optimization)
        const PredictionState prediction = predict_step(room, odometry_prior, is_localized);

        // ===== FILTER MEASUREMENTS (TOP-DOWN PREDICTION) =====
        // Optional: filter outliers based on model fit
        auto filter_result = filter_measurements(points_r, room, prediction);
        // Could use filtered points: auto points = filter_result.filtered_set;
        const auto points = points_r;  // For now, use all points

        // ===== CONVERT MEASUREMENTS TO TENSOR =====
        const torch::Tensor points_tensor = convert_points_to_tensor(points);
        auto pose_before_opt = room->get_robot_pose();

        // ===== EKF UPDATE STEP =====
        // - Optimization-based update: x_new = argmin { L(z|x) + ||x - x_pred||²_P }. Likelihood + prior/P
        // - Starts from x_pred (set in predict_step)
        // - Measurement term: L(z|x) pulls toward LiDAR fit
        // - Prior term: ||x - x_pred||² pulls toward prediction
        // - Result: x_new = x_pred + K*(innovation), where K is implicit in optimization
        Result res = update_step(points_tensor, room, odometry_prior, prediction, is_localized,
                                 num_iterations, min_loss_threshold, learning_rate);

        // Store the prior for debugging
        res.prior = odometry_prior;

        // ===== COMPUTE EKF RESIDUAL (Innovation) =====
        // residual = x_optimized - x_predicted
        // This is the "innovation" in EKF terminology: where measurement pulled us
        // Used for long-term calibration learning
        res.optimized_pose = room->get_robot_pose();

        // ===== UPDATE STATE MANAGEMENT =====
        // Use measurement loss only for freezing decisions (prior loss is intentionally high when preventing drift)
        float measurement_loss_for_freezing = res.final_loss - res.prior_loss;
        update_state_management(res, room, is_localized, measurement_loss_for_freezing);

        // ===== PREDICT REAL-TIME POSE (LATENCY COMPENSATION) =====
        if (!velocity_history.empty())
        {
            // Get timestamp of the very latest command (effectively "NOW")
            auto t_latest = velocity_history.back().timestamp;

            // Only predict forward
            if (t_latest > lidar_timestamp)
            {
                // Integrate velocity from t_lidar (Optimized Time) -> t_latest (Real Time)
                // Note: 'room' currently holds the optimized pose at t_lidar.
                // Your integrate_velocity_over_window uses room->get_robot_pose() to start,
                // so it correctly accumulates from the optimized orientation.
                Eigen::Vector3f motion_lag = integrate_velocity_over_window(
                    room,
                    velocity_history,
                    lidar_timestamp,
                    t_latest
                );

                // Since integrate_velocity_over_window returns GLOBAL displacement
                // (it applies cos/sin internally), we simply add it.
                float x_pred = res.optimized_pose[0] + motion_lag.x();
                float y_pred = res.optimized_pose[1] + motion_lag.y();
                float t_pred = res.optimized_pose[2] + motion_lag.z();

                // Normalize theta
                while (t_pred > M_PI) t_pred -= 2 * M_PI;
                while (t_pred < -M_PI) t_pred += 2 * M_PI;

                res.predicted_realtime_pose = {x_pred, y_pred, t_pred};
            }
            else
            {
                // No future data available (rare), fallback to optimized
                res.predicted_realtime_pose = res.optimized_pose;
            }
        }
        else
        {
            res.predicted_realtime_pose = res.optimized_pose;
        }

        // ===== ADAPTIVE PRIOR WEIGHTING =====
        // Compute innovation
        Eigen::Matrix3f cov_eigen = Eigen::Matrix3f::Zero();
        float innovation_norm = 0.0f;
        float motion_magnitude = 0.0f;
        if (is_localized and prediction.have_propagated and odometry_prior.valid)
        {
            Eigen::Vector3f innovation(
                res.optimized_pose[0] - prediction.predicted_pose[0],
                res.optimized_pose[1] - prediction.predicted_pose[1],
                res.optimized_pose[2] - prediction.predicted_pose[2]
            );
            cov_eigen = Eigen::Matrix3f::Zero();
            if (prediction.have_propagated && prediction.propagated_cov.size(0) == 3)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        cov_eigen(i, j) = prediction.propagated_cov[i][j].item<float>();
            else // Fallback
                cov_eigen = compute_motion_covariance(odometry_prior);

            Eigen::Matrix3f inv_cov = (cov_eigen + 1e-6f * Eigen::Matrix3f::Identity()).inverse();
            innovation_norm = std::sqrt(innovation.transpose() * inv_cov * innovation);
            motion_magnitude = std::sqrt(std::hypot(odometry_prior.delta_pose[0], odometry_prior.delta_pose[1]));

            // Principled adaptive weight
            prior_weight = compute_adaptive_prior_weight(
                innovation_norm,
                motion_magnitude,
                cov_eigen,
                prior_weight);
        }
        else prior_weight = 1.0f; // No adaptation in MAPPING mode

        res.prior.prior_weight = prior_weight;
        res.innovation_norm = innovation_norm;
        res.motion_magnitude = motion_magnitude;
        res.prediction_state = prediction;
        res.state = room_freezing_manager.get_state();
        res.timestamp = std::chrono::high_resolution_clock::now();
        room->state_ = res.state;

        // ===== FINALIZE RESULT =====
        frame_number++;
        return res;
    }

    // ============================================================================
    // PREDICT PHASE
    // ============================================================================

    RoomConcept::PredictionState RoomConcept::predict_step(
        std::shared_ptr<RoomModel> &room,
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

        // Propagate covariance using integrated delta_pose
        // This is correct even with multiple velocity commands in the window
        prediction.propagated_cov = uncertainty_manager.propagate_with_delta_pose( odometry_prior.delta_pose,  // Already integrated correctly
                                                                                   prev_cov,
                                                                                   is_localized);

        // Propagate covariance using velocity command
        /*prediction.propagated_cov = uncertainty_manager.propagate_with_velocity(
            odometry_prior.velocity_cmd,
            odometry_prior.dt,
            prev_cov,
            is_localized
        );*/
        prediction.have_propagated = true;

        // ===== EKF PREDICT: Initialize model to predicted pose =====
        // This sets the starting point for optimization
        // Update model to predicted pose (for LOCALIZED mode)
        if (is_localized)
        {
            // CRITICAL: Save previous pose BEFORE computing prediction
            const auto current_pose = room->get_robot_pose();
            prediction.previous_pose = current_pose;  // Store for prior loss computation

            // Predict using the integrated delta (more reliable than velocity commands)
            torch::Tensor current_pose_tensor = torch::tensor(
                {current_pose[0], current_pose[1], current_pose[2]},
                torch::kFloat32
            );

            torch::Tensor predicted_tensor = room->predict_pose_from_delta(
                current_pose_tensor,
                odometry_prior.delta_pose
            );

            // Extract predicted pose
            const auto predicted_acc = predicted_tensor.accessor<float, 1>();
            prediction.predicted_pose = {predicted_acc[0], predicted_acc[1], predicted_acc[2]};

            // Initialize model to prediction (starting point for optimization)
            // Update model parameters (bypassing autograd with .data())
            room->robot_pos_.data()[0] = prediction.predicted_pose[0];
            room->robot_pos_.data()[1] = prediction.predicted_pose[1];
            room->robot_theta_.data()[0] = prediction.predicted_pose[2];

        }

        return prediction;
    }

    ModelBasedFilter::Result RoomConcept::filter_measurements(
        const RoboCompLidar3D::TPoints &points,
       std::shared_ptr<RoomModel> &room,
        const PredictionState &prediction)
    {
        const bool is_localized = room_freezing_manager.should_freeze_room();

        if (!is_localized) {
            return {};  // No filtering in MAPPING mode
        }

        // Get robot uncertainty from predicted covariance
        float robot_uncertainty = 0.0f;
        if (prediction.have_propagated && prediction.propagated_cov.size(0) == 3)
        {
            const auto cov_acc = prediction.propagated_cov.accessor<float, 2>();
            robot_uncertainty = std::sqrt(cov_acc[0][0] + cov_acc[1][1]);
        }

        // Filter points based on model fit
        auto filter_result = model_based_filter.filter(points, room, robot_uncertainty);
        //model_based_filter.print_result(points.size(), filter_result);

        return filter_result;
    }

    // ============================================================================
    // UPDATE PHASE
    // ============================================================================

    RoomConcept::Result RoomConcept::update_step(
        const torch::Tensor &points_tensor,
       std::shared_ptr<RoomModel> &room,
        const OdometryPrior &odometry_prior,
        const PredictionState &prediction,
        bool is_localized,
        int num_iterations,
        float min_loss_threshold,
        float learning_rate)
    {
        Result res;
        res.prior = odometry_prior;

        // Select parameters to optimize based on mode
        auto params_to_optimize = select_optimization_parameters(room, is_localized);

        // Save pose before optimization for diagnostics
        auto pose_before_opt = room->get_robot_pose();

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
            if (not has_meaningful_motion)
                if (odometry_prior.dt < min_dt)
                    use_odometry_prior = false;  // Disable only if dt too small

            // Get TRUE previous pose (BEFORE prediction was applied)
            // CRITICAL FIX: room->get_robot_pose() returns PREDICTED pose at this point
            // Use prediction.previous_pose which was saved in predict_step()
            if (not prediction.previous_pose.empty())
            {
                prev_pose_tensor = torch::tensor(
                    {prediction.previous_pose[0], prediction.previous_pose[1], prediction.previous_pose[2]},
                    torch::kFloat32
                ).requires_grad_(false);
            } else {
                // Fallback: use current (predicted) pose if previous not available
                auto prev_pose = room->get_robot_pose();
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
            prediction,
            use_odometry_prior,
            num_iterations,
            min_loss_threshold
        );

        res.optimized_pose = room->get_robot_pose();

        // === JUMP GUARD ===
        // If the optimizer moved the robot > 20cm or > 45deg from prediction, REJECT IT.
        // This prevents the 180 degree flip.

        std::vector<float> ref_pose;
        if (prediction.predicted_pose.size() == 3) ref_pose = prediction.predicted_pose;
        else if (prediction.previous_pose.size() == 3) ref_pose = prediction.previous_pose;

        if (ref_pose.size() == 3)
        {
            float dx = res.optimized_pose[0] - ref_pose[0];
            float dy = res.optimized_pose[1] - ref_pose[1];
            float dtheta = res.optimized_pose[2] - ref_pose[2];

            // Normalize angle
            while (dtheta > M_PI) dtheta -= 2 * M_PI;
            while (dtheta < -M_PI) dtheta += 2 * M_PI;

            if (std::sqrt(dx*dx + dy*dy) > 0.2f || std::abs(dtheta) > 0.78f) // 0.2m, 45deg
            {
                qWarning() << "RoomConcept: REJECTED OPTIMIZATION JUMP. Resetting to prediction."
                           << "dTheta:" << dtheta;

                // Revert to prediction
                room->robot_pos_.data()[0] = prediction.predicted_pose[0];
                room->robot_pos_.data()[1] = prediction.predicted_pose[1];
                room->robot_theta_.data()[0] = prediction.predicted_pose[2];
                res.optimized_pose = prediction.predicted_pose;
            }
        }

        // Estimate uncertainty after optimization
        // Pass the propagated covariance (after motion growth) for proper EKF update
        res = estimate_uncertainty(room, points_tensor, is_localized, opt_result.total_loss,
                                   prediction.have_propagated ? prediction.propagated_cov : torch::Tensor());
        res.prior_loss = opt_result.prior_loss;  // Store for calibration learning

        return res;
    }

    std::vector<torch::Tensor> RoomConcept::select_optimization_parameters(
       std::shared_ptr<RoomModel> &room,
        bool is_localized)
    {
        std::vector<torch::Tensor> params_to_optimize;

        if (is_localized) {
            // LOCALIZED: Optimize robot pose only
            params_to_optimize = room->get_robot_parameters();
            room->freeze_room_parameters();
        } else {
            // MAPPING: Optimize everything (room + robot)
            params_to_optimize = room->parameters();
            room->unfreeze_room_parameters();
        }

        return params_to_optimize;
    }

    RoomConcept::OptimizationResult RoomConcept::run_optimization_loop(
        const torch::Tensor &points_tensor,
       std::shared_ptr<RoomModel> &room,
        const torch::Tensor &prev_pose_tensor,
        const OdometryPrior &odometry_prior,
        torch::optim::Optimizer &optimizer,
        const PredictionState &prediction,
        bool use_odometry_prior,
        int num_iterations,
        float min_loss_threshold)
    {
        OptimizationResult result;
        torch::Tensor prior_loss = torch::zeros({}, torch::kFloat32);
        torch::Tensor measurement_loss = torch::zeros({}, torch::kFloat32);
        torch::Tensor total_loss = torch::zeros({}, torch::kFloat32);
        for (int iter = 0; iter < num_iterations; ++iter)
        {
            optimizer.zero_grad();

            // ===== MEASUREMENT LIKELIHOOD: p(z|x) =====
            // SDF-based loss: how well robot pose explains LiDAR measurements
            measurement_loss = RoomLoss::compute_loss(points_tensor, room, wall_thickness);
            total_loss = measurement_loss;

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

                prior_loss = compute_prior_loss(room, predicted_pose, odometry_prior, prediction);
                total_loss = total_loss + prior_weight * prior_loss;
            }

            // ===== OPTIMIZATION STEP (Implicit Kalman Gain) =====
            // In standard EKF: x_new = x_pred + K*(z - h(x_pred))
            // Here: x_new = argmin { ||z - h(x)||² + ||x - x_pred||²_P }
            // The balance between terms acts like Kalman gain

            // Backward pass
            total_loss.backward();

            optimizer.step();

            // Early stopping
            // Don't stop early when using odometry prior - need multiple iterations
            // for prior loss to constrain the pose (gradient is 0 at iter 0)
            if (!use_odometry_prior && total_loss.item<float>() < min_loss_threshold)
                break;
        }

        // Store final losses in Result
        result.total_loss = total_loss.item<float>();
        result.measurement_loss = measurement_loss.item<float>();
        if (prior_loss.defined() and prior_loss.numel() > 0)
            result.prior_loss = prior_loss.item<float>();

        return result;
    }

    // ============================================================================
    // LOSS COMPUTATION
    // ============================================================================

    torch::Tensor RoomConcept::compute_prior_loss(std::shared_ptr<RoomModel> &room,
                                                     const torch::Tensor &predicted_pose,
                                                     const OdometryPrior &odometry_prior,
                                                     const PredictionState &prediction)
    {
        // Build pose difference (connected to computation graph)
        torch::Tensor predicted_pos = predicted_pose.slice(0, 0, 2);     // [x, y]
        torch::Tensor predicted_theta = predicted_pose.slice(0, 2, 3);   // [theta]

        torch::Tensor pos_diff = room->robot_pos_ - predicted_pos;
        torch::Tensor theta_diff = room->robot_theta_ - predicted_theta;
        torch::Tensor pose_diff = torch::cat({pos_diff, theta_diff});

        // Compute motion-based covariance using consistent helper method. Use propagated covariance if available
        torch::Tensor reg_cov;
        if (prediction.propagated_cov.defined() and prediction.propagated_cov.numel() > 0)
            reg_cov = prediction.propagated_cov;
        else
        {
            // Fallback: compute simple covariance
            Eigen::Matrix3f cov_eigen = compute_motion_covariance(odometry_prior);
            reg_cov = torch::zeros({3, 3}, torch::kFloat32);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    reg_cov[i][j] = cov_eigen(i, j);
        }

        reg_cov = reg_cov + 1e-6 * torch::eye(3);
        torch::Tensor info_matrix = torch::inverse(reg_cov);
        torch::Tensor prior_loss = 0.5 * torch::matmul(pose_diff.unsqueeze(0),
                                                       torch::matmul(info_matrix, pose_diff.unsqueeze(1))).squeeze();

        return prior_loss;
    }

    // ============================================================================
    // UNCERTAINTY ESTIMATION
    // ============================================================================

    RoomConcept::Result RoomConcept::estimate_uncertainty(
       std::shared_ptr<RoomModel> &room,
        const torch::Tensor &points_tensor,
        bool is_localized,
        float final_loss,
        const torch::Tensor &propagated_cov)
    {
        Result res;

        // For LOCALIZED mode with odometry prior, use proper EKF information form:
        // Information matrix fusion: Λ_total = Λ_measurement + Λ_prior
        // Then: Σ_total = Λ_total^(-1)

        if (is_localized && propagated_cov.defined() && propagated_cov.numel() > 0) {
            // Store the propagated covariance for visualization
            res.propagated_cov = propagated_cov.clone();

            try
            {
                // Step 1: Compute measurement covariance from Hessian of MEASUREMENT LOSS ONLY
                // This excludes the prior loss to avoid ill-conditioning
                torch::Tensor measurement_cov = UncertaintyEstimator::compute_covariance(
                    points_tensor,
                    room,
                    wall_thickness
                );

                // Check for valid measurement covariance
                if (torch::any(torch::isnan(measurement_cov)).item<bool>() ||
                    torch::any(torch::isinf(measurement_cov)).item<bool>()) {
                    throw std::runtime_error("Measurement covariance contains NaN/Inf");
                    }

                // Step 2: Convert both covariances to information form (inverse)
                // Information matrix = inverse of covariance matrix
                torch::Tensor info_measurement = torch::inverse(measurement_cov);
                torch::Tensor info_prior = torch::inverse(propagated_cov);

                // Step 3: Fuse information (additive in information space)
                // This is the key EKF update: combine prior and measurement information
                // Λ_total = Λ_meas + Λ_prior
                torch::Tensor info_total = info_measurement + info_prior;

                // Step 4: Convert back to covariance
                // Σ_total = Λ_total^(-1)
                res.covariance = torch::inverse(info_total);

                // Apply inflation factor to correct for systematic overconfidence
                res.covariance = res.covariance * calib_config.uncertainty_inflation;

                // Validate final result
                if (torch::any(torch::isnan(res.covariance)).item<bool>() ||
                    torch::any(torch::isinf(res.covariance)).item<bool>()) {
                    throw std::runtime_error("Final covariance contains NaN/Inf");
                    }

                res.uncertainty_valid = true;
                res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);
                res.final_loss = final_loss;

                return res;

            }
            catch (const std::exception& e)
            {
                qWarning() << "EKF information fusion failed:" << e.what()
                          << "- using propagated covariance only";

                // Fallback: Use propagated covariance with conservative inflation
                // This means we trust only the motion model, not the measurements
                res.propagated_cov = propagated_cov.clone();
                res.covariance = propagated_cov * 2.0f;  // Conservative: double the uncertainty
                res.uncertainty_valid = false;
                res.std_devs = UncertaintyEstimator::get_std_devs(res.covariance);
                res.final_loss = final_loss;
                return res;
            }
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
            size_t expected_dim = is_localized ? 3 : 5;
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

    void RoomConcept::update_state_management(
        const Result &res,
        std::shared_ptr<RoomModel> &room,
        bool is_localized,
        float final_loss)
    {
        // Store current covariance and pose for next prediction
        uncertainty_manager.set_previous_cov(res.covariance);
        uncertainty_manager.set_previous_pose(room->get_robot_pose());

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

        const auto room_params = room->get_room_parameters();
        bool state_changed = room_freezing_manager.update(
            room_params,
            room_std_devs,
            robot_std_devs,
            room->get_robot_pose(),
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

    torch::Tensor RoomConcept::convert_points_to_tensor(const RoboCompLidar3D::TPoints &points)
    {
        torch::Tensor points_tensor = torch::empty({static_cast<long>(points.size()), 2}, torch::kFloat32);
        auto accessor = points_tensor.accessor<float, 2>();

        for (size_t i = 0; i < points.size(); ++i)
        {
            accessor[i][0] = points[i].x; // Convert mm to meters
            accessor[i][1] = points[i].y;
        }

        return points_tensor;
    }

    Eigen::Vector3f RoomConcept::integrate_velocity_over_window(
        const std::shared_ptr<RoomModel>& room,
        const boost::circular_buffer<VelocityCommand> &velocity_history,
        const std::chrono::time_point<std::chrono::high_resolution_clock> &t_start,
        const std::chrono::time_point<std::chrono::high_resolution_clock> &t_end)
    {
        Eigen::Vector3f total_delta = Eigen::Vector3f::Zero();

        const auto current_pose = room->get_robot_pose();
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
            const float dx_local = (adv_x * dt);
            const float dy_local = (adv_z * dt);
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

RoomConcept::OdometryPrior RoomConcept::compute_odometry_prior(
        const std::shared_ptr<RoomModel> &room,
        const boost::circular_buffer<VelocityCommand>& velocity_history,
        const TimePoints &points_)
    {
        OdometryPrior prior;
        prior.valid = false;
        const auto &[points, lidar_timestamp] = points_;

        if (last_lidar_timestamp == std::chrono::time_point<std::chrono::high_resolution_clock>{})
        {
            last_lidar_timestamp = lidar_timestamp;
            return prior;
        }

        // Calculate dt
        const float dt = std::chrono::duration<float>(lidar_timestamp - last_lidar_timestamp).count();
        if (dt <= 0 || dt > 0.5f) {
            last_lidar_timestamp = lidar_timestamp;
            return prior;
        }
        prior.dt = dt;

        // 1. Try Lidar Odometry
        // (Move the physics check logic here as discussed previously)
        const auto res = lidar_odometry.update(points, lidar_timestamp);

        bool accept_lidar = false;
        if (res.success) {
            float speed_lin = std::sqrt(res.delta_pose[0]*res.delta_pose[0] +
                                      res.delta_pose[1]*res.delta_pose[1]) / dt;
            float speed_rot = std::abs(res.delta_pose[2]) / dt;
            if (speed_lin < 2.0f && speed_rot < 2.0f) accept_lidar = true;
            else qWarning() << "RoomConcept: REJECTED LidarOdom jump!";
        }

        if (accept_lidar)
        {
            prior.delta_pose = res.delta_pose;
            prior.valid = true;
        }
        else
        {
            // 2. Fallback to Velocity or Zero
            if (!velocity_history.empty()) {
                prior.delta_pose = integrate_velocity_over_window(room, velocity_history,
                                                    last_lidar_timestamp, lidar_timestamp);
            } else {
                // If no history, assume STATIONARY (Zero motion)
                // This protects us when sitting still!
                prior.delta_pose = Eigen::Vector3f::Zero();
            }
            prior.valid = true; // ALWAYS valid now
        }

        // Compute covariance
        Eigen::Matrix3f cov_eigen = compute_motion_covariance(prior);
        prior.covariance = torch::eye(3, torch::kFloat32);
        prior.covariance[0][0] = cov_eigen(0, 0);
        prior.covariance[1][1] = cov_eigen(1, 1);
        prior.covariance[2][2] = cov_eigen(2, 2);

        last_lidar_timestamp = lidar_timestamp;
        return prior;
    }

    // ===== HELPER METHOD: Compute motion-based covariance =====
    /**
     * Compute motion-based covariance consistently
     * σ = base + k * distance
     */
    Eigen::Matrix3f RoomConcept::compute_motion_covariance(const OdometryPrior &odometry_prior)
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

    /**
     * Compute adaptive prior weight using information-theoretic principles
     *
     * Key ideas:
     * 1. Weight should be HIGH when prior has high information (tight covariance)
     * 2. Weight should be LOW when innovation is large (prior doesn't match data)
     * 3. Weight should be 1.0 when stationary (no motion uncertainty)
     * 4. Transition should be SMOOTH (no discontinuities)
     *
     * @param innovation_norm Mahalanobis distance between prior and optimized pose
     * @param motion_magnitude Distance traveled since last frame
     * @param prior_cov Prior covariance matrix (3x3)
     * @param current_weight Current weight (for EMA smoothing)
     * @return New weight in [0, 1]
     */
    float RoomConcept::compute_adaptive_prior_weight(float innovation_norm,
                                                       float motion_magnitude,
                                                       const Eigen::Matrix3f& prior_cov,
                                                       float current_weight)
    {
        // ===== SPECIAL CASE: STATIONARY =====
        // When robot doesn't move, trust odometry completely
        if (motion_magnitude < 0.01f) {  // 10mm threshold
            return 1.0f;  // Full weight, no smoothing needed
        }

        // ===== INFORMATION CONTENT =====
        // Prior with tight covariance has high information → deserves high weight
        // Prior with loose covariance has low information → deserves low weight

        // Information matrix = inverse of covariance
        // Use trace (sum of eigenvalues) as scalar measure
        Eigen::Matrix3f info_matrix = prior_cov.inverse();
        float information_content = info_matrix.trace();

        // Normalize: typical range is [10, 1000] depending on uncertainty
        // Map to [0.3, 1.0] so even weak priors have some influence
        const float min_info = 10.0f;    // Very uncertain prior
        const float max_info = 1000.0f;  // Very certain prior
        float info_factor = 0.3f + 0.7f * std::clamp(
            (information_content - min_info) / (max_info - min_info),
            0.0f, 1.0f
        );

        // ===== INNOVATION FACTOR =====
        // Large innovation → prior doesn't match reality → reduce weight
        // Small innovation → prior matches well → keep high weight

        // Exponential decay: smooth, continuous, theoretically grounded
        // λ = 0.1 means weight drops to ~0.37 at innovation_norm = 3σ
        const float lambda = 0.1f;
        float innovation_factor = std::exp(-lambda * innovation_norm * innovation_norm);

        // ===== MOTION FACTOR =====
        // More motion → more uncertainty → reduce prior weight
        // Less motion → less uncertainty → keep high weight

        // Normalize motion: 100mm = 0.1m is reference
        const float motion_scale = 0.1f;  // 100mm reference
        float normalized_motion = motion_magnitude / motion_scale;

        // Exponential decay with motion
        float motion_factor = std::exp(-0.5f * normalized_motion);

        // ===== COMBINE FACTORS =====
        // Multiplicative: all factors must agree for high weight
        float target_weight = info_factor * innovation_factor * motion_factor;

        // Clamp to sensible range
        const float min_weight = 0.01f;   // Always some regularization
        const float max_weight = 1.0f;
        target_weight = std::clamp(target_weight, min_weight, max_weight);

        // ===== TEMPORAL SMOOTHING =====
        // Use EMA to prevent abrupt changes
        const float alpha = 0.3f;  // 30% new, 70% old
        float smoothed_weight = alpha * target_weight + (1.0f - alpha) * current_weight;

        return smoothed_weight;
    }

    /**
     * Simplified version using only innovation norm
     */
    float RoomConcept::compute_adaptive_prior_weight_simple(float innovation_norm,
                                               float motion_magnitude,
                                               float current_weight)
    {
        // Stationary: full trust
        if (motion_magnitude < 0.01f) {
            return 1.0f;
        }

        // Exponential decay based on innovation
        // At 1σ: weight ≈ 0.90
        // At 2σ: weight ≈ 0.67
        // At 3σ: weight ≈ 0.41
        // At 4σ: weight ≈ 0.20
        const float lambda = 0.05f;
        float target_weight = std::exp(-lambda * innovation_norm * innovation_norm);

        // Clamp
        target_weight = std::clamp(target_weight, 0.01f, 1.0f);

        // Smooth
        const float alpha = 0.3f;
        return alpha * target_weight + (1.0f - alpha) * current_weight;
    }
};