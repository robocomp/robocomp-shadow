/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 */

// Undefine Qt macros before including PyTorch headers
#ifdef slots
#undef slots
#endif
#ifdef signals
#undef signals
#endif
#ifdef emit
#undef emit
#endif

#include "door_concept.h"
#include <iostream>
#include <QDebug>

namespace rc
{
    void DoorConcept::initialize(const RoboCompLidar3D::TPoints& roi_points,
                                float initial_width,
                                float initial_height,
                                float initial_angle)
    {
        if (roi_points.empty())
        {
            qWarning() << "DoorConcept::initialize() - Empty ROI point cloud!";
            return;
        }

        qInfo() << "DoorConcept::initialize() - Initializing with" << roi_points.size() << "points";

        door_model_.init(roi_points, initial_width, initial_height, initial_angle);
        initialized_ = true;

        //door_model_.print_info();
    }

    void DoorConcept::reset()
    {
        initialized_ = false;
        qInfo() << "DoorConcept::reset() - Door concept reset";
    }

    torch::Tensor DoorConcept::convert_points_to_tensor(const RoboCompLidar3D::TPoints& points)
    {
        std::vector<float> points_data;
        points_data.reserve(points.size() * 3);

        for (const auto& p : points)
        {
            points_data.push_back(p.x / 1000.0f);  // mm to meters
            points_data.push_back(p.y / 1000.0f);
            points_data.push_back(p.z / 1000.0f);
        }

        return torch::from_blob(
            points_data.data(),
            {static_cast<long>(points.size()), 3},
            torch::kFloat32
        ).clone();
    }

    void DoorConcept::predict_step(const Eigen::Vector3f& robot_motion)
    {
        // When robot moves, door appears to move in opposite direction
        // This is a simplified prediction - in full SLAM, we'd also update uncertainty

        if (robot_motion.norm() < 1e-6f)
            return;  // No motion, skip prediction

        // Get current door pose
        auto pose = door_model_.get_door_pose();
        float door_x = pose[0];
        float door_y = pose[1];
        float door_z = pose[2];
        float door_theta = pose[3];

        // Transform robot motion to door frame
        // (Simplified: assumes small rotations)
        const float dx_robot = robot_motion[0];
        const float dy_robot = robot_motion[1];
        const float dtheta_robot = robot_motion[2];

        // Door moves opposite to robot in robot's frame
        door_x -= dx_robot;
        door_y -= dy_robot;
        door_theta -= dtheta_robot;

        // Update door model parameters (no_grad to avoid building computation graph)
        door_model_.set_pose(door_x, door_y, door_z , door_theta);

        qDebug() << "DoorConcept::predict_step() - Applied motion: dx=" << dx_robot
                 << "dy=" << dy_robot << "dθ=" << dtheta_robot;
    }

    torch::Tensor DoorConcept::compute_measurement_loss(const torch::Tensor& points_tensor)
    {
        // SDF-based surface fitting loss
        // We want all points to lie on the door surface (SDF ≈ 0)

        const torch::Tensor sdf_values = door_model_.sdf(points_tensor);

        // L1 loss: mean absolute distance to surface
        return torch::mean(torch::abs(sdf_values));
    }

    torch::Tensor DoorConcept::compute_geometry_regularization()
    {
        if (!config_.use_geometry_regularization)
            return torch::tensor(0.0f);

        // Penalize door dimensions that deviate from typical values
        // Uses Gaussian prior: exp(-0.5 * ((x - μ) / σ)²)
        // Which gives loss: 0.5 * ((x - μ) / σ)²

        const auto ps = door_model_.get_door_geometry();
        const float width = ps[0];
        const float height = ps[1];

        const float width_dev = (width - config_.typical_width) / config_.size_std;
        const float height_dev = (height - config_.typical_height) / config_.size_std;

        const float reg_loss = 0.5f * (width_dev * width_dev + height_dev * height_dev);

        return torch::tensor(reg_loss) * config_.geometry_reg_weight;
    }

    DoorConcept::OptimizationResult DoorConcept::run_optimization(
        const torch::Tensor& points_tensor,
        torch::optim::Optimizer& optimizer)
    {
        OptimizationResult result;

        float best_loss = std::numeric_limits<float>::max();
        int patience_counter = 0;

        for (int iter = 0; iter < config_.max_iterations; ++iter)
        {
            // Zero gradients
            optimizer.zero_grad();

            // Compute losses
            const torch::Tensor meas_loss = compute_measurement_loss(points_tensor);
            const torch::Tensor reg_loss = compute_geometry_regularization();
            const torch::Tensor total_loss = meas_loss + reg_loss;

            // Backpropagate
            total_loss.backward();

            // Update parameters
            optimizer.step();

            // Constrain parameters to valid ranges
            {
                torch::NoGradGuard no_grad;

                // Positive dimensions

                door_model_.door_width_.clamp_(0.3f, 2.0f);   // 30cm to 2m
                door_model_.door_height_.clamp_(1.5f, 3.0f);  // 1.5m to 3m

                // Opening angle: -π to π
                door_model_.opening_angle_.clamp_(-M_PI, M_PI);

                // Door orientation: -π to π
                auto theta_val = door_model_.door_theta_.item<float>();
                while (theta_val > M_PI) theta_val -= 2 * M_PI;
                while (theta_val < -M_PI) theta_val += 2 * M_PI;
                door_model_.door_theta_[0] = theta_val;
            }

            // Track best loss
            const float current_loss = total_loss.item<float>();

            if (iter % 50 == 0)
            {
                qDebug() << "  Iter" << iter << ": loss=" << current_loss
                         << "(meas=" << meas_loss.item<float>()
                         << ", reg=" << reg_loss.item<float>() << ")";
            }

            // Check convergence
            if (std::abs(best_loss - current_loss) < config_.convergence_delta)
            {
                patience_counter++;
                if (patience_counter >= config_.convergence_patience)
                {
                    qInfo() << "DoorConcept::run_optimization() - Converged at iteration" << iter;
                    result.converged = true;
                    result.iterations = iter;
                    result.total_loss = current_loss;
                    result.measurement_loss = meas_loss.item<float>();
                    break;
                }
            }
            else
            {
                patience_counter = 0;
                best_loss = current_loss;
            }

            // Early termination if loss is very low
            if (current_loss < config_.min_loss_threshold)
            {
                qInfo() << "DoorConcept::run_optimization() - Loss below threshold at iteration" << iter;
                result.converged = true;
                result.iterations = iter;
                result.total_loss = current_loss;
                result.measurement_loss = meas_loss.item<float>();
                break;
            }
        }

        // If we didn't break early, still return final values
        if (!result.converged)
        {
            result.iterations = config_.max_iterations;
            const torch::Tensor final_meas = compute_measurement_loss(points_tensor);
            const torch::Tensor final_reg = compute_geometry_regularization();
            result.total_loss = (final_meas + final_reg).item<float>();
            result.measurement_loss = final_meas.item<float>();
        }

        return result;
    }

    torch::Tensor DoorConcept::estimate_uncertainty(const torch::Tensor& points_tensor,
                                                   float final_loss)
    {
        // Estimate parameter uncertainty using Hessian approximation
        // Similar to RoomOptimizer's approach

        try
        {
            // Compute loss one more time to build computation graph
            const torch::Tensor loss = compute_measurement_loss(points_tensor);

            // Get all parameters
            auto params = door_model_.parameters();

            // Compute Hessian using finite differences (simplified)
            // For now, use diagonal approximation from gradient magnitudes
            std::vector<float> param_stds;

            for (const auto& param : params)
            {
                if (param.grad().defined())
                {
                    const float grad_norm = torch::norm(param.grad()).item<float>();
                    // Simple heuristic: std ≈ sqrt(loss) / grad_norm
                    const float std_dev = (grad_norm > 1e-6f)
                        ? std::sqrt(final_loss) / grad_norm
                        : 0.1f;  // Fallback

                    const int n_elements = param.numel();
                    for (int i = 0; i < n_elements; ++i)
                        param_stds.push_back(std_dev);
                }
            }

            // Build diagonal covariance matrix
            const int n_params = static_cast<int>(param_stds.size());
            torch::Tensor cov = torch::zeros({n_params, n_params});

            for (int i = 0; i < n_params; ++i)
            {
                const float variance = param_stds[i] * param_stds[i];
                cov[i][i] = variance;
            }

            return cov;
        }
        catch (const std::exception& e)
        {
            qWarning() << "DoorConcept::estimate_uncertainty() - Failed:" << e.what();

            // Return large uncertainty as fallback
            const int n_params = 7;  // x, y, z, theta, width, height, angle
            torch::Tensor cov = torch::eye(n_params) * 0.1f;  // 10cm std dev
            return cov;
        }
    }

    DoorConcept::Result DoorConcept::update_step(const torch::Tensor& points_tensor)
    {
        Result result;
        result.num_points_used = points_tensor.size(0);

        if (result.num_points_used < 10)
        {
            qWarning() << "DoorConcept::update_step() - Too few points:" << result.num_points_used;
            result.success = false;
            return result;
        }

        qInfo() << "DoorConcept::update_step() - Optimizing with" << result.num_points_used << "points";

        // Create optimizer (optimize all parameters)
        auto params = door_model_.parameters();
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(config_.learning_rate));

        // Run optimization loop
        const auto opt_result = run_optimization(points_tensor, optimizer);

        // Store results
        result.final_loss = opt_result.total_loss;
        result.measurement_loss = opt_result.measurement_loss;
        result.success = opt_result.converged || opt_result.total_loss < config_.min_loss_threshold * 10.0f;

        // Estimate uncertainty
        result.covariance = estimate_uncertainty(points_tensor, result.final_loss);

        // Extract standard deviations
        for (int i = 0; i < result.covariance.size(0); ++i)
        {
            const float variance = result.covariance[i][i].item<float>();
            result.std_devs.push_back(std::sqrt(std::max(variance, 0.0f)));
        }

        // Get optimized parameters
        result.optimized_params = door_model_.get_door_parameters();

        // Compute mean residual
        {
            torch::NoGradGuard no_grad;
            const torch::Tensor sdf_vals = door_model_.sdf(points_tensor);
            result.mean_residual = torch::mean(torch::abs(sdf_vals)).item<float>();
        }

        qInfo() << "DoorConcept::update_step() - Optimization"
                << (result.success ? "SUCCESS" : "FAILED")
                << "- Loss:" << result.final_loss
                << "Mean residual:" << result.mean_residual << "m";

        return result;
    }

    DoorConcept::Result DoorConcept::update(const RoboCompLidar3D::TPoints& roi_points,
                                           const Eigen::Vector3f& robot_motion)
    {
        if (!initialized_)
        {
            qWarning() << "DoorConcept::update() - Not initialized!";
            return Result{};
        }

        // PREDICT: Adjust door pose for robot motion
        predict_step(robot_motion);

        // Convert points to tensor
        const torch::Tensor points_tensor = convert_points_to_tensor(roi_points);

        // UPDATE: Optimize door parameters
        return update_step(points_tensor);
    }
};