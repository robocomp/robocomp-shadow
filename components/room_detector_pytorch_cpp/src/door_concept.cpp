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
#include "door_projection.h"  // NEW: Include DoorProjection for model-based ROI
#include <iostream>
#include <QDebug>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rc
{
    std::optional<DoorConcept::Result> DoorConcept::update(const RoboCompCamera360RGBD::TRGBD &rgbd,
                                                           const Eigen::Vector3f& robot_motion)
    {
        std::vector<Eigen::Vector3f> roi_points;

        // =====================================================================
        // TOP-DOWN PREDICTION APPROACH
        // =====================================================================

        if (door == nullptr || !tracking_active_)
        {
            // -----------------------------------------------------------------
            // DETECTION MODE: No door tracked, run full YOLO detection
            // -----------------------------------------------------------------
            qInfo() << "DoorConcept::update() - Running YOLO detection (no active tracking)";

            auto door_candidates = detect(rgbd);
            if (door_candidates.empty())
            {
                qWarning() << "DoorConcept::update() - No doors detected!";
                return {};
            }

            // Take the best candidate (first one, typically highest confidence)
            door = std::make_shared<DoorModel>(door_candidates[0]);
            roi_points = door->roi_points;
            initialized_ = true;
            tracking_active_ = true;
            consecutive_tracking_failures_ = 0;

            // qInfo() << "DoorConcept::update() - Door initialized from YOLO detection,"
            //         << roi_points.size() << "points";
        }
        else
        {
            // -----------------------------------------------------------------
            // TRACKING MODE: Use model prediction to extract ROI (bypass YOLO)
            // -----------------------------------------------------------------
            //qDebug() << "DoorConcept::update() - Using model-based ROI extraction";

            // PREDICT: Adjust door pose for robot motion FIRST
            predict_step(robot_motion);

            // Extract ROI points using the predicted door model
            roi_points = extract_roi_from_model(rgbd, *door);

            // Check if we got enough points
            if (static_cast<int>(roi_points.size()) < config_.min_points_for_tracking)
            {
                consecutive_tracking_failures_++;
                qWarning() << "DoorConcept::update() - Insufficient points for tracking:"
                           << roi_points.size() << "(need" << config_.min_points_for_tracking << ")"
                           << "- Failure" << consecutive_tracking_failures_ << "/" << MAX_TRACKING_FAILURES;

                if (consecutive_tracking_failures_ >= MAX_TRACKING_FAILURES)
                {
                    // Tracking lost, trigger redetection on next call
                    qWarning() << "DoorConcept::update() - Tracking lost, will redetect";
                    tracking_active_ = false;
                    return {};
                }

                // Try to continue with whatever points we have
                if (roi_points.empty())
                    return {};
            }

            // Update door's roi_points for visualization/debugging
            door->roi_points = roi_points;
        }

        // =====================================================================
        // OPTIMIZATION
        // =====================================================================

        // Convert points to tensor
        const torch::Tensor points_tensor = convert_points_to_tensor(roi_points);

        // UPDATE: Optimize door parameters
        auto result = update_step(points_tensor);
        result.print();

        // Check tracking quality
        if (not check_tracking_quality(result))
        {
            consecutive_tracking_failures_++;
            if (consecutive_tracking_failures_ >= MAX_TRACKING_FAILURES)
            {
                qWarning() << "DoorConcept::update() - Tracking quality too low, will redetect";
                tracking_active_ = false;
            }
        }
        else
        {
            // Good tracking, reset failure counter
            consecutive_tracking_failures_ = 0;
            result.door = door;
        }

        return result;
    }

    // =========================================================================
    // TOP-DOWN ROI EXTRACTION FROM MODEL PREDICTION
    // =========================================================================

    std::vector<Eigen::Vector3f> DoorConcept::extract_roi_from_model(
        const RoboCompCamera360RGBD::TRGBD &rgbd,
        const DoorModel& door_model)
    {
        std::vector<Eigen::Vector3f> roi_points;

        // =====================================================================
        // USE DoorProjection::predictROI() for consistent equirectangular projection
        // =====================================================================

        // Camera height for projection (sensor mounted at ~1.2m typically)
        const Eigen::Vector3f camera_pos(0.0f, 0.0f, roi_config_.camera_height);

        // Get predicted ROI using the proper equirectangular projection
        // This accounts for the full door geometry (frame + leaf)
        const int margin_pixels = static_cast<int>(
            std::max(rgbd.width, rgbd.height) * roi_config_.margin_factor);

        PredictedROI predicted_roi = DoorProjection::predictROI(
            std::make_shared<DoorModel>(door_model),  // Create shared_ptr for the API
            rgbd.width,
            rgbd.height,
            camera_pos,
            margin_pixels,
            15  // num_samples for projection
        );

        if (!predicted_roi.valid)
        {
            qWarning() << "DoorConcept::extract_roi_from_model() - ROI projection failed";
            return roi_points;
        }

        // qDebug() << "DoorConcept::extract_roi_from_model() - Predicted ROI:"
        //          << predicted_roi.u_min << "-" << predicted_roi.u_max << "x"
        //          << predicted_roi.v_min << "-" << predicted_roi.v_max
        //          << (predicted_roi.wraps_around ? "(wraps)" : "");

        // Get door pose for depth filtering
        auto pose = door_model.get_door_pose();
        const float door_depth = std::sqrt(pose[0]*pose[0] + pose[1]*pose[1]);

        // Depth range based on door position and geometry
        auto geom = door_model.get_door_geometry();
        const float door_width = geom[0];
        const float depth_margin = door_width * 0.5f;  // Half door width margin

        const float min_depth = std::max(roi_config_.min_depth, door_depth - depth_margin);
        const float max_depth = std::min(roi_config_.max_depth, door_depth + depth_margin);

        // Extract points from depth data within the predicted ROI
        const auto depth_ptr = reinterpret_cast<const cv::Vec3f*>(rgbd.depth.data());

        // Helper lambda to process a rectangular region
        auto process_region = [&](int u_start, int u_end, int v_start, int v_end)
        {
            for (int v = v_start; v < v_end; ++v)
            {
                for (int u = u_start; u < u_end; ++u)
                {
                    const int index = v * rgbd.width + u;
                    const cv::Vec3f& point = depth_ptr[index];

                    // Skip invalid points
                    if (std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))
                        continue;
                    if (point[0] == 0.0f && point[1] == 0.0f && point[2] == 0.0f)
                        continue;

                    // Convert to meters
                    const float px = point[0] / 1000.f;
                    const float py = point[1] / 1000.f;
                    const float pz = point[2] / 1000.f;

                    // Depth filtering
                    if (roi_config_.filter_by_depth)
                    {
                        const float point_depth = std::sqrt(px*px + py*py);
                        if (point_depth < min_depth || point_depth > max_depth)
                            continue;
                    }

                    roi_points.emplace_back(px, py, pz);
                }
            }
        };

        // Process primary ROI region
        process_region(predicted_roi.u_min, predicted_roi.u_max,
                       predicted_roi.v_min, predicted_roi.v_max);

        // Process secondary region if ROI wraps around image boundary
        if (predicted_roi.wraps_around)
        {
            process_region(predicted_roi.u_min_2, predicted_roi.u_max_2,
                           predicted_roi.v_min, predicted_roi.v_max);
        }

        //qDebug() << "DoorConcept::extract_roi_from_model() - Extracted"
        //          << roi_points.size() << "points from model prediction";

        return roi_points;
    }

    // =========================================================================
    // LEGACY: compute_projected_roi - Kept for backward compatibility
    // Now internally uses DoorProjection::predictROI
    // =========================================================================

    cv::Rect DoorConcept::compute_projected_roi(const DoorModel& door_model,
                                                 int image_width, int image_height)
    {
        const Eigen::Vector3f camera_pos(0.0f, 0.0f, roi_config_.camera_height);
        const int margin = static_cast<int>(image_width * roi_config_.margin_factor);

        PredictedROI roi = DoorProjection::predictROI(
            std::make_shared<DoorModel>(door_model),
            image_width,
            image_height,
            camera_pos,
            margin
        );

        if (!roi.valid)
        {
            // Return a default ROI at image center if projection fails
            return cv::Rect(image_width/4, image_height/4,
                            image_width/2, image_height/2);
        }

        // For cv::Rect, we can only return one rectangle
        // If wrapping, merge the two regions or return the larger one
        if (roi.wraps_around)
        {
            // For simplicity, return a rect spanning the entire horizontal range
            // This is conservative but ensures we don't miss the door
            int v_min = roi.v_min;
            int v_max = roi.v_max;
            return cv::Rect(0, v_min, image_width, v_max - v_min);
        }

        return roi.toCvRect();
    }

    cv::Point2f DoorConcept::project_point_to_image(const Eigen::Vector3f& point_3d,
                                                     int image_width, int image_height)
    {
        // Use EquirectangularProjection for consistency
        int u, v;
        if (EquirectangularProjection::projectPoint(point_3d, image_width, image_height, u, v))
        {
            return cv::Point2f(static_cast<float>(u), static_cast<float>(v));
        }

        // Fallback to center if projection fails
        return cv::Point2f(image_width / 2.0f, image_height / 2.0f);
    }

    bool DoorConcept::check_tracking_quality(const Result& result)
    {
        // Check if optimization succeeded
        if (not result.success)
        { qWarning() << __FUNCTION__ << "No result success"; return false;}

        // Check if mean residual is below threshold
        if (result.mean_residual > config_.tracking_lost_threshold)
        { qWarning() << __FUNCTION__ << "Mean residual" << result.mean_residual << "below threshold"; return false;}

        // Check if we have enough points
        if (result.num_points_used < config_.min_points_for_tracking)
        { qWarning() << __FUNCTION__ << "Not enough points" << result.num_points_used; return false;}


        return true;
    }

    void DoorConcept::reset()
    {
        door = nullptr;
        initialized_ = false;
        tracking_active_ = false;
        consecutive_tracking_failures_ = 0;
    }

    // =========================================================================
    // DETECT: Full YOLO-based detection (initialization or recovery)
    // =========================================================================

    std::vector<DoorModel> DoorConcept::detect(const RoboCompCamera360RGBD::TRGBD &rgbd)
    {
        std::vector<DoorModel> detected_doors;

        // Get RGB image for YOLO
        cv::Mat rgb_image(rgbd.height, rgbd.width, CV_8UC3,
                          const_cast<uint8_t*>(rgbd.rgb.data()));

        // Run YOLO detection
        auto detections = yolo_detector->detect(rgb_image);

        if (detections.empty())
        {
            //qDebug() << "DoorConcept::detect() - No YOLO detections";
            return detected_doors;
        }

        //qInfo() << "DoorConcept::detect() - YOLO found" << detections.size() << "candidate(s)";

        // Process each detection
        const auto depth_ptr = reinterpret_cast<const cv::Vec3f*>(rgbd.depth.data());

        for (const auto& det : detections)
        {
            // Extract ROI from detection bounding box
            cv::Rect roi = det.roi;

            // Expand ROI slightly
            const int margin_x = static_cast<int>(roi.width * roi_config_.margin_factor);
            const int margin_y = static_cast<int>(roi.height * roi_config_.margin_factor);

            roi.x = std::max(0, roi.x - margin_x);
            roi.y = std::max(0, roi.y - margin_y);
            roi.width = std::min(rgbd.width - roi.x, roi.width + 2 * margin_x);
            roi.height = std::min(rgbd.height - roi.y, roi.height + 2 * margin_y);

            // Extract 3D points from ROI
            std::vector<Eigen::Vector3f> roi_points;

            for (int y = roi.y; y < roi.y + roi.height; ++y)
            {
                for (int x = roi.x; x < roi.x + roi.width; ++x)
                {
                    const int index = y * rgbd.width + x;
                    const cv::Vec3f& point = depth_ptr[index];

                    // Skip invalid points
                    if (std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))
                        continue;
                    if (point[0] == 0.0f && point[1] == 0.0f && point[2] == 0.0f)
                        continue;

                    // Convert to meters and filter by depth
                    const float px = point[0] / 1000.f;
                    const float py = point[1] / 1000.f;
                    const float pz = point[2] / 1000.f;

                    const float depth = std::sqrt(px*px + py*py);
                    if (depth < roi_config_.min_depth || depth > roi_config_.max_depth)
                        continue;

                    roi_points.emplace_back(px, py, pz);
                }
            }

            if (roi_points.size() < static_cast<size_t>(config_.min_points_for_tracking))
            {
                qDebug() << "DoorConcept::detect() - Skipping detection with only"
                         << roi_points.size() << "points";
                continue;
            }

            // Create door model from ROI points
            DoorModel door_model;
            door_model.init(roi_points, roi, det.classId, det.label);
            door_model.roi_points = roi_points;
            door_model.roi = roi;

            detected_doors.push_back(door_model);

            //qInfo() << "DoorConcept::detect() - Created door model with"
            //        << roi_points.size() << "points";
        }

        return detected_doors;
    }

    // =========================================================================
    // PREDICT STEP: Adjust door pose for robot motion
    // =========================================================================

    void DoorConcept::predict_step(const Eigen::Vector3f& robot_motion)
    {
        if (!door)
            return;

        // robot_motion contains [dx, dy, dtheta] in robot frame
        const float dx_robot = robot_motion.x();
        const float dy_robot = robot_motion.y();
        const float dtheta_robot = robot_motion.z();

        // Get current door pose in robot frame
        auto pose = door->get_door_pose();
        float door_x = pose[0];
        float door_y = pose[1];
        float door_z = pose[2];
        float door_theta = pose[3];

        // Transform door pose: door moves in opposite direction to robot motion
        // 1. First rotate around robot (new origin)
        const float cos_dtheta = std::cos(-dtheta_robot);
        const float sin_dtheta = std::sin(-dtheta_robot);

        float new_x = cos_dtheta * door_x - sin_dtheta * door_y;
        float new_y = sin_dtheta * door_x + cos_dtheta * door_y;

        // 2. Then translate (robot moved forward, so door appears to move backward)
        new_x -= dx_robot;
        new_y -= dy_robot;

        // 3. Update door orientation (relative to robot)
        float new_theta = door_theta - dtheta_robot;

        // Normalize theta to [-π, π]
        while (new_theta > M_PI) new_theta -= 2 * M_PI;
        while (new_theta < -M_PI) new_theta += 2 * M_PI;

        door_x = new_x;
        door_y = new_y;
        door_theta = new_theta;

        door->set_pose(door_x, door_y, door_z, door_theta);

        //qDebug() << "DoorConcept::predict_step() - Applied motion: dx=" << dx_robot
        //         << "dy=" << dy_robot << "dθ=" << dtheta_robot;
    }

    // =========================================================================
    // OPTIMIZATION FUNCTIONS
    // =========================================================================

    torch::Tensor DoorConcept::convert_points_to_tensor(const std::vector<Eigen::Vector3f> &points)
    {
        if (points.empty())
            return torch::empty({0, 3});

        std::vector<float> data;
        data.reserve(points.size() * 3);

        for (const auto& p : points)
        {
            data.push_back(p.x());
            data.push_back(p.y());
            data.push_back(p.z());
        }

        return torch::from_blob(data.data(), {static_cast<long>(points.size()), 3},
                                torch::kFloat32).clone();
    }

    torch::Tensor DoorConcept::compute_measurement_loss(const torch::Tensor& points_tensor)
    {
        const torch::Tensor sdf_values = door->sdf(points_tensor);
        return torch::mean(torch::abs(sdf_values));
    }

    torch::Tensor DoorConcept::compute_geometry_regularization()
    {
        if (!config_.use_geometry_regularization)
            return torch::tensor(0.0f);

        // Use tensors directly for proper gradient flow
        const torch::Tensor width = door->door_width_;
        const torch::Tensor height = door->door_height_;

        // Typical door dimensions as tensors
        const torch::Tensor typical_width = torch::tensor(config_.typical_width);
        const torch::Tensor typical_height = torch::tensor(config_.typical_height);
        const torch::Tensor size_std = torch::tensor(config_.size_std);

        // Normalized deviations (Mahalanobis-style)
        const torch::Tensor width_dev = (width - typical_width) / size_std;
        const torch::Tensor height_dev = (height - typical_height) / size_std;

        // Squared error (Gaussian prior)
        const torch::Tensor reg_loss = 0.5f * (width_dev * width_dev + height_dev * height_dev);

        return reg_loss * config_.geometry_reg_weight;
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
            optimizer.zero_grad();

            const torch::Tensor meas_loss = compute_measurement_loss(points_tensor);
            const torch::Tensor reg_loss = compute_geometry_regularization();
            const torch::Tensor total_loss = meas_loss + reg_loss;

            total_loss.backward();
            optimizer.step();

            // Constrain parameters
            {
                torch::NoGradGuard no_grad;
                door->door_width_.clamp_(0.3f, 2.0f);
                door->door_height_.clamp_(1.5f, 3.0f);
                door->opening_angle_.clamp_(-M_PI, M_PI);

                auto theta_val = door->door_theta_.item<float>();
                while (theta_val > M_PI) theta_val -= 2 * M_PI;
                while (theta_val < -M_PI) theta_val += 2 * M_PI;
                door->door_theta_[0] = theta_val;
            }

            const float current_loss = total_loss.item<float>();

            if (iter % 50 == 0)
            {
                //qDebug() << "  Iter" << iter << ": loss=" << current_loss
                //         << "(meas=" << meas_loss.item<float>()
                //         << ", reg=" << reg_loss.item<float>() << ")";
            }

            if (std::abs(best_loss - current_loss) < config_.convergence_delta)
            {
                patience_counter++;
                if (patience_counter >= config_.convergence_patience)
                {
                    //qInfo() << "DoorConcept::run_optimization() - Converged at iteration" << iter;
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

            if (current_loss < config_.min_loss_threshold)
            {
                //qInfo() << "DoorConcept::run_optimization() - Loss below threshold at iteration" << iter;
                result.converged = true;
                result.iterations = iter;
                result.total_loss = current_loss;
                result.measurement_loss = meas_loss.item<float>();
                break;
            }
        }

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
        try
        {
            const torch::Tensor loss = compute_measurement_loss(points_tensor);
            auto params = door->parameters();

            std::vector<float> param_stds;

            for (const auto& param : params)
            {
                if (param.grad().defined())
                {
                    const float grad_norm = torch::norm(param.grad()).item<float>();
                    const float std_dev = (grad_norm > 1e-6f)
                        ? std::sqrt(final_loss) / grad_norm
                        : 0.1f;

                    const int n_elements = param.numel();
                    for (int i = 0; i < n_elements; ++i)
                        param_stds.push_back(std_dev);
                }
            }

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
            const int n_params = 7;
            torch::Tensor cov = torch::eye(n_params) * 0.1f;
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

        //qInfo() << "DoorConcept::update_step() - Optimizing with" << result.num_points_used << "points";

        const auto params = door->parameters();
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(config_.learning_rate));

        const auto opt_result = run_optimization(points_tensor, optimizer);

        result.final_loss = opt_result.total_loss;
        result.measurement_loss = opt_result.measurement_loss;
        result.success = opt_result.converged || opt_result.total_loss < config_.min_loss_threshold * 10.0f;  // TODO:: why?
        result.iterations = opt_result.iterations;

        result.covariance = estimate_uncertainty(points_tensor, result.final_loss);

        for (int i = 0; i < result.covariance.size(0); ++i)
        {
            const float variance = result.covariance[i][i].item<float>();
            result.std_devs.push_back(std::sqrt(std::max(variance, 0.0f)));
        }

        result.optimized_params = door->get_door_parameters();

        {
            torch::NoGradGuard no_grad;
            const torch::Tensor sdf_vals = door->sdf(points_tensor);
            result.mean_residual = torch::mean(torch::abs(sdf_vals)).item<float>();
        }

        // qInfo() << "DoorConcept::update_step() - Optimization"
        //         << (result.success ? "SUCCESS" : "FAILED")
        //         << "- Loss:" << result.final_loss
        //         << "Mean residual:" << result.mean_residual << "m";

        return result;
    }

};