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
        result.door = door;

        // Check tracking quality
        if (!check_tracking_quality(result))
        {
            consecutive_tracking_failures_++;
            qWarning() << "DoorConcept::update() - Poor tracking quality, residual:"
                       << result.mean_residual
                       << "- Failure" << consecutive_tracking_failures_ << "/" << MAX_TRACKING_FAILURES;

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

        // Compute the 2D projected ROI from the 3D door model
        cv::Rect projected_roi = compute_projected_roi(door_model, rgbd.width, rgbd.height);

        // Expand ROI slightly to catch points at the edges
        const int margin_x = static_cast<int>(projected_roi.width * roi_config_.margin_factor);
        const int margin_y = static_cast<int>(projected_roi.height * roi_config_.margin_factor);

        projected_roi.x = std::max(0, projected_roi.x - margin_x);
        projected_roi.y = std::max(0, projected_roi.y - margin_y);
        projected_roi.width = std::min(rgbd.width - projected_roi.x,
                                       projected_roi.width + 2 * margin_x);
        projected_roi.height = std::min(rgbd.height - projected_roi.y,
                                        projected_roi.height + 2 * margin_y);

        //qDebug() << "DoorConcept::extract_roi_from_model() - Projected ROI:"
        //         << projected_roi.x << "," << projected_roi.y
        //         << projected_roi.width << "x" << projected_roi.height;

        // Get door pose for depth filtering
        auto pose = door_model.get_door_pose();
        const float door_depth = std::sqrt(pose[0]*pose[0] + pose[1]*pose[1]);

        // Depth range based on door position and geometry
        auto geom = door_model.get_door_geometry();
        const float door_width = geom[0];
        const float depth_margin = door_width * 0.5f;  // Half door width margin

        const float min_depth = std::max(roi_config_.min_depth, door_depth - depth_margin);
        const float max_depth = std::min(roi_config_.max_depth, door_depth + depth_margin);

        // Extract points from depth data within the projected ROI
        const auto depth_ptr = reinterpret_cast<const cv::Vec3f*>(rgbd.depth.data());

        for (int y = projected_roi.y; y < projected_roi.y + projected_roi.height; ++y)
        {
            for (int x = projected_roi.x; x < projected_roi.x + projected_roi.width; ++x)
            {
                const int index = y * rgbd.width + x;
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

        //qDebug() << "DoorConcept::extract_roi_from_model() - Extracted"
        //          << roi_points.size() << "points from model prediction";

        return roi_points;
    }

    cv::Rect DoorConcept::compute_projected_roi(const DoorModel& door_model,
                                                 int image_width, int image_height)
    {
        // Get door pose and geometry
        auto pose = door_model.get_door_pose();
        auto geom = door_model.get_door_geometry();

        const float door_x = pose[0];      // X position in robot frame
        const float door_y = pose[1];      // Y position (forward)
        const float door_z = pose[2];      // Z position
        const float door_theta = pose[3];  // Orientation

        const float door_width = geom[0];
        const float door_height = geom[1];

        // Compute 8 corners of the door bounding box in 3D
        // Door local frame: width along X, depth along Y, height along Z
        const float half_width = door_width / 2.0f;
        const float frame_depth = door_model.frame_depth_;
        const float half_depth = frame_depth / 2.0f;

        // Corners in door local frame (before rotation)
        std::vector<Eigen::Vector3f> corners_local = {
            {-half_width, -half_depth, 0.0f},           // Bottom-left-front
            { half_width, -half_depth, 0.0f},           // Bottom-right-front
            {-half_width,  half_depth, 0.0f},           // Bottom-left-back
            { half_width,  half_depth, 0.0f},           // Bottom-right-back
            {-half_width, -half_depth, door_height},    // Top-left-front
            { half_width, -half_depth, door_height},    // Top-right-front
            {-half_width,  half_depth, door_height},    // Top-left-back
            { half_width,  half_depth, door_height}     // Top-right-back
        };

        // Rotation matrix around Z-axis
        const float cos_theta = std::cos(door_theta);
        const float sin_theta = std::sin(door_theta);

        // Transform corners to robot frame and project to image
        float min_u = std::numeric_limits<float>::max();
        float max_u = std::numeric_limits<float>::lowest();
        float min_v = std::numeric_limits<float>::max();
        float max_v = std::numeric_limits<float>::lowest();

        for (const auto& corner_local : corners_local)
        {
            // Rotate around Z-axis
            float rotated_x = cos_theta * corner_local.x() - sin_theta * corner_local.y();
            float rotated_y = sin_theta * corner_local.x() + cos_theta * corner_local.y();
            float rotated_z = corner_local.z();

            // Translate to door position in robot frame
            Eigen::Vector3f corner_robot(
                door_x + rotated_x,
                door_y + rotated_y,
                door_z + rotated_z
            );

            // Project to image
            cv::Point2f pixel = project_point_to_image(corner_robot, image_width, image_height);

            min_u = std::min(min_u, pixel.x);
            max_u = std::max(max_u, pixel.x);
            min_v = std::min(min_v, pixel.y);
            max_v = std::max(max_v, pixel.y);
        }

        // Convert to cv::Rect with bounds checking
        int x = static_cast<int>(std::floor(min_u));
        int y = static_cast<int>(std::floor(min_v));
        int w = static_cast<int>(std::ceil(max_u - min_u));
        int h = static_cast<int>(std::ceil(max_v - min_v));

        // Clamp to image bounds
        x = std::max(0, std::min(x, image_width - 1));
        y = std::max(0, std::min(y, image_height - 1));
        w = std::max(1, std::min(w, image_width - x));
        h = std::max(1, std::min(h, image_height - y));

        // Apply expansion factor for safety margin
        const int expand_x = static_cast<int>(w * (config_.roi_expansion_factor - 1.0f) / 2.0f);
        const int expand_y = static_cast<int>(h * (config_.roi_expansion_factor - 1.0f) / 2.0f);

        x = std::max(0, x - expand_x);
        y = std::max(0, y - expand_y);
        w = std::min(image_width - x, w + 2 * expand_x);
        h = std::min(image_height - y, h + 2 * expand_y);

        return cv::Rect(x, y, w, h);
    }

    cv::Point2f DoorConcept::project_point_to_image(const Eigen::Vector3f& point_3d,
                                                     int image_width, int image_height)
    {
        // Equirectangular projection for 360° camera
        // Robot frame: X+ right, Y+ forward, Z+ up

        const float x = point_3d.x();
        const float y = point_3d.y();
        const float z = point_3d.z();

        // Compute horizontal angle (azimuth) from Y-axis (forward)
        // atan2(x, y) gives angle from forward direction
        const float azimuth = std::atan2(x, y);  // Range: [-π, π]

        // Compute vertical angle (elevation)
        const float range_xy = std::sqrt(x*x + y*y);
        const float elevation = std::atan2(z, range_xy);  // Range: [-π/2, π/2]

        // Map to image coordinates
        // Horizontal: azimuth [-π, π] -> [0, width]
        // For equirectangular: center of image is forward (azimuth=0)
        float u = (azimuth / M_PI + 1.0f) * 0.5f * image_width;

        // Vertical: elevation [-π/2, π/2] -> [height, 0] (image Y increases downward)
        // Center of image is horizon (elevation=0)
        float v = (0.5f - elevation / M_PI) * image_height;

        // Clamp to image bounds
        u = std::max(0.0f, std::min(u, static_cast<float>(image_width - 1)));
        v = std::max(0.0f, std::min(v, static_cast<float>(image_height - 1)));

        return cv::Point2f(u, v);
    }

    bool DoorConcept::check_tracking_quality(const Result& result)
    {
        // Check if optimization succeeded
        if (!result.success)
            return false;

        // Check if mean residual is below threshold
        if (result.mean_residual > config_.tracking_lost_threshold)
            return false;

        // Check if we have enough points
        if (result.num_points_used < config_.min_points_for_tracking)
            return false;

        return true;
    }

    // =========================================================================
    // YOLO DETECTION (for initialization and recovery)
    // =========================================================================

    std::vector<DoorModel> DoorConcept::detect(const RoboCompCamera360RGBD::TRGBD &rgbd)
    {
        cv::Mat cv_img(cv::Size(rgbd.width, rgbd.height), CV_8UC3, const_cast<unsigned char*>(rgbd.rgb.data()));
        cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);

        // Use panoramic detection for 360° images
        auto doors_raw = yolo_detector->detectPanoramic(cv_img);
        qInfo() << __FUNCTION__ << "Detected" << doors_raw.size() << "doors";

        std::vector<DoorModel> doors;
        for (auto &d : doors_raw)
        {
            // Expand ROI slightly
            const int x_offset = d.roi.width / 10;
            const int y_offset = d.roi.height / 10;
            d.roi.x = std::max(0, d.roi.x - x_offset);
            d.roi.y = std::max(0, d.roi.y - y_offset);
            d.roi.width = std::min(rgbd.width - d.roi.x, d.roi.width + 2 * x_offset);
            d.roi.height = std::min(rgbd.height - d.roi.y, d.roi.height + 2 * y_offset);

            const cv::Rect roi(d.roi.x, d.roi.y, d.roi.width, d.roi.height);

            // Extract points within ROI
            std::vector<Eigen::Vector3f> roi_points;
            const auto depth_ptr = reinterpret_cast<const cv::Vec3f*>(rgbd.depth.data());

            for (int y = 0; y < rgbd.height; ++y)
            {
                for (int x = 0; x < rgbd.width; ++x)
                {
                    if (x < roi.x || x >= roi.x + roi.width ||
                        y < roi.y || y >= roi.y + roi.height)
                        continue;

                    const int index = y * rgbd.width + x;
                    const cv::Vec3f& point = depth_ptr[index];

                    if (std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))
                        continue;
                    if (point[0] == 0.0f && point[1] == 0.0f && point[2] == 0.0f)
                        continue;

                    roi_points.emplace_back(point[0]/1000.f, point[1]/1000.f, point[2]/1000.f);
                }
            }

            auto dm = DoorModel{};
            dm.init(roi_points, d.roi, doors.size(), d.label, 1.0f, 2.0f, 0.0f);
            doors.emplace_back(dm);
        }
        return doors;
    }

    // =========================================================================
    // REMAINING METHODS (unchanged from original)
    // =========================================================================

    void DoorConcept::initialize(const RoboCompLidar3D::TPoints& roi_points,
                                 float initial_width,
                                 float initial_height,
                                 float initial_angle)
    {
        if (roi_points.empty())
        {
            return;
        }
    }

    void DoorConcept::reset()
    {
        initialized_ = false;
        tracking_active_ = false;
        consecutive_tracking_failures_ = 0;
        door = nullptr;
        qInfo() << "DoorConcept::reset() - Door concept reset";
    }

    torch::Tensor DoorConcept::convert_points_to_tensor(const std::vector<Eigen::Vector3f> &points)
    {
        std::vector<float> points_data;
        points_data.reserve(points.size() * 3);

        for (const auto& p : points)
        {
            points_data.push_back(p.x());
            points_data.push_back(p.y());
            points_data.push_back(p.z());
        }

        return torch::from_blob(
            points_data.data(),
            {static_cast<long>(points.size()), 3},
            torch::kFloat32
        ).clone();
    }

    void DoorConcept::predict_step(const Eigen::Vector3f& robot_motion)
    {
        if (robot_motion.norm() < 1e-6f)
            return;

        auto pose = door->get_door_pose();
        float door_x = pose[0];
        float door_y = pose[1];
        float door_z = pose[2];
        float door_theta = pose[3];

        const float dx_robot = robot_motion[0];
        const float dy_robot = robot_motion[1];
        const float dtheta_robot = robot_motion[2];

        // Door moves opposite to robot in robot's frame
        door_x -= dx_robot;
        door_y -= dy_robot;
        door_theta -= dtheta_robot;

        door->set_pose(door_x, door_y, door_z, door_theta);

        //qDebug() << "DoorConcept::predict_step() - Applied motion: dx=" << dx_robot
        //         << "dy=" << dy_robot << "dθ=" << dtheta_robot;
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

        const auto ps = door->get_door_geometry();
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

        auto params = door->parameters();
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(config_.learning_rate));

        const auto opt_result = run_optimization(points_tensor, optimizer);

        result.final_loss = opt_result.total_loss;
        result.measurement_loss = opt_result.measurement_loss;
        result.success = opt_result.converged || opt_result.total_loss < config_.min_loss_threshold * 10.0f;

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