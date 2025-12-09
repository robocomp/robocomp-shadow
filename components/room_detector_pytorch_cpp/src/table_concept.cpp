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

#include "table_concept.h"
#include <QDebug>

namespace rc
{

std::optional<r> TableConcept::update(const RoboCompCamera360RGBD::TRGBD &rgbd,
                                              const Eigen::Vector3f& robot_motion)
{
    // If not tracking, try YOLO detection
    if (!tracking_active_)
    {
        auto detections = detect(rgbd);
        if (detections.empty())
        {
            qDebug() << "TableConcept: No tables detected";
            return std::nullopt;
        }

        // Take first detection
        table = std::make_shared<TableModel>(detections[0]);
        tracking_active_ = true;
        initialized_ = true;
        consecutive_tracking_failures_ = 0;

        qInfo() << "TableConcept: Table initialized from YOLO detection";
    }

    // PREDICT: Adjust pose for robot motion
    if (robot_motion.norm() > 1e-6f)
    {
        predict_step(robot_motion);
    }

    // Extract ROI from predicted model
    auto roi_points = extract_roi_from_model(rgbd, *table);
    
    if (static_cast<int>(roi_points.size()) < config_.min_points_for_tracking)
    {
        qWarning() << "TableConcept: Insufficient points in ROI:" << roi_points.size();
        consecutive_tracking_failures_++;
        
        if (consecutive_tracking_failures_ >= MAX_TRACKING_FAILURES)
        {
            qWarning() << "TableConcept: Tracking lost after" << MAX_TRACKING_FAILURES << "failures";
            tracking_active_ = false;
            consecutive_tracking_failures_ = 0;
        }
        
        return std::nullopt;
    }

    // Convert to tensor
    auto points_tensor = convert_points_to_tensor(roi_points);

    // UPDATE: Optimize with measurements
    auto result = update_step(points_tensor);

    // Check tracking quality
    if (!check_tracking_quality(result))
    {
        consecutive_tracking_failures_++;
        
        if (consecutive_tracking_failures_ >= MAX_TRACKING_FAILURES)
        {
            qWarning() << "TableConcept: Poor tracking quality, forcing redetection";
            tracking_active_ = false;
            consecutive_tracking_failures_ = 0;
        }
        
        return std::nullopt;
    }

    // Tracking successful
    consecutive_tracking_failures_ = 0;
    return result;
}

std::vector<TableModel> TableConcept::detect(const RoboCompCamera360RGBD::TRGBD &rgbd)
{
    std::vector<TableModel> tables;

    // Run YOLO detector
    auto detections = yolo_detector->detect(rgbd.image);

    for (const auto& det : detections)
    {
        // Filter for table class (adjust based on your YOLO model)
        if (det.label != "table" && det.label != "dining table")
            continue;

        // Extract 3D points within ROI
        std::vector<Eigen::Vector3f> roi_points;
        
        for (int v = det.bbox.y; v < det.bbox.y + det.bbox.height && v < rgbd.depth.rows; ++v)
        {
            for (int u = det.bbox.x; u < det.bbox.x + det.bbox.width && u < rgbd.depth.cols; ++u)
            {
                float depth = rgbd.depth.at<float>(v, u);
                
                // Filter invalid depths
                if (depth < roi_config_.min_depth || depth > roi_config_.max_depth)
                    continue;

                // Convert pixel + depth to 3D point (equirectangular)
                float theta = (u / static_cast<float>(rgbd.depth.cols)) * 2.0f * M_PI - M_PI;
                float phi = (v / static_cast<float>(rgbd.depth.rows)) * M_PI - M_PI / 2.0f;

                float x = depth * std::cos(phi) * std::sin(theta);
                float y = depth * std::cos(phi) * std::cos(theta);
                float z = depth * std::sin(phi);

                roi_points.emplace_back(x, y, z);
            }
        }

        if (roi_points.size() < 50)  // Minimum points threshold
        {
            qDebug() << "TableConcept: Skipping detection with insufficient points:" << roi_points.size();
            continue;
        }

        // Create table model
        TableModel table_model;
        table_model.init(roi_points, det.bbox, tables.size(), det.label);
        tables.push_back(table_model);

        qInfo() << "TableConcept: Detected table with" << roi_points.size() << "points";
    }

    return tables;
}

std::vector<Eigen::Vector3f> TableConcept::extract_roi_from_model(
    const RoboCompCamera360RGBD::TRGBD &rgbd,
    const TableModel& table)
{
    // Compute projected ROI
    auto roi = compute_projected_roi(table, rgbd.image.cols, rgbd.image.rows);

    // Expand ROI by margin factor
    int margin_x = static_cast<int>(roi.width * config_.roi_expansion_factor * 0.5f);
    int margin_y = static_cast<int>(roi.height * config_.roi_expansion_factor * 0.5f);
    
    roi.x = std::max(0, roi.x - margin_x);
    roi.y = std::max(0, roi.y - margin_y);
    roi.width = std::min(rgbd.image.cols - roi.x, roi.width + 2 * margin_x);
    roi.height = std::min(rgbd.image.rows - roi.y, roi.height + 2 * margin_y);

    // Extract points within ROI
    std::vector<Eigen::Vector3f> points;
    
    for (int v = roi.y; v < roi.y + roi.height && v < rgbd.depth.rows; ++v)
    {
        for (int u = roi.x; u < roi.x + roi.width && u < rgbd.depth.cols; ++u)
        {
            float depth = rgbd.depth.at<float>(v, u);
            
            if (depth < roi_config_.min_depth || depth > roi_config_.max_depth)
                continue;

            // Convert to 3D
            float theta = (u / static_cast<float>(rgbd.depth.cols)) * 2.0f * M_PI - M_PI;
            float phi = (v / static_cast<float>(rgbd.depth.rows)) * M_PI - M_PI / 2.0f;

            float x = depth * std::cos(phi) * std::sin(theta);
            float y = depth * std::cos(phi) * std::cos(theta);
            float z = depth * std::sin(phi);

            points.emplace_back(x, y, z);
        }
    }

    return points;
}

cv::Rect TableConcept::compute_projected_roi(const TableModel& table,
                                             int image_width, int image_height)
{
    // Get table parameters
    auto params = table.get_table_parameters();
    float x = params[0], y = params[1], z = params[2], theta = params[3];
    float width = params[4], depth = params[5], height = params[6];

    // Compute 8 corners of table bounding box (tabletop + legs)
    std::vector<Eigen::Vector3f> corners;
    
    float half_w = width / 2.0f;
    float half_d = depth / 2.0f;
    
    // Tabletop corners
    for (float dx : {-half_w, half_w})
    {
        for (float dy : {-half_d, half_d})
        {
            corners.push_back({dx, dy, 0.0f});  // Top surface
            corners.push_back({dx, dy, -height});  // Bottom (floor)
        }
    }

    // Transform corners to robot frame and project
    float min_u = image_width, max_u = 0;
    float min_v = image_height, max_v = 0;

    for (const auto& corner_local : corners)
    {
        // Transform to robot frame
        Eigen::Vector3f corner_robot;
        corner_robot.x() = corner_local.x() * std::cos(theta) - corner_local.y() * std::sin(theta) + x;
        corner_robot.y() = corner_local.x() * std::sin(theta) + corner_local.y() * std::cos(theta) + y;
        corner_robot.z() = corner_local.z() + z;

        // Project to image
        auto proj = project_point_to_image(corner_robot, image_width, image_height);
        
        min_u = std::min(min_u, proj.x);
        max_u = std::max(max_u, proj.x);
        min_v = std::min(min_v, proj.y);
        max_v = std::max(max_v, proj.y);
    }

    // Clamp to image bounds
    min_u = std::max(0.0f, min_u);
    max_u = std::min(static_cast<float>(image_width - 1), max_u);
    min_v = std::max(0.0f, min_v);
    max_v = std::min(static_cast<float>(image_height - 1), max_v);

    return cv::Rect(
        static_cast<int>(min_u),
        static_cast<int>(min_v),
        static_cast<int>(max_u - min_u),
        static_cast<int>(max_v - min_v)
    );
}

cv::Point2f TableConcept::project_point_to_image(const Eigen::Vector3f& point_3d,
                                                 int image_width, int image_height)
{
    // Equirectangular projection
    float x = point_3d.x();
    float y = point_3d.y();
    float z = point_3d.z();

    float dist_xy = std::sqrt(x*x + y*y);
    float theta = std::atan2(x, y);  // Azimuth
    float phi = std::atan2(z, dist_xy);  // Elevation

    // Map to image coordinates
    float u = (theta + M_PI) / (2.0f * M_PI) * image_width;
    float v = (phi + M_PI / 2.0f) / M_PI * image_height;

    return cv::Point2f(u, v);
}

void TableConcept::reset()
{
    table = nullptr;
    initialized_ = false;
    tracking_active_ = false;
    consecutive_tracking_failures_ = 0;
    clearConsensusPrior();
    qInfo() << "TableConcept: Reset complete";
}

} // namespace rc

void TableConcept::predict_step(const Eigen::Vector3f& robot_motion)
{
    // Robot moved - adjust table pose in opposite direction
    // Table is "static" in world frame, but we track it in robot frame
    
    auto current_pose = table->get_table_pose();
    float x = current_pose[0];
    float y = current_pose[1];
    float z = current_pose[2];
    float theta = current_pose[3];

    // Robot motion: [dx, dy, dtheta] in robot frame
    float dx_robot = robot_motion.x();
    float dy_robot = robot_motion.y();
    float dtheta_robot = robot_motion.z();

    // Update table position (inverse of robot motion)
    float new_x = x - dx_robot;
    float new_y = y - dy_robot;
    float new_theta = theta - dtheta_robot;

    // Normalize angle
    while (new_theta > M_PI) new_theta -= 2.0f * M_PI;
    while (new_theta < -M_PI) new_theta += 2.0f * M_PI;

    table->set_pose(new_x, new_y, z, new_theta);

    qDebug() << "TableConcept: Predicted pose:"
             << "x=" << new_x << "y=" << new_y << "theta=" << new_theta;
}

TableConcept::Result TableConcept::update_step(const torch::Tensor& points_tensor)
{
    Result result;
    result.table = table;
    result.num_points_used = points_tensor.size(0);

    // Setup optimizer
    auto params = table->parameters();
    torch::optim::Adam optimizer(params, torch::optim::AdamOptions(config_.learning_rate));

    // Run optimization
    auto opt_result = run_optimization(points_tensor, optimizer);

    result.final_loss = opt_result.total_loss;
    result.measurement_loss = opt_result.measurement_loss;
    result.success = opt_result.converged;
    result.optimized_params = table->get_table_parameters();

    // Estimate uncertainty
    if (result.success)
    {
        result.covariance = estimate_uncertainty(points_tensor, result.final_loss);
        
        // Extract std devs
        if (result.covariance.defined() && result.covariance.size(0) > 0)
        {
            auto std_tensor = torch::sqrt(result.covariance.diagonal());
            auto std_acc = std_tensor.accessor<float, 1>();
            for (int i = 0; i < std_tensor.size(0); ++i)
                result.std_devs.push_back(std_acc[i]);
        }

        // Compute mean residual
        auto distances = table->sdf(points_tensor);
        result.mean_residual = torch::abs(distances).mean().item<float>();
    }

    return result;
}

torch::Tensor TableConcept::convert_points_to_tensor(const std::vector<Eigen::Vector3f> &points)
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

torch::Tensor TableConcept::compute_measurement_loss(const torch::Tensor& points_tensor)
{
    // Compute SDF distances
    auto distances = table->sdf(points_tensor);
    
    // L2 loss (penalize deviation from surface)
    return torch::mean(distances * distances);
}

torch::Tensor TableConcept::compute_geometry_regularization()
{
    if (!config_.use_geometry_regularization)
        return torch::zeros({1});

    auto geometry = table->get_table_geometry();
    float width = geometry[0];
    float depth = geometry[1];
    float height = geometry[2];

    // Regularization: penalize deviation from typical dimensions
    float width_dev = (width - config_.typical_width) / config_.size_std;
    float depth_dev = (depth - config_.typical_depth) / config_.size_std;
    float height_dev = (height - config_.typical_height) / config_.size_std;

    float reg_loss = width_dev * width_dev + depth_dev * depth_dev + height_dev * height_dev;

    return torch::tensor(reg_loss * config_.geometry_reg_weight);
}

torch::Tensor TableConcept::compute_consensus_prior_loss()
{
    if (!consensus_prior_set_ || !config_.use_consensus_prior)
        return torch::zeros({1});

    // Get current pose
    auto current_pose = table->get_table_pose();
    
    // Current pose as vector [x, y, z, theta]
    Eigen::Vector4f current{current_pose[0], current_pose[1], 
                           current_pose[2], current_pose[3]};
    
    // Difference from prior
    Eigen::Vector4f diff = current - consensus_prior_pose_;

    // Mahalanobis distance: diff^T * Info * diff
    float mahalanobis = diff.transpose() * consensus_prior_info_ * diff;

    // Separate weights for position and orientation
    float pos_loss = diff.head<3>().squaredNorm() * config_.consensus_prior_weight;
    float ori_loss = diff[3] * diff[3] * config_.consensus_orientation_weight;

    return torch::tensor(pos_loss + ori_loss);
}

TableConcept::OptimizationResult TableConcept::run_optimization(
    const torch::Tensor& points_tensor,
    torch::optim::Optimizer& optimizer)
{
    OptimizationResult result;
    
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;

    for (int iter = 0; iter < config_.max_iterations; ++iter)
    {
        optimizer.zero_grad();

        // Compute loss components
        auto measurement_loss = compute_measurement_loss(points_tensor);
        auto geometry_reg = compute_geometry_regularization();
        auto prior_loss = compute_consensus_prior_loss();

        auto total_loss = measurement_loss + geometry_reg + prior_loss;

        // Backward pass
        total_loss.backward();
        optimizer.step();

        // Track convergence
        float current_loss = total_loss.item<float>();
        
        if (current_loss < best_loss - config_.convergence_delta)
        {
            best_loss = current_loss;
            patience_counter = 0;
        }
        else
        {
            patience_counter++;
        }

        // Check convergence
        if (patience_counter >= config_.convergence_patience)
        {
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }

        // Check minimum threshold
        if (current_loss < config_.min_loss_threshold)
        {
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }
    }

    result.total_loss = best_loss;
    result.measurement_loss = compute_measurement_loss(points_tensor).item<float>();

    if (!result.converged)
        result.iterations = config_.max_iterations;

    return result;
}

torch::Tensor TableConcept::estimate_uncertainty(const torch::Tensor& points_tensor,
                                                float final_loss)
{
    // Laplace approximation: estimate Hessian at optimum
    
    auto params = table->parameters();
    int n_params = 0;
    for (const auto& p : params)
        n_params += p.numel();

    try
    {
        // Compute loss
        auto loss = compute_measurement_loss(points_tensor);

        // Compute gradients
        loss.backward(torch::nullopt, true);  // retain_graph

        // Approximate Hessian diagonal using gradient magnitudes
        torch::Tensor hessian_diag = torch::zeros({n_params});
        
        int idx = 0;
        for (const auto& p : params)
        {
            if (p.grad().defined())
            {
                auto grad_flat = p.grad().flatten();
                auto grad_sq = grad_flat * grad_flat;
                
                for (int i = 0; i < grad_flat.numel(); ++i)
                {
                    hessian_diag[idx++] = grad_sq[i].item<float>();
                }
            }
            else
            {
                idx += p.numel();
            }
        }

        // Add small regularization
        hessian_diag = hessian_diag + 1e-6f;

        // Covariance = inverse of Hessian (diagonal approximation)
        auto cov_diag = 1.0f / hessian_diag;

        // Construct diagonal covariance matrix
        return torch::diag(cov_diag);
    }
    catch (const std::exception& e)
    {
        qWarning() << "TableConcept: Uncertainty estimation failed:" << e.what();
        return torch::eye(n_params) * 0.1f;  // Default uncertainty
    }
}

bool TableConcept::check_tracking_quality(const Result& result)
{
    // Check mean residual
    if (result.mean_residual > config_.tracking_lost_threshold)
    {
        qWarning() << "TableConcept: High residual:" << result.mean_residual;
        return false;
    }

    // Check number of points
    if (result.num_points_used < config_.min_points_for_tracking)
    {
        qWarning() << "TableConcept: Insufficient points:" << result.num_points_used;
        return false;
    }

    // Check optimization success
    if (!result.success)
    {
        qWarning() << "TableConcept: Optimization failed";
        return false;
    }

    return true;
}

void TableConcept::setConsensusPrior(const Eigen::Vector4f& pose, 
                                     const Eigen::Matrix4f& covariance)
{
    consensus_prior_pose_ = pose;
    consensus_prior_covariance_ = covariance;
    
    // Compute precision matrix (inverse)
    consensus_prior_info_ = covariance.inverse();
    
    consensus_prior_set_ = true;

    qInfo() << "TableConcept: Consensus prior set at ("
            << pose[0] << "," << pose[1] << "," << pose[2] 
            << "), theta=" << pose[3];
}

void TableConcept::clearConsensusPrior()
{
    consensus_prior_set_ = false;
    qInfo() << "TableConcept: Consensus prior cleared";
}

} // namespace rc
