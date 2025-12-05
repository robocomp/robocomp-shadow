/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 */

#ifndef DOOR_CONCEPT_H
#define DOOR_CONCEPT_H

#include <torch/torch.h>
#include <Eigen/Dense>
#include <Lidar3D.h>
#include "door_model.h"
#include "common_types.h"
#include "yolo_detector_onnx.h"
#include <Camera360RGBD.h>
#include <QtCore>

namespace rc
{
    /**
     * @brief Door detection and tracking using Bayesian optimization
     *
     * Implements a predict-update cycle for door state estimation:
     * 1. PREDICT: Use robot motion to predict door pose relative to robot
     * 2. UPDATE: Optimize door parameters using LiDAR measurements within ROI
     *
     * Top-down prediction approach:
     * - First detection: Full YOLO detection to find doors
     * - Subsequent iterations: Use optimized door model to predict ROI,
     *   extract points directly, bypass YOLO unless tracking is lost
     */

    class DoorConcept
    {
        public:
            struct Result
            {
                torch::Tensor covariance;           // Full parameter covariance
                std::vector<float> std_devs;        // Standard deviations
                float final_loss = 0.0f;            // Total optimization loss
                float measurement_loss = 0.0f;      // SDF fitting loss
                bool success = false;               // Whether optimization succeeded
                std::vector<float> optimized_params; // Final door parameters
                int num_points_used = 0;            // Number of LiDAR points in ROI
                float mean_residual = 0.0f;         // Mean SDF residual
                std::shared_ptr<DoorModel> door;
                void print() const
                {
                    qInfo() << "==============================";
                    qInfo() << "Door " << door->id << "optimization result:"
                            << "	Final loss:" <<  final_loss
                            << "	Measurement loss:"  << measurement_loss
                            << "	Num points:" << num_points_used
                            << "	Success:" << success;
                    const auto params = optimized_params;
                    qInfo() << "Optimized door parameters:"
                            << "	x:"  << params[0]
                            << "	y:"  << params[1]
                            << "	z:"  << params[2]
                            << "	theta:"   << params[3]
                            << "	width:"   << params[4]
                            << "	height:"  << params[5]
                            << "	angle:"  << params[6];
                }
            };

            struct OptimizationConfig
            {
                int max_iterations = 200;
                float learning_rate = 0.01f;
                float min_loss_threshold = 0.001f;

                // Convergence criteria
                float convergence_patience = 10;     // Iterations without improvement
                float convergence_delta = 1e-5f;     // Minimum loss change

                // Regularization
                bool use_geometry_regularization = true;
                float geometry_reg_weight = 0.01f;   // Penalize unrealistic sizes

                // Standard door dimensions for regularization (meters)
                float typical_width = 0.9f;          // 90cm
                float typical_height = 2.0f;         // 2m
                float size_std = 0.2f;               // 20cm tolerance

                // Tracking quality thresholds
                float tracking_lost_threshold = 0.5f;   // Mean residual above this triggers redetection
                int min_points_for_tracking = 50;       // Minimum points to maintain tracking
                float roi_expansion_factor = 1.3f;      // Expand predicted ROI by this factor
            };

            /**
             * @brief ROI extraction configuration
             */
            struct ROIConfig
            {
                float margin_factor = 0.15f;         // Expand ROI by 15% in each direction
                float max_depth = 5.0f;              // Maximum depth in meters
                float min_depth = 0.3f;              // Minimum depth in meters
                bool filter_by_depth = true;         // Filter points outside depth range
            };

            explicit DoorConcept(const RoboCompCamera360RGBD::Camera360RGBDPrxPtr &camera_360rgbd_proxy_)
            {
                camera360rgbd_proxy = camera_360rgbd_proxy_;
                std::string model_path = "best.onnx";
                yolo_detector = std::make_unique<YOLODetectorONNX>(model_path, std::vector<std::string>{},
                                     0.25f, 0.45f, 640, true);
            }

            /**
             * @brief Main update cycle with top-down prediction
             *
             * Flow:
             * 1. If no door tracked: run full YOLO detection
             * 2. If door exists: predict new pose, extract ROI from prediction
             * 3. Check tracking quality, redetect if necessary
             * 4. Optimize door parameters with extracted points
             */
            std::optional<Result> update(const RoboCompCamera360RGBD::TRGBD &rgbd,
                                         const Eigen::Vector3f& robot_motion = Eigen::Vector3f::Zero());

            /**
             * @brief Full YOLO-based detection (used for initialization or recovery)
             */
            std::vector<DoorModel> detect(const RoboCompCamera360RGBD::TRGBD &rgbd);

            /**
             * @brief Extract ROI points using door model prediction (top-down approach)
             *
             * Projects the 3D door bounding box to image space, then extracts
             * depth points within that region.
             *
             * @param rgbd Current RGBD frame
             * @param door Door model with predicted pose
             * @return Points within the predicted door ROI
             */
            std::vector<Eigen::Vector3f> extract_roi_from_model(
                const RoboCompCamera360RGBD::TRGBD &rgbd,
                const DoorModel& door);

            /**
             * @brief Compute 2D bounding box from 3D door model projection
             *
             * Projects door corners to image space and computes enclosing rectangle
             */
            cv::Rect compute_projected_roi(const DoorModel& door,
                                           int image_width, int image_height);

            /**
             * @brief Initialize door from YOLO detection
             */
            void initialize(const RoboCompLidar3D::TPoints& roi_points,
                           float initial_width = 1.0f,
                           float initial_height = 2.0f,
                           float initial_angle = 0.0f);

            /**
             * @brief Get current door model
             */
            std::shared_ptr<DoorModel> get_model() const { return door; }

            /**
             * @brief Get optimization configuration (for tuning)
             */
            OptimizationConfig& get_config() { return config_; }

            /**
             * @brief Get ROI extraction configuration
             */
            ROIConfig& get_roi_config() { return roi_config_; }

            /**
             * @brief Check if door has been initialized
             */
            bool is_initialized() const { return initialized_; }

            /**
             * @brief Check if tracking is currently active (vs. detection mode)
             */
            bool is_tracking() const { return tracking_active_; }

            /**
             * @brief Reset door concept (clear state)
             */
            void reset();

            /**
             * @brief Force redetection on next update
             */
            void force_redetection() { tracking_active_ = false; }

        private:
            OptimizationConfig config_;
            ROIConfig roi_config_;
            bool initialized_ = false;
            bool tracking_active_ = false;
            int consecutive_tracking_failures_ = 0;
            static constexpr int MAX_TRACKING_FAILURES = 3;

            std::unique_ptr<YOLODetectorONNX> yolo_detector;
            RoboCompCamera360RGBD::Camera360RGBDPrxPtr camera360rgbd_proxy;
            std::shared_ptr<DoorModel> door = nullptr;

            RoboCompCamera360RGBD::TRGBD read_image();

            /**
             * @brief Predict step: adjust door pose for robot motion
             */
            void predict_step(const Eigen::Vector3f& robot_motion);

            /**
             * @brief Update step: optimize door parameters using measurements
             */
            Result update_step(const torch::Tensor& points_tensor);

            /**
             * @brief Convert point cloud to tensor
             */
            torch::Tensor convert_points_to_tensor(const std::vector<Eigen::Vector3f> &points);

            /**
             * @brief Compute measurement loss (SDF-based)
             */
            torch::Tensor compute_measurement_loss(const torch::Tensor& points_tensor);

            /**
             * @brief Compute geometry regularization loss
             */
            torch::Tensor compute_geometry_regularization();

            /**
             * @brief Estimate uncertainty from Hessian
             */
            torch::Tensor estimate_uncertainty(const torch::Tensor& points_tensor,
                                              float final_loss);

            /**
             * @brief Run gradient descent optimization
             */
            struct OptimizationResult
            {
                float total_loss = 0.0f;
                float measurement_loss = 0.0f;
                int iterations = 0;
                bool converged = false;
            };

            OptimizationResult run_optimization(const torch::Tensor& points_tensor,
                                               torch::optim::Optimizer& optimizer);

            /**
             * @brief Check if tracking quality is sufficient
             */
            bool check_tracking_quality(const Result& result);

            /**
             * @brief Project 3D point to 2D image coordinates (equirectangular)
             *
             * For 360° camera, uses equirectangular projection
             */
            cv::Point2f project_point_to_image(const Eigen::Vector3f& point_3d,
                                               int image_width, int image_height);
    };
};

#endif // DOOR_CONCEPT_H