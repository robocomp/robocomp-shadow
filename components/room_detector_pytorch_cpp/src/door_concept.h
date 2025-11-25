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
#include <Camera360RGB.h>
#include <QtCore>
namespace rc
{
    /**
     * @brief Door detection and tracking using Bayesian optimization
     *
     * Implements a predict-update cycle for door state estimation:
     * 1. PREDICT: Use robot motion to predict door pose relative to robot
     * 2. UPDATE: Optimize door parameters using LiDAR measurements within YOLO ROI
     *
     * Similar to RoomOptimizer but simpler:
     * - No adaptive mode switching (doors don't "freeze")
     * - Operates on ROI from YOLO detector
     * - Optimizes door pose, geometry, and articulation state
     * - Maintains uncertainty estimates
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
            };

            explicit DoorConcept(const RoboCompCamera360RGB::Camera360RGBPrxPtr &camera_360rgb_proxy_)
            {
                camera360rgb_proxy = camera_360rgb_proxy_;
                std::string model_path = "best.onnx";
                yolo_detector = std::make_unique<YOLODetectorONNX>(model_path, std::vector<std::string>{}, 0.25f, 0.45f, 640, true);
            }

            std::tuple<std::vector<DoorModel>, cv::Mat> detect()
            {
                const auto img = read_image();
                const auto doors_raw = yolo_detector->detect(img);
                qInfo() << __FUNCTION__ << "Detected" << doors_raw.size() << "doors";
                std::vector<DoorModel> doors;
                for (const auto &door : doors_raw)
                {
                    auto dm = DoorModel{};
                    dm.init(RoboCompLidar3D::TPoints{}, door.roi, door.classId, door.label,
                  1.0f, 2.0f, 0.0f);
                    doors.emplace_back(dm);
                }
                return std::make_tuple(doors, img.clone());
            }

            /**
             * @brief Initialize door from YOLO detection
             *
             * @param roi_points LiDAR points within YOLO bounding box
             * @param initial_width Initial door width estimate (meters)
             * @param initial_height Initial door height estimate (meters)
             * @param initial_angle Initial opening angle (radians)
             */
            void initialize(const RoboCompLidar3D::TPoints& roi_points,
                           float initial_width = 1.0f,
                           float initial_height = 2.0f,
                           float initial_angle = 0.0f);

            /**
             * @brief Main predict-update cycle
             *
             * @param roi_points Current LiDAR points in YOLO ROI
             * @param robot_motion Robot motion since last update (dx, dy, dtheta)
             * @return Optimization result with updated door state
             */
            Result update(const RoboCompLidar3D::TPoints& roi_points,
                         const Eigen::Vector3f& robot_motion = Eigen::Vector3f::Zero());

            /**
             * @brief Get current door model
             */
            DoorModel& get_model() { return door_model_; }
            const DoorModel& get_model() const { return door_model_; }

            /**
             * @brief Get optimization configuration (for tuning)
             */
            OptimizationConfig& get_config() { return config_; }

            /**
             * @brief Check if door has been initialized
             */
            bool is_initialized() const { return initialized_; }

            /**
             * @brief Reset door concept (clear state)
             */
            void reset();

        private:
            DoorModel door_model_;
            OptimizationConfig config_;
            bool initialized_ = false;
            std::unique_ptr<YOLODetectorONNX> yolo_detector;
            RoboCompCamera360RGB::Camera360RGBPrxPtr camera360rgb_proxy;
            std::vector<DoorModel> doors;

            cv::Mat read_image();

            /**
             * @brief Predict step: adjust door pose for robot motion
             * Robot moved -> door appears to move in opposite direction
             */
            void predict_step(const Eigen::Vector3f& robot_motion);

            /**
             * @brief Update step: optimize door parameters using measurements
             */
            Result update_step(const torch::Tensor& points_tensor);

            /**
             * @brief Convert point cloud to tensor
             */
            torch::Tensor convert_points_to_tensor(const RoboCompLidar3D::TPoints& points);

            /**
             * @brief Compute measurement loss (SDF-based)
             */
            torch::Tensor compute_measurement_loss(const torch::Tensor& points_tensor);

            /**
             * @brief Compute geometry regularization loss
             * Penalizes door dimensions far from typical values
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
    };
};

#endif // DOOR_CONCEPT_H
