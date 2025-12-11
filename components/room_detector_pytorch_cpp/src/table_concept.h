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

#ifndef TABLE_CONCEPT_H
#define TABLE_CONCEPT_H

#include <torch/torch.h>
#include <Eigen/Dense>
#include <Lidar3D.h>
#include "table_model.h"
#include "table_projection.h"
#include "common_types.h"
#include "yolo_detector_onnx.h"
#include <Camera360RGBD.h>
#include <QtCore>

namespace rc
{
    /**
     * @brief Table detection and tracking using Bayesian optimization
     *
     * Implements a predict-update cycle for table state estimation:
     * 1. PREDICT: Use robot motion to predict table pose relative to robot
     * 2. UPDATE: Optimize table parameters using LiDAR measurements within ROI
     *
     * Top-down prediction approach:
     * - First detection: Full YOLO detection to find tables
     * - Subsequent iterations: Use optimized table model to predict ROI,
     *   extract points directly, bypass YOLO unless tracking is lost
     */

    class TableConcept
    {
        public:
            struct Result
            {
                torch::Tensor covariance;           // Full parameter covariance
                std::vector<float> std_devs;        // Standard deviations
                float final_loss = 0.0f;            // Total optimization loss
                float measurement_loss = 0.0f;      // SDF fitting loss
                bool success = false;               // Whether optimization succeeded
                std::vector<float> optimized_params; // Final table parameters
                int num_points_used = 0;            // Number of LiDAR points in ROI
                float mean_residual = 0.0f;         // Mean SDF residual
                std::shared_ptr<TableModel> table;
                void print() const
                {
                    qInfo() << "==============================";
                    qInfo() << "Table " << table->id << "optimization result:"
                            << "	Final loss:" <<  final_loss
                            << "	Measurement loss:"  << measurement_loss
                            << "	Num points:" << num_points_used
                            << "	Success:" << success;
                    const auto params = optimized_params;
                    qInfo() << "Optimized table parameters:"
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
                float convergence_patience = 5;      // Iterations without improvement (reduced)
                float convergence_delta = 1e-4f;     // Minimum loss change (increased for faster exit)

                // Regularization
                bool use_geometry_regularization = true;
                float geometry_reg_weight = 0.5f;    // Strong prior on table dimensions

                // Standard table dimensions for regularization (meters)
                float typical_width = 1.0f;          // 100cm standard table width
                float typical_height = 0.75f;         // 75cm standard table height
                float size_std = 0.2f;              // 20cm tolerance (tighter)

                // Tracking quality thresholds
                float tracking_lost_threshold = 0.5f;   // Mean residual above this triggers redetection
                int min_points_for_tracking = 50;       // Minimum points to maintain tracking
                float roi_expansion_factor = 1.3f;      // Expand predicted ROI by this factor

                // Consensus prior (from factor graph)
                bool use_consensus_prior = true;        // Enable consensus prior in loss
                float consensus_prior_weight = 1.0f;    // Weight for position prior term
                float consensus_orientation_weight = 5.0f;  // Weight for orientation alignment (stronger)
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
                float camera_height = 1.2f;          // Camera/sensor height for projection (meters)
            };

            explicit TableConcept(const RoboCompCamera360RGBD::Camera360RGBDPrxPtr &camera_360rgbd_proxy_)
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
             * 1. If no table tracked: run full YOLO detection
             * 2. If table exists: predict new pose, extract ROI from prediction
             * 3. Check tracking quality, redetect if necessary
             * 4. Optimize table parameters with extracted points
             */
            std::optional<Result> update(const RoboCompCamera360RGBD::TRGBD &rgbd,
                                         const Eigen::Vector3f& robot_motion = Eigen::Vector3f::Zero());

            /**
             * @brief Full YOLO-based detection (used for initialization or recovery)
             */
            std::vector<TableModel> detect(const RoboCompCamera360RGBD::TRGBD &rgbd);

            /**
             * @brief Extract ROI points using table model prediction (top-down approach)
             *
             * Projects the 3D table bounding box to image space, then extracts
             * depth points within that region.
             *
             * @param rgbd Current RGBD frame
             * @param table Table model with predicted pose
             * @return Points within the predicted table ROI
             */
            std::vector<Eigen::Vector3f> extract_roi_from_model(
                const RoboCompCamera360RGBD::TRGBD &rgbd,
                const TableModel& table);

            /**
             * @brief Compute 2D bounding box from 3D table model projection
             *
             * Projects table corners to image space and computes enclosing rectangle
             */
            cv::Rect compute_projected_roi(const TableModel& table,
                                           int image_width, int image_height);

            /**
             * @brief Initialize table from YOLO detection
             */
            void initialize(const RoboCompLidar3D::TPoints& roi_points,
                           float initial_width = 1.0f,
                           float initial_height = 2.0f,
                           float initial_angle = 0.0f);

            /**
             * @brief Get current table model
             */
            std::shared_ptr<TableModel> get_model() const { return table; }

            /**
             * @brief Get optimization configuration (for tuning)
             */
            OptimizationConfig& get_config() { return config_; }

            /**
             * @brief Get ROI extraction configuration
             */
            ROIConfig& get_roi_config() { return roi_config_; }

            /**
             * @brief Check if table has been initialized
             */
            bool is_initialized() const { return initialized_; }

            /**
             * @brief Check if tracking is currently active (vs. detection mode)
             */
            bool is_tracking() const { return tracking_active_; }

            /**
             * @brief Reset table concept (clear state)
             */
            void reset();

            /**
             * @brief Force redetection on next update
             */
            void force_redetection() { tracking_active_ = false; }

            /**
             * @brief Set consensus prior from factor graph optimization
             *
             * The prior is a (pose, covariance) pair that constrains the table
             * to lie on a wall. This is computed by ConsensusManager after
             * running the factor graph optimization.
             *
             * @param pose Prior pose [x, y, theta] from consensus
             * @param covariance Prior covariance (3x3) from consensus marginals
             */
            void setConsensusPrior(const Eigen::Vector3f& pose, const Eigen::Matrix3f& covariance);

            /**
             * @brief Clear consensus prior (disable prior term)
             */
            void clearConsensusPrior();

            /**
             * @brief Check if consensus prior is set
             */
            bool hasConsensusPrior() const { return consensus_prior_set_; }

        private:
            OptimizationConfig config_;
            ROIConfig roi_config_;
            bool initialized_ = false;
            bool tracking_active_ = false;
            int consecutive_tracking_failures_ = 0;
            static constexpr int MAX_TRACKING_FAILURES = 3;

            // Consensus prior from factor graph
            bool consensus_prior_set_ = false;
            Eigen::Vector3f consensus_prior_pose_;
            Eigen::Matrix3f consensus_prior_covariance_;
            Eigen::Matrix3f consensus_prior_info_;  // Inverse of covariance (precision)

            std::unique_ptr<YOLODetectorONNX> yolo_detector;
            RoboCompCamera360RGBD::Camera360RGBDPrxPtr camera360rgbd_proxy;
            std::shared_ptr<TableModel> table = nullptr;

            RoboCompCamera360RGBD::TRGBD read_image();

            /**
             * @brief Predict step: adjust table pose for robot motion
             */
            void predict_step(const Eigen::Vector3f& robot_motion);

            /**
             * @brief Update step: optimize table parameters using measurements
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
             * @brief Compute consensus prior loss (Mahalanobis distance to prior)
             */
            torch::Tensor compute_consensus_prior_loss();

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

#endif // TABLE_CONCEPT_H