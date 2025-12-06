/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 *
 *    Door model projection utilities for visualizing door model on 360 images
 */

#ifndef DOOR_PROJECTION_H
#define DOOR_PROJECTION_H

#include "door_model.h"
#include "equirectangular_projection.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>

/**
 * @brief ROI (Region of Interest) structure for predicted door bounding box
 */
struct PredictedROI
{
    int u_min = 0;
    int u_max = 0;
    int v_min = 0;
    int v_max = 0;

    bool valid = false;
    bool wraps_around = false;

    int u_min_2 = 0;
    int u_max_2 = 0;

    cv::Rect toCvRect() const;
    cv::Rect toCvRect2() const;
    PredictedROI expanded(int margin, int image_width, int image_height) const;
    int area() const;
    Eigen::Vector2i center() const;
};

/**
 * @brief Door projection utilities for 360 equirectangular images
 */
class DoorProjection
{
public:
    static std::vector<Eigen::Vector3f> sampleFrameSurface(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int num_samples = 20);

    static std::vector<Eigen::Vector3f> sampleLeafSurface(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int num_samples = 20);

    static void projectDoorOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& frame_color = cv::Scalar(255, 165, 0),
        const cv::Scalar& leaf_color = cv::Scalar(0, 255, 0),
        int num_samples = 30,
        int point_radius = 2);

    static void projectFrameOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& color = cv::Scalar(255, 165, 0),
        int num_samples = 30,
        int point_radius = 2);

    static void projectLeafOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        int num_samples = 30,
        int point_radius = 2);

    static PredictedROI predictROI(
        const std::shared_ptr<DoorModel> &door,
        int image_width,
        int image_height,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f(0.0f, 0.0f, 1.2f),
        int margin_pixels = 20,
        int num_samples = 15);

    static PredictedROI predictAndDrawROI(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f(0.0f, 0.0f, 1.2f),
        const cv::Scalar& color = cv::Scalar(0, 255, 255),
        int margin_pixels = 20,
        int line_thickness = 2);

    static bool isInsideROI(const PredictedROI& roi, int u, int v);

    static std::vector<size_t> filterPointsByROI(
        const std::vector<Eigen::Vector3f>& points_3d,
        const PredictedROI& roi,
        int image_width,
        int image_height);

    static std::vector<Eigen::Vector3f> sampleDoorSurfaceDense(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int grid_resolution = 30);
};

#endif // DOOR_PROJECTION_H