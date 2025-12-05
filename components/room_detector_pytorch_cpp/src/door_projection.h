/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 *
 *    Door model projection utilities for visualizing door model on 360° images
 */

#ifndef DOOR_PROJECTION_H
#define DOOR_PROJECTION_H

#include "door_model.h"
#include "equirectangular_projection.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>

class DoorProjection
{
public:
    /**
     * @brief Sample 3D points from the door frame (jambs and lintel) with backface culling
     *
     * @param door Door model to sample from
     * @param camera_pos Camera/robot position in world frame (for backface culling)
     * @param num_samples Number of samples along each edge
     * @return Vector of 3D points on visible frame surfaces
     */
    static std::vector<Eigen::Vector3f> sampleFrameSurface(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int num_samples = 20);

    /**
     * @brief Sample 3D points from the door leaf with backface culling
     *
     * The leaf opens INWARD (negative Y direction in door frame, into the room)
     *
     * @param door Door model to sample from
     * @param camera_pos Camera/robot position (for backface culling)
     * @param num_samples Number of samples along each edge
     * @return Vector of 3D points on visible leaf surface
     */
    static std::vector<Eigen::Vector3f> sampleLeafSurface(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int num_samples = 20);

    /**
     * @brief Project door model onto equirectangular image with separate colors
     *
     * @param door Door model to project
     * @param image Background equirectangular image (will be modified)
     * @param camera_pos Camera position for backface culling (default: origin)
     * @param frame_color Color for frame (BGR format)
     * @param leaf_color Color for leaf (BGR format)
     * @param num_samples Number of samples per edge
     * @param point_radius Radius of projected points
     */
    static void projectDoorOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& frame_color = cv::Scalar(255, 165, 0),   // RGB: Orange
        const cv::Scalar& leaf_color = cv::Scalar(0, 255, 0),      // RGB: Green
        const int num_samples = 30,
        const int point_radius = 2);

    /**
     * @brief Project only the door frame (no leaf)
     */
    static void projectFrameOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& color = cv::Scalar(255, 165, 0),
        int num_samples = 30,
        int point_radius = 2);

    /**
     * @brief Project only the door leaf (no frame)
     */
    static void projectLeafOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        int num_samples = 30,
        int point_radius = 2);

    /**
     * @brief Sample door surface for dense visualization (with backface culling)
     */
    static std::vector<Eigen::Vector3f> sampleDoorSurfaceDense(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int grid_resolution = 30);
};

#endif // DOOR_PROJECTION_H

