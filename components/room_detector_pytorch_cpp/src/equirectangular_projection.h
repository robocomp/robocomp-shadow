/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 *
 *    Equirectangular projection utilities for projecting 3D points onto 360° images
 *
 *    Coordinate system:
 *      X+ = right
 *      Y+ = forward (maps to horizontal center of image)
 *      Z+ = up
 */

#ifndef EQUIRECTANGULAR_PROJECTION_H
#define EQUIRECTANGULAR_PROJECTION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

class EquirectangularProjection
    {
    public:
        /**
         * @brief Project 3D points onto a 2D equirectangular image
         *
         * Coordinate system:
         *   X+ = right
         *   Y+ = forward (maps to horizontal center of image)
         *   Z+ = up
         *
         * @param points_3d Vector of 3D points in robot/camera frame
         * @param image Background equirectangular image (will be modified)
         * @param color Color for projected points (BGR format)
         * @param point_radius Radius of drawn points (default 2)
         */
        static void projectPointsOnImage(
            const std::vector<Eigen::Vector3f>& points_3d,
            cv::Mat& image,
            const cv::Scalar& color = cv::Scalar(0, 255, 0),
            int point_radius = 2);

        /**
         * @brief Project 3D points and return 2D pixel coordinates
         *
         * @param points_3d Vector of 3D points
         * @param image_width Width of equirectangular image
         * @param image_height Height of equirectangular image
         * @return Vector of 2D pixel coordinates (u, v) for valid points
         */
        static std::vector<Eigen::Vector2i> projectPoints(
            const std::vector<Eigen::Vector3f>& points_3d,
            int image_width,
            int image_height);

        /**
         * @brief Project a single 3D point to 2D pixel coordinates
         *
         * @param point_3d 3D point in robot/camera frame
         * @param image_width Width of equirectangular image
         * @param image_height Height of equirectangular image
         * @param[out] u_out Horizontal pixel coordinate
         * @param[out] v_out Vertical pixel coordinate
         * @return true if projection is valid, false if point is at origin
         */
        static bool projectPoint(
            const Eigen::Vector3f& point_3d,
            int image_width,
            int image_height,
            int& u_out,
            int& v_out);

        /**
         * @brief Inverse projection: convert 2D pixel to 3D ray direction
         *
         * @param u Horizontal pixel coordinate
         * @param v Vertical pixel coordinate
         * @param image_width Width of equirectangular image
         * @param image_height Height of equirectangular image
         * @return Unit vector representing ray direction in 3D
         */
        static Eigen::Vector3f unprojectPixel(
            int u,
            int v,
            int image_width,
            int image_height);
};

#endif // EQUIRECTANGULAR_PROJECTION_H
