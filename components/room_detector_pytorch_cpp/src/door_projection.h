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
        int num_samples = 20)
    {
        std::vector<Eigen::Vector3f> points;

        const auto params = door->get_door_parameters();
        const float door_x = params[0];
        const float door_y = params[1];
        const float door_z = params[2];
        const float theta = params[3];
        const float width = params[4];
        const float height = params[5];

        const float ft = door->frame_thickness_;
        const float fd = door->frame_depth_;

        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);

        // Door center position (only X-Y for horizontal culling)
        const Eigen::Vector2f door_center_xy(door_x, door_y);

        // Door's local Y axis (forward direction) in world frame - horizontal only
        const Eigen::Vector2f door_forward(sin_t, cos_t);

        // Vector from door to camera - HORIZONTAL ONLY for vertical surfaces
        const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
        const Eigen::Vector2f to_camera_xy = camera_xy - door_center_xy;

        // Determine which side of the door the camera is on (horizontal plane only)
        const float dot = to_camera_xy.dot(door_forward);
        const float front_y = (dot >= 0) ? -fd / 2.0f : fd / 2.0f;  // Show front face facing camera

        auto transformToRobot = [&](float lx, float ly, float lz) -> Eigen::Vector3f
        {
            const float rx = cos_t * lx - sin_t * ly + door_x;
            const float ry = sin_t * lx + cos_t * ly + door_y;
            const float rz = lz + door_z;
            return Eigen::Vector3f(rx, ry, rz);
        };

        // Left jamb - front face only
        const float left_x = -width / 2.0f - ft / 2.0f;
        for (int i = 0; i <= num_samples; ++i)
        {
            const float z = (static_cast<float>(i) / num_samples) * height;
            // Vertical edges of front face
            points.push_back(transformToRobot(left_x - ft / 2, front_y, z));
            points.push_back(transformToRobot(left_x + ft / 2, front_y, z));
        }
        // Horizontal edges of front face
        for (int i = 0; i <= num_samples / 4; ++i)
        {
            const float x = left_x - ft / 2 + (static_cast<float>(i) / (num_samples / 4)) * ft;
            points.push_back(transformToRobot(x, front_y, 0));
            points.push_back(transformToRobot(x, front_y, height));
        }

        // Right jamb - front face only
        const float right_x = width / 2.0f + ft / 2.0f;
        for (int i = 0; i <= num_samples; ++i)
        {
            const float z = (static_cast<float>(i) / num_samples) * height;
            points.push_back(transformToRobot(right_x - ft / 2, front_y, z));
            points.push_back(transformToRobot(right_x + ft / 2, front_y, z));
        }
        for (int i = 0; i <= num_samples / 4; ++i)
        {
            const float x = right_x - ft / 2 + (static_cast<float>(i) / (num_samples / 4)) * ft;
            points.push_back(transformToRobot(x, front_y, 0));
            points.push_back(transformToRobot(x, front_y, height));
        }

        // Lintel - front and bottom faces
        const float lintel_half_w = (width + 2 * ft) / 2.0f;
        for (int i = 0; i <= num_samples; ++i)
        {
            const float x = -lintel_half_w + (static_cast<float>(i) / num_samples) * 2 * lintel_half_w;
            // Front face edges
            points.push_back(transformToRobot(x, front_y, height));
            points.push_back(transformToRobot(x, front_y, height + ft));
            // Bottom face edge (always visible from below)
            points.push_back(transformToRobot(x, -fd / 2, height));
            points.push_back(transformToRobot(x, fd / 2, height));
        }

        return points;
    }

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
        int num_samples = 20)
    {
        std::vector<Eigen::Vector3f> points;

        const auto params = door->get_door_parameters();
        const float door_x = params[0];
        const float door_y = params[1];
        const float door_z = params[2];
        const float theta = params[3];
        const float width = params[4];
        const float height = params[5];
        const float open_angle = params[6];

        const float lt = door->leaf_thickness_;

        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);

        // Leaf opens INWARD: negative angle direction
        // When open_angle > 0, door swings into -Y direction (into the room)
        const float cos_a = std::cos(-open_angle);  // Negate for inward opening
        const float sin_a = std::sin(-open_angle);

        const float hinge_x = -width / 2.0f;

        auto transformLeafToRobot = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
            // Rotate around hinge (Z-axis)
            const float rel_x = lx - hinge_x;
            const float rotated_x = hinge_x + cos_a * rel_x - sin_a * ly;
            const float rotated_y = sin_a * rel_x + cos_a * ly;
            // Then to world frame
            const float rx = cos_t * rotated_x - sin_t * rotated_y + door_x;
            const float ry = sin_t * rotated_x + cos_t * rotated_y + door_y;
            const float rz = lz + door_z;
            return Eigen::Vector3f(rx, ry, rz);
        };

        // Compute leaf center in world frame (for backface culling reference)
        const float leaf_center_local_x = 0.0f;  // Center of leaf in door frame
        const float leaf_center_local_z = height / 2.0f;
        const Eigen::Vector3f leaf_center = transformLeafToRobot(
            leaf_center_local_x, 0.0f, leaf_center_local_z);

        // Leaf normal in door-local frame after hinge rotation
        // Original normal is (0, 1, 0) pointing in +Y
        // After rotation by -open_angle around Z:
        // Rotating (0,1,0) by angle alpha around Z gives (-sin(alpha), cos(alpha), 0)
        // With alpha = -open_angle: (-sin(-open_angle), cos(-open_angle), 0) = (sin(open_angle), cos(open_angle), 0)
        // Since cos_a = cos(-open_angle) = cos(open_angle) and sin_a = sin(-open_angle) = -sin(open_angle)
        // The rotated normal is (-sin_a, cos_a, 0) in door-local frame

        const float normal_local_x = -sin_a;
        const float normal_local_y = cos_a;

        // Transform normal to world frame (rotation only, no translation)
        const Eigen::Vector2f leaf_normal_xy(
            cos_t * normal_local_x - sin_t * normal_local_y,
            sin_t * normal_local_x + cos_t * normal_local_y);

        // Vector from leaf center to camera - HORIZONTAL ONLY
        const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
        const Eigen::Vector2f leaf_center_xy(leaf_center.x(), leaf_center.y());
        const Eigen::Vector2f to_camera_xy = camera_xy - leaf_center_xy;

        // Determine which face is visible (horizontal plane only)
        const float dot = to_camera_xy.dot(leaf_normal_xy);
        const float visible_y = (dot >= 0) ? lt / 2.0f : -lt / 2.0f;

        // Sample leaf edges on the visible face
        const float leaf_left = -width / 2.0f;
        const float leaf_right = width / 2.0f;

        // Vertical edges
        for (int i = 0; i <= num_samples; ++i)
        {
            const float z = (static_cast<float>(i) / num_samples) * height;
            // Left edge (hinge side)
            points.push_back(transformLeafToRobot(leaf_left, visible_y, z));
            // Right edge (free side)
            points.push_back(transformLeafToRobot(leaf_right, visible_y, z));
        }

        // Horizontal edges
        for (int i = 0; i <= num_samples; ++i)
        {
            const float x = leaf_left + (static_cast<float>(i) / num_samples) * width;
            // Bottom edge
            points.push_back(transformLeafToRobot(x, visible_y, 0));
            // Top edge
            points.push_back(transformLeafToRobot(x, visible_y, height));
        }

        // Add some internal structure lines (like door panels)
        // Horizontal mid-lines
        for (int i = 0; i <= num_samples; ++i)
        {
            const float x = leaf_left + (static_cast<float>(i) / num_samples) * width;
            points.push_back(transformLeafToRobot(x, visible_y, height * 0.4f));
            points.push_back(transformLeafToRobot(x, visible_y, height * 0.6f));
        }

        // Vertical mid-line
        for (int i = 0; i <= num_samples; ++i)
        {
            const float z = (static_cast<float>(i) / num_samples) * height;
            points.push_back(transformLeafToRobot(0.0f, visible_y, z));
        }

        return points;
    }

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
        const cv::Scalar& frame_color = cv::Scalar(265, 165, 0),   // BGR: Orange (R=255, G=165, B=0)
        const cv::Scalar& leaf_color = cv::Scalar(0, 255, 0),      // BGR: Green
        const int num_samples = 30,
        const int point_radius = 2)
    {
        if (image.empty())
            return;

        // Sample and project frame
        const auto frame_points = sampleFrameSurface(door, camera_pos, num_samples);
        EquirectangularProjection::projectPointsOnImage(frame_points, image, frame_color, point_radius);

        // Sample and project leaf
        const auto leaf_points = sampleLeafSurface(door, camera_pos, num_samples);
        EquirectangularProjection::projectPointsOnImage(leaf_points, image, leaf_color, point_radius);
    }

    /**
     * @brief Project only the door frame (no leaf)
     */
    static void projectFrameOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& color = cv::Scalar(0, 165, 255),
        const int num_samples = 30,
        const int point_radius = 2)
    {
        if (image.empty())
            return;

        auto frame_points = sampleFrameSurface(door, camera_pos, num_samples);
        EquirectangularProjection::projectPointsOnImage(frame_points, image, color, point_radius);
    }

    /**
     * @brief Project only the door leaf (no frame)
     */
    static void projectLeafOnImage(
        const std::shared_ptr<DoorModel> &door,
        cv::Mat& image,
        const Eigen::Vector3f& camera_pos = Eigen::Vector3f::Zero(),
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        int num_samples = 30,
        int point_radius = 2)
    {
        if (image.empty())
            return;

        auto leaf_points = sampleLeafSurface(door, camera_pos, num_samples);
        EquirectangularProjection::projectPointsOnImage(leaf_points, image, color, point_radius);
    }

    /**
     * @brief Sample door surface for dense visualization (with backface culling)
     */
    static std::vector<Eigen::Vector3f> sampleDoorSurfaceDense(
        const std::shared_ptr<DoorModel> &door,
        const Eigen::Vector3f& camera_pos,
        int grid_resolution = 30)
    {
        std::vector<Eigen::Vector3f> points;

        const auto params = door->get_door_parameters();
        const float door_x = params[0];
        const float door_y = params[1];
        const float door_z = params[2];
        const float theta = params[3];
        const float width = params[4];
        const float height = params[5];
        const float open_angle = params[6];

        const float ft = door->frame_thickness_;
        const float fd = door->frame_depth_;
        const float lt = door->leaf_thickness_;

        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);

        // Door forward direction (horizontal only)
        const Eigen::Vector2f door_forward(sin_t, cos_t);
        const Eigen::Vector2f door_center_xy(door_x, door_y);
        const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
        const Eigen::Vector2f to_camera_xy = camera_xy - door_center_xy;

        const float dot = to_camera_xy.dot(door_forward);
        const float front_y = (dot >= 0) ? -fd / 2.0f : fd / 2.0f;

        auto transformToRobot = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
            const float rx = cos_t * lx - sin_t * ly + door_x;
            const float ry = sin_t * lx + cos_t * ly + door_y;
            const float rz = lz + door_z;
            return Eigen::Vector3f(rx, ry, rz);
        };

        // Sample front face of left jamb
        const float left_x_center = -width / 2.0f - ft / 2.0f;
        for (int i = 0; i < grid_resolution / 4; ++i)
        {
            for (int j = 0; j < grid_resolution; ++j)
            {
                const float x = left_x_center - ft / 2 + (static_cast<float>(i) / (grid_resolution / 4 - 1)) * ft;
                const float z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
                points.push_back(transformToRobot(x, front_y, z));
            }
        }

        // Sample front face of right jamb
        const float right_x_center = width / 2.0f + ft / 2.0f;
        for (int i = 0; i < grid_resolution / 4; ++i)
        {
            for (int j = 0; j < grid_resolution; ++j)
            {
                const float x = right_x_center - ft / 2 + (static_cast<float>(i) / (grid_resolution / 4 - 1)) * ft;
                const float z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
                points.push_back(transformToRobot(x, front_y, z));
            }
        }

        // Sample front face of lintel
        const float lintel_half_w = (width + 2 * ft) / 2.0f;
        for (int i = 0; i < grid_resolution; ++i)
        {
            for (int j = 0; j < grid_resolution / 6; ++j)
            {
                const float x = -lintel_half_w + (static_cast<float>(i) / (grid_resolution - 1)) * 2 * lintel_half_w;
                const float z = height + (static_cast<float>(j) / (grid_resolution / 6 - 1)) * ft;
                points.push_back(transformToRobot(x, front_y, z));
            }
        }

        // Sample leaf (inward opening)
        const float cos_a = std::cos(-open_angle);
        const float sin_a = std::sin(-open_angle);
        const float hinge_x = -width / 2.0f;

        // Leaf normal for backface culling (horizontal only)
        const float normal_local_x = -sin_a;
        const float normal_local_y = cos_a;
        const Eigen::Vector2f leaf_normal_xy(
            cos_t * normal_local_x - sin_t * normal_local_y,
            sin_t * normal_local_x + cos_t * normal_local_y);

        const float leaf_dot = to_camera_xy.dot(leaf_normal_xy);
        const float visible_ly = (leaf_dot >= 0) ? lt / 2.0f : -lt / 2.0f;

        for (int i = 0; i < grid_resolution; ++i)
        {
            for (int j = 0; j < grid_resolution; ++j)
            {
                const float local_x = -width / 2 + (static_cast<float>(i) / (grid_resolution - 1)) * width;
                const float local_z = (static_cast<float>(j) / (grid_resolution - 1)) * height;

                const float rel_x = local_x - hinge_x;
                const float rotated_x = hinge_x + cos_a * rel_x - sin_a * visible_ly;
                const float rotated_y = sin_a * rel_x + cos_a * visible_ly;

                points.push_back(transformToRobot(rotated_x, rotated_y, local_z));
            }
        }

        return points;
    }
};

#endif // DOOR_PROJECTION_H