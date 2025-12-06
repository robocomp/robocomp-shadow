/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 */

#include "door_projection.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// PredictedROI Implementation
// ============================================================================

cv::Rect PredictedROI::toCvRect() const {
    if (!valid) return cv::Rect();
    return cv::Rect(u_min, v_min, u_max - u_min, v_max - v_min);
}

cv::Rect PredictedROI::toCvRect2() const {
    if (!valid || !wraps_around) return cv::Rect();
    return cv::Rect(u_min_2, v_min, u_max_2 - u_min_2, v_max - v_min);
}

PredictedROI PredictedROI::expanded(int margin, int image_width, int image_height) const {
    PredictedROI exp = *this;
    if (!valid) return exp;
    exp.u_min = std::max(0, u_min - margin);
    exp.u_max = std::min(image_width, u_max + margin);
    exp.v_min = std::max(0, v_min - margin);
    exp.v_max = std::min(image_height, v_max + margin);
    if (wraps_around) {
        exp.u_min_2 = std::max(0, u_min_2 - margin);
        exp.u_max_2 = std::min(image_width, u_max_2 + margin);
    }
    return exp;
}

int PredictedROI::area() const {
    if (!valid) return 0;
    int a = (u_max - u_min) * (v_max - v_min);
    if (wraps_around) a += (u_max_2 - u_min_2) * (v_max - v_min);
    return a;
}

Eigen::Vector2i PredictedROI::center() const {
    return Eigen::Vector2i((u_min + u_max) / 2, (v_min + v_max) / 2);
}

// ============================================================================
// DoorProjection - Frame Sampling
// ============================================================================

std::vector<Eigen::Vector3f> DoorProjection::sampleFrameSurface(
    const std::shared_ptr<DoorModel> &door,
    const Eigen::Vector3f& camera_pos,
    int num_samples)
{
    std::vector<Eigen::Vector3f> points;
    const auto params = door->get_door_parameters();
    const float door_x = params[0], door_y = params[1], door_z = params[2];
    const float theta = params[3], width = params[4], height = params[5];
    const float ft = door->frame_thickness_, fd = door->frame_depth_;
    const float cos_t = std::cos(theta), sin_t = std::sin(theta);

    const Eigen::Vector2f door_center_xy(door_x, door_y);
    const Eigen::Vector2f door_forward(sin_t, cos_t);
    const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
    const Eigen::Vector2f to_camera_xy = camera_xy - door_center_xy;
    const float dot = to_camera_xy.dot(door_forward);
    const float front_y = (dot >= 0) ? -fd / 2.0f : fd / 2.0f;

    auto transformToCamera = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        return Eigen::Vector3f(
            cos_t * lx - sin_t * ly + door_x - camera_pos.x(),
            sin_t * lx + cos_t * ly + door_y - camera_pos.y(),
            lz + door_z - camera_pos.z());
    };

    // Left jamb
    const float left_x = -width / 2.0f - ft / 2.0f;
    for (int i = 0; i <= num_samples; ++i) {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformToCamera(left_x - ft/2, front_y, z));
        points.push_back(transformToCamera(left_x + ft/2, front_y, z));
    }
    for (int i = 0; i <= num_samples/4; ++i) {
        const float x = left_x - ft/2 + (static_cast<float>(i) / (num_samples/4)) * ft;
        points.push_back(transformToCamera(x, front_y, 0));
        points.push_back(transformToCamera(x, front_y, height));
    }

    // Right jamb
    const float right_x = width / 2.0f + ft / 2.0f;
    for (int i = 0; i <= num_samples; ++i) {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformToCamera(right_x - ft/2, front_y, z));
        points.push_back(transformToCamera(right_x + ft/2, front_y, z));
    }
    for (int i = 0; i <= num_samples/4; ++i) {
        const float x = right_x - ft/2 + (static_cast<float>(i) / (num_samples/4)) * ft;
        points.push_back(transformToCamera(x, front_y, 0));
        points.push_back(transformToCamera(x, front_y, height));
    }

    // Lintel
    const float lintel_half_w = (width + 2*ft) / 2.0f;
    for (int i = 0; i <= num_samples; ++i) {
        const float x = -lintel_half_w + (static_cast<float>(i) / num_samples) * 2 * lintel_half_w;
        points.push_back(transformToCamera(x, front_y, height));
        points.push_back(transformToCamera(x, front_y, height + ft));
        points.push_back(transformToCamera(x, -fd/2, height));
        points.push_back(transformToCamera(x, fd/2, height));
    }
    return points;
}

// ============================================================================
// DoorProjection - Leaf Sampling
// ============================================================================

std::vector<Eigen::Vector3f> DoorProjection::sampleLeafSurface(
    const std::shared_ptr<DoorModel> &door,
    const Eigen::Vector3f& camera_pos,
    int num_samples)
{
    std::vector<Eigen::Vector3f> points;
    const auto params = door->get_door_parameters();
    const float door_x = params[0], door_y = params[1], door_z = params[2];
    const float theta = params[3], width = params[4], height = params[5];
    const float open_angle = params[6];
    const float lt = door->leaf_thickness_;
    const float cos_t = std::cos(theta), sin_t = std::sin(theta);
    const float cos_a = std::cos(open_angle), sin_a = std::sin(open_angle);
    const float hinge_x = -width / 2.0f;

    auto transformLeafToCamera = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        const float rel_x = lx - hinge_x;
        const float rotated_x = hinge_x + cos_a * rel_x - sin_a * ly;
        const float rotated_y = sin_a * rel_x + cos_a * ly;
        return Eigen::Vector3f(
            cos_t * rotated_x - sin_t * rotated_y + door_x - camera_pos.x(),
            sin_t * rotated_x + cos_t * rotated_y + door_y - camera_pos.y(),
            lz + door_z - camera_pos.z());
    };

    auto transformLeafToWorld = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        const float rel_x = lx - hinge_x;
        const float rotated_x = hinge_x + cos_a * rel_x - sin_a * ly;
        const float rotated_y = sin_a * rel_x + cos_a * ly;
        return Eigen::Vector3f(
            cos_t * rotated_x - sin_t * rotated_y + door_x,
            sin_t * rotated_x + cos_t * rotated_y + door_y,
            lz + door_z);
    };

    const Eigen::Vector3f leaf_center_world = transformLeafToWorld(0.0f, 0.0f, height/2.0f);
    const float normal_local_x = -sin_a, normal_local_y = cos_a;
    const Eigen::Vector2f leaf_normal_xy(
        cos_t * normal_local_x - sin_t * normal_local_y,
        sin_t * normal_local_x + cos_t * normal_local_y);
    const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
    const Eigen::Vector2f leaf_center_xy(leaf_center_world.x(), leaf_center_world.y());
    const Eigen::Vector2f to_camera_xy = camera_xy - leaf_center_xy;
    const float dot = to_camera_xy.dot(leaf_normal_xy);
    const float visible_y = (dot >= 0) ? lt / 2.0f : -lt / 2.0f;

    // Leaf edges
    for (int i = 0; i <= num_samples; ++i) {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformLeafToCamera(-width/2, visible_y, z));
        points.push_back(transformLeafToCamera(width/2, visible_y, z));
    }
    for (int i = 0; i <= num_samples; ++i) {
        const float x = -width/2 + (static_cast<float>(i) / num_samples) * width;
        points.push_back(transformLeafToCamera(x, visible_y, 0));
        points.push_back(transformLeafToCamera(x, visible_y, height));
    }
    // Panel details
    const float panel_margin = 0.08f;
    const float panel_left = -width/2 + panel_margin;
    const float panel_right = width/2 - panel_margin;
    const float panel_bottom = panel_margin;
    const float panel_top = height - panel_margin;
    for (int i = 0; i <= num_samples/2; ++i) {
        const float t = static_cast<float>(i) / (num_samples/2);
        points.push_back(transformLeafToCamera(panel_left, visible_y, panel_bottom + t*(panel_top-panel_bottom)));
        points.push_back(transformLeafToCamera(panel_right, visible_y, panel_bottom + t*(panel_top-panel_bottom)));
        points.push_back(transformLeafToCamera(panel_left + t*(panel_right-panel_left), visible_y, panel_bottom));
        points.push_back(transformLeafToCamera(panel_left + t*(panel_right-panel_left), visible_y, panel_top));
    }
    // Hinge edge
    for (int i = 0; i <= num_samples/3; ++i) {
        const float z = (static_cast<float>(i) / (num_samples/3)) * height;
        points.push_back(transformLeafToCamera(-width/2, 0.0f, z));
    }
    return points;
}

// ============================================================================
// DoorProjection - Image Projection Functions
// ============================================================================

void DoorProjection::projectDoorOnImage(
    const std::shared_ptr<DoorModel> &door,
    cv::Mat& image,
    const Eigen::Vector3f& camera_pos,
    const cv::Scalar& frame_color,
    const cv::Scalar& leaf_color,
    int num_samples,
    int point_radius)
{
    if (image.empty()) return;
    const auto frame_points = sampleFrameSurface(door, camera_pos, num_samples);
    EquirectangularProjection::projectPointsOnImage(frame_points, image, frame_color, point_radius);
    const auto leaf_points = sampleLeafSurface(door, camera_pos, num_samples);
    EquirectangularProjection::projectPointsOnImage(leaf_points, image, leaf_color, point_radius);
}

void DoorProjection::projectFrameOnImage(
    const std::shared_ptr<DoorModel> &door,
    cv::Mat& image,
    const Eigen::Vector3f& camera_pos,
    const cv::Scalar& color,
    int num_samples,
    int point_radius)
{
    if (image.empty()) return;
    auto frame_points = sampleFrameSurface(door, camera_pos, num_samples);
    EquirectangularProjection::projectPointsOnImage(frame_points, image, color, point_radius);
}

void DoorProjection::projectLeafOnImage(
    const std::shared_ptr<DoorModel> &door,
    cv::Mat& image,
    const Eigen::Vector3f& camera_pos,
    const cv::Scalar& color,
    int num_samples,
    int point_radius)
{
    if (image.empty()) return;
    auto leaf_points = sampleLeafSurface(door, camera_pos, num_samples);
    EquirectangularProjection::projectPointsOnImage(leaf_points, image, color, point_radius);
}

// ============================================================================
// DoorProjection - ROI Prediction
// ============================================================================

PredictedROI DoorProjection::predictROI(
    const std::shared_ptr<DoorModel> &door,
    int image_width,
    int image_height,
    const Eigen::Vector3f& camera_pos,
    int margin_pixels,
    int num_samples)
{
    PredictedROI roi;
    roi.valid = false;
    roi.wraps_around = false;

    auto frame_points = sampleFrameSurface(door, camera_pos, num_samples);
    auto leaf_points = sampleLeafSurface(door, camera_pos, num_samples);

    std::vector<Eigen::Vector3f> all_points;
    all_points.reserve(frame_points.size() + leaf_points.size());
    all_points.insert(all_points.end(), frame_points.begin(), frame_points.end());
    all_points.insert(all_points.end(), leaf_points.begin(), leaf_points.end());

    if (all_points.empty()) return roi;

    auto projected = EquirectangularProjection::projectPoints(all_points, image_width, image_height);
    if (projected.empty()) return roi;

    int v_min = image_height, v_max = 0;
    std::vector<int> u_values;
    u_values.reserve(projected.size());

    for (const auto& p : projected) {
        u_values.push_back(p.x());
        v_min = std::min(v_min, p.y());
        v_max = std::max(v_max, p.y());
    }

    std::sort(u_values.begin(), u_values.end());

    int max_gap = 0, gap_start = 0;
    for (size_t i = 1; i < u_values.size(); ++i) {
        int gap = u_values[i] - u_values[i-1];
        if (gap > max_gap) { max_gap = gap; gap_start = u_values[i-1]; }
    }

    const int wrap_threshold = image_width / 2;
    if (max_gap > wrap_threshold) {
        roi.wraps_around = true;
        auto it = std::upper_bound(u_values.begin(), u_values.end(), gap_start);
        roi.u_min = (it != u_values.end()) ? *it : u_values.back();
        roi.u_max = u_values.back();
        roi.u_min_2 = u_values.front();
        roi.u_max_2 = gap_start;
    } else {
        roi.u_min = u_values.front();
        roi.u_max = u_values.back();
    }

    roi.v_min = v_min;
    roi.v_max = v_max;
    roi.valid = true;
    return roi.expanded(margin_pixels, image_width, image_height);
}

PredictedROI DoorProjection::predictAndDrawROI(
    const std::shared_ptr<DoorModel> &door,
    cv::Mat& image,
    const Eigen::Vector3f& camera_pos,
    const cv::Scalar& color,
    int margin_pixels,
    int line_thickness)
{
    if (image.empty()) return PredictedROI{};
    PredictedROI roi = predictROI(door, image.cols, image.rows, camera_pos, margin_pixels);
    if (roi.valid) {
        cv::rectangle(image, roi.toCvRect(), color, line_thickness);
        if (roi.wraps_around) cv::rectangle(image, roi.toCvRect2(), color, line_thickness);
    }
    return roi;
}

bool DoorProjection::isInsideROI(const PredictedROI& roi, int u, int v)
{
    if (!roi.valid) return false;
    if (v < roi.v_min || v > roi.v_max) return false;
    if (u >= roi.u_min && u <= roi.u_max) return true;
    if (roi.wraps_around && u >= roi.u_min_2 && u <= roi.u_max_2) return true;
    return false;
}

std::vector<size_t> DoorProjection::filterPointsByROI(
    const std::vector<Eigen::Vector3f>& points_3d,
    const PredictedROI& roi,
    int image_width,
    int image_height)
{
    std::vector<size_t> inside_indices;
    if (!roi.valid) return inside_indices;
    inside_indices.reserve(points_3d.size() / 4);
    for (size_t i = 0; i < points_3d.size(); ++i) {
        int u, v;
        if (EquirectangularProjection::projectPoint(points_3d[i], image_width, image_height, u, v)) {
            if (isInsideROI(roi, u, v)) inside_indices.push_back(i);
        }
    }
    return inside_indices;
}

// ============================================================================
// DoorProjection - Dense Surface Sampling
// ============================================================================

std::vector<Eigen::Vector3f> DoorProjection::sampleDoorSurfaceDense(
    const std::shared_ptr<DoorModel> &door,
    const Eigen::Vector3f& camera_pos,
    int grid_resolution)
{
    std::vector<Eigen::Vector3f> points;
    const auto params = door->get_door_parameters();
    const float door_x = params[0], door_y = params[1], door_z = params[2];
    const float theta = params[3], width = params[4], height = params[5];
    const float open_angle = params[6];
    const float ft = door->frame_thickness_, fd = door->frame_depth_, lt = door->leaf_thickness_;
    const float cos_t = std::cos(theta), sin_t = std::sin(theta);

    const Eigen::Vector2f door_forward(sin_t, cos_t);
    const Eigen::Vector2f door_center_xy(door_x, door_y);
    const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
    const Eigen::Vector2f to_camera_xy = camera_xy - door_center_xy;
    const float dot = to_camera_xy.dot(door_forward);
    const float front_y = (dot >= 0) ? -fd / 2.0f : fd / 2.0f;

    auto transformToCamera = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        return Eigen::Vector3f(
            cos_t * lx - sin_t * ly + door_x - camera_pos.x(),
            sin_t * lx + cos_t * ly + door_y - camera_pos.y(),
            lz + door_z - camera_pos.z());
    };

    // Left jamb
    const float left_x_center = -width / 2.0f - ft / 2.0f;
    for (int i = 0; i < grid_resolution/4; ++i) {
        for (int j = 0; j < grid_resolution; ++j) {
            const float x = left_x_center - ft/2 + (static_cast<float>(i) / (grid_resolution/4 - 1)) * ft;
            const float z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
            points.push_back(transformToCamera(x, front_y, z));
        }
    }

    // Right jamb
    const float right_x_center = width / 2.0f + ft / 2.0f;
    for (int i = 0; i < grid_resolution/4; ++i) {
        for (int j = 0; j < grid_resolution; ++j) {
            const float x = right_x_center - ft/2 + (static_cast<float>(i) / (grid_resolution/4 - 1)) * ft;
            const float z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
            points.push_back(transformToCamera(x, front_y, z));
        }
    }

    // Lintel
    const float lintel_half_w = (width + 2*ft) / 2.0f;
    for (int i = 0; i < grid_resolution; ++i) {
        for (int j = 0; j < grid_resolution/6; ++j) {
            const float x = -lintel_half_w + (static_cast<float>(i) / (grid_resolution - 1)) * 2 * lintel_half_w;
            const float z = height + (static_cast<float>(j) / (grid_resolution/6 - 1)) * ft;
            points.push_back(transformToCamera(x, front_y, z));
        }
    }

    // Leaf
    const float cos_a = std::cos(open_angle), sin_a = std::sin(open_angle);
    const float hinge_x = -width / 2.0f;
    const float normal_local_x = -sin_a, normal_local_y = cos_a;
    const Eigen::Vector2f leaf_normal_xy(
        cos_t * normal_local_x - sin_t * normal_local_y,
        sin_t * normal_local_x + cos_t * normal_local_y);
    const float leaf_dot = to_camera_xy.dot(leaf_normal_xy);
    const float visible_ly = (leaf_dot >= 0) ? lt / 2.0f : -lt / 2.0f;

    for (int i = 0; i < grid_resolution; ++i) {
        for (int j = 0; j < grid_resolution; ++j) {
            const float local_x = -width/2 + (static_cast<float>(i) / (grid_resolution - 1)) * width;
            const float local_z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
            const float rel_x = local_x - hinge_x;
            const float rotated_x = hinge_x + cos_a * rel_x - sin_a * visible_ly;
            const float rotated_y = sin_a * rel_x + cos_a * visible_ly;
            points.push_back(transformToCamera(rotated_x, rotated_y, local_z));
        }
    }
    return points;
}