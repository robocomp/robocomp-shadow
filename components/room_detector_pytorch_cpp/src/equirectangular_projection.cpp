#include "equirectangular_projection.h"
#include <algorithm>
#include <cmath>

namespace
{
constexpr float kEps = 1e-6f;
}

void EquirectangularProjection::projectPointsOnImage(
    const std::vector<Eigen::Vector3f>& points_3d,
    cv::Mat& image,
    const cv::Scalar& color,
    int point_radius)
{
    if (image.empty() || points_3d.empty())
        return;

    const int h = image.rows;
    const int w = image.cols;

    for (const auto& point : points_3d)
    {
        const float r = point.norm();
        if (r < kEps)
            continue;

        const float theta = std::atan2(point.x(), point.y());
        const float u = (theta / (2.0f * M_PI) + 0.5f) * w;

        const float phi = std::asin(-point.z() / r);
        const float v = (phi / M_PI + 0.5f) * h;

        const int u_idx = std::clamp(static_cast<int>(u), 0, w - 1);
        const int v_idx = std::clamp(static_cast<int>(v), 0, h - 1);

        if (point_radius <= 1)
            image.at<cv::Vec3b>(v_idx, u_idx) = cv::Vec3b(color[0], color[1], color[2]);
        else
            cv::circle(image, cv::Point(u_idx, v_idx), point_radius, color, -1);
    }
}

std::vector<Eigen::Vector2i> EquirectangularProjection::projectPoints(
    const std::vector<Eigen::Vector3f>& points_3d,
    int image_width,
    int image_height)
{
    std::vector<Eigen::Vector2i> projected;
    projected.reserve(points_3d.size());

    for (const auto& point : points_3d)
    {
        const float r = point.norm();
        if (r < kEps)
            continue;

        const float theta = std::atan2(point.x(), point.y());
        const float u = (theta / (2.0f * M_PI) + 0.5f) * image_width;

        const float phi = std::asin(-point.z() / r);
        const float v = (phi / M_PI + 0.5f) * image_height;

        const int u_idx = std::clamp(static_cast<int>(u), 0, image_width - 1);
        const int v_idx = std::clamp(static_cast<int>(v), 0, image_height - 1);

        projected.emplace_back(u_idx, v_idx);
    }

    return projected;
}

bool EquirectangularProjection::projectPoint(
    const Eigen::Vector3f& point_3d,
    int image_width,
    int image_height,
    int& u_out,
    int& v_out)
{
    const float r = point_3d.norm();
    if (r < kEps)
        return false;

    const float theta = std::atan2(point_3d.x(), point_3d.y());
    const float u = (theta / (2.0f * M_PI) + 0.5f) * image_width;

    const float phi = std::asin(-point_3d.z() / r);
    const float v = (phi / M_PI + 0.5f) * image_height;

    u_out = std::clamp(static_cast<int>(u), 0, image_width - 1);
    v_out = std::clamp(static_cast<int>(v), 0, image_height - 1);

    return true;
}

Eigen::Vector3f EquirectangularProjection::unprojectPixel(
    int u,
    int v,
    int image_width,
    int image_height)
{
    const float theta = ((static_cast<float>(u) / image_width) - 0.5f) * 2.0f * M_PI;
    const float phi = ((static_cast<float>(v) / image_height) - 0.5f) * M_PI;

    const float cos_phi = std::cos(phi);
    const float x = cos_phi * std::sin(theta);
    const float y = cos_phi * std::cos(theta);
    const float z = -std::sin(phi);

    return {x, y, z};
}

