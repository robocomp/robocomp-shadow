#include "door_projection.h"
#include <algorithm>

namespace
{
constexpr float kFrameThicknessEps = 1e-6f;
}

std::vector<Eigen::Vector3f> DoorProjection::sampleFrameSurface(
    const std::shared_ptr<DoorModel> &door,
    const Eigen::Vector3f& camera_pos,
    int num_samples)
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

    const Eigen::Vector2f door_center_xy(door_x, door_y);
    const Eigen::Vector2f door_forward(sin_t, cos_t);
    const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
    const Eigen::Vector2f to_camera_xy = camera_xy - door_center_xy;

    const float dot = to_camera_xy.dot(door_forward);
    const float front_y = (dot >= 0) ? -fd / 2.0f : fd / 2.0f;

    auto transformToCamera = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    {
        const float rx = cos_t * lx - sin_t * ly + door_x - camera_pos.x();
        const float ry = sin_t * lx + cos_t * ly + door_y - camera_pos.y();
        const float rz = lz + door_z - camera_pos.z();
        return {rx, ry, rz};
    };

    const float left_x = -width / 2.0f - ft / 2.0f;
    for (int i = 0; i <= num_samples; ++i)
    {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformToCamera(left_x - ft / 2, front_y, z));
        points.push_back(transformToCamera(left_x + ft / 2, front_y, z));
    }
    for (int i = 0; i <= std::max(1, num_samples / 4); ++i)
    {
        const float factor = (num_samples / 4 == 0) ? 0.0f : static_cast<float>(i) / (num_samples / 4);
        const float x = left_x - ft / 2 + factor * ft;
        points.push_back(transformToCamera(x, front_y, 0));
        points.push_back(transformToCamera(x, front_y, height));
    }

    const float right_x = width / 2.0f + ft / 2.0f;
    for (int i = 0; i <= num_samples; ++i)
    {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformToCamera(right_x - ft / 2, front_y, z));
        points.push_back(transformToCamera(right_x + ft / 2, front_y, z));
    }
    for (int i = 0; i <= std::max(1, num_samples / 4); ++i)
    {
        const float factor = (num_samples / 4 == 0) ? 0.0f : static_cast<float>(i) / (num_samples / 4);
        const float x = right_x - ft / 2 + factor * ft;
        points.push_back(transformToCamera(x, front_y, 0));
        points.push_back(transformToCamera(x, front_y, height));
    }

    const float lintel_half_w = (width + 2 * ft) / 2.0f;
    for (int i = 0; i <= num_samples; ++i)
    {
        const float x = -lintel_half_w + (static_cast<float>(i) / num_samples) * 2 * lintel_half_w;
        points.push_back(transformToCamera(x, front_y, height));
        points.push_back(transformToCamera(x, front_y, height + ft));
        points.push_back(transformToCamera(x, -fd / 2, height));
        points.push_back(transformToCamera(x, fd / 2, height));
    }

    return points;
}

std::vector<Eigen::Vector3f> DoorProjection::sampleLeafSurface(
    const std::shared_ptr<DoorModel> &door,
    const Eigen::Vector3f& camera_pos,
    int num_samples)
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

    const float cos_a = std::cos(open_angle);
    const float sin_a = std::sin(open_angle);

    const float hinge_x = -width / 2.0f;

    auto transformLeafToCamera = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        const float rel_x = lx - hinge_x;
        const float rotated_x = hinge_x + cos_a * rel_x - sin_a * ly;
        const float rotated_y = sin_a * rel_x + cos_a * ly;
        const float rx = cos_t * rotated_x - sin_t * rotated_y + door_x - camera_pos.x();
        const float ry = sin_t * rotated_x + cos_t * rotated_y + door_y - camera_pos.y();
        const float rz = lz + door_z - camera_pos.z();
        return {rx, ry, rz};
    };

    auto transformLeafToWorld = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        const float rel_x = lx - hinge_x;
        const float rotated_x = hinge_x + cos_a * rel_x - sin_a * ly;
        const float rotated_y = sin_a * rel_x + cos_a * ly;
        const float rx = cos_t * rotated_x - sin_t * rotated_y + door_x;
        const float ry = sin_t * rotated_x + cos_t * rotated_y + door_y;
        const float rz = lz + door_z;
        return {rx, ry, rz};
    };

    const float leaf_center_local_x = 0.0f;
    const float leaf_center_local_z = height / 2.0f;
    const Eigen::Vector3f leaf_center_world = transformLeafToWorld(
        leaf_center_local_x, 0.0f, leaf_center_local_z);

    const float normal_local_x = sin_a;
    const float normal_local_y = cos_a;

    const Eigen::Vector2f leaf_normal_xy(
        cos_t * normal_local_x - sin_t * normal_local_y,
        sin_t * normal_local_x + cos_t * normal_local_y);

    const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
    const Eigen::Vector2f leaf_center_xy(leaf_center_world.x(), leaf_center_world.y());
    const Eigen::Vector2f to_camera_xy = camera_xy - leaf_center_xy;

    const float dot = to_camera_xy.dot(leaf_normal_xy);
    const float visible_y = (dot >= 0) ? lt / 2.0f : -lt / 2.0f;

    const float leaf_left = -width / 2.0f;
    const float leaf_right = width / 2.0f;

    for (int i = 0; i <= num_samples; ++i)
    {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformLeafToCamera(leaf_left, visible_y, z));
        points.push_back(transformLeafToCamera(leaf_right, visible_y, z));
    }

    for (int i = 0; i <= num_samples; ++i)
    {
        const float x = leaf_left + (static_cast<float>(i) / num_samples) * width;
        points.push_back(transformLeafToCamera(x, visible_y, 0));
        points.push_back(transformLeafToCamera(x, visible_y, height));
    }

    for (int i = 0; i <= num_samples; ++i)
    {
        const float x = leaf_left + (static_cast<float>(i) / num_samples) * width;
        points.push_back(transformLeafToCamera(x, visible_y, height * 0.4f));
        points.push_back(transformLeafToCamera(x, visible_y, height * 0.6f));
    }

    for (int i = 0; i <= num_samples; ++i)
    {
        const float z = (static_cast<float>(i) / num_samples) * height;
        points.push_back(transformLeafToCamera(0.0f, visible_y, z));
    }

    return points;
}

void DoorProjection::projectDoorOnImage(
    const std::shared_ptr<DoorModel> &door,
    cv::Mat& image,
    const Eigen::Vector3f& camera_pos,
    const cv::Scalar& frame_color,
    const cv::Scalar& leaf_color,
    int num_samples,
    int point_radius)
{
    if (image.empty())
        return;

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
    if (image.empty())
        return;

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
    if (image.empty())
        return;

    auto leaf_points = sampleLeafSurface(door, camera_pos, num_samples);
    EquirectangularProjection::projectPointsOnImage(leaf_points, image, color, point_radius);
}

std::vector<Eigen::Vector3f> DoorProjection::sampleDoorSurfaceDense(
    const std::shared_ptr<DoorModel> &door,
    const Eigen::Vector3f& camera_pos,
    int grid_resolution)
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

    const Eigen::Vector2f door_forward(sin_t, cos_t);
    const Eigen::Vector2f door_center_xy(door_x, door_y);
    const Eigen::Vector2f camera_xy(camera_pos.x(), camera_pos.y());
    const Eigen::Vector2f to_camera_xy = camera_xy - door_center_xy;

    const float dot = to_camera_xy.dot(door_forward);
    const float front_y = (dot >= 0) ? -fd / 2.0f : fd / 2.0f;

    auto transformToCamera = [&](float lx, float ly, float lz) -> Eigen::Vector3f {
        const float rx = cos_t * lx - sin_t * ly + door_x - camera_pos.x();
        const float ry = sin_t * lx + cos_t * ly + door_y - camera_pos.y();
        const float rz = lz + door_z - camera_pos.z();
        return {rx, ry, rz};
    };

    const float left_x_center = -width / 2.0f - ft / 2.0f;
    const int jamb_res = std::max(2, grid_resolution / 4);
    for (int i = 0; i < jamb_res; ++i)
    {
        for (int j = 0; j < grid_resolution; ++j)
        {
            const float x = left_x_center - ft / 2 + (static_cast<float>(i) / (jamb_res - 1)) * ft;
            const float z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
            points.push_back(transformToCamera(x, front_y, z));
        }
    }

    const float right_x_center = width / 2.0f + ft / 2.0f;
    for (int i = 0; i < jamb_res; ++i)
    {
        for (int j = 0; j < grid_resolution; ++j)
        {
            const float x = right_x_center - ft / 2 + (static_cast<float>(i) / (jamb_res - 1)) * ft;
            const float z = (static_cast<float>(j) / (grid_resolution - 1)) * height;
            points.push_back(transformToCamera(x, front_y, z));
        }
    }

    const float lintel_half_w = (width + 2 * ft) / 2.0f;
    const int lintel_res = std::max(2, grid_resolution / 6);
    for (int i = 0; i < grid_resolution; ++i)
    {
        for (int j = 0; j < lintel_res; ++j)
        {
            const float x = -lintel_half_w + (static_cast<float>(i) / (grid_resolution - 1)) * 2 * lintel_half_w;
            const float z = height + (static_cast<float>(j) / (lintel_res - 1)) * ft;
            points.push_back(transformToCamera(x, front_y, z));
        }
    }

    const float cos_a = std::cos(open_angle);
    const float sin_a = std::sin(open_angle);
    const float hinge_x = -width / 2.0f;

    const float normal_local_x = sin_a;
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

            points.push_back(transformToCamera(rotated_x, rotated_y, local_z));
        }
    }

    return points;
}

