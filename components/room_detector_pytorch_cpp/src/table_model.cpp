/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 */

// Undefine Qt macros before including PyTorch headers
#ifdef slots
#undef slots
#endif
#ifdef signals
#undef signals
#endif
#ifdef emit
#undef emit
#endif

#include "table_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <QDebug>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void TableModel::init(const std::vector<Eigen::Vector3f>& roi_points_,
                      const cv::Rect &roi_,
                      unsigned int id_,
                      const std::string &label_,
                      float initial_width,
                      float initial_depth,
                      float initial_height)
{
    // Store metadata
    roi_points = roi_points_;
    roi = roi_;
    id = id_;
    label = label_;
    score = 1.0f;

    // Compute initial pose from point cloud centroid
    if (roi_points.empty())
    {
        qWarning() << "TableModel::init() - No points in ROI, using default pose";
        table_position_ = torch::tensor({0.0f, 1.0f, initial_height/2.0f}, torch::requires_grad(true));
        table_theta_ = torch::tensor({0.0f}, torch::requires_grad(true));
    }
    else
    {
        // Compute centroid
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& p : roi_points)
            centroid += p;
        centroid /= static_cast<float>(roi_points.size());

        // Initialize position at centroid (z at tabletop level)
        // For height estimation, use max z from points
        float max_z = roi_points[0].z();
        for (const auto& p : roi_points)
            max_z = std::max(max_z, p.z());

        table_position_ = torch::tensor({centroid.x(), centroid.y(), max_z},
                                       torch::requires_grad(true));

        // Initial orientation: align with robot's Y-axis (forward)
        table_theta_ = torch::tensor({0.0f}, torch::requires_grad(true));
    }

    // Initialize geometry
    table_width_ = torch::tensor({initial_width}, torch::requires_grad(true));
    table_depth_ = torch::tensor({initial_depth}, torch::requires_grad(true));
    table_height_ = torch::tensor({initial_height}, torch::requires_grad(true));
    top_thickness_ = torch::tensor({0.05f}, torch::requires_grad(true));  // 5cm tabletop

    // Register parameters
    register_parameter("table_position", table_position_);
    register_parameter("table_theta", table_theta_);
    register_parameter("table_width", table_width_);
    register_parameter("table_depth", table_depth_);
    register_parameter("table_height", table_height_);
    register_parameter("top_thickness", top_thickness_);

    qInfo() << "TableModel" << id << "initialized at position:"
            << table_position_[0].item<float>()
            << table_position_[1].item<float>()
            << table_position_[2].item<float>()
            << "theta:" << table_theta_[0].item<float>();
}

torch::Tensor TableModel::transform_to_table_frame(const torch::Tensor& points_robot) const
{
    // Transform points from robot frame to table local frame
    // 1. Translate by -table_position
    // 2. Rotate by -table_theta around Z-axis

    // Get table position [3]
    const torch::Tensor pos = table_position_;

    // Get rotation angle
    const torch::Tensor theta = table_theta_.squeeze();
    const torch::Tensor cos_theta = torch::cos(theta);
    const torch::Tensor sin_theta = torch::sin(theta);

    // Create 3D rotation matrix around Z-axis [3x3]
    const torch::Tensor rotation = torch::stack({
        torch::stack({cos_theta, -sin_theta, torch::zeros_like(theta)}),
        torch::stack({sin_theta, cos_theta, torch::zeros_like(theta)}),
        torch::stack({torch::zeros_like(theta), torch::zeros_like(theta), torch::ones_like(theta)})
    });

    // Translate points to table center: [N, 3] - [3]
    const torch::Tensor translated = points_robot - pos;

    // Rotate points: [N, 3] @ [3, 3]^T = [N, 3]
    const torch::Tensor points_table = torch::matmul(translated, rotation.transpose(0, 1));

    return points_table;
}

torch::Tensor TableModel::sdBox(const torch::Tensor& p, const torch::Tensor& b) const
{
    // SDF for axis-aligned box centered at origin
    // p: [N, 3] query points
    // b: [3] half-extents

    const torch::Tensor q = torch::abs(p) - b;
    const torch::Tensor outside = torch::norm(torch::clamp_min(q, 0.0), 2, /*dim=*/1);
    const torch::Tensor inside = torch::clamp_max(
        torch::max(torch::max(q.select(1, 0), q.select(1, 1)), q.select(1, 2)),
        0.0
    );

    return outside + inside;
}

torch::Tensor TableModel::sdCylinder(const torch::Tensor& p, float h, float r) const
{
    // SDF for vertical cylinder (Z-axis aligned) centered at origin
    // p: [N, 3] query points
    // h: cylinder height
    // r: cylinder radius

    // Distance in XY plane
    const torch::Tensor xy = torch::stack({p.select(1, 0), p.select(1, 1)}, 1);
    const torch::Tensor d_xy = torch::norm(xy, 2, /*dim=*/1) - r;

    // Distance along Z axis
    const torch::Tensor d_z = torch::abs(p.select(1, 2)) - h / 2.0f;

    // Combine distances
    const torch::Tensor outside = torch::norm(
        torch::stack({torch::clamp_min(d_xy, 0.0), torch::clamp_min(d_z, 0.0)}, 1),
        2, /*dim=*/1
    );
    const torch::Tensor inside = torch::clamp_max(torch::max(d_xy, d_z), 0.0);

    return outside + inside;
}

torch::Tensor TableModel::sdf_tabletop(const torch::Tensor& points_table) const
{
    // Tabletop is a box centered at origin (in table frame)
    // The table_position_ in robot frame is at the TOP of the tabletop
    // So in table local frame, tabletop center is at z = -top_thickness/2

    const float half_width = table_width_.item<float>() / 2.0f;
    const float half_depth = table_depth_.item<float>() / 2.0f;
    const float half_thickness = top_thickness_.item<float>() / 2.0f;

    // Offset points so tabletop center is at (0, 0, -half_thickness)
    torch::Tensor offset = torch::tensor({0.0f, 0.0f, -half_thickness}, points_table.options());
    torch::Tensor points_offset = points_table - offset;

    // Box half-extents
    torch::Tensor half_extents = torch::tensor(
        {half_width, half_depth, half_thickness},
        points_table.options()
    );

    return sdBox(points_offset, half_extents);
}

torch::Tensor TableModel::sdf_legs(const torch::Tensor& points_table) const
{
    // Four cylindrical legs at corners
    // In table local frame:
    // - Tabletop is at z = 0 (top surface)
    // - Legs extend downward from z = -top_thickness to z = -(table_height)
    // - Leg positions: (±width/2, ±depth/2)

    const float half_width = table_width_.item<float>() / 2.0f;
    const float half_depth = table_depth_.item<float>() / 2.0f;
    const float top_thickness = top_thickness_.item<float>();
    const float table_height = table_height_.item<float>();
    const float leg_height = table_height - top_thickness;
    const float leg_center_z = -(top_thickness + leg_height / 2.0f);

    // Leg corner positions in XY
    std::vector<std::pair<float, float>> leg_positions = {
        {half_width, half_depth},      // Front-right
        {half_width, -half_depth},     // Back-right
        {-half_width, half_depth},     // Front-left
        {-half_width, -half_depth}     // Back-left
    };

    // Compute SDF for each leg and take minimum (union)
    torch::Tensor min_dist = torch::full({points_table.size(0)}, 1e10f, points_table.options());

    for (const auto& [leg_x, leg_y] : leg_positions)
    {
        // Offset points to leg center
        torch::Tensor offset = torch::tensor({leg_x, leg_y, leg_center_z}, points_table.options());
        torch::Tensor points_leg = points_table - offset;

        // Compute cylinder SDF
        torch::Tensor leg_sdf = sdCylinder(points_leg, leg_height, leg_radius_);

        // Update minimum distance
        min_dist = torch::min(min_dist, leg_sdf);
    }

    return min_dist;
}

torch::Tensor TableModel::sdf_union(const torch::Tensor& d1, const torch::Tensor& d2) const
{
    return torch::min(d1, d2);
}

torch::Tensor TableModel::sdf(const torch::Tensor& points_robot) const
{
    // Transform to table local frame
    const torch::Tensor points_table = transform_to_table_frame(points_robot);

    // Compute SDF for tabletop and legs
    const torch::Tensor sdf_top = sdf_tabletop(points_table);
    const torch::Tensor sdf_leg = sdf_legs(points_table);

    // Union (minimum)
    return sdf_union(sdf_top, sdf_leg);
}

std::vector<float> TableModel::get_table_parameters() const
{
    auto pos_acc = table_position_.accessor<float, 1>();
    auto theta_acc = table_theta_.accessor<float, 1>();
    auto width_acc = table_width_.accessor<float, 1>();
    auto depth_acc = table_depth_.accessor<float, 1>();
    auto height_acc = table_height_.accessor<float, 1>();
    auto thickness_acc = top_thickness_.accessor<float, 1>();

    return {
        pos_acc[0], pos_acc[1], pos_acc[2],     // x, y, z
        theta_acc[0],                            // theta
        width_acc[0], depth_acc[0],              // width, depth
        height_acc[0], thickness_acc[0]          // height, top_thickness
    };
}

std::vector<float> TableModel::get_table_pose() const
{
    auto pos_acc = table_position_.accessor<float, 1>();
    auto theta_acc = table_theta_.accessor<float, 1>();
    return {pos_acc[0], pos_acc[1], pos_acc[2], theta_acc[0]};
}

std::vector<float> TableModel::get_table_geometry() const
{
    auto width_acc = table_width_.accessor<float, 1>();
    auto depth_acc = table_depth_.accessor<float, 1>();
    auto height_acc = table_height_.accessor<float, 1>();
    auto thickness_acc = top_thickness_.accessor<float, 1>();
    return {width_acc[0], depth_acc[0], height_acc[0], thickness_acc[0]};
}

std::vector<torch::Tensor> TableModel::parameters() const
{
    return {table_position_, table_theta_, table_width_, table_depth_,
            table_height_, top_thickness_};
}

std::vector<torch::Tensor> TableModel::get_pose_parameters() const
{
    return {table_position_, table_theta_};
}

std::vector<torch::Tensor> TableModel::get_geometry_parameters() const
{
    return {table_width_, table_depth_, table_height_, top_thickness_};
}

void TableModel::freeze_geometry()
{
    if (table_width_.defined()) table_width_.set_requires_grad(false);
    if (table_depth_.defined()) table_depth_.set_requires_grad(false);
    if (table_height_.defined()) table_height_.set_requires_grad(false);
    if (top_thickness_.defined()) top_thickness_.set_requires_grad(false);
}

void TableModel::unfreeze_geometry()
{
    if (table_width_.defined()) table_width_.set_requires_grad(true);
    if (table_depth_.defined()) table_depth_.set_requires_grad(true);
    if (table_height_.defined()) table_height_.set_requires_grad(true);
    if (top_thickness_.defined()) top_thickness_.set_requires_grad(true);
}

bool TableModel::is_geometry_frozen() const
{
    return table_width_.defined() && !table_width_.requires_grad();
}

void TableModel::set_table_position(float x, float y, float z)
{
    table_position_ = torch::tensor({x, y, z}, torch::requires_grad(true));
}

void TableModel::set_theta(float theta)
{
    table_theta_ = torch::tensor({theta}, torch::requires_grad(true));
}

void TableModel::set_pose(float x, float y, float z, float theta)
{
    set_table_position(x, y, z);
    set_theta(theta);
}

void TableModel::print_info() const
{
    const auto params = get_table_parameters();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== TABLE MODEL (ID: " << id << ") ===\n";
    std::cout << "  Position: (" << params[0] << ", " << params[1] << ", " << params[2] << ") m\n";
    std::cout << "  Orientation: " << params[3] << " rad (" << (params[3] * 180.0 / M_PI) << " deg)\n";
    std::cout << "  Dimensions: " << params[4] << " x " << params[5] << " x " << params[6] << " m (W x D x H)\n";
    std::cout << "  Top thickness: " << params[7] << " m\n";
    std::cout << "  Leg radius: " << leg_radius_ << " m\n";
    std::cout << "  Frozen: " << (is_geometry_frozen() ? "YES" : "NO") << "\n";
}
