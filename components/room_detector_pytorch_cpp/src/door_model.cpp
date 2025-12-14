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

#include "door_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <QDebug>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void DoorModel::init(const std::vector<Eigen::Vector3f>& roi_points_,
                     const cv::Rect &roi_,
                     unsigned int id_,
                     const std::string &label_,
                     float initial_width,
                     float initial_height,
                     float initial_angle)
{
    roi_points = roi_points_;
    roi = roi_;
    id = id_;
    label = label_;

    if (roi_points.empty())
    {
        return;
    }

    // Compute median of depth (Y+ is forward)
    std::vector<float> y_depths;
    y_depths.reserve(roi_points.size());
    for (const auto& p : roi_points)
        y_depths.push_back(p.y());

    std::ranges::sort(y_depths);
    const float median_depth = y_depths[y_depths.size() / 2];

    // Filter points inside a reasonable range (door opening + leaf is at most 1 meter deep)
    const float depth_tolerance = 1.0f;
    std::vector<Eigen::Vector3f> filtered_points;
    filtered_points.reserve(roi_points.size());

    for (const auto& p : roi_points)
    {
        const float y = p.y();
        if (std::abs(y - median_depth) <= depth_tolerance)
            filtered_points.push_back(p);
    }

    if (filtered_points.empty())
    {
        std::cout << "Warning: All points filtered out, using original set\n";
        filtered_points = roi_points;
    }

    std::cout << "Filtered points: " << filtered_points.size()
              << " of " << roi_points.size()
              << " (median depth: " << median_depth << " m)\n";

    // Estimate door center from ROI point cloud
    float x_sum = 0.0f, y_sum = 0.0f, z_sum = 0.0f;
    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();

    for (const auto& p : filtered_points)
    {
        float x = p.x();
        float y = p.y();
        float z = p.z();

        x_sum += x;
        y_sum += y;
        z_sum += z;

        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
        z_min = std::min(z_min, z);
        z_max = std::max(z_max, z);
    }

    const float n = static_cast<float>(filtered_points.size());
    const float center_x = x_sum / n;
    const float center_y = y_sum / n;
    // z position is at floor level (use z_min or assume 0)
    const float center_z = 0.0f;

    // Estimate door orientation from principal direction using PCA on XY plane
    // The door surface normal should point toward the robot (at origin)
    float initial_theta = 0.0f;

    if (filtered_points.size() >= 3)
    {
        // Compute covariance matrix in XY plane
        float cov_xx = 0.0f, cov_xy = 0.0f, cov_yy = 0.0f;
        for (const auto& p : filtered_points)
        {
            float dx = p.x() - center_x;
            float dy = p.y() - center_y;
            cov_xx += dx * dx;
            cov_xy += dx * dy;
            cov_yy += dy * dy;
        }
        cov_xx /= n;
        cov_xy /= n;
        cov_yy /= n;

        // Find principal eigenvector (direction of maximum variance = along door surface)
        // For 2x2 symmetric matrix, eigenvector can be computed directly
        float trace = cov_xx + cov_yy;
        float det = cov_xx * cov_yy - cov_xy * cov_xy;
        float discriminant = std::sqrt(std::max(0.0f, trace * trace / 4.0f - det));

        // Larger eigenvalue corresponds to the door surface direction
        float lambda1 = trace / 2.0f + discriminant;

        // Eigenvector for lambda1: (cov_xy, lambda1 - cov_xx) or (lambda1 - cov_yy, cov_xy)
        float ev_x, ev_y;
        if (std::abs(cov_xy) > 1e-6f)
        {
            ev_x = cov_xy;
            ev_y = lambda1 - cov_xx;
        }
        else
        {
            // Diagonal matrix - pick axis with larger variance
            ev_x = (cov_xx > cov_yy) ? 1.0f : 0.0f;
            ev_y = (cov_xx > cov_yy) ? 0.0f : 1.0f;
        }

        // Normalize
        float ev_len = std::sqrt(ev_x * ev_x + ev_y * ev_y);
        if (ev_len > 1e-6f)
        {
            ev_x /= ev_len;
            ev_y /= ev_len;
        }

        // The door normal is perpendicular to the surface direction
        // Normal = (-ev_y, ev_x) or (ev_y, -ev_x)
        // Door's Y+ axis should point AWAY from robot (the direction the door opens toward)
        // So we want the normal pointing away from robot (opposite to toward-robot)
        float normal_x = -ev_y;
        float normal_y = ev_x;

        // Check if normal points AWAY from robot (dot product with center should be positive)
        // center is the door position relative to robot, so normal should align with center direction
        if (normal_x * center_x + normal_y * center_y < 0)
        {
            normal_x = ev_y;
            normal_y = -ev_x;
        }

        // Theta is the angle of the door's forward direction (Y+ in door frame)
        // Door frame: Y+ is the normal direction (pointing away from robot)
        initial_theta = std::atan2(normal_x, normal_y);

        std::cout << "  PCA eigenvector (surface dir): (" << ev_x << ", " << ev_y << ")\n";
        std::cout << "  Door normal (away from robot): (" << normal_x << ", " << normal_y << ")\n";
        std::cout << "  Estimated theta: " << initial_theta << " rad (" << (initial_theta * 180.0 / M_PI) << " deg)\n";
    }

    // Refine size estimates from point cloud bounds if available
    // Width is along X, height is along Z
    const float estimated_width = std::max(x_max - x_min, initial_width);
    const float estimated_height = std::max(z_max - z_min, initial_height);

    std::cout << "\n=== DoorModel::init() ===\n";
    std::cout << "  ROI points: " << filtered_points.size() << "\n";
    std::cout << "  Point cloud bounds:\n";
    std::cout << "    X: [" << x_min << ", " << x_max << "] -> width: " << (x_max - x_min) << " m\n";
    std::cout << "    Y: [" << y_min << ", " << y_max << "] -> depth: " << (y_max - y_min) << " m\n";
    std::cout << "    Z: [" << z_min << ", " << z_max << "] -> height: " << (z_max - z_min) << " m\n";
    std::cout << "  Initial door pose: (" << center_x << ", " << center_y << ", "
              << center_z << ", theta=" << initial_theta << ")\n";
    std::cout << "  Initial geometry: " << estimated_width << " x " << estimated_height << " m (W x H)\n";
    std::cout << "  Initial opening angle: " << initial_angle << " rad ("
              << (initial_angle * 180.0 / M_PI) << " deg)\n";
    std::cout << "========================\n";

    // Initialize trainable parameters
    door_position_ = torch::tensor({center_x, center_y, center_z}, torch::requires_grad(true));
    door_theta_ = torch::tensor({initial_theta}, torch::requires_grad(true));
    door_width_ = torch::tensor({estimated_width}, torch::requires_grad(true));
    door_height_ = torch::tensor({estimated_height}, torch::requires_grad(true));
    opening_angle_ = torch::tensor({initial_angle}, torch::requires_grad(true));

    // Register parameters
    register_parameter("door_position", door_position_);
    register_parameter("door_theta", door_theta_);
    register_parameter("door_width", door_width_);
    register_parameter("door_height", door_height_);
    register_parameter("opening_angle", opening_angle_);
}

torch::Tensor DoorModel::transform_to_door_frame(const torch::Tensor& points_robot) const
{
    // Transform: p_door = R^T(theta) * (p_robot - door_position)
    // Rotation is around Z-axis (vertical)

    // 1. Translate to door origin
    const torch::Tensor translated = points_robot - door_position_;

    // 2. Rotate by -theta around Z-axis
    const torch::Tensor cos_theta = torch::cos(-door_theta_);
    const torch::Tensor sin_theta = torch::sin(-door_theta_);

    // Rotation matrix around Z-axis (inverse rotation):
    // [cos   sin  0]   [x]
    // [-sin  cos  0] * [y]
    // [0     0    1]   [z]

    const torch::Tensor x = translated.index({torch::indexing::Slice(), 0});
    const torch::Tensor y = translated.index({torch::indexing::Slice(), 1});
    const torch::Tensor z = translated.index({torch::indexing::Slice(), 2});

    const torch::Tensor x_door = cos_theta * x + sin_theta * y;
    const torch::Tensor y_door = -sin_theta * x + cos_theta * y;
    const torch::Tensor z_door = z;

    return torch::stack({x_door, y_door, z_door}, /*dim=*/1);
}

torch::Tensor DoorModel::sdBox(const torch::Tensor& p, const torch::Tensor& b) const
{
    // SDF for axis-aligned box
    // p: [N, 3] query points
    // b: [3] half-extents

    const torch::Tensor q = torch::abs(p) - b.to(p.device());

    // Outside distance
    const torch::Tensor outside_sq = torch::sum(
        torch::clamp(q, /*min=*/0.0) * torch::clamp(q, /*min=*/0.0),
        /*dim=*/1
    );
    const torch::Tensor outside_dist = torch::sqrt(outside_sq + 1e-10f);

    // Inside distance
    const torch::Tensor max_q = std::get<0>(torch::max(q, /*dim=*/1));
    const torch::Tensor inside_dist = torch::clamp_max(max_q, /*max=*/0.0);

    return outside_dist + inside_dist;
}

torch::Tensor DoorModel::sdf_frame(const torch::Tensor& points_door) const
{
    // Door frame consists of three boxes:
    // 1. Top lintel (horizontal beam at top)
    // 2. Left jamb (vertical post)
    // 3. Right jamb (vertical post)
    //
    // Coordinate system:
    //   X = width direction (left jamb at -w/2, right at +w/2)
    //   Y = depth direction (frame centered at y=0)
    //   Z = height direction (floor at z=0, top at z=height)

    const float w = door_width_.item<float>();
    const float h = door_height_.item<float>();
    const float ft = frame_thickness_;
    const float fd = frame_depth_;

    // Top lintel: spans full width including jambs
    // Half-extents: ((w + 2*ft)/2, fd/2, ft/2)
    // Center: (0, 0, h + ft/2)
    const torch::Tensor b_top = torch::tensor({(w + 2*ft) / 2.0f, fd / 2.0f, ft / 2.0f});
    const torch::Tensor c_top = torch::tensor({0.0f, 0.0f, h + ft / 2.0f});
    const torch::Tensor d_top = sdBox(points_door - c_top, b_top);

    // Left jamb
    // Half-extents: (ft/2, fd/2, h/2)
    // Center: (-w/2 - ft/2, 0, h/2)
    const torch::Tensor b_jamb = torch::tensor({ft / 2.0f, fd / 2.0f, h / 2.0f});
    const torch::Tensor c_left = torch::tensor({-w/2.0f - ft/2.0f, 0.0f, h / 2.0f});
    const torch::Tensor d_left = sdBox(points_door - c_left, b_jamb);

    // Right jamb
    // Center: (w/2 + ft/2, 0, h/2)
    const torch::Tensor c_right = torch::tensor({w/2.0f + ft/2.0f, 0.0f, h / 2.0f});
    const torch::Tensor d_right = sdBox(points_door - c_right, b_jamb);

    // Union of three boxes (minimum distance)
    return torch::minimum(d_top, torch::minimum(d_left, d_right));
}

torch::Tensor DoorModel::sdf_leaf(const torch::Tensor& points_door) const
{
    // Door leaf rotates around hinge at left edge
    // Hinge position: (-width/2, 0, 0) in door frame
    // Rotation is around Z-axis (vertical)
    //
    // When closed (opening_angle=0): leaf is in XZ plane at y=0
    // When open: leaf rotates around the hinge (left edge)

    const float w = door_width_.item<float>();
    const float h = door_height_.item<float>();
    const float lt = leaf_thickness_;

    // Hinge at left edge, floor level
    const torch::Tensor hinge_pos = torch::tensor({-w / 2.0f, 0.0f, 0.0f});

    // Leaf half-extents: (w/2, lt/2, h/2)
    const torch::Tensor b_leaf = torch::tensor({w / 2.0f, lt / 2.0f, h / 2.0f});

    // Leaf center in leaf-local frame (relative to hinge after rotation)
    // Center is at (w/2, 0, h/2) from hinge
    const torch::Tensor leaf_center_local = torch::tensor({w / 2.0f, 0.0f, h / 2.0f});

    // Transform points to leaf local frame:
    // 1. Translate so hinge is at origin
    const torch::Tensor p1 = points_door - hinge_pos;

    // 2. Rotate by -opening_angle around Z-axis
    const torch::Tensor neg_angle = opening_angle_;
    const torch::Tensor c = torch::cos(neg_angle);
    const torch::Tensor s = torch::sin(neg_angle);

    // Rotation around Z-axis (inverse):
    // [cos   sin  0]
    // [-sin  cos  0]
    // [0     0    1]
    const torch::Tensor p2_x = p1.index({torch::indexing::Slice(), 0}) * c
                              + p1.index({torch::indexing::Slice(), 1}) * s;
    const torch::Tensor p2_y = -p1.index({torch::indexing::Slice(), 0}) * s
                              + p1.index({torch::indexing::Slice(), 1}) * c;
    const torch::Tensor p2_z = p1.index({torch::indexing::Slice(), 2});

    // 3. Translate to leaf's local center
    const torch::Tensor p_local_x = p2_x - leaf_center_local[0];
    const torch::Tensor p_local_y = p2_y - leaf_center_local[1];
    const torch::Tensor p_local_z = p2_z - leaf_center_local[2];

    const torch::Tensor p_local = torch::stack({p_local_x, p_local_y, p_local_z}, /*dim=*/1);

    return sdBox(p_local, b_leaf);
}

torch::Tensor DoorModel::sdf(const torch::Tensor& points_robot) const
{
    // Transform points to door local frame
    const torch::Tensor points_door = transform_to_door_frame(points_robot);

    // Compute SDF for frame and leaf
    const torch::Tensor sdf_frame_vals = sdf_frame(points_door);
    const torch::Tensor sdf_leaf_vals = sdf_leaf(points_door);

    // Union: minimum distance to either component
    return torch::minimum(sdf_frame_vals, sdf_leaf_vals);
}


std::vector<float> DoorModel::get_door_parameters() const
{
    const auto pos = door_position_.accessor<float, 1>();
    const float theta = door_theta_.item<float>();
    const float width = door_width_.item<float>();
    const float height = door_height_.item<float>();
    const float angle = opening_angle_.item<float>();

    return {pos[0], pos[1], pos[2], theta, width, height, angle};
}

std::vector<float> DoorModel::get_door_pose() const
{
    const auto pos = door_position_.accessor<float, 1>();
    const float theta = door_theta_.item<float>();
    return {pos[0], pos[1], pos[2], theta};
}

std::vector<float> DoorModel::get_door_geometry() const
{
    return {door_width_.item<float>(), door_height_.item<float>()};
}

float DoorModel::get_opening_angle() const
{
    return opening_angle_.item<float>();
}

std::vector<torch::Tensor> DoorModel::parameters() const
{
    return {door_position_, door_theta_, door_width_, door_height_, opening_angle_};
}

std::vector<torch::Tensor> DoorModel::get_pose_parameters() const
{
    return {door_position_, door_theta_, opening_angle_};
}

std::vector<torch::Tensor> DoorModel::get_geometry_parameters() const
{
    return {door_width_, door_height_};
}

void DoorModel::freeze_geometry()
{
    if (door_width_.defined()) door_width_.set_requires_grad(false);
    if (door_height_.defined()) door_height_.set_requires_grad(false);
}

void DoorModel::unfreeze_geometry()
{
    if (door_width_.defined()) door_width_.set_requires_grad(true);
    if (door_height_.defined()) door_height_.set_requires_grad(true);
}

bool DoorModel::is_geometry_frozen() const
{
    return door_width_.defined() && !door_width_.requires_grad();
}

void DoorModel::set_door_position(float x, float y, float z)
{
    torch::NoGradGuard no_grad;
    door_position_[0] = x;
    door_position_[1] = y;
    door_position_[2] = z;
}

void DoorModel::set_theta(float theta)
{
    torch::NoGradGuard no_grad;
    door_theta_[0] = theta;
}

void DoorModel::set_pose(float x, float y, float z, float theta)
{
    torch::NoGradGuard no_grad;
    door_position_[0] = x;
    door_position_[1] = y;
    door_position_[2] = z;
    door_theta_[0] = theta;
}

void DoorModel::print_info() const
{
    const auto pose = get_door_pose();
    const auto geom = get_door_geometry();
    const float angle = get_opening_angle();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== DOOR MODEL ===\n";
    std::cout << "  Coordinate system: X=right, Y=forward, Z=up\n";
    std::cout << "  Position: (" << pose[0] << ", " << pose[1] << ", " << pose[2] << ") m\n";
    std::cout << "  Orientation: theta=" << pose[3] << " rad (" << (pose[3] * 180.0 / M_PI) << " deg)\n";
    std::cout << "  Dimensions: " << geom[0] << " x " << geom[1] << " m (W x H)\n";
    std::cout << "  Opening angle: " << angle << " rad (" << (angle * 180.0 / M_PI) << " deg)\n";
    std::cout << "  Frame: thickness=" << frame_thickness_ << "m, depth=" << frame_depth_ << "m\n";
    std::cout << "  Leaf: thickness=" << leaf_thickness_ << "m\n";
    std::cout << "  Geometry frozen: " << (is_geometry_frozen() ? "YES" : "NO") << "\n";
}