/*
 * Elastic Band Local Path Planner with Obstacle Avoidance — Implementation
 */

#include "elastic_band.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
ElasticBand::ElasticBand(const Params &params) : params_(params) {}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::initialize(const Eigen::Vector2f &start, const Eigen::Vector2f &target)
{
    bubbles_.clear();
    current_target_ = target;

    float dist = (target - start).norm();
    if (dist < 1.f)
    {
        initialized_ = false;
        return;
    }

    // Build a sequence of waypoints that avoids the robot body.
    // Robot body occupies approx [-semi_width, semi_width] x [-semi_length, semi_length]
    // around the origin (start). We add a margin so bubbles stay clear.
    float margin = params_.robot_semi_width + params_.safety_margin * 0.5f;
    float body_front = params_.robot_semi_length + margin;
    float body_side  = params_.robot_semi_width + margin;

    std::vector<Eigen::Vector2f> waypoints;
    waypoints.push_back(start);

    // Check if the target is inside or behind the robot body zone
    bool target_behind = target.y() < body_front;
    bool target_inside_lateral = std::abs(target.x()) < body_side;

    if (target_behind && target_inside_lateral && dist > params_.robot_semi_length)
    {
        // Target is behind or inside the robot — route around the side
        // Pick the side closest to the target (or left if centered)
        float exit_x = (target.x() >= 0.f) ? body_side : -body_side;

        // Waypoint 1: go to the side at the front of the robot
        waypoints.emplace_back(exit_x, body_front);

        // If target is actually behind (negative Y), add a waypoint at the rear side
        if (target.y() < -params_.robot_semi_length * 0.5f)
        {
            waypoints.emplace_back(exit_x, -body_front);
        }

        // Waypoint: approach target laterally from the side
        waypoints.emplace_back(exit_x, target.y());
    }
    else if (target.y() < 0.f && dist > params_.robot_semi_length)
    {
        // Target is behind but outside the lateral body zone — still route forward first
        // Go forward a bit, then curve toward target
        float exit_x = (target.x() >= 0.f) ? body_side : -body_side;
        waypoints.emplace_back(exit_x, body_front);
        waypoints.emplace_back(exit_x, target.y());
    }

    waypoints.push_back(target);

    // Now create evenly-spaced bubbles along the waypoint polyline
    // First compute total polyline length
    float total_length = 0.f;
    for (size_t i = 1; i < waypoints.size(); ++i)
        total_length += (waypoints[i] - waypoints[i - 1]).norm();

    if (total_length < 1.f)
    {
        initialized_ = false;
        return;
    }

    int n_bubbles = std::max(3, static_cast<int>(std::ceil(total_length / params_.bubble_spacing)) + 1);
    float step_dist = total_length / static_cast<float>(n_bubbles - 1);

    bubbles_.resize(n_bubbles);
    bubbles_[0].pos = waypoints.front();
    bubbles_[0].clearance = std::numeric_limits<float>::max();

    size_t wp_idx = 1;
    float seg_consumed = 0.f;
    Eigen::Vector2f seg_start = waypoints[0];
    Eigen::Vector2f seg_dir = (waypoints[1] - waypoints[0]);
    float seg_len = seg_dir.norm();
    if (seg_len > 1e-6f) seg_dir /= seg_len;

    for (int i = 1; i < n_bubbles; ++i)
    {
        float target_dist = step_dist * static_cast<float>(i);
        // Walk along polyline segments
        while (wp_idx < waypoints.size() && seg_consumed + seg_len < target_dist)
        {
            seg_consumed += seg_len;
            seg_start = waypoints[wp_idx];
            wp_idx++;
            if (wp_idx < waypoints.size())
            {
                seg_dir = waypoints[wp_idx] - seg_start;
                seg_len = seg_dir.norm();
                if (seg_len > 1e-6f) seg_dir /= seg_len;
            }
            else
            {
                seg_len = 0.f;
            }
        }
        float along = target_dist - seg_consumed;
        bubbles_[i].pos = seg_start + seg_dir * along;
        bubbles_[i].clearance = std::numeric_limits<float>::max();
    }
    // Ensure last bubble is exactly at target
    bubbles_.back().pos = target;

    initialized_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::optimize(const std::vector<Eigen::Vector3f> &lidar_points)
{
    if (!initialized_ || bubbles_.size() < 3) return;

    // Pin start bubble to robot origin (robot frame, always 0,0)
    bubbles_.front().pos = Eigen::Vector2f(0.f, 0.f);

    // Update clearances first
    update_clearances(lidar_points);

    for (int iter = 0; iter < params_.iterations; ++iter)
    {
        // Apply forces to internal bubbles (not first and last)
        std::vector<Eigen::Vector2f> displacements(bubbles_.size(), Eigen::Vector2f::Zero());

        for (size_t i = 1; i < bubbles_.size() - 1; ++i)
        {
            // ---- Internal contraction force (spring toward midpoint of neighbors) ----
            Eigen::Vector2f midpoint = (bubbles_[i - 1].pos + bubbles_[i + 1].pos) * 0.5f;
            Eigen::Vector2f f_internal = params_.k_internal * (midpoint - bubbles_[i].pos);

            // Suppress internal force when bubble is near an obstacle.
            // This prevents the spring from pulling the band back into walls in corridors.
            float clearance_ratio = std::clamp(
                bubbles_[i].clearance / (params_.robot_semi_width + params_.safety_margin),
                0.f, 1.f);
            f_internal *= clearance_ratio;

            // ---- External repulsion force (away from obstacles) ----
            Eigen::Vector2f f_external = Eigen::Vector2f::Zero();
            for (const auto &lp : lidar_points)
            {
                // Skip zero-valued points
                if (lp.x() == 0.f && lp.y() == 0.f) continue;

                Eigen::Vector2f obstacle(lp.x(), lp.y());
                Eigen::Vector2f diff = bubbles_[i].pos - obstacle;
                float dist = diff.norm();

                if (dist < 1.f) dist = 1.f; // Avoid division by zero

                if (dist < params_.obstacle_influence)
                {
                    // Quadratic potential field repulsion
                    float ratio = (params_.obstacle_influence - dist) / params_.obstacle_influence;
                    float magnitude = params_.k_external * ratio * ratio * (params_.obstacle_influence / dist);
                    f_external += magnitude * diff.normalized();
                }
            }

            displacements[i] = f_internal + f_external;
        }

        // Apply displacements
        for (size_t i = 1; i < bubbles_.size() - 1; ++i)
        {
            // Clamp displacement to avoid instability
            float max_disp = params_.bubble_spacing;
            if (displacements[i].norm() > max_disp)
                displacements[i] = displacements[i].normalized() * max_disp;

            bubbles_[i].pos += displacements[i];

            // Push bubble out if it ended up inside the robot body
            push_out_of_robot_body(bubbles_[i].pos);
        }

        // Redistribute bubbles every few iterations to maintain spacing
        if (iter % 3 == 0)
            redistribute_bubbles();

        // Update clearances after each iteration for the clearance-based suppression
        update_clearances(lidar_points);
    }

    // Pin start again after redistribution
    bubbles_.front().pos = Eigen::Vector2f(0.f, 0.f);

    // Remove any bubbles (except first and last) that are inside the robot body
    prune_bubbles_inside_robot();

    // Final clearance update
    update_clearances(lidar_points);
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::redistribute_bubbles()
{
    if (bubbles_.size() < 2) return;

    std::vector<Bubble> new_bubbles;
    new_bubbles.push_back(bubbles_.front());

    for (size_t i = 1; i < bubbles_.size(); ++i)
    {
        float dist = (bubbles_[i].pos - new_bubbles.back().pos).norm();

        // If too far apart, insert intermediate bubbles
        if (dist > params_.bubble_spacing * 2.0f)
        {
            int n_inserts = static_cast<int>(std::floor(dist / params_.bubble_spacing));
            Eigen::Vector2f step = (bubbles_[i].pos - new_bubbles.back().pos) / static_cast<float>(n_inserts + 1);
            Eigen::Vector2f base = new_bubbles.back().pos;
            for (int j = 1; j <= n_inserts; ++j)
            {
                Bubble b;
                b.pos = base + step * static_cast<float>(j);
                new_bubbles.push_back(b);
            }
        }
        // If too close and not the last bubble, skip
        else if (dist < params_.bubble_spacing * 0.4f && i < bubbles_.size() - 1)
        {
            continue;
        }

        new_bubbles.push_back(bubbles_[i]);
    }

    // Ensure last bubble is preserved
    if ((new_bubbles.back().pos - bubbles_.back().pos).norm() > 1.f)
        new_bubbles.push_back(bubbles_.back());

    bubbles_ = std::move(new_bubbles);

    // Limit path length
    if (bubbles_.size() > 200)
        bubbles_.resize(200);
}

// ─────────────────────────────────────────────────────────────────────────────
float ElasticBand::compute_clearance(const Eigen::Vector2f &pos,
                                      const std::vector<Eigen::Vector3f> &lidar_points) const
{
    float min_dist = std::numeric_limits<float>::max();
    for (const auto &lp : lidar_points)
    {
        if (lp.x() == 0.f && lp.y() == 0.f) continue;
        float d = (pos - Eigen::Vector2f(lp.x(), lp.y())).norm();
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::update_clearances(const std::vector<Eigen::Vector3f> &lidar_points)
{
    for (auto &b : bubbles_)
        b.clearance = compute_clearance(b.pos, lidar_points);
}

// ─────────────────────────────────────────────────────────────────────────────
bool ElasticBand::is_inside_robot_body(const Eigen::Vector2f &pos) const
{
    // Robot body exclusion zone: a rectangle centered at origin with a margin
    float margin = params_.safety_margin * 0.3f;
    float half_w = params_.robot_semi_width + margin;
    float half_l = params_.robot_semi_length + margin;
    return std::abs(pos.x()) < half_w && std::abs(pos.y()) < half_l;
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::push_out_of_robot_body(Eigen::Vector2f &pos) const
{
    float margin = params_.safety_margin * 0.3f;
    float half_w = params_.robot_semi_width + margin;
    float half_l = params_.robot_semi_length + margin;

    if (std::abs(pos.x()) >= half_w || std::abs(pos.y()) >= half_l)
        return;  // Not inside the robot body

    // Find the closest edge to push the point to
    float dx_right = half_w - pos.x();
    float dx_left  = half_w + pos.x();
    float dy_front = half_l - pos.y();
    float dy_back  = half_l + pos.y();

    float min_dist = std::min({dx_right, dx_left, dy_front, dy_back});

    // Push to the closest edge, preferring forward and lateral directions
    if (min_dist == dy_front)
        pos.y() = half_l;       // Push forward (preferred for navigation)
    else if (min_dist == dx_right)
        pos.x() = half_w;       // Push right
    else if (min_dist == dx_left)
        pos.x() = -half_w;      // Push left
    else
        pos.y() = -half_l;      // Push backward (least preferred)
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::prune_bubbles_inside_robot()
{
    if (bubbles_.size() < 3) return;

    std::vector<Bubble> pruned;
    pruned.push_back(bubbles_.front());  // Always keep first (robot origin)

    for (size_t i = 1; i < bubbles_.size() - 1; ++i)
    {
        if (!is_inside_robot_body(bubbles_[i].pos))
            pruned.push_back(bubbles_[i]);
    }

    pruned.push_back(bubbles_.back());  // Always keep last (target)
    bubbles_ = std::move(pruned);
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<float, float, float> ElasticBand::compute_velocities() const
{
    if (!initialized_ || bubbles_.size() < 2)
        return {0.f, 0.f, 0.f};

    // Pure-pursuit: find a carrot point at lookahead distance along the path
    Eigen::Vector2f carrot = bubbles_.back().pos; // Default: last point
    float accum = 0.f;

    for (size_t i = 1; i < bubbles_.size(); ++i)
    {
        float seg_len = (bubbles_[i].pos - bubbles_[i - 1].pos).norm();
        if (accum + seg_len >= params_.lookahead)
        {
            float alpha = (params_.lookahead - accum) / seg_len;
            carrot = bubbles_[i - 1].pos + alpha * (bubbles_[i].pos - bubbles_[i - 1].pos);
            break;
        }
        accum += seg_len;
    }

    // Compute velocities toward carrot point
    // In robot frame: x = lateral (side), y = forward (advance)
    float adv = std::clamp(carrot.y(), 0.f, params_.max_adv);
    float side = std::clamp(carrot.x(), -params_.max_side, params_.max_side);
    float rot = std::clamp(std::atan2(carrot.x(), carrot.y()), -params_.max_rot, params_.max_rot);

    // Scale advance by rotation (slow down when turning) — gentler scaling
    float rot_factor = 1.f - 0.5f * std::abs(rot) / params_.max_rot;
    adv *= rot_factor;

    // Scale by minimum clearance along the path (skip first 2 bubbles near robot origin)
    // Only slow down when obstacles are really close to the path
    float min_cl = path_min_clearance(2);  // skip first 2 bubbles
    float danger_zone = params_.robot_semi_width + params_.safety_margin;
    if (min_cl < danger_zone)
    {
        float clearance_factor = std::clamp(min_cl / danger_zone, 0.15f, 1.f);
        adv *= clearance_factor;
        side *= clearance_factor;
    }

    return {adv, side, rot};
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<Eigen::Vector2f> ElasticBand::get_path() const
{
    std::vector<Eigen::Vector2f> path;
    path.reserve(bubbles_.size());
    for (const auto &b : bubbles_)
        path.push_back(b.pos);
    return path;
}

// ─────────────────────────────────────────────────────────────────────────────
bool ElasticBand::is_valid() const
{
    if (!initialized_ || bubbles_.size() < 2) return false;
    float threshold = params_.robot_semi_width;
    for (const auto &b : bubbles_)
        if (b.clearance < threshold) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
float ElasticBand::min_clearance() const
{
    float mc = std::numeric_limits<float>::max();
    for (const auto &b : bubbles_)
        if (b.clearance < mc) mc = b.clearance;
    return mc;
}

// ─────────────────────────────────────────────────────────────────────────────
float ElasticBand::path_min_clearance(size_t skip_n) const
{
    float mc = std::numeric_limits<float>::max();
    for (size_t i = skip_n; i < bubbles_.size(); ++i)
        if (bubbles_[i].clearance < mc) mc = bubbles_[i].clearance;
    return mc;
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::clear()
{
    bubbles_.clear();
    initialized_ = false;
}

// ─────────────────────────────────────────────────────────────────────────────
bool ElasticBand::update_target(const Eigen::Vector2f &new_target, float threshold)
{
    if (!initialized_)
    {
        current_target_ = new_target;
        return true;
    }
    if ((new_target - current_target_).norm() > threshold)
    {
        current_target_ = new_target;
        return true;
    }
    // Update the last bubble position to track target movement
    if (!bubbles_.empty())
        bubbles_.back().pos = new_target;

    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
void ElasticBand::draw(QGraphicsScene *scene, QColor path_color, QColor bubble_color)
{
    // Clean previous items
    for (auto *item : draw_items_)
    {
        scene->removeItem(item);
        delete item;
    }
    draw_items_.clear();

    if (!initialized_ || bubbles_.size() < 2) return;

    // Draw path polyline segments
    for (size_t i = 0; i + 1 < bubbles_.size(); ++i)
    {
        auto *line = scene->addLine(
            bubbles_[i].pos.x(), bubbles_[i].pos.y(),
            bubbles_[i + 1].pos.x(), bubbles_[i + 1].pos.y(),
            QPen(path_color, 8));
        draw_items_.push_back(line);
    }

    // Draw bubbles as circles with radius proportional to clearance (capped)
    for (const auto &b : bubbles_)
    {
        float r = std::min(b.clearance, params_.obstacle_influence) * 0.15f; // Scale for visibility
        r = std::clamp(r, 15.f, 80.f);

        auto *ellipse = scene->addEllipse(
            -r, -r, 2 * r, 2 * r,
            QPen(path_color, 3),
            QBrush(bubble_color));
        ellipse->setPos(b.pos.x(), b.pos.y());
        draw_items_.push_back(ellipse);
    }

    // Draw start point (green) and target point (red)
    if (!bubbles_.empty())
    {
        float r = 30.f;
        auto *start = scene->addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor("green"), 5), QBrush(QColor(0, 255, 0, 100)));
        start->setPos(bubbles_.front().pos.x(), bubbles_.front().pos.y());
        draw_items_.push_back(start);

        auto *end = scene->addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor("red"), 5), QBrush(QColor(255, 0, 0, 100)));
        end->setPos(bubbles_.back().pos.x(), bubbles_.back().pos.y());
        draw_items_.push_back(end);
    }
}

