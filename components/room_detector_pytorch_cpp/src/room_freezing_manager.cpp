/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 */

#include "room_freezing_manager.h"
#include <numeric>
#include <algorithm>
#include <cmath>

bool RoomFreezingManager::update(const std::vector<float>& room_params,
                                  const std::vector<float>& room_std_devs,
                                  const std::vector<float>& robot_std_devs,
                                  const std::vector<float>& robot_pose,
                                  float mean_residual,
                                  int iteration)
{
    observation_count_ = iteration;

    //std::cout << "Frame " << observation_count_ << ", dist=" << cumulative_distance_
    //          << "m, can_freeze=" << check_freeze_conditions(room_params, room_std_devs, mean_residual) << "\n";

    // Update movement tracking
    update_movement_tracking(robot_pose);

    // Update histories
    room_history_.push_back(room_params);
    if (room_history_.size() > params_.history_size)
        room_history_.pop_front();

    residual_history_.push_back(mean_residual);
    if (residual_history_.size() > params_.history_size)
        residual_history_.pop_front();

    // Check minimum time in current state (prevent rapid switching)
    float time_in_state = get_time_in_current_state();
    if (time_in_state < params_.min_time_in_state)
        return false;  // Not enough time elapsed

    RoomState previous_state = state_;

    // State machine logic
    switch(state_)
    {
        case RoomState::MAPPING:
        {
            // Check if we should transition to LOCALIZED (freeze room)
            if (check_freeze_conditions(room_params, room_std_devs, mean_residual))
            {
                std::cout << "\n🔒 FREEZING ROOM: Confident estimate achieved\n";
                std::cout << "   Observations: " << observation_count_ << "\n";
                std::cout << "   Mean uncertainty: " << compute_mean_uncertainty(room_std_devs) << " m\n";
                std::cout << "   Stability: " << compute_room_stability() << " m\n";
                std::cout << "   Residual: " << mean_residual << "\n";
                std::cout << "   Distance traveled: " << cumulative_distance_ << " m\n";
                std::cout << "   Rotation traveled: " << cumulative_rotation_ << " rad\n";
                state_ = RoomState::LOCALIZED;
                state_entry_time_ = std::chrono::steady_clock::now();
                reset_movement_tracking(robot_pose);
            }
            break;
        }

        case RoomState::LOCALIZED:
        {
            // Check if we should transition back to MAPPING (unfreeze due to anomaly)
            if (check_unfreeze_conditions(room_params, robot_std_devs, mean_residual))
            {
                std::cout << "\n🔓 UNFREEZING ROOM: Structural change detected!\n";
                std::cout << "   Residual: " << mean_residual << " (threshold: "
                          << params_.residual_unfreeze_threshold << ")\n";
                std::cout << "   Robot uncertainty: " << compute_mean_uncertainty(robot_std_devs)
                          << " m\n";
                if (check_structural_change(room_params))
                    std::cout << "   Significant room shape change detected!\n";
                state_ = RoomState::MAPPING;
                state_entry_time_ = std::chrono::steady_clock::now();
                reset_movement_tracking(robot_pose);
            }
            break;
        }

        case RoomState::TRANSITIONING:
            // Could add a transition state if needed for smoothing
            break;
    }

    return state_ != previous_state;
}

bool RoomFreezingManager::check_freeze_conditions(const std::vector<float>& room_params,
                                                   const std::vector<float>& room_std_devs,
                                                   float mean_residual)
{
    // Condition 1: Enough observations
    if (observation_count_ < params_.min_observations_to_freeze)
        return false;

    // Condition 2: Low uncertainty in room parameters
    float mean_uncertainty = compute_mean_uncertainty(room_std_devs);
    if (mean_uncertainty > params_.uncertainty_freeze_threshold)
        return false;

    // Condition 3: Stable estimates (not changing much)
    float stability = compute_room_stability();
    if (stability > params_.stability_freeze_threshold)
        return false;

    // Condition 4: Good fit (low residual)
    if (mean_residual > params_.residual_unfreeze_threshold)
        return false;

    // Condition 5 (NEW): Robot has moved enough to explore the room
    if (cumulative_distance_ < params_.min_distance_traveled)
    {
        std::cout << "   ⏳ Waiting for more exploration: " << cumulative_distance_
                  << " / " << params_.min_distance_traveled << " m\n";
        return false;
    }

    if (cumulative_rotation_ < params_.min_rotation_traveled)
    {
        std::cout << "   ⏳ Waiting for more rotation: " << cumulative_rotation_
                  << " / " << params_.min_rotation_traveled << " rad\n";
        return false;
    }

    // All conditions met!
    return true;
}

bool RoomFreezingManager::check_unfreeze_conditions(const std::vector<float>& room_params,
                                                     const std::vector<float>& robot_std_devs,
                                                     float mean_residual)
{
    // Grace period: Don't unfreeze immediately after freezing
    // Give the system a few frames to settle in LOCALIZED mode
    float time_in_localized = get_time_in_current_state();
    if (time_in_localized < 2.0f) {  // 2 second grace period
        return false;
    }

    // Compute mean of recent residuals
    float recent_mean_residual = 0.0f;
    if (not residual_history_.empty())
        recent_mean_residual = std::accumulate(residual_history_.begin(),
                                               residual_history_.end(), 0.0f)
                              / residual_history_.size();


    // Use RELATIVE threshold: unfreeze if current residual is much worse than recent average
    const float relative_threshold = std::max(
        0.25f,                               // Absolute minimum (increased from 0.15)
        recent_mean_residual * 4.0f          // Or 4x recent average (increased from 3x)
    );

    // Condition 1: High residual (poor fit with current room model)
    if (mean_residual > relative_threshold)
    {
        std::cout << "   ⚠️  High residual: " << mean_residual
                  << " (threshold: " << relative_threshold
                  << ", recent avg: " << recent_mean_residual << ")\n";
        return true;
    }

    // Condition 2: Robot pose uncertainty is growing (lost track)
    float mean_robot_uncertainty = compute_mean_uncertainty(robot_std_devs);
    if (mean_robot_uncertainty > params_.uncertainty_unfreeze_threshold)
    {
        std::cout << "   ⚠️  High robot uncertainty: " << mean_robot_uncertainty << "\n";
        return true;
    }

    // Condition 3: Structural change detected (e.g., L-shaped room revealed)
    if (check_structural_change(room_params))
    {
        std::cout << "   ⚠️  Structural change detected\n";
        return true;
    }
    return false;
}


float RoomFreezingManager::compute_room_stability() const
{
    if (room_history_.size() < 2)
        return 0.0f;

    // Compute maximum change in any room parameter over recent history
    float max_change = 0.0f;

    const auto& latest = room_history_.back();
    for (size_t i = 0; i < room_history_.size() - 1; ++i)
    {
        const auto& prev = room_history_[i];
        for (size_t j = 0; j < latest.size() && j < prev.size(); ++j)
        {
            float change = std::abs(latest[j] - prev[j]);
            max_change = std::max(max_change, change);
        }
    }

    return max_change;
}

float RoomFreezingManager::compute_mean_uncertainty(const std::vector<float>& std_devs) const
{
    if (std_devs.empty())
        return 0.0f;

    float sum = std::accumulate(std_devs.begin(), std_devs.end(), 0.0f);
    return sum / static_cast<float>(std_devs.size());
}

bool RoomFreezingManager::check_structural_change(const std::vector<float>& room_params) const
{
    if (room_history_.size() < params_.history_size / 2)
        return false;  // Not enough history

    // Compute median of historical room parameters
    std::vector<float> median_params(room_params.size(), 0.0f);
    for (size_t param_idx = 0; param_idx < room_params.size(); ++param_idx)
    {
        std::vector<float> values;
        for (const auto& hist : room_history_)
        {
            if (param_idx < hist.size())
                values.push_back(hist[param_idx]);
        }

        if (!values.empty())
        {
            std::sort(values.begin(), values.end());
            median_params[param_idx] = values[values.size() / 2];
        }
    }

    // Check if current params deviate significantly from median
    for (size_t i = 0; i < room_params.size() && i < median_params.size(); ++i)
    {
        float relative_change = std::abs(room_params[i] - median_params[i]) /
                               (median_params[i] + 1e-6f);

        if (relative_change > params_.structural_change_threshold)
        {
            std::cout << "   Room param[" << i << "]: " << room_params[i]
                      << " vs median: " << median_params[i]
                      << " (change: " << relative_change * 100 << "%)\n";
            return true;
        }
    }

    return false;
}

float RoomFreezingManager::get_time_in_current_state() const
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - state_entry_time_);
    return duration.count() / 1000.0f;  // Convert to seconds
}

void RoomFreezingManager::print_status() const
{
    std::cout << "\n=== ROOM FREEZING STATUS ===\n";
    std::cout << "State: " << get_state_string() << "\n";
    std::cout << "Time in state: " << get_time_in_current_state() << " s\n";
    std::cout << "Observations: " << observation_count_ << "\n";
    std::cout << "Room history size: " << room_history_.size() << "\n";
    std::cout << "Stability: " << compute_room_stability() << " m\n";
    std::cout << "Distance traveled: " << cumulative_distance_ << " m\n";
    std::cout << "Rotation traveled: " << cumulative_rotation_ << " rad\n";

    if (!residual_history_.empty())
    {
        float mean_residual = std::accumulate(residual_history_.begin(),
                                              residual_history_.end(), 0.0f) /
                             residual_history_.size();
        std::cout << "Mean residual: " << mean_residual << "\n";
    }

    std::cout << "============================\n";
}

RoomFreezingManager::Statistics RoomFreezingManager::get_statistics() const
{
    Statistics stats;
    stats.observation_count = observation_count_;
    stats.time_in_current_state = get_time_in_current_state();
    stats.room_stability = compute_room_stability();
    stats.distance_traveled = cumulative_distance_;
    stats.rotation_traveled = cumulative_rotation_;

    // Compute mean residual
    if (!residual_history_.empty())
    {
        stats.mean_residual = std::accumulate(residual_history_.begin(),
                                              residual_history_.end(), 0.0f) /
                             residual_history_.size();
    }
    else
    {
        stats.mean_residual = 0.0f;
    }

    // These would need to be passed in or cached
    stats.mean_room_uncertainty = 0.0f;
    stats.mean_robot_uncertainty = 0.0f;
    stats.can_freeze = false;
    stats.should_unfreeze = false;

    return stats;
}

void RoomFreezingManager::update_movement_tracking(const std::vector<float>& robot_pose)
{
    // robot_pose = [x, y, theta]
    if (robot_pose.size() < 3)
        return;

    // Store in history
    robot_pose_history_.push_back(robot_pose);
    if (robot_pose_history_.size() > params_.history_size)
        robot_pose_history_.pop_front();

    // Initialize pose at state change if needed
    if (pose_at_state_change_.empty())
    {
        pose_at_state_change_ = robot_pose;
        cumulative_distance_ = 0.0f;
        cumulative_rotation_ = 0.0f;
        return;
    }

    // Compute incremental movement since last update
    if (robot_pose_history_.size() >= 2)
    {
        const auto& prev_pose = robot_pose_history_[robot_pose_history_.size() - 2];

        // Compute translation
        float dx = robot_pose[0] - prev_pose[0];
        float dy = robot_pose[1] - prev_pose[1];
        float distance = std::sqrt(dx * dx + dy * dy);
        cumulative_distance_ += distance;

        // Compute rotation (handle wrap-around)
        float dtheta = robot_pose[2] - prev_pose[2];
        // Normalize to [-pi, pi]
        while (dtheta > M_PI) dtheta -= 2 * M_PI;
        while (dtheta < -M_PI) dtheta += 2 * M_PI;
        cumulative_rotation_ += std::abs(dtheta);
    }
}

void RoomFreezingManager::reset_movement_tracking(const std::vector<float>& robot_pose)
{
    pose_at_state_change_ = robot_pose;
    cumulative_distance_ = 0.0f;
    cumulative_rotation_ = 0.0f;
    robot_pose_history_.clear();
    robot_pose_history_.push_back(robot_pose);
}