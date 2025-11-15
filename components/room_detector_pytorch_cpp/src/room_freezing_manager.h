/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp
 */

#ifndef ROOM_FREEZING_MANAGER_H
#define ROOM_FREEZING_MANAGER_H

#include <torch/torch.h>
#include <vector>
#include <deque>
#include <iostream>
#include <chrono>

/**
 * @brief Manages adaptive freezing of room parameters with hysteresis
 * 
 * State machine:
 * MAPPING: Optimize both room shape and robot pose (initial exploration)
 * LOCALIZED: Optimize only robot pose (confident in room shape)
 * 
 * Transitions:
 * MAPPING → LOCALIZED: When room estimate is confident and stable
 * LOCALIZED → MAPPING: When structural changes detected (anomaly)
 * 
 * Uses hysteresis to prevent oscillation between states.
 */
class RoomFreezingManager
{
public:
    enum class State
    {
        MAPPING,      // Optimize all parameters (room + robot)
        LOCALIZED,    // Optimize only robot pose (room frozen)
        TRANSITIONING // Optional intermediate state
    };

    struct Params
    {
        // Thresholds for MAPPING → LOCALIZED (freezing)
        float uncertainty_freeze_threshold = 0.05f;    // Max std dev for room params (meters)
        float stability_freeze_threshold = 0.02f;      // Max change in room params between updates
        int min_observations_to_freeze = 100;           // Min number of optimization iterations
        float confidence_freeze_threshold = 0.95f;     // Confidence level (0-1)
        
        // Thresholds for LOCALIZED → MAPPING (unfreezing)
        float residual_unfreeze_threshold = 0.15f;     // Max acceptable mean residual
        float uncertainty_unfreeze_threshold = 0.15f;   // If robot pose uncertainty gets too high
        float structural_change_threshold = 0.20f;     // Significant change in room shape detected
        
        // Hysteresis factors (prevent oscillation)
        float freeze_hysteresis = 1.2f;                // Unfreeze threshold = freeze threshold * hysteresis
        
        // History tracking
        int history_size = 20;                         // Number of past estimates to track
        
        // Time-based constraints
        float min_time_in_state = 5.0f;                // Minimum seconds before state change
    };
    RoomFreezingManager() = default;


    /**
     * @brief Update state based on current room estimate and optimization results
     * 
     * @param room_params Current room parameters [half_width, half_height]
     * @param room_std_devs Uncertainty in room parameters
     * @param robot_std_devs Uncertainty in robot pose
     * @param mean_residual Mean SDF residual (loss value)
     * @param iteration Current optimization iteration
     * @return true if state changed, false otherwise
     */
    bool update(const std::vector<float>& room_params,
                const std::vector<float>& room_std_devs,
                const std::vector<float>& robot_std_devs,
                float mean_residual,
                int iteration);

    /**
     * @brief Check if room parameters should be frozen (not optimized)
     */
    bool should_freeze_room() const { return state_ == State::LOCALIZED; }

    /**
     * @brief Get current state
     */
    State get_state() const { return state_; }

    /**
     * @brief Get state as string for logging
     */
    std::string get_state_string() const
    {
        switch(state_)
        {
            case State::MAPPING: return "MAPPING";
            case State::LOCALIZED: return "LOCALIZED";
            case State::TRANSITIONING: return "TRANSITIONING";
            default: return "UNKNOWN";
        }
    }

    /**
     * @brief Print current state and statistics
     */
    void print_status() const;

    /**
     * @brief Force state change (for testing or external triggers)
     */
    void force_state(State new_state)
    {
        if (state_ != new_state)
        {
            std::cout << "🔧 Forcing state change: " << get_state_string() 
                      << " → " << state_to_string(new_state) << "\n";
            state_ = new_state;
            state_entry_time_ = std::chrono::steady_clock::now();
        }
    }

    /**
     * @brief Reset to initial MAPPING state
     */
    void reset()
    {
        state_ = State::MAPPING;
        observation_count_ = 0;
        room_history_.clear();
        residual_history_.clear();
        state_entry_time_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief Get statistics for visualization/debugging
     */
    struct Statistics
    {
        int observation_count;
        float time_in_current_state;
        float mean_room_uncertainty;
        float mean_robot_uncertainty;
        float mean_residual;
        float room_stability;
        bool can_freeze;
        bool should_unfreeze;
    };
    
    Statistics get_statistics() const;

private:
    Params params_;
    State state_ = State::MAPPING;
    int observation_count_;
    
    // History tracking
    std::deque<std::vector<float>> room_history_;     // Past room parameters
    std::deque<float> residual_history_;              // Past residuals
    
    // Timing
    std::chrono::steady_clock::time_point state_entry_time_;

    // Helper methods
    bool check_freeze_conditions(const std::vector<float>& room_params,
                                  const std::vector<float>& room_std_devs,
                                  float mean_residual);
    
    bool check_unfreeze_conditions(const std::vector<float>& room_params,
                                    const std::vector<float>& robot_std_devs,
                                    float mean_residual);
    
    float compute_room_stability() const;
    float compute_mean_uncertainty(const std::vector<float>& std_devs) const;
    bool check_structural_change(const std::vector<float>& room_params) const;
    float get_time_in_current_state() const;
    
    static std::string state_to_string(State s)
    {
        switch(s)
        {
            case State::MAPPING: return "MAPPING";
            case State::LOCALIZED: return "LOCALIZED";
            case State::TRANSITIONING: return "TRANSITIONING";
            default: return "UNKNOWN";
        }
    }
};

#endif // ROOM_FREEZING_MANAGER_H
