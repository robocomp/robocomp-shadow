//
// Manages uncertainty estimation and propagation for robot localization
//

#pragma once
#include <torch/torch.h>
#include <vector>
#include "common_types.h"
#include "room_model.h"

/**
 * @brief Manages uncertainty estimation and propagation for robot localization.
 *
 * This class combines two sources of uncertainty:
 * 1. Kinematic propagation: based on velocity commands and motion noise model
 * 2. Geometric measurements: computed via Laplace approximation over sensor data
 *
 * Uses information form fusion to combine both sources, providing validated
 * covariance matrices (no NaN/Inf) that describe the robot's pose uncertainty
 * in mapping mode (3 DOF: x,y,θ) or localized mode (5 DOF: includes room orientation).
 *
 * Key features:
 * - Optional odometry-based propagation
 * - Automatic covariance validation
 * - Fallback mechanisms for degenerate cases
 * - Support for state transitions (mapping/localized)
 */

class UncertaintyManager
{
    public:
        struct Result
        {
            torch::Tensor covariance;    // 3x3 or 5x5
            std::vector<float> std_devs; // Standard deviations
            bool is_valid = true;        // False if NaN/Inf detected

            // Diagnostics
            bool used_propagation = false;
            bool used_fusion = false;
            std::string error_message;
        };

        UncertaintyManager() = default;

        // Configuration
        void set_motion_noise(float translation_per_meter, float rotation_per_radian);
        void enable_motion_propagation(bool enable) { use_propagation_ = enable; }

        // State management
        void reset(); // Clear history (e.g., when switching mapping/localized states)
        bool has_history() const { return has_history_; }
        void set_previous_cov(const torch::Tensor& cov);
        void set_previous_pose(const std::vector<float>& pose);
        torch::Tensor propagate_with_velocity( const VelocityCommand& cmd, float dt,
                                               const torch::Tensor& prev_cov,
                                               bool is_localized) const;

        torch::Tensor propagate_with_delta_pose( const Eigen::Vector3f &delta_pose,
                                                 const torch::Tensor& prev_cov,
                                                 bool is_localized) const;

        torch::Tensor get_previous_cov() const ;
         // Fuse propagated and measurement covariances using information form
        torch::Tensor fuse_covariances(const torch::Tensor& propagated,const torch::Tensor& measurement ) const;

    private:
        // ===== STATE =====
        std::vector<float> previous_pose_;  // [x, y, theta]
        torch::Tensor previous_cov_;         // Last valid covariance
        bool has_history_ = false;

        // ===== CONFIGURATION =====
        bool use_propagation_ = false;
        float noise_trans_ = 0.02f;  // 2cm stddev per meter
        float noise_rot_ = 0.1f;     // 0.1 rad stddev per radian
        const float reg_factor_ = 1e-6f;  // Regularization for numerical stability

        // ===== INTERNAL METHODS =====

        /**
         * Compute measurement covariance via Laplace approximation
         */
        torch::Tensor compute_measurement_covariance(
            const torch::Tensor& points,
            RoomModel& room,
            float huber_delta,
            bool is_localized
        ) const;

        /**
         * Validate covariance matrix (check for NaN, Inf, wrong dimensions)
         */
        bool is_valid_covariance(const torch::Tensor& cov, int expected_dim) const;

        /**
         * Create fallback covariance when computation fails
         */
        torch::Tensor create_fallback_covariance(int dim) const;

        /**
         * Compute motion metrics between poses
         */
        void compute_motion_metrics(
            const std::vector<float>& current_pose,
            float& translation,
            float& rotation
        ) const;
};