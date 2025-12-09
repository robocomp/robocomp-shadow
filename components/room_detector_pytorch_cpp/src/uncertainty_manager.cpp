//
// Implementation of UncertaintyManager
//

#include "uncertainty_manager.h"
#include "uncertainty_estimator.h"
#include <QDebug>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

torch::Tensor UncertaintyManager::compute_measurement_covariance(
    const torch::Tensor& points,
    std::shared_ptr<RoomModel>& room,
    float huber_delta,
    bool is_localized) const
{
    try {
        return UncertaintyEstimator::compute_covariance(points, room, huber_delta);
    } catch (const std::exception& e) {
        qWarning() << "Measurement covariance computation failed:" << e.what();
        return torch::full({is_localized ? 3 : 5, is_localized ? 3 : 5},
                          std::numeric_limits<float>::quiet_NaN());
    }
}

torch::Tensor UncertaintyManager::propagate_with_velocity(
    const VelocityCommand& cmd,
    float dt,
    const torch::Tensor& prev_cov,
    bool is_localized) const
{
    int dim = is_localized ? 3 : 5;

    // Get current pose for Jacobian computation
    if (previous_pose_.empty()) {
        // Fallback: simple additive noise
        return prev_cov + noise_trans_ * noise_trans_ * torch::eye(dim, torch::kFloat32);
    }

    float theta = previous_pose_[2];  // Current heading

    // Compute motion in robot frame
    float dx_local = (cmd.adv_x * dt);
    float dy_local = (cmd.adv_z * dt);
    float dtheta = -cmd.rot * dt;

    // ===== MOTION MODEL JACOBIAN =====
    // State: [x, y, theta] (for localized) or [w, h, x, y, theta] (for mapping)
    // Motion: x' = x + dx*cos(θ) - dy*sin(θ)
    //         y' = y + dx*sin(θ) + dy*cos(θ)
    //         θ' = θ + dθ

    torch::Tensor F = torch::eye(dim, torch::kFloat32);

    if (is_localized) {
        // Jacobian for robot pose only [x, y, theta]
        // ∂x'/∂θ = -dx*sin(θ) - dy*cos(θ)
        // ∂y'/∂θ =  dx*cos(θ) - dy*sin(θ)

        F[0][2] = -dx_local * std::sin(theta) - dy_local * std::cos(theta);
        F[1][2] =  dx_local * std::cos(theta) - dy_local * std::sin(theta);
        // F[2][2] = 1.0 (already set by eye)
    } else {
        // Jacobian for full state [w, h, x, y, theta]
        // Room dimensions don't change with motion
        F[2][4] = -dx_local * std::sin(theta) - dy_local * std::cos(theta);
        F[3][4] =  dx_local * std::cos(theta) - dy_local * std::sin(theta);
    }


    // ===== PROCESS NOISE =====
    // Proper motion model: σ = base + k * distance
    float motion_dist = std::sqrt(dx_local*dx_local + dy_local*dy_local);

    // Base uncertainty (sensor noise, discretization, etc) + motion-proportional uncertainty
    float base_trans_noise = 0.002f;  // 2mm base uncertainty
    float trans_noise = base_trans_noise + noise_trans_ * motion_dist;

    float base_rot_noise = 0.005f;  // 5 mrad base uncertainty
    float rot_noise = base_rot_noise + noise_rot_ * std::abs(dtheta);

    torch::Tensor Q = torch::zeros({dim, dim}, torch::kFloat32);

    if (is_localized) {
        // Process noise in robot frame, transformed to global frame
        // This accounts for the fact that motion uncertainty depends on heading
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        // Build noise covariance in robot frame
        float sigma_x = trans_noise;
        float sigma_y = trans_noise;
        float sigma_theta = rot_noise;

        // Transform to global frame: Q_global = R * Q_local * R^T
        Q[0][0] = sigma_x*sigma_x * cos_t*cos_t + sigma_y*sigma_y * sin_t*sin_t;
        Q[0][1] = (sigma_x*sigma_x - sigma_y*sigma_y) * cos_t * sin_t;
        Q[1][0] = Q[0][1];
        Q[1][1] = sigma_x*sigma_x * sin_t*sin_t + sigma_y*sigma_y * cos_t*cos_t;
        Q[2][2] = sigma_theta * sigma_theta;
    } else {
        // Full state: room doesn't accumulate noise, only robot pose
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        float sigma_x = trans_noise;
        float sigma_y = trans_noise;
        float sigma_theta = rot_noise;

        Q[2][2] = sigma_x*sigma_x * cos_t*cos_t + sigma_y*sigma_y * sin_t*sin_t;
        Q[2][3] = (sigma_x*sigma_x - sigma_y*sigma_y) * cos_t * sin_t;
        Q[3][2] = Q[2][3];
        Q[3][3] = sigma_x*sigma_x * sin_t*sin_t + sigma_y*sigma_y * cos_t*cos_t;
        Q[4][4] = sigma_theta * sigma_theta;
    }

    // ===== EKF PREDICTION =====
    // P_pred = F * P_prev * F^T + Q
    torch::Tensor propagated = torch::matmul(torch::matmul(F, prev_cov), F.t()) + Q;

    return propagated;
}

torch::Tensor UncertaintyManager::propagate_with_delta_pose( const Eigen::Vector3f &delta_pose,
                                                             const torch::Tensor& prev_cov,
                                                             bool is_localized) const
{
    int dim = is_localized ? 3 : 5;

    // Get current pose for Jacobian computation
    if (previous_pose_.empty()) {
        // Fallback: simple additive noise
        return prev_cov + noise_trans_ * noise_trans_ * torch::eye(dim, torch::kFloat32);
    }

    float theta = previous_pose_[2];  // Current heading

    // Motion already integrated in global frame!
    // delta_pose = [dx_global, dy_global, dtheta]
    // But we need motion in robot frame for anisotropic noise computation

    // Transform global delta back to robot frame for noise computation
    float cos_t = std::cos(theta);
    float sin_t = std::sin(theta);

    float dx_global = delta_pose[0];
    float dy_global = delta_pose[1];
    float dtheta = delta_pose[2];

    // Inverse rotation: robot_frame = R^T * global_frame
    float dx_local = dx_global * cos_t + dy_global * sin_t;
    float dy_local = -dx_global * sin_t + dy_global * cos_t;

    // ===== MOTION MODEL JACOBIAN =====
    // State: [x, y, theta] (for localized) or [w, h, x, y, theta] (for mapping)

    torch::Tensor F = torch::eye(dim, torch::kFloat32);

    if (is_localized) {
        // Jacobian for robot pose only [x, y, theta]
        // ∂x'/∂θ = -dy_local*sin(θ) - dx_local*cos(θ)
        // ∂y'/∂θ =  dy_local*cos(θ) - dx_local*sin(θ)

        F[0][2] = -dy_local * sin_t - dx_local * cos_t;
        F[1][2] =  dy_local * cos_t - dx_local * sin_t;
    } else {
        // Jacobian for full state [w, h, x, y, theta]
        F[2][4] = -dy_local * sin_t - dx_local * cos_t;
        F[3][4] =  dy_local * cos_t - dx_local * sin_t;
    }

    // ===== PROCESS NOISE =====
    // ANISOTROPIC: For Y=forward, X=right coordinate system:
    //   dy_local = FORWARD motion (should get larger noise)
    //   dx_local = LATERAL motion (should get smaller noise)

    float forward_motion = std::abs(dy_local);   // Forward (Y in robot frame)
    float lateral_motion = std::abs(dx_local);   // Lateral (X in robot frame)

    float base_trans_noise = 0.002f;  // 2mm base uncertainty

    // Forward uncertainty: grows with forward motion
    float forward_noise = base_trans_noise + noise_trans_ * forward_motion;

    // Lateral uncertainty: much smaller (differential drive, 10% of forward)
    float lateral_noise = base_trans_noise + 0.1f * noise_trans_ * lateral_motion;

    float base_rot_noise = 0.005f;  // 5 mrad base uncertainty
    float rot_noise = base_rot_noise + noise_rot_ * std::abs(dtheta);

    torch::Tensor Q = torch::zeros({dim, dim}, torch::kFloat32);

    if (is_localized) {
        // Build noise covariance in robot frame: Q = diag(lateral², forward², theta²)
        float sigma_x = lateral_noise;   // X = right/lateral (smaller)
        float sigma_y = forward_noise;   // Y = forward (larger)
        float sigma_theta = rot_noise;

        // Transform to global frame: Q_global = R * Q_local * R^T
        // For X=right, Y=forward:
        Q[0][0] = sigma_x*sigma_x * sin_t*sin_t + sigma_y*sigma_y * cos_t*cos_t;
        Q[0][1] = (sigma_y*sigma_y - sigma_x*sigma_x) * cos_t * sin_t;
        Q[1][0] = Q[0][1];
        Q[1][1] = sigma_x*sigma_x * cos_t*cos_t + sigma_y*sigma_y * sin_t*sin_t;
        Q[2][2] = sigma_theta * sigma_theta;
    } else {
        // Full state: room doesn't accumulate noise, only robot pose
        float sigma_x = lateral_noise;
        float sigma_y = forward_noise;
        float sigma_theta = rot_noise;

        Q[2][2] = sigma_y*sigma_y * cos_t*cos_t + sigma_x*sigma_x * sin_t*sin_t;
        Q[2][3] = (sigma_y*sigma_y - sigma_x*sigma_x) * cos_t * sin_t;
        Q[3][2] = Q[2][3];
        Q[3][3] = sigma_y*sigma_y * sin_t*sin_t + sigma_x*sigma_x * cos_t*cos_t;
        Q[4][4] = sigma_theta * sigma_theta;
    }

    // ===== EKF PREDICTION =====
    // P_pred = F * P_prev * F^T + Q
    torch::Tensor propagated = torch::matmul(torch::matmul(F, prev_cov), F.t()) + Q;

    return propagated;

}

torch::Tensor UncertaintyManager::fuse_covariances( const torch::Tensor& propagated, const torch::Tensor& measurement)
const
{
    try {
        // Information form fusion: I_fused = I_prop + I_meas
        auto I = torch::eye(propagated.size(0), propagated.options());

        // Regularize for numerical stability
        torch::Tensor prop_reg = propagated + reg_factor_ * I;
        torch::Tensor meas_reg = measurement + reg_factor_ * I;

        // Convert to information matrices
        torch::Tensor I_prop = torch::inverse(prop_reg);
        torch::Tensor I_meas = torch::inverse(meas_reg);

        // Check for NaN after inversion
        if (torch::any(torch::isnan(I_prop)).item<bool>() ||
            torch::any(torch::isnan(I_meas)).item<bool>()) {
            return torch::full_like(propagated, std::numeric_limits<float>::quiet_NaN());
        }

        // Fuse information
        torch::Tensor I_fused = I_prop + I_meas;
        torch::Tensor cov_fused = torch::inverse(I_fused);

        return cov_fused;

    } catch (const std::exception& e) {
        qWarning() << "Covariance fusion exception:" << e.what();
        return torch::full_like(propagated, std::numeric_limits<float>::quiet_NaN());
    }
}

bool UncertaintyManager::is_valid_covariance(const torch::Tensor& cov, int expected_dim)
const
{
    if (cov.numel() == 0) return false;
    if (cov.dim() != 2) return false;
    if (cov.size(0) != expected_dim || cov.size(1) != expected_dim) return false;
    if (torch::any(torch::isnan(cov)).item<bool>()) return false;
    if (torch::any(torch::isinf(cov)).item<bool>()) return false;
    return true;
}

torch::Tensor UncertaintyManager::create_fallback_covariance(int dim)
const
{
    return torch::eye(dim, torch::kFloat32) * 0.01f;
}

void UncertaintyManager::compute_motion_metrics(
    const std::vector<float>& current_pose,
    float& translation,
    float& rotation) const
{
    if (!has_history_) {
        translation = 0.0f;
        rotation = 0.0f;
        return;
    }

    float dx = current_pose[0] - previous_pose_[0];
    float dy = current_pose[1] - previous_pose_[1];
    float dtheta = current_pose[2] - previous_pose_[2];

    // Normalize angle difference
    while (dtheta > M_PI) dtheta -= 2 * M_PI;
    while (dtheta < -M_PI) dtheta += 2 * M_PI;

    translation = std::sqrt(dx*dx + dy*dy);
    rotation = std::abs(dtheta);
}

void UncertaintyManager::set_motion_noise(float translation_per_meter, float rotation_per_radian)
{
    noise_trans_ = translation_per_meter;
    noise_rot_ = rotation_per_radian;
    qInfo() << "Motion noise set: trans=" << noise_trans_
            << "m/m, rot=" << noise_rot_ << "rad/rad";
}

void UncertaintyManager::reset()
{
    previous_pose_.clear();
    previous_cov_ = torch::Tensor();
    has_history_ = false;
    qInfo() << "UncertaintyManager history reset";
}

void UncertaintyManager::set_previous_cov(const torch::Tensor& cov)
{
    previous_cov_ = cov.clone();
}

void UncertaintyManager::set_previous_pose(const std::vector<float>& pose)
{
    previous_pose_ = pose;
    has_history_ = true;
}

torch::Tensor UncertaintyManager::get_previous_cov() const
{
    return previous_cov_;
}