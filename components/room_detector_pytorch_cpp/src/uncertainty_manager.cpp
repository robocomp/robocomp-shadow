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

UncertaintyManager::Result UncertaintyManager::compute(
    const torch::Tensor& points,
    RoomModel& room,
    float huber_delta,
    bool is_localized)
{
    Result result;
    const auto current_pose = room.get_robot_pose();
    const int expected_dim = is_localized ? 3 : 5;

    // ===== STEP 1: MEASUREMENT UPDATE =====
    torch::Tensor measurement_cov = compute_measurement_covariance(
        points, room, huber_delta, is_localized
    );

    if (!is_valid_covariance(measurement_cov, expected_dim)) {
        qWarning() << "Invalid measurement covariance. Using fallback.";
        result.covariance = create_fallback_covariance(expected_dim);
        result.std_devs = UncertaintyEstimator::get_std_devs(result.covariance);
        result.is_valid = false;
        result.error_message = "Measurement covariance invalid";
        return result;
    }

    // ===== STEP 2: MOTION PROPAGATION (if enabled and history exists) =====
    torch::Tensor propagated_cov;
    bool can_propagate = (use_propagation_ && has_history_ &&
                         previous_cov_.numel() > 0 &&
                         previous_cov_.size(0) == expected_dim);

    if (can_propagate) {
        propagated_cov = propagate_with_motion(current_pose, previous_cov_, is_localized);

        if (!is_valid_covariance(propagated_cov, expected_dim)) {
            qWarning() << "Invalid propagated covariance. Skipping propagation.";
            can_propagate = false;
        } else {
            result.used_propagation = true;
        }
    }

    // ===== STEP 3: FUSION (if propagation was successful) =====
    if (can_propagate) {
        result.covariance = fuse_covariances(propagated_cov, measurement_cov);

        if (!is_valid_covariance(result.covariance, expected_dim)) {
            qWarning() << "Fusion produced invalid covariance. Using measurement only.";
            result.covariance = measurement_cov;
            result.used_fusion = false;
        } else {
            result.used_fusion = true;
        }
    } else {
        // First frame or invalid propagation - use measurement only
        result.covariance = measurement_cov;
    }

    // ===== STEP 4: UPDATE HISTORY =====
    if (is_valid_covariance(result.covariance, expected_dim)) {
        previous_pose_ = current_pose;
        previous_cov_ = result.covariance.clone();
        has_history_ = true;
    }

    // ===== STEP 5: COMPUTE STANDARD DEVIATIONS =====
    result.std_devs = UncertaintyEstimator::get_std_devs(result.covariance);
    result.is_valid = true;

    return result;
}

torch::Tensor UncertaintyManager::compute_measurement_covariance(
    const torch::Tensor& points,
    RoomModel& room,
    float huber_delta,
    bool is_localized)
{
    try {
        return UncertaintyEstimator::compute_covariance(points, room, huber_delta);
    } catch (const std::exception& e) {
        qWarning() << "Measurement covariance computation failed:" << e.what();
        return torch::full({is_localized ? 3 : 5, is_localized ? 3 : 5},
                          std::numeric_limits<float>::quiet_NaN());
    }
}

torch::Tensor UncertaintyManager::propagate_with_motion(
    const std::vector<float>& current_pose,
    const torch::Tensor& previous_cov,
    bool is_localized)
{
    // Calculate motion
    float translation, rotation;
    compute_motion_metrics(current_pose, translation, rotation);

    // Compute process noise
    float trans_var = std::pow(noise_trans_ * translation, 2);
    float rot_var = std::pow(noise_rot_ * rotation, 2);

    // Start from previous covariance
    torch::Tensor propagated = previous_cov.clone();

    // Add process noise to appropriate indices
    if (is_localized) {
        // 3x3: [robot_x, robot_y, robot_theta]
        auto cov_acc = propagated.accessor<float, 2>();
        cov_acc[0][0] += trans_var;  // X variance
        cov_acc[1][1] += trans_var;  // Y variance
        cov_acc[2][2] += rot_var;    // Theta variance

        // Add motion-correlated noise
        if (translation > 0.01f) {
            float dx = current_pose[0] - previous_pose_[0];
            float dy = current_pose[1] - previous_pose_[1];
            float motion_angle = std::atan2(dy, dx);
            float cos_a = std::cos(motion_angle);
            float sin_a = std::sin(motion_angle);

            cov_acc[0][1] += trans_var * cos_a * sin_a;
            cov_acc[1][0] = cov_acc[0][1];
        }
    } else {
        // 5x5: [half_width, half_height, robot_x, robot_y, robot_theta]
        auto cov_acc = propagated.accessor<float, 2>();
        // Room parameters don't grow with motion
        cov_acc[2][2] += trans_var;  // Robot X
        cov_acc[3][3] += trans_var;  // Robot Y
        cov_acc[4][4] += rot_var;    // Robot theta

        if (translation > 0.01f) {
            float dx = current_pose[0] - previous_pose_[0];
            float dy = current_pose[1] - previous_pose_[1];
            float motion_angle = std::atan2(dy, dx);
            float cos_a = std::cos(motion_angle);
            float sin_a = std::sin(motion_angle);

            cov_acc[2][3] += trans_var * cos_a * sin_a;
            cov_acc[3][2] = cov_acc[2][3];
        }
    }

    return propagated;
}

torch::Tensor UncertaintyManager::propagate_with_velocity( const VelocityCommand& cmd, float dt,
                                                           const torch::Tensor& prev_cov,
                                                           bool is_localized)
{
    // Process noise stddev (proportional to command magnitude)
    float trans_noise = (std::abs(cmd.adv_x) + std::abs(cmd.adv_z)) / 1000.0f * dt * noise_trans_;
    float rot_noise = std::abs(cmd.rot) * dt * noise_rot_;

    // Build process noise matrix Q (diagonal)
    int dim = is_localized ? 3 : 5;
    torch::Tensor Q = torch::eye(dim, torch::kFloat32);

    if (is_localized)
    {
        Q[0][0] = trans_noise * trans_noise;  // X
        Q[1][1] = trans_noise * trans_noise;  // Y
        Q[2][2] = rot_noise * rot_noise;      // Theta
    } else
    {
        Q[2][2] = trans_noise * trans_noise;  // Robot X
        Q[3][3] = trans_noise * trans_noise;  // Robot Y
        Q[4][4] = rot_noise * rot_noise;      // Robot Theta
        // Room parameters don't drift
    }

    // State transition Jacobian F = I (simple integration)
    // For more accuracy, compute actual Jacobian:
    // F[0][2] = -dx*sin(theta) - dy*cos(theta)
    // F[1][2] =  dx*cos(theta) - dy*sin(theta)

    torch::Tensor F = torch::eye(dim, torch::kFloat32);
    torch::Tensor propagated = torch::matmul(torch::matmul(F, prev_cov), F.t()) + Q;

    // // Regularización para evitar singularidad
    // auto I = torch::eye(propagated.size(0), propagated.options());
    // propagated = propagated + reg_factor_ * I;
    //
    // // Verificar que la matriz sea válida antes de continuar
    // if (torch::any(torch::isnan(propagated)).item<bool>() ||
    //     torch::any(torch::isinf(propagated)).item<bool>())
    // {
    //     qWarning() << "Propagated covariance contains invalid values";
    //     return torch::full_like(prev_cov, std::numeric_limits<float>::quiet_NaN());
    // }

    return propagated;
}

torch::Tensor UncertaintyManager::fuse_covariances( const torch::Tensor& propagated, const torch::Tensor& measurement)
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
{
    if (cov.numel() == 0) return false;
    if (cov.dim() != 2) return false;
    if (cov.size(0) != expected_dim || cov.size(1) != expected_dim) return false;
    if (torch::any(torch::isnan(cov)).item<bool>()) return false;
    if (torch::any(torch::isinf(cov)).item<bool>()) return false;
    return true;
}

torch::Tensor UncertaintyManager::create_fallback_covariance(int dim)
{
    return torch::eye(dim, torch::kFloat32) * 0.01f;
}

void UncertaintyManager::compute_motion_metrics(
    const std::vector<float>& current_pose,
    float& translation,
    float& rotation)
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