//
// Created by pbustos on 16/11/25.
//

#include "uncertainty_estimator.h"
#include <QDebug>

// ============================================================================
// Uncertainty Estimation using Laplace Approximation - ADAPTIVE VERSION
// ============================================================================

torch::Tensor UncertaintyEstimator::compute_covariance(const torch::Tensor &points,
                                                       RoomModel &room,
                                                       float wall_thickness)
{
    // Get all parameters
    const auto all_params = room.parameters();

    // Filter only parameters that require gradients
    std::vector<torch::Tensor> params;
    for (const auto &p: all_params)
        if (p.requires_grad())
            params.push_back(p);

    if (params.empty())
    {
        std::cerr << "ERROR: No parameters require gradients!\n";
        return torch::eye(5, torch::kFloat32); // Return identity as fallback
    }

    // Count total number of parameters
    int total_params = 0;
    for (const auto &p: params)
        total_params += p.numel();

    // Compute loss (ensure requires_grad is true)
    const auto loss = RoomLoss::compute_loss(points, room, wall_thickness);

    // Compute gradients (first derivatives)
    const auto gradients = torch::autograd::grad({loss}, params,
                                                 {}, // grad_outputs
                                                 true, // retain_graph
                                                 true); // create_graph

    // Flatten all gradients into a single vector
    std::vector<torch::Tensor> grad_flat;
    for (const auto &g: gradients)
        grad_flat.push_back(g.flatten());
    const auto grad_vector = torch::cat(grad_flat);

    // Compute Hessian matrix
    torch::Tensor hessian = torch::zeros({total_params, total_params}, torch::kFloat32);

    // Compute each row of Hessian (second derivatives)
    for (int i = 0; i < total_params; ++i)
    {
        torch::Tensor grad_for_this_param = torch::zeros_like(grad_vector);
        grad_for_this_param[i] = 1.0;

        const auto second_derivs = torch::autograd::grad({grad_vector}, params,
                                                         {grad_for_this_param},
                                                         true, // retain_graph
                                                         false, // create_graph
                                                         true); // allow_unused

        std::vector<torch::Tensor> second_flat;
        for (const auto &sd: second_derivs)
        {
            if (sd.defined())
                 second_flat.push_back(sd.flatten());
            else
                 second_flat.push_back(torch::zeros({total_params}, torch::kFloat32));
        }
        const auto second_vector = torch::cat(second_flat);
        hessian[i] = second_vector;
    }

    // Ensure numerical symmetry
    hessian = 0.5 * (hessian + hessian.transpose(0, 1));

    // Robust inversion strategy
    torch::Tensor covariance;
    const auto I = torch::eye(total_params, hessian.options());
    double reg = 1e-6;
    const int max_attempts = 6;
    bool success = false;

    for (int attempt = 0; attempt < max_attempts; ++attempt)
    {
        const auto h_reg = hessian + reg * I;
        try
        {
            // Try Cholesky and build inverse from the factor
            const auto L = torch::linalg_cholesky(h_reg, /*upper=*/false);
            covariance = torch::cholesky_inverse(L, /*upper=*/false);
            success = true;
            break;
        } catch (const c10::Error &) { reg *= 10.0; }
    }

    if (not success)
    {
        // First fallback: pseudo-inverse (can handle singular/indefinite matrices)
        try
        {
            covariance = torch::pinverse(hessian + reg * I);

            // Check if result is valid
            if (torch::any(torch::isnan(covariance)).item<bool>() or
                torch::any(torch::isinf(covariance)).item<bool>())
                 throw std::runtime_error("Pinverse produced NaN/Inf");
         } catch (const std::exception &)
         {
            // Last-resort fallback: conservative diagonal covariance
            qWarning() << "Pinverse failed, using diagonal fallback";
            const torch::Tensor diag = torch::abs(torch::diag(hessian)).clone();

            // Replace zeros/small values with conservative estimate
            float *diag_ptr = diag.data_ptr<float>();
            for (int i = 0; i < diag.size(0); ++i)
                if (diag_ptr[i] < 1e-6f)
                    diag_ptr[i] = 1e-2f; // Conservative: high uncertainty

            covariance = torch::diag(1.0 / diag);
        }
    }

    // Final validation before returning
    if (torch::any(torch::isnan(covariance)).item<bool>() ||
        torch::any(torch::isinf(covariance)).item<bool>())
    {
        qWarning() << "Covariance computation produced NaN/Inf, using identity fallback";
        covariance = torch::eye(total_params, torch::kFloat32) * 0.01f; // 10cm variance
    }

    covariance = covariance / static_cast<float>(points.size(0));

    // One more check after normalization
    if (torch::any(torch::isnan(covariance)).item<bool>() ||
        torch::any(torch::isinf(covariance)).item<bool>())
    {
        qWarning() << "NaN after normalization, using safe fallback";
        covariance = torch::eye(total_params, torch::kFloat32) * 0.01f;
    }

    return covariance.detach();
}

 std::vector<float> UncertaintyEstimator::get_std_devs(const torch::Tensor &covariance)
 {
     const auto variances = torch::diagonal(covariance);
     const auto std_devs = torch::sqrt(torch::clamp(variances, 0.0, 1e10));

     std::vector<float> result;
     const auto acc = std_devs.accessor<float, 1>();
    for (int i = 0; i < std_devs.size(0); ++i)
        result.push_back(acc[i]);

     return result;
 }

 torch::Tensor UncertaintyEstimator::get_correlation_matrix(const torch::Tensor &covariance)
 {
     auto std_devs = torch::sqrt(torch::diagonal(covariance));
     std_devs = torch::clamp(std_devs, 1e-10, 1e10);

     const int n = covariance.size(0);
     torch::Tensor correlation = torch::zeros({n, n}, torch::kFloat32);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            correlation[i][j] = covariance[i][j] / (std_devs[i] * std_devs[j]);

     return correlation;
 }

 void UncertaintyEstimator::print_uncertainty(const torch::Tensor &covariance, const RoomModel &room)
 {
     const auto std_devs = get_std_devs(covariance);
     const auto room_params = room.get_room_parameters();
     const auto robot_pose = room.get_robot_pose();
     const bool room_frozen = room.are_room_parameters_frozen();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== UNCERTAINTY ESTIMATES ===\n";
    std::cout << "(1-sigma / 68% confidence intervals)\n\n";

    if (room_frozen)
    {
        std::cout << "Room Parameters (FROZEN):\n";
        std::cout << "  Half-width:   " << room_params[0] << " m (frozen)\n";
        std::cout << "  Half-height:  " << room_params[1] << " m (frozen)\n";
        std::cout << "\nRobot Pose (OPTIMIZED):\n";
        std::cout << "  Position X:   " << robot_pose[0] << " ± " << std_devs[0] << " m\n";
        std::cout << "  Position Y:   " << robot_pose[1] << " ± " << std_devs[1] << " m\n";
        std::cout << "  Orientation:  " << robot_pose[2] << " ± " << std_devs[2]
                << " rad (" << (std_devs[2] * 180.0 / M_PI) << " deg)\n";
    } else
    {
        std::cout << "Room Parameters (center FIXED at origin):\n";
        std::cout << "  Half-width:   " << room_params[0] << " ± " << std_devs[0] << " m\n";
        std::cout << "  Half-height:  " << room_params[1] << " ± " << std_devs[1] << " m\n";

        std::cout << "\nRobot Pose (relative to room):\n";
        std::cout << "  Position X:   " << robot_pose[0] << " ± " << std_devs[2] << " m\n";
        std::cout << "  Position Y:   " << robot_pose[1] << " ± " << std_devs[3] << " m\n";
        std::cout << "  Orientation:  " << robot_pose[2] << " ± " << std_devs[4]
                << " rad (" << (std_devs[4] * 180.0 / M_PI) << " deg)\n";

        std::cout << "\nFull Room Dimensions:\n";
        const float width = 2 * room_params[0];
        const float height = 2 * room_params[1];
        const float width_std = 2 * std_devs[0];
        const float height_std = 2 * std_devs[1];
        std::cout << "  Width:        " << width << " ± " << width_std << " m\n";
        std::cout << "  Height:       " << height << " ± " << height_std << " m\n";
    }

    // Compute correlation matrix
    const auto correlation = get_correlation_matrix(covariance);

    std::cout << "\n=== KEY CORRELATIONS ===\n";
    std::cout << std::setprecision(3);

    const auto corr_acc = correlation.accessor<float, 2>();

    if (room_frozen)
    {
        // Only 3x3 covariance (robot pose only)
        std::cout << "Robot X vs Robot Y:         " << corr_acc[0][1] << "\n";
        std::cout << "Robot X vs Robot Theta:     " << corr_acc[0][2] << "\n";
        std::cout << "Robot Y vs Robot Theta:     " << corr_acc[1][2] << "\n";
    } else
    {
        // Full 5x5 covariance
        std::cout << "Robot X vs Robot Y:         " << corr_acc[2][3] << "\n";
        std::cout << "Robot X vs Robot Theta:     " << corr_acc[2][4] << "\n";
        std::cout << "Robot X vs Room Width:      " << corr_acc[2][0] << "\n";
        std::cout << "Robot Y vs Room Height:     " << corr_acc[3][1] << "\n";
        std::cout << "Room Width vs Height:       " << corr_acc[0][1] << "\n";
    }

    std::cout << "\n(Correlations close to ±1 indicate strong dependence)\n";
}
