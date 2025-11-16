//
// Created by pbustos on 16/11/25.
//

#ifndef UNCERTAINTY_ESTIMATOR_H
#define UNCERTAINTY_ESTIMATOR_H

#include <torch/torch.h>
#include <vector>
#include "room_model.h"
#include "room_loss.h"

/**
 * @brief Compute covariance matrix using Laplace approximation (ADAPTIVE)
 *
 * Automatically detects which parameters require gradients and computes
 * covariance only for those parameters.
 *
 * When room is frozen (LOCALIZED):
 *   - Returns 3×3 covariance for [robot_x, robot_y, robot_theta]
 * When room is unfrozen (MAPPING):
 *   - Returns 5×5 covariance for [half_width, half_height, robot_x, robot_y, robot_theta]
 *
 * @param points LiDAR points tensor [N, 2] in robot frame
 * @param room Room model at MAP estimate
 * @param wall_thickness Same as used in loss computation
 * @return Covariance matrix [n, n] where n = number of unfrozen parameters
 */

class UncertaintyEstimator
{
    public:

        static torch::Tensor compute_covariance(const torch::Tensor& points,
                                               RoomModel& room,
                                               float wall_thickness = 0.1f);

        /**
         * @brief Extract standard deviations (sqrt of diagonal of covariance)
         *
         * @param covariance Covariance matrix [n, n]
         * @return Vector of std devs (size depends on frozen state)
         */
        static std::vector<float> get_std_devs(const torch::Tensor& covariance);

        /**
         * @brief Extract correlation matrix from covariance matrix
         *
         * @param covariance Covariance matrix [n, n]
         * @return Correlation matrix [n, n] with values in [-1, 1]
         */
        static torch::Tensor get_correlation_matrix(const torch::Tensor& covariance);

        /**
         * @brief Print uncertainty information in readable format (ADAPTIVE)
         *
         * Adapts output based on whether room is frozen or not.
         *
         * @param covariance Covariance matrix [n, n]
         * @param room Room model to label parameters
         */
        static void print_uncertainty(const torch::Tensor& covariance, const RoomModel& room);
};

#endif //UNCERTAINTY_ESTIMATOR_H
