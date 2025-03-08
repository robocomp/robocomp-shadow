//
// Created by robolab on 12/5/24.
//

#ifndef BETA_ROBOTICA_CLASS_PRIVATE_HUNGARIAN_H
#define BETA_ROBOTICA_CLASS_PRIVATE_HUNGARIAN_H

#include <Eigen/Geometry>
#include "munkres.hpp"
#include "common_types.h"

namespace rc
{
    /**
    * @brief Hungarian function. Computes the combinatorial minimum cost matching of two sets of points.
    * @param measurement_corners in global frame
    * @param nominal_corners_in_robot in global frame
    * @param robot_current_pose from global to robot frame
    * @param max_corner_diff
    * @return Match
    */
    Match hungarian(const std::vector<Eigen::Vector3d> &measurement_corners,
              const std::vector<Eigen::Vector3d> &nominal_corners_in_robot,
              const Eigen::Affine2d &robot_current_pose,
              double max_corner_diff = std::numeric_limits<double>::max());

    /**
    * @brief Overloading of the hungarian function to work with corners.
    * @param measurement_corners
    * @param nominal_corners_in_robot
    * @param robot_current_pose
    * @param max_corner_diff
    * @return Match
    */
    Match hungarian(const Corners &measurement_corners,
              const std::vector<Eigen::Vector3d> &nominal_corners_in_robot,
              const Eigen::Affine2d &robot_current_pose,
              double max_corner_diff = std::numeric_limits<double>::max());
} // rc

#endif //BETA_ROBOTICA_CLASS_PRIVATE_HUNGARIAN_H
