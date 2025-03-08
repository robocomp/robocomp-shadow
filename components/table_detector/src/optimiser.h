//
// Created by robolab on 12/10/24.
//

#ifndef ROOM_DETECTOR_OPTIMISER_H
#define ROOM_DETECTOR_OPTIMISER_H

#include "pch.h"
#include "common_types.h"

namespace rc
{
    std::tuple<double, double, double>
    optimise(const Match &matches,  const Eigen::Affine2d &robot_pose);     // measurement in room - nominal - distance - angle - error
    double keep_angle_between_minus_pi_and_pi(double angle);

} // rc

#endif //ROOM_DETECTOR_OPTIMISER_H
