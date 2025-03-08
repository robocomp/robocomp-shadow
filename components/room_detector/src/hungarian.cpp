//
// Created by robolab on 12/5/24.
//
#include "hungarian.h"

namespace rc
{
    Match hungarian(const std::vector<Eigen::Vector3d> &measurement_corners,
              const std::vector<Eigen::Vector3d> &nominal_corners_in_robot,
              const Eigen::Affine2d &robot_current_pose,
              double max_corner_diff)
    {
        // create cost matrix for Hungarian //
        std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double, double, double>> matches;   //  measurement - nominal

        std::vector<double> costs;
        for (const auto &c: measurement_corners)    // rows
            for (const auto &rc: nominal_corners_in_robot)   // cols
                costs.emplace_back((c - rc).norm());
        const auto rows = measurement_corners.size();
        const auto cols = nominal_corners_in_robot.size();

        // if costs is empty, return empty matches
        if (costs.empty())
            return matches;

        // lambda to access the costs matrix
        auto cost = [costs, cols](const unsigned r, const unsigned c) { return costs[r * cols + c]; };
        const auto matching = munkres_algorithm<double>(rows, cols, cost);
        for (const auto &[r, c]: matching)
        {
            if (cost(r, c) < max_corner_diff)
            {
                // Transform the corner to the robot reference system
                auto measurement_corner_robot = robot_current_pose.inverse() * measurement_corners[r];  // TODO: por qué conviertes al robot measurement_corners?
                matches.emplace_back(nominal_corners_in_robot[c],
                                     measurement_corners[r],
                                     measurement_corner_robot.norm(),
                                     std::atan2(measurement_corner_robot.x(), measurement_corner_robot.y()),
                                    (nominal_corners_in_robot[c] - measurement_corners[r]).norm());
            }
        }
        return matches;
    }


    Match hungarian(const Corners &measurement_corners,
             const std::vector<Eigen::Vector3d> &nominal_corners_in_robot,
             const Eigen::Affine2d &robot_current_pose,
             double max_corner_diff)
    {
        std::vector<Eigen::Vector3d> m_corners;
        std::ranges::transform(measurement_corners, std::back_inserter(m_corners), [](const auto &c)
                { const auto &[_, p, timestamp] = c; return Eigen::Vector3d{p.x(), p.y(), 1.0f};});
    }
}

