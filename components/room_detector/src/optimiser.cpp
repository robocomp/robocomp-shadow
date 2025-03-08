//
// Created by robolab on 12/10/24.
//

#include "optimiser.h"

namespace rc
{
    std::tuple<double, double, double>
    optimise(const Match &matches, const Eigen::Affine2d &robot_pose)
    {
        // 1. Define the initial pose of the robot in the room frame
        Eigen::Rotation2Df rot(robot_pose.rotation());
        // Get rot angle, diff PI/2 and keep the value between -PI and PI
        const double mod_angle = keep_angle_between_minus_pi_and_pi(rot.angle() + M_PI_2);

        const gtsam::Pose2 initialPose(robot_pose.translation().x() / 1000, robot_pose.translation().y() / 1000, mod_angle); // Example: x=1, y=2, theta=30 degrees

        // 2. Create a GTSAM factor graph
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initialEstimate;

        // 3. Add a prior factor for the initial pose
        const gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
                gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(1, 1, 1)); // Adjust noise as needed
        graph.add(gtsam::PriorFactor<gtsam::Pose2>(gtsam::Symbol('x', 0), initialPose, priorNoise));
        initialEstimate.insert(gtsam::Symbol('x', 0), initialPose);

        // 4. Add prior factors for the nominal landmark positions
        for (const auto &[i, match] : matches | iter::enumerate)
        {
            const auto &[measurement_corner, nominal_corner, range, bearing, error] = match;
            if(error < 750)
            {
                // Add a prior factor for the landmark positions in the room frame. TODO:: no serían las nominales directamente?
                gtsam::noiseModel::Diagonal::shared_ptr landmarkNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.01, 0.01)); // Adjust noise as needed
                gtsam::PriorFactor<gtsam::Point2> pfactor(gtsam::Symbol('l', i),
                                                            gtsam::Point2(measurement_corner.x() / 1000, measurement_corner.y() / 1000),
                                                                 landmarkNoise);
                graph.add(pfactor);
                // Add a BearingRange factor as the difference between the nominal and measurement corners (error) pre-computed in the match
                gtsam::noiseModel::Diagonal::shared_ptr measurement_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.001, 0.01));
                gtsam::BearingRangeFactor<gtsam::Pose2, Eigen::Vector2d> brfactor(gtsam::Symbol('x', 0),
                                                                                  gtsam::Symbol('l', i),
                                                                                          gtsam::Rot2(-bearing),
                                                                                          range / 1000,
                                                                                          measurement_noise);

                // gtsam::BearingRangeFactor<gtsam::Symbol, gtsam::Symbol, gtsam::Rot2, double>
                //     brfactor_(gtsam::Symbol('x', 0), gtsam::Symbol('l', i),
                //         gtsam::Rot2(-bearing), range / 1000, measurement_noise);  // TODO: check this

                graph.add(brfactor);
                initialEstimate.insert(gtsam::Symbol('l', i), gtsam::Point2(nominal_corner.x() / 1000, nominal_corner.y() / 1000));
            }
        }

        // 6. Optimize the factor graph
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
        const gtsam::Values result = optimizer.optimize();

        // 9. Get the optimized pose
        const auto optimizedPose = result.at<gtsam::Pose2>(gtsam::Symbol('x', 0));

        // 10. Print the results
        //    std::cout << __FUNCTION__ << " Initial pose: " << initialPose << std::endl;
        //    std::cout << __FUNCTION__ << " Optimized pose: " << optimizedPose << std::endl;

        // Transform angle adding PI/2 and keep the value between -PI and PI
        double theta = keep_angle_between_minus_pi_and_pi(optimizedPose.theta() - M_PI_2);
        return {optimizedPose.x() * 1000, optimizedPose.y() * 1000, theta};
    }

    // Keep the angle between -PI and PI
    double keep_angle_between_minus_pi_and_pi(double angle) // TODO: SACAR A UNA LIBRERIA DE UTILIDADES???
    {
        while (angle > M_PI)
            angle -= 2 * M_PI;
        while (angle < -M_PI)
            angle += 2 * M_PI;
        return angle;
    }
}