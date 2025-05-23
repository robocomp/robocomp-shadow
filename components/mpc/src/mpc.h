//
// Created by pbustos on 3/04/22.
//

#ifndef LOCAL_GRID_MPC_H
#define LOCAL_GRID_MPC_H

#include <casadi/casadi.hpp>
#include <casadi/core/optistack.hpp>
#include <Eigen/Dense>
#include <tuple>
#include <optional>
#include <QtCore>
#include <QGraphicsEllipseItem>

namespace rc
{
    class MPC
    {
        public:
            casadi::Opti initialize_differential(int N);
            casadi::Opti initialize_omni(int N);
            std::optional<std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector2f>>> // control and state vectors
            update(const std::vector<Eigen::Vector2f> &path, std::vector<Eigen::Vector2f> &obstacles);
            casadi::Opti opti;

        private:
            struct Constants
            {
                unsigned int num_steps = 8;             // MPC steps ahead
                float time_interval = 0.2;              // seconds
                const float robot_radius = 250;
                double gauss_dist = 0.1;                // minimun distance to a lidar-gaussian as constraint
                double point_dist = 0.2;                // value in lidar-gaussian at lidar points (corners)
                double point_sigma = 0.07;              // variance for point (corners) gaussians
                double gauss_value_for_point = 0.3;     // minimun distance to a corner gaussian as constraint
                float min_dist_to_target = 0.9;         // min distance to target at which the robot stops
                double max_rotation_value = 2;        // max rotation constraint in rads/sg
                double min_rotation_value = -2;           // max advance constraint in m/sg
                double max_advance_value = 1.2;           // max advance constraint in m/sg
                double min_advance_value = -0.3;           // min advance constraint in m/sg
                double max_side_value = 1;              // max advance constraint in m/sg
                double min_side_value = -1;              // min advance constraint in m/sg
                double xset_gaussian = 0.5;              // gaussian break x set value
                double yset_gaussian = 0.5;             // gaussian break y set value
                double min_line_dist = 0.4;
                float max_RDP_deviation = 70;
                float laser_noise_sigma  = 15;
                float max_ball_radius = 1 ;             // meters
                const float peak_threshold = 500;       // opening detector
                const float target_noise_sigma = 50;
                const int num_lidar_affected_rays_by_hard_noise = 1;
                double MAX_DIST_TO_OBSTACLE = 0.5;      // meters
            };
            Constants consts;

            std::vector<double> previous_values_of_solution, previous_control_of_solution;
            casadi::MX state;
            casadi::MX pos;
            casadi::MX rot;
            casadi::MX phi;
            casadi::MX control;
            casadi::MX adv, side;
            casadi::MX slack_vector;

            std::vector<double> e2v(const Eigen::Vector2d &v);
            void draw_path(const std::vector<double> &path_robot_meters, QGraphicsPolygonItem *robot_polygon, QGraphicsScene *scene);
            float gaussian(float x);
    };
} // mpc
#endif //LOCAL_GRID_MPC_H
