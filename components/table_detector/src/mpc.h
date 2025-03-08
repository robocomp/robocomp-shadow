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
            update(const std::vector<Eigen::Vector2f> &path, const std::vector<Eigen::Vector2f> &obstacles);
            casadi::Opti opti;

        private:
            struct Constants
            {
                unsigned int num_steps = 10;             // MPC steps ahead
                float time_interval = 0.3;              // seconds
                float min_dist_to_target = 1.2;         // min distance to target at which the robot stops
                double max_rotation_value = 2;        // max rotation constraint in rads/sg
                double min_rotation_value = -2;           // max advance constraint in m/sg
                double max_rotation_acc = 1;        // max rotation constraint in rads/sg
                double min_rotation_acc = -1;           // max advance constraint in m/sg
                double max_advance_value = 1.8;           // max advance constraint in m/sg
                double min_advance_value = -1;           // min advance constraint in m/sg
                double max_side_value = 1;              // max advance constraint in m/sg
                double min_side_value = -1;              // min advance constraint in m/sg
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

            inline std::vector<double> e2v(const Eigen::Vector2d &v);
            void draw_path(const std::vector<double> &path_robot_meters, QGraphicsPolygonItem *robot_polygon, QGraphicsScene *scene);
    };
} // mpc
#endif //LOCAL_GRID_MPC_H
