//
// Created by pbustos on 9/12/24.
//

#ifndef ACTIONABLE_THING_H
#define ACTIONABLE_THING_H

#include <vector>
#include "room.h"
#include "common_types.h"
#include "dbscan.h"
#include "actionable_room.h"

#include <Eigen/Dense>
#include "custom_factor_lidarpoints.h"
//#include "custom_expression_point_factor.h"

namespace rc
{
    class ActionableThing
    {
        public:
            ActionableThing();
            bool initialize(const ActionableThing &progenitor,
                            const std::vector<ActionableThing> &actionables,
                            const rc::ActionableRoom &best_actionable_room,
                            const std::vector<Eigen::Vector3d> residuals,
                            QGraphicsScene *scene);

            gtsam::Vector5 create_factor_graph(const std::vector<Eigen::Vector3d> &residual_points, const Eigen::Affine2d &robot_pose, QGraphicsScene *scene);
            void plotUncertaintyEllipses(const Eigen::MatrixXd& covariance, const Eigen::Matrix<double, 5, 1> &params, QGraphicsScene* scene);

            Eigen::AlignedBox2d fit_rectangle_to_lidar_points(const LidarPoints &points,const rc::ActionableRoom &current_room);
            LidarPoints project(const LidarPoints &points, const rc::ActionableRoom &current_room);
            void draw_box(const std::vector<Eigen::AlignedBox2d> &boxes, QGraphicsScene *scene);
            void draw_fridge(const auto &corners, const QColor &color, QGraphicsScene *scene) const;
            void draw_clusters(const std::vector<cv::RotatedRect> &rects, QGraphicsScene *scene);
            void draw_qrects(const std::vector<QRectF> &rects, QGraphicsScene *scene);

            Match project_corners(const Corners &corners, long lidar_timestamp);
            double compute_total_error(const Match &matches);
            Eigen::Affine2d update_robot_pose(const Eigen::Affine2d &pose);
            bool check_viability();
            void remove_thing_draw(QGraphicsScene *scene);
            void set_thing_opacity(float opacity);
            void define_actionable_color(const QColor &color);
            void set_error(double error);
            std::vector<Eigen::Vector3d> get_corners_3d() const;
            Eigen::Affine2d get_robot_pose() const;
            QGraphicsItem * get_thing_draw() const;
            QColor get_actionable_color() const;
            double get_error() const;
            double get_buffer_error() const;
            double get_prediction_fitness(double eps = 0.001) const;
            void set_energy(double energy_);
            double geAlignedBox2dt_energy() const;
            bool operator==(const ActionableThing &other) const;

            static gtsam::Vector2 distance(const gtsam::Vector5 &b, gtsam::OptionalJacobian<1, 5> H)
            {
                return gtsam::Vector2{b(3), b(4)};
            }

        private:
            Eigen::Affine2d robot_pose;
            QGraphicsPolygonItem *thing_draw = nullptr;
            Eigen::AlignedBox2d box;
            Eigen::Vector3d target;
            boost::circular_buffer<double> error_buffer;
            bool initialized = false;
            QColor color;
            double error = 0.0f;
            double energy = 0.0;
            //Create std::set of residuals with a custom comparator
            struct CompareEigenVector3d
            {
                bool operator()(const Eigen::Vector3d &lhs, const Eigen::Vector3d &rhs) const
                {
                    //Return norm of the difference gretter than 1e-3
                    return (lhs - rhs).norm() > 1e-3;
                }
            };
std::set<Eigen::Vector3d, CompareEigenVector3d> residuals_set;

    };
} // rc

#endif //ACTIONABLE_ROOM_H
