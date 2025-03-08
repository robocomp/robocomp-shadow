//
// Created by pbustos on 9/12/24.
//

#ifndef ACTIONABLE_ORIGIN_H
#define ACTIONABLE_ORIGIN_H

#include <vector>
#include "room.h"
#include "common_types.h"

namespace rc
{
    class ActionableOrigin
    {
        public:
            ActionableOrigin();
            void initialize(const rc::ActionableOrigin &parent, const std::tuple<QGraphicsPolygonItem*, QGraphicsEllipseItem*> &robot_drawing);
            Match project_corners(const Corners &corners, long lidar_timestamp);
            Eigen::Affine2d update_robot_pose(const Eigen::Affine2d &pose);
            Eigen::Vector3d update_target(const LidarPoints &points);
            bool ready_to_procreate() const;
            bool check_viability();
            void remove_robot_draw(QGraphicsScene *scene);
            void set_robot_opacity(float opacity);
            void define_actionable_color(const QColor &color);
            void set_error(double error);
            std::vector<Eigen::Vector3d> get_corners_3d() const;
            Eigen::Affine2d get_robot_pose() const;
            QGraphicsItem *get_robot_draw() const;
            QColor get_actionable_color() const;
            double get_error() const;
            double get_buffer_error() const;
            double get_normalized_error(double eps = 0.001) const;
            rc::Room get_room() const;
            void set_energy(double energy_);
            double get_energy() const;
            boost::circular_buffer<Corner> get_corner_buffer() const;
            boost::circular_buffer<Corner> get_corner_robot_buffer() const;
            bool operator ==(const ActionableOrigin &other) const;

        private:
            rc::Room room;
            Eigen::Affine2d robot_pose;
            QGraphicsPolygonItem *robot_draw = nullptr;
            QGraphicsEllipseItem *white_circle = nullptr;
            Eigen::Vector3d target;
            boost::circular_buffer<Corner> corner_buffer;
            boost::circular_buffer<Corner> corner_robot_buffer;
            boost::circular_buffer<double> error_buffer;
            bool reproduce = false;
            bool initialized = false;
            QColor color;
            double error = 0.0f;
            double energy = 0.0;
    };
} // rc

#endif //ACTIONABLE_ORIGIN_H
