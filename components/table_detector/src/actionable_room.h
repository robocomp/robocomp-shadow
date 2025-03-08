//
// Created by pbustos on 9/12/24.
//

#ifndef ACTIONABLE_ROOM_H
#define ACTIONABLE_ROOM_H

#include <vector>
#include "room.h"
#include "common_types.h"
#include "dbscan.h"
#include "actionable_origin.h"

namespace rc
{
    class ActionableRoom
    {
        public:
            ActionableRoom();

            bool initialize(const rc::ActionableRoom &progenitor,
                            const std::vector<ActionableRoom> &actionables,
                            QGraphicsPolygonItem *robot_draw_);
            bool initialize(const ActionableOrigin &progenitor,
                            const std::vector<ActionableRoom> &actionables,
                            QGraphicsPolygonItem *robot_draw_);
            Match project_corners(const Corners &corners, long lidar_timestamp);
            double compute_total_error(const Match &matches);
            Eigen::Affine2d update_robot_pose(const Eigen::Affine2d &pose);
            Eigen::Vector3d update_target(const LidarPoints &points);
            bool check_viability();
            void remove_robot_draw(QGraphicsScene *scene);
            void set_robot_opacity(float opacity);
            void define_actionable_color(const QColor &color);
            void set_error(double error);
            std::vector<Eigen::Vector3d> get_corners_3d() const;
            Eigen::Affine2d get_robot_pose() const;
            Eigen::Affine3d get_robot_pose_3d() const;
            Eigen::Affine2d get_robot_pose_in_meters() const;
            QGraphicsItem *get_robot_draw() const;
            QGraphicsItem *get_room_draw() const;
            QColor get_actionable_color() const;
            double get_error() const;
            double get_buffer_error() const;

            void compute_prediction_fitness(double eps = 0.001);
            rc::Room get_room() const;
            void set_energy(double energy_);
            double get_energy() const;
            double get_value() const;
            boost::circular_buffer<Corner> get_corner_buffer() const;
            std::vector<Eigen::Vector3d> get_corners_by_timestamp(long timestamp) const;
            bool operator==(const ActionableRoom &other) const;

            rc::Room room;

        private:
            Eigen::Affine2d robot_pose;
            Eigen::Affine3d robot_pose_3d;
            QGraphicsPolygonItem *robot_draw = nullptr;
            QGraphicsPolygonItem *room_draw = nullptr;
            Eigen::Vector3d target;
            boost::circular_buffer<Corner> corner_buffer;
            boost::circular_buffer<Corner> corner_robot_buffer;
            boost::circular_buffer<double> error_buffer;
            bool initialized = false;
            QColor color;
            double error = 0.0;
            double energy = 0.0;
            double value = 0.0;
    };
} // rc

#endif //ACTIONABLE_ROOM_H
