//
// Created by pbustos on 9/10/22.
//

#ifndef ROOM_H
#define ROOM_H

#include <QtCore>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QPolygonF>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "hungarian.h"

namespace rc
{
    class Room
    {
        public:
            std::pair<cv::RotatedRect, Eigen::Affine2d> initialize(const std::vector<QPointF>& corners,
                                                                   const Eigen::Affine2d& robot_pose);
            bool operator==(const Room &r) const
                { return (center-r.center).norm() < 100.f and fabs(rot-r.rot) < 1.f and
                          fabs(rsize.height() - r.rsize.height()) < 100.f and
                          fabs(rsize.width() - r.rsize.width()) < 100.f; };
            double keep_angle_between_minus_pi_and_pi(double angle);
            double get_width() const;

            double get_width_meters() const;

            double get_depth() const;

            double get_depth_meters() const;

            double get_height() const;
            double get_largest_dim() const;
            double get_smallest_dim() const;
            Eigen::Vector2d get_center() const;
            double get_center_x() const;
            double get_center_y() const;
            double get_rotation() const;
            Eigen::Vector2d get_closest_corner(const Eigen::Vector2d &c);
            Eigen::Vector2d get_closest_corner_in_robot_coor(const Eigen::Vector2d &c);
            Eigen::Vector2d get_closest_corner_in_robot_coor2(const Eigen::Vector2d &c) const;
            std::pair<double, QLineF> get_closest_side(const QLineF &line);
            std::vector<QLineF> get_room_lines_qt() const;
            std::vector<Eigen::ParametrizedLine<double, 2>> get_room_lines_eigen() const;
            QPolygonF get_qt_polygon() const;
            Eigen::Vector2d to_room_coor(const Eigen::Vector2d &p) const;
            Eigen::Matrix<double, 4, 2> get_corners_mat() const;
            std::vector<Eigen::Vector2d> get_corners() const;
            //std::vector<Eigen::Vector3d> get_3d_corners_in_robot_coor();
            std::vector<Eigen::Vector3d> get_corners_3d() const;            // with projective coordinates
            Eigen::Vector2d to_local_coor(const Eigen::Vector2d &p);
            void rotate(double delta);  //degrees
            void print();
            void draw_2D(const QString &color, QGraphicsScene *scene) const;
            bool is_valid() const { return is_initialized; };
            void set_valid(const bool v) { is_initialized = v; };
            double get_minX() const;           // returns the minimum x value of the room
            double get_minY() const;
            double get_maxX() const;
            double get_maxY() const;
            std::vector<QPolygonF> get_walls_as_polygons(const std::vector<QPolygonF> &obstacles, double robot_width) const;

    private:
            cv::RotatedRect rect;
            Eigen::Vector2d center = {0.0, 0.0};    // mm
            double rot = 0.0; // radians
            QSizeF rsize = {0.0, 0.0}; // mm
            bool is_initialized = false;
            [[nodiscard]] double size_dist(const QSizeF &p1, const QSizeF &p2) const;
            double euc_distance_between_points(const QPointF &p1, const QPointF &p2) const;
            double euc_distance_between_points(const cv::Point2d &p1, const QPointF &p2) const;
            int csign = 1;
            double delta = 0.1;
            QSizeF tmp_size = {0.0, 0.0};
            double size_confidence = -10.0;
            void compute_corners();
            QPointF to_qpoint(const cv::Point2d &p) const;

    };
} //rc
#endif //ROOM_H
