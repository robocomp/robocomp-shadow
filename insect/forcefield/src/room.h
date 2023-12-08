//
// Created by pbustos on 9/10/22.
//

#ifndef LOCAL_GRID_ROOM_H
#define LOCAL_GRID_ROOM_H

#include <QtCore>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace rc
{
    class Room
    {
    public:
//        void update(const QSizeF &new_size, const Eigen::Vector2f &center_, float rot_);
//        void update(const QSizeF &new_size, const Eigen::Vector2f &center_);
//        void update(const QPointF &p1, const QPointF &p2, const QPointF &p3);
        Eigen::Vector2f get_closest_corner(const Eigen::Vector2f &c);
        Eigen::Vector2f get_closest_corner_in_robot_coor(const Eigen::Vector2f &c);
        Eigen::Matrix<float, 4, 2> get_eigen_corners_in_robot_coor2();
        Eigen::Vector2f get_closest_corner_in_robot_coor2(const Eigen::Vector2f &c) const;
        std::pair<float, QLineF> get_closest_side(const QLineF &line);
        std::vector<QLineF> get_room_lines_qt() const;
        Eigen::Vector2f to_room_coor(const Eigen::Vector2f &p) const;
        Eigen::Matrix<float, 4, 2> get_corners();
        std::vector<Eigen::Vector3f> get_3d_corners_in_robot_coor();
        Eigen::Matrix<float, 4, 2> get_eigen_corners_in_robot_coor();
        Eigen::Vector2f to_local_coor(const Eigen::Vector2f &p);
        void rotate(float delta);  //degrees
        void print();
        void draw_on_2D_tab(const Room &room, const QString &color, AbstractGraphicViewer *viewer);

        cv::RotatedRect rect;
        Eigen::Vector2f center = {0.f, 0.f};    // mm
        float rot = 0.f; // radians
        QSizeF rsize = {0.f, 0.f}; // mm
        bool is_initialized = false;

    private:
        [[nodiscard]] float size_dist(const QSizeF &p1, const QSizeF &p2) const;
        float euc_distance_between_points(const QPointF &p1, const QPointF &p2) const;

        int csign = 1;
        float delta = 0.1;
        QSizeF tmp_size = {0.f, 0.f};
        float size_confidence = -10.f;
        void compute_corners();
        QPointF to_qpoint(const cv::Point2f &p) const;
        float euc_distance_between_points(const cv::Point2f &p1, const QPointF &p2) const;
    };
} //rc
#endif //LOCAL_GRID_ROOM_H
