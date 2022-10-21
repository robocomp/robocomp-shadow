//
// Created by pbustos on 9/10/22.
//

#ifndef LOCAL_GRID_ROOM_H
#define LOCAL_GRID_ROOM_H

#include <QtCore>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>


class Room
{
    public:
        void update(const QSizeF &new_size, const Eigen::Vector2f &center_, float rot_);
        void update(const QSizeF &new_size, const Eigen::Vector2f &center_);
        Eigen::Vector2f get_closest_corner(const Eigen::Vector2f &c);
        QLineF get_closest_side(const QLineF &line);

        Eigen::Matrix<float, 4, 2> get_corners();
        void draw_on_2D_tab(QGraphicsScene *scene, const std::vector<std::tuple<QPointF, QPointF, int>> &points);
        Eigen::Vector2f to_local_coor(const Eigen::Vector2f &p);
        void rotate(float delta);  //degrees
        void print();

        Eigen::Vector2f center = {0.f, 0.f};
        float rot = 0.f;
        QSizeF rsize = {0.f, 0.f};

    private:
        [[nodiscard]] float size_dist(const QSizeF &p1, const QSizeF &p2) const;
        int csign = 1;
        float delta = 0.1;
        QSizeF tmp_size = {0.f, 0.f};
        float size_confidence = -10.f;
        void compute_corners();
        Eigen::Matrix<float,4,2> get_corners_in_robot_coor();

};

#endif //LOCAL_GRID_ROOM_H
