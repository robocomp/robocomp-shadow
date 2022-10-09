//
// Created by pbustos on 9/10/22.
//

#ifndef LOCAL_GRID_ROOM_H
#define LOCAL_GRID_ROOM_H

#include <QtCore>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class Room
{
    public:
        void update(const QSizeF &new_size, const Eigen::Vector2f &center_, float rot_);
        Eigen::Vector2f get_closest_corner(const Eigen::Vector2f &c);
        void print();
        Eigen::Vector2f center = {0.f, 0.f};
        float rot = 0.f;
        QSizeF rsize = {0.f, 0.f};

    private:
        float size_dist(const QSizeF &p1, const QSizeF &p2) const;
        int csign = 1;
        float delta = 0.1;
        QSizeF tmp_size = {0.f, 0.f};
        float size_confidence = -10.f;
        void compute_corners();
        Eigen::Matrix<float,4,2> get_corners_in_robot_coor();

    Eigen::Matrix<float, 4, 2> get_corners();
};

#endif //LOCAL_GRID_ROOM_H
