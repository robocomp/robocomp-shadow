//
// Created by pbustos on 9/10/22.
//

#include "room.h"

float Room::size_dist(const QSizeF &p1, const QSizeF &p2) const
{
    return sqrt((p1.width()-p2.width())*(p1.width()-p2.width())+(p1.height()-p2.height())*(p1.height()-p2.height()));
}
void Room::update(const QSizeF &new_size, const Eigen::Vector2f &center_, float rot_)
{
    center = center_;
    rot = rot_;
    if(size_dist(new_size, tmp_size) < 500)
    {
        size_confidence = std::clamp(size_confidence+csign*delta, -10.f, 10.f);
    }
    else
    {
        tmp_size = new_size;
        csign = -csign;
        size_confidence = std::clamp(size_confidence+csign*delta, -10.f, 10.f);
    }
    double res = 1.0/(1.0 + exp(-size_confidence));
    if(res > 0.9 or res < 0.1)
    {
        rsize = tmp_size;
    }
    //qInfo() << __FUNCTION__ << size_confidence << res << tmp_size << size;
}
Eigen::Matrix<float, 4, 2> Room::get_corners_in_robot_coor()
{
    Eigen::Matrix<float, 4, 2> corners;
    Eigen::Matrix<float, 2, 2> rt_from_robot;
    rt_from_robot << cos(rot), -sin(rot) , sin(rot), cos(rot);

    corners << rt_from_robot * Eigen::Vector2f(rsize.width()/2, rsize.height()/2) + center,
               rt_from_robot * Eigen::Vector2f(rsize.width()/2, -rsize.height()/2) + center,
               rt_from_robot * Eigen::Vector2f(-rsize.width()/2, rsize.height()/2) + center,
               rt_from_robot * Eigen::Vector2f(-rsize.width()/2, -rsize.height()/2) + center;
    return corners;
}
Eigen::Matrix<float, 4, 2> Room::get_corners()
{
    Eigen::Matrix<float, 4, 2> corners;
    float sw = rsize.width()/2.f;
    float sh = rsize.height()/2.f;
    corners << Eigen::Vector2f(sw, sh),
               Eigen::Vector2f(sw, -sh),
               Eigen::Vector2f(-sw, sh),
               Eigen::Vector2f(-sw, -sh);
    return corners;
}
Eigen::Vector2f Room::get_closest_corner(const Eigen::Vector2f &c)
{
    auto corners = get_corners_in_robot_coor();
    Eigen::Index index;
    (corners.rowwise() - c.transpose()).rowwise().squaredNorm().minCoeff(&index);
    //std::cout << "min value at " << index << std::endl;
    return corners.row(index);
}
void Room::print()
{
    std::cout << "Room: " << std::endl;
    std::cout << "  size: [" << rsize.width() << ", " << rsize.height() << "]" << std::endl;;
    std::cout << "  center: [" << center.x() << ", " << center.y() << "]" << std::endl;;
    std::cout << "  rot: " << rot << std::endl;
    auto rcorners = get_corners_in_robot_coor();
    std::cout << "  corners: [" << rcorners << "]" << std::endl;
    auto corners = get_corners();
    std::cout << "  corners: [" << corners << "]" << std::endl;
}