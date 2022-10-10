//
// Created by pbustos on 9/10/22.
//

#include "room.h"
#include <cppitertools/range.hpp>

float Room::size_dist(const QSizeF &p1, const QSizeF &p2) const
{
    return sqrt((p1.width()-p2.width())*(p1.width()-p2.width())+(p1.height()-p2.height())*(p1.height()-p2.height()));
}
void Room::update(const QSizeF &new_size, const Eigen::Vector2f &center_, float rot_)
{
    center = center_;
    rot = rot_;
    if(size_dist(new_size, tmp_size) < 500)
        size_confidence = std::clamp(size_confidence+csign*delta, -10.f, 10.f);
    else
    {
        tmp_size = new_size;
        csign = -csign;
        size_confidence = std::clamp(size_confidence+csign*delta, -10.f, 10.f);
    }
    double res = 1.0/(1.0 + exp(-size_confidence));
    if(res > 0.9 or res < 0.1)
        rsize = tmp_size;
}
void Room::update(const QSizeF &new_size, const Eigen::Vector2f &center_)
{
    center = center_;
    if(size_dist(new_size, tmp_size) < 500)
        size_confidence = std::clamp(size_confidence+csign*delta, -10.f, 10.f);
    else
    {
        tmp_size = new_size;
        csign = -csign;
        size_confidence = std::clamp(size_confidence+csign*delta, -10.f, 10.f);
    }
    double res = 1.0/(1.0 + exp(-size_confidence));
    if(res > 0.9 or res < 0.1)
        rsize = tmp_size;
}
Eigen::Matrix<float, 4, 2> Room::get_corners_in_robot_coor()
{
    //Eigen::Matrix<float, 4, 2> corners = get_corners();
    Eigen::Matrix<float, 2, 2> rt_from_robot;
    float rrot = qDegreesToRadians(rot);
    rt_from_robot << cos(rrot), -sin(rrot) , sin(rrot), cos(rrot);
    Eigen::Matrix<float, 4, 2> rcorners = ((rt_from_robot * get_corners().transpose()).colwise() + center).transpose();
    return rcorners;
}
Eigen::Matrix<float, 4, 2> Room::get_corners()
{
    Eigen::Matrix<float, 4, 2> corners;
    float sw = rsize.width()/2.f;
    float sh = rsize.height()/2.f;
    corners.row(0) = Eigen::Vector2f(sw, sh);
    corners.row(1) = Eigen::Vector2f(sw, -sh);
    corners.row(2) = Eigen::Vector2f(-sw, sh);
    corners.row(3) = Eigen::Vector2f(-sw, -sh);
    return corners;
}
Eigen::Vector2f Room::get_closest_corner(const Eigen::Vector2f &c)
{
    auto corners = get_corners_in_robot_coor();
    Eigen::Index index;
    (corners.rowwise() - c.transpose()).rowwise().squaredNorm().minCoeff(&index);
    return corners.row(index);
}
QLineF Room::get_closest_side(const QLineF &l)
{
//    auto sides = get_sides_in_robot_coor();
//
//    return QLineF();
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

void Room::draw_on_2D_tab(QGraphicsScene *scene, const std::vector<std::tuple<QPointF, QPointF, int>> &points)
{
    static std::vector<QGraphicsItem*> items;
    for(const auto &g: items)
        scene->removeItem(g);
    items.clear();

    QColor col("yellow");
    // draw model
    col.setAlpha(30);
    auto size = rsize;
    auto item = scene->addRect(-size.height()/2, -size.width()/2, size.height(), size.width(), QPen(QColor(col), 60), QBrush(QColor(col)));
    item->setPos(center.x(), center.y());
    item->setRotation(-rot);
    item->setZValue(1);
    items.push_back(item);

    // draw connected corners
    for(const auto &[p1, p2, signo] : points)
    {
        //auto closest = get_closest_corner(Eigen::Vector2f(c.p1().x(), c.p1().y()));
        if(signo > 0)  col = QColor("red"); else col = QColor("blue");
        items.push_back(scene->addLine(p1.x(), p1.y(), p2.x(), p2.y(), QPen(col, 40)));
    }
}
Eigen::Vector2f Room::to_local_coor(const Eigen::Vector2f &p)
{
    Eigen::Matrix<float, 2, 2> rt_from_robot;
    rt_from_robot << cos(rot), -sin(rot) , sin(rot), cos(rot);
    return rt_from_robot.transpose()*(p - center);
}
void Room::rotate(float delta)
{
    delta = std::clamp(delta, -5.f, 5.f);
    rot += delta;
    qInfo() << __FUNCTION__ << delta << rot;
}

