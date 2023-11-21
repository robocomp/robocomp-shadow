//
// Created by pbustos on 1/12/22.
//

#include "door_detector.h"
#include <cppitertools/range.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/combinations.hpp>
#include <cppitertools/enumerate.hpp>

std::vector<DoorDetector::Door> DoorDetector::detect(const std::vector<std::vector<Eigen::Vector2f>> &lines, AbstractGraphicViewer *viewer)
{
    // lambda to check that all points between door edges are further that the edges themselves
    auto distances_between_points_greater_than_distances_to_points = [](auto &p0, auto &p1, auto &line)
            { for(auto &&i: iter::range(std::max(p0,p1), std::min(p0,p1)))
                if(line[i].norm() > line[p0].norm() or line[i].norm() > line[p1].norm())
                    return false;
              return true;
            };

    std::vector<std::vector<Door>> doors(n_lines);
    for(auto &&i: iter::range(1, n_lines))
    {
        const auto &line = lines[i];
        std::vector<float> derivatives(line.size() - 1);
        for (auto &&[i, p]: line | iter::sliding_window(2) | iter::enumerate)
            derivatives[i] = p[1].norm() - p[0].norm();

        std::vector<std::tuple<int, bool>> peaks;
        for (auto &&[i, d]: derivatives | iter::enumerate)
            if (d > der_threshold) peaks.push_back(std::make_tuple(i, true));
            else if (d < -der_threshold) peaks.push_back(std::make_tuple(i + 1, false));

        for (auto &&p: peaks | iter::combinations(2))  //QUITAR
        {
            auto &[p0, p0pos] = p[0]; auto &[p1, p1pos] = p[1];
            auto v0 = line[p0]; auto v1 = line[p1];
            if (((not p0pos and p1pos) or (p0pos and not p1pos))  // alternating signs
                and (v0 - v1).norm() < max_door_width             // width contraint
                and (v0 - v1).norm() > min_door_width
                and distances_between_points_greater_than_distances_to_points(p0, p1, line)
                and v0.y()>200 and v1.y()>200) // to remove spureous peaks
            {
                Door door;
                door.center_floor = (v0 + v1) / 2.f;
                door.points = {Eigen::Vector3f(v0.x(), v0.y(), 0.f), Eigen::Vector3f(v1.x(), v1.y(), 0.f),
                               Eigen::Vector3f(v1.x(), v1.y(), door.height), Eigen::Vector3f(v0.x(), v0.y(), door.height)};
                door.p0 = v0;door.p1 = v1;
                door.idx0 = p0; door.idx1 = p1;
                if (i == 1)
                    doors[i].emplace_back(std::move(door));
                else    //if there is also a door in the previous level
                if(auto it=std::ranges::find_if(doors[i-1], [door, max_sep = max_door_separation](auto &a)
                                {return (door.center_floor-a.center_floor).norm() < max_sep;}); it != doors[i-1].end())
                    doors[i].emplace_back(std::move(door));
            }
        }
        draw_peaks(peaks, line, viewer);
    }
    draw_doors(doors[n_lines-1], viewer);
    return doors[n_lines-1];
}

void DoorDetector::draw_peaks(const std::vector<std::tuple<int, bool>> &peaks, const std::vector<Eigen::Vector2f> &line, AbstractGraphicViewer *viewer)
{
    if(viewer==nullptr) return;
    static std::vector<QGraphicsItem *> items;
    for(const auto &item: items)
        viewer->scene.removeItem(item);
    items.clear();
    QColor color;
    for(auto &[d, pos]: peaks)
    {
        if(pos)
            color = QColor("yellow");
        else
            color = QColor("green");
        auto item = viewer->scene.addEllipse(-80, -80, 160, 160, color, QBrush(color));
        item->setPos(line[d].x(), line[d].y());
        items.push_back(item);
    }
}
void DoorDetector::draw_doors(const std::vector<Door> &doors, AbstractGraphicViewer *viewer)    //one vector for each height level
{
    if (viewer == nullptr) return;
    static std::vector<QGraphicsItem *> items;
    for (const auto &item: items)
        viewer->scene.removeItem(item);
    items.clear();

    for (auto &&d: doors)
    {
        auto item = viewer->scene.addEllipse(-80, -80, 160, 160, QColor("magenta"), QBrush(QColor("magenta")));
        item->setPos(d.center_floor.x(), d.center_floor.y());
        items.push_back(item);
    }
}
std::vector<Eigen::Vector2f> DoorDetector::filter_out_points_beyond_doors(const std::vector<Eigen::Vector2f> &floor_line_cart, const std::vector<DoorDetector::Door> &doors)
{
    std::vector<Eigen::Vector2f> inside_points(floor_line_cart);
    std::vector<std::pair<u_long, Eigen::Vector2f>> ignore;
    //qInfo() << __FUNCTION__ << "Before" << inside_points.size();
    for (const auto &door: doors)       // all in robot's reference system
    {
        QLineF door_line(door.p0.x(), door.p0.y(), door.p1.x(), door.p1.y());
        for (auto &&i: iter::range(std::min(door.idx0, door.idx1), std::max(door.idx0, door.idx1)))
        {
            QLineF r_to_p(0.f, 0.f, floor_line_cart[i].x(), floor_line_cart[i].y());
            QPointF point;
            if (auto res = r_to_p.intersect(door_line, &point); res == QLineF::BoundedIntersection)
                inside_points[i] = Eigen::Vector2f(point.x(), point.y());
        }
    }
    //qInfo() << __FUNCTION__ << "After" << inside_points.size();
    return inside_points;
}