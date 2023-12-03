//
// Created by pbustos on 1/12/22.
//

#include "door_detector.h"
#include <cppitertools/range.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/combinations.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/slice.hpp>

DoorDetector::DoorDetector()
{
    // Define the value of each range for all levels of lidar lines extracted from the pointcloud
    height_ranges = {{0, 350}};
    //    height_ranges = {{0, 350}, {1000, 2000}, {2000, 3000}, {3000, 4000}};
}
DoorDetector::Doors
DoorDetector::detect(const RoboCompLidar3D::TPoints &points, QGraphicsScene *scene)
{
    auto lines = extract_lines(points, height_ranges);
    auto peaks = extract_peaks(lines);
    auto doors = get_doors(peaks);
    auto final_doors = filter_doors(doors);

    draw_doors(final_doors, Door(), scene);
    return final_doors;
}

DoorDetector::Lines DoorDetector::extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges)
{
    Lines lines(ranges.size());
    for(const auto &p: points)
    {
        for(const auto &[i, r] : ranges | iter::enumerate)
            if(p.z > r.first and p.z < r.second)
                lines[i].emplace_back(p.x, p.y);
    }
    return lines;
}
DoorDetector::Lines DoorDetector::extract_peaks(const DoorDetector::Lines &lines)
{
    Lines peaks(lines.size());
    const float THRES_PEAK = 1000;

    for(const auto &[i, line] : lines | iter::enumerate)
        for (const auto &both: iter::sliding_window(line, 2))
            if (fabs(both[1].norm() - both[0].norm()) > THRES_PEAK)
            {
                if (both[0].norm() < both[1].norm()) peaks[i].push_back(both[0]);
                else peaks[i].push_back(both[1]);
            }

    return peaks;
}
std::vector<DoorDetector::Doors> DoorDetector::get_doors(const DoorDetector::Lines &peaks)
{
    std::vector<Doors> doors_list(peaks.size());
    const float THRES_DOOR = 500;
    const float MAX_DOOR_WIDTH = 1400;
    const float MIN_DOOR_WIDTH = 500;
    auto near_door = [THRES_DOOR](auto &doors_list, auto d)
    {
        return std::ranges::any_of(doors_list, [d, THRES_DOOR](auto &old)
            {return (old.p0-d.p0).norm() < THRES_DOOR or
                    (old.p1-d.p1).norm() < THRES_DOOR or
                    (old.p1-d.p0).norm() < THRES_DOOR or
                    (old.p0-d.p1).norm() < THRES_DOOR;});
    };

    for(const auto &[i, peak] : peaks | iter::enumerate)
        for(auto &par : peak | iter::combinations(2))
            if((par[0]-par[1]).norm() < MAX_DOOR_WIDTH and (par[0]-par[1]).norm() > MIN_DOOR_WIDTH)
            {
                auto door = Door{par[0], par[1]};
                if(not near_door(doors_list[i], door))
                    doors_list[i].emplace_back(par[0], par[1]);
            }
    return doors_list;
}
DoorDetector::Doors
DoorDetector::filter_doors(const std::vector<Doors> &doors_list)
{
    Doors final_doors;
    auto lowest_doors = doors_list[0];
    for(const auto &dl: lowest_doors)
    {
        bool match = true;
        for(const auto &doors: iter::slice(doors_list, 1, (int)doors_list.size(), 1))  // start from second element
            match = match and std::ranges::find(doors, dl) != doors.end();

        if (match)
            final_doors.push_back(dl);
    }
    return final_doors;
}

void DoorDetector::draw_doors(const Doors &doors, const Door &door_target, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> borrar;
    for (auto &b: borrar)
    {
        scene->removeItem(b);
        delete b;
    }
    borrar.clear();

    QColor target_color;
    for (const auto &d: doors)
    {
        if(d == door_target)
        {
            target_color = QColor("magenta");
            auto middle = scene->addRect(-100, -100, 200, 200, QColor("blue"), QBrush(QColor("blue")));
            auto perp = door_target.point_perpendicular_to_door_at();
            middle->setPos(perp.x(), perp.y());
            borrar.push_back(middle);
            auto middle_line = scene->addLine(perp.x(), perp.y(), d.middle.x(), d.middle.y(), QPen(QColor("blue"), 20));
            borrar.push_back(middle_line);
        }
        else
            target_color = color;
        auto point = scene->addRect(-50, -50, 100, 100, QPen(target_color), QBrush(target_color));
        point->setPos(d.p0.x(), d.p0.y());
        borrar.push_back(point);
        point = scene->addRect(-50, -50, 100, 100, QPen(target_color), QBrush(target_color));
        point->setPos(d.p1.x(), d.p1.y());
        borrar.push_back(point);
        auto line = scene->addLine(d.p0.x(), d.p0.y(), d.p1.x(), d.p1.y(), QPen(target_color, 50));
        borrar.push_back(line);
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
            if (auto res = r_to_p.intersects(door_line, &point); res == QLineF::BoundedIntersection)
                inside_points[i] = Eigen::Vector2f(point.x(), point.y());
        }
    }
    //qInfo() << __FUNCTION__ << "After" << inside_points.size();
    return inside_points;
}