//
// Created by pbustos on 11/11/25.
//

#include "door_detector.h"

#include <expected>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/combinations.hpp>
#include <QGraphicsItem>


Doors DoorDetector::detect(const RoboCompLidar3D::TPoints &points,
                           const Eigen::Affine2d &robot_pose,
                           bool localised,
                           QGraphicsScene *robot_scene,
                           QGraphicsScene *scene_room)
{
    if(points.empty()) return {};

    // get the peaks
    Peaks peaks;
    for (const auto &p : iter::sliding_window(points, 2))
    {
        const auto &p1 = p[0]; const auto &p2 = p[1];
        const float d1 = p1.distance2d; const float d2 = p2.distance2d;
        if (const float dd_da1 = abs(d2 - d1); dd_da1 > 1000.f)
        {
            const auto m = std::ranges::min_element(p, [](auto &pa, auto &pb){return pa.distance2d < pb.distance2d;});
            peaks.emplace_back(Eigen::Vector2f(m->x, m->y), m->phi);
        }
    }

    // non-maximum suppression of peaks: remove peaks closer than 500mm
    Peaks nms_peaks;
    for (const auto &[p, a] : peaks)
    {
        const bool too_close = std::ranges::any_of(nms_peaks, [&p](const auto &p2) { return (p - std::get<0>(p2)).norm() < 500.f; });
        if (not too_close)
            nms_peaks.emplace_back(p, a);
    }
    peaks = nms_peaks;

    // find doors as pairs of peaks separated by a gap < 1200mm and > 800mm
    Doors doors;
    for(const auto &p : peaks | iter::combinations(2))
    {
        const auto &[p0,a0] = p[0]; const auto &[p1, a1] = p[1];
        const float gap = (p1-p0).norm();
        //qInfo() << "Gap: " << gap;
        if(gap < 1300.f and gap > 800.f)
            doors.emplace_back(p0, a0, p1, a1);
    }
    //qInfo() << __FUNCTION__ << "Peaks found: " << peaks.size() << "Doors found: " << doors.size();

    doors_cache = doors;

    return doors;
}

// Method to use the Doors vector to filter out the LiDAR points that como from a room outside the current one
RoboCompLidar3D::TPoints DoorDetector::filter_points(const RoboCompLidar3D::TPoints &points)
{
    const auto doors = detect(points);
    if(doors.empty()) return points;

    // for each door, check if the distance from the robot to each lidar point is smaller than the distance from the robot to the door
    RoboCompLidar3D::TPoints filtered;
    for(const auto &d : doors)
    {
        const float dist_to_door = d.center().norm();
        // Check if the angular range wraps around the -π/+π boundary
        const bool angle_wraps = d.p2_angle < d.p1_angle;
        for(const auto &p : points)
        {
            // Determine if point is within the door's angular range
            bool point_in_angular_range;
            if (angle_wraps)
            {
                // If the range wraps around, point is in range if it's > p1_angle OR < p2_angle
                point_in_angular_range = (p.phi > d.p1_angle) or (p.phi < d.p2_angle);
            }
            else
            {
                // Normal case: point is in range if it's between p1_angle and p2_angle
                point_in_angular_range = (p.phi > d.p1_angle) and (p.phi < d.p2_angle);
            }

            // Filter out points that are through the door (in angular range and farther than door)
            if(point_in_angular_range and p.distance2d >= dist_to_door)
                continue;

            //qInfo() << __FUNCTION__ << "Point angle: " << p.phi << " Door angles: " << d.p1_angle << ", " << d.p2_angle << " Point distance: " << p.distance2d << " Door distance: " << dist_to_door;
            filtered.emplace_back(p);
        }
    }
    return filtered;
}

std::expected<Door, std::string> DoorDetector::get_current_door() const
{
    if (doors_cache.empty())
        return std::unexpected<std::string>{"No doors detected"};
    return doors_cache[0];
}

void DoorDetector::draw_doors(bool localised, QGraphicsScene *scene_robot, QGraphicsScene *scene_room, const Eigen::Affine2d &robot_pose)
{
    // draw doors in robot viewer
    static std::vector<QGraphicsItem *> doors_draw;
    if(scene_robot)
    {
        for (const auto &p : doors_draw)
        { scene_robot->removeItem(p); delete p;}
        doors_draw.clear();
        const QColor color("blue");
        const QPen pen(color);
        const QBrush brush(color);
        for (const auto &d : doors_cache)
        {
            const auto peak_A_draw = scene_robot->addEllipse(-100, -100, 200, 200, pen, brush);
            const auto peak_B_draw = scene_robot->addEllipse(-100, -100, 200, 200, pen, brush);
            peak_A_draw->setPos(d.p1.x(), d.p1.y());
            peak_B_draw->setPos(d.p2.x(), d.p2.y());
            const auto line_draw = scene_robot->addLine(d.p1.x(), d.p1.y(), d.p2.x(), d.p2.y(), QPen(Qt::cyan, 30));
            doors_draw.push_back(peak_A_draw);
            doors_draw.push_back(peak_B_draw);
            doors_draw.push_back(line_draw);
        }
    }
    // if robot is localised, draw doors in room viewer
    static std::vector<QGraphicsItem *> doors_draw_room;
    if(localised and scene_room)
    {
        for (const auto &p : doors_draw_room)
        { scene_room->removeItem(p); delete p;}
        doors_draw_room.clear();

        const QColor color("blue");
        const QPen pen(color);
        for (const auto &d : doors_cache)
        {
            // transform door points to room frame
            Eigen::Vector2d p1_room = robot_pose * d.p1.cast<double>();
            Eigen::Vector2d p2_room = robot_pose * d.p2.cast<double>();
            const auto peak_A_draw = scene_room->addEllipse(-100, -100, 200, 200, pen, QBrush(color));
            const auto peak_B_draw = scene_room->addEllipse(-100, -100, 200, 200, pen, QBrush(color));
            peak_A_draw->setPos(p1_room.x(), p1_room.y());
            peak_B_draw->setPos(p2_room.x(), p2_room.y());
            doors_draw.push_back(peak_A_draw);
            doors_draw.push_back(peak_B_draw);
            const auto line_draw = scene_room->addLine(p1_room.x(), p1_room.y(), p2_room.x(), p2_room.y(), pen);
            doors_draw.push_back(line_draw);
        }
    }
}