//
// Created by pbustos on 1/12/22.
//

#ifndef FORCEFIELD_DOOR_DETECTOR_H
#define FORCEFIELD_DOOR_DETECTOR_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <QGraphicsItem>
#include <QColor>
#include <QtCore>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <opencv2/core.hpp>
#include <Lidar3D.h>

class DoorDetector
{
public:
    DoorDetector();
    struct Door
    {
        Eigen::Vector2f p0, p1, middle, middle_measured, p0_measured, p1_measured;
        int idx_in_peaks_0, idx_in_peaks_1, id, wall_id;
        float angle;
        const float THRESHOLD = 500; //door equality

        Door(){ p0 = p1 = middle = Eigen::Vector2f::Zero();};
        Door(const Eigen::Vector2f &p0_,
             const Eigen::Vector2f &p1_,
             int idx_0, int idx_1) : p0(p0_), p1(p1_), idx_in_peaks_0(idx_0), idx_in_peaks_1(idx_1)
        {
            middle = (p0 + p1)/2.f;
        };
        //Create new constructor to create a door from measured peaks, the wall id and the door id
        Door(const Eigen::Vector2f &p0_,
             const Eigen::Vector2f &p1_,
             const Eigen::Vector2f &p0_measured_,
             const Eigen::Vector2f &p1_measured_) : p0(p0_), p1(p1_), p0_measured(p0_measured_), p1_measured(p1_measured_)  /// Consider not using constructor to calculate middle point
        {
            angle = door_angle_respect_to_robot();
            middle = (p0 + p1)/2.f;
            middle_measured = (p0_measured + p1_measured) * 0.5;
        }
        Door(const Eigen::Vector2f &central_point,
             int width, int wall_id_, int id_, float angle_) : middle(central_point), wall_id(wall_id_), id(id_), angle(angle_)
        {
            /// Considering angle respect to robot and the width of the door, calculate the left and right points
            Eigen::Vector2f p0_, p1_;
            p0 = middle + Eigen::Vector2f{cos(angle), sin(angle)} * width/2;
            p1 = middle - Eigen::Vector2f{cos(angle), sin(angle)} * width/2;
        };

        Door(const Eigen::Vector2f &central_point,
             int width, int wall_id, int id) : middle(central_point), wall_id(wall_id), id(id)
        {
            angle = door_angle_respect_to_robot();
            /// Considering angle respect to robot and the width of the door, calculate the left and right points
            Eigen::Vector2f p0_, p1_;
            p0 = middle + Eigen::Vector2f{cos(angle), sin(angle)} * width/2;
            p1 = middle - Eigen::Vector2f{cos(angle), sin(angle)} * width/2;
        };
        bool operator==(const Door &d) const
        {
            return (d.middle - middle).norm() < THRESHOLD;
        };
        Door& operator=(const Door &d)
        {
            p0 = d.p0;
            p1 = d.p1;
            middle = d.middle;
            return *this;
        };
        void print()
        {
            qInfo() << "Door:";
            qInfo() << "    left:" << p0.x() << p0.y();
            qInfo() << "    right:" << p1.x() << p1.y();
        };
        float dist_to_robot() const
        { return middle.norm();}
        float angle_to_robot() const
        { return (float)atan2(middle.x(), middle.y());}
        std::pair<Eigen::Vector2f, Eigen::Vector2f> point_perpendicular_to_door_at(float dist=1000.f) const // mm
        {
            // Calculate the direction vector from p1 to p2 and rotate it by 90 degrees
            Eigen::Vector2f d_perp{-(p0.y() - p1.y()), p0.x() - p1.x()};
            // Normalize the perpendicular vector to get the unit vector
            Eigen::Vector2f u_perp = d_perp.normalized();
            // Calculate the points P1 and P2 at a distance of 1 meter from M along the perpendicular
            Eigen::Vector2f a, b;
            a = middle + (u_perp * dist); // 1 meter in the direction of u_perp
            b = middle - (u_perp * dist); // 1 meter in the opposite direction of u_perp
            return a.norm() < b.norm() ? std::make_pair(a,b) : std::make_pair(b,a);
        }
        float door_angle_respect_to_robot()
        {
            /// Get the vector from door center to the perpendicular point to door closer to the robot
            auto p = point_perpendicular_to_door_at_measured();
            auto closest = p.first.norm() < p.second.norm() ? p.first : p.second;
            /// Considering robot pose is (0, 0), generate a Eigen::Vector2f from center point to 100 mm in front of the robot
            Eigen::Vector2f robot_front{0, 100};
            /// Get vector starting in robot center and ending in the closest point to the door
            Eigen::Vector2f door_to_robot = middle_measured - closest;
            /// Calculate the angle between the robot front and the closest point to the robot
            return min_angle_between_vectors(robot_front, door_to_robot);
        }
        std::pair<Eigen::Vector2f, Eigen::Vector2f> point_perpendicular_to_door_at_measured(float dist=1000.f) const // mm
        {
            // Calculate the direction vector from p1 to p2 and rotate it by 90 degrees
            Eigen::Vector2f d_perp{-(p0_measured.y() - p1_measured.y()), p0_measured.x() - p1_measured.x()};
            // Normalize the perpendicular vector to get the unit vector
            Eigen::Vector2f u_perp = d_perp.normalized();
            // Calculate the points P1 and P2 at a distance of 1 meter from M along the perpendicular
            Eigen::Vector2f a, b;
            a = middle_measured + (u_perp * dist); // 1 meter in the direction of u_perp
            b = middle_measured - (u_perp * dist); // 1 meter in the opposite direction of u_perp
            return a.norm() < b.norm() ? std::make_pair(a,b) : std::make_pair(b,a);
        }
        float perp_dist_to_robot() const
        {
            auto p = point_perpendicular_to_door_at();
            return p.first.norm();
        }
        float perp_angle_to_robot() const
        {
            auto p = point_perpendicular_to_door_at();
            return atan2(p.first.x(), p.first.y());
        }
        float width() const
        {
            return (p0 - p1).norm();
        }
        float height() const
        {
            return 2000; // mm TODO: get from lidar
        }
        float position_in_wall(const std::vector<Eigen::Vector2f> &corners) const
        {
            // we need to compute the distance from the middle of the door to the wall's corner CCW
            // first we define a lambda to check if two floats have the same sign, so we can compare angles
            // and separate the case where each point of a door falls in quadrants 2 and 3 (positive and negative angles)
            // Normally, the point with the smallest angle wrt the robot is the closest one, but if the points fall in
            // quadrants 2 and 3, the point with the smallest angle is the one with the greatest angle wrt the robot.
            auto same_sign = [](double a, double b) { return a*b >= 0.0; };
            // we start by computing the wall's corner with the smallest angle wrt the robot
            if(corners.empty()) { qWarning() << __FUNCTION__  << "Corner vector empty"; return -1.f;};
            auto closest_corner = std::ranges::min_element(corners, [same_sign](auto &c1, auto &c2)
            {  double a1 = atan2(c1.x(), c1.y());
                double a2 = atan2(c2.x(), c2.y());
                return same_sign(a1, a2) ? fabs(a1)<fabs(a2) : a1 > a2;
            });
            // the distance to this wall's corner is the norm of the vector from the middle of the door to the corner
            return (middle - *closest_corner).norm();
        }
        // Function to calculate the signed angle between two vectors in radians using Eigen
        float min_angle_between_vectors(const Eigen::Vector2f& u, const Eigen::Vector2f& v) {
            if (u.size() != v.size()) {
                throw std::invalid_argument("Vectors must be of the same dimension.");
            }
            float dot_product = u.dot(v);
            float cross_product = u.x() * v.y() - u.y() * v.x(); // Equivalent to the z-component of the 3D cross product
            return std::atan2(cross_product, dot_product);
        }
    };

    using Doors = std::vector<Door>;
    using Doors_list = std::vector<Doors>;
    using Line = std::vector<Eigen::Vector2f>;
    using Lines = std::vector<Line>;
    using Peaks_list = std::vector<std::vector<int>>;
    struct Constants
    {
        const float SAME_DOOR = 400;   // same door threshold (mm)
        const float MAX_DOOR_WIDTH = 1500;  // mm
        const float MIN_DOOR_WIDTH = 500;   // mm
    };
    Constants consts;
    Doors detect(const Lines &lines, QGraphicsScene *scene, const std::vector<Eigen::Vector2f> &corners);
    Line filter_out_points_beyond_doors(const Line&floor_line, const Doors &doors);

private:
    Peaks_list extract_peaks(const Lines &lines);
    Doors_list get_doors(const Peaks_list &peaks, const Lines &lines, const std::vector<Eigen::Vector2f> &corners);
    Doors filter_doors(const Doors_list &doors);
    void draw_doors(const Doors &doors, const Door &current_door, QGraphicsScene *scene, QColor=QColor("yellow"));
    void draw_peaks(const Peaks_list &peaks_list, const Lines &lines, QGraphicsScene *scene);
};


#endif //FORCEFIELD_DOOR_DETECTOR_H
