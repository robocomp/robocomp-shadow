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
            const float THRESHOLD = 500; //door equality
            Eigen::Vector2f center_floor;
            std::vector<Eigen::Vector3f> points;  //floor and high
            Eigen::Vector2f p0, p1, middle;
            int idx_in_peaks_0, idx_in_peaks_1;
            const float height = 1000.f;
            Door(){ p0 = p1 = middle = Eigen::Vector2f{0.f,0.f}; idx_in_peaks_0 = idx_in_peaks_1 = -1;}
            Door(Eigen::Vector2f &&p0_, Eigen::Vector2f &&p1_, int idx0, int idx1) : p0(p0_), p1(p1_), idx_in_peaks_0(idx0), idx_in_peaks_1(idx1)
            {
                middle = (p0 + p1) / 2.f;
                idx_in_peaks_0 = idx_in_peaks_1 = -1;
            };
            Door(const Eigen::Vector2f &p0_, const Eigen::Vector2f &p1_, int idx0, int idx1) : p0(p0_), p1(p1_), idx_in_peaks_0(idx0), idx_in_peaks_1(idx1)
            {
                middle = (p0 + p1) / 2.f;
                idx_in_peaks_0 = idx_in_peaks_1 = -1;
            };
            bool operator==(const Door &d) const
            {
                return (d.middle-middle).norm() < THRESHOLD;
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
                qInfo() << "    p0:" << p0.x() << p0.y();
                qInfo() << "    p1:" << p1.x() << p1.y();
            };
            float dist_to_robot() const { return middle.norm();}
            float angle_to_robot() const { return atan2(middle.x(), middle.y());}
            Eigen::Vector2f point_perpendicular_to_door_at(float dist=1000) const  //mm
            {
                // Calculate the direction vector from p1 to p2 and rotate it by 90 degrees
                Eigen::Vector2f d_perp{ - (p0.y() - p1.y()), p0.x() - p1.x()};
                Eigen::Vector2f d_perp_n = d_perp.normalized();

                // Calculate the points P1 and P2 at a distance of +-1 meter along the perpendicular
                Eigen::Vector2f a, b;
                a = middle + d_perp_n * dist;
                b = middle - d_perp_n * dist;
                return a.norm() < b.norm() ? a : b;
            }
            float perp_dist_to_robot() const { return point_perpendicular_to_door_at().norm(); }
            float perp_angle_to_robot() const
            {
                auto p = point_perpendicular_to_door_at();
                return atan2(p.x(), p.y());
            }
            float door_angle_to_robot() const
            {
                Eigen::Vector2f d_perp{ - (p0.y() - p1.y()), p0.x() - p1.x()};
                Eigen::Vector2f d_perp_n = d_perp.normalized();
                return atan2(d_perp_n.x(), d_perp_n.y());
            }
        };
        
        using Doors = std::vector<Door>;
        using Doors_list = std::vector<Doors>;
        using Lines = std::vector<std::vector<Eigen::Vector2f>>;
        using Peaks = std::vector<uint>;
        using Peaks_list = std::vector<std::vector<uint>>;  // indices of peaks in each line
        std::vector<Eigen::Vector2f> current_line; // to hold the current line being processed (lowest line for now)

        std::vector<Eigen::Vector2f> filter_out_points_beyond_doors(const std::vector<Eigen::Vector2f> &floor_line_cart, const Doors &doors);
        Doors detect(const RoboCompLidar3D::TPoints &points, QGraphicsScene *viewer);
        Lines extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges);
        Peaks_list extract_peaks(const Lines &lines);
        Doors_list get_doors(const Peaks_list &peaks, const Lines &lines);
        Doors filter_doors(const Doors_list &doors_list);
        void draw_doors(const Doors &doors, const Door &door_target, QGraphicsScene *scene, QColor color=QColor("blue"));
    private:
            std::vector<std::pair<float, float>> height_ranges;

};


#endif //FORCEFIELD_DOOR_DETECTOR_H
