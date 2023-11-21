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

class DoorDetector
{
    public:
        struct Door
        {
            Eigen::Vector2f center_floor;
            std::vector<Eigen::Vector3f> points;  //floor and high
            Eigen::Vector2f p0, p1;
            int idx0, idx1;
            const float height = 1000.f;
        };
        std::vector<Door> detect(const std::vector<std::vector<Eigen::Vector2f>> &lines, AbstractGraphicViewer *viewer=nullptr);
        std::vector<Eigen::Vector2f> filter_out_points_beyond_doors(const std::vector<Eigen::Vector2f> &floor_line_cart, const std::vector<DoorDetector::Door> &doors);

private:
        const float der_threshold = 800.f;
        const int n_lines = 4;
        const float max_door_width = 1100;
        const float min_door_width = 700; // mm
        const float max_door_separation = 100; //mm
        void draw_peaks(const std::vector<std::tuple<int, bool>> &peaks, const std::vector<Eigen::Vector2f> &line, AbstractGraphicViewer *viewer=nullptr);
        void draw_doors(const std::vector<Door> &doors, AbstractGraphicViewer *viewer=nullptr);
};

#endif //FORCEFIELD_DOOR_DETECTOR_H
