#ifndef DOOR_DETECTOR_ALEJ_H
#define DOOR_DETECTOR_ALEJ_H

#include <Eigen/Dense>
#include <genericworker.h>
#include <timer/timer.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

class Door_detector
{
    public:
        struct Door
        {
            Eigen::Vector2f punto_medio;
            Eigen::Vector2f punto1;
            Eigen::Vector2f punto2;
            Eigen::Vector2f pre_middle_point;
            Eigen::Vector2f post_middle_point;
            float dist_pmedio;
        };

        Door_detector() = default;
        std::vector<Door> detector(const std::vector<Eigen::Vector2f> &line);
        void draw(AbstractGraphicViewer *viewer, vector<Door_detector::Door> doors);

    private:

};
#endif