//
// Created by pbustos on 25/04/24.
//

#ifndef FORCEFIELD_AGENT_PARAMS_H
#define FORCEFIELD_AGENT_PARAMS_H

#include <QColor>
#include <opencv2/core/cvdef.h>

namespace rc
{
    struct Params
    {
        int ROBOT_ID = 200;
        std::string robot_name = "Shadow";
        std::string lidar_name = "helios";
        std::vector<std::pair<float, float>> ranges_list = {{1200, 2500}};
        float MAX_LIDAR_LOW_RANGE = 10000;  // mm
        float MAX_LIDAR_HIGH_RANGE = 10000;  // mm
        float MAX_LIDAR_RANGE = 10000;  // mm used in the grid
        int LIDAR_LOW_DECIMATION_FACTOR = 2;
        int LIDAR_HIGH_DECIMATION_FACTOR = 1;
        int PERIOD = 50;    // ms (20 Hz) for compute timer
        int LIDAR_SLEEP_PERIOD = PERIOD;    // ms (10 Hz) for lidar reading
        bool DISPLAY = true;
        unsigned int SECS_TO_GET_IN = 1; // secs
        unsigned int SECS_TO_GET_OUT = 2; // sec//
        int max_distance = 2500; // mm

        // colors
        QColor TARGET_COLOR = {"orange"};
        QColor LIDAR_COLOR = {"LightBlue"};
        QColor PATH_COLOR = {"orange"};
        QColor SMOOTHED_PATH_COLOR = {"magenta"};

        // Hough
        double rhoMin = -8000.0;
        double rhoMax = 8000.0;
        double rhoStep = 20;
        double thetaMin = 0;
        double thetaMax = CV_PI;
        double thetaStep = CV_PI / 180.0f;
        double MIN_VOTES = 100;
        int LINES_MAX = 10;
        int LINE_THRESHOLD = 25;
        float MAX_LIDAR_DISTANCE = 8000;  // mm
        double NMS_DELTA = 0.3;  // degrees  //TODO
        double NMS_DIST = 300;  // mm
        float PAR_LINES_ANG_THRESHOLD = 10;  //degrees
        float CORNERS_PERP_ANGLE_THRESHOLD = 0.2; // radians
        const float NMS_MIN_DIST_AMONG_CORNERS = 200; // mm
    };
 }
#endif //FORCEFIELD_AGENT_PARAMS_H
