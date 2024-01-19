/*
 *    Copyright (C) 2024 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <fps/fps.h>
#include <person.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);
        void VisualElementsPub_setVisualObjects(RoboCompVisualElementsPub::TData data);

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;

        //Graphics
        AbstractGraphicViewer *viewer;

        // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // Pilar cone parameters
        float cone_radius = 3000;
        float cone_angle = 1;

        struct Params
        {
            float ROBOT_WIDTH = 460;  // mm
            float ROBOT_LENGTH = 480;  // mm
            float ROBOT_SEMI_WIDTH = ROBOT_WIDTH / 2.f;     // mm
            float ROBOT_SEMI_LENGTH = ROBOT_LENGTH / 2.f;    // mm
            float TILE_SIZE = 100;   // mm
            float MIN_DISTANCE_TO_TARGET = ROBOT_WIDTH / 2.f; // mm
            std::string LIDAR_NAME_LOW = "bpearl";
            std::string LIDAR_NAME_HIGH = "helios";
            float MAX_LIDAR_LOW_RANGE = 10000;  // mm
            float MAX_LIDAR_HIGH_RANGE = 10000;  // mm
            float MAX_LIDAR_RANGE = 10000;  // mm used in the grid
            int LIDAR_LOW_DECIMATION_FACTOR = 2;
            int LIDAR_HIGH_DECIMATION_FACTOR = 1;
            QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
            float CARROT_DISTANCE = 400;   // mm
            float CARROT_ANGLE = M_PI_4 / 6.f;   // rad
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            int PERIOD = 100;    // ms (10 Hz) for compute timer
            float MIN_ANGLE_TO_TARGET = 1.f;   // rad
            int MPC_HORIZON = 8;
            bool USE_MPC = true;
            unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
            int NUM_PATHS_TO_SEARCH = 3;
            float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm

            // colors
            QColor TARGET_COLOR= {"orange"};
            QColor LIDAR_COLOR = {"LightBlue"};
            QColor PATH_COLOR = {"orange"};
            QColor SMOOTHED_PATH_COLOR = {"magenta"};
        };
        Params params;

        // People
        Person wanted_person;
        using People = std::vector<Person>;
        People people;

        // Robot path
        std::vector<QGraphicsEllipseItem*> points;

        // Visual elements
        DoubleBuffer<RoboCompVisualElementsPub::TData, RoboCompVisualElementsPub::TData> buffer_visual_elements;
        DoubleBuffer<RoboCompVisualElementsPub::TData, RoboCompVisualElementsPub::TData> buffer_room_elements;
        void draw_lidar(const vector<Eigen::Vector3f> &points, int decimate);
        void draw_room(const RoboCompVisualElementsPub::TObject &obj);
        void draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only);
        void process_visual_elements(const RoboCompVisualElementsPub::TData &data);
        void process_room_elements(const RoboCompVisualElementsPub::TData &data);
        void print_people(const People &ppol);

        // fps
        FPSCounter fps;
        int hz = 0;
};

#endif
