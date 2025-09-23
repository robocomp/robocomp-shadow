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
#include "grid.h"
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include "Gridder.h"
#include <fps/fps.h>
#include <timer/timer.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

        bool Gridder_IsPathBlocked(RoboCompGridder::TPath path);
        bool Gridder_LineOfSightToTarget(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, float robotRadius);
        RoboCompGridder::TPoint Gridder_getClosestFreePoint(RoboCompGridder::TPoint source);
        RoboCompGridder::TDimensions Gridder_getDimensions();
        RoboCompGridder::Result Gridder_getPaths(RoboCompGridder::TPoint source,
                                                 RoboCompGridder::TPoint target,
                                                 int max_paths,
                                                 bool tryClosestFreePoint,
                                                 bool targetIsHuman);
        RoboCompGridder::Result Gridder_getPaths_unlocked(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, int max_paths,
        bool tryClosestFreePoint, bool targetIsHuman);

	bool Gridder_setGridDimensions(RoboCompGridder::TDimensions dimensions);
	
	RoboCompGridder::Result Gridder_setLocationAndGetPath(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, RoboCompGridder::TPointVector freePoints, RoboCompGridder::TPointVector obstaclePoints);
    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;
        bool cancel_from_mouse = false;     // cancel current target from mouse right click

        //Graphics
        AbstractGraphicViewer *viewer;

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
            float MAX_LIDAR_LOW_RANGE = 15000;  // mm
            float MAX_LIDAR_HIGH_RANGE = 15000;  // mm
            float MAX_LIDAR_RANGE = MAX_LIDAR_LOW_RANGE;  // mm used in the grid
            int LIDAR_LOW_DECIMATION_FACTOR = 1;
            int LIDAR_HIGH_DECIMATION_FACTOR = 1;
            QRectF GRID_MAX_DIM{-5000, -5000, 10000, 10000};
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            int PERIOD = 100;    // ms (20 Hz) for compute timer
            unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
            int NUM_PATHS_TO_SEARCH = 3;
            float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm
            bool DISPLAY = true ; //TODO: config file
        };
        Params params;

        // Timer
        rc::Timer<> clock;

        // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // grid
        Grid grid;

        // FPS
        FPSCounter fps;
        int hz = 0;

		// DSR
		void insert_path_node(Eigen::Vector2f target, std::vector<Eigen::Vector2f> points);


        // Draw
        void draw_paths(const vector<std::vector<Eigen::Vector2f>> &paths, QGraphicsScene *scene, bool erase_only=false);
        void draw_path(const vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only=false);

        // mutex
        std::mutex mutex_path;

        // Do some work
        //RoboCompGridPlanner::TPlan compute_line_of_sight_target(const Target &target);
        //RoboCompGridPlanner::TPlan compute_plan_from_grid(const Target &target);
        //void adapt_grid_size(const Target &target,  const RoboCompGridPlanner::Points &path);   // EXPERIMENTAL


};

#endif
