/*
 *    Copyright (C) 2023 by YOUR NAME HERE
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
#include "mpc.h"
#include <doublebuffer/DoubleBuffer.h>
#include <fps/fps.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);
        RoboCompGridPlanner::TPlan GridPlanner_modifyPlan(RoboCompGridPlanner::TPlan plan);
        void GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan);

public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;
        FPSCounter fps;

        // we need a vector of MPCs to be able to handle plans with less than NUM_STEPS points
        rc::MPC mpc;
        std::vector<rc::MPC> mpcs;

        struct Params
        {
            int NUM_STEPS = 8;
            int MIN_NUM_STEPS = 2;
            int PERIOD = 50; // ms
            std::string LIDAR_NAME_LOW = "bpearl";
            std::string LIDAR_NAME_HIGH = "helios";
            float MAX_LIDAR_LOW_RANGE = 700;  // mm
            float MAX_LIDAR_HIGH_RANGE = 700;  // mm
            int LIDAR_LOW_DECIMATION_FACTOR = 1;
            int LIDAR_HIGH_DECIMATION_FACTOR = 1;
            int MAX_OBSTACLES = 5;
            float MAX_DIST_TO_OBSTACLES = 0.5; // m
        };
        Params params;
        DoubleBuffer<RoboCompGridPlanner::Points, std::vector<Eigen::Vector2f>> path_buffer;
        DoubleBuffer<std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector2f>>,
                     std::pair<RoboCompGridPlanner::Control, RoboCompGridPlanner::Points>> control_buffer; // adv, side and rot for each point of the path

        // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

};

#endif
