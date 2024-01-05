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
        rc::MPC mpc;
        struct Params
        {
            unsigned int num_steps = 10;
        };
        Params params;
        DoubleBuffer<RoboCompGridPlanner::Points, std::vector<Eigen::Vector2f>> path_buffer;
        DoubleBuffer<std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector2f>>,
                     std::pair<RoboCompGridPlanner::Control, RoboCompGridPlanner::Points>> control_buffer; // adv, side and rot for each point of the path

};

#endif
