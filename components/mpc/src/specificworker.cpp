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
#include "specificworker.h"
#include <cppitertools/range.hpp>

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
//	try
//	{
//		RoboCompCommonBehavior::Parameter par = params.at("InnerModelPath");
//		std::string innermodel_path = par.value;
//		innerModel = std::make_shared(innermodel_path);
//	}
//	catch(const std::exception &e) { qFatal("Error reading config params"); }
	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = period;
	if(this->startup_check_flag)
        this->startup_check();
	else
    {
        auto opti = mpc.initialize_omni(params.num_steps);
        this->Period = 50;
        timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    double slack_weight = 10;
    if(auto path = path_buffer.try_get(); path.has_value())
    {
        if(auto result = mpc.update(path.value(), slack_weight); result.has_value())
        {
            auto control_and_path = result.value();
            //qInfo() << adv << side << rot;
            control_buffer.put(std::move(control_and_path), [](auto &&control_and_path, auto &grid_planner)
            {
                std::ranges::transform(control_and_path.first, std::back_inserter(grid_planner.first),  // controls
                                       [](auto &&c){ return RoboCompGridPlanner::TControl{.adv=c[0], .side=c[1], .rot=c[2]}; });
                std::ranges::transform(control_and_path.second, std::back_inserter(grid_planner.second),    // points
                                       [](auto &&p){ return RoboCompGridPlanner::TPoint{.x=p[0], .y=p[1]}; });
            });
        }
        else
            control_buffer.put(std::make_pair(std::vector<Eigen::Vector3f>{}, std::vector<Eigen::Vector2f>{}));
    }
    fps.print("FPS:");
}

/////////////////////////////////////////////////////////////////////////////////
/// MPC Interface Implementation
/////////////////////////////////////////////////////////////////////////////////

//RoboCompMPC::Control SpecificWorker::MPC_newPath(RoboCompMPC::Path newpath)
//{
//    if(newpath.size() < num_steps)
//        return RoboCompMPC::Control{.valid=false};
//    if(newpath.size() > num_steps)
//        // reduce vector size by removing intermediate points.
//
//        for(auto p : newpath)
//        qInfo() << p.x << p.y;
//    qInfo() << "-----------------------";
//    path_buffer.put(std::move(newpath), [nsteps=num_steps](auto &&newpath, auto &path)
//                { path.reserve(nsteps);
//                  for(auto &&p: newpath)
//                      path.push_back(Eigen::Vector2f{p.x, p.y});
//                });
//    const auto &[adv, side, rot] = control_buffer.get();
//    return RoboCompMPC::Control{.valid=true, .adv=adv, .side=side, .rot=rot};
//}
RoboCompGridPlanner::TPlan SpecificWorker::GridPlanner_modifyPlan(RoboCompGridPlanner::TPlan plan)
{
    if(not plan.valid)
    {
        qWarning() << __FUNCTION__ << "Invalid plan";
        return RoboCompGridPlanner::TPlan{.valid=false};
    }

    if(plan.path.size() < params.num_steps)
    {
        qWarning() << __FUNCTION__ << "Path too short. Returning original path";
        return plan;
    }
    path_buffer.put(std::move(plan.path), [nsteps=params.num_steps](auto &&new_path, auto &path)
    {  std::transform(new_path.begin(), new_path.begin()+nsteps, std::back_inserter(path),
                      [](auto &&p){ return Eigen::Vector2f{p.x, p.y}; });});

    const auto& control_and_path = control_buffer.get();
    if(control_and_path.first.empty() and control_and_path.second.empty())
    {
        qWarning() << __FUNCTION__ << "Empty control or path";
        return RoboCompGridPlanner::TPlan{.valid=false};
    }
    else
    {
        return RoboCompGridPlanner::TPlan{.path=control_and_path.second,
                                      .controls=control_and_path.first,
                                      .timestamp=std::chrono::system_clock::now().time_since_epoch().count(),
                                      .valid=true,};
    }
}
void SpecificWorker::GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan)
{
    qWarning() << "GridPlanner_setPlan not implemented";
}

///////////////////////////////////////////////////////////////////////////////

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


//
//std::vector<Eigen::Vector2f> path{Eigen::Vector2f{100, 0}, Eigen::Vector2f{200, 0},
//                                  Eigen::Vector2f{300, 0}, Eigen::Vector2f{400, 0},
//                                  Eigen::Vector2f{500, 0}, Eigen::Vector2f{600, 0},
//                                  Eigen::Vector2f{700, 0}, Eigen::Vector2f{800, 0},
//                                  Eigen::Vector2f{800, 0},Eigen::Vector2f{1000, 0}};