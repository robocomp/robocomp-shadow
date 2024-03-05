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
#include <cppitertools/enumerate.hpp>
#include <cppitertools/slice.hpp>

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
        // Opti
        //auto opti = mpc.initialize_omni(params.NUM_STEPS);
        for(auto i: iter::range(params.MIN_NUM_STEPS, params.NUM_STEPS+1))
        {
            mpcs.emplace_back();
            mpcs.back().opti = mpcs.back().initialize_omni(i);
        }

        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar, this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        this->Period = params.PERIOD;
        timer.start(Period);
	}
}
void SpecificWorker::compute()
{
    /// read LiDAR
    std::vector<Eigen::Vector2f> obstacles;
    if(auto obs = buffer_lidar_data.try_get(); obs.has_value()) // meters
        obstacles = obs.value();


    /// read path
    if(auto path = path_buffer.try_get(); path.has_value())
    {
        int idx = path.value().size() - params.MIN_NUM_STEPS;
        qInfo() << "Path size: " << path.value().size() << " idx: " << idx;
        if(auto result = mpcs[idx].update(path.value(), obstacles); result.has_value())
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
void SpecificWorker::read_lidar()
{
    auto wait_period = std::chrono::milliseconds (this->Period);
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_LOW,
                                                                   params.MAX_LIDAR_LOW_RANGE,
                                                                   params.LIDAR_LOW_DECIMATION_FACTOR);
            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysteresis of 2ms
            if (wait_period > std::chrono::milliseconds((long) data.period + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds((long) data.period - 2)) wait_period++;
            // convert to Eigen
            std::vector<Eigen::Vector3f> eig_data(data.points.size());
            for (const auto &[i, p]: data.points | iter::enumerate)
                eig_data[i] = {p.x, p.y, p.z};
            // get the N closest lidar points to the robot by sorting the array and taking the first N
            std::ranges::sort(eig_data, [](auto &&p1, auto &&p2){ return p1.norm() < p2.norm(); });
            std::vector<Eigen::Vector2f> obstacles;
            for(const auto &p: eig_data | iter::slice(0, std::min(static_cast<int>(eig_data.size()), params.MAX_OBSTACLES), 1))
                obstacles.emplace_back(p[0]/1000.f, p[1]/1000.f);   // to meters for MPC

            buffer_lidar_data.put(std::move(obstacles));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
} // Thread to read the lidar

/////////////////////////////////////////////////////////////////////////////////
/// MPC Interface Implementation
/////////////////////////////////////////////////////////////////////////////////

//RoboCompMPC::Control SpecificWorker::MPC_newPath(RoboCompMPC::Path newpath)
//{
//    if(newpath.size() < NUM_STEPS)
//        return RoboCompMPC::Control{.valid=false};
//    if(newpath.size() > NUM_STEPS)
//        // reduce vector size by removing intermediate points.
//
//        for(auto p : newpath)
//        qInfo() << p.x << p.y;
//    qInfo() << "-----------------------";
//    path_buffer.put(std::move(newpath), [nsteps=NUM_STEPS](auto &&newpath, auto &path)
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

    auto last_path_point = plan.path.back();
    // Print distance to last point
    if(std::hypot(last_path_point.x, last_path_point.y) < 300)
    {
        qWarning() << __FUNCTION__ << "Distance to last point is less than 300mm. Returning original path";
        return plan;
    }

    if(static_cast<int>(plan.path.size()) < params.MIN_NUM_STEPS)
    {
        qWarning() << __FUNCTION__ << "Path with one element or less. Returning original path";
        return plan;
    }
    //qInfo() << __FUNCTION__ << plan.path[params.NUM_STEPS - 1].x << plan.path[params.NUM_STEPS - 1].y;

    int nsteps = std::clamp(static_cast<int>(plan.path.size()), params.MIN_NUM_STEPS, params.NUM_STEPS);
    path_buffer.put(std::move(plan.path), [nsteps](auto &&new_path, auto &path)
    {  std::transform(new_path.begin(), new_path.begin()+nsteps, std::back_inserter(path),
                      [](auto &&p){ return Eigen::Vector2f{p.x, p.y}; });});

    const auto& control_and_path = control_buffer.get();
    if(control_and_path.first.empty() and control_and_path.second.empty())
    {
        qWarning() << __FUNCTION__ << "No solution. Empty control or path";
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