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
	// Uncomment if there's too many debug messages
	// but it removes the possibility to see the messages
	// shown in the console with qDebug()
//	QLoggingCategory::setFilterRules("*.debug=false\n");
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
        auto opti = mpc.initialize_omni(num_steps);
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
            auto &[adv, side, rot, res] = result.value();
            qInfo() << adv << side << rot;
            control_buffer.put(std::make_tuple(adv, side, rot));
        }
    }
    fps.print("FPS:");
}

RoboCompMPC::Control SpecificWorker::MPC_newPath(RoboCompMPC::Path newpath)
{
    if(newpath.size() < num_steps)
        return RoboCompMPC::Control{.valid=false};
    if(newpath.size() > num_steps)
        // reduce vector size by removing intermediate points.

        for(auto p : newpath)
        qInfo() << p.x << p.y;
    qInfo() << "-----------------------";
    path_buffer.put(std::move(newpath), [nsteps=num_steps](auto &&newpath, auto &path)
                { path.reserve(nsteps);
                  for(auto &&p: newpath)
                      path.push_back(Eigen::Vector2f{p.x, p.y});
                });
    const auto &[adv, side, rot] = control_buffer.get();
    return RoboCompMPC::Control{.valid=true, .adv=adv, .side=side, .rot=rot};
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