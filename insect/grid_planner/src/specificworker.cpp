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
#include "cppitertools/filter.hpp"

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


	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = 100;

	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
        viewer = new AbstractGraphicViewer(this->frame, QRectF(-4000, -2000,8000,6000 ));

        QRectF dim{-2000,-500,4000,3000};
        viewer -> draw_contour();
        viewer ->add_robot(500,500,0,0, QColor("Blue"));
        grid.initialize(dim, 200, &viewer->scene, false);

		timer.start(Period);
	}
}

void SpecificWorker::compute()
{

//    if (auto target = target_buffer.try_get(); target.has_value())

    {

        auto points = get_lidar_data();
        qInfo() << points.size();
        grid.update_map(points, Eigen::Vector2f{0.0, 0.0}, 3500);
    }
//    else //NO TARGET
//    {
//
//    }
}

std::vector<Eigen::Vector3f> SpecificWorker::get_lidar_data()
{
    std::vector<Eigen::Vector3f> points;
    try
    {
        auto ldata = lidar3d_proxy->getLidarData(0, 900);
        for ( auto &&[i,p] : iter::filter([](auto p){return p.z < -1200 and p.z > -1400 and sqrt(p.x*p.x+p.y*p.y+p.z*p.z) < 10000; }, ldata) | iter::enumerate)
            points.emplace_back(Eigen::Vector3f {p.x, p.y, p.z});



        return points;
    }
    catch(const Ice::Exception &e)
    {  std::cout << "Error reading from Lidar3D" << e << std::endl; }
    return points;
}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


RoboCompGridPlanner::TPlan SpecificWorker::GridPlanner_getPlan()
{
//implementCODE

}

//SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack(int track)
{
    target_buffer.put(Eigen::Vector2f{100.0,100.0});
}



/**************************************/
// From the RoboCompLidar3D you can call this methods:
// this->lidar3d_proxy->getLidarData(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint

/**************************************/
// From the RoboCompGridPlanner you can use this types:
// RoboCompGridPlanner::Tpoint
// RoboCompGridPlanner::TPlan

