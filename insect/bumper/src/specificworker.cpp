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
#include <cppitertools/enumerate.hpp>
#include <cppitertools/filter.hpp>


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
	{
		this->startup_check();
	}
	else
	{
		// Viewer
 		viewer = new AbstractGraphicViewer(this->frame, QRectF(-1000, -1000, 2000, 2000));
        viewer->draw_contour();
        viewer->add_robot(500, 500, 0, 0, QColor("Blue"));

		map_of_points = create_map_of_points();

		timer.start(Period);
	}

}

void SpecificWorker::compute()
{
	auto ldata = get_lidar_data();

	// check for a repulsion force
	for(const auto p: ldata)
	{
		float ang = atan2(p.x, p.y);
		int index = ang*() +  ;
		if(map_of_points.at(index) > p.norm())
		{
			// something is inside the perimeter

		}

	}

	// if resultant force is not zero, inhibit DWA proposal and move the robot
	

}

//////////////////////////////////////////////////////////////////////////////
std::vector<float> SpecificWorker::create_map_of_points()
{
	std::vector<float> dists;
	for(const auto i: iter::range(DEGREES_NUMBER))
	{
		// get the corresponding angle: 0..360 -> -pi, +pi
		float alf = qDegreesToRadians(i)
		//float alf = i*() + ;
		bool found = false;
		for(const int r : iter::range(OUTER_RIG_DISTANCE))
		{
			float x = r * sin(alf);
			float y = r * cos(alf);
			if( not contour.containsPoint(QPointf(x, y), Qt::OddEvenFill))
			{
				dists.push_back(Eigen::Vector2f(x, y).norm() - 1);
				found = true;
				break;
			}

		}
		if(not found) { qFatal("ERROR: Could not find limit for angle " + QString::number(i));	}
	}
}	


std::vector<Eigen::Vector3f> SpecificWorker::get_lidar_data()
{
    std::vector <Eigen::Vector3f> points;
    try 
	{
        auto ldata = lidar3d_proxy->getLidarData("bpearl", 0, 900, 1);
        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
        {
            float dist = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            return p.z < 300		// uppper limit
                   and p.z > -500	// floor limit
                   and dist < 750	// range limit
                   and dist > 550;	// body out limit. This should be computed using the robot's contour
        }, ldata) | iter::enumerate)
            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});
    }
    catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
    return points;
}

//////////////////////////////////////////////////////////////////////////////

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

//////////////////////////// Interfaces //////////////////////////////////////////

void SpecificWorker::OmniRobot_correctOdometer(int x, int z, float alpha)
{
//implementCODE

}

void SpecificWorker::OmniRobot_getBasePose(int &x, int &z, float &alpha)
{
//implementCODE

}

void SpecificWorker::OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state)
{
//implementCODE

}

void SpecificWorker::OmniRobot_resetOdometer()
{
//implementCODE

}

void SpecificWorker::OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state)
{
//implementCODE

}

void SpecificWorker::OmniRobot_setOdometerPose(int x, int z, float alpha)
{
//implementCODE

}

void SpecificWorker::OmniRobot_setSpeedBase(float advx, float advz, float rot)
{
//implementCODE

}

void SpecificWorker::OmniRobot_stopBase()
{
//implementCODE

}

//SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack(RoboCompVisualElements::TObject target)
{
//subscribesToCODE

}



/**************************************/
// From the RoboCompLidar3D you can call this methods:
// this->lidar3d_proxy->getLidarData(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

