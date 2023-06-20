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
	{
		this->startup_check();
	}
	else
	{
		// Viewer
 		viewer = new AbstractGraphicViewer(this->frame, QRectF(-3000, -3000, 6000, 6000));
        //viewer->draw_contour();
        viewer->add_robot(200, 200, 0, 0, QColor("Blue"));

        // creatºe map from degrees (0..360)  -> edge distances
        int robot_width = 460;
        int robot_heigth = 480;
        robot_contour << QPointF(-230,240) << QPointF(230, 240) << QPointF(230, -240) << QPointF(-230, -240);
		map_of_points = create_map_of_points();
        draw_ring(map_of_points, &viewer->scene);

        timer.start(Period);
	}

}

void SpecificWorker::compute()
{
	auto ldata = get_lidar_data();

    // check for a repulsion force
    Eigen::Vector2f result{0.f, 0.f};
    std::vector<QPointF> draw_points;
    for(const auto &p: ldata)
	{
		float ang = atan2(p.x(), p.y());
        int index  = qRadiansToDegrees(ang);
        if(index <0) index += 360;
        float diff = map_of_points.at(index) - p.head(2).norm();
        //qInfo() << "diff" << diff;
        if(diff < 0 and diff> -BAND_WIDTH)
        {
			// something is inside the perimeter
            float modulus = std::clamp(diff*(1.f/BAND_WIDTH), 0.f, 1.f);
            result -= p.head(2).normalized() * modulus;
            draw_points.push_back(QPointF(p.x(), p.y()));
		}
	}

    draw_ring_points(draw_points, &viewer->scene);
    //draw_all_points(ldata, &viewer->scene);



    // if resultant force is zero and there are NOT points inside the ring, inhibit DWA proposal and move the robot

    // if resultant force is zero and there are points inside de ring, the robot is TRAPPED
	

}

//////////////////////////////////////////////////////////////////////////////
std::vector<float> SpecificWorker::create_map_of_points()
{
	std::vector<float> dists;
	for(const auto &i: iter::range(DEGREES_NUMBER))
	{
		// get the corresponding angle: 0..360 -> -pi, +pi
		float alf = qDegreesToRadians(i);
		bool found = false;
		for(const int r : iter::range(OUTER_RIG_DISTANCE))
		{
			float x = r * sin(alf);
			float y = r * cos(alf);
            if( not robot_contour.containsPoint(QPointF(x, y), Qt::OddEvenFill))
			{
				dists.push_back(Eigen::Vector2f(x, y).norm() - 1);
				found = true;
				break;
			}
		}
		if(not found) { qFatal("ERROR: Could not find limit for angle ");	}
	}
    return dists;
}

std::vector<Eigen::Vector3f> SpecificWorker::get_lidar_data()
{
    std::vector <Eigen::Vector3f> points;
    try 
	{
        auto ldata = lidar3d_proxy->getLidarData("bpearl", 0, 360, 1);
        for (auto &&[i, p]: iter::filter([this](auto p)
        {
            float dist = sqrt(p.x*p.x + p.y*p.y);
            float ang = atan2(p.x, p.y);
            int index  = qRadiansToDegrees(ang);
            if(index <0) index += 360;
            return dist > map_of_points.at(index) and p.z > -700 and p.z < 0;
        }, ldata) | iter::enumerate)
            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});
    }
    catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
    return points;
}

//            return p.z < 100		// uppper limit
//                   and p.z > -600	// floor limit
//                   and dist < 1000	// range limit
//                   and dist > 100;	// body out limit. This should be computed using the robot's contour

void SpecificWorker::draw_ring(const std::vector<float> &dists, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for(const auto &p : draw_points) {
        scene->removeItem(p);
        delete p;
    }
    draw_points.clear();

    QPolygonF poly;
    for(const auto &[i, p]: dists | iter::enumerate)
        poly << QPointF(p * cos(qDegreesToRadians(i)), p * sin(qDegreesToRadians(i)));

    qInfo() << poly;
    auto o = scene->addPolygon(poly, QPen(QColor("red"), 10));
    draw_points.push_back(o);
}

void SpecificWorker::draw_ring_points(const std::vector<QPointF> &points, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for(const auto &p : draw_points) {
        scene->removeItem(p);
        delete p;
    }
     draw_points.clear();

     for(const auto p: points)
     {

         auto o = scene->addRect(-10, 10, 20, 20, QPen(QColor("green")), QBrush(QColor("green")));
         o->setPos(p);
         draw_points.push_back(o);
     }
}

void SpecificWorker::draw_all_points(const std::vector<Eigen::Vector3f> &points, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;

    for(const auto &p : draw_points) {
        scene->removeItem(p);
        delete p;
    }
    draw_points.clear();

    for(const auto p: points)
    {
        auto o = scene->addRect(-10, 10, 20, 20, QPen(QColor("blue")), QBrush(QColor("blue")));
        o->setPos(p.x(), p.y());
        draw_points.push_back(o);
    }
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

