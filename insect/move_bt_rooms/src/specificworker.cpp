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
#include <cppitertools/enumerate.hpp>
#include "specificworker.h"

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
//	THE FOLLOWING IS JUST AN EXAMPLE
//	To use innerModelPath parameter you should uncomment specificmonitor.cpp readConfig method content
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

        timer.start(Period);
	}

    this->min_distance = 0.0f;
    this->current_phi = M_2_PI;

    viewer = new AbstractGraphicViewer(this->frame, QRectF(-3000, -3000, 6000, 6000), false);
    viewer->add_robot(460, 480, 0, 100, QColor("Blue"));
    viewer->show();
    std::cout << "Started viewer" << std::endl;

//    BT::NodeConfiguration config;
//    config.blackboard = BT::Blackboard::create();
//    config.blackboard->set("doors", this->detector);
//    LookForNewDoor lookForNewDoorNode("LookForNewDoor", config);
//    factory.registerSimpleAction("OpenWorker", [&](BT::TreeNode&){ return this->getLidar2D(); } );
//    this->factory.registerNodeType<GoMiddleOfTheRoom>("GoMiddleOfTheRoom",this, std::bind(&RoboCompLidar3D::Lidar3DPrx::getLidarDataWithThreshold2d, this->lidar3d_proxy));

    this->dataPtr = std::make_shared<Data>();

    this->robot_comunication_th = std::move(std::thread(&SpecificWorker::robot_comunication,this));

    this->factory.registerNodeType<LookForNewDoor>("LookForNewDoor",this->dataPtr);
    this->factory.registerNodeType<InFrontDoor>("InFrontDoor", this->dataPtr);
    this->factory.registerNodeType<GoThroughDoor>("GoThroughDoor",this->dataPtr);
    this->factory.registerNodeType<GoMiddleOfTheRoom>("GoMiddleOfTheRoom");

    std::cout << "Node registered" << std::endl;

    // TreeNodes are destroyed
    try
    {
        //Executing in /bin
        this->tree = factory.createTreeFromFile("./src/my_tree.xml");
//        auto tree = factory.createTreeFromFile("/home/robolab/robocomp/components/robocomp-shadow/insect/move_bt_rooms/src/my_tree.xml");

    } catch (const std::exception& e) {
        std::cerr << "Error al crear el árbol de comportamiento: " << e.what() << std::endl;
    }

}

void SpecificWorker::compute()
{

    auto lidar_points = this->lidar3d_proxy->getLidarData("helios",0,360,1).points;

    std::ranges::sort(lidar_points, {}, &RoboCompLidar3D::TPoint::phi);

    this->current_phi = trunc((lidar_points[0].phi * 180) / M_PI);
    this->aux_point = lidar_points[0];
    
    std::vector<Eigen::Vector2f> lidar2D;

    for(auto &&point: lidar_points)
    {
       if (point.z > 500.0 && point.z < 700.0)
       {
            int check_phi = std::trunc((point.phi * 180) / M_PI);

            if(this->current_phi != check_phi)
            {
                lidar2D.push_back(Eigen::Vector2f{this->aux_point.x,this->aux_point.y});
                this->aux_point = point;
                this->current_phi = check_phi;
            }
            else
            if(point.distance2d < this->aux_point.distance2d)
            {
                this->aux_point = point;
            }
        }
    }

    //Insert last point to the filtered lidar
    lidar2D.push_back(Eigen::Vector2f{this->aux_point.x,this->aux_point.y});

    draw_floor_line(lidar2D);
    Lines lines = this->extract_lines(lidar_points,this->consts.ranges_list);

    //Get doors from lidar2D
    this->dataPtr->detected_doors = this->detector.detect(lines,&this->viewer->scene);

//    std::cout << __FUNCTION__ << "ROT_POINT:" << this->dataPtr->rot_point << std::endl;

    this->detector.draw_doors(this->dataPtr->detected_doors,this->dataPtr->target_door,&this->viewer->scene);

    draw_floor_line(lidar2D);

    this->tree.tickWhileRunning();
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

void SpecificWorker::draw_floor_line(const vector<Eigen::Vector2f> &lines)
{
    static std::vector<QGraphicsItem *> draw_points;
    for(const auto &p : draw_points)
    {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_points.clear();

    for(const auto &p: lines)
    {
        auto o = viewer->scene.addRect(-10, 10, 20, 20, QPen(QColor("green")), QBrush(QColor("green")));
        o->setPos(QPointF{p.x(),p.y()});
        draw_points.push_back(o);
    }
}

SpecificWorker::Lines SpecificWorker::extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges)
{
    Lines lines(ranges.size());
    for(const auto &p: points)
        for(const auto &[i, r] : ranges | iter::enumerate)
            if(p.z > r.first and p.z < r.second)
                lines[i].emplace_back(p.x, p.y);

    return lines;
}

void SpecificWorker::robot_comunication()
{
    while(true)
    {
        try
        {
            omnirobot_proxy->setSpeedBase(this->dataPtr->advy_point,-this->dataPtr->advx_point,-this->dataPtr->rot_point);
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error talking to OmniRobot " << e.what() << std::endl; }
    }
}

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



/**************************************/
// From the RoboCompLidar3D you can call this methods:
// this->lidar3d_proxy->getLidarData(...)
// this->lidar3d_proxy->getLidarDataArrayProyectedInImage(...)
// this->lidar3d_proxy->getLidarDataProyectedInImage(...)
// this->lidar3d_proxy->getLidarDataWithThreshold2d(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData

/**************************************/
// From the RoboCompOmniRobot you can call this methods:
// this->omnirobot_proxy->correctOdometer(...)
// this->omnirobot_proxy->getBasePose(...)
// this->omnirobot_proxy->getBaseState(...)
// this->omnirobot_proxy->resetOdometer(...)
// this->omnirobot_proxy->setOdometer(...)
// this->omnirobot_proxy->setOdometerPose(...)
// this->omnirobot_proxy->setSpeedBase(...)
// this->omnirobot_proxy->stopBase(...)

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

