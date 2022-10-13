/*
 *    Copyright (C) 2022 by YOUR NAME HERE
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
        // grphics
        viewer_dimensions = QRectF(-2500, -2500, 5000, 5000);
        viewer = new AbstractGraphicViewer(this->beta_frame, this->viewer_dimensions);
        this->resize(900,650);
        const auto &[rp, re] = viewer->add_robot(consts.robot_length, consts.robot_length);
        robot_polygon = rp;
        laser_in_robot_polygon = new QGraphicsRectItem(-10, 10, 20, 20, robot_polygon);
        laser_in_robot_polygon->setPos(0, 190);     // move this to abstract
        viewer->draw_contour();

        timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    RoboCompGenericBase::TBaseState bState;
    try
    {  omnirobot_proxy->getBaseState(bState); }
    catch(const Ice::Exception &e)
    { std::cout << e.what() << " Error reading omnibase_proxy" << std::endl; return;}
    robot_polygon->setPos(bState.x, bState.z);
    robot_polygon->setRotation(qRadiansToDegrees(bState.alpha));

    RoboCompCameraRGBDSimple::TImage omni_depth;       //omni_camera depth comes as RGB
	try
	{
        omni_depth = camerargbdsimple_proxy->getImage("/Shadow/omnicamera/sensorDepth");
        if(omni_depth.image.empty())
        { qInfo() << "Empty image"; return;}
    }
    catch(const Ice::Exception &e)
    { std::cout << e.what() << " Error reading camerargbdsimple_proxy" << std::endl; return;}

    cv::Mat omni_depth_frame(cv::Size(omni_depth.width, omni_depth.height), CV_8UC3, &omni_depth.image[0], cv::Mat::AUTO_STEP);
    cv::cvtColor(omni_depth_frame, omni_depth_frame, cv::COLOR_RGB2GRAY);
    cv::Mat omni_depth_float;
    omni_depth_frame.convertTo(omni_depth_float, CV_32FC1);

    auto floor_lines = get_multi_level_3d_points(omni_depth_float);
    draw_floor_line(floor_lines);

    Eigen::Vector2f force = compute_repulsion_forces(floor_lines[1]);
    draw_forces(force); // in robot coordinate system

    try
    {
        Eigen::Vector2f gains{5.0, 3.0};
        force = force.cwiseProduct(gains);
        float rot = atan2(force.x(), force.y());
        float adv = force.y() ;
        float side = force.x() ;
        omnirobot_proxy->setSpeedBase(side, adv, rot);
    }
    catch (const Ice::Exception &e){ std::cout << e.what() << std::endl;}

}
////////////////////////////////////////////////////////////////////////
Eigen::Vector2f SpecificWorker::compute_repulsion_forces(std::vector<Eigen::Vector2f> &floor_line)
{
    Eigen::Vector2f res = {0.f, 0.f};
    for(const auto &ray: floor_line)
    {
        if (ray.norm() > 1500) continue;
        res += -ray.normalized() / fabs(pow(ray.norm()/1000, 3));
    }
    return res;  //scale factor
    // return std::accumulate(floor_line.begin(), floor_line.end(), Eigen::Vector2f{0.f, 0.f},[](auto a, auto b){return a -b.normalized()/b.norm();});
}
std::vector<std::vector<Eigen::Vector2f>> SpecificWorker::get_multi_level_3d_points(const cv::Mat &depth_frame)
{
    std::vector<std::vector<Eigen::Vector2f>> points(int((2000-400)/200));  //height steps
    for(auto &p: points)
        p.resize(360, Eigen::Vector2f(consts.max_camera_depth_range, consts.max_camera_depth_range));   // angular resolution

    int semi_height = depth_frame.rows/2;
    float hor_ang, dist, x, y, z, proy;
    float ang_slope = 2*M_PI/depth_frame.cols;
    const float ang_bin = 2.0*M_PI/consts.num_angular_bins;

    for(int u=0; u<depth_frame.rows; u++)
        for(int v=0; v<depth_frame.cols; v++)
        {
            hor_ang = ang_slope * v - M_PI; // cols to radians
            dist = depth_frame.ptr<float>(u)[v] * consts.scaling_factor;  // pixel to dist scaling factor  -> to mm
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
            proy = dist * cos( atan2((semi_height - u), 128.f));
            z = (semi_height - u)/128.f * proy; // 128 focal as PI fov angle for 256 pixels
            z += consts.omni_camera_height; // get from DSR
            // add Y axis displacement

            if(z < 10) continue; // filter out floor

            // add only to its bin if less than current value
            for(auto &&[level, step] : iter::range(400, 2000, 200) | iter::enumerate)
                if(z > step and z < step+200)
                {
                    int ang_index = floor((M_PI + atan2(x, y)) / ang_bin);
                    Eigen::Vector2f new_point(x, y);
                    if(new_point.norm() <  points[level][ang_index].norm())
                        points[level][ang_index] = new_point;
                }
        };
    return points;
}
void SpecificWorker::draw_floor_line(const vector<vector<Eigen::Vector2f>> &lines)    //one vector for each height level
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &item: items)
        viewer->scene.removeItem(item);
    items.clear();

    QPen pen(QColor("orange"));
    QBrush brush(QColor("orange"));

    static QStringList my_color = {"red", "orange", "blue", "magenta", "black", "yellow", "brown", "cyan"};
    std::vector<vector<Eigen::Vector2f>> copy(lines.begin()+1, lines.end()-3);
    for(auto &&[k, line]: copy | iter::enumerate)
        for(const auto &p: line)
        {
            auto item = viewer->scene.addEllipse(-10, -10, 20, 20, QPen(QColor(my_color.at(k))), QBrush(QColor(my_color.at(k))));
            item->setPos(robot_polygon->mapToScene(p.x(), p.y()));
            items.push_back(item);
        }
}
void SpecificWorker::draw_forces(const Eigen::Vector2f &force)
{
    static QGraphicsItem* item=nullptr;
    if(item != nullptr) viewer->scene.removeItem(item);
    delete item;
    auto large_force = force * 3.f;
    QPointF tip = robot_polygon->mapToScene(large_force.x(), large_force.y());
    item = viewer->scene.addLine(robot_polygon->pos().x(), robot_polygon->pos().y(), tip.x(), tip.y(), QPen(QColor("red"), 50));
}


////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}





/**************************************/
// From the RoboCompCameraRGBDSimple you can call this methods:
// this->camerargbdsimple_proxy->getAll(...)
// this->camerargbdsimple_proxy->getDepth(...)
// this->camerargbdsimple_proxy->getImage(...)
// this->camerargbdsimple_proxy->getPoints(...)

/**************************************/
// From the RoboCompCameraRGBDSimple you can use this types:
// RoboCompCameraRGBDSimple::Point3D
// RoboCompCameraRGBDSimple::TPoints
// RoboCompCameraRGBDSimple::TImage
// RoboCompCameraRGBDSimple::TDepth
// RoboCompCameraRGBDSimple::TRGBD

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

