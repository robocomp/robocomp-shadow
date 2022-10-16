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
        viewer_dimensions = QRectF(-2500, -2500, 5000, 5000);  //robot view
        viewer = new AbstractGraphicViewer(this->beta_frame, this->viewer_dimensions);
        this->resize(900,650);
        const auto &[rp, re] = viewer->add_robot(consts.robot_length, consts.robot_length);
        robot_polygon = rp;
        laser_in_robot_polygon = new QGraphicsRectItem(-10, 10, 20, 20, robot_polygon);
        laser_in_robot_polygon->setPos(0, 190);     // move this to abstract
        viewer->draw_contour();

        // global
        IS_COPPELIA = false;

        timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    cv::Mat omni_rgb_frame;
    cv::Mat omni_depth_frame;
    RoboCompCameraRGBDSimple::TPoints points;

    if(IS_COPPELIA)
    {
        omni_depth_frame = read_depth_coppelia();
        if(omni_depth_frame.empty()) { qWarning() << "omni_depth_frame empty"; return;}
        omni_rgb_frame = read_rgb("/Shadow/omnicamera/sensorRGB");
        if(omni_rgb_frame.empty()) { qWarning() << "omni_rgb_frame empty"; return;}
    }
    else    // DREAMVU
    {
        omni_rgb_frame = read_rgb("");
        if(omni_rgb_frame.empty()) { qWarning() << "omni_rgb_frame empty dreamvu"; return;}
        omni_depth_frame = read_depth_dreamvu(omni_rgb_frame);
        if(omni_rgb_frame.empty()) { qWarning() << "omni_depth_frame dreamvu empty"; return;}
        //points = read_points_dreamvu(omni_depth_frame);
        //  draw_3d_points(omni_points);
    }

    draw_lines_on_image(omni_rgb_frame, omni_depth_frame);
    imshow("rgb", omni_rgb_frame);
    //cv::normalize(omni_depth_frame_norm, omni_depth_frame_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //imshow("depth", omni_depth_frame_norm);

    // compute level_lines
    //auto lines = get_multi_level_3d_points(omni_depth_frame);
    //draw_floor_line(lines);

    // potential field algorithm
    //Eigen::Vector2f force = compute_repulsion_forces(floor_lines[1]);
    //draw_forces(force); // in robot coordinate system

    try
    {
        Eigen::Vector2f gains{5.0, 3.0};
        //force = force.cwiseProduct(gains);
        //float rot = atan2(force.x(), force.y());
        //float adv = force.y() ;
        //float side = force.x() ;
        //omnirobot_proxy->setSpeedBase(side, adv, rot);
    }
    catch (const Ice::Exception &e){ std::cout << e.what() << std::endl;}

}
////////////////////////////////////////////////////////////////////////
cv::Mat SpecificWorker::read_depth_dreamvu(cv::Mat omni_rgb_frame)
{
    RoboCompCameraRGBDSimple::TDepth omni_depth;       //omni_camera depth comes as RGB
    cv::Mat omni_depth_frame;
    try
    {
        omni_depth = camerargbdsimple_proxy->getDepth("");
        if( not omni_depth.depth.empty())
        {
            omni_depth_frame = cv::Mat(cv::Size(omni_depth.width, omni_depth.height), CV_32FC1, &omni_depth.depth[0], cv::Mat::AUTO_STEP);
            cv::resize(omni_depth_frame, omni_depth_frame, cv::Size(omni_rgb_frame.cols, omni_rgb_frame.rows));
        }
    }
    catch (const Ice::Exception &e)
    { std::cout << e.what() << " Error reading camerargbdsimple_proxy" << std::endl; }
    return omni_depth_frame.clone();
}
cv::Mat SpecificWorker::read_rgb(const std::string &camera_name)
{
    RoboCompCameraRGBDSimple::TImage omni_rgb;
    cv::Mat omni_rgb_frame;
    try
    {
        omni_rgb = camerargbdsimple_proxy->getImage(camera_name);
        qInfo() << __FUNCTION__ << omni_rgb.height << omni_rgb.width;
        if (not omni_rgb.image.empty())
            omni_rgb_frame= cv::Mat(cv::Size(omni_rgb.width, omni_rgb.height), CV_8UC3, &omni_rgb.image[0], cv::Mat::AUTO_STEP);
    }
    catch (const Ice::Exception &e)
    { std::cout << e.what() << " Error reading camerargbdsimple_proxy::getImage" << std::endl;}
    return omni_rgb_frame.clone();
}

cv::Mat SpecificWorker::read_depth_coppelia()
{
    RoboCompCameraRGBDSimple::TImage omni_depth;       //omni_camera depth comes as RGB
    cv::Mat omni_depth_float;
    try
    {
        omni_depth = camerargbdsimple_proxy->getImage("/Shadow/omnicamera/sensorDepth");
        if(not omni_depth.image.empty())
        {
            cv::Mat omni_depth_frame(cv::Size(omni_depth.width, omni_depth.height), CV_8UC3, &omni_depth.image[0], cv::Mat::AUTO_STEP);
            cv::cvtColor(omni_depth_frame, omni_depth_frame, cv::COLOR_RGB2GRAY);
            omni_depth_frame.convertTo(omni_depth_float, CV_32FC1);
        }
    }
    catch(const Ice::Exception &e)
    { std::cout << e.what() << " Error reading camerargbdsimple_proxy::getImage for depth" << std::endl;}
    return omni_depth_float.clone();
}
RoboCompCameraRGBDSimple::TPoints SpecificWorker::read_points_dreamvu(cv::Mat omni_depth_frame)
{
    RoboCompCameraRGBDSimple::TPoints omni_points;
    try
    {
        omni_points = camerargbdsimple_proxy->getPoints("");
        if(omni_points.points.empty())
            qInfo() << "Empty 3D points";
    }
    catch (const Ice::Exception &e){ std::cout << e.what() << " Error connecting to CameraSimple::getPoints" << std::endl;}
    return omni_points;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////77
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
            dist = depth_frame.ptr<float>(u)[v] * consts.dreamvu_depth_scaling_factor;  // pixel to dist scaling factor  -> to mm
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
//            proy = dist * cos( atan2((semi_height - u), 128.f));
//            z = (semi_height - u)/128.f * proy; // 128 focal as PI fov angle for 256 pixels
            float fov = 224.f / tan(qDegreesToRadians(111.f));
            proy = dist * cos( atan2((semi_height - u), fov));
            z = (semi_height - u)/fov * proy; // 128 focal as PI fov angle for 256 pixels
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
    std::vector<vector<Eigen::Vector2f>> copy(lines.begin(), lines.end());
    for(auto &&[k, line]: copy | iter::enumerate)
        for(const auto &p: line)
        {
            auto item = viewer->scene.addEllipse(-10, -10, 20, 20, QPen(QColor(my_color.at(k))), QBrush(QColor(my_color.at(k))));
            item->setPos(robot_polygon->mapToScene(p.x(), p.y()));
            items.push_back(item);
        }
}
void SpecificWorker::draw_3d_points(const RoboCompCameraRGBDSimple::TPoints &scan)    //one vector for each height level
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &item: items)
        viewer->scene.removeItem(item);
    items.clear();

    QPen pen(QColor("orange"));
    QBrush brush(QColor("orange"));
    int cont=0;
    static QStringList my_color = {"red", "orange", "blue", "magenta", "black", "yellow", "brown", "cyan"};

//    std::vector<std::vector<Eigen::Vector2f>> points(int((2000)/200));  //height steps
//    for(auto &p: points)
//        p.resize(360, Eigen::Vector2f(consts.max_camera_depth_range, consts.max_camera_depth_range));   // angular resolution
//    const float ang_bin = 2.0*M_PI/consts.num_angular_bins;
//    for(const auto &p: scan.points)
//    {
//        for(auto &&[level, step] : iter::range(-1000, 1000, 200) | iter::enumerate)
//            if(-p.z*10 > step and -p.z*10 < step+200)
//            {
//                int ang_index = floor((M_PI + atan2(p.x*10, p.y*10)) / ang_bin);
//                Eigen::Vector2f new_point(p.x*10, p.y*10);
//                if(new_point.norm() <  points[level][ang_index].norm())
//                    points[level][ang_index] = new_point;
//            }
//        if(-p.z > 10 and -p.z < 110)
    for(int j=0; j<384; j++)
        for(int i=74; i<76; i++)
        {
            auto item = viewer->scene.addEllipse(-10, -10, 20, 20, QPen(QColor(my_color.at(1))), QBrush(QColor(my_color.at(1))));
            qInfo() << i*384+j;
            item->setPos(robot_polygon->mapToScene(scan.points[j*128+i].x, scan.points[j*128+i].y)); // to mm
            items.push_back(item);
            cont++;
        }
    //draw_floor_line(points);
}
void SpecificWorker::draw_lines_on_image(cv::Mat &rgb, const cv::Mat &depth_frame)
{
    int semi_height = depth_frame.rows/2;
    float dist, z, proy;
    for(int u=0; u<depth_frame.rows; u++)
        for(int v=0; v<depth_frame.cols; v++)
        {
            dist = depth_frame.ptr<float>(u)[v] * consts.dreamvu_depth_scaling_factor;  // pixel to dist scaling factor  -> to mm
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            float fov = (depth_frame.rows/2.f) / tan(qDegreesToRadians(111.f/2.f));   // 111º vertical angle of dreamvu
            proy = dist * cos( atan2((semi_height - u), fov));
            z = (semi_height - u)/fov * proy;  // z is negative
            z += consts.omni_camera_height; // get from DSR
            // add Y axis displacement
            if(z < 0 or z > 100) continue; // filter out floor
            //qInfo() << z << dist;
            cv::circle(rgb, cv::Point(v, u), 1, cv::Scalar(0,255,0));
        };
}
RoboCompGenericBase::TBaseState SpecificWorker::read_robot_state()
{
    RoboCompGenericBase::TBaseState bState;
    try
    {  omnirobot_proxy->getBaseState(bState); }
    catch(const Ice::Exception &e)
    { std::cout << e.what() << " Error reading omnibase_proxy" << std::endl;}
    robot_polygon->setPos(bState.x, bState.z);
    robot_polygon->setRotation(qRadiansToDegrees(bState.alpha));
    return bState;
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

