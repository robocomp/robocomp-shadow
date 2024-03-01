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
	try
	{
		display = params.at("display").value == "True" or params.at("display").value == "true";
        lidar_name = params.at("lidar_name").value;
        dim.setLeft(std::stod(params.at("viewer_left").value));
        dim.setTop(std::stod(params.at("viewer_top").value));
        dim.setWidth(std::stod(params.at("viewer_width").value));
        dim.setHeight(std::stod(params.at("viewer_height").value));
	}
	catch(const std::exception &e) { qFatal("Error reading config params"); }
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
        if(display)
        {
            viewer = new AbstractGraphicViewer(this->frame, dim);
            auto [rob, las] = viewer->add_robot(400, 500, 0, 100, QColor("Blue"));
            robot_polygon = rob;
            viewer->show();
        }
        else
            hide();

        // Lidar thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar_thread,this));
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // reset button
        connect(resetButton, &QPushButton::clicked, [this]()
        {
            qInfo() << "[UI] Reset";
            fastgicp.reset();
            draw_path(true);
            path.clear();
        });

        // pause variable
        last_read.store(std::chrono::high_resolution_clock::now());

        Period = 100;
		timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    /// check idle time
    if(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - last_read.load()).count() > MAX_INACTIVE_TIME)
    {
        fps.print("No requests in the last 5 seconds. Pausing. Reset to continue", 3000);
        return;
    }
    /// read LiDAR
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_source;
    if( auto res = read_lidar(); not res.has_value()) return;
    else pcl_cloud_source = res.value();

    // gets accumulated pose and instantaneous change (first, second)
    auto robot_pose = fastgicp.align(pcl_cloud_source);

    
    buffer_odometry.put(std::move(robot_pose), [](auto &&input, auto &output)
            {

                output.pose.m00 = input.first(0, 0); output.pose.m01 = input.first(0, 1); output.pose.m02 = input.first(0, 2);
                output.pose.m03 = input.first(0, 3); output.pose.m10 = input.first(1, 0); output.pose.m11 = input.first(1, 1);
                output.pose.m12 = input.first(1, 2); output.pose.m13 = input.first(1, 3); output.pose.m20 = input.first(2, 0);
                output.pose.m21 = input.first(2, 1); output.pose.m22 = input.first(2, 2); output.pose.m23 = input.first(2, 3);
                output.pose.m30 = input.first(3, 0); output.pose.m31 = input.first(3, 1); output.pose.m32 = input.first(3, 2);
                output.pose.m33 = input.first(3, 3);

                output.change.m00 = input.second(0, 0); output.change.m01 = input.second(0, 1); output.change.m02 = input.second(0, 2);
                output.change.m03 = input.second(0, 3); output.change.m10 = input.second(1, 0); output.change.m11 = input.second(1, 1);
                output.change.m12 = input.second(1, 2); output.change.m13 = input.second(1, 3); output.change.m20 = input.second(2, 0);
                output.change.m21 = input.second(2, 1); output.change.m22 = input.second(2, 2); output.change.m23 = input.second(2, 3);
                output.change.m30 = input.second(3, 0); output.change.m31 = input.second(3, 1); output.change.m32 = input.second(3, 2);
                output.change.m33 = input.second(3, 3);
            });


    // draw
    if(display)
        draw_robot(robot_pose.first);

    // print
    std::string pose = std::to_string(robot_pose.first.translation().x()*1000) + "mm " +
                       std::to_string(robot_pose.first.translation().y()*1000) + "mm " +
                       std::to_string(robot_pose.first.rotation().eulerAngles(0, 1, 2).z()) + "rad";
    fps.print("Pose: [" + pose + "]", 3000);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<pcl::PointCloud<pcl::PointXYZ>::Ptr> SpecificWorker::read_lidar()
{
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value()) { qWarning() << __FUNCTION__  << "No data"; return {};} // No data
    auto ldata = res_.value();

    // convert points to Eigen
    std::vector<Eigen::Vector3f> points;
    points.reserve(ldata.points.size());
    for (const auto &p: ldata.points)
        points.emplace_back(p.x, p.y, p.z);

    // compute odometry
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_source(new pcl::PointCloud <pcl::PointXYZ>);
    pcl_cloud_source->reserve(ldata.points.size());
    RoboCompLidar3D::TPoints lidar_points;
    for (const auto &[i, p]: ldata.points | iter::enumerate)
        if(p.z > 1000)  // only points above 2m
        {
            pcl_cloud_source->emplace_back(pcl::PointXYZ{p.x / 1000.f, p.y / 1000.f, p.z / 1000.f});
            lidar_points.emplace_back(p);
        };
    return pcl_cloud_source;
}
// Thread to read the lidar
void SpecificWorker::read_lidar_thread()
{
    auto wait_period = std::chrono::milliseconds (this->Period);
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d("helios", 10000, 1); // TODO: move to constants
            // concatenate both lidars
            //data.points.insert(data.points.end(), data_helios.points.begin(), data_helios.points.end());


            //Filter data points wwith z > threshold using a lambda function //TODO: move threshold to config file
            data.points.erase(std::remove_if(data.points.begin(), data.points.end(), [](const RoboCompLidar3D::TPoint &p) { return p.z < 1500; }), data.points.end());

            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysteresis of 2ms
            if (wait_period > std::chrono::milliseconds((long) data.period + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds((long) data.period - 2)) wait_period++;
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
}
void SpecificWorker::draw_lidar(const RoboCompLidar3D::TPoints &points, int decimate)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points) {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_points.clear();

    for (const auto &[i, p]: points |iter::enumerate)
    {
        // skip 2 out of 3 points
        if(i % decimate == 0)
        {
            auto o = viewer->scene.addRect(-20, 20, 40, 40, QPen(QColor("green")), QBrush(QColor("green")));
            o->setPos(p.x, p.y);
            draw_points.push_back(o);
        }
    }
}
void SpecificWorker::draw_path(bool only_clean)
{
    static std::vector<QGraphicsItem *> draw_path;
    for (const auto &p: draw_path) {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_path.clear();

    if(not only_clean)
        for (const auto &[i, p]: path |iter::enumerate)
        {
            auto o = viewer->scene.addRect(-20, 20, 40, 40, QPen(QColor("green")), QBrush(QColor("green")));
            o->setPos(p.x(), p.y());
            draw_path.push_back(o);
        }
}
void SpecificWorker::draw_robot(const Eigen::Isometry3d &robot_pose)
{
    robot_polygon->setPos(robot_pose.translation().x()*1000.0, robot_pose.translation().y()*1000.0);
    robot_polygon->setRotation(robot_pose.rotation().eulerAngles(0, 1, 2).z() * 180.0 / M_PI);
    path.emplace_back(robot_pose.translation().x()*1000, robot_pose.translation().y()*1000);
    if(path.size() > 1000) path.erase(path.begin(), path.begin() + 1);
    draw_path();
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
RoboCompFullPoseEstimation::FullPoseEuler SpecificWorker::LidarOdometry_getFullPoseEuler()
{
    qWarning() << "Not implemented";
    return RoboCompFullPoseEstimation::FullPoseEuler();
}
RoboCompFullPoseEstimation::FullPoseMatrix SpecificWorker::LidarOdometry_getFullPoseMatrix()
{
    last_read.store(std::chrono::high_resolution_clock::now());
    return buffer_odometry.get_idemp().pose;
}
void SpecificWorker::LidarOdometry_reset()
{
    last_read.store(std::chrono::high_resolution_clock::now());
    fastgicp.reset();   // TODO: this might crash without a mutex
}
RoboCompLidarOdometry::PoseAndChange SpecificWorker::LidarOdometry_getPoseAndChange()
{
    last_read.store(std::chrono::high_resolution_clock::now());
    //print buffer_odo
    return buffer_odometry.get_idemp();
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
// From the RoboCompFullPoseEstimation you can use this types:
// RoboCompFullPoseEstimation::FullPoseMatrix
// RoboCompFullPoseEstimation::FullPoseEuler

