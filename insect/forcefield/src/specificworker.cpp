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
#include <cppitertools/filter.hpp>
#include <cppitertools/chunked.hpp>


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
	std::cout << "Initializing worker" << std::endl;
	this->Period = period;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
        // graphics
        viewer = new AbstractGraphicViewer(this->beta_frame,  QRectF(-2500, -2500, 5000, 5000), true);
        this->resize(900,900);
        viewer->add_robot(400, 400);


        //QCustomPlot
        custom_plot.setParent(this->customplot);
        custom_plot.xAxis->setLabel("time");
        custom_plot.yAxis->setLabel("sice_a adv_a track");
        custom_plot.xAxis->setRange(0, 200);
        custom_plot.yAxis->setRange(-500, 500);
        track_err = custom_plot.addGraph();
        track_err->setPen(QColor("blue"));
        side_acc = custom_plot.addGraph();
        side_acc->setPen(QColor("orange"));
        adv_acc = custom_plot.addGraph();
        adv_acc->setPen(QColor("green"));
        custom_plot.resize(this->customplot->size());
        custom_plot.show();

        // room_detector
        room_detector.init(&custom_plot);

        // A thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        // timers
        Period = 50;
        timer.start(Period);
        std::cout << "Worker initialized OK" << std::endl;
	}
}
// cameras, distance-lines, Yolo, eye-tracker, repulsion-forces
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (res_.has_value() == false) {   /*qWarning() << "No data Lidar";*/ return; }
    auto ldata = res_.value();
    //qInfo() << ldata.points.size();
    draw_lidar(ldata);

    ///////////  EARLY DETECTORS  /////////////
    //preobjects.clear();

    /// Door detector
    auto doors = door_detector.detect(ldata.points, &viewer->scene);
    //qInfo() << doors.size();

    //auto pdoor = rc::PreObject::add_doors(doors);
    //preobjects.insert(preobjects.end(), pdoor.begin(), pdoor.end());

    /// Room detector
    //std::vector<Eigen::Vector2f> filtered_line = door_detector.filter_out_points_beyond_doors(door_detector.current_line, doors);
    //draw_floor_line({filtered_line}, {0});
    auto current_room = room_detector.detect({door_detector.current_line}, viewer);
    current_room.print();

    // refresh current_target
//    if (auto it = std::find_if(preobjects.begin(), preobjects.end(), [r = robot](auto &a)
//        { return a.type == r.get_current_target().type; }); it != preobjects.end())
//    {
//        if((it->get_robot_coordinates() - robot.get_current_target().get_robot_coordinates()).norm() < 300)
//            robot.set_current_target(*it);
//    }
//
//    // Move robot
//    robot.goto_target(current_line);

    ///////// DRAWING  //////////////////////
 //   {
//        for (const auto &d: doors)
//            top_camera.project_polygon_3d(d.points, robot.get_tf_base_to_cam(), top_rgb_frame, cv::Scalar(255, 0, 0), "door");
//        top_camera.project_floor(current_room.get_3d_corners_in_robot_coor(), robot.get_tf_base_to_cam(),
//                                      top_rgb_frame, cv::Scalar(0, 255, 0));
////        top_camera.project_walls(current_room.get_3d_corners_in_robot_coor(), robot.get_tf_base_to_cam(),
////                                 top_rgb_frame, cv::Scalar(0, 255, 0));
//
//        /// draw top image
//        cv::imshow("top", top_rgb_frame); cv::waitKey(1);
//        /// draw yolo_objects on 2D view
//        draw_objects_on_2dview(preobjects, rc::PreObject());
//    }

    //qInfo() << t2-t1 << t3-t2 << t4-t3 << t5-t4 << t6-t5 << t8-t7 << t8-t1;
//    static float side_ant = 0, adv_ant = 0;
//    draw_timeseries(side-side_ant, adv-adv_ant, track_err);
//    side_ant = side; adv_ant = adv;
//    draw_timeseries(robot.get_distance_to_target()/10, robot.get_current_advance_speed(), robot.get_current_side_speed()/10);

}

//////////////////// LIDAR /////////////////////////////////////////////////
void SpecificWorker::read_lidar()
{
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarData("bpearl", -90, 360, 1);
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D " << e << std::endl; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));}
}

/////////////////// Draw  /////////////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const RoboCompLidar3D::TData &data)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items)
        viewer->scene.removeItem(i);
    items.clear();

    // draw points
    for(const auto &p: data.points)
    {
        auto item = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20), QBrush(QColor("green")));
        item->setPos(p.x, p.y);
        items.push_back(item);
    }
}

void SpecificWorker::draw_timeseries(float side, float adv, float track)
{
    static int cont = 0;
    side_acc->addData(cont, side);
    track_err->addData(cont, qRadiansToDegrees(track));  // degrees
    adv_acc->addData(cont++, adv);
    custom_plot.xAxis->setRange(cont++, 200, Qt::AlignRight);
    custom_plot.replot();
}

//////////////////////////////////////////////////////////////////////////////
// SUBSCRIPTION to sendData method from JoystickAdapter interface
/////////////////////////////////////////////////////////////////////////////
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    Eigen::Vector2f target_force{0.f, 0.f};
//    for(const auto &axe : data.axes)
//    {
//        if (axe.name == "advance") target_coordinates += Eigen::Vector2f{0.f, axe.value/1000.f};
//        if (axe.name == "rotate") target_coordinates += Eigen::Vector2f{axe.value, 0.f};
//    }
    //set_target_force(target_coordinates);
}
//////////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}


//        int row = (o.top + o.bot)/2;
//        int col = (o.left + o.right)/2;
//        float dist = depth_frame.ptr<float>(row)[col]*1000.f;  //  -> to mm   WARN:  assuming depth and rgb are  the same size
//        // TODO: do not assume depth and rgb are the same size
//        if(std::isnan(dist)) {qWarning() << " Distance value un depth frame coors " << o.x << o.y << "is nan:";};
//        if(dist > consts.max_camera_depth_range or dist < consts.min_camera_depth_range) continue;
//        // compute axis coordinates according to the camera's coordinate system (Y outwards and Z up)
//        float y = dist;
//        float x = (col-(depth_frame.cols/2.f)) * y / focalx;
//        float z = -(row-(depth_frame.rows/2.f)) * y / focaly;


/// compose with target
//if(target_coordinates.norm() > 1000.f) target_coordinates = target_coordinates.normalized()*1000.f;
//Eigen::Vector2f force = rep_force  /*+ target_coordinates*/; // comes from joystick
//    if(fabs(rep_force.norm() - target_coordinates.norm()) < consts.forces_similarity_threshold)  // probable deadlock. Move sideways
//    {
//        qInfo() << __FUNCTION__  << "-------------------------- SIDE FORCE";
//        Eigen::Rotation2Df rot(M_PI_2);
//        force += (rot * target_coordinates.normalized()) * 1000;
//    }
//draw_forces(rep_force, target_coordinates, force); // in robot coordinate system