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
        viewer->add_robot(robot.width, robot.length);

        // initialize robot
        robot.initialize(omnirobot_proxy, viewer);
        //robot.add_camera(tf, {"z"}, jointmotorsimple_proxy);
        //robot.create_bumper();  // create bumper

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

        // rep state machine
        sm_search_and_approach.init(this->graph_frame);

        // A thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        // timers
        Period = 50;
        timer.start(Period);
        //connect(&timer2, SIGNAL(timeout()), this, SLOT(compute2()));
        connect(&timer2, SIGNAL(timeout()), this, SLOT(compute_explore_rooms()));
        //timer2.start(200);
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

    //auto pdoor = rc::PreObject::add_doors(doors);
    //preobjects.insert(preobjects.end(), pdoor.begin(), pdoor.end());

    /// Room detector
    //std::vector<Eigen::Vector2f> filtered_line = door_detector.filter_out_points_beyond_doors(current_line, doors);
    //auto current_room = room_detector.detect({filtered_line}, viewer);   // inside, index has to be 0

    // refresh current_target
    if (auto it = std::find_if(preobjects.begin(), preobjects.end(), [r = robot](auto &a)
        { return a.type == r.get_current_target().type; }); it != preobjects.end())
    {
        if((it->get_robot_coordinates() - robot.get_current_target().get_robot_coordinates()).norm() < 300)
            robot.set_current_target(*it);
    }

    // Move robot
    robot.goto_target(current_line);

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

///////////////////////////////////////////////////////////////////////////////////////////
// mission: explore and build space
void SpecificWorker::compute_explore_rooms()
{
    // state machine to activate basic behaviours. Returns a  target_coordinates vector
    auto state = sm_search_and_approach.update(robot, preobjects, yolo_object_names);
}
// mission: follow person
void SpecificWorker::compute_follow_human()
{
    for(auto &&o: preobjects)
        o.print();

    for(const auto &o: preobjects)
        if(o.type == 0) // human
            robot.set_current_target(o);

}
//////////////////// ELEMENTS OF CONTROL/////////////////////////////////////////////////
// perception
// Thread to read the lidar
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
RoboCompYoloObjects::TObjects SpecificWorker::yolo_detect_objects(cv::Mat rgb)
{
    RoboCompYoloObjects::TObjects objects;
    RoboCompYoloObjects::TData yolo_objects;
    try
    { yolo_objects = yoloobjects_proxy->getYoloObjects(); }
    catch(const Ice::Exception &e){ std::cout << e.what() << std::endl; return objects;}

    // remove unwanted types
    yolo_objects.objects.erase(std::remove_if(yolo_objects.objects.begin(), yolo_objects.objects.end(), [names = yolo_object_names](auto p)
    { return names[p.type] != "person" and
             names[p.type] != "chair" and
             names[p.type] != "potted plant" and
             names[p.type] != "toilet" and
             names[p.type] != "sink" and
             names[p.type] != "refrigerator" and
             names[p.type] != "tv"; }), yolo_objects.objects.end());
    // draw boxes
    for(auto &&o: yolo_objects.objects | iter::filter([th = consts.yolo_threshold](auto &o){return o.score > th;}))
    {
        objects.push_back(o);
        auto tl = round(0.002 * (rgb.cols + rgb.rows) / 2) + 1; // line / fontthickness
        const auto &c = COLORS.row(o.type);
        cv::Scalar color(c.x(), c.y(), c.z()); // box color
        cv::Point c1(o.left, o.top);
        cv::Point c2(o.right, o.bot);
        cv::rectangle(rgb, c1, c2, color, tl, cv::LINE_AA);
        int tf = (int) std::max(tl - 1, 1.0);  // font thickness
        int baseline = 0;
        std::string label = yolo_object_names.at(o.type) + " " + std::to_string((int) (o.score * 100)) + "%";
        auto t_size = cv::getTextSize(label, 0, tl / 3.f, tf, &baseline);
        c2 = {c1.x + t_size.width, c1.y - t_size.height - 3};
        cv::rectangle(rgb, c1, c2, color, -1, cv::LINE_AA);  // filled
        cv::putText(rgb, label, cv::Size(c1.x, c1.y - 2), 0, tl / 3, cv::Scalar(225, 255, 255), tf, cv::LINE_AA);
    }
    return objects;
}

// control
Eigen::Vector2f SpecificWorker::compute_repulsion_forces(vector<Eigen::Vector2f> &floor_line)
{
    // update threshold with speed
    //    if( fabs(robot.current_adv_speed) > 10.f)
    //        consts.dynamic_threshold = consts.quadratic_dynamic_threshold_coefficient * (robot.current_adv_speed * robot.current_adv_speed);
    //    else
    //        consts.dynamic_threshold = robot.width;
    //qInfo() << __FUNCTION__ << consts.dynamic_threshold << robot.current_adv_speed  << "[" << target_coordinates.x() << target_coordinates.y()  << "]";

    //  computation in meters to reduce the size of the numbers
    Eigen::Vector2f res = {0.f, 0.f};
    float threshold = consts.dynamic_threshold/1000.f;   // to meters
    float max_dist = consts.max_distance_for_repulsion/1000.f;
    for(const auto &ray: floor_line)
    {
        const float &dist = (ray/1000.f).norm();
        if (dist <= threshold)
            res += consts.nu * (1.0 / dist - 1.0 / max_dist) * (1.0 / (dist * dist)) * (-(ray/1000.f) / dist);  // as in original paper
    }
    return res*1000.f; //mm
}
void SpecificWorker::set_target_force(const Eigen::Vector3f &vec)
{
    target_coordinates = vec * 1000;  //to mm/sg
}
void SpecificWorker::move_robot(Eigen::Vector2f force)
{
    //auto sigmoid = [](auto x){ return std::clamp(x / 1000.f, 0.f, 1.f);};
    try
    {
        Eigen::Vector2f gains{0.8, 0.8};
        force = force.cwiseProduct(gains);
        float rot = atan2(force.x(), force.y())  - 0.9*current_servo_angle;  // dumps rotation for small resultant force
        float adv = force.y() ;
        float side = force.x() ;
        qInfo() << __FUNCTION__ << side << adv << rot;
        omnirobot_proxy->setSpeedBase(side, adv, rot);
    }
    catch (const Ice::Exception &e){ std::cout << e.what() << std::endl;}
}

///////////////////// Aux //////////////////////////////////////////////////////////////////
float SpecificWorker::closest_distance_ahead(const std::vector<Eigen::Vector2f> &line)
{
    // minimum distance in central sector
    if(line.empty()) { qWarning() << __FUNCTION__ << "Empty line vector"; return 0.f;}
    size_t offset = 3*line.size()/7;
    auto res = std::min_element(line.begin()+offset, line.end()-offset, [](auto &a, auto &b){ return a.norm() < b.norm();});
    return res->norm();

}
//// IOU auxiliary function
float SpecificWorker::iou(const RoboCompYoloObjects::TBox &a, const RoboCompYoloObjects::TBox &b)
{
    // coordinates of the area of intersection.
    float ix1 = std::max(a.left, b.left);
    float iy1 = std::max(a.top, b.top);
    float ix2 = std::min(a.right, b.right);
    float iy2 = std::min(a.bot, b.bot);

    // Intersection height and width.
    float i_height = std::max(iy2 - iy1 + 1, 0.f);
    float i_width = std::max(ix2 - ix1 + 1, 0.f);

    float area_of_intersection = i_height * i_width;

    // Ground Truth dimensions.
    float a_height = a.bot - a.top + 1;
    float a_width = a.right - a.left + 1;

    // Prediction dimensions.
    float b_height = b.bot - b.top + 1;
    float b_width = b.right - b.left + 1;

    float area_of_union = a_height * a_width + b_height * b_width - area_of_intersection;
    return area_of_intersection / area_of_union;
}
float SpecificWorker::gaussian(float x)
{
    const double xset = consts.xset_gaussian;
    const double yset = consts.yset_gaussian;
    const double s = -xset*xset/log(yset);
    return exp(-x*x/s);
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
void SpecificWorker::draw_floor_line(const std::vector<std::vector<Eigen::Vector2f>> &lines, std::initializer_list<int> list)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &item: items)
        viewer->scene.removeItem(item);
    items.clear();

    if(list.size() > lines.size()) {qWarning()<< "Requested list bigger than data. Returning"; return;}
    std::vector<vector<Eigen::Vector2f>> copy_of_line(list.size());
    for(auto &&[i, e]: list|iter::enumerate)
        copy_of_line[i] =  lines[e];

    for(auto &&[k, line]: copy_of_line | iter::enumerate)
    {
        //qInfo() << __FUNCTION__ << k << (int)COLORS.row(k).x() << (int)COLORS.row(k).y() << (int)COLORS.row(k).z();
        QColor color((int)COLORS.row(k).x(), (int)COLORS.row(k).y(), (int)COLORS.row(k).z());
        QBrush brush(color);
        for(const auto &p: line)
        {
            auto item = viewer->scene.addEllipse(-20, -20, 40, 40, color, brush);
            item->setPos(p.x(), p.y());
            items.push_back(item);
        }
    }
}
void SpecificWorker::draw_objects_on_2dview(const std::vector<rc::PreObject> &objects, const rc::PreObject &selected)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items)
        viewer->scene.removeItem(i);
    items.clear();

    // draw selected
//    auto item = viewer->scene.addRect(-200, -200, 400, 400, QPen(QColor("green"), 20));
//    Eigen::Vector2f corrected = (m * Eigen::Vector3f(selected.x, selected.y, selected.z)).head(2);
//    item->setPos(corrected.x(), corrected.y());
//    items.push_back(item);

    // draw rest
    for(const auto &o: objects)
    {
        const auto &c = COLORS.row(o.type);
        QColor color(c.z(), c.y(), c.x());  //BGR
        auto item = viewer->scene.addRect(-200, -200, 400, 400, QPen(color, 20));

        Eigen::Vector2f corrected = (robot.get_tf_cam_to_base() * Eigen::Vector3f(o.x, o.y, o.z)).head(2);
        item->setPos(corrected.x(), corrected.y());
        items.push_back(item);
        //Eigen::Vector3f yolo = robot.get_tf_cam_to_base() * Eigen::Vector3f(o.x, o.y, o.z);
        //qInfo() << __FUNCTION__ << corrected.x() << corrected.y() << yolo.x() << yolo.y();
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

///
// SUBSCRIPTION to sendData method from JoystickAdapter interface
///
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