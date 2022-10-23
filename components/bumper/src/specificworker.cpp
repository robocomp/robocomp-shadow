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
        //viewer->draw_contour();

        // global
        IS_COPPELIA = true;

        try
        { jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition(0, 1));}
        catch(const Ice::Exception &e){ std::cout << e.what() << std::endl; return;}

        timer.start(50);
	}
}

void SpecificWorker::compute()
{
    wtimer.tick();

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

    /// top camera
    auto top_rgb_frame = read_rgb("/Shadow/camera_top");

    /// YOLO
    RoboCompYoloObjects::TObjects people = yolo_detect_people(top_rgb_frame);
    cv::imshow("top", top_rgb_frame); cv::waitKey(1);
    
    /// / update leader
    Eigen::Vector2f target_force(0.f, 0.f);
    auto [active_leader, leader, tf] = update_leader(people);
    target_force = tf;
    draw_humans(people, leader);

    /// eye tracking
    eye_track(active_leader, leader);

    /// compute level_lines
    auto lines = get_multi_level_3d_points(omni_depth_frame);
    draw_floor_line(lines, 1);

    /// lidar leg detector
    auto legs = leg_detector(lines[1]);
    //draw_legs(legs);

    /// remove leader from detected legs
    remove_leader_from_detected_legs(legs, leader);

    /// add repulsion force form detected legs;
    std::vector<Eigen::Vector2f> legs_forces;
    for(const auto &l: legs)
        legs_forces.emplace_back(Eigen::Vector2f(l.x, l.y));
    lines[1].insert(lines[1].end(), legs_forces.begin(), legs_forces.end());

    /// remove points belonging to leader
    remove_lidar_points_from_leader(lines[1], leader);

    /// potential field algorithm
    Eigen::Vector2f rep_force = compute_repulsion_forces(lines[1]);

    /// compose with target
    Eigen::Vector2f force = rep_force + target_force;
    draw_forces(rep_force, target_force, force); // in robot coordinate system
    auto sigmoid = [](auto x){ return std::clamp(x / 1000.f, 0.f, 1.f);};
    try
    {
        Eigen::Vector2f gains{0.8, 0.8};
        force = force.cwiseProduct(gains);
        float rot = atan2(force.x(), force.y()) * sigmoid(force.norm()) - 0.9*current_servo_angle;  // dumps rotation for small resultant force
        float adv = force.y() ;
        float side = force.x() ;
        omnirobot_proxy->setSpeedBase(side, adv, rot);
    }
    catch (const Ice::Exception &e){ std::cout << e.what() << std::endl;}

    //wtimer.tock();
    //qInfo() << "Elapsed:" << wtimer.duration();
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
Eigen::Vector2f SpecificWorker::compute_repulsion_forces(vector<Eigen::Vector2f> &floor_line)
{
    Eigen::Vector2f res = {0.f, 0.f};
    const float th_distance = 1500;
    for(const auto &ray: floor_line)
    {
        if (ray.norm() < th_distance)
            res -= ray.normalized() / pow(ray.norm()/th_distance, 4);
        else
            res -= ray.normalized() / pow(ray.norm()/th_distance, 2);
    }
    return res;
    //return std::accumulate(floor_line.begin(), floor_line.end(), Eigen::Vector2f{0.f, 0.f},[](auto a, auto b){return a -b.normalized()/b.norm();});
}
std::vector<std::vector<Eigen::Vector2f>> SpecificWorker::get_multi_level_3d_points(const cv::Mat &depth_frame)
{
    std::vector<std::vector<Eigen::Vector2f>> points(int((1550-350)/100));  //height steps
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
            if(IS_COPPELIA)
                dist = depth_frame.ptr<float>(u)[v] * consts.coppelia_depth_scaling_factor;  // pixel to dist scaling factor  -> to mm
            else
                dist = depth_frame.ptr<float>(u)[v] * consts.dreamvu_depth_scaling_factor;  // pixel to dist scaling factor  -> to mm
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
            float fov;
            if(IS_COPPELIA)
                fov = 128;
            else
                fov = (depth_frame.rows / 2.f) / tan(qDegreesToRadians(111.f / 2.f));   // 111º vertical angle of dreamvu

            proy = dist * cos(atan2((semi_height - u), fov));
            z = (semi_height - u) / fov * proy;
            z += consts.omni_camera_height; // get from DSR
            // add Y axis displacement

            if(z < 0) continue; // filter out floor

            // add only to its bin if less than current value
            for(auto &&[level, step] : iter::range(350, 1550, 100) | iter::enumerate)
                if(z > step and z < step+100 )
                {
                    int ang_index = floor((M_PI + atan2(x, y)) / ang_bin);
                    Eigen::Vector2f new_point(x, y);
                    if(new_point.norm() <  points[level][ang_index].norm() and new_point.norm() > 400)
                        points[level][ang_index] = new_point;
                }
        };
    return points;
}
void SpecificWorker::eye_track(bool active_person, const RoboCompYoloObjects::TBox &person_box)
{
    static float error_ant = 0.f;
    if(active_person)
    {
        float hor_angle = atan2(person_box.x, person_box.y);  // angle wrt camera origin
        current_servo_angle = jointmotorsimple_proxy->getMotorState("camera_pan_joint").pos;
        if (std::isnan(current_servo_angle) or std::isnan(hor_angle))
        { qWarning() << "NAN value in servo position";  return; }

        if( fabs(hor_angle) > consts.max_hor_angle_error)  // saccade
        {
            qInfo() << __FUNCTION__ << "saccade" << hor_angle;
            try
            {
                float error = 0.5 * (current_servo_angle - hor_angle);
                float new_angle = std::clamp(error, -1.f, 1.f);  // dumping
                jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition(new_angle, 1));
            }
            catch (const Ice::Exception &e)
            {  std::cout << e.what() << " Error connecting to MotorGoalPosition" << std::endl; return; }
        }
        else    // smooth pursuit
        {
            try
            {
                float new_vel = -0.7 * hor_angle + 0.3 * (hor_angle - error_ant);
                new_vel = std::clamp(new_vel, -1.f, 1.f);  // dumping
                jointmotorsimple_proxy->setVelocity("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalVelocity(new_vel, 1));
                qInfo() << __FUNCTION__ << "smooth" << hor_angle << current_servo_angle << new_vel;
            }
            catch (const Ice::Exception &e)
            {  std::cout << e.what() << " Error connecting to MotorGoalPosition" << std::endl; return; }
        }
    }
    else  // inhibition of return
        try
        { jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition(0, 1)); }
        catch(const Ice::Exception &e){ std::cout << e.what() << std::endl; return;}

}
RoboCompLegDetector2DLidar::Legs SpecificWorker::leg_detector(vector<Eigen::Vector2f> &lidar_line)
{
    RoboCompLaser::TLaserData ldata;
    for(const auto &l: lidar_line)
        ldata.emplace_back(RoboCompLaser::TData{atan2(l.x(),l.y()), l.norm()});
    RoboCompLegDetector2DLidar::Legs legs;
    try
    {
        legs = legdetector2dlidar_proxy->getLegs(ldata);
        for(auto &l : legs)
        {
            l.x *= 1000.f;
            l.y *= 1000.f;
        }
    }
    catch(const Ice::Exception &e){ std::cout << e.what() << " Error connecting to LegDetector" << std::endl;}
    return legs;
}
RoboCompYoloObjects::TObjects SpecificWorker::yolo_detect_people(cv::Mat rgb)
{
    RoboCompYoloObjects::TObjects people;
    RoboCompYoloObjects::TData yolo_objects;
    try
    { yolo_objects = yoloobjects_proxy->getYoloObjects(); }
    catch(const Ice::Exception &e){ std::cout << e.what() << std::endl; return people;}

    Eigen::Vector2f target_force = {0.f, 0.f};
    bool active_person = false;
    RoboCompYoloObjects::TBox person_box;
    for(const auto &o: yolo_objects.objects)
        if(o.type == 0)
        {
            people.push_back(o);
            cv::rectangle(rgb, cv::Point(o.left, o.top), cv::Point(o.right, o.bot), cv::Scalar(0, 255, 0), 3);
        }
    return people;
}
std::tuple<bool, RoboCompYoloObjects::TBox, Eigen::Vector2f> SpecificWorker::update_leader(RoboCompYoloObjects::TObjects &people)
{
    static RoboCompYoloObjects::TBox leader;
    static bool active_leader = false;
    auto get_force = [](const RoboCompYoloObjects::TBox &b)
                {   Eigen::Vector2f target_force{leader.x, leader.y};
                    if(target_force.norm() < 1300) target_force = {0.f, 0.f};
                    if(target_force.norm() > 3500) target_force = target_force.normalized()*3000;
                    target_force /= 2;
                    return target_force;
                };
    Eigen::Vector2f target_force = {0.f, 0.f};
    if(not active_leader)
    {
        if(people.empty()) return std::make_tuple(false, RoboCompYoloObjects::TBox(), Eigen::Vector2f(0.f,0.f));
        leader = people.front();
        active_leader = true;
        return std::make_tuple(true, leader, get_force(leader));
    }
    // update: check if any in people matches leader
    if(people.empty()) return std::make_tuple(true, leader, get_force(leader));
    std::vector<float> matches;
    for(const auto &person: people)
        matches.push_back(iou(leader, person));
    // get them closest match
    auto min = std::ranges::min_element(matches);
    int ind = std::distance(matches.begin(), min);
    leader = people[ind];
    // remove leader from people
    people.erase(people.begin()+ind);
    return std::make_tuple(true, leader, get_force(leader));
}
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
    return  area_of_intersection / area_of_union;
}
void SpecificWorker::remove_leader_from_detected_legs(RoboCompLegDetector2DLidar::Legs &legs, const RoboCompYoloObjects::TBox &leader)
{
    std::vector<float> leg_matches;
    for(const auto &l: legs)
    {
        RoboCompYoloObjects::TBox leg_box{.left=(int)l.x-200, .top=(int)l.y-200, .right=(int)l.x+200, .bot=(int)l.y+200};
        leg_matches.push_back(iou(leader, leg_box));
    }
    // get the closest match
    if(auto min = std::ranges::min_element(leg_matches); min != leg_matches.end())
        if(*min > 0.5)
            leg_matches.erase(leg_matches.begin() + std::distance(leg_matches.begin(), min));
}
void SpecificWorker::remove_lidar_points_from_leader(std::vector<Eigen::Vector2f> line, const RoboCompYoloObjects::TBox &leader)
{
    QRectF rect(leader.left, leader.right, leader.right-leader.left, leader.bot-leader.top);
    float delta = 0;
    rect.adjust(-delta, -delta, delta, delta);
    QPolygonF pol(rect);
    line.erase(std::remove_if(line.begin(), line.end(), [pol](auto p) { return pol.containsPoint(QPointF(p.x(),p.y()), Qt::OddEvenFill);}), line.end());
}
////////////////////////////////////////
void SpecificWorker::draw_floor_line(const vector<vector<Eigen::Vector2f>> &lines, int i)    //one vector for each height level
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &item: items)
        viewer->scene.removeItem(item);
    items.clear();

    QPen pen(QColor("orange"));
    QBrush brush(QColor("orange"));

    static QStringList my_color = {"red", "orange", "blue", "magenta", "black", "yellow", "brown", "cyan"};
    std::vector<vector<Eigen::Vector2f>> copy;
    if(i == -1)
        copy.assign(lines.begin(), lines.end());
    else
        copy.assign(lines.begin()+i, lines.begin()+i+1);

    for(auto &&[k, line]: copy | iter::enumerate)
        for(const auto &p: line)
        {
            auto item = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor(my_color.at(k))), QBrush(QColor(my_color.at(k))));
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
void SpecificWorker::draw_forces(const Eigen::Vector2f &force, const Eigen::Vector2f &target, const Eigen::Vector2f &res)
{
    static QGraphicsItem* item1=nullptr;
    static QGraphicsItem* item2=nullptr;
    static QGraphicsItem* item3=nullptr;
    if(item1 != nullptr) viewer->scene.removeItem(item1);
    if(item2 != nullptr) viewer->scene.removeItem(item2);
    if(item2 != nullptr) viewer->scene.removeItem(item3);
    delete item1; delete item2; delete item3;

    auto large_force = force * 3.f;
    QPointF tip1 = robot_polygon->mapToScene(large_force.x(), large_force.y());
    QPointF tip2 = robot_polygon->mapToScene(target.x(), target.y());
    QPointF tip3 = robot_polygon->mapToScene(res.x(), res.y());
    item1 = viewer->scene.addLine(robot_polygon->pos().x(), robot_polygon->pos().y(), tip1.x(), tip1.y(), QPen(QColor("red"), 50));
    item2 = viewer->scene.addLine(robot_polygon->pos().x(), robot_polygon->pos().y(), tip2.x(), tip2.y(), QPen(QColor("blue"), 50));
    item3 = viewer->scene.addLine(robot_polygon->pos().x(), robot_polygon->pos().y(), tip3.x(), tip3.y(), QPen(QColor("green"), 50));
}
void SpecificWorker::draw_humans(RoboCompYoloObjects::TObjects objects, const RoboCompYoloObjects::TBox &leader)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items)
        viewer->scene.removeItem(i);
    items.clear();

    // delete leader from objects before drawing
    // objects.erase(std::remove_if(objects.begin(), objects.end(), [leader](auto o) { iui == obj; }));
    float tilt = qDegreesToRadians(20.f);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(tilt, Eigen::Vector3f::UnitX());
    m = m * Eigen::AngleAxisf(current_servo_angle, Eigen::Vector3f::UnitZ());
    // draw leader
    auto item = viewer->scene.addRect(-200, -200, 400, 400, QPen(QColor("green"), 20));
    Eigen::Vector2f corrected = (m * Eigen::Vector3f(leader.x, leader.y, leader.z)).head(2);
    item->setPos(corrected.x(), corrected.y());
    items.push_back(item);
    // draw rest
    for(const auto &o: objects)
    {
        auto item = viewer->scene.addRect(-200, -200, 400, 400, QPen(QColor("blue"), 20));
        Eigen::Vector2f corrected = (m * Eigen::Vector3f(o.x, o.y, o.z)).head(2);
        item->setPos(corrected.x(), corrected.y());
        items.push_back(item);
    }
}
void SpecificWorker::draw_legs(RoboCompLegDetector2DLidar::Legs legs)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items)
        viewer->scene.removeItem(i);
    items.clear();

    for(const auto &l: legs)
    {
        auto item = viewer->scene.addRect(-200,-200, 400, 400, QPen(QColor("magenta"), 20));
        // compensate angle as R * (ox,oy)
        const float &a = current_servo_angle;
        Eigen::Matrix2f rot; rot << cos(a), -sin(a), sin(a), cos(a);
        Eigen::Vector2f corrected = rot * Eigen::Vector2f(l.x, l.y);
        item->setPos(corrected.x(), corrected.y());
        items.push_back(item);
    }
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

