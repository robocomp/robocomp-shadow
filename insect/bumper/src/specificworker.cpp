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
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/slice.hpp>

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
	this->Period = consts.PERIOD;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		// Viewer
 		viewer = new AbstractGraphicViewer(this->frame, consts.viewer_dim, false);
        //viewer->add_robot(consts.ROBOT_WIDTH, consts.ROBOT_LENGTH, 0, 100, QColor("Blue"));
        viewer->show();
        std::cout << "Started viewer" << std::endl;

        robot_contour << QPointF(-consts.ROBOT_SEMI_WIDTH, consts.ROBOT_SEMI_LENGTH) <<
                      QPointF(consts.ROBOT_SEMI_WIDTH, consts.ROBOT_SEMI_LENGTH) <<
                      QPointF(consts.ROBOT_SEMI_WIDTH, -consts.ROBOT_SEMI_LENGTH) <<
                      QPointF(-consts.ROBOT_SEMI_WIDTH, -consts.ROBOT_SEMI_LENGTH);
        robot_safe_band << QPointF(-consts.ROBOT_SEMI_WIDTH - consts.BAND_WIDTH, consts.ROBOT_SEMI_LENGTH + consts.BAND_WIDTH) <<
                        QPointF(consts.ROBOT_SEMI_WIDTH + consts.BAND_WIDTH, consts.ROBOT_SEMI_LENGTH + consts.BAND_WIDTH) <<
                        QPointF(consts.ROBOT_SEMI_WIDTH + consts.BAND_WIDTH, -consts.ROBOT_SEMI_LENGTH - consts.BAND_WIDTH) <<
                        QPointF(-consts.ROBOT_SEMI_WIDTH - consts.BAND_WIDTH, -consts.ROBOT_SEMI_LENGTH - consts.BAND_WIDTH);

        // create list of edge points (polar) from robot_safe_band
		edge_points = create_edge_points(robot_safe_band);
        //draw_edge(edge_points, &viewer->scene);
        draw_robot_contour(robot_contour, robot_safe_band, &viewer->scene);
        std::cout << __FUNCTION__  << "Robot is drawn" << std::endl;

        // Lidar thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        timer.start(this->Period);
	}
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (res_.has_value() == false) {   /*qWarning() << "No data Lidar";*/ return; }
    auto ldata = res_.value();
    //draw_lidar(ldata.points);
    //qInfo() << ldata.points.size();

    /// Check for new external target
    if(const auto ext = buffer_target.try_get(); ext.has_value())
    {
        const auto &[side, adv, rot, debug] = ext.value();
        target.set(side, adv, rot, true);
        draw_target_original(target, false, 1);
    }

    /// Check bumper for a security breach
    std::vector<Eigen::Vector2f> displacements = check_safety(ldata.points);
    draw_displacements(displacements,&viewer->scene);
    bool security_breach = not displacements.empty();
    //security_breach = false;
    if(not security_breach) draw_target_breach(target, true);

    //////////////////////////////////
    /// We have now four possibilities
    //////////////////////////////////
    // (1) target active and security breach.  Choose displacement best aligned with target
    if(target.active and security_breach)
    {
        qInfo() << "Target active and security breach -------------------------------";
        if (not displacements.empty())
        {
            // we need to find the element that is the minimum in the following cost function: J = target.eigen().transpose * a + a.norm()
            // meaning that it is aligned with the target and has small module
            auto res = std::ranges::max_element(displacements, [t = target](auto &a, auto &b)
            {
                Eigen::Vector2f tv = t.eigen().transpose().normalized();
                return tv.dot(a.normalized())/a.norm() < tv.dot(b.normalized())/b.norm();  //maximum angle scaled by norm
            });
            // float landa = 0.5;
            // *res = (*res)*landa + target.eigen()*(1-landa); // add target to displacement
            reaction.set(res->x(), res->y(), 0.f);
            draw_target_original(target, false, 1);
        }
    }

    // (2) no target and no security breach. Stop robot
    if(not target.active and not security_breach)
    {
        //qInfo() << "NO target active and NOT security breach -------------------------------";
        reaction.active = false;
        stop_robot("No target, no breach");
    }

    // (3) no target and security breach. Choose displacement that maximizes sum of distances to obstacles
    if(not target.active and security_breach) // choose displacement that maximizes sum of distances to obstacles
    {
        qInfo() << "NO target active and security breach -------------------------------";
        if (not displacements.empty())
        {
            // select the minimum displacement that sets the robot free
            auto res = std::ranges::min(displacements,[](auto &a, auto &b)
                { return a.norm() < b.norm(); });
            res *= consts.REPULSION_GAIN;
            reaction.set(res.x(), res.y(), 0.f);
            draw_target_breach(reaction);
        } else  {  stop_robot("Collision but no solution found");  }
    }

    // (4) target and no security breach. Keep going

    // Check if target is zero speed to deactivate target
    if(target.active and target.eigen().norm() < 50.f and fabs(target.rot)<0.1f)
    {
        target.active = false;
        stop_robot();
    }

    // Move the robot
    if(target.active or reaction.active)
        move_robot(target, reaction);

    // Adjust band size
    robot_safe_band = adjust_band_size(robot_current_speed);
    edge_points = create_edge_points(robot_safe_band);
    draw_robot_contour(robot_contour, robot_safe_band, &viewer->scene);
    //fps.print("FPS:");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Checks if there are points inside the bumper and computes a set of safe displacements to avoid them
std::vector<Eigen::Vector2f> SpecificWorker::check_safety(const RoboCompLidar3D::TPoints &points)
{
    // compute reachable positions at max acc that sets the robot free. Choose the one closest to target or of minimum length if not target.
    // get lidar points closer than safety band. If none return
    std::vector<Eigen::Vector2f> displacements;
    // lambda to compute if a point is inside the robot's body
    auto point_in_body = [this](auto ang, auto dist)
          {
              // find the first element in edge_points with angle greater than ang. In edge, x, y -> ang, dist
              auto r = std::upper_bound(edge_points.cbegin(), edge_points.cend(), ang,
                                        [](float value, const Eigen::Vector2f &p) { return p.x() >= value; });
              if (r != edge_points.end() and dist < r->y()) return true;
              else return false;
          };

    // gather all points inside safety belt in the close_points container
    std::vector<Eigen::Vector2f> close_points;  // cartesian coordinates
    for(const auto &p: points)
        if (point_in_body(p.phi, p.r)) close_points.emplace_back(p.x, p.y);
    draw_points_in_belt(close_points); // cartesian coordinates

    // max dist to reach from current situation
    // const float delta_t = 1; // 0.050; //200ms
    // const float MAX_ACC = 300; // mm/sg2
    // double dist = robot_current_speed.head(2).norm() * delta_t + (MAX_ACC * delta_t * delta_t);
    // float MAX_DIST = (MAX_ACC * delta_t * delta_t);

    if(not close_points.empty())
    {
        // iterate over all possible displacements
        for (const float dist: iter::range(0.f, consts.MAX_DIST_TO_LOOK_AHEAD, consts.BELT_LINEAR_STEP))
        {
            for (const double ang: iter::range(-M_PI, M_PI, consts.BELT_ANGULAR_STEP))
            {
                // compute robot's new pos (d*sin(ang), d*cos(ang))
                Eigen::Vector2f t{dist * sin(ang), dist * cos(ang)};
                bool free = true;
                // check if all points are outside the belt
                for (const auto &p: close_points)
                {
                    // translate point to new robot position
                    float dist_p = (p - t).norm();
                    // check if point is inside belt after being moved. If so, break and try next displacement
                    if (point_in_body(atan2((p - t).x(), (p - t).y()), dist_p))
                    {
                        free = false;
                        break;
                    }
                }
                // if all points are outside the belt add displacement to list
                if (free)
                    displacements.emplace_back(t); // add displacement to list
            }
        }
      // Check if the selected displacements induce new points inside the belt
      std::vector<Eigen::Vector2f> final_displacement;
      for (const auto &d : displacements)
      {
          bool success = true;
          Eigen::Vector2f p_cart;
          for(const auto &p: points)
          {
              // translate point to new robot position
              p_cart.x() = p.x - d.x();
              p_cart.y() = p.y - d.y();
              if(point_in_body(atan2(p_cart.x(), p_cart.y()), p_cart.norm()))
              {
                  success = false;
                  break;
              }
          }
          if (success)
              final_displacement.emplace_back(d);
      }

      return final_displacement;
    }
    else return {};
}
// Thread to read the lidar
void SpecificWorker::read_lidar()
{
    auto wait_period = std::chrono::milliseconds (this->Period);
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d(consts.LIDAR_NAME, consts.MAX_LIDAR_RANGE, consts.LIDAR_DECIMATION_FACTOR);
            if(wait_period > std::chrono::milliseconds((long)data.period+consts.PERIOD_HYSTERESIS)) wait_period--;
            else if(wait_period < std::chrono::milliseconds((long)data.period-consts.PERIOD_HYSTERESIS)) wait_period++;
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
}
void SpecificWorker::move_robot(Target &target, const Target &reaction)
{
    // check speed limits
    float t_adv = std::clamp(target.y, -consts.MAX_ADV_SPEED, consts.MAX_ADV_SPEED);
    float t_side = std::clamp(target.x, -consts.MAX_SIDE_SPEED, consts.MAX_SIDE_SPEED);
    float t_rot = std::clamp(target.ang, -consts.MAX_ROT_SPEED, consts.MAX_ROT_SPEED);  // angle wrt axis y
    float r_adv = std::clamp(reaction.y, -consts.MAX_ADV_SPEED, consts.MAX_ADV_SPEED);
    float r_side = std::clamp(reaction.x, -consts.MAX_SIDE_SPEED, consts.MAX_SIDE_SPEED);

    qInfo() << "ROBOT SPEEDS " << t_adv << t_side << t_rot;

    float lambda = 0.3;

    // check activation status of targets and combine results
    if(target.active and not reaction.active)
        robot_current_speed = {t_side, t_adv, t_rot};
    else if(target.active ) // also reaction.active is true
        robot_current_speed = {()lambda * t_side+ (1-lambda) * r_side, lambda * t_adv+(1-lambda) * r_adv, t_rot};
    else if(reaction.active) // also target.active is false
        robot_current_speed = {r_side, r_adv, 0.f};
    else return;  // no targets active

    try
    {
        // TODO: Webots has order changed
        omnirobot_proxy->setSpeedBase(robot_current_speed.x()*0.4 ,
                                      robot_current_speed.y()*0.4 ,
                                      robot_current_speed.z()*0.7);
        robot_stopped = false;
    }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__  << " Error talking to OmniRobot " << e.what() << std::endl; }
}
void SpecificWorker::stop_robot()
{
    try
    {
        // TODO: Webots has order changed
        omnirobot_proxy->setSpeedBase(0.f ,
                                      0.f ,
                                      0.f);
        robot_stopped = true;
    }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__  << " Error talking to OmniRobot " << e.what() << std::endl; }
}
std::vector<Eigen::Vector2f> SpecificWorker::create_edge_points(const QPolygonF &robot_safe_band)
{
    std::vector<Eigen::Vector2f> edges;
    for (const double ang: iter::range(-M_PI, M_PI, consts.BELT_ANGULAR_STEP))
        {
        bool found = false;
        // iter from 0 to OUTER_RIG_DISTANCE until the point falls outside the polygon
        for(const int r : iter::range(consts.OUTER_RIG_DISTANCE))
        {
            double x = r * sin(ang);
            double y = r * cos(ang);
            if( not robot_safe_band.containsPoint(QPointF(x, y), Qt::OddEvenFill))
            {
                edges.emplace_back(ang, r);
                found = true;
                break;
            }
        }
        if(not found) { qFatal("[Create_edge_points] ERROR: Could not find limit for angle ");	}
    }
    return edges;
}
void SpecificWorker::stop_robot(const std::string_view txt)
{
    if(not robot_stopped)
    {
        target.active = false;
        try
        {
            omnirobot_proxy->setSpeedBase(0, 0, 0);
            qInfo() << __FUNCTION__ << "Robot stopped";
            draw_target(target, true);
            robot_current_speed = {0.f, 0.f, 0.f};
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error talking to OmniRobot " << e.what() << std::endl; }
        robot_stopped = true;
        std::cout << "Robot stopped due to " << txt << std::endl;
    }
};
QPolygonF SpecificWorker::adjust_band_size(const Eigen::Vector3f &velocity)
{
    // if advance velocity (y) is positive, make the width of the band proportional to it
    // according to the following constraints: for MAX_ADV_SPEED -> MIN_BAND_WIDTH; for MIN_ADV_SPEED -> MAX_BAND_WIDTH
    float width = -(consts.MAX_BAND_WIDTH - consts.MIN_BAND_WIDTH) * fabs(velocity.y()) / consts.MAX_ADV_SPEED + consts.MAX_BAND_WIDTH;
    float height = -(consts.MAX_BAND_WIDTH - consts.MIN_BAND_WIDTH) * fabs(velocity.x()) / consts.MAX_SIDE_SPEED + consts.MAX_BAND_WIDTH;
    QRectF new_rect{-consts.ROBOT_SEMI_WIDTH - width,
                    consts.ROBOT_SEMI_LENGTH + height,
                    consts.ROBOT_WIDTH + 2 * width,
                    -(consts.ROBOT_LENGTH + 2 * height)};

    return QPolygonF{new_rect};
}
/////////////////////////////////// DRAW ////////////////////////////////////////
void SpecificWorker::draw_edge(const std::vector<Eigen::Vector2f> &edge, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points) {
        scene->removeItem(p);
        delete p;
    }
    draw_points.clear();

    QPolygonF poly;
    for (const auto &e : edge)
        poly << QPointF(e.y() * sin(e.x()), e.y() * cos(e.x()));    // x, y -> ang, dist

    auto o = scene->addPolygon(poly, QPen(QColor("DarkBlue"), 10));
    draw_points.push_back(o);
}
void SpecificWorker::draw_target(const Target &t, bool erase)
{
    static QGraphicsItem * line = nullptr;
    static QGraphicsItem * ball = nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr; }
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr; }
    if(not erase)
    {
        line = viewer->scene.addLine(0, 0, t.x, t.y, QPen(QColor("green"), 20));
        ball = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20));
        ball->setPos(t.x, t.y);
    }
}
void SpecificWorker::draw_target(double x, double y, bool erase)
{
    static QGraphicsItem * line = nullptr;
    static QGraphicsItem * ball = nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr;}
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr;}
    if(not erase)
    {
        line = viewer->scene.addLine(0, 0, x, y, QPen(QColor("green"), 20));
        ball = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20));
        ball->setPos(x, y);
    }
}
void SpecificWorker::draw_target_original(const Target &t, bool erase, float scale)
{
    static QGraphicsItem * line = nullptr;
    static QGraphicsItem * ball = nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr;}
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr;}
    if(not erase)
    {
        //qInfo() << __FUNCTION__ << t.x << t.y;
        line = viewer->scene.addLine(0, 0, t.x*scale, t.y*scale, QPen(QColor("blue"), 20));
        ball = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("blue"), 20));
        ball->setPos(t.x*scale, t.y*scale);
    }
}
void SpecificWorker::draw_target_breach(const Target &t, bool erase)
{
    static QGraphicsItem *line=nullptr, *ball=nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr; }
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr; }
    if(not erase)
    {
        line = viewer->scene.addLine(0, 0, t.x, t.y, QPen(QColor("magenta"), 20));
        ball = viewer->scene.addEllipse(-30, -30, 60, 60, QPen(QColor("magenta"), 20));;
        ball->setPos(t.x, t.y);
    }
}
void SpecificWorker::draw_displacements(std::vector<Eigen::Matrix<float, 2, 1>> displacement_points, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_displacements;
    for(const auto &p : draw_displacements)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_displacements.clear();

    for(auto &d  : displacement_points)
    {
        //d *= 5;
        QLineF l{0,0, d.x(),d.y()};
        auto o = scene->addLine(l, QPen(QColor("magenta"), 8));
        auto o_p = scene->addEllipse(-50,-50,100,100 , QPen(QColor("magenta"), 8));
        o_p->setPos(d.x(),d.y());
        draw_displacements.push_back(o);
        draw_displacements.push_back(o_p);
    }
}
void SpecificWorker::draw_points_in_belt(const std::vector<Eigen::Vector2f> &points_in_belt) // cartesian coordinates
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points) {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_points.clear();

    for (const auto &p: points_in_belt)
    {
        auto o = viewer->scene.addRect(-10, 10, 20, 20, QPen(QColor("red")), QBrush(QColor("red")));
        o->setPos(p.x(), p.y());
        draw_points.push_back(o);
    }
}
void SpecificWorker::draw_lidar(const RoboCompLidar3D::TPoints &points)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points) {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_points.clear();

    for (const auto &p: points)
    {
        auto o = viewer->scene.addRect(-10, 10, 20, 20, QPen(QColor("green")), QBrush(QColor("green")));
        o->setPos(p.x, p.y);
        draw_points.push_back(o);
    }
}
void SpecificWorker::draw_robot_contour(const QPolygonF &robot_contour, const QPolygonF &robot_safe_band, QGraphicsScene *scene)
{
    // draw two polygons: robot_contour and robot_safe_band
    static std::vector<QGraphicsItem *> items;
    for (const auto &p: items) {
        scene->removeItem(p);
        delete p;
    }
    items.clear();
    auto r = scene->addPolygon(robot_contour, QPen(QColor("green"), 15));
    auto s = scene->addPolygon(robot_safe_band, QPen(QColor("orange"), 15));
    items.push_back(r);
    items.push_back(s);
}
//////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}
void SpecificWorker::self_adjust_period(int new_period)
{
    if(abs(new_period - this->Period) < 2 || new_period < 1)      // do it only if period changes
        return;

    if(new_period > this->Period)
    {
        this->Period += 1;
        timer.setInterval(this->Period);
    } else
    {
        this->Period -= 1;
        this->timer.setInterval(this->Period);
    }
}

//////////////////////////// Interfaces //////////////////////////////////////////
/// Implements GridPlanner
/////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan)
{
    if(plan.valid) // Possible failure variable
    {
        buffer_target.put(std::make_tuple(plan.subtarget.x, plan.subtarget.y, 0, false));
    }
}
void SpecificWorker::new_mouse_coordinates(QPointF p)
{
    buffer_target.put(std::make_tuple(p.x(), p.y(), 0, true)); // for debug
}

/////////////////////////////////////////////////////////////////////////////////
// SUBSCRIPTION to sendData method from JoystickAdapter interface
/////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    float side=0.f, adv=0.f, rot=0.f;
    // Take joystick data as an external. It comes in m/sg, so we need to scale it to mm/s
    for (const auto &axis : data.axes)
    {
        if(axis.name == "rotate")
            rot = axis.value;
        else if (axis.name == "advance")
            adv = axis.value;
        else if (axis.name == "side")
            side = -axis.value;
        else
            cout << "[ JoystickAdapter ] Warning: Using a non-defined axes (" << axis.name << ")." << endl;
    }
    buffer_target.put(std::make_tuple(side*1000.f, adv*1000.f, rot, false));
}


////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//  TODO: Divide into methods draw_polar_graph and draw_cartesian_graph

//  Draw the Gpoints vector in cartesian coordinates angles/distance
//    for(const auto &[ang, dist] : gpoints)
//    {
//        cv::Point pt1(ang * scaleXcartesian, height - dist * scaleYcartesian);
//        cv::circle(graphCartesian, pt1, 2, cv::Scalar(0, 0, 255),-8);
//    }
// Show cartesian graph
//const float scaleXcartesian = width / 360.0;
//const float scaleYcartesian = height / max_distance;
//cv::namedWindow("GraphCartesian", cv::WINDOW_AUTOSIZE);
//cv::imshow("GraphCartesian", graphCartesian);

//void SpecificWorker::draw_histogram(const RoboCompLidar3D::TPoints &ldata)
//{
//    // Draw a white image one for each graph
//    const int width = 400, height = 400;
//    cv::Mat graphPolar = cv::Mat::zeros(height, width, CV_8UC3) + cv::Scalar(255, 255, 255);
//
//    // Max_distance of the laser used for scaling
//    const float max_distance = 4000.0; //mm
//
//    // scale factors
//
//    const float scaleX = width / (2 * max_distance);  // Asumiendo que el ángulo varía de 0 a 360
//    const float scaleY = height / (2 * max_distance);
//
//    for(auto &ps : ldata )
//    {
//        cv::Point p{static_cast<int>(width/2.0 - (ps.r * scaleX)), static_cast<int>(height/2.0 - (ps.r * scaleY))};
//        cv::circle(graphPolar, p, 3, cv::Scalar(255,0,0));
//    }
//    cv::line(graphPolar, cv::Point{0, height/2}, cv::Point{width, height/2}, cv::Scalar(0, 255, 0), 1);
//    cv::line(graphPolar, cv::Point{width/2, 0}, cv::Point{width/2, height}, cv::Scalar(0, 255, 0), 1);
//    cv::namedWindow("GraphPolar", cv::WINDOW_AUTOSIZE);
//    cv::imshow("GraphPolar", graphPolar);
//    cv::waitKey(1);
//}
//void SpecificWorker::draw_histogram(const std::vector<std::tuple<float, float>> &pdata)
//{
//    // Draw a white image one for each graph
//    const int width = 400, height = 400;
//    cv::Mat graphPolar = cv::Mat::zeros(height, width, CV_8UC3) + cv::Scalar(255, 255, 255);
//
//    // Max_distance of the laser used for scaling
//    const float max_distance = 1500.0; //mm
//
//    // scale factors
//
//    const float scaleXpolar = width / (2 * max_distance);  // Asumiendo que el ángulo varía de 0 a 360
//    const float scaleYpolar = height / (2 * max_distance);
//
//    //  Draw the Gpoints vector in polar coordinates
//    for(auto &&ps : pdata | iter::sliding_window(2))
//    {
//        const auto &[d1, ang1] = ps[0];
//        cv::Point p1{ static_cast<int>(width/2.0 - (d1 * scaleXpolar * std::sin(qDegreesToRadians((float)ang1)))),
//                      static_cast<int>(height/2.0 - (d1 * scaleYpolar * std::cos(qDegreesToRadians((float)ang1))))};
//        const auto &[d2, ang2] = ps[1];
//        cv::Point p2{ static_cast<int>(width/2.0 - (d2 * scaleXpolar * std::sin(qDegreesToRadians((float)ang2)))),
//                      static_cast<int>(height/2.0 - (d2 * scaleYpolar * std::cos(qDegreesToRadians((float)ang2))))};
//
//        cv::line(graphPolar, p1, p2, cv::Scalar(255, 0, 0), 2);
//    }
//    cv::line(graphPolar, cv::Point{0, height/2}, cv::Point{width, height/2}, cv::Scalar(0, 255, 0), 1);
//    cv::line(graphPolar, cv::Point{width/2, 0}, cv::Point{width/2, height}, cv::Scalar(0, 255, 0), 1);
//    cv::namedWindow("GraphPolar", cv::WINDOW_AUTOSIZE);
//    cv::imshow("GraphPolar", graphPolar);
//    cv::waitKey(1);
//}



//qInfo() << "Result norm from last point" << result.norm();
// TODO: Modify to one try catch calling setSpeedBase and only modify adv and side variables
//    if(result.norm() > 3)    // a clean bumper should be zero
//    {
//        // Use std::clamp to ensure that the value of side is within the range [-value, value]
//        robot_speed.adv_speed = std::clamp(result.x() * x_gain,-max_adv,max_adv);
//        robot_speed.side_speed = std::clamp(result.y() * y_gain,-max_side,max_side);
//        robot_speed.rot_speed = 0.0f;
//
//        robot_stop = false;
//    }
//    else

//#if DEBUG
//    auto start = std::chrono::high_resolution_clock::now();
//#endif
//#if DEBUG
//        qInfo() << "Post get_lidar_data" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        start = std::chrono::high_resolution_clock::now();
//    #endif
//    #if DEBUG
//        qInfo() << "Post result" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        start = std::chrono::high_resolution_clock::now();
//    #endif
//    #if DEBUG
//        qInfo() << "Post sending adv, side, rot" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        start = std::chrono::high_resolution_clock::now();
//    #endif
//    #if DEBUG
//        qInfo() << "Post draw_all_points" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        qInfo() << "";
//    #endif

//std::tuple<float, float> SpecificWorker::cost_function(const std::vector<std::tuple<float, float>> &points, const Target &target)
//{
//    std::vector<std::tuple<float, float>> p_costs(points);
//    auto angle_diff = [](auto a, auto b){ return atan2(sin(a - b), cos(a - b));};
//    const float k1 = 10.f; const float k2 = 1.f; const float k3 = 0.001;
//
//    for(auto &[ang, dist] : p_costs)
//    {
//        float hg = angle_diff(p.ang, target.ang);
//        float ho = fabs(p.ang);
//        //Eigen::Vector2f d{p.dist*sin(p.ang), p.dist*cos(p.ang)};
//        //float proy = (robot_current_speed.transpose() * d ) ;
//        //proy = proy / d.norm();
//        //if(proy < d.norm())
//        //if(not blocks[p.block].concave)
//        {
//            //p.coste = (blocks[p.block].dist() / ( k1 * hg + k2 * ho + k3));
//            p.coste = (p.dist / ( k1 * hg + k2 * ho + k3));
//        }
//    }
//    LPoint max_point = std::ranges::max_element(p_costs, [](auto &a, auto &b){ return a.coste > b.coste;}).operator*();
//    max_point.dist *= 0.8;  // to avoid being on the border
//    return  max_point;
//}

/// Check for debug target (later for LOST target)
//    if(target_ext.active and target_ext.debug)
//    {
//        // How is target in world coordinates seen from the robot's frame
//        Eigen::Vector4d target_in_robot =
//                robot_pose.inverse().matrix() * Eigen::Vector4d(target_original.x / 1000.f, target_original.y / 1000.f, 0.f, 1.f) * 1000;
//        target_ext.set(target_in_robot(0)/2, target_in_robot(1)/2, 0, true);
//
//        // Check if the robot is at target
//        qInfo() << __FUNCTION__ << "Dist to target:" << target_in_robot.norm();
//        if (target_in_robot.norm() < 600)
//        {
//            stop_robot("Robot arrived to target");
//            target_ext.debug = false;
//            target_ext.active = false;
//            draw_target(target, true);
//        }
//    }
