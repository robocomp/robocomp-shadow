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
#include <cppitertools/filter.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/sliding_window.hpp>


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
    return true;
}
void SpecificWorker::initialize(int period)
{
    std::cout << "Initialize worker" << std::endl;
    this->Period = 50;

    if (this->startup_check_flag) {
        this->startup_check();
    }
    else
    {
        // Viewer
        viewer = new AbstractGraphicViewer(this->frame, QRectF(consts.xMin, consts.yMin, consts.grid_width, consts.grid_length));
        viewer->add_robot(500, 500, 0, 100, QColor("Blue"));
        viewer->show();

        // grid
        QRectF dim{consts.xMin, consts.yMin, static_cast<qreal>(consts.grid_width), static_cast<qreal>(consts.grid_length)};
        grid.initialize(dim, consts.tile_size, &viewer->scene, false);

        // reset lidar odometry
        try{ lidarodometry_proxy->reset(); }
        catch (const Ice::Exception &e) { std::cout << "Error reading from LidarOdometry" << e << std::endl;};

        // Lidar thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
            {
                qInfo() << "[MOUSE] New global target arrived:" << p;
                Target target;
                target.set(p, true);    // true = in global coordinates
                target_buffer.put(std::move(target)); // false = in robot's frame - true = in global frame
                try{ lidarodometry_proxy->reset(); }
                catch (const Ice::Exception &e) { std::cout << "Error reading from LidarOdometry" << e << std::endl;};
            });

        timer.start(Period);
    }
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (res_.has_value() == false)  {   /*qWarning() << "No data Lidar";*/ return; }
    auto ldata = res_.value();

    /// convert points to Eigen
    std::vector<Eigen::Vector3f> points; points.reserve(ldata.points.size());
    for (const auto &p: ldata.points)
        points.emplace_back(p.x, p.y, p.z);
    //draw_lidar(ldata.points, 5);

    /// get target from buffer
    RoboCompGridPlanner::TPlan returning_plan;
    Target target;
    auto res = target_buffer.try_get();
    if(res.has_value())
        target = res.value();
    else
    {
        target = Target::invalid();
        adapt_grid_size(target);    // restore max grid size
    }

    /// clear grid and update it
    grid.clear();
    // TODO: pass target to remove occupied cells.
    //qInfo() << "update";
    grid.update_map(points, Eigen::Vector2f{0.0, 0.0}, consts.MAX_LASER_RANGE);
    //qInfo() << "costs";
    grid.update_costs( consts.robot_semi_width, true);
    //qInfo() << "after costs";
    /// if new valid target
    if (target.is_valid())
    {
        /// get robot pose
        Eigen::Transform<double, 3, 1> robot_pose = get_robot_pose();

        if(target.completed) // if robot arrived to target in the past iteration
        {
            returning_plan.valid = true;
            returning_plan.subtarget.x = 0.f;
            returning_plan.subtarget.y = 0.f;
            send_and_publish_plan(returning_plan);  // send zero plan to stop robot in bumper
            return;
        }
        auto original_target = target; // save original target to reinject it into the buffer

        if(target.global)  // global = true -> target in global coordinates
            set_target_global(robot_pose, target, original_target);    // transform target to robot's frame

        /// search for closest point to target in grid
        if(auto closest_ = grid.closest_free(target.pos_qt()); not closest_.has_value())
        {
            qInfo() << __FUNCTION__ << "No closest_point found. Returning. Cancelling target";
            target_buffer.try_get();   // empty buffer so it doesn't enter a loop
            return;
        }
        else target.set(closest_.value(), target.global); // keep current status of global
        draw_global_target(target.pos_eigen(), &viewer->scene);

        /// if target is not in line of sight, compute path
        if (not_line_of_sight_path(target.pos_qt()))
            returning_plan = compute_not_line_of_sight_target(target);
        else
            returning_plan = compute_line_of_sight_target(target);
        /// send plan to remote interface and publish it to rcnode
        send_and_publish_plan(returning_plan);

        /// adapt grid size and resolution for next iteration
        adapt_grid_size(target);
    }
    viewer->update();
    fps.print("FPS:");
}

////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_lidar()
{
    auto wait_period = std::chrono::milliseconds (this->Period);
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d("bpearl", 10000, 1); // TODO: move to constants
            auto data_helios = lidar3d1_proxy->getLidarDataWithThreshold2d("helios", 10000, 1); // TODO: move to constants
            // concatenate both lidars
            data.points.insert(data.points.end(), data_helios.points.begin(), data_helios.points.end());
            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysteresis of 2ms
            if (wait_period > std::chrono::milliseconds((long) data.period + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds((long) data.period - 2)) wait_period++;
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
} // Thread to read the lidar
void SpecificWorker::adapt_grid_size(const Target &target)
{
    // TODO: change TILE_SIZE in map according to the existence of target, distance to the target, closeness to obstacles, etc.
    // adjust grid size to target distance and path
    if(not target.is_valid() and grid.dim.width() != consts.grid_width) // if no target and first time
    {
        grid.reset();
        grid.initialize(QRectF(consts.xMin, consts.yMin, consts.grid_width, consts.grid_length), consts.tile_size, &viewer->scene, false);
        return;
    }
    if(target.is_valid())
    {
        std::vector<Eigen::Vector2f> points;
        points.emplace_back(0, 0);
        points.emplace_back(target.pos_eigen().x(), target.pos_eigen().y());
        float xmin = std::ranges::min(points, [](auto &a, auto &b)
        { return a.x() < b.x(); }).x();
        float ymin = std::ranges::min(points, [](auto &a, auto &b)
        { return a.y() < b.y(); }).y();
        float xmax = std::ranges::max(points, [](auto &a, auto &b)
        { return a.x() < b.x(); }).x();
        float ymax = std::ranges::max(points, [](auto &a, auto &b)
        { return a.y() < b.y(); }).y();
        QRectF dim{xmin-1000, ymin-1000, fabs(xmax - xmin)+2000, fabs(ymax - ymin)+2000};
        if (fabs(dim.width() - grid.dim.width()) > 500 or fabs(dim.height() > grid.dim.height()) > 500)
        {
            grid.reset();
            grid.initialize(dim, consts.tile_size, &viewer->scene, false);
        }
    }
    // tile size
//    if (not target.is_valid() and consts.tile_size < 150)    // if no target and first time increase TILE_SIZE
//    {
//        grid.reset();
//        consts.tile_size = 150;
//        grid.initialize(dim, consts.tile_size, &viewer->scene, false);
//    }
//    if (target.is_valid() and consts.tile_size > 50)
//    {
//        grid.reset();
//        consts.tile_size = 50;
//        grid.initialize(dim, consts.tile_size, &viewer->scene, false);
//    }
}
Eigen::Transform<double, 3, 1> SpecificWorker::get_robot_pose()
{
    Eigen::Transform<double, 3, 1> robot_pose;
    try
    {
        auto pose = lidarodometry_proxy->getFullPoseMatrix();
        robot_pose.matrix() << pose.m00, pose.m01, pose.m02, pose.m03,
                pose.m10, pose.m11, pose.m12, pose.m13,
                pose.m20, pose.m21, pose.m22, pose.m23,
                pose.m30, pose.m31, pose.m32, pose.m33;
    }
    catch (const Ice::Exception &e)
    { std::cout << "Error reading from LidarOdometry" << e << std::endl; }

    return robot_pose;
}
void SpecificWorker::set_target_global(const Eigen::Transform<double, 3, 1> &robot_pose, Target &target, Target &original_target)
{
    // transform target to robot's frame TODO: Check if robot_pose exists
    Eigen::Vector4d target_in_robot =
            robot_pose.inverse().matrix() * Eigen::Vector4d(target.pos_eigen().x() / 1000.0, target.pos_eigen().y() / 1000.0, 0.0, 1.0) * 1000.0;
    target.set(Eigen::Vector2f{target_in_robot(0), target_in_robot(1)}, true); // mm
    if(target.pos_eigen().norm() < consts.min_distance_to_target)   // robot at target
    {
        qInfo() << __FUNCTION__  << "Target reached. Sending zero target";
        target_buffer.put(Target{.active=true,
                                    .global=false,
                                    .completed=true,
                                    .point=Eigen::Vector2f::Zero(),
                                    .qpoint=QPointF{0.f,0.f}}); // reinject zero target to notify bumper of target reached
    }
    else    // still moving
        target_buffer.put(Target{.active=true,
                                    .global=true,
                                    .completed=false,
                                    .point=original_target.pos_eigen(),
                                    .qpoint=original_target.pos_qt()});  // reinject global target so it is not lost
}
RoboCompGridPlanner::TPlan SpecificWorker::compute_line_of_sight_target(const Target &target)
{
    RoboCompGridPlanner::TPlan returning_plan;
    returning_plan.valid = true;
    auto point_close_to_robot = target.point_at_distance(550);  // TODO: move to constants
    returning_plan.subtarget.x = point_close_to_robot.x();
    returning_plan.subtarget.y = point_close_to_robot.y();
    //std::cout << __FUNCTION__ << " target x,y:" << qtarget.x() << " " << qtarget.y() << std::endl;
    //std::cout << __FUNCTION__ << " returningplan_subtarget x,y:" << returning_plan.subtarget.x << " "<< returning_plan.subtarget.y << std::endl;
    draw_subtarget(point_close_to_robot, &viewer->scene);
    return returning_plan;
}
RoboCompGridPlanner::TPlan SpecificWorker::compute_not_line_of_sight_target(const Target &target)
{
    //qInfo() << __FUNCTION__ << "No Line of Sight path to target!";
    RoboCompGridPlanner::TPlan returning_plan;
    std::vector<Eigen::Vector2f> path;
    if(not last_path.empty())
    {
        auto frechet_distance =get_frechet_distance(last_path, path);
        qInfo() << "FRECHET DISTANCE " << frechet_distance;
        if(frechet_distance > 600 and (path_not_found_counter < path_not_found_limit))
        {
            path = last_path;
            path_not_found_counter++;
        }
        else
        {
            path_not_found_counter = 0;
        }
    }
    path = grid.compute_path(QPointF(0, 0), target.pos_qt());
    //qInfo() << "Path size" << path.size();
    if (not path.empty() and path.size() > 0)
    {
        auto subtarget = send_path(path, 550, M_PI_4 / 6);
        draw_path(path, &viewer->scene);
        draw_subtarget(Eigen::Vector2f(subtarget.x, subtarget.y), &viewer->scene);

        // Constructing the plan for interface type TPlan
        returning_plan.subtarget = subtarget;
        returning_plan.valid = true;
        last_path = path;
        for (auto &&p: iter::sliding_window(path, 1))
        {
            auto point = RoboCompGridPlanner::TPoint(p[0][0], p[0][1]);
            returning_plan.path.push_back(point);
        }
    } else  // EMPTY PATH
    {
        qWarning() << __FUNCTION__ << "Empty path";
        returning_plan.valid = false;
    }
    return returning_plan;
}
RoboCompGridPlanner::TPoint SpecificWorker::send_path(const std::vector<Eigen::Vector2f> &path, float threshold_dist, float threshold_angle)
{
    RoboCompGridPlanner::TPoint subtarget;

    static float h = sin(M_PI_2 + (M_PI - threshold_angle)/ 2.0);
    float len = 0.0;

    if (path.empty())
        return RoboCompGridPlanner::TPoint{.x=0.f, .y=0.f};

    if(path.size() < 3)
    {
        subtarget.x = path.back().x();
        subtarget.y = path.back().y();
    }

    auto local_path = path;
    local_path.erase(local_path.begin());

    for(auto &&p : iter::sliding_window(local_path, 3))
    {
        float d0d1 = (p[1]-p[0]).norm();
        float d1d2 = (p[2]-p[1]).norm();
        float d0d2 = (p[2]-p[0]).norm();
        len += d0d1;
//        qInfo()<< len << (d0d2 - (d1d2 + d0d1)) << (d0d1 * h) << (((d1d2 + d0d1) - d0d2) > (d0d1 * h));
        if (len > threshold_dist or (abs((d1d2 + d0d1) - d0d2) > (d0d1 * h)))
        {
            subtarget.x = p[1].x();
            subtarget.y = p[1].y();
            break;
        }
        subtarget.x = p[2].x();
        subtarget.y = p[2].y();
    }

//    qInfo()<< "t.duration"<<t.duration();
//    if((Eigen::Vector2f{last_subtarget.x,last_subtarget.y}-Eigen::Vector2f{subtarget.x,subtarget.y}).norm()< 600 or t.duration() > 500)
//    {
//        t.tick();
//        last_subtarget=subtarget;
//        return subtarget;
//    }
//    else
//        return last_subtarget;
    return subtarget;
}
double SpecificWorker::get_frechet_distance(const std::vector<Eigen::Vector2f>& pathA, const std::vector<Eigen::Vector2f>& pathB) {
    double maxDist = 0.0;
    for (size_t i = 0; i < (std::min(pathA.size(), pathB.size())); ++i) {
        double dist = (pathA[i] - pathB[i]).norm();
        maxDist = std::max(maxDist, dist);
    }
    return maxDist;
}
Eigen::Vector2f SpecificWorker::border_subtarget(const RoboCompVisualElements::TObject &target)    // TODO: Explicar qué hace el método
{
    Eigen::Vector2f target2f {target.x,target.y};
    float dist = target2f.norm();
    Eigen::Vector2f corner_left_top {consts.xMin, consts.yMax};
    Eigen::Vector2f corner_right_bottom {consts.xMax, consts.yMin};

    // Vertical
    if (target2f.x() == 0)
    {
        target2f.y() = (target.y > 0) ? corner_left_top.y() : corner_right_bottom.y();
        return target2f;
    }
    double m = target2f.y() / target2f.x();  // Pendiente de la línea

    // Calculamos las intersecciones con los lados del rectángulo
    Eigen::Vector2f interseccionIzquierda(consts.xMin, m * consts.xMin);
    Eigen::Vector2f interseccionDerecha(consts.xMax, m * consts.xMax);
    Eigen::Vector2f interseccionSuperior(consts.xMax / m, consts.yMax);
    Eigen::Vector2f interseccionInferior(consts.xMin / m, consts.yMin);

    // Comprobamos si las intersecciones están dentro del rectángulo
    Eigen::Vector2f intersecciones[4] = { interseccionIzquierda, interseccionDerecha, interseccionSuperior, interseccionInferior };
    Eigen::Vector2f resultado;

    for (int i = 0; i < 4; ++i)
    {
        float x = intersecciones[i].x();
        float y = intersecciones[i].y();
        if (consts.xMin <= x && x <= consts.xMax && consts.yMin <= y && y <= consts.yMax)
        {
            if((intersecciones[i]-target2f).norm() < dist)
            {
                resultado = intersecciones[i];
                break;
            }
        }
    }
    return resultado;
}
bool SpecificWorker::not_line_of_sight_path(const QPointF &f)
{
    int tile_size = 100;
    std::vector<Eigen::Vector2f> path;
    Eigen::Vector2f target(f.x(),f.y());
    Eigen::Vector2f origin(0.0,0.0);
    float steps = (target - origin).norm() / tile_size;
    Eigen::Vector2f step((target-origin)/steps);

    for ( int i = 0 ; i <= steps-3; ++i)
    {
        path.push_back(origin + i*step);
    }
    draw_path(path, &viewer->scene);
    return  grid.is_path_blocked(path);
}
void SpecificWorker::send_and_publish_plan(RoboCompGridPlanner::TPlan plan)
{
    // Sending plan to remote interface
    try
    { gridplanner_proxy->setPlan(plan); }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__ << " Error setting valid plan" << e << std::endl; }

    // Publishing the plan to rcnode
    try
    { gridplanner_pubproxy->setPlan(plan); }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__ << " Error publishing valid plan" << e << std::endl; }
}

//////////////////////////////// Draw ///////////////////////////////////////////////////////
void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    int s = 80;
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("green")), QBrush(QColor("green")));
        ptr->setPos(QPointF(p.x(), p.y()));
        points.push_back(ptr);
    }
}
void SpecificWorker::draw_subtarget(const Eigen::Vector2f &point, QGraphicsScene *scene)
{
    static QGraphicsEllipseItem* subtarget;
    scene->removeItem(subtarget);

    int s = 120;
    subtarget = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("red")), QBrush(QColor("red")));
    subtarget->setPos(QPointF(point.x(), point.y()));
}
void SpecificWorker::draw_global_target(const Eigen::Vector2f &point, QGraphicsScene *scene)
{
    static QGraphicsEllipseItem* subtarget;
    scene->removeItem(subtarget);

    int s = 120;
    subtarget = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("magenta")), QBrush(QColor("magenta")));
    subtarget->setPos(QPointF(point.x(), point.y()));
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
///////////////////////////////////////////////////////////////////////////////////////////

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// Interfaces
///////////////////////////////////////////////////////////////////////////////////////////////
/// SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack (RoboCompVisualElements::TObject target)
{
    // TODO: Check here if line of sight is blocked
    // qInfo()<< "TARGET " << target.x << target.y;
    Target t;
    if (target.x < consts.xMax and target.x > consts.xMin and target.y > consts.yMin and target.y < consts.yMax)
        t.set(Eigen::Vector2f{target.x, target.y}, false); /// false = in robot's frame
    else
    {
        t.set(border_subtarget(target), false); /// false = in robot's frame
        qInfo() << "TARGET OUT OF GRID" << border_subtarget(target).x() << border_subtarget(target).y();
    }
    target_buffer.put(std::move(t));
}

///// MORALLA


//
//std::vector<Eigen::Vector3f> SpecificWorker::get_lidar_data()
//{
//    std::vector <Eigen::Vector3f> points;
//    try
//    {
//        string lidar_name = "bpearl";
//        //auto ldata = lidar3d_proxy->getLidarData(lidar_name, 315, 90, 5);
//        auto ldata = lidar3d_proxy->getLidarDataWithThreshold2d("bpearl", 10000);
//        //std::cout << "LIDAR LENGHT: " << ldata.points.size() << std::endl;
//        //HELIOS
////        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
////        {
////            float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
////            return p.z < 300
////                   and p.z > -900
////                   and dist < 5000
////                   and dist > 550;
////        }, ldata) | iter::enumerate)
////            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});
//
//        // BPEARL
////        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
////                                         {
////                                             float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
////                                             return p.z < 1000 and dist > 50 and p.z > 100;
////                                         }, ldata.points) | iter::enumerate)
////            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});
//
//        for (const auto &p : ldata.points)
//            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});
//
//        return points;
//    }
//    catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
//    return points;
//}


//std::optional<Eigen::Vector2f> SpecificWorker::closest_point_to_target(const QPointF &p)
//{
//    for (double t = 1.0; t >= 0.0; t -= 0.001)
//    {
//        auto act_point = Eigen::Vector2f(p.x() * t, p.y() * t);
//        auto point_as_key = grid.pointToKey(act_point);
//        auto free_neighboors = grid.neighboors_8(point_as_key);
////        std::cout << "exists_los " << exists_los << " " << p.x() * t << " " << p.y() * t << std::endl;
//        if(free_neighboors.empty())
//        {
//            return act_point;
//        }
//    }
//    return {};
//}