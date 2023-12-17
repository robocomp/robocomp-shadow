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
        viewer = new AbstractGraphicViewer(this->frame, QRectF(xMin, yMin, grid_width, grid_length));
        viewer->add_robot(500, 500, 0, 0, QColor("Blue"));
        viewer->show();
        QRectF dim{xMin, yMin, static_cast<qreal>(grid_width), static_cast<qreal>(grid_length)};
        grid.initialize(dim, tile_size, &viewer->scene, false);

        // Lidar thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
            {
                qInfo() << "New global target arrived:" << p;
                target.set(p);
                Eigen::Vector2f pt{p.x(), p.y()};
//                draw_global_target(pt, &viewer->scene);
                target_buffer.put(std::make_tuple(pt, true));
            });// false = in robot's frame  });

        timer.start(Period);
    }
}
void SpecificWorker::compute()
{
    //auto cstart = std::chrono::high_resolution_clock::now();
    //clock.tick();
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (res_.has_value() == false)
    {   /*qWarning() << "No data Lidar";*/ return; }
    auto ldata = res_.value();
    std::vector<Eigen::Vector3f> points;
    points.reserve(ldata.points.size());
    for (const auto &p: ldata.points)
        points.emplace_back(p.x, p.y, p.z);
    //draw_lidar(ldata.points, 5);

    grid.clear();
    grid.update_map(points, Eigen::Vector2f{0.0, 0.0}, 7000);
    grid.update_costs(true);

    /// compute odometry
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_source(new pcl::PointCloud <pcl::PointXYZ>);
    pcl_cloud_source->reserve(ldata.points.size());
    for (const auto &[i, p]: ldata.points | iter::enumerate)
        pcl_cloud_source->emplace_back(pcl::PointXYZ{p.x / 1000.f, p.y / 1000.f, p.z / 1000.f});
    auto robot_pose = fastgicp.align(pcl_cloud_source);
    // QPointF robot_tr(robot_pose(0, 3)*1000, robot_pose(1, 3)*1000);

    if (auto res = target_buffer.try_get(); res.has_value())
    {
        auto [pos, global] = res.value();
        //if(target.active)
        //{
        //     std::cout << "Time updating grid 2: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cstart).count() <<std::endl;
        QPointF init_Qtarget;
        if(not global)
            init_Qtarget = {pos.x(), pos.y()};
        else    // target is set in global coordinates. Transform to robot's frame
        {
            qInfo() << "Target in global ref frame" << target.pos().x() << target.pos().y();
            Eigen::Vector4d target_in_robot =
                  robot_pose.inverse().matrix() * Eigen::Vector4d(target.pos().x() / 1000.f, target.pos().y() / 1000.f, 0.f, 1.f) * 1000;
            init_Qtarget = {target_in_robot(0), target_in_robot(1)};
            //draw_global_target(Eigen::Vector2f{target_in_robot(0), target_in_robot(1)}, &viewer->scene);
            target_buffer.put(std::make_tuple(target.pos(), true)); // reinject global target
        }
        Eigen::Vector2f closest_point;
        //if (auto res = closest_point_to_target(init_Qtarget); not res.has_value())
        if(auto res = grid.closest_free(init_Qtarget); not res.has_value())
        {
            qInfo() << __FUNCTION__ << "No closest_point found. Returning";
            return;
        }
        else closest_point = Eigen::Vector2f{res.value().x(), res.value().y()};
        draw_global_target(closest_point, &viewer->scene);
        qInfo() << closest_point.x() << closest_point.y() << init_Qtarget.x() << init_Qtarget.y();
        QPointF Qtarget(closest_point.x(), closest_point.y());

        RoboCompGridPlanner::TPlan returning_plan;
        if (not_line_of_sight_path(Qtarget))
        {
            qInfo() << __FUNCTION__ << "No Line of Sight path to target!";
            bool path_found = false;
            std::vector<Eigen::Vector2f> path;
//            while(not path_found)
//            {
//                path = grid.compute_path(QPointF(0, 0), Qtarget);
//                auto frechet_distance = frechetDistance(path, last_path);
//                std::cout << "FRECHET DISTANCE: " << frechet_distance << std::endl;
//
//                if(frechet_distance < 500)
//                {
//                    std::cout << "ENTRAAAAAAAAAAAAaaa" << std::endl;
//                    path_found = true;
//                }
//            }
            path = grid.compute_path(QPointF(0, 0), Qtarget);
            qInfo() << "Path size" << path.size();
            if (not path.empty() and path.size() > 0)
            {
                auto subtarget = send_path(path, 650, M_PI_4 / 6);
                draw_path(path, &viewer->scene);
                draw_subtarget(Eigen::Vector2f(subtarget.x, subtarget.y), &viewer->scene);

                // Constructing the plan for interface type TPlan
                //Tplan (path, timestamps, Tpoint subtarget,bool valid)
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
                draw_subtarget(Eigen::Vector2f{0.0, 0.0}, &viewer->scene);
                returning_plan.valid = false;
            }
        }
        else // Target in line of sight
        {
            returning_plan.valid = true;
            returning_plan.subtarget.x = Qtarget.x();
            returning_plan.subtarget.y = Qtarget.y();
            std::cout << __FUNCTION__ << " target x,y:" << Qtarget.x() << " "<< Qtarget.y() << std::endl;
            std::cout << __FUNCTION__ << " returningplan_subtarget x,y:" << returning_plan.subtarget.x << " "<< returning_plan.subtarget.y << std::endl;
            draw_subtarget(Eigen::Vector2f {Qtarget.x(), Qtarget.y()}, &viewer->scene);
        }
        // Sending plan to remote interface
        try
        { gridplanner_proxy->setPlan(returning_plan); }
        catch (const Ice::Exception &e)
        { std::cout << __FUNCTION__ << " Error setting valid plan" << e << std::endl; }

        // Publishing the plan to rcnode
        try
        { gridplanner_pubproxy->setPlan(returning_plan); }
        catch (const Ice::Exception &e)
        { std::cout << __FUNCTION__ << " Error publishing valid plan" << e << std::endl; }
    }
    else //NO TARGET
    { }

    viewer->update();
    fps.print("FPS:");
}

////////////////////////////////////////////////////////////////////////////////////////////
std::vector<Eigen::Vector3f> SpecificWorker::get_lidar_data()
{
    std::vector <Eigen::Vector3f> points;
    try
    {
       string lidar_name = "bpearl";
       //auto ldata = lidar3d_proxy->getLidarData(lidar_name, 315, 90, 5);
       auto ldata = lidar3d_proxy->getLidarDataWithThreshold2d("bpearl", 10000);
        //std::cout << "LIDAR LENGHT: " << ldata.points.size() << std::endl;
        //HELIOS
//        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
//        {
//            float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
//            return p.z < 300
//                   and p.z > -900
//                   and dist < 5000
//                   and dist > 550;
//        }, ldata) | iter::enumerate)
//            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});

        // BPEARL
//        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
//                                         {
//                                             float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
//                                             return p.z < 1000 and dist > 50 and p.z > 100;
//                                         }, ldata.points) | iter::enumerate)
//            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});

        for (const auto &p : ldata.points)
            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});

        return points;
    }
    catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
    return points;
}
// Thread to read the lidar
void SpecificWorker::read_lidar()
{
    auto wait_period = std::chrono::milliseconds (this->Period);
    while(true)
    {
        try
        {
            // Use with simulated lidar in webots using "pearl" name
            //auto data = lidar3d_proxy->getLidarData("bpearl", -90, 360, 1); // TODO: move to contants
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d("bpearl", 10000); // TODO: move to contants
            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysterisis of 2ms
            if (wait_period > std::chrono::milliseconds((long) data.period + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds((long) data.period - 2)) wait_period++;
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
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
float SpecificWorker::euclideanDistance(const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    return (a - b).norm();
}
float SpecificWorker::frechetDistanceUtil(const std::vector<Eigen::Vector2f>& path1,
                          const std::vector<Eigen::Vector2f>& path2,
                          int i, int j,
                          std::vector<std::vector<float>>& dp)
{
    if (i == 0 && j == 0) {
        return euclideanDistance(path1[0], path2[0]);
    }

    if (i < 0 || j < 0) {
        return std::numeric_limits<float>::max();
    }

    if (dp[i][j] != -1) {
        return dp[i][j];
    }

    float cost = euclideanDistance(path1[i], path2[j]);
    float minPrevious = std::min({frechetDistanceUtil(path1, path2, i - 1, j, dp),
                                  frechetDistanceUtil(path1, path2, i - 1, j - 1, dp),
                                  frechetDistanceUtil(path1, path2, i, j - 1, dp)});

    dp[i][j] = std::max(cost, minPrevious);
    return dp[i][j];
}
float SpecificWorker::frechetDistance(const std::vector<Eigen::Vector2f>& path1, const std::vector<Eigen::Vector2f>& path2)
{
    if(path1.empty() or path2.empty())
    {
        std::cout << "Empty path found." << std::endl;
        return -1;
    }
    std::vector<std::vector<float>> dp(path1.size(), std::vector<float>(path2.size(), -1));
    return frechetDistanceUtil(path1, path2, path1.size() - 1, path2.size() - 1, dp);
}
Eigen::Vector2f SpecificWorker::border_subtarget(RoboCompVisualElements::TObject target)    // TODO: Explicar qué hace el método
{
    Eigen::Vector2f target2f {target.x,target.y};

    float dist = target2f.norm();

    Eigen::Vector2f corner_left_top {xMin, yMax};
    Eigen::Vector2f corner_right_bottom {xMax, yMin};

    //Vertical
    if (target2f.x() == 0)
    {
        target2f.y() = (target.y > 0) ? corner_left_top.y() : corner_right_bottom.y();
        return target2f;
    }
    double m = target2f.y() / target2f.x();  // Pendiente de la línea

    // Calculamos las intersecciones con los lados del rectángulo
    Eigen::Vector2f interseccionIzquierda(xMin, m * xMin);
    Eigen::Vector2f interseccionDerecha(xMax, m * xMax);
    Eigen::Vector2f interseccionSuperior(xMax / m, yMax);
    Eigen::Vector2f interseccionInferior(xMin / m, yMin);

    // Comprobamos si las intersecciones están dentro del rectángulo
    Eigen::Vector2f intersecciones[4] = { interseccionIzquierda, interseccionDerecha, interseccionSuperior, interseccionInferior };
    Eigen::Vector2f resultado;

    for (int i = 0; i < 4; ++i) {
        float x = intersecciones[i].x();
        float y = intersecciones[i].y();
        if (xMin <= x && x <= xMax && yMin <= y && y <= yMax)
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
std::optional<Eigen::Vector2f> SpecificWorker::closest_point_to_target(const QPointF &p)
{
    for (double t = 1.0; t >= 0.0; t -= 0.001)
    {
        auto act_point = Eigen::Vector2f(p.x() * t, p.y() * t);
        auto point_as_key = grid.pointToKey(act_point);
        auto free_neighboors = grid.neighboors_8(point_as_key);
//        std::cout << "exists_los " << exists_los << " " << p.x() * t << " " << p.y() * t << std::endl;
        if(free_neighboors.empty())
        {
            return act_point;
        }
    }
    return {};
}
///////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// Inrterfaces
///////////////////////////////////////////////////////////////////////////////////////////////

/// SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack (RoboCompVisualElements::TObject target)
{
    //    qInfo()<< "TARGET " << target.x << target.y;
    if (target.x < xMax and target.x > xMin and target.y > yMin and target.y < yMax)
        target_buffer.put(std::make_tuple(Eigen::Vector2f{target.x, target.y}, false)); // false = in robot's frame
    else
    {
        target_buffer.put(std::make_tuple(border_subtarget(target), false));
        qInfo()<<"TARGET OUT OF GRID" << border_subtarget(target).x() << border_subtarget(target).y();
    }
}
