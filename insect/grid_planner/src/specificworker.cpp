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
#include <cppitertools/range.hpp>


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
        viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
        //QRectF(params.xMin, params.yMin, params.grid_width, params.grid_length));
        viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
        viewer->show();

        // grid
        //QRectF dim{params.xMin, params.yMin, static_cast<qreal>(params.grid_width), static_cast<qreal>(params.grid_length)};
        grid.initialize(params.GRID_MAX_DIM, static_cast<int>(params.TILE_SIZE), &viewer->scene, false);

        // reset lidar odometry
        try{ lidarodometry_proxy->reset(); }
        catch (const Ice::Exception &e) { std::cout << "Error reading from LidarOdometry" << e << std::endl;}

        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
            {
                qInfo() << "[MOUSE] New global target arrived:" << p;
                Target target;
                if(grid.get_dim().contains(p))
                    target.set(p, true); /// false = in robot's frame; true = in global frame
                else
                    target.set(border_subtarget(Eigen::Vector2f{p.x(), p.y()}), true); /// false = in robot's frame
                target_buffer.put(std::move(target)); // false = in robot's frame - true = in global frame
                try{ lidarodometry_proxy->reset(); }
                catch (const Ice::Exception &e) { std::cout << "Error reading from LidarOdometry" << e << std::endl;};
            });
        connect(viewer, &AbstractGraphicViewer::right_click, [this](QPointF p)
        {
            qInfo() <<  "RIGHT CLICK. Cancelling target";
            cancel_from_mouse = true;
        });

        timer.start(Period);
    }
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value())  {   /*qWarning() << "No data Lidar";*/ return; }
    auto points = res_.value();

    /// clear grid and update it
    grid.clear();   // TODO: pass human target to remove occupied cells.
    grid.update_map(points, Eigen::Vector2f{0.0, 0.0}, params.MAX_LIDAR_RANGE);
    grid.update_costs( params.ROBOT_SEMI_WIDTH, true);

    /// get target from buffer
    RoboCompGridPlanner::TPlan returning_plan;
    Target target;
    auto res = target_buffer.try_get();
    if(res.has_value())
        target = res.value();
    else
    {
        target = Target::invalid();
        adapt_grid_size(target, {});    // restore max grid size
    }
    if(cancel_from_mouse)
    {
        target.completed = true;
        cancel_from_mouse = false;
    }

    /// if new valid target arrived process it
    if (target.is_valid())
    {
        /// get robot pose
        this->robot_pose = get_robot_pose();

        if(target.completed) // if robot arrived to target in the past iteration
        {
            returning_plan.valid = true;
            returning_plan.subtarget.x = 0.f;
            returning_plan.subtarget.y = 0.f;
            send_and_publish_plan(returning_plan);  // send zero plan to stop robot in bumper
            draw_paths({}, &viewer->scene, true);   // erase paths
            return;
        }
        auto original_target = target; // save original target to reinject it into the buffer

        if(target.global)  // global = true -> target in global coordinates
            target = transform_target_to_global_frame(robot_pose, target);    // transform target to robot's frame

        /// check if target has been reached
        check_if_robot_at_target(target, original_target);

        /// search for closest point to target in grid
        if(auto closest_ = grid.closest_free(target.pos_qt()); not closest_.has_value())
        {
            qInfo() << __FUNCTION__ << "No closest_point found. Returning. Cancelling target";
            //target_buffer.try_get();   // empty buffer so it doesn't enter a loop
            target_buffer.put(Target{.active=true,
                    .global=false,
                    .completed=true,
                    .point=Eigen::Vector2f::Zero(),
                    .qpoint=QPointF{0.f,0.f}}); // reinject zero target to notify bumper of target reached
            return;
        }
        else target.set(closest_.value(), target.global); // keep current status of global
        draw_global_target(target.pos_eigen(), &viewer->scene);

        /// if target is not in line of sight, compute path
        if(not grid.is_line_of_sigth_to_target_free(target.pos_eigen(), Eigen::Vector2f::Zero(),
                                                    params.ROBOT_SEMI_WIDTH))
            returning_plan = compute_plan_from_grid(target);
        else
            returning_plan = compute_line_of_sight_target(target);

        if(not returning_plan.valid)   // no valid path found. Cancel target
        {
            qWarning() << __FUNCTION__ << "No valid path found. Cancelling target";
            target_buffer.put(Target{.active=true,
                    .global=false,
                    .completed=true,
                    .point=Eigen::Vector2f::Zero(),
                    .qpoint=QPointF{0.f,0.f}}); // reinject zero target to notify bumper of target reached
            return;
        }
//        else
//            /// adapt grid size and resolution for next iteration
//            adapt_grid_size(target, returning_plan.path);

        /// send plan to remote interface and publish it to rcnode
        send_and_publish_plan(returning_plan);

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
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_LOW,
                                                                           params.MAX_LIDAR_LOW_RANGE,
                                                                           params.LIDAR_LOW_DECIMATION_FACTOR);
            auto data_helios = lidar3d1_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_HIGH,
                                                                           params.MAX_LIDAR_HIGH_RANGE,
                                                                           params.LIDAR_HIGH_DECIMATION_FACTOR);
            // concatenate both lidars
            data.points.insert(data.points.end(), data_helios.points.begin(), data_helios.points.end());
            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysteresis of 2ms
            if (wait_period > std::chrono::milliseconds((long) data.period + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds((long) data.period - 2)) wait_period++;
            std::vector<Eigen::Vector3f> eig_data(data.points.size());
            for (const auto &[i, p]: data.points | iter::enumerate)
               eig_data[i] = {p.x, p.y, p.z};
            buffer_lidar_data.put(std::move(eig_data));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
} // Thread to read the lidar
bool SpecificWorker::check_if_robot_at_target(const Target &target_, const Target &original_target_)
{
    if(target_.pos_eigen().norm() < params.MIN_DISTANCE_TO_TARGET)   // robot at target
    {
        qInfo() << __FUNCTION__ << "Target reached. Sending zero target";
        target_buffer.put(Target{.active=true,
                                    .global=false,
                                    .completed=true,
                                    .point=Eigen::Vector2f::Zero(),
                                    .qpoint=QPointF{0.f, 0.f}}); // reinject zero target to notify bumper of target reached
    }
    else /// still moving
    {
        // if global target, reinject the original one into the buffer since it establishes robot's zero frame
        if (target_.global)
            target_buffer.put(Target{.active=true,
                                        .global=true,
                                        .completed=false,
                                        .point=original_target_.pos_eigen(),
                                        .qpoint=original_target_.pos_qt()});
    }
}
void SpecificWorker::adapt_grid_size(const Target &target, const RoboCompGridPlanner::Points &path)
{
    // TODO: EXPERIMENTAL change TILE_SIZE in map according to the existence of target, distance to the target, closeness to obstacles, etc.
    // adjust grid size to target distance and path
    if(not target.is_valid() and grid.get_dim().width() != params.GRID_MAX_DIM.width()) // if no target and first time
    {
        grid.reset();
        grid.initialize(params.GRID_MAX_DIM, params.TILE_SIZE, &viewer->scene, false);
        return;
    }
    if(target.is_valid())
    {
        std::vector<Eigen::Vector2f> points; points.reserve(path.size()+2);
        points.emplace_back(0, 0);
        points.emplace_back(target.pos_eigen().x(), target.pos_eigen().y());
        for(auto &&p: path)
            points.emplace_back(p.x, p.y);
        // compute enclosing rectangle of points
        float xmin = std::ranges::min(points, [](auto &a, auto &b)
        { return a.x() < b.x(); }).x();
        float ymin = std::ranges::min(points, [](auto &a, auto &b)
        { return a.y() < b.y(); }).y();
        float xmax = std::ranges::max(points, [](auto &a, auto &b)
        { return a.x() < b.x(); }).x();
        float ymax = std::ranges::max(points, [](auto &a, auto &b)
        { return a.y() < b.y(); }).y();
        QRectF dim{xmin-1200, ymin-1200, fabs(xmax - xmin)+2400, fabs(ymax - ymin)+2400};
        if (fabs(dim.width() - grid.get_dim().width()) > 500 or fabs(dim.height() > grid.get_dim().height()) > 500)
        {
            grid.reset();
            grid.initialize(dim, 50, &viewer->scene, false);
        }
    }
    // tile size
//    if (not target.is_valid() and params.tile_size < 150)    // if no target and first time increase TILE_SIZE
//    {
//        grid.reset();
//        params.tile_size = 150;
//        grid.initialize(dim, params.tile_size, &viewer->scene, false);
//    }
//    if (target.is_valid() and params.tile_size > 50)
//    {
//        grid.reset();
//        params.tile_size = 50;
//        grid.initialize(dim, params.tile_size, &viewer->scene, false);
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
SpecificWorker::Target SpecificWorker::transform_target_to_global_frame(const Eigen::Transform<double, 3, 1> &robot_pose, const Target &target)
{
    // transform target to robot's frame
    Target t = target;
    Eigen::Vector4d target_in_robot =
            robot_pose.inverse().matrix() * Eigen::Vector4d(target.pos_eigen().x() / 1000.0, target.pos_eigen().y() / 1000.0, 0.0, 1.0) * 1000.0;
    t.set(Eigen::Vector2f{target_in_robot(0), target_in_robot(1)}, true); // mm
    return t;
}
RoboCompGridPlanner::TPlan SpecificWorker::compute_line_of_sight_target(const Target &target)
{
    RoboCompGridPlanner::TPlan returning_plan;
    returning_plan.valid = true;
    auto point_close_to_robot = target.point_at_distance(params.CARROT_DISTANCE);  // TODO: move to constants
    returning_plan.subtarget.x = point_close_to_robot.x();
    returning_plan.subtarget.y = point_close_to_robot.y();
    draw_paths({}, &viewer->scene, true);   // erase paths
    draw_path({}, &viewer->scene, true);          // erase path
    draw_subtarget(point_close_to_robot, &viewer->scene);
    return returning_plan;
}
RoboCompGridPlanner::TPlan SpecificWorker::compute_plan_from_grid(const Target &target)
{
    // keep the current path in a static variable
    static std::vector<Eigen::Vector2f> current_path = {}, original_path = {};
    // it has to be reset when a new target arrives
    if (target.completed)
        current_path.clear();
    // if it is empty (first time or after a target has been reached), compute a new path
    if (current_path.empty())
    {
        current_path = grid.compute_path(Eigen::Vector2f::Zero(), target.pos_eigen());
        original_path = current_path;
    }
    // the robot has to turn to the target before recomputing the path. Otherwise, the path becomes unstable.

    // check if params.ELAPSED_TIME has passed since last path computation. We update path once very ELAPSED_TIME mseconds
//    static auto last_time = std::chrono::steady_clock::now();
//    auto now = std::chrono::steady_clock::now();
//    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);
//    if (elapsed_time.count() > 5000)
//        current_path = grid.compute_path(Eigen::Vector2f::Zero(), target.pos_eigen());
//    else
//    {
//        std::vector<Eigen::Vector2f> local_current_path;
//        local_current_path.reserve(current_path.size());
//        std::ranges::transform(original_path, std::back_inserter(local_current_path), [rp = robot_pose](auto &p)
//        {
//            Eigen::Vector4d p4d =
//                    rp.inverse().matrix() * Eigen::Vector4d(p.x() / 1000.f, p.y() / 1000.f, 0.0, 1.0) * 1000.f;
//            return Eigen::Vector2f{p4d.x(), p4d.y()};
//        });
//        current_path = local_current_path;
//    }

//    std::vector<std::vector<Eigen::Vector2f>> paths;
//    static std::vector<std::pair<int, std::vector<Eigen::Vector2f>>> tracklets;
//
//    // compute the K optimal paths that differ "min_max_dist" among them using the Yen algorithm and the max_distance approximation to Frechet distance
//    paths = grid.compute_k_paths(Eigen::Vector2f::Zero(), target.pos_eigen(), 3, 500.f);
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> distances(paths.size(), paths.size());
//    for (auto &&[i, path]: paths | iter::enumerate)
//        for(auto &&[j, vote]: tracklets | iter::enumerate)
//            distances(static_cast<long>(i), static_cast<long>(j)) = max_distance(path, vote.second);
//
//    // match votes against tracklets
//    for(auto &&[j, tracklet]: tracklets | iter::enumerate)
//    {
//        auto min_dist = std::min_element(distances.col(j).data(), distances.col(j).data() + distances.rows());
//        if(*min_dist < 500.f)
//        {
//            auto i = std::distance(distances.col(j).data(), min_dist);
//            tracklet.second = paths[i];
//            tracklet.first++;
//        }
//        else
//            tracklet.first--;
//    }

    //   if(paths.empty()) { qWarning() << __FUNCTION__ << "No paths found"; return RoboCompGridPlanner::TPlan{.valid=false}; }

//     if current_path is empty, select the first path and return
//    if(current_path.empty())
//        current_path = paths.front();
//    else
//    {

    // check if params.ELAPSED_TIME has passed since last path computation. We update path once very ELAPSED_TIME mseconds
    static auto last_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);
    if (elapsed_time.count() > 3000)
    {
        // compute the K optimal paths that differ "min_max_dist" among them using the Yen algorithm and the max_distance approximation to Frechet distance
        std::vector<std::vector<Eigen::Vector2f>> paths;
        paths = grid.compute_k_paths(Eigen::Vector2f::Zero(), target.pos_eigen(), 3, 500.f);
        if (paths.empty())
        {
            qWarning() << __FUNCTION__ << "No paths found";
            return RoboCompGridPlanner::TPlan{.valid=false};
        }
        // compute a vector with the <max_distances, index> of current_path to all paths
        std::vector<std::pair<float, int>> distances_to_current;
        for (auto &&[i, path]: paths | iter::enumerate)
            distances_to_current.emplace_back(max_distance(current_path, path), i);
        std::ranges::sort(distances_to_current, [](auto &a, auto &b)
        { return a.first < b.first; });

        // assign current_path to the closest path
        current_path = paths[distances_to_current.front().second];
        original_path = current_path;
        qInfo() << __FUNCTION__ << "Dist to current path: " << distances_to_current.front().first << "Num paths: "
                << distances_to_current.size();
        draw_paths(paths, &viewer->scene);
    } else  // if not enough time has passed, transform current_path to global frame
    {
        // transform current_path to global frame
        std::vector<Eigen::Vector2f> local_current_path;
        local_current_path.reserve(current_path.size());
        std::ranges::transform(original_path, std::back_inserter(local_current_path), [rp = robot_pose](auto &p)
        {
            Eigen::Vector4d p4d =
                    rp.inverse().matrix() * Eigen::Vector4d(p.x() / 1000.f, p.y() / 1000.f, 0.0, 1.0) * 1000.f;
            return Eigen::Vector2f{p4d.x(), p4d.y()};
        });
        current_path = local_current_path;
    }

    draw_path(current_path, &viewer->scene);

    // match the set of paths with the current_path by selecting the path with the minimum max_distance to current_path,
    // that is above a threshold. If no path is found, select the first path
    // if no match is found, keep with the current path for 5 iterations
//    static int counter = 0;
//    if (distances_to_current.front().first > params.ROBOT_WIDTH*4) // TODO: draw distance to current_path (front) to check magnitudes
//    {
//        if (counter < 50)
//        {   // if no match in N consecutive iterations, change to the first path
//            counter++;
//            // transform current_path to global frame
//            std::vector<Eigen::Vector2f> local_current_path;
//            local_current_path.reserve(current_path.size());
//            std::ranges::transform(original_path, std::back_inserter(local_current_path), [rp = robot_pose](auto &p)
//            {
//                Eigen::Vector4d p4d =
//                        rp.inverse().matrix() * Eigen::Vector4d(p.x() / 1000.f, p.y() / 1000.f, 0.0, 1.0) * 1000.f;
//                return Eigen::Vector2f{p4d.x(), p4d.y()};
//            });
//            current_path = local_current_path;
//        } else
//        {
//            counter = 0;
//            current_path = paths[distances_to_current.front().second];
//            original_path = current_path;
//        }
//    } else    // match found, reset counter
//    {
//        counter = 0;
//        current_path = paths[distances_to_current.front().second];
//        original_path = current_path;
//    }


    // fill subtarget and returning_plan
    RoboCompGridPlanner::TPlan returning_plan;
    if (not current_path.empty())
    {
        auto subtarget = get_carrot_from_path(current_path, params.CARROT_DISTANCE, params.CARROT_ANGLE);
        draw_subtarget(Eigen::Vector2f(subtarget.x, subtarget.y), &viewer->scene);

        // Constructing the interface type TPlan
        returning_plan.subtarget = subtarget;
        returning_plan.valid = true;
        current_path = current_path;
        returning_plan.path.reserve(current_path.size());
        for (auto &&p: current_path)
            returning_plan.path.emplace_back(p.x(), p.y());
    }
    else  // EMPTY PATH
    {
        qWarning() << __FUNCTION__ << "Empty path";
        returning_plan.valid = false;
    }
    return returning_plan;
}
//    if(not last_path.empty())
//    {
//        auto frechet_distance = frechet_distance(last_path, path);
//        qInfo() << __FUNCTION__ << ": Frechet distance = " << frechet_distance;
//        if(frechet_distance > 600 and (path_not_found_counter < path_not_found_limit))
//        {
//            path = last_path;
//            path_not_found_counter++;
//        }
//        else
//        {
//            path_not_found_counter = 0;
//        }
//    }
//auto path = paths.front();

//    static float last_path_distance;
//    if(not last_path.empty())
//    {
//        float path_distance = 0.f;
//        for(const auto &pp: path | iter::sliding_window(2))
//            path_distance += (pp[1] - pp[0]).norm();
//        //qInfo() << __FUNCTION__ << ": Frechet distance = " << frechet_distance;
//        if((fabs(path_distance - last_path_distance) > last_path_distance * 0.7))
//        {
//            path = last_path;
//            last_path_distance = path_distance;
//            path_not_found_counter++;
//        }
//                else
//        {
//            path_not_found_counter = 0;
//        }
//    }
RoboCompGridPlanner::TPoint SpecificWorker::get_carrot_from_path(const std::vector<Eigen::Vector2f> &path, float threshold_dist, float threshold_angle)
{
    // computes a subtarget from a path that is closer than threshold_dist and has an angle with the robot smaller than threshold_angle

    // Admissible conditions
    if (path.empty())
        return RoboCompGridPlanner::TPoint{.x=0.f, .y=0.f};

    RoboCompGridPlanner::TPoint subtarget;
    double h = sin(M_PI_2 + (M_PI - threshold_angle)/ 2.f);
    float len = 0.f;

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
        if (len > threshold_dist or (abs((d1d2 + d0d1) - d0d2) > (d0d1 * h)))
        {
            subtarget.x = p[1].x();
            subtarget.y = p[1].y();
            break;
        }
        subtarget.x = p[2].x();
        subtarget.y = p[2].y();
    }
    return subtarget;
}
float SpecificWorker::max_distance(const std::vector<Eigen::Vector2f> &pathA, const std::vector<Eigen::Vector2f> &pathB)
{
    // Approximates Frechet distance
    std::vector<float> dists; dists.reserve(std::min(pathA.size(), pathB.size()));
    for(auto &&i: iter::range(std::min(pathA.size(), pathB.size())))
        dists.emplace_back((pathA[i] - pathB[i]).norm());
    return std::ranges::max(dists);
}
double SpecificWorker::frechet_distance(const std::vector<Eigen::Vector2f>& pathA, const std::vector<Eigen::Vector2f>& pathB)
{
    std::vector<float> dists;
    for(auto &&i: iter::range(std::min(pathA.size(), pathB.size())))
        dists.emplace_back((pathA[i] - pathB[i]).norm());
    return std::ranges::max(dists);
}
// Check if target is within the grid and otherwise, compute the intersection with the border
Eigen::Vector2f SpecificWorker::border_subtarget(const Eigen::Vector2f &target)
{
    QRectF dim = grid.get_dim();
    float xMin = dim.left();
    float xMax = dim.right();
    float yMin = dim.top();
    float yMax = dim.bottom();

    Eigen::Vector2f target2f{target.x(), target.y()};
    float dist = target2f.norm();
    Eigen::Vector2f corner_left_top {xMin, yMax};
    Eigen::Vector2f corner_right_bottom {xMax, yMin};

    // Vertical
    if (target2f.x() == 0)
    {
        target2f.y() = (target.y() > 0) ? corner_left_top.y() : corner_right_bottom.y();
        return target2f;
    }
    double m = target2f.y() / target2f.x();  // Pendiente de la línea

    // Compute intersections with the rectangle
    Eigen::Vector2f interseccionIzquierda(xMin, m * xMin);
    Eigen::Vector2f interseccionDerecha(xMax, m * xMax);
    Eigen::Vector2f interseccionSuperior(xMax / m, yMax);
    Eigen::Vector2f interseccionInferior(xMin / m, yMin);

    // Check if intersections are inside the rectangle
    Eigen::Vector2f intersecciones[4] = { interseccionIzquierda, interseccionDerecha, interseccionSuperior, interseccionInferior };
    Eigen::Vector2f resultado;

    for (int i = 0; i < 4; ++i)
    {
        float x = intersecciones[i].x();
        float y = intersecciones[i].y();
        if (xMin <= x and x <= xMax and yMin <= y and y <= yMax)
        {
            if((intersecciones[i]-target2f).norm() < dist)
            {
                resultado = intersecciones[i] - params.TILE_SIZE * intersecciones[i].normalized();
                break;
            }
        }
    }
    qInfo() << __FUNCTION__ << "Target out of grid. Computing intersection with border: [" << resultado.x() << resultado.y() << "]";
    return resultado;
}

///////////////////////////////// Send and publish plan ///////////////////////////////////////
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
void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    if(erase_only) return;

    float s = 100;
    auto color = QColor("green");
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(color), QBrush(color));
        ptr->setPos(QPointF(p.x(), p.y()));
        points.push_back(ptr);
    }
}
void SpecificWorker::draw_paths(const std::vector<std::vector<Eigen::Vector2f>> &paths, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    static QColor colors[] = {QColor("cyan"), QColor("blue"), QColor("red"), QColor("orange"), QColor("magenta"), QColor("cyan")};
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    if(erase_only) return;

    float s = 80;
    for(const auto &[i, path]: paths | iter::enumerate)
    {
        // pick a consecutive color
        auto color = colors[i];
        for(const auto &p: path)
        {
            auto ptr = scene->addEllipse(-s/2.f, -s/2.f, s, s, QPen(color), QBrush(color));
            ptr->setPos(QPointF(p.x(), p.y()));
            points.push_back(ptr);
        }
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
    // qInfo()<< "TARGET " << target.x << target.y;
    Target t;
    if(grid.get_dim().contains(QPointF{target.x, target.y}))
        t.set(Eigen::Vector2f{target.x, target.y}, false); /// false = in robot's frame; true = in global frame
    else
    {
        Eigen::Vector2f tt = border_subtarget(Eigen::Vector2f{target.x, target.y});
        t.set(tt, false); /// false = in robot's frame
        //qInfo() << "TARGET OUT OF GRID" << tt.x() << tt.y();
    }

//    if (target.x < params.xMax and target.x > params.xMin and target.y > params.yMin and target.y < params.yMax)
//        t.set(Eigen::Vector2f{target.x, target.y}, false); /// false = in robot's frame
//    else
//    {
//        t.set(border_subtarget(target), false); /// false = in robot's frame
//        qInfo() << "TARGET OUT OF GRID" << border_subtarget(target).x() << border_subtarget(target).y();
//    }
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
//        auto point_as_key = grid.point_to_key(act_point);
//        auto free_neighboors = grid.neighboors_8(point_as_key);
////        std::cout << "exists_los " << exists_los << " " << p.x() * t << " " << p.y() * t << std::endl;
//        if(free_neighboors.empty())
//        {
//            return act_point;
//        }
//    }
//    return {};
//}