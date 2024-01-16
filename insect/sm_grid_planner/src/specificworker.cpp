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
#include <cppitertools/takewhile.hpp>


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

        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // reset lidar odometry
        try{ lidarodometry_proxy->reset(); }
        catch (const Ice::Exception &e) { std::cout << "Error reading from LidarOdometry" << e << std::endl;}

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
            {
                qInfo() << "[MOUSE] New global target arrived:" << p;
                Target target;
                current_path.clear();
                QRectF dim;
                try
                {
                    params.gdim = gridder_proxy->getDimensions();
                    dim = QRectF{params.gdim.left, params.gdim.top, params.gdim.width, params.gdim.height};
                    if (dim.contains(p))
                        target.set(p, true);
                    else
                        target.set(compute_closest_target_to_grid_border(Eigen::Vector2f{p.x(), p.y()}), true); /// false = in robot's frame
                    target.set_original(target.pos_eigen());
                    target.set_new(true);
                }
                catch (const Ice::Exception &e)
                {
                    std::cout << "[MOUSE] Error reading from Gridder" << e << std::endl;
                    target = Target::invalid();
                    return;
                }
                target_buffer.put(std::move(target));
                // wait until odometry is properly reset: matrix trace = 4 aka identity matrix
                try
                {
                    lidarodometry_proxy->reset();   // empty buffer
                    std::optional<std::pair<Eigen::Transform<double, 3, 1>, Eigen::Transform<double, 3, 1>>> rp;
                    auto start = std::chrono::high_resolution_clock::now();
                    do
                    {   rp = get_robot_pose_and_change();
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    } while (rp->first.matrix().diagonal().sum() < 3.7 and
                             std::chrono::duration_cast<std::chrono::milliseconds>(start - std::chrono::high_resolution_clock::now()).count() < 4000);
                    if(rp->first.matrix().diagonal().sum() < 3.7)
                        qWarning() << "Odometry not properly reset. Matrix trace too large: " << rp->first.matrix().diagonal().sum();
                }
                catch (const Ice::Exception &e)
                {
                    std::cout << "[MOUSE] Error reading from LidarOdometry" << e << std::endl;
                    target = Target::invalid();
                    return;
                };
            });
        connect(viewer, &AbstractGraphicViewer::right_click, [this](QPointF p)
        {
            qInfo() <<  "RIGHT CLICK. Cancelling target";
            cancel_from_mouse = true;
            // erase paths in scene
            draw_smoothed_path({}, &viewer->scene, QColor(),true);
            draw_global_target({}, &viewer->scene, true);
            draw_paths({}, &viewer->scene, true);
        });

        timer.start(params.PERIOD);
    }
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value())  {   /*qWarning() << "No data Lidar";*/ return; }
    auto points = res_.value();
    draw_lidar(points, params.LIDAR_LOW_DECIMATION_FACTOR);

    /// get robot pose
    if(auto res_ = get_robot_pose_and_change(); res_.has_value())
        this->robot_pose_and_change = res_.value();
    else
    {
        qWarning() << __FUNCTION__ << "No robot pose available. Returning. Check LidarOdometry component status";
        return;
    }

    /// get target from buffer
    Target target = Target::invalid();
    if(auto res = target_buffer.try_get(); res.has_value())
        target = res.value();
    else { fps.print("No Target - FPS:"); return;}
    //draw target
    draw_global_target(target.pos_eigen(), &viewer->scene);

    qInfo() << __FUNCTION__ << "Target: " << target.pos_eigen().x() << target.pos_eigen().y();

    /// check if target has been cancelled
    if(cancel_from_mouse or target.is_completed())
    {
        inject_ending_plan();
        cancel_from_mouse = false;
        return;
    }

    /// transform target to robot's frame
    target = transform_target_to_global_frame(robot_pose_and_change.first, target);    // transform target to robot's frame
    draw_point_color(target.pos_eigen(), &viewer->scene, false, QColor{"pink"});

    /// check if target has been reached
    if(robot_is_at_target(target))
    {
        inject_ending_plan();
        return;
    }

    /// compute path
    RoboCompGridder::Result returning_plan = compute_path(Eigen::Vector2f::Zero(), target);
    if (not returning_plan.valid)
    {
        inject_ending_plan();
        return;
    }
    else draw_paths(returning_plan.paths, &viewer->scene);

//    /// MPC
    RoboCompGridPlanner::TPlan final_plan;
    if(params.USE_MPC) /// convert plan to list of control actions calling MPC
        final_plan = convert_plan_to_control(returning_plan, target);
    this->lcdNumber_length->display((int)final_plan.path.size());
//
////    /// send plan to remote interface and publish it to rcnode
    if(not this->pushButton_stop->isChecked())
        send_and_publish_plan(final_plan);





    target_buffer.put(std::move(target));
//    try
//    {
//        lidarodometry_proxy->reset();   // empty buffer
//        std::optional<std::pair<Eigen::Transform<double, 3, 1>, Eigen::Transform<double, 3, 1>>> rp;
//        auto start = std::chrono::high_resolution_clock::now();
//        do
//        {   rp = get_robot_pose_and_change();
//            std::this_thread::sleep_for(std::chrono::milliseconds(50));
//        } while (rp->first.matrix().diagonal().sum() < 3.7 and
//                 std::chrono::duration_cast<std::chrono::milliseconds>(start - std::chrono::high_resolution_clock::now()).count() < 4000);
//        if(rp->first.matrix().diagonal().sum() < 3.7)
//            qWarning() << "Odometry not properly reset. Matrix trace too large: " << rp->first.matrix().diagonal().sum();
//    }
//    catch (const Ice::Exception &e)
//    {
//        std::cout << "[MOUSE] Error reading from LidarOdometry" << e << std::endl;
//        target = Target::invalid();
//        return;
//    };



    hz = fps.print("FPS:", 3000);
    this->lcdNumber_hz->display(this->hz);
}
//void SpecificWorker::compute()
//{
//    /// get robot pose
//    this->robot_pose_and_change = get_robot_pose_and_change();
//
//    /// get target from buffer
//    Target target = Target::invalid();
//    if(auto res = target_buffer.try_get(); res.has_value())
//        target = res.value();
//
//    if(cancel_from_mouse)
//    {
//        target.completed = true;
//        cancel_from_mouse = false;
//    }
//
//    /// A new valid target has arrived. We have to process it for one iteration and compute a TPlan for Bumper
//    RoboCompGridPlanner::TPlan returning_plan;
//    if (target.is_valid())
//    {
//        if(target.completed) // if robot arrived to target in the past iteration
//        {
//            returning_plan.valid = true;
//            returning_plan.subtarget.x = 0.f;
//            returning_plan.subtarget.y = 0.f;
//            returning_plan.controls.emplace_back(RoboCompGridPlanner::TControl{.adv=0.f, .side=0.f, .rot=0.f});
//            send_and_publish_plan(returning_plan);  // send zero plan to stop robot in bumper
//            draw_paths({}, &viewer->scene, true);   // erase paths
//            return;
//        }
//        auto original_target = target; // save original target to reinject it into the buffer
//
//        if(target.global)  // global = true -> target in global coordinates
//            target = transform_target_to_global_frame(robot_pose_and_change.first, target);    // transform target to robot's frame
//
//        /// check if target has been reached
//        if(not robot_is_at_target(target, original_target))
//        {
//            /// search for closest point to target in grid
//            if (auto closest_ = grid.closest_free(target.pos_qt()); not closest_.has_value())
//            {
//                qInfo() << __FUNCTION__ << "No closest_point found. Returning. Cancelling target";
//                //target_buffer.try_get();   // empty buffer so it doesn't enter a loop
//                target_buffer.put(Target{.active=true,
//                        .global=false,
//                        .completed=true,
//                        .point=Eigen::Vector2f::Zero(),
//                        .qpoint=QPointF{0.f, 0.f}}); // reinject zero target to notify bumper of target reached
//                return;
//            }
//            else target.set(closest_.value(), target.global); // keep current status of global
//            draw_global_target(target.pos_eigen(), &viewer->scene);
//
//            /// if target is not in line of sight, compute path
//            if (not grid.is_line_of_sigth_to_target_free(target.pos_eigen(), Eigen::Vector2f::Zero(),
//                                                         params.ROBOT_SEMI_WIDTH))
//                returning_plan = compute_plan_from_grid(target);
//            else
//                returning_plan = compute_line_of_sight_target(target);
//
//            if(params.USE_MPC) /// convert plan to list of control actions calling MPC
//                returning_plan = convert_plan_to_control(returning_plan, target);
//            // else CARROT
//            // new_returning_plan = mpc_buffer.get();  // get plan from MPC thread
//            // mpc_buffer.put(returning_plan); // empty buffer
//
//            if (not returning_plan.valid)   // no valid path found. Cancel target
//            {
//                qWarning() << __FUNCTION__ << "No valid path found. Cancelling target";
//                target_buffer.put(Target{.active=true,
//                        .global=false,
//                        .completed=true,
//                        .point=Eigen::Vector2f::Zero(),
//                        .qpoint=QPointF{0.f, 0.f}}); // reinject zero target to notify bumper of target reached
//                return;
//            }
//            //        else
//            //            /// adapt grid size and resolution for next iteration
//            //            adapt_grid_size(target, returning_plan.path);
//
//            /// send plan to remote interface and publish it to rcnode
//            send_and_publish_plan(returning_plan);
//        }
//    }
//    viewer->update();
//    fps.print("FPS:");
//}

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
RoboCompGridder::Result SpecificWorker::compute_path(const Eigen::Vector2f &source, const Target &target)
{

    
    RoboCompGridder::TPoint closest_point;
    Target target_plan = target;

    RoboCompGridder::Result returning_plan;
    try
    {
        auto dim = QRectF{params.gdim.left, params.gdim.top, params.gdim.width, params.gdim.height};

        if (not dim.contains(QPointF{target.pos_eigen().x(),target.pos_eigen().y()}))
        {
            closest_point = gridder_proxy->getClosestFreePoint(RoboCompGridder::TPoint{target.pos_eigen().x(), target.pos_eigen().y()});
            target_plan.pos_eigen() = Eigen::Vector2f{closest_point.x, closest_point.y};
        }

        if (gridder_proxy->LineOfSightToTarget(RoboCompGridder::TPoint{source.x(), source.y()},
                                               RoboCompGridder::TPoint{target_plan.pos_eigen().x(), target_plan.pos_eigen().y()},
                                               params.ROBOT_SEMI_WIDTH))
            returning_plan = compute_line_of_sight_target(target_plan);
        else
            returning_plan = compute_plan_from_grid(target_plan);
    }
    catch (const Ice::Exception &e)
    { std::cout << "Error reading Line of Sight from Gridder" << e << std::endl; }

    return returning_plan;
}
void SpecificWorker::inject_ending_plan()
{
    qWarning() << __FUNCTION__ << "Cancelling target due to robot at target, target cancelled by user or no valid path found";
    RoboCompGridPlanner::TPlan returning_plan;
    returning_plan.valid = true;
    returning_plan.controls.emplace_back(RoboCompGridPlanner::TControl{.adv=0.f, .side=0.f, .rot=0.f});
    send_and_publish_plan(returning_plan);  // send zero plan to stop robot in bumper
    target_buffer.try_get();   // empty buffer so it doesn't enter a loop
}
bool SpecificWorker::robot_is_at_target(const Target &target_)
{
    //qInfo() << __FUNCTION__ << "Checking if robot is at target [" << target_.pos_eigen().x() << target_.pos_eigen().y() << "]";
    if(target_.pos_eigen().norm() < params.MIN_DISTANCE_TO_TARGET)   // robot at target
    {
        qInfo() << __FUNCTION__ << "Target reached. Sending zero target";
        target_buffer.put(Target{.active=true,
                                    .global=false,
                                    .completed=true,
                                    .point=Eigen::Vector2f::Zero(),
                                    .qpoint=QPointF{0.f, 0.f}}); // reinject zero target to notify bumper of target reached
        return true;
    }
    else /// still moving. Reinject original target
    {
        target_buffer.put(Target{.active=true,
                                    .global=true,
                                    .completed=false,
                                    .point=target_.get_original(),
                                    .qpoint=QPointF{target_.get_original().x(), target_.get_original().y()},
                                    .original = target_.get_original(),
                                    .new_target = false});
        return false;
    }
}
std::optional<std::pair<Eigen::Transform<double, 3, 1>, Eigen::Transform<double, 3, 1>>> SpecificWorker::get_robot_pose_and_change()
{
    Eigen::Transform<double, 3, 1> robot_pose;
    Eigen::Transform<double, 3, 1> robot_change;
    try
    {
        //  auto pose = lidarodometry_proxy->getFullPoseMatrix();
        const auto pose_and_change = lidarodometry_proxy->getPoseAndChange();
        const auto &pose = pose_and_change.pose;
        const auto &change = pose_and_change.change;
        robot_pose.matrix() << pose.m00, pose.m01, pose.m02, pose.m03,
                               pose.m10, pose.m11, pose.m12, pose.m13,
                               pose.m20, pose.m21, pose.m22, pose.m23,
                               pose.m30, pose.m31, pose.m32, pose.m33;
        robot_change.matrix() << change.m00, change.m01, change.m02, change.m03,
                                 change.m10, change.m11, change.m12, change.m13,
                                 change.m20, change.m21, change.m22, change.m23,
                                 change.m30, change.m31, change.m32, change.m33;
    }
    catch (const Ice::Exception &e)
    {
        std::cout << "Error reading from LidarOdometry" << e << std::endl;
        return {};
    }
    return std::make_pair(robot_pose, robot_change);
}
SpecificWorker::Target SpecificWorker::transform_target_to_global_frame(const Eigen::Transform<double, 3, 1> &robot_pose, const Target &target)
{
    // transform target to robot's frame
    const Eigen::Vector2f &tp = target.get_original();
    //qInfo() << __FUNCTION__ <<  "Original target: " << tp.x() << tp.y();
    Eigen::Vector2f target_in_robot =
            (robot_pose.inverse().matrix() * Eigen::Vector4d(tp.x() / 1000.0, tp.y() / 1000.0, 0.0, 1.0) * 1000.0).head(2).cast<float>();
    //qInfo() << __FUNCTION__ <<  "transformed target: " << target_in_robot.x() << target_in_robot.y();
    Target target_aux = target;     // keep original target's attributes
    target_aux.set(target_in_robot);
    //qInfo() << __FUNCTION__ <<  "final target: " << target_aux.pos_eigen().x() << target_aux.pos_eigen().y();
    return target_aux;
}
RoboCompGridder::Result SpecificWorker::compute_line_of_sight_target(const Target &target)
{
    //qInfo() << __FUNCTION__;
    RoboCompGridder::Result returning_plan;
    returning_plan.valid = true;

    // fill path with equally spaced points from the robot to the target at a distance of consts.ROBOT_LENGTH
    int npoints = ceil(target.pos_eigen().norm() / params.ROBOT_SEMI_LENGTH);
    if(npoints > 1)
    {
        Eigen::Vector2f dir = target.pos_eigen().normalized();  // direction vector from robot to target
        RoboCompGridder::TPath plan;
        for (const auto &i: iter::range(npoints))
        {
            Eigen::Vector2f p = dir * (params.ROBOT_SEMI_LENGTH * i);
            plan.emplace_back(RoboCompGridder::TPoint(p.x(), p.y()));
        }
        returning_plan.paths.emplace_back(plan);
    }
    else
    {
        qWarning() << __FUNCTION__ << "Target too close. Cancelling target";
        returning_plan.valid = false;
        returning_plan.error_msg = "[compute_line_of_sight_target] Target too close";
    }
    return returning_plan;
}
RoboCompGridder::Result SpecificWorker::compute_plan_from_grid(const Target &target)
{
    //qInfo() << __FUNCTION__ << "Entering method";
    // static original path
    static std::vector<Eigen::Vector2f> original_path = {};

    // lambda to compute the length of a path
    //    auto path_length = [](const std::vector<Eigen::Vector2f> &path)
    //        {
    //            float acum = 0.f;
    //            for(const auto &pp: path | iter::sliding_window(2))
    //                acum += (pp[1]-pp[0]).norm();
    //            return acum;
    //        };

    if(target.is_new())  // if target is new, compute a new path
    {
        try
        {
            auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{0.f, 0.f},
                                                         RoboCompGridder::TPoint{target.pos_eigen().x(),target.pos_eigen().y()},
                                                         params.NUM_PATHS_TO_SEARCH,
                                                         true,
                                                         true);
            if(not result.valid or result.paths.empty())   //TODO: try a few times
            {
                qWarning() << __FUNCTION__ << "No path found while initializing current_path";
                current_path = original_path = {};
                return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] No path found while initializing current_path", .valid=false};
            }
            for(auto &&p : result.paths)
                qInfo() << __FUNCTION__ << " [NEW] New paths: " << p.size();
            qInfo() << __FUNCTION__ << " [NEW] Original path length: " << original_path.size();
            current_path.clear();
            for(const auto &p: result.paths.front())
                current_path.emplace_back(p.x, p.y);
            original_path = current_path;
            qInfo() << __FUNCTION__  << " [NEW] New current path: " << current_path.size() << original_path.size();
        }
        catch (const Ice::Exception &e)
        {
            std::cout << "Error reading plans from Gridder" << e << std::endl;
            current_path = original_path = {};
            return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] Error reading plan from Gridder", .valid=false};
        }
    }
    else // Transform target and check if it is blocked, transform current_path according to robot's new position
    {
        // Transforming
        std::vector<Eigen::Vector2f> local_path;
        local_path.reserve(current_path.size());
        // get the inverse of the robot current pose wrt the last reset
        const auto &inv = robot_pose_and_change.first.inverse().matrix();
        // it has to be the original path, not the current_path, since the robot_pose is accumulated from the last reset to lidar_odometry
        // the type in the lambda return must be explicit, otherwise it does not work
        std::ranges::transform(original_path, std::back_inserter(local_path), [inv](auto &p)
        { return Eigen::Vector2f{(inv * Eigen::Vector4d(p.x() / 1000.f, p.y() / 1000.f, 0.0, 1.0) * 1000.f).head(2).cast<float>()};});

        // remove points in the path that fall behind the robot
        std::vector<Eigen::Vector2f> trimmed;
        for(const auto &pp: local_path | iter::sliding_window(2))
            if (pp[0].norm() < pp[1].norm() )
                trimmed.emplace_back(pp[0]);
        current_path = trimmed;

        // check if it is blocked
        RoboCompGridder::TPath lpath; lpath.reserve(current_path.size());
        for (auto &&p: current_path)
                lpath.push_back(RoboCompGridder::TPoint{p.x(), p.y()});
        if(gridder_proxy->IsPathBlocked(lpath)) // not blocked. Transform current_path
            try
            {
                qInfo() << __FUNCTION__ << " Blocked. Computing new path";
                auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{0.f, 0.f},
                                                      RoboCompGridder::TPoint{target.pos_eigen().x(), target.pos_eigen().y()},
                                                      1, true, true);
                if(not result.valid or result.paths.empty())  //TODO: try a few times
                {
                    qWarning() << __FUNCTION__ << "No path found while initializing current_path";
                    current_path = original_path = {};
                    return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] No path found after blocked current_path", .valid=false};
                }
                else
                {
                    current_path.clear();
                    for (const auto &p: result.paths.front())
                        current_path.emplace_back(p.x, p.y);
                    original_path = current_path;
                    for(auto && [i, p] : result.paths | iter::enumerate)
                        qInfo() << __FUNCTION__ << " [BLOCKED] New path: " << i << p.size();
                    qInfo() << __FUNCTION__ << " [BLOCKED] Original path length: " << original_path.size();
                }
            }
            catch (const Ice::Exception &e)
            {
                std::cout << "Error reading plans from Gridder" << e << std::endl;
                current_path = original_path = {};
                return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] Error reading plan from Gridder", .valid=false};
            }
    }   // TODO:: check for ATAJO

    // if current_path not blocked and there is no other much shorter path available
//    if (not grid.is_path_blocked(current_path))
//    {
////                std::vector<std::pair<std::vector<Eigen::Vector2f>, float>> path_lenghts;
////                std::ranges::transform(paths, std::back_inserter(path_lenghts),
////                                       [path_length](auto &p) { return std::make_pair(p, path_length(p)); });
////                std::ranges::sort(path_lenghts, [](auto &a, auto &b) { return a.second < b.second; });
////
////                // and there is a shorter path. ATAJO
////                if (path_lenghts.front().second < path_length(current_path) * 0.6)
////                    current_path = path_lenghts.front().first;
////                // else do nothing;
//    } else
//    {
//        // else, current_path is NOT valid, so we need a fresh new one  BLOCK
//        auto paths = grid.compute_k_paths(Eigen::Vector2f::Zero(), target.pos_eigen(), params.NUM_PATHS_TO_SEARCH, params.MIN_DISTANCE_BETWEEN_PATHS);
//        current_path = paths.front();
//        qInfo() << "====================================================dasdf=============";
//    }
    //}

    /// fill returning_plan
    RoboCompGridder::Result returning_plan;
    if (not current_path.empty())
    {
        returning_plan.valid = true;
        RoboCompGridder::TPath path; path.reserve(current_path.size());
        for (auto &&p: current_path)
            path.emplace_back(p.x(), p.y());
        returning_plan.paths.emplace_back(path);
    }
    else  // EMPTY PATH
    {
        qWarning() << __FUNCTION__ << "Empty path";
        returning_plan.valid = false;
        returning_plan.error_msg = "Not path found in [compute_plan_from_grid]";
    }
    return returning_plan;
}
RoboCompGridPlanner::TPlan
SpecificWorker::convert_plan_to_control(const RoboCompGridder::Result &res, const Target &target)
{
    // call MPC to compute a new sequence of positions that meet the control constraints
    // get both the new sequence of positions and the new sequence of controls
    RoboCompGridPlanner::TPlan plan;
    plan.valid = true;
    plan.path.reserve(res.paths[0].size());
    for (auto &&p: res.paths[0])
        plan.path.emplace_back(p.x, p.y);
    try
    {
        auto mod_plan = gridplanner1_proxy->modifyPlan(plan);   // call MPC
        if(mod_plan.valid and not mod_plan.controls.empty())
        {
            //qInfo() << __FUNCTION__ <<  "Step: 0" << mod_plan.controls.front().adv << mod_plan.controls.front().side << mod_plan.controls.front().rot;
            plan = mod_plan;
            draw_smoothed_path(plan.path, &viewer->scene, params.SMOOTHED_PATH_COLOR);
        }
        else    // no valid plan found
        {
            qWarning() << __FUNCTION__ << "No valid optimization found in MPC. Returning original plan";
            draw_smoothed_path(plan.path, &viewer->scene, params.PATH_COLOR);
        }
    }
    catch (const Ice::Exception &e)
    {
        std::cout << __FUNCTION__ << "Error reading from MPC. Check component status. Returning original plan" << e << std::endl;
        draw_smoothed_path(plan.path, &viewer->scene, params.PATH_COLOR);
    }
    return plan;
}
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
// Check if target is within the grid and otherwise, compute the intersection with the border
Eigen::Vector2f SpecificWorker::compute_closest_target_to_grid_border(const Eigen::Vector2f &target)
{
    QRectF dim;
    try{ auto grid = gridder_proxy->getDimensions(); dim = QRectF{grid.left, grid.top, grid.width, grid.height}; }
    catch (const Ice::Exception &e)
    {
        std::cout << __FUNCTION__ << " Error reading from Gridder. Using grid dimension from PARAMS" << e << std::endl;
        dim = params.GRID_MAX_DIM;
    }
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
void SpecificWorker::send_and_publish_plan(const RoboCompGridPlanner::TPlan &plan)
{
    // Sending plan to remote interface
    try
    { gridplanner_proxy->setPlan(plan); }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__ << " Error setting valid plan" << e << std::endl; }

    // Publishing the plan to rcnode
//    try
//    { gridplanner_pubproxy->setPlan(plan); }
//    catch (const Ice::Exception &e)
//    { std::cout << __FUNCTION__ << " Error publishing valid plan" << e << std::endl; }
}

//////////////////////////////// Draw ///////////////////////////////////////////////////////
void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
    { scene->removeItem(p); delete p; }
    points.clear();

    if(erase_only) return;

    QPen pen = QPen(params.PATH_COLOR);
    QBrush brush = QBrush(params.PATH_COLOR);
    float s = 100;
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
        ptr->setPos(QPointF(p.x(), p.y()));
        points.push_back(ptr);
    }
}
void SpecificWorker::draw_smoothed_path(const RoboCompGridPlanner::Points &path, QGraphicsScene *scene,
                                        const QColor &color, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
    { scene->removeItem(p); delete p; }
    points.clear();

    if(erase_only) return;

    QPen pen = QPen(color);
    QBrush brush = QBrush(color);
    float s = 100;
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
        ptr->setPos(QPointF{p.x, p.y});
        points.push_back(ptr);
    }
}
void SpecificWorker::draw_paths(const RoboCompGridder::TPaths &paths, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    static QColor colors[] = {QColor("green"), QColor("blue"), QColor("red"), QColor("orange"), QColor("magenta"), QColor("cyan")};
    for(auto p : points)
    { scene->removeItem(p); delete p; }
    points.clear();

    if(erase_only) return;

    float s = 80;
    for(const auto &[i, path]: paths | iter::enumerate)
    {
        // pick a consecutive color
        QBrush brush = QBrush(colors[i]);
        for(const auto &p: path)
        {
            auto ptr = scene->addEllipse(-s/2.f, -s/2.f, s, s, colors[i], brush);
            ptr->setPos(QPointF(p.x, p.y));
            points.push_back(ptr);
        }
    }
}
void SpecificWorker::draw_global_target(const Eigen::Vector2f &point, QGraphicsScene *scene, bool erase_only)
{
    static QGraphicsEllipseItem *item = nullptr;
    if(item != nullptr)
        scene->removeItem(item);

    if(erase_only) return;

    QPen pen = QPen(params.TARGET_COLOR);
    QBrush brush = QBrush(params.TARGET_COLOR);
    int s = 150;
    item = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
    item->setPos(QPointF(point.x(), point.y()));
}

void SpecificWorker::draw_point_color(const Eigen::Vector2f &point, QGraphicsScene *scene, bool erase_only, QColor color)
{
    static QGraphicsEllipseItem *item = nullptr;
    if(item != nullptr)
        scene->removeItem(item);

    if(erase_only) return;

    QPen pen = QPen(color);
    QBrush brush = QBrush(color);
    int s = 150;
    item = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
    item->setPos(QPointF(point.x(), point.y()));
}

void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &points, int decimate)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points)
    {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_points.clear();

    QPen pen = QPen(params.LIDAR_COLOR);
    QBrush brush = QBrush(params.LIDAR_COLOR);
    for (const auto &[i, p]: points |iter::enumerate)
    {
        // skip 2 out of 3 points
        if(i % decimate == 0)
        {
            auto o = viewer->scene.addRect(-20, 20, 40, 40, pen, brush);
            o->setPos(p.x(), p.y());
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
void SpecificWorker::SegmentatorTrackingPub_setTrack (RoboCompVisualElementsPub::TObject target)
{
    Target t;
    QRectF dim;

    try{ auto grid = gridder_proxy->getDimensions(); dim = QRectF{grid.left, grid.top, grid.width, grid.height}; }
    catch (const Ice::Exception &e)
    {
        std::cout << __FUNCTION__ << " Error reading from Gridder. Using grid dimension from PARAMS" << e << std::endl;
        dim = params.GRID_MAX_DIM;
    }

    auto x_pos = target.attributes.find("x_pos");
    auto y_pos = target.attributes.find("y_pos");

    if ( x_pos == target.attributes.end() and y_pos == target.attributes.end()) 
    {
        //qInfo() << __FUNCTION__ << "No element selected to track in target";
        return;
    }

    //std::cout << x_pos->first << " " << std::stof(x_pos->second) << std::endl;
    //std::cout << y_pos->first << " " << std::stof(y_pos->second) << std::endl;

    auto pos = Eigen::Vector2f{std::stof(x_pos->second), std::stof(y_pos->second)};
//    if(pos.isMuchSmallerThan(Eigen::Vector2f::Ones(), 1))
//    {
//        qInfo() << __FUNCTION__ << "Target element at position [0, 0], Returning";
//        return;
//    }
    if(dim.contains(QPointF{pos.x(), pos.y()}))
    {
        t.set(pos, false); /// false = in robot's frame; true = in global frame
        t.set_original(pos);
        t.set_new(true);
    }
    else
    {
        auto pp = compute_closest_target_to_grid_border(pos);
        t.set(pp, false);
        t.set_original(pp);
        t.set_new(true);
    }
    //Reset lidar odometry
    try
    { lidarodometry_proxy->reset(); }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__ << " Error resetting LidarOdometry" << e << std::endl; }

    target_buffer.put(std::move(t));
}

/// check if params.ELAPSED_TIME has passed since last path computation. We update path once very ELAPSED_TIME_BETWEEN_PATH_UPDATES mseconds
//    static auto last_time = std::chrono::steady_clock::now();
//    auto now = std::chrono::steady_clock::now();
//    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);
//    if (elapsed_time.count() > params.ELAPSED_TIME_BETWEEN_PATH_UPDATES) // if enough time has passed, compute a new path. It should check for a sudden blockage
//    {
//        // compute the K optimal paths that differ "min_max_dist" among them using the Yen algorithm and the max_distance approximation to Frechet distance
//        std::vector<std::vector<Eigen::Vector2f>> paths;
//        paths = grid.compute_k_paths(Eigen::Vector2f::Zero(), target.pos_eigen(),
//                                     params.NUM_PATHS_TO_SEARCH, params.MIN_DISTANCE_BETWEEN_PATHS);
//        if (paths.empty())
//        {
//            qWarning() << __FUNCTION__ << "No paths found in compute_k_paths";
//            return RoboCompGridPlanner::TPlan{.valid=false};
//        }
//        // compute a sorted vector with the <max_distances, index> of current_path to all paths
//        std::vector<std::pair<float, int>> distances_to_current;
//        for (auto &&[i, path]: paths | iter::enumerate)
//            distances_to_current.emplace_back(max_distance(current_path, path), i);
//        std::ranges::sort(distances_to_current, [](auto &a, auto &b)
//        { return a.first < b.first; });
//
//        // assign current_path to the closest path
//        current_path = paths[distances_to_current.front().second];
//        original_path = current_path;
//        //draw_paths(paths, &viewer->scene);
//    } else  // if not enough time has passed, transform current_path to global frame
//    {
