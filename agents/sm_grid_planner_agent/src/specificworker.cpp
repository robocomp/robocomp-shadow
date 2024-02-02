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
#include <cppitertools/reversed.hpp>


/**
* \brief Default constructor
*/

SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
    this->startup_check_flag = startup_check;
	// Uncomment if there's too many debug messages
	// but it removes the possibility to see the messages
	// shown in the console with qDebug()
	QLoggingCategory::setFilterRules("*.debug=false\n");

}
/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
    std::cout << "Destroying SpecificWorker" << std::endl;
	std::cout << "Destroying SpecificWorker" << std::endl;
	//G->write_to_json_file("./"+agent_name+".json");
	auto grid_nodes = G->get_nodes_by_type("grid");
	for (auto grid : grid_nodes)
	{
		G->delete_node(grid);
	}
	G.reset();
}
bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
    try
    {
        this->params.DISPLAY = params.at("display").value == "True" or params.at("display").value == "true";
	agent_name = params.at("agent_name").value;
	agent_id = stoi(params.at("agent_id").value);
	tree_view = params.at("tree_view").value == "true";
	graph_view = params.at("graph_view").value == "true";
	qscene_2d_view = params.at("2d_view").value == "true";
	osg_3d_view = params.at("3d_view").value == "true";
    }
    catch (const std::exception &e)
    { qWarning("Error reading config params. Withdrawing to defaults"); }

    qInfo() << "Config parameters:";
    qInfo() << "    display" << this->params.DISPLAY;

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
	// create graph
	G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
    rt = G->get_rt_api();
	std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;  

	//dsr update signals
	connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
	connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
	connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
	connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

	// Graph viewer
	using opts = DSR::DSRViewer::view;
	int current_opts = 0;
	opts main = opts::none;
	if(tree_view)
	{
	    current_opts = current_opts | opts::tree;
	}
	if(graph_view)
	{
	    current_opts = current_opts | opts::graph;
	    main = opts::graph;
	}
	if(qscene_2d_view)
	{
	    current_opts = current_opts | opts::scene;
	}
	if(osg_3d_view)
	{
	    current_opts = current_opts | opts::osg;
	}
	graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
	setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

	/***
	Custom Widget
	In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
	The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
	either with a QtDesigner or directly from scratch in a class of its own.
	The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
	***/

    widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));

        // mouse
        connect(widget_2d, &DSR::QScene2dViewer::mouse_left_click, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New click left arrived:" << p;
            // mouse_click eigen point from QPointF p
            mouse_click = Eigen::Vector2f(p.x(), p.y());



            // Check item at position corresponding to person representation (person can't be selected if cone layer is superposed)
            // TODO:  selectedItem->setZValue() changes the order of the items in the scene. Use this to avoid the cone layer to be superposed
//            QList<QGraphicsItem *> itemsAtPosition = widget_2d->scene.items(p, Qt::IntersectsItemShape, Qt::DescendingOrder, QTransform());
//            QGraphicsItem *selectedItem = nullptr;
//
//            if(auto res = std::ranges::find_if(itemsAtPosition, [p](auto it)
//                { return (Eigen::Vector2f(it->pos().x(), it->pos().y()) - Eigen::Vector2f(p.x(), p.y())).norm() < 100;}); res != itemsAtPosition.end())
//                selectedItem = *res;
//            else return;

        });
        //Right click
        connect(widget_2d, &DSR::QScene2dViewer::mouse_right_click, [this](int x, int y, uint64_t id)
        {
            qInfo() << "[MOUSE] New right click arrived:";
        });


        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // reset lidar odometry
        try{ lidarodometry_proxy->reset();
            std::cout << __FUNCTION__ << "Odometry Reset" << std::endl;}
        catch (const Ice::Exception &e) { std::cout << "Error reading from LidarOdometry" << e << std::endl;}

        //Set grid dimensions
        try
        {
            gridder_proxy->setGridDimensions(RoboCompGridder::TDimensions{-7500, -7500, 15000, 15000});
            std::cout << __FUNCTION__ << "Setting Grid Dimm" << std::endl;
        }
        catch (const Ice::Exception &e)
        {
            std::cout << "Error setting grid dim" << e << std::endl;
            std::terminate();
        }
        if(not params.DISPLAY)
            hide();
        timer.start(params.PERIOD);
    }
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value())  {   /*qWarning() << "No data Lidar";*/ return; }
    auto points = res_.value();

    draw_scenario(points, &widget_2d->scene);


    /// get robot pose //TODO: move odometry
    RobotPose robot_pose_and_change;
    if(auto res = get_robot_pose_and_change(); res.has_value())
    {
        robot_pose_and_change = res.value();
    }
    else
    {
        qWarning() << __FUNCTION__ << "No robot pose available. Returning. Check LidarOdometry component status";
        return;
    }
    ////DEBUG ODOMETRY
    /// transform mouse_click to global frame
    mouse_click = (robot_pose_and_change.first.inverse().matrix() * Eigen::Vector4d(mouse_click.x() / 1000.0, mouse_click.y() / 1000.0, 0.0, 1.0) * 1000.0).head(2).cast<float>();


    /// get target from buffer
    Target target = Target::invalid();
    if(auto res = target_buffer.try_get(); res.has_value())
        target = res.value();
    else { fps.print("No Target "); return;}
    //print target
    qInfo() << __FUNCTION__ <<  "Target: " << target.pos_eigen().x() << target.pos_eigen().y();

    /// transform target to robot's frame
    target = transform_target_to_global_frame(robot_pose_and_change.first, target);    // transform target to robot's frame

    qInfo() << __FUNCTION__ <<  "Target: " << target.pos_eigen().x() << target.pos_eigen().y();
    /// if robot is not tracking and at target, stop
    if(not target.is_tracked() and robot_is_at_target(target))
    {
        inject_ending_plan();
        return;
    }  /// check if target has been reached. Is reached, inject zero plan, empty buffer and return

    /// compute path
    RoboCompGridder::Result returning_plan = compute_path(Eigen::Vector2f::Zero(), target, robot_pose_and_change);
    if (not returning_plan.valid)
    {
        inject_ending_plan();
        return;
    }
    //else  if(params.DISPLAY)(draw_paths(returning_plan.paths, &viewer->scene));

    /// if target is tracked, set the real target at a distance of params.TRACKING_DISTANCE_TO_TARGET from the original target
    if(target.is_tracked())
    {
        // move forward through path until the distance to the target is equal to params.TRACKING_DISTANCE_TO_TARGET
        float acum = 0.f;
        bool success = false;
        for(auto &&pp: returning_plan.paths.front() |
                                     iter::reversed |
                                     iter::sliding_window(2) |
                                     iter::filter([acum, dist = params.TRACKING_DISTANCE_TO_TARGET](auto &pp) mutable
                                            { acum += (Eigen::Vector2f{pp[0].x, pp[0].y} - Eigen::Vector2f{pp[1].x, pp[1].y}).norm();
                                              return acum > dist;}))
        {
                target.set(Eigen::Vector2f{pp[1].x, pp[1].y}, false);
                success = true;
                break;
        }
        if(not success)
        {
            qWarning() << __FUNCTION__ << "Target too close. Cancelling target";
            inject_ending_plan();
            return;
        }
    }


    /// MPC
    RoboCompGridPlanner::TPlan final_plan;
    if(params.USE_MPC) /// convert plan to list of control actions calling MPC
        final_plan = convert_plan_to_control(returning_plan, target);
//    if(params.DISPLAY)(this->lcdNumber_length->display((int)final_plan.path.size()));

   /// send plan to remote interface and publish it to rcnode
//    if(not this->pushButton_stop->isChecked())
    send_and_publish_plan(final_plan);

    /// reinject target to buffer
    target_buffer.put(std::move(target));

    hz = fps.print("FPS:", 3000);
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
RoboCompGridder::Result SpecificWorker::compute_path(const Eigen::Vector2f &source, const Target &target, const RobotPose &robot_pose_and_change)
{
    RoboCompGridder::Result returning_plan;
    try
    { returning_plan = compute_plan_from_grid(target, robot_pose_and_change);
    //print target
    std::cout << "Target: " << target.pos_eigen().x() << " " << target.pos_eigen().y() << std::endl;
    }
    catch (const Ice::Exception &e)
    { std::cout << "Error reading path from Gridder" << e << std::endl; }
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
    qInfo() << __FUNCTION__ <<  "final target: " << target_aux.pos_eigen().x() << target_aux.pos_eigen().y();

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
RoboCompGridder::Result SpecificWorker::compute_plan_from_grid(const Target &target, const RobotPose &robot_pose_and_change)
{
    static std::vector<Eigen::Vector2f> original_path = {};      // static original path
    if(target.is_new())  // if target is new, compute a new path
    {
        try
        {
            auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{.x=0.f, .y=0.f, .radius=400 },    // TODO: move to params
                                                         RoboCompGridder::TPoint{.x=target.pos_eigen().x(),
                                                                                       .y=target.pos_eigen().y(),
                                                                                       .radius=400},
                                                         params.NUM_PATHS_TO_SEARCH,
                                                         true,
                                                         true);
            //qInfo() << __FUNCTION__ <<  "New target arrived. Computing new path. Obtained " << result.paths.size() << " paths";
            if(params.DISPLAY) draw_paths(result.paths, &viewer->scene);
            if(not result.valid or result.paths.empty())   //TODO: try a few times
            {
                qWarning() << __FUNCTION__ << " Message from Gridder:" + QString::fromStdString(result.error_msg);
                current_path = original_path = {};
                return RoboCompGridder::Result{.error_msg = result.error_msg, .valid=false};
            }
            else    // path found. Copy to current_path and original_path
            {
                //Print message
                qInfo() << __FUNCTION__ << "New path: ";
                current_path.clear();
                for (const auto &p: result.paths.front())
                    current_path.emplace_back(p.x, p.y);
                original_path = current_path;
            }
        }
        catch (const Ice::Exception &e)
        {
            std::cout << __FUNCTION__ << " [CATCH] Error reading plans from Gridder in new path " << std::endl;
            std::cout << e << std::endl;
            current_path = original_path = {};
            return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] Error reading plan from Gridder in new path", .valid=false};
        }
    }
    else // Known target. Transform target and check if it is blocked, transform current_path according to robot's new position
    {
        //Print known target
        qInfo() << __FUNCTION__ << "Known target: " << target.pos_eigen().x() << target.pos_eigen().y();
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
        if(gridder_proxy->IsPathBlocked(lpath))     // blocked. Compute new path
            try
            {
                qInfo() << __FUNCTION__ << " Blocked. Computing new path";
                auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{.x=0.f, .y=0.f, .radius=250},
                                                              RoboCompGridder::TPoint{.x=target.pos_eigen().x(),
                                                                                            .y=target.pos_eigen().y(),
                                                                                            .radius=400},
                                                           1, true, true);
                if(not result.valid or result.paths.empty())  //TODO: try a few times
                {
                    qWarning() << __FUNCTION__ << " No path found after blocked current_path";
                    current_path = original_path = {};
                    return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] No path found after blocked current_path", .valid=false};
                }
                else    // path found. Assign to current_path and original_path
                {
                    current_path.clear();
                    for (const auto &p: result.paths.front())
                        current_path.emplace_back(p.x, p.y);
                    original_path = current_path;
                    //for(auto && [i, p] : result.paths | iter::enumerate)
                    //    qInfo() << __FUNCTION__ << " [BLOCKED] New path: " << i << p.size();
                    //qInfo() << __FUNCTION__ << " [BLOCKED] Original path length: " << original_path.size();
                }
            }
            catch (const Ice::Exception &e)
            {
                std::cout << " [CATCH] Error reading plans from Gridder in blocked path" << std::endl;
                std::cout << e << std::endl;
                current_path = original_path = {};
                return RoboCompGridder::Result{.error_msg = "[compute_plan_from_grid] Error reading plan from Gridder in blocked path", .valid=false};
            }
    }   // TODO:: check for ATAJO

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
        qWarning() << __FUNCTION__ << " No paths found. Returning not valid plan";
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
float SpecificWorker::distance_between_paths(const std::vector<Eigen::Vector2f> &pathA, const std::vector<Eigen::Vector2f> &pathB)
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
void SpecificWorker::draw_point(const Eigen::Vector2f &point, QGraphicsScene *scene )
{
    static QGraphicsEllipseItem* point_item;

    if(point_item != nullptr)
    { scene->removeItem(point_item); delete point_item; }

    // Draw point in the scene using RED color
    QPen pen = QPen(QColor("red"));
    QBrush brush = QBrush(QColor("red"));
    float s = 100;
    point_item = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
    point_item->setPos(QPointF(point.x(), point.y()));

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



///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////// DRAW /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &points, int decimate, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points)
    { scene->removeItem(p); delete p; }
    draw_points.clear();

    QPen pen = QPen(params.LIDAR_COLOR);
    QBrush brush = QBrush(params.LIDAR_COLOR);
    for (const auto &[i, p]: points |iter::enumerate)
    {
        // skip 2 out of 3 points
        if(i % decimate == 0)
        {
            auto o = widget_2d->scene.addRect(-20, 20, 40, 40, pen, brush);
            o->setPos(p.x(), p.y());
            draw_points.push_back(o);
        }
    }
}
void SpecificWorker::draw_room(const RoboCompVisualElementsPub::TObject &obj)
{
    //check if obj.attributes.contains the key name if it does print the value
    if(obj.attributes.contains("name"))
    {
        if (obj.attributes.at("name") == "room")
        {
            //save the attributes of the room width, depth,height,center_x,center_y,rotation
            float width = std::stof(obj.attributes.at("width"));
            float depth = std::stof(obj.attributes.at("depth"));
            //float height = std::stof(obj.attributes.at("height"));
            float center_x = std::stof(obj.attributes.at("center_x"));
            float center_y = std::stof(obj.attributes.at("center_y"));
            float rotation = std::stof(obj.attributes.at("rotation"));

            static QGraphicsRectItem *item = nullptr;
            if (item != nullptr)
                widget_2d->scene.removeItem(item);

            item = widget_2d->scene.addRect(-width / 2, -depth / 2, width, depth, QPen(QColor("black"),50));
            item->setPos(QPointF(center_x, center_y));
            item->setRotation(rotation+90);
        }
        else
            qWarning() << __FUNCTION__ << " Error. The object by parameter is not a room";
    }
    else
        qWarning() << __FUNCTION__ << " Error. The object does not contain the key name";

}
void SpecificWorker::draw_paths(const RoboCompGridder::TPaths &paths, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    static QColor colors[] = {QColor("blue"), QColor("red"), QColor("orange"), QColor("magenta"), QColor("cyan")};
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
void SpecificWorker::draw_room_graph(QGraphicsScene *scene)
{
    static std::pair<uint64_t , QGraphicsItem*> draw_room = {0, nullptr};  // holds draw element

    // admissibility conditions
    auto room_nodes = G->get_nodes_by_type("room");
    if(room_nodes.empty()) return;

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    auto room_node = room_nodes.front();
    if (auto edge_robot = rt->get_edge_RT(robot_node.value(), room_node.id()); edge_robot.has_value())
    {
        auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
        auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
        auto width_ = G->get_attrib_by_name<width_att>(room_node);
        auto depth_ = G->get_attrib_by_name<depth_att>(room_node);
        if (tr.has_value() and rot.has_value() and width_.has_value() and depth_.has_value())
        {
            auto x = tr.value().get()[0], y = tr.value().get()[1];
            auto ang = rot.value().get()[2];
            if (draw_room.first == room_node.id())  // update room
            {
                draw_room.second->setPos(x, y);
                draw_room.second->setRotation(ang + 90);
            }
            else   // add it
            {
                auto item = scene->addRect(-width_.value()/2.f, -depth_.value()/2.f, width_.value(), depth_.value(),
                                           QPen(QColor("black"), 50));
                item->setPos(x, y);
                item->setRotation(ang + 90);
                draw_room = {room_node.id(), item};
            }
        }
    }
}
void SpecificWorker::draw_objects_graph(QGraphicsScene *scene)
{
    static std::map<uint64_t , QGraphicsItem*> draw_objects; // holds draw elements

    // admissibility conditions
    auto object_nodes = G->get_nodes_by_type("chair");
    if(object_nodes.empty()) return;

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    for(const auto &object_node : object_nodes)
    {
        if (auto edge_robot = rt->get_edge_RT(robot_node.value(), object_node.id()); edge_robot.has_value())
        {
            auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
            auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
            //auto width_ = G->get_attrib_by_name<width_att>(object_node);
            //auto depth_ = G->get_attrib_by_name<depth_att>(object_node);
            if (tr.has_value() and rot.has_value() /*and width_.has_value() and depth_.has_value()*/)
            {
                auto x = tr.value().get()[0], y = tr.value().get()[1];
                auto ang = rot.value().get()[2];
                if (draw_objects.contains(object_node.id()))  // update room
                {
                    draw_objects.at(object_node.id())->setPos(x, y);
                    draw_objects.at(object_node.id())->setRotation(ang + 90);
                } else   // add it
                {
                    auto width = 400, depth = 400;  // TODO: get width and depth from graph
                    auto item = scene->addRect(-width / 2.f, -depth / 2.f, width, depth,
                                               QPen(QColor("magenta"), 50));
                    item->setPos(x, y);
                    item->setRotation(ang + 90);    // TODO: orient with wall
                    draw_objects[object_node.id()] =  item;
                }
            }
        }
    }
    // check if object is not in the graph, but it is in the draw_map. Remove it.
    for (auto it = draw_objects.begin(); it != draw_objects.end();)
    {
        if (std::ranges::find_if(object_nodes, [&it, this](const DSR::Node &p)
        { return p.id() == it->first;}) == object_nodes.end())
        {
            scene->removeItem(it->second);
            delete it->second;
            it = draw_objects.erase(it);
        }
        else ++it;
    }
}
void SpecificWorker::draw_people_graph(QGraphicsScene *scene)
{
    float s = 500;
    auto color = QColor("orange");

    // draw people
    static std::map<uint64_t , QGraphicsItem *> draw_map;
    auto people_nodes = G->get_nodes_by_type("person");

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    for(const auto &person_node : people_nodes)
    {
        auto id = person_node.id();
        if (auto edge_robot = rt->get_edge_RT(robot_node.value(), id); edge_robot.has_value())
        {
            auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
            auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
            if (tr.has_value() and rot.has_value())
            {
                auto x = tr.value().get()[0], y = tr.value().get()[1];
                auto ang = qRadiansToDegrees(rot.value().get()[2]);
                if (draw_map.contains(id))
                {
                    draw_map.at(id)->setPos(x, y);
                    draw_map.at(id)->setRotation(ang);
                }
                else    // add person
                {
                    auto circle = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("black"), 20),
                                                    QBrush(color));
                    auto line = scene->addLine(0, 0, 0, 250,
                                               QPen(QColor("black"), 20, Qt::SolidLine, Qt::RoundCap));
                    line->setParentItem(circle);
                    circle->setPos(x, y);
                    circle->setRotation(ang);
                    circle->setZValue(100);  // set it higher than the grid so it can be selected with mouse
                    draw_map.emplace(id, circle);
                }
            }
        }
    }
    // check person is not in the graph, but it is in the draw_map. Remove it.
    for (auto it = draw_map.begin(); it != draw_map.end();)
    {
        if (std::ranges::find_if(people_nodes, [&it, this](const DSR::Node &p)
        { return p.id() == it->first;}) == people_nodes.end())
        {
            scene->removeItem(it->second);
            delete it->second;
            it = draw_map.erase(it);
        }
        else ++it;  // removes without dismantling the iterators
    }
}
void SpecificWorker::draw_scenario(const std::vector<Eigen::Vector3f> &points, QGraphicsScene *scene)
{

    draw_lidar(points, 3, scene);
//    draw_room_graph(scene);
//    draw_objects_graph(scene);
    draw_people_graph(scene);

    draw_point(mouse_click, scene );
}
///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////     SLOTS     ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

void SpecificWorker::modify_node_slot(std::uint64_t, const std::string &type)
{

}
void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{

}
void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    if(type == "following_action")
    {
        // get robot node
        if (auto robot_node = G->get_node("Shadow"); robot_node.has_value())
        {
            if(auto edge = rt->get_edge_RT(robot_node.value(), to); edge.has_value())
            {
                if(auto  target_pose = G->get_attrib_by_name<rt_translation_att>(edge.value()); target_pose.has_value())
                {
                    auto target_pose_val = target_pose.value().get();
                    // get target x, y
                    auto target_x = target_pose_val[0];
                    auto target_y = target_pose_val[1];

                    // Create target
                    Target t;
                    t.active = true;
                    t.new_target = true;
                    t.point = Eigen::Vector2f{target_x, target_y};
                    //put t in target_buffer
                    target_buffer.put(std::move(t));
                    //print target
                    qInfo() << __FUNCTION__ << "Slot Target: " << target_x << target_y;
                }
            }
        }
    }
}
void SpecificWorker::modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names)
{

}
void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{

}
void SpecificWorker::del_node_slot(std::uint64_t from)
{

}
////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// Interfaces
///////////////////////////////////////////////////////////////////////////////////////////////
/// SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack (RoboCompVisualElementsPub::TObject target)
{
    //Cancel target from intention_predictor publisher
    if(target.id == -1)
    {
        cancel_from_mouse = true;
        return;
    }
    
    Target t;
    QRectF dim;
    try{ auto grid = gridder_proxy->getDimensions(); dim = QRectF{grid.left, grid.top, grid.width, grid.height}; }
    catch (const Ice::Exception &e)
    {
        std::cout << __FUNCTION__ << " Error reading from Gridder. Using grid dimension from PARAMS" << e << std::endl;
        dim = params.GRID_MAX_DIM;
    }

    // get target position from string attributes
    auto x_pos = target.attributes.find("x_pos");
    auto y_pos = target.attributes.find("y_pos");
    if ( x_pos == target.attributes.end() and y_pos == target.attributes.end())
    {
        //qInfo() << __FUNCTION__ << "No element selected to track in target";
        return;
    }

    auto pos = Eigen::Vector2f{std::stof(x_pos->second), std::stof(y_pos->second)};
    if(dim.contains(QPointF{pos.x(), pos.y()}))
    {
        t.set(pos, false);
        t.set_original(pos);
        t.set_new(true);
        t.set_being_tracked(true);
    }
    else
    {
        auto pp = compute_closest_target_to_grid_border(pos);
        t.set(pp, false);
        t.set_original(pp);
        t.set_new(true);
        t.set_being_tracked(true);
    }

    // Reset lidar odometry
    try
    { lidarodometry_proxy->reset(); }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__ << " Error resetting LidarOdometry" << e << std::endl; }

    target_buffer.put(std::move(t));
}

//////////////////////////////////////////////////////////////////////////////////////////////
double SpecificWorker::path_length(const vector<Eigen::Vector2f> &path) const
{
    double acum = 0.0;
    if(path.size() > params.MAX_PATH_STEPS)
    {
        qWarning() << __FUNCTION__ << "Path too long. Returning 0";
        return 0.0;
    }
    for(const auto &pp: path | iter::sliding_window(2))
        acum += (pp[1]-pp[0]).norm();
    return acum;
}

//////////////////////////////////// TESTING  ///////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}
