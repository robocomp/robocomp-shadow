/*
 *    Copyright (C) 2024 by YOUR NAME HERE
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
#include <cppitertools/reversed.hpp>
#include <cppitertools/zip.hpp>
/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif
        this->loadRT = this->configLoader.get<bool>("setRTPoseDSR");

		//dsr update signals
		//connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/


		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period,
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/
	    
		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();
        QLoggingCategory::setFilterRules("*.debug=false\n");
        if (auto error = statemachine.errorString(); error.length() > 0)
        {
			qWarning() << error;
			throw error;
		}
	}
}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
}

void SpecificWorker::initialize()
{
    // Graph viewer
    using opts = DSR::DSRViewer::view;
    int current_opts = 0;
    opts main = opts::none;
    if(this->configLoader.get<bool>("ViewAgent.tree"))
    {
        current_opts = current_opts | opts::tree;
    }
    if(this->configLoader.get<bool>("ViewAgent.graph"))
    {
        current_opts = current_opts | opts::graph;
        main = opts::graph;
    }
    if(this->configLoader.get<bool>("ViewAgent.2d"))
    {
        current_opts = current_opts | opts::scene;
    }
    if(this->configLoader.get<bool>("ViewAgent.3d"))
    {
        current_opts = current_opts | opts::osg;
    }

    graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
    setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

    // get pointer to 2D viewer
    widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(DSR::DSRViewer::view::scene));

    room_widget = new CustomWidget();
    widget_2d->set_draw_axis(true);
    widget_2d->scale(0.08, 0.08);
    graph_viewer->add_custom_widget_to_dock("room view", room_widget);

    // Lidar thread
    // read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
    std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

	// Local apis
    rt_api = G->get_rt_api();
    inner_api = G->get_inner_eigen_api();

    // check for current room
    auto current_edges = G->get_edges_by_type("current");
    if(current_edges.empty())
    {qWarning() << __FUNCTION__ << " No current edges in graph"; return;}

    auto room_node_ = G->get_node(current_edges[0].to());
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room level in graph"; return; }
    auto room_node = room_node_.value();
    params.current_room = room_node.name();
}

/// ProtoCODE
// 1. Read LiDAR
// 2. Check if there is a "has_intention" edge in G. If so get the intention edge
// 3. Check if there is offset position, final orientation and thresholds in the intention edge
// 4. Get translation vector from offset in target node
// 5. Check if robot distance to target offset is lower than the threshold in the intention edge, then stop
// 6. Else, if target position is in line of sight, then send velocity commands to bumper.
//    else, if there is current room,
//             if there is no current_path or the target_position in the room has changed,
//                ask room_gridder for a path to the target position
//             else send velocity commands to the bumper for the next path step
//          ask local_gridder for a path to the target position

void SpecificWorker::compute()
{
    std::vector<Eigen::Vector3f> lidar_points;
    try
    {
        auto data = lidar3d_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_LOW,
                                                               params.MAX_LIDAR_LOW_RANGE,
                                                               params.LIDAR_LOW_DECIMATION_FACTOR);

        std::vector<Eigen::Vector3f> eig_data(data.points.size());
        for (const auto &[i, p]: data.points | iter::enumerate)
            if (p.distance2d > 500) eig_data[i] = {p.x, p.y, p.z};
             /// read LiDAR
        lidar_points = eig_data;
        draw_lidar_in_robot_frame(lidar_points, &widget_2d->scene, 50);
        // draw robot and lidar in room frame
        draw_room(&room_widget->viewer->scene, lidar_points);
    }
    catch (const Ice::Exception &e)
    { std::cout << "Error reading from Lidar3D " << e << std::endl; return; }

    /// check if pushButton_stop is checked and stop robot and return if so
    if(room_widget->ui->pushButton_stop->isChecked())
    { qWarning() << __FUNCTION__ << " Stop button checked"; stop_robot(); return; }

    /// check if there is a "has_intention" edge in G. If so get the intention edge
    DSR::Edge intention_edge;
    if(const auto intention_edge_ =  there_is_intention_edge_marked_as_active(); not intention_edge_.has_value() )
    {
        std::cout << __FUNCTION__ << " There is no intention edge. Returning." << std::endl;
        stop_robot();
        return;
    }
    else intention_edge = intention_edge_.value();

    /// check if there is a "state_att" attribute in the intention edge
    const auto intention_status_ = G->get_attrib_by_name<state_att>(intention_edge);
    if(not intention_status_.has_value())
    {
        std::cout << __FUNCTION__ << " Error getting intention status. Returning." << std::endl;
        return;
    }

    auto intention_status = intention_status_.value();


    /// Get translation vector from target node with offset applied to it
    Eigen::Vector3d vector_to_target = Eigen::Vector3d::Zero();

    // Check if "current_path" node exists in G
    if(not current_path_room.empty())
    {
        //Transform path to robot using RT API
        std::vector<Eigen::Vector3d> path_in_robot;
        path_in_robot.reserve(current_path_room.size());

        for(const auto &[i, point]: current_path_room | iter::enumerate)
        {
            auto point_in_robot = get_translation_vector_to_point(point);
            if(not point_in_robot.has_value())
            {qWarning() << __FUNCTION__ << "Error getting translation from the robot to the path point"; continue;}
            path_in_robot.push_back(point_in_robot.value());
        }
        vector_to_target = computePathTarget(path_in_robot);

    }
    else
    {
        if(const auto vector_to_target_ = get_translation_vector_from_target_node(intention_edge); not vector_to_target_.has_value())
        {
            qWarning() << __FUNCTION__ << "Error getting translation from the robot to the target node. Returning.";
            return;
        }
        else vector_to_target = vector_to_target_.value();
    }

    draw_vector_to_target(vector_to_target, &widget_2d->scene );
    auto [norm_at_target, angle_at_target] = robot_at_target(vector_to_target, intention_edge);

    /// If robot is at target, stop
    if(norm_at_target)
    {
//        if(not current_path_room.empty())
//        {
//            // Print path size
//            std::cout << "Path size: " << current_path_room.size() << std::endl;
//            qInfo() << "Robot at current path target.";
//        }
//        else
//        {
            qInfo() << "Robot at target. Stopping.";

            auto is_finite_intention = G->get_attrib_by_name<finite_att>(intention_edge);
            if(not is_finite_intention.has_value())
            {
                std::cout << "There is not finite attribute. Considering not finite intention." << std::endl;
                set_intention_edge_state(intention_edge, "completed");
                stop_robot();
            }
            else
            {
                auto finite = is_finite_intention.value();
                if(finite)
                {
                    std::cout << "Intention is finite. Deleting intention edge." << std::endl;
                    set_intention_edge_state(intention_edge, "completed");
                    stop_robot();
                }
                else
                {
                    if(angle_at_target)
                    {
                        stop_robot();
                    }
                    std::cout << "Intention is not finite. Keeping going." << std::endl;
                }
            }
            return;
//        }
    }

    set_intention_edge_state(intention_edge, "in_progress");

    auto [adv, side, rot] = compute_line_of_sight_target_velocities(vector_to_target);
    
    // if(norm_at_target) adv = 0;
    // if(angle_at_target) rot = 0;
    qInfo() << "Robot speed" << adv << side << -rot;
    move_robot(adv, side, -rot);

    /// Check if the target position is in line of sight
    // if( line_of_sight(vector_to_target, lidar_points, &widget_2d->scene) )
    // {
    //     auto [adv, side, rot] = compute_line_of_sight_target_velocities(vector_to_target);
    //     move_robot(adv, side, rot);
    //     return;
    // }

    /// not in line of sight
    // if there is a node of type ROOM and it has a self edge of type "current" in it
//    bool found = false;  // flag to check if the current room has been found
//    auto current_edges = G->get_edges_by_type("current");
//    for(const auto &e : current_edges)
//        if(G->get_node(e.from())->type() == "room" and G->get_node(e.to())->type() == "room")
//        {
//            // if there is no current_path or the target_position in the room has changed
//            if (current_path.empty() or vector_to_target != vector_to_target_ant)
//                current_path = room_gridder->get_path_to_target(vector_to_target);
//            found = true;
//            break;
//        }
//    if(not found) // ask local_gridder for a path to the target position
//        current_path = local_gridder->get_path_to_target(vector_to_target);
//
//    // update path and send it to MPC and get velocity commands for the bumper
//    auto [new_path, adv, side, rot] = update_and_smooth_path(current_path);
//    send_velocity_commands(adv, side, rot);
//    current_path = new_path;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_lidar()
{
    auto wait_period = std::chrono::milliseconds (this->getPeriod("Compute"));
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_LOW,
                                                                   params.MAX_LIDAR_LOW_RANGE,
                                                                   params.LIDAR_LOW_DECIMATION_FACTOR);
            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysteresis of 2ms
            if (wait_period > std::chrono::milliseconds(static_cast<long>(data.period) + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds(static_cast<long>(data.period) - 2)) wait_period++;
            std::vector<Eigen::Vector3f> eig_data(data.points.size());
            for (const auto &[i, p]: data.points | iter::enumerate)
                if (p.distance2d > 500) eig_data[i] = {p.x, p.y, p.z};
            buffer_lidar_data.put(std::move(eig_data));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D " << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
} // Thread to read the lidar
std::optional<DSR::Edge> SpecificWorker::there_is_intention_edge_marked_as_active()
{
    auto edges = G->get_edges_by_type("has_intention");
    for(const auto &edge: edges)
    {
        if(G->get_attrib_by_name<active_att>(edge).has_value() && G->get_attrib_by_name<active_att>(edge).value())
            return edge;
    }
    return {};
}
bool SpecificWorker::target_node_is_measurement(const DSR::Edge &edge)
{
    DSR::Node robot_node;
    if(auto robot_node_ = G->get_node("Shadow"); not robot_node_.has_value())
    { std::cout << __FUNCTION__ << "Robot node not found. Returning." << std::endl;}
    else robot_node = robot_node_.value();
    if(G->get_node(edge.from()).value().id() == robot_node.id())
       return true; // measurement
    else // nominal
        return false;
}
std::optional<Eigen::Vector3d> SpecificWorker::get_translation_vector_from_target_node(const DSR::Edge &intention_edge)
{
    // get offset for target node
    Eigen::Vector3d offset = Eigen::Vector3d::Zero();
    if( auto offset_ = G->get_attrib_by_name<offset_xyz_att>(intention_edge); offset_.has_value())
        offset = { offset_.value().get()[0], offset_.value().get()[1], offset_.value().get()[2] };
    
    // get the pose of the target_node + offset in the intention edge
    if(auto translation_ = inner_api->transform(params.robot_name, offset,G->get_name_from_id(intention_edge.to()).value(), 9999999999999); not translation_.has_value())
    { std::cout << __FUNCTION__ << "Error getting translation from the robot to the target node plus offset. Returning." << std::endl; return {};}
    else
        return translation_.value();

    /// check if the intention edge starts from the robot or else is a nominal node
    if( target_node_is_measurement(intention_edge))
    {
        // get the pose of the target_node + offset in the intention edge
        if(auto translation_ = inner_api->transform(params.robot_name, offset,G->get_name_from_id(intention_edge.to()).value()); not translation_.has_value())
        { std::cout << __FUNCTION__ << "Error getting translation from the robot to the target node plus offset. Returning." << std::endl; return {};}
        else
            return translation_.value();
    }
    else // nominal
    {
        if(auto translation_ = inner_api->transform(params.robot_name, offset,G->get_name_from_id(intention_edge.to()).value()); not translation_.has_value())
        { std::cout << __FUNCTION__ << "Error getting translation from the robot to the target node. Returning." << std::endl;}
        else return translation_.value();
    }
    return {};
}

std::optional<Eigen::Vector3d> SpecificWorker::get_translation_vector_to_point(const Eigen::Vector2f &current_target)
{
    Eigen::Vector3d current_target_3d;
    current_target_3d << current_target.x(), current_target.y(), 0;

    // get the pose of the target_node + offset in the intention edge
    if(auto translation_ = inner_api->transform(params.robot_name, current_target_3d, params.current_room, 9999999999999); not translation_.has_value())
    { std::cout << __FUNCTION__ << "Error getting translation from the robot to the target node plus offset. Returning." << std::endl; return {};}
    else
        return translation_.value();
    return {};
}

/////////////////// Draw  /////////////////////////////////////////////////////////////
void SpecificWorker::draw_lidar_in_robot_frame(const std::vector<Eigen::Vector3f> &data, QGraphicsScene *scene, QColor color, int step)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &i: iter::range(0, (int)data.size(), step))
    {
        const auto item = scene->addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20), QBrush(QColor("green")));
        auto p = data[i];
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
}
void SpecificWorker::draw_vector_to_target(const Eigen::Vector3d &target, QGraphicsScene *pScene)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ pScene->removeItem(i); delete i;}
    items.clear();

    items.push_back(pScene->addLine(0, 0, target.x(), target.y(), QPen(QColor("red"), 10)));
    auto item = pScene->addEllipse(-50, -50, 100, 100, QPen(QColor("red"), 20), QBrush(QColor("red")));
    item->setPos(target.x(), target.y());
    items.push_back(item);
}

/////////////////////////////////////////////////////////////////////////////////////
std::pair<bool, bool> SpecificWorker::robot_at_target(const Eigen::Vector3d &target, const DSR::Edge &edge)
{
    // get tolerance from intention edge
    Eigen::Vector<float, 6> tolerance = Eigen::Vector<float, 6>::Zero().array() - 1.f; // -1 means no tolerance is applied
    if(auto threshold_ = G->get_attrib_by_name<tolerance_att>(edge); threshold_.has_value())
        tolerance = { threshold_.value().get()[0], threshold_.value().get()[1], threshold_.value().get()[2],
                      threshold_.value().get()[3], threshold_.value().get()[4], threshold_.value().get()[5] };
    auto angle_to_target = atan2(target[0], target[1]);
    /// Print target and tolerance
    std::cout << "Target: " << target.transpose() << std::endl;
    std::cout << "Tolerance: " << tolerance.transpose() << std::endl;
    std::cout << "Distance to target: " << target.head(2).norm() << std::endl;
    std::cout << "Tolerance norm: " << tolerance.head(2).norm() << std::endl;
    std::cout << "Angle tolerance: " << tolerance[5] << std::endl;
    std::cout << "Angle to target: " << angle_to_target << std::endl;
    
    return std::make_pair(target.head(2).norm() <= tolerance.head(2).norm(), angle_to_target <= tolerance[5]);
}
void SpecificWorker::stop_robot()
{
    try
    {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        RoboCompGridPlanner::TPlan plan = {.path={}, .controls={}, .timestamp=now, .subtarget={} , .valid=false};
        gridplanner_proxy->setPlan(plan);
    }
    catch (const Ice::Exception &e) { std::cout << "Error sending target to bumper" << e << std::endl; }
}
std::tuple<float, float, float> SpecificWorker::compute_line_of_sight_target_velocities(const Eigen::Vector3d &vector_to_target)
{
    // final point control. adv velocity is proportional to the distance to the target, clamped with params.MAX_ADV_VELOCITY
    float adv = std::clamp( (float)vector_to_target.y(), 0.f, params.MAX_ADVANCE_VELOCITY);
    float side = std::clamp( (float)vector_to_target.x(), -params.MAX_SIDE_VELOCITY, params.MAX_SIDE_VELOCITY);
    float rot = std::clamp( (float)atan2(vector_to_target.x(), vector_to_target.y()), -params.MAX_ROTATION_VELOCITY, params.MAX_ROTATION_VELOCITY);

    // Apply a proportional controller to the adv velocity controlled by the rotation of the robot
    adv = adv * pow((1 - std::abs(rot) / params.MAX_ROTATION_VELOCITY), 2);

    return {adv, side, rot};
}
void SpecificWorker::move_robot(float adv, float side, float rot)
{
    try
    {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        RoboCompGridPlanner::TPlan plan = {.path={},
                                           .controls={{RoboCompGridPlanner::TControl{adv, side, -rot}}},
                                           .timestamp=now,
                                           .subtarget={} ,
                                           .valid=true};
        gridplanner_proxy->setPlan(plan);
    }
    catch (const Ice::Exception &e) { std::cout << "Error sending target to bumper" << e << std::endl; }
}
bool SpecificWorker::line_of_sight(const Eigen::Vector3d &target, const std::vector<Eigen::Vector3f >& ldata, QGraphicsScene *pScene)
{
    /// Static variables for drawing
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ pScene->removeItem(i); delete i;}
    items.clear();

    // check if there is a line of sight from the robot to the target using only local info: lidar3D
    // create a 2D rectangular shape linking the robot and the target with the width of the robot plus a safe zone and check that no lidar points fall inside
    // filter the 3D lidar points to remove those with z component close to zero
    auto points = iter::filter([](const auto &p){ return p.z() > 0.1; }, ldata);

    // given a 2D point t and the robot width, compute two vectors l and r placed at 90 degrees from the target and with length equal to the robot width
    Eigen::Vector2d target_2d = target.head(2).normalized();
    Eigen::Vector2d l = Eigen::Vector2d(target_2d.y(), -target_2d.x()) * params.ROBOT_WIDTH;
    Eigen::Vector2d r = Eigen::Vector2d (-target_2d.y(), target_2d.x()) * params.ROBOT_WIDTH;

    // create a QPolygon rectangular shape linking the robot and the target with the width of the robot plus a safe zone
    QPolygonF robot_safe_band;
    auto sup_iz = l + target.head(2);
    auto sup_de = r + target.head(2);
    robot_safe_band << QPointF(l.x(), l.y()) <<
                       QPointF(sup_iz.x(), sup_iz.y()) <<
                       QPointF(sup_de.x(), sup_de.y()) <<
                       QPointF(r.x(), r.y());

    // draw safe band
    items.push_back(pScene->addPolygon(robot_safe_band, QPen(QColor("blue"), 10)));

    // check if there are points inside the rectangular shape
    // for(const auto &p: points)
    //     if(robot_safe_band.containsPoint(QPointF(p.x(), p.y()), Qt::FillRule::OddEvenFill))
    //     {
    //         qInfo() << __FUNCTION__ << "Line of sight blocked by point " << p.transpose().x() << " " << p.transpose().y();
    //         return false;
    //     }


    return true;
}
RoboCompGridPlanner::Points SpecificWorker::compute_line_of_sight_target(const Eigen::Vector2d &target)
{
    RoboCompGridPlanner::Points path;

    if(target.norm() < params.MIN_DISTANCE_TO_TARGET or target.norm() < params.ROBOT_WIDTH)
    {
        qWarning() << __FUNCTION__ << "Target too close. Cancelling target";
        return path;
    }

    // fill path with equally spaced points from the robot to the target at a distance of consts.ROBOT_LENGTH
    int npoints = ceil(target.norm() / params.ROBOT_SEMI_LENGTH);
    Eigen::Vector2d dir = target.normalized();  // direction vector from robot to target
    for (const auto &i: iter::range(npoints))
    {
        Eigen::Vector2f p = (dir * (params.ROBOT_SEMI_LENGTH * (float)i)).cast<float>();
        path.emplace_back(RoboCompGridPlanner::TPoint(p.x(), p.y()));
    }
    return path;
}
void SpecificWorker::set_intention_edge_state(DSR::Edge &edge, const std::string &string)
{
    /// Set edge attribute state to string
    qInfo() << __FUNCTION__ << "Setting intention edge state to " << string.c_str();
    G->add_or_modify_attrib_local<state_att>(edge, string);
    if(G->insert_or_assign_edge(edge))
        std::cout << __FUNCTION__ << "Intention edge state set to " << string << std::endl;
    else
        std::cout << __FUNCTION__ << "Error setting intention edge state to " << string << std::endl;
}

void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    if(type == "TARGET")
    {
        auto path_node_ = G->get_node("current_path");
        if(path_node_.has_value())
        {
            auto node = path_node_.value();
            auto x_values = G->get_attrib_by_name<path_x_values_att>(node);
            auto y_values = G->get_attrib_by_name<path_y_values_att>(node);
            if(x_values.has_value() and y_values.has_value())
            {
                auto x = x_values.value().get();
                auto y = y_values.value().get();
                std::vector<Eigen::Vector2f> path; path.reserve(x.size());
                for (auto &&[x, y] : iter::zip(x, y))
                    path.push_back(Eigen::Vector2f(x, y));

                current_path_room = path;
                std::cout << __FUNCTION__ << " New path received with " << path.size() << " points." << std::endl;
            }
        }
    }
    if(type == "current")
    {
        auto room_node_ = G->get_node(to);
        if(not room_node_.has_value())
        { qWarning() << __FUNCTION__ << " No room level in graph"; return; }
        auto room_node = room_node_.value();
        params.current_room = room_node.name();
        qInfo() << __FUNCTION__ << " Current room set to " << params.current_room.c_str();
    }
}

void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{
    if (edge_tag == "TARGET")
        current_path_room.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// computePathTarget:
// - current_path: vector de puntos en el sistema del robot (Eigen::Vector2f), modificado in-place
// - D: distancia mínima para eliminar puntos cercanos (mm)
// - stop_distance: si la distancia al final < stop_distance, path y objetivo se borran (mm)
// - lookahead: distancia a buscar a lo largo del path para colocar el objetivo (mm)
// - min_length: si el path total es menor que min_length, se fija objetivo en el punto final (mm)
// Retorna Eigen::Vector3d {x, y, yaw} en mm/radianes. Si no hay objetivo devuelve {NaN,NaN,NaN}.
Eigen::Vector3d SpecificWorker::computePathTarget(std::vector<Eigen::Vector3d>& current_path,
                                  float D,
                                  float stop_distance,
                                  float lookahead,
                                  float min_length)
{
    using std::sqrt;
    using std::atan2;

    const double NAN_D = std::numeric_limits<double>::quiet_NaN();

    // 1) Si path vacío -> sin objetivo
    if (current_path.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    // 2) Eliminar puntos detrás del robot (x < 0) o demasiado cercanos (norma < D)
    std::vector<Eigen::Vector3d> filtered;
    filtered.reserve(current_path.size());

//    Find closest point to robot in path
    auto closest_point = std::min_element(current_path.begin(), current_path.end(),
                               [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                                   return a.norm() < b.norm();
                               });
    if (closest_point != current_path.end()) {
        //Remove previous points
        current_path.erase(current_path.begin(), closest_point);
    }


//    for (const auto &p : current_path) {
//        float dist = p.norm(); // distancia desde el robot (0,0)
//        // Detrás del robot -> proyección en eje X negativa en sistema del robot
//        bool is_behind = (p.x() < 0.0f);
//        if (dist >= D) {
//            filtered.push_back(p);
//        }
//        else
//        {
//            // std::cout << "Removing point " << p.transpose() << " dist=" << dist << (is_behind ? " behind" : " too close") << std::endl;
//            current_path_room.erase(current_path_room.begin());
//        }
//    }



//
//    current_path.swap(filtered);

    if (current_path.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    // 3) Calcular longitud total del path y distancia al final
    double total_len = 0.0;
    for (size_t i = 1; i < current_path.size(); ++i) {
        Eigen::Vector3d d = current_path[i] - current_path[i-1];
        total_len += d.norm();
    }
    // distancia desde el robot (origen) al último punto
    Eigen::Vector3d last_pt = current_path.back();
    double dist_to_last = last_pt.norm();

    // Si estamos lo suficientemente cerca del final -> borrar path y devolver sin objetivo
    if (dist_to_last < stop_distance) {
        current_path.clear();
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    // Si la longitud total es menor que min_length -> objetivo = punto final
    if (total_len < min_length) {
        // determinar yaw mirando desde el penúltimo punto al último (si existe)
        double yaw = 0.0;
        if (current_path.size() >= 2) {
            Eigen::Vector3d pen = current_path[current_path.size() - 2];
            Eigen::Vector3d diff = last_pt - pen;
            if (diff.squaredNorm() > 1e-6f) {
                yaw = atan2((double)diff.y(), (double)diff.x());
            }
        } else {
            // solo un punto: orientar hacia ese punto desde el origen
            yaw = atan2((double)last_pt.y(), (double)last_pt.x());
        }
        return Eigen::Vector3d((double)last_pt.x(), (double)last_pt.y(), yaw);
    }

    // 4) Estrategia lookahead (pure pursuit-like):
    // buscamos el primer punto a una distancia acumulada >= lookahead desde el robot (inicio del path).
    // Path está en coordenadas del robot, empezando por el primer punto (el más cercano hacia adelante).
    double accum = 0.0;
    Eigen::Vector2d target_pt( NAN_D, NAN_D );
    double target_yaw = 0.0;
    bool found = false;

    // Si el primer punto no está en x>=0 puede darse que tenga un gap; asumimos current_path[0] es el más cercano adelante.
    // Recorremos segmentos y acumulamos longitudes.
    Eigen::Vector3d prev;
    // Si el primer punto está muy lejos en longitud acumulada, asumimos desde el origen al primer punto parte de la longitud:
    // para simplicidad consideramos el path como secuencia de puntos (no añadimos origen como punto explícito).
    prev = current_path[0];

    // Si lookahead <= distancia al primer punto, interpolamos desde origen (0,0) al primer punto
    double dist_origin_to_first = prev.norm();
    if (lookahead <= dist_origin_to_first) {
        // Interpolar entre origen (0,0) y current_path[0]
        double t = lookahead / dist_origin_to_first;
        if (dist_origin_to_first < 1e-6) t = 1.0;
        Eigen::Vector2d p0(0.0, 0.0);
        Eigen::Vector2d p1(prev.x(), prev.y());
        target_pt = p0 + t * (p1 - p0);
        target_yaw = atan2((double)prev.y(), (double)prev.x()); // dirección hacia first
        found = true;
    } else {
        // Caso general: buscamos a lo largo de segmentos entre puntos
        accum = dist_origin_to_first;
        for (size_t i = 1; i < current_path.size() && !found; ++i) {
            Eigen::Vector3d cur = current_path[i];
            Eigen::Vector3d seg = cur - prev;
            double seg_len = seg.norm();
            if (seg_len <= 1e-6) {
                prev = cur;
                continue;
            }
            if (accum + seg_len >= lookahead) {
                // Interpolar dentro de este segmento
                double need = lookahead - accum;      // cuánto necesitamos dentro de este segmento
                double alpha = need / seg_len;        // 0..1
                Eigen::Vector2d p_prev(prev.x(), prev.y());
                Eigen::Vector2d p_cur(cur.x(), cur.y());
                target_pt = p_prev + alpha * (p_cur - p_prev);
                target_yaw = atan2((double)seg.y(), (double)seg.x());
                found = true;
                break;
            } else {
                accum += seg_len;
                prev = cur;
            }
        }
        // Si no encontramos porque lookahead > acumulado hasta el final, usar punto final
        if (!found) {
            target_pt = Eigen::Vector2d(last_pt.x(), last_pt.y());
            // set yaw from penultimate->last if possible
            if (current_path.size() >= 2) {
                Eigen::Vector3d p2 = current_path[current_path.size()-2];
                Eigen::Vector3d diff = last_pt - p2;
                if (diff.squaredNorm() > 1e-6f) {
                    target_yaw = atan2((double)diff.y(), (double)diff.x());
                } else {
                    target_yaw = atan2((double)last_pt.y(), (double)last_pt.x());
                }
            } else {
                target_yaw = atan2((double)last_pt.y(), (double)last_pt.x());
            }
            found = true;
        }
    }

    // 5) Comprobar condición de parada (distancia al final)
    if (dist_to_last < stop_distance) {
        current_path.clear();
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    // Devolver objetivo
    if (found) {
        return Eigen::Vector3d(target_pt.x(), target_pt.y(), target_yaw);
    } else {
        return Eigen::Vector3d(NAN_D, NAN_D, NAN_D);
    }
}

void SpecificWorker::draw_room(QGraphicsScene *pScene, const std::vector<Eigen::Vector3f> &lidar_data) // points in robot frame
{
    static bool drawn_room = false; // TODO: Probably needs to be reseted when room changes
    static std::vector<QGraphicsItem *> room;
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ pScene->removeItem(i); delete i;}
    items.clear();

    // check for current room
    auto current_edges = G->get_edges_by_type("current");
    if(current_edges.empty())
    {qWarning() << __FUNCTION__ << " No current edges in graph"; return;}

    auto room_node_ = G->get_node(current_edges[0].to());
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room level in graph"; return; }
    auto room_node = room_node_.value();

    if(not drawn_room)
    {
        /// Get room corners
        auto corner_nodes = G->get_nodes_by_type("corner");
        /// Get corners that not contains "measured" in name string
        auto corners = iter::filter([](const auto &c){ return c.name().find("measured") == std::string::npos; }, corner_nodes);
        QPolygonF room_poly;
        for(const auto &c: corners)
        {
            auto corner_to_room_transformation_ = inner_api->transform(room_node.name(), c.name());
            if(not corner_to_room_transformation_.has_value())
            { std::cout << __FUNCTION__ << "Error getting corner to room transformation. Returning." << std::endl; return;}
            auto corner_to_room_transformation = corner_to_room_transformation_.value();
            room_poly << QPointF(corner_to_room_transformation.x(), corner_to_room_transformation.y());
        }
        room.push_back(pScene->addPolygon(room_poly, QPen(QColor("black"), 30)));
        drawn_room = true;
    }

    /// Get robot pose in room
    if(auto rt_room_robot = inner_api->get_transformation_matrix(room_node.name(), params.robot_name); not rt_room_robot.has_value())
    { std::cout << __FUNCTION__ << "Error getting robot pose in room. Returning." << std::endl; return;}
    else
    {
        auto rt_room_robot_ = rt_room_robot.value();
        auto matrix_translation = rt_room_robot_.translation();
        auto matrix_angles = rt_room_robot_.rotation().eulerAngles(0, 1, 2);
        room_widget->viewer->robot_poly()->setPos(matrix_translation.x(), matrix_translation.y());
        room_widget->viewer->robot_poly()->setRotation(qRadiansToDegrees(matrix_angles[2]));

        /// Draw lidar data in room frame
        for(const auto &i: iter::range(0, static_cast<int>(lidar_data.size()), 50))
        {
            // transform lidar data to room frame multiplying rt_room_robot by point with extended projective coordinate
            auto p = rt_room_robot_ * Eigen::Vector4d(lidar_data[i].x(), lidar_data[i].y(), lidar_data[i].z(), 1);
            auto item = pScene->addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20), QBrush(QColor("green")));
            item->setPos(p.x(), p.y());
            items.push_back(item);
        }
    }

    /// Get doors from graph
    /// Iterate over doors
    for(auto door_nodes = G->get_nodes_by_type("door"); const auto &door_node: door_nodes)
    {
        auto door_width_ = G->get_attrib_by_name<width_att>(door_node);
        if(not door_width_.has_value())
        { std::cout << __FUNCTION__ << "Error getting door width. Returning." << std::endl; return;}
        auto door_width = door_width_.value();

        /// Get door corner pose
        auto door_peak_0_ = inner_api->transform(room_node.name(), Eigen::Vector3d{-door_width/2.f, 0.f, 0.f}, door_node.name());
        if(not door_peak_0_.has_value())
        { std::cout << __FUNCTION__ << "Error getting door corner pose. Returning." << std::endl; return;}
        auto door_peak_0 = door_peak_0_.value();

        auto door_peak_1_ = inner_api->transform(room_node.name(), Eigen::Vector3d{door_width/2.f, 0.f, 0.f}, door_node.name());
        if(not door_peak_1_.has_value())
        { std::cout << __FUNCTION__ << "Error getting door corner pose. Returning." << std::endl; return;}
        auto door_peak_1 = door_peak_1_.value();
        /// Check if door name contains "pre"
        if(door_node.name().find("pre") != std::string::npos)
        {
            /// Draw door
            items.push_back(pScene->addLine(door_peak_0.x(), door_peak_0.y(), door_peak_1.x(), door_peak_1.y(), QPen(QColor("blue"), 30)));
            /// Draw door peaks
            items.push_back(pScene->addEllipse(door_peak_0.x()-30, door_peak_0.y()-30, 60, 60, QPen(QColor("blue"), 30), QBrush(QColor("blue"))));
            items.push_back(pScene->addEllipse(door_peak_1.x()-30, door_peak_1.y()-30, 60, 60, QPen(QColor("blue"), 30), QBrush(QColor("blue"))));
        }
        else
        {
            /// Draw door
            items.push_back(pScene->addLine(door_peak_0.x(), door_peak_0.y(), door_peak_1.x(), door_peak_1.y(), QPen(QColor("red"), 30)));
            /// Draw door peaks
            items.push_back(pScene->addEllipse(door_peak_0.x()-30, door_peak_0.y()-30, 60, 60, QPen(QColor("red"), 30), QBrush(QColor("red"))));
            items.push_back(pScene->addEllipse(door_peak_1.x()-30, door_peak_1.y()-30, 60, 60, QPen(QColor("red"), 30), QBrush(QColor("red"))));
        }
    }
}

/////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //computeCODE
    //
    //if (SUCCESSFUL)
    //  emmit goToRestore()
}

//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //computeCODE
    //Restore emergency component

}

//////////////////////////////////////////////////////////////////////////
//SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
//////////////////////////////////////////////////////////////////////////
void SpecificWorker::FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)
{
    #ifdef HIBERNATION_ENABLED
        hibernation = true;
    #endif

//    auto robot_node = G->get_node(params.robot_name);
//    if(not robot_node.has_value())
//    {  qWarning() << "Robot node" << QString::fromStdString(params.robot_name) << "not found"; return; }
//
//    auto robot_node_value = robot_node.value();
//    G->add_or_modify_attrib_local<robot_current_advance_speed_att>(robot_node_value, (float)  pose.vx);
//    G->add_or_modify_attrib_local<robot_current_side_speed_att>(robot_node_value, (float) pose.vy);
//    G->add_or_modify_attrib_local<robot_current_angular_speed_att>(robot_node_value, (float) pose.vrz);
//    G->add_or_modify_attrib_local<timestamp_alivetime_att>(robot_node_value, (uint64_t) pose.timestamp);
//    G->update_node(robot_node_value);

    // if (this->loadRT)
    // {
    //     auto robot_edge = G->get_edge("root", params.robot_name, "RT");
    //     if(not robot_edge.has_value())
    //     {  qWarning() << "Robot edge" << QString::fromStdString(params.robot_name) << "not found"; return; }
    //     auto robot_edge_value = robot_edge.value();
    //     G->add_or_modify_attrib_local<rt_translation_att>(robot_edge_value, (std::vector<float>)  std::vector<float>{pose.x, pose.y, pose.z});
    //     G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(robot_edge_value, (std::vector<float>)  std::vector<float>{pose.rx, pose.ry, pose.rz});
    //     G->add_or_modify_attrib_local<rt_translation_velocity_att>(robot_edge_value, (std::vector<float>)  std::vector<float>{pose.vx, pose.vy, pose.vz});
    //     G->add_or_modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(robot_edge_value, (std::vector<float>)  std::vector<float>{pose.vrx, pose.vry, pose.vrz});
    //     G->add_or_modify_attrib_local<rt_timestamps_att>(robot_edge_value, (std::vector<uint64_t>) std::vector<uint64_t>{pose.timestamp});
    //     G->insert_or_assign_edge(robot_edge_value);
    // }


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

