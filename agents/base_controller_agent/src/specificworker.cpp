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
/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
	QLoggingCategory::setFilterRules("*.debug=false\n");
}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	//G->write_to_json_file("./"+agent_name+".json");
	auto grid_nodes = G->get_nodes_by_type("grid");
	for (auto grid : grid_nodes)
		G->delete_node(grid);
	G.reset();
}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
	try
	{
		agent_name = params.at("agent_name").value;
		agent_id = stoi(params.at("agent_id").value);
		tree_view = params.at("tree_view").value == "true";
		graph_view = params.at("graph_view").value == "true";
		qscene_2d_view = params.at("2d_view").value == "true";
		osg_3d_view = params.at("3d_view").value == "true";
	}
	catch(const std::exception &e){ std::cout << e.what() << " Error reading params from config file" << std::endl;};
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
		// create graph
		G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
		std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;  

		//dsr update signals
		//connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

		// Graph viewer
		using opts = DSR::DSRViewer::view;
		int current_opts = 0;
		opts main = opts::none;
		if(tree_view)
		    current_opts = current_opts | opts::tree;
		if(graph_view)
        {
		    current_opts = current_opts | opts::graph;
		    main = opts::graph;
		}
		if(qscene_2d_view)
		    current_opts = current_opts | opts::scene;
		if(osg_3d_view)
		    current_opts = current_opts | opts::osg;
		graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
		setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));
        // get pointer to 2D viewer
        widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));
        widget_2d->set_draw_axis(true);

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/

        room_widget = new CustomWidget();
   	    graph_viewer->add_custom_widget_to_dock("room view", room_widget);


        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        rt_api = G->get_rt_api();
        inner_api = G->get_inner_eigen_api();

        timer.start(Period);
	}
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

    /// read LiDAR
    auto ldata = buffer_lidar_data.get_idemp();

    /// draw lidar in robot frame
    draw_lidar_in_robot_frame(ldata, &widget_2d->scene, 50);


    draw_robot_in_room(&room_widget->viewer->scene, ldata);

    // check if there is a room with a current self pointing edge
    //draw_room_frame(ldata, &room_widget->viewer->scene);  // draws lidar and robot in room frame

    /// check if pushButton_stop is checked and stop robot and return if so
    if(room_widget->ui->pushButton_stop->isChecked())
    { qWarning() << __FUNCTION__ << " Stop button checked"; stop_robot(); return; }

    /// check if there is a "has_intention" edge in G. If so get the intention edge
    DSR::Edge intention_edge;
    if( auto intention_edge_ =  there_is_intention_edge_marked_as_active(); not intention_edge_.has_value() )
    {
        std::cout << __FUNCTION__ << " There is no intention edge. Returning." << std::endl;
        stop_robot();
        return;
    }
    else intention_edge = intention_edge_.value();

    // Print from and to nodes of the intention edge
    std::cout << "Intention edge from: " << intention_edge.from() << intention_edge.to() << std::endl;

    auto intention_status_ = G->get_attrib_by_name<state_att>(intention_edge);
    if(not intention_status_.has_value())
    {
        std::cout << __FUNCTION__ << " Error getting intention status. Returning." << std::endl;
        return;
    }
    auto intention_status = intention_status_.value();

    /// Set string to set intention state
    std::string intention_state = "waiting";


    /// Check if there is final orientation and tolerance in the intention edge
//    Eigen::Vector3f orientation = Eigen::Vector3f::Zero();
//    if(auto orientation_ = G->get_attrib_by_name<orientation_att>(intention_edge); orientation_.has_value())
//        orientation = { orientation_.value().get()[0], orientation_.value().get()[1], orientation_.value().get()[2] };
//
    /// Get translation vector from target node with offset applied to it
    Eigen::Vector3d vector_to_target = Eigen::Vector3d::Zero();
    if(auto vector_to_target_ = get_translation_vector_from_target_node(intention_edge); not vector_to_target_.has_value())
    {
        qWarning() << __FUNCTION__ << "Error getting translation from the robot to the target node. Returning.";
        return;
    }
    else vector_to_target = vector_to_target_.value();

    std::cout << __FUNCTION__ << "\n [" << vector_to_target << "]" << std::endl;
    draw_vector_to_target(vector_to_target, &widget_2d->scene );

    /// If robot is at target, stop
    if(robot_at_target(vector_to_target, intention_edge))
    {
        qInfo() << "Robot at target. Stopping.";
        stop_robot();
        set_intention_edge_state(intention_edge, "completed");
        return;
    }

    set_intention_edge_state(intention_edge, "in_progress");

    /// Check if the target position is in line of sight
    if( line_of_sight(vector_to_target, ldata, &widget_2d->scene) )
    {
        auto [adv, side, rot] = compute_line_of_sight_target_velocities(vector_to_target);
        move_robot(adv, side, rot);
        return;
    }





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
    if(auto translation_ = inner_api->transform(params.robot_name, offset,G->get_name_from_id(intention_edge.to()).value()); not translation_.has_value())
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
/////////////////// Draw  /////////////////////////////////////////////////////////////
void SpecificWorker::draw_lidar_in_robot_frame(const std::vector<Eigen::Vector3f> &data, QGraphicsScene *scene, QColor color, int step)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &i: iter::range(0, (int)data.size(), step))
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20), QBrush(QColor("green")));
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
bool SpecificWorker::robot_at_target(const Eigen::Vector3d &target, const DSR::Edge &edge)
{
    // get tolerance from intention edge
    Eigen::Vector<float, 6> tolerance = Eigen::Vector<float, 6>::Zero().array() - 1.f; // -1 means no tolerance is applied
    if(auto threshold_ = G->get_attrib_by_name<tolerance_att>(edge); threshold_.has_value())
        tolerance = { threshold_.value().get()[0], threshold_.value().get()[1], threshold_.value().get()[2],
                      threshold_.value().get()[3], threshold_.value().get()[4], threshold_.value().get()[5] };
    /// Print target and tolerance
    std::cout << "Target: " << target.transpose() << std::endl;
    std::cout << "Tolerance: " << tolerance.transpose() << std::endl;
    std::cout << "Distance to target: " << target.head(2).norm() << std::endl;
    std::cout << "Tolerance norm: " << tolerance.head(2).norm() << std::endl;
    return target.head(2).norm() <= tolerance.head(2).norm();
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

    return {adv, side, rot};
}
void SpecificWorker::move_robot(float adv, float side, float rot)
{
    try
    {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        RoboCompGridPlanner::TPlan plan = {.path={},
                                           .controls={{RoboCompGridPlanner::TControl{adv, side, rot}}},
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
    for(const auto &p: points)
        if(robot_safe_band.containsPoint(QPointF(p.x(), p.y()), Qt::FillRule::OddEvenFill))
        {
            qInfo() << __FUNCTION__ << "Line of sight blocked by point " << p.transpose().x() << " " << p.transpose().y();
            return false;
        }


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
void SpecificWorker::set_intention_edge_state(DSR::Edge &edge, const string &string)
{
    /// Set edge attribute state to string
    qInfo() << __FUNCTION__ << "Setting intention edge state to " << string.c_str();
    G->add_or_modify_attrib_local<state_att>(edge, string);
    if(G->insert_or_assign_edge(edge))
        std::cout << __FUNCTION__ << "Intention edge state set to " << string << std::endl;
    else
        std::cout << __FUNCTION__ << "Error setting intention edge state to " << string << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

void SpecificWorker::draw_robot_in_room(QGraphicsScene *pScene, const vector<Eigen::Vector3f> &lidar_data)
{
    static bool drawn_room = false; // TODO: Probably needs to be reseted when room changes
    static std::vector<QGraphicsItem *> room;
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ pScene->removeItem(i); delete i;}
    items.clear();

// 2. check for current room
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
    }

    /// Get doors from graph
    auto door_nodes = G->get_nodes_by_type("door");
    /// Iterate over doors
    for(const auto &door_node: door_nodes)
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

