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

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/
		//graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);

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
    //draw_lidar(ldata, &widget_2d->scene, 50);

    /// check if there is a "has_intention" edge in G. If so get the intention edge
//    DSR::Edge intention_edge;
//    if( auto intention_edge_ =  there_is_intention_edge_marked_as_valid(); not intention_edge_.has_value() )
//    { std::cout << __FUNCTION__ << "There is no intention edge. Returning." << std::endl; return;}

    auto intention_edge_ = there_is_intention_edge_marked_as_valid();
    if(not intention_edge_.has_value())
    { std::cout << __FUNCTION__ << "There is no intention edge. Returning." << std::endl; return;}
    auto intention_edge = intention_edge_.value();

    /// Check if there is final orientation and tolerance in the intention edge
//    Eigen::Vector3f orientation = Eigen::Vector3f::Zero();
//    if(auto orientation_ = G->get_attrib_by_name<orientation_att>(intention_edge); orientation_.has_value())
//        orientation = { orientation_.value().get()[0], orientation_.value().get()[1], orientation_.value().get()[2] };
//

    /// Get translation vector from target node with offset applied to it
    Eigen::Vector3d vector_to_target = Eigen::Vector3d::Zero();
    if(auto vector_to_target_ = get_translation_vector_from_target_node(intention_edge); not vector_to_target_.has_value())
    { qWarning() << __FUNCTION__ << "Error getting translation from the robot to the target node. Returning."; return;}
    else vector_to_target = vector_to_target_.value();

    /// If robot is at target, stop
    if(robot_at_target(vector_to_target, intention_edge))
    {
        qInfo() << "Robot at target. Stopping.";
        stop_robot();
        return;
    }

    /// Check if the target position is in line of sight
    if( line_of_sight(vector_to_target) )
    {
        // compute rotation, advance and side speed. Send to bumper
        auto [adv, side, rot] = compute_velocity_commands(vector_to_target);
        send_velocity_commands(adv, side, rot);
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
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
} // Thread to read the lidar
std::optional<DSR::Edge> SpecificWorker::there_is_intention_edge_marked_as_valid()
{
    auto edges = G->get_edges_by_type("has_intention");
    for(const auto &edge: edges)
    {
        if(G->get_attrib_by_name<valid_att>(edge).has_value() && G->get_attrib_by_name<valid_att>(edge).value())
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

    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
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
void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &data, QGraphicsScene *scene, QColor color, int step)
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
    send_velocity_commands(0.f, 0.f, 0.f);
    return;
}
bool SpecificWorker::line_of_sight(const Eigen::Vector3d &matrix)
{
    return false;
}
std::tuple<float, float, float> SpecificWorker::compute_velocity_commands(const Eigen::Vector3d &matrix)
{

}
void SpecificWorker::send_velocity_commands(float advx, float advz, float rot)
{

}
/////////////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
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

