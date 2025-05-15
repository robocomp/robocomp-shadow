/*
 *    Copyright (C) 2025 by YOUR NAME HERE
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
 *    adouble with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"

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
		
		//dsr update signals
		//connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
//		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
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

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
    exit(0);
	//G->write_to_json_file("./"+agent_name+".json");
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

    room_widget->viewer->robot_poly()->hide();

    std::cout << "initialize worker" << std::endl;
    //initializeCODE
    rt_api = G->get_rt_api();
    inner_api = G->get_inner_eigen_api();

    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 

//    room_widget = new CustomWidget();
//    graph_viewer->add_custom_widget_to_dock("room view", room_widget);

    // Get robot id
    if(auto robot_node = G->get_node("Shadow"); robot_node.has_value())
        robot_id = robot_node.value().id();
    else
    {
        std::cerr << "Robot node not found" << std::endl;
        std::terminate();
    }

//    // Initialize time points with the current system time
//    rt_set_last_time = std::chrono::system_clock::now();
//    last_update_with_corners = std::chrono::system_clock::now();
//    elapsed = std::chrono::system_clock::now();

    // Insert prior pose to GTSAM graph (TEST)
//    gtsam_graph.insert_prior_pose(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));

//    robot_pose = Eigen::Affine3d::Identity();

//    initialize_graph();
}

void SpecificWorker::compute()
{
    // Statemaching
    // Pop odometry value
    auto odom  = odom_buffer.try_get();
    if(!odom.has_value()) return;
    const auto [timestamp, translation, rotation] = odom.value();
    // State 1: Graph not initialized
//    std::cout << "timestamp: " << timestamp << std::endl;
    switch (init_graph)
    {
        case 0:
        {
            qInfo() << "Initializing graph...";
            if(initialize_graph())
            {
                std::cout << "Graph initialized" << std::endl;
                gtsam_graph.set_start_time(timestamp);
                init_graph = 1;
            }
            else{std::cerr << "Graph not initialized" << std::endl;};
            break;
        }
        case 1:
        {

            if(timestamp != last_odometry_timestamp)
            {
                // print odometry value
//                std::cout << "Odometry: " << timestamp << " " << translation.transpose() << " " << rotation.transpose() << std::endl;
//                std::cout << "Timestamp diff: " << timestamp - last_odometry_timestamp << std::endl;
                auto [last_pose_timestamp, robot_translation_from_last_timestamp] = integrate_odometry(odom.value());
                gtsam_graph.insert_odometry_pose(last_pose_timestamp, gtsam::Pose3(robot_translation_from_last_timestamp.matrix()));

                // Get measured corners and nodes graph timestamps
                auto actual_measured_corners = get_measured_corners(last_pose_timestamp);
                draw_measured_corners(&room_widget->viewer->scene, actual_measured_corners);
                gtsam_graph.add_landmark_measurement(actual_measured_corners);

                // Print optimized pose
                auto act_robot_pose = gtsam_graph.get_robot_pose();
                const gtsam::Point3 translation = act_robot_pose.translation();
                const double x = translation.x();
                const double y = translation.y();
                const double z = translation.z();
                // Extract rotation as Euler angles (Z-Y-X convention, radians)
                const gtsam::Rot3 rotation = act_robot_pose.rotation();
                const double roll = rotation.roll();   // X-axis rotation
                const double pitch = rotation.pitch(); // Y-axis rotation
                const double yaw = rotation.yaw();     // Z-axis rotation
                // Convert to degrees for more intuitive display
                // Print formatted output

                std::cout << "  X: " << std::setw(8) << x
                          << "  Y: " << std::setw(8) << y
                          << "  alpha: " << std::setw(8) << yaw << std::endl;
                update_robot_dsr_pose(x, y, yaw, timestamp);
                draw_robot_in_room(&room_widget->viewer->scene);
                gtsam_graph.draw_graph_nodes(&room_widget->viewer->scene);
            }
            break;
        }
    }
    last_odometry_timestamp = timestamp;

    // State 2: Graph initialized. Update
}

// Thread-safe copy of circular buffer
std::vector<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>>
SpecificWorker::copy_odometry_buffer(
        boost::circular_buffer<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>>& buffer,
        std::mutex& buffer_mutex,
        double min_timestamp)  // Default to all values if timestamp not specified
{
    std::lock_guard<std::mutex> lock(buffer_mutex);

    // Find first element meeting the timestamp requirement
    auto it = std::find_if(buffer.begin(), buffer.end(),
                           [min_timestamp](const auto& sample) {
                               return std::get<0>(sample) >= min_timestamp;
                           });

    // Return all elements from found position to end
    return std::vector<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>>(it, buffer.end());
}

std::tuple<double, Eigen::Affine3d> SpecificWorker::integrate_odometry(
        const std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>& odometry_value)
{
    auto [current_time, translation, rotation] = odometry_value;
    Eigen::Affine3d current_pose = Eigen::Affine3d::Identity();
    auto last_timestamp = last_odometry_timestamp;
//    qInfo() << "Start timestamp: " << last_timestamp;


//            qInfo() << "Integrating odometry values...";
//            qInfo() << "Current time: " << current_time << "Prev time" << prev_time;
    // Calculate time difference in seconds
    double delta_t = (current_time - last_timestamp);  // Convert milliseconds to seconds

    // Compute rotation change (using exponential map)
    const double angle = rotation.norm() * delta_t;
//    if (angle > 1e-6) {  // Threshold to avoid numerical instability
//        current_pose.rotate(Eigen::AngleAxisd(angle, rotation.normalized()));
//    }
    current_pose.rotate(Eigen::AngleAxisd(angle, rotation.normalized()));
    // Print translation and delta_t for debugging
//    std::cout << "Translation: " << translation.transpose() << std::endl;
//    std::cout << "Delta t: " << delta_t << std::endl;
    // Compute and apply translation
    current_pose.translate(current_pose.rotation() * (translation * delta_t));

    return std::make_tuple(current_time, current_pose);
}

std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> SpecificWorker::get_measured_corners(double timestamp)
{
    std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> returned_corners;

    // Get nominal corners from graph ("corner" type nodes in which name doesnt appear "measured")
    auto corners = G->get_nodes_by_type("corner");
    std::vector<DSR::Node> corners_vector;
    for (const auto& corner : corners)
        if (corner.name().find("measured") != std::string::npos)
            corners_vector.push_back(corner);

    // Order corners by corner_id attribute
    std::sort(corners_vector.begin(), corners_vector.end(),
              [this](const DSR::Node& a, const DSR::Node& b) {
                  auto a_id = G->get_attrib_by_name<corner_id_att>(a);
                  auto b_id = G->get_attrib_by_name<corner_id_att>(b);

                  // Handle cases where attributes might be missing
                  if (!a_id.has_value() && !b_id.has_value()) return false; // equal
                  if (!a_id.has_value()) return true;  // a comes first if missing ID
                  if (!b_id.has_value()) return false; // b comes first if missing ID

                  return a_id.value() < b_id.value();
              });

    // Iterate over corners and insert them into GTSAM graph
    for (const auto& corner : corners_vector)
    {
        auto corner_is_valid = G->get_attrib_by_name<valid_att>(corner); if (!corner_is_valid) return returned_corners;
        auto corner_id = G->get_attrib_by_name<corner_id_att>(corner); if (!corner_id) return returned_corners;
        if(auto parent_node = G->get_parent_node(corner); parent_node.has_value())
        {
            auto parent_node_value = parent_node.value();
            if(auto rt_corner_wall = rt_api->get_edge_RT(parent_node_value, corner.id()); rt_corner_wall.has_value())
            {
                auto corner_edge = rt_corner_wall.value();
                if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(corner_edge); rt_translation.has_value())
                {
                    auto rt_corner_value = rt_translation.value().get();
                    if(auto rt_timestamp = G->get_attrib_by_name<rt_timestamps_att>(corner_edge); rt_timestamp.has_value())
                    {
                        auto rt_timestamp_value = rt_timestamp.value().get();
                        auto corner_timestamp = rt_timestamp_value[0] / 1000.0;
//                        std::cout << "Corner timestamp: " << corner_timestamp / 1000.0 << std::endl;
                        Eigen::Vector3f corner_pose(rt_corner_value[0] / 1000, rt_corner_value[1] / 1000, 0.f);
                        auto corner_pose_double = corner_pose.cast<double>();
                        returned_corners.push_back(std::make_tuple(corner_id.value(), corner_timestamp, corner_pose_double, corner_is_valid.value()));
                    }
                }
            }
        }
    }
    return returned_corners;
}

void SpecificWorker::update_robot_dsr_pose(double x, double y, double ang, double timestamp)
{
    // Get room node
    auto room_node_ = G->get_node("room");
    if (!room_node_) return;
    auto room_node = room_node_.value();

    // Get robot node
    auto robot_node_ = G->get_node("Shadow");
    if (!robot_node_) return;
    auto robot_node = robot_node_.value();

//    // Get RT edge from robot to room
//    auto rt_edge_ = rt_api->get_edge_RT(room_node, robot_node.id());
//    qInfo() << __FUNCTION__ << " Requested RT edge from robot to room: " ;
//    if (!rt_edge_)
//    {
//        qInfo() << __FUNCTION__ << " No RT edge between robot and room";
//        return;
//    }
//    // Get rt_timestamps attribute
//    auto rt_pose_ = G->get_attrib_by_name<rt_translation_att>(rt_edge_.value());
//    if (!rt_pose_)
//    {
//        qInfo() << __FUNCTION__ << " No RT pose";
//        return;
//    }
//    auto rt_pose_value = rt_pose_.value().get();
//    // Get rt_timestamps attribute
//    auto rt_rotation_ = G->get_attrib_by_name<rt_rotation_euler_xyz_att >(rt_edge_.value());
//    if (!rt_rotation_)
//    {
//        qInfo() << __FUNCTION__ << " No RT pose";
//        return;
//    }
//    auto rt_rotation_value = rt_rotation_.value().get();
//    // Get rt_timestamps attribute
//    auto rt_timestamp_ = G->get_attrib_by_name<rt_timestamps_att>(rt_edge_.value());
//    if (!rt_timestamp_)
//    {
//        qInfo() << __FUNCTION__ << " No RT timestamp";
//        return;
//    }
//    auto rt_timestamp_value = rt_timestamp_.value().get();
//
//    // Get RT pose from robot to room
//    auto rt_edge = DSR::Edge::create<RT_edge_type>(room_node.id(), robot_node.id());
//    std::vector<uint64_t> timestamps {(uint64_t )(timestamp * 1000.0)};
//    std::vector<float> rotation = {0, 0, (float)ang};
//    std::vector<float> traslation = {(float)(x * 1000.f), (float)(y * 1000.f), 0};
//
//    // COncatenate vectors with required data from rt_edge_
//    rt_pose_value.insert(rt_pose_value.end(), traslation.begin(), traslation.end());
//    rt_rotation_value.insert(rt_rotation_value.end(), rotation.begin(), rotation.end());
//    rt_timestamp_value.insert(rt_timestamp_value.end(), timestamps.begin(), timestamps.end());
//
//    G->add_or_modify_attrib_local<rt_timestamps_att >(rt_edge, rt_timestamp_value );
//    G->add_or_modify_attrib_local< rt_rotation_euler_xyz_att>(rt_edge, rt_rotation_value);
//    G->add_or_modify_attrib_local<rt_translation_att>(rt_edge, rt_pose_value);
//    G->insert_or_assign_edge(rt_edge);

    qInfo() << __FUNCTION__ << " Inserting RT edge from robot to room";
    rt_api->insert_or_assign_edge_RT(room_node, robot_node.id(), { (float)(x * 1000.f), (float)(y * 1000.f), 0 }, {0, 0, (float)ang}, (uint64_t) (timestamp * 1000.0));
    qInfo() << __FUNCTION__ << " RT edge inserted from robot to room";
    // Get RT edge from robot to room
    auto rt_edge_req = rt_api->get_edge_RT(room_node, robot_node.id());
    qInfo() << __FUNCTION__ << " Requested RT edge from robot to room: " ;
    if (!rt_edge_req)
    {
        qInfo() << __FUNCTION__ << " No RT edge between robot and room";
        return;
    }
    // Get rt_timestamps attribute
    auto rt_pose = G->get_attrib_by_name<rt_translation_att>(rt_edge_req.value());
    if (!rt_pose)
    {
        qInfo() << __FUNCTION__ << " No RT pose";
        return;
    }
    // Get rt_timestamps attribute
    auto rt_timestamp = G->get_attrib_by_name<rt_timestamps_att>(rt_edge_req.value());
    if (!rt_timestamp)
    {
        qInfo() << __FUNCTION__ << " No RT timestamp";
        return;
    }
    auto rt_timestamp_value_ = rt_timestamp.value().get();
    // Print rt timestamp vector
    for(const auto& value : rt_timestamp_value_)
    {
        std::cout << "RT timestamp: " << value << std::endl;
    }
}
void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}

//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}

void SpecificWorker::draw_robot_in_room(QGraphicsScene *pScene)
{
    /// Get robot pose in room
    if(auto rt_room_robot = inner_api->get_transformation_matrix("room", "Shadow"); not rt_room_robot.has_value())
    { std::cout << __FUNCTION__ << "Error getting robot pose in room. Returning." << std::endl; return;}
    else
    {
        auto rt_room_robot_ = rt_room_robot.value();
        auto matrix_translation = rt_room_robot_.translation();
        auto matrix_angles = rt_room_robot_.rotation().eulerAngles(0, 1, 2);
        room_widget->viewer->robot_poly()->setPos(matrix_translation.x(), matrix_translation.y());
        room_widget->viewer->robot_poly()->setRotation(qRadiansToDegrees(matrix_angles[2]));
    }
}

void SpecificWorker::draw_nominal_corners(QGraphicsScene *pScene, const std::vector<std::tuple<int, double, Eigen::Vector3d>> &nominal_corners)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ pScene->removeItem(i); delete i;}
    items.clear();
    QPolygonF poly;
    // draw corners
    for(const auto &corner: nominal_corners)
    {
        auto corner_position = std::get<2>(corner);
        // Print corner position
        std::cout << "Corner position: " << corner_position.x() << " " << corner_position.y() << std::endl;
        const auto item = pScene->addEllipse(-200, -200, 400, 400, QPen(QColor("red"), 100), QBrush(QColor("red")));
        poly << QPointF(corner_position.x(), corner_position.y());
        item->setPos(corner_position.x(), corner_position.y());
        items.push_back(item);

    }
    auto poly_item = pScene->addPolygon(poly, QPen(QColor("red"), 100));
    items.push_back(poly_item);
}

void SpecificWorker::draw_measured_corners(QGraphicsScene *pScene, const std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> &measured_corners)
{
//    static std::vector<QGraphicsItem *> items;
//    for(const auto &i: items){ pScene->removeItem(i); delete i;}
//    items.clear();
//
//    // Get robot pose
//    auto act_robot_pose = gtsam_graph.get_robot_pose();
//    const gtsam::Point3 translation = act_robot_pose.translation();
//    const double x = translation.x();
//    const double y = translation.y();
//    // Extract rotation as Euler angles (Z-Y-X convention, radians)
//    const gtsam::Rot3 rotation = act_robot_pose.rotation();
//    const double yaw = rotation.yaw();     // Z-axis rotation
//
//    // draw corners
//    for(const auto &corner: measured_corners)
//    {
//        auto corner_position = std::get<2>(corner);
//        // Apply transformation to measured corners
//        auto trans_corner_position = transform * corner_position.head<2>();
//        // Print transformed corner position
//        std::cout << "Transformed corner position: " << trans_corner_position.x() << " " << trans_corner_position.y() << std::endl;
//        auto corner_is_valid = std::get<3>(corner);
//        if(corner_is_valid)
//        {
//            const auto item = pScene->addEllipse(-100, -100, 200, 200, QPen(QColor("green"), 100), QBrush(QColor("green")));
//            item->setPos(trans_corner_position.x(), trans_corner_position.y());
//            items.push_back(item);
//        }
//    }
}

// ******************************* CREATE INITIALIZE G2O GRAPH FUNCTION BASED ON GTSAM AGENT FROM GERARDO'S PC ********************************
bool SpecificWorker::initialize_graph()
{
    //get shadow robot node from G
    auto robot_node_ = G->get_node("Shadow"); if (!robot_node_) return false;
    auto& robot_node = robot_node_.value();

    auto current_edge = G->get_edges_by_type("current"); if (current_edge.empty()) return false;
    auto current_room_node_ = G->get_node(current_edge[0].to()); if (!current_room_node_) return false;
    auto& current_room_node = current_room_node_.value();

    odometry_node_id = robot_node.id();
    //get attribute room_id from current room node
    auto current_room_id = G->get_attrib_by_name<room_id_att>(current_room_node); if (!current_room_id) return false;

    if(actual_room_id != current_room_node.id())
        last_room_id = actual_room_id;
    actual_room_id = current_room_id.value();
    actual_room_name = current_room_node.name();

//    # Get first robot pose
    auto robot_pose = get_dsr_robot_pose(current_room_node, robot_node.id());
    if (!robot_pose.has_value())
    {
        std::cerr << "Robot pose not found" << std::endl;
        return false;
    }
    auto [act_pose_timestamp, robot_pose_value] = robot_pose.value();
    // Print robot pose
    std::cout << "Robot pose: " << robot_pose_value.translation() << " " << robot_pose_value.rotation().yaw() << " " << act_pose_timestamp << std::endl;

    // Get timestamp from the first odom_value in the buffer
    auto first_odom_value = odometry_queue.front();
    odometry_queue.pop_front();
    // Get the timestamp from the first odometry value
    auto first_odom_timestamp = std::get<0>(first_odom_value);
    gtsam_graph.insert_prior_pose(first_odom_timestamp, robot_pose_value);

    auto nominal_corners_data = get_nominal_corners();
    if(nominal_corners_data.empty())
    {
        std::cerr << "No nominal corners found" << std::endl;
        return false;
    }
    for (const auto& corner_data : nominal_corners_data)
    {
        auto [corner_id, corner_timestamp, corner_pose] = corner_data;
        std::cout << "Corner id: " << corner_id << " Corner timestamp: " << corner_timestamp / 1000.0<< std::endl;
        gtsam_graph.insert_landmark_prior(corner_timestamp, corner_id, Point3(corner_pose.x()/ 1000.0, corner_pose.y()/ 1000.0, 0.0));
    }
    draw_nominal_corners(&room_widget->viewer->scene, nominal_corners_data);
    return true;
}

std::vector<std::tuple<int, double, Eigen::Vector3d>> SpecificWorker::get_nominal_corners()
{
    std::vector<std::tuple<int, double, Eigen::Vector3d>> returned_corners;

    // Get nominal corners from graph ("corner" type nodes in which name doesnt appear "measured")
    auto corners = G->get_nodes_by_type("corner");
    std::vector<DSR::Node> corners_vector;
    for (const auto& corner : corners)
        if (corner.name().find("measured") == std::string::npos)
            corners_vector.push_back(corner);

    // Order corners by corner_id attribute
    std::sort(corners_vector.begin(), corners_vector.end(),
              [this](const DSR::Node& a, const DSR::Node& b) {
                  auto a_id = G->get_attrib_by_name<corner_id_att>(a);
                  auto b_id = G->get_attrib_by_name<corner_id_att>(b);

                  // Handle cases where attributes might be missing
                  if (!a_id.has_value() && !b_id.has_value()) return false; // equal
                  if (!a_id.has_value()) return true;  // a comes first if missing ID
                  if (!b_id.has_value()) return false; // b comes first if missing ID

                  return a_id.value() < b_id.value();
              });

    // Iterate over corners and insert them into GTSAM graph
    for (const auto& corner : corners_vector)
    {
        auto corner_id = G->get_attrib_by_name<corner_id_att>(corner); if (!corner_id) return returned_corners;
        if(auto parent_node = G->get_parent_node(corner); parent_node.has_value())
        {
            auto parent_node_value = parent_node.value();
            if(auto rt_corner_wall = rt_api->get_edge_RT(parent_node_value, corner.id()); rt_corner_wall.has_value())
            {
                auto corner_edge = rt_corner_wall.value();
                if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(corner_edge); rt_translation.has_value())
                {
                    auto rt_corner_value = rt_translation.value().get();
                    if(auto rt_timestamp = G->get_attrib_by_name<rt_timestamps_att>(corner_edge); rt_timestamp.has_value())
                    {
                        auto rt_timestamp_value = rt_timestamp.value().get();
                        auto corner_timestamp = rt_timestamp_value[0] / 1000.0;
                        Eigen::Vector3f corner_pose(rt_corner_value[0], rt_corner_value[1], 0.f);
                        auto corner_pose_double = corner_pose.cast<double>();

                        if (auto corner_room_pose = inner_api->transform(actual_room_name,
                                                                         corner_pose_double,
                                                                         parent_node_value.name()); corner_room_pose.has_value())
                        {
                            returned_corners.push_back(std::make_tuple(corner_id.value(), corner_timestamp, corner_room_pose.value()));
                        }
                    }
                }
            }
        }
    }
    return returned_corners;
}


std::optional<std::pair<double, gtsam::Pose3>> SpecificWorker::get_dsr_robot_pose(DSR::Node room_node, uint64_t robot_id)
{
    // Print room node name and robot id
    std::cout << "Room node name: " << room_node.name() << " Robot id: " << robot_id << std::endl;
    //get first robot pose
    auto robot_rt = rt_api->get_edge_RT(room_node, robot_id); if (!robot_rt) return{};
    auto robot_translation = G->get_attrib_by_name<rt_translation_att>(robot_rt.value()); if (!robot_translation) {qInfo() << "Robot translation not found"; return{};};
    auto robot_rotation = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(robot_rt.value()); if (!robot_rotation) {qInfo() << "Robot rotastion not found"; return{};};
    auto robot_pose_timestamp = G->get_attrib_by_name<rt_timestamps_att>(robot_rt.value()); if (!robot_pose_timestamp) {qInfo() << "Robot timestamp not found"; return{};};
    return std::make_pair(robot_pose_timestamp.value().get()[0] / 1000.0, Pose3(Rot3::RzRyRx(robot_rotation.value().get()[0], robot_rotation.value().get()[1], robot_rotation.value().get()[2]),
                            Point3(robot_translation.value().get()[0] / 1000.0, robot_translation.value().get()[1] / 1000.0, robot_translation.value().get()[2] / 1000.0)));
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{
    // Check if id == robot_id
    if(id == robot_id)
    {
        if(auto robot_node_ = G->get_node(robot_id); robot_node_.has_value())
        {
            auto robot_node = robot_node_.value();
            // Check if "robot_current_advance_speed" is in att_names
            if(std::find(att_names.begin(), att_names.end(), "robot_current_advance_speed") != att_names.end()) {
                auto timestamp_ = G->get_attrib_by_name<timestamp_alivetime_att>(robot_node);
                    auto adv_speed = G->get_attrib_by_name<robot_current_advance_speed_att>(robot_node);
                    auto ang_speed = G->get_attrib_by_name<robot_current_angular_speed_att>(robot_node);
                    auto side_speed = G->get_attrib_by_name<robot_current_side_speed_att>(robot_node);

                    // Check both values exists
                    if (adv_speed.has_value() && ang_speed.has_value() && side_speed.has_value()) {
                        // Get the values
                        auto adv_speed_val = adv_speed.value();
                        auto ang_speed_val = ang_speed.value();
                        auto side_speed_val = side_speed.value();
                        auto timestamp_val = timestamp_.value();
                        // Print values
//                        std::cout << "Timestamp: " << timestamp_val << " Advance speed: " << adv_speed_val << " Angular speed: " << ang_speed_val << " Side speed: " << side_speed_val << std::endl;
                        // Insert into odometry queue
                        //
//                        odometry_queue.push_back(std::make_tuple(
//                                timestamp_val / 1000.0, Eigen::Vector3d(side_speed_val, adv_speed_val, 0),
//                                Eigen::Vector3d(0, 0, ang_speed_val)));
                         odom_buffer.put(std::make_tuple(
                                timestamp_val / 1000.0, Eigen::Vector3d(side_speed_val, adv_speed_val, 0),
                                Eigen::Vector3d(0, 0, ang_speed_val)));
//                        qInfo() << "Odometry queue size: " << odometry_queue.size();
                    } else {
                        std::cerr << "Problem reading values" << std::endl;
                        return;

                    }
            }
        }
        else
        {
            std::cerr << "Robot node not found" << std::endl;
            std::terminate();
        }
    }
}

void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    //Check if from node is room type?
    auto from_node = G->get_node(from); if(!from_node) return;

    if(type == "current" and from_node.value().type() == "room")
    {
        room_initialized = false;
        auto from_node_room_id = G->get_attrib_by_name<room_id_att>(from_node.value()); if(!from_node_room_id) return;

        if(actual_room_id > 0 and actual_room_id != from_node_room_id)
        {
            last_room_id = actual_room_id;
            current_edge_set = true;
        }
        return;
    }

    auto to_node = G->get_node(to); if(!to_node) return;
    if(type == "RT" and from_node.value().type() == "room" and to_node.value().name() == "Shadow")
    {
        auto rt_edge = G->get_edge(from, to, "RT");
        auto now = std::chrono::system_clock::now();
        auto time_difference = std::chrono::duration_cast<std::chrono::seconds>(now - rt_set_last_time);
        //get rt translation from rt_edge
        if (rt_edge->agent_id() == agent_id and time_difference.count() > rt_time_min)
        {
            room_initialized = false;
            first_rt_set = true;
            rt_set_last_time = std::chrono::system_clock::now();
            //get translation from rt_edge
            auto rt_translation = G->get_attrib_by_name<rt_translation_att>(rt_edge.value()); if(!rt_translation) return;
            translation_to_set = rt_translation.value().get();
            //get rotation from rt_edge
            auto rt_rotation = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(rt_edge.value()); if(!rt_rotation) return;
            rotation_to_set = rt_rotation.value().get();
        }
    }
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}



