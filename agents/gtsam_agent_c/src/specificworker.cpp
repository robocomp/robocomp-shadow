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
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
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
}

void SpecificWorker::compute()
{
    // static double last_timestamp = 0.0;
    // Statemaching
    // Pop odometry value
    auto odom  = odom_buffer.try_get();
    if(!odom.has_value()) return;

    update_robot_odometry_data_in_DSR(odom.value());
    auto [timestamp, translation, rotation] = odom.value();

    // qInfo() << "Odom time difference" << timestamp - last_timestamp;
    // last_timestamp = timestamp;
    timestamp = timestamp / 1000.0; // Convert milliseconds to seconds

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
                // print the complete odom timestamp number, e.g. 1700000000.123456
//                std::cout << std::fixed << std::setprecision(10) << timestamp << std::endl;

//                std::cout << "Timestamp diff: " << timestamp - last_odometry_timestamp << std::endl;
                auto [last_pose_timestamp, robot_translation_from_last_timestamp] = integrate_odometry(timestamp, translation, rotation);
                gtsam_graph.insert_odometry_pose(last_pose_timestamp, gtsam::Pose3(robot_translation_from_last_timestamp.matrix()));

                auto robot_without_corners_pose = gtsam_graph.get_robot_pose();
                Eigen::Vector2d robot_position(robot_without_corners_pose.translation().x(),
                                              robot_without_corners_pose.translation().y());
                if(not safe_polygon.empty() and is_pose_inside_polygon(robot_position, safe_polygon))
                {
                    // Get measured corners and nodes graph timestamps
                    auto actual_measured_corners = get_measured_corners(last_pose_timestamp);
                    draw_measured_corners(&room_widget->viewer->scene, actual_measured_corners);
                    gtsam_graph.add_landmark_measurement(actual_measured_corners);
                }

                // Print optimized pose
                auto act_robot_pose = gtsam_graph.get_robot_pose();
                const gtsam::Point3 translation = act_robot_pose.translation();
                const double x = translation.x();
                const double y = translation.y();
                const gtsam::Rot3 rotation = act_robot_pose.rotation();
                const double yaw = rotation.yaw();     // Z-axis rotation
                // Print formatted output
//               std::cout << "  X: " << std::setw(8) << x
//                         << "  Y: " << std::setw(8) << y
//                         << "  alpha: " << std::setw(8) << yaw << std::endl;
                update_robot_dsr_pose(x, y, yaw, timestamp);
                gtsam_graph.draw_graph_nodes(&room_widget->viewer->scene);
            }
            break;
        }
    }
    last_odometry_timestamp = timestamp;

    // State 2: Graph initialized. Update
}

void SpecificWorker::update_robot_odometry_data_in_DSR(const std::tuple<double, Eigen::Vector3d, Eigen::Vector3d> &odom_value)
{
    auto robot_node = G->get_node(params.robot_name);
    if(not robot_node.has_value())
    {  qWarning() << "Robot node" << QString::fromStdString(params.robot_name) << "not found"; return; }

    auto robot_node_value = robot_node.value();
    const auto [timestamp, translation, rotation] = odom_value;
    G->add_or_modify_attrib_local<robot_current_advance_speed_att>(robot_node_value, (float)  translation.y());
    G->add_or_modify_attrib_local<robot_current_side_speed_att>(robot_node_value, (float) translation.x());
    G->add_or_modify_attrib_local<robot_current_angular_speed_att>(robot_node_value, (float) rotation.z());
    G->add_or_modify_attrib_local<timestamp_alivetime_att>(robot_node_value, (uint64_t) timestamp);
    G->update_node(robot_node_value);
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
        double current_time, Eigen::Vector3d translation, Eigen::Vector3d rotation)
{
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

bool SpecificWorker::is_pose_inside_polygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon)
{
    int count = 0;
    size_t n = polygon.size();
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector2d a = polygon[i];
        Eigen::Vector2d b = polygon[(i + 1) % n];

        // Ver si hay cruce con el eje horizontal del punto
        if ((a.y() > point.y()) != (b.y() > point.y())) {
            double xIntersect = (b.x() - a.x()) * (point.y() - a.y()) / (b.y() - a.y()) + a.x();
            if (point.x() < xIntersect)
                count++;
        }
    }
    return count % 2 == 1; // true si número impar de cruces
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
//                qInfo() << "Getting measured corner" << corner_id.value();
                auto corner_edge = rt_corner_wall.value();

//                // Get rt_timestamps attribute
//                auto rt_timestamps = G->get_attrib_by_name<rt_timestamps_att>(corner_edge);
//                if (rt_timestamps.has_value())
//                {
//                    auto rt_timestamps_value = rt_timestamps.value().get();
//                    for(const auto& rt_timestamp : rt_timestamps_value)
//                    {
//                        qInfo() << "RT timestamp: " << rt_timestamp;
//                    }
//                }


                auto corner_pose = rt_api->get_edge_RT_as_stamped_rtmat(corner_edge, (uint64_t)(timestamp * 1000));
                if (!corner_pose.has_value()) {std::cout << "There is no corner pose for corner id " << corner_id.value() << std::endl; continue;}
                auto corner_pose_value = corner_pose.value();
                auto chosen_corner_timestamp = corner_pose_value.first;

//                std::cout << "Chosen corner timestamp: " << chosen_corner_timestamp << " Comparing: " << corners_last_update_timestamp[corner_id.value()] << std::endl;

                if(corners_last_update_timestamp[corner_id.value()] == chosen_corner_timestamp) continue;
                else corners_last_update_timestamp[corner_id.value()] = chosen_corner_timestamp;

                // Considering corner_pose_value is a Eigen::Transform<double, 3, Eigen::Affine>, get x and y pose
                auto corner_pose_matrix = corner_pose_value.second.matrix();

//                std::cout << "Corner " << corner_id.value() << " timestamp: " << corner_pose_value.first << " Ordered:" << (uint64_t)(timestamp * 1000) << " position: " << corner_pose_matrix(0, 3) << " " <<  corner_pose_matrix(1, 3) << std::endl;
                Eigen::Vector3f corner_pose_(corner_pose_matrix(0, 3) / 1000, corner_pose_matrix(1, 3) / 1000, 0.f);
                auto corner_pose_double = corner_pose_.cast<double>();
                returned_corners.push_back(std::make_tuple(corner_id.value(), corner_pose_value.first / 1000.0, corner_pose_double, corner_is_valid.value()));
            }
        }
    }
    return returned_corners;
}

void SpecificWorker::update_robot_dsr_pose(double x, double y, double ang, double timestamp)
{
//    // Get room node
//    auto room_node_ = G->get_node("room");
//    if (!room_node_) return;
//    auto room_node = room_node_.value();

    // Get robot node
    auto robot_node_ = G->get_node("Shadow");
    if (!robot_node_) return;
    auto robot_node = robot_node_.value();

    // Get robot node parent node
    auto robot_parent_node = G->get_parent_node(robot_node); if (!robot_parent_node.has_value()) return;
    auto robot_parent_node_value = robot_parent_node.value();

    rt_api->insert_or_assign_edge_RT(robot_parent_node_value, robot_node.id(), { (float)(x * 1000.f), (float)(y * 1000.f), 0 }, {0, 0, (float)ang}, (uint64_t) (timestamp * 1000.0));
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

void SpecificWorker::draw_nominal_corners(QGraphicsScene *pScene, const std::vector<std::tuple<int, double, Eigen::Vector3d>> &nominal_corners, const std::vector<Eigen::Vector2d> &security_polygon)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ pScene->removeItem(i); delete i;}
    items.clear();
    QPolygonF poly, sec_poly;
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
    for(const auto &corner: security_polygon)
    {
        sec_poly << QPointF(corner.x() * 1000, corner.y() * 1000);
    }
    auto poly_item = pScene->addPolygon(poly, QPen(QColor("red"), 100));
    items.push_back(poly_item);
    auto sec_poly_item = pScene->addPolygon(sec_poly, QPen(QColor("green"), 100));
    items.push_back(sec_poly_item);
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
    gtsam_graph.reset_graph();

    //get shadow robot node from G
    auto robot_node_ = G->get_node("Shadow"); if (!robot_node_) return false;
    auto& robot_node = robot_node_.value();

    // Get robot node parent node
    auto robot_parent_node = G->get_parent_node(robot_node); if (!robot_parent_node.has_value()) return false;
    auto robot_parent_node_value = robot_parent_node.value();

    // Get timestamp from the first odom_value in the buffer
    auto first_odom_value = odometry_queue.front();
    odometry_queue.pop_front();
    // Get the timestamp from the first odometry value
    auto first_odom_timestamp = std::get<0>(first_odom_value);

    // If robot parent node is "root":
    if (robot_parent_node_value.name() == "root")
    {
        std::cerr << "Robot parent node is root. Integrating pure odometry." << std::endl;
        auto robot_origin = Pose3(Rot3::RzRyRx(0, 0, 0),
              Point3(0, 0, 0));
        gtsam_graph.insert_prior_pose(first_odom_timestamp, robot_origin);
        return true;
    }

    odometry_node_id = robot_node.id();

//    # Get first robot pose
    auto robot_pose = get_dsr_robot_pose();
    if (!robot_pose.has_value())
    {
        std::cerr << "Robot pose not found" << std::endl;
        return false;
    }
    auto [act_pose_timestamp, robot_pose_value] = robot_pose.value();
    // Print robot pose
    std::cout << "Robot pose: " << robot_pose_value.translation() << " " << robot_pose_value.rotation().yaw() << " " << act_pose_timestamp << std::endl;

    gtsam_graph.insert_prior_pose(first_odom_timestamp, robot_pose_value);

    auto current_edge = G->get_edges_by_type("current");
    if (!current_edge.empty())
    {
        auto current_room_node_ = G->get_node(current_edge[0].to()); if (!current_room_node_) return false;
        auto& current_room_node = current_room_node_.value();
        //get attribute room_id from current room node
        auto current_room_id = G->get_attrib_by_name<room_id_att>(current_room_node); if (!current_room_id) return false;

        if(actual_room_id != current_room_node.id())
            last_room_id = actual_room_id;
        actual_room_id = current_room_id.value();
        actual_room_name = current_room_node.name();


        auto nominal_corners_data = get_nominal_corners();
        if(nominal_corners_data.empty())
        {
            std::cerr << "No nominal corners found" << std::endl;
//        return false;
        }
        // Generate aux vector only with the Eigen::Vector3d corners
        std::vector<Eigen::Vector2d> corners_2D;
        for (const auto& corner_data : nominal_corners_data)
        {
            auto [corner_id, corner_timestamp, corner_pose] = corner_data;
            corners_2D.emplace_back(corner_pose.x() / 1000.0, corner_pose.y() / 1000.0);
            std::cout << "Corner id: " << corner_id << " Corner timestamp: " << corner_timestamp / 1000.0<< std::endl;
            gtsam_graph.insert_landmark_prior(corner_timestamp, corner_id, Point3(corner_pose.x()/ 1000.0, corner_pose.y()/ 1000.0, 0.0));
            corners_last_update_timestamp[corner_id] = (uint64_t )corner_timestamp * 1000;
        }
        safe_polygon = shrink_polygon_to_safe_zone(corners_2D, 0.4);
        draw_nominal_corners(&room_widget->viewer->scene, nominal_corners_data, safe_polygon);
    }

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

// Proyectar las esquinas y reducir el polígono
std::vector<Eigen::Vector2d> SpecificWorker::shrink_polygon_to_safe_zone(const std::vector<Eigen::Vector2d>& corners_2D, double margin_meters)
{
    ClipperOffset offsetter;
    Path64 path = toClipper2Path(corners_2D);

    offsetter.AddPath(path, JoinType::Miter, EndType::Polygon);
    Paths64 solution;
    offsetter.Execute(-margin_meters * SCALE, solution);

    if (solution.empty()) return {};
    return fromClipper2Path(solution[0]);
}

// Convierte de Eigen a Clipper2
Path64 SpecificWorker::toClipper2Path(const std::vector<Eigen::Vector2d>& poly) {
    Path64 path;
    for (const auto& p : poly) {
        path.emplace_back(static_cast<int64_t>(p.x() * SCALE), static_cast<int64_t>(p.y() * SCALE));
    }
    return path;
}

// Convierte de Clipper2 a Eigen
std::vector<Eigen::Vector2d> SpecificWorker::fromClipper2Path(const Path64& path) {
    std::vector<Eigen::Vector2d> poly;
    for (const auto& pt : path) {
        poly.emplace_back(pt.x / SCALE, pt.y / SCALE);
    }
    return poly;
}


std::optional<std::pair<double, gtsam::Pose3>> SpecificWorker::get_dsr_robot_pose()
{
    auto robot_node = G->get_node("Shadow");
    if (!robot_node) {qInfo() << "Robot node not found"; return{};};
    // Get robot node parent node
    auto robot_parent_node = G->get_parent_node(robot_node.value()); if (!robot_parent_node.has_value()) {qInfo() << "Robot parent node not found"; return{};};
    // Print room node name and robot id
    std::cout << "Room node name: " << robot_parent_node.value().name() << " Robot id: " << robot_id << std::endl;
    //get first robot pose
    auto robot_rt = rt_api->get_edge_RT(robot_parent_node.value(), robot_node.value().id()); if (!robot_rt) return{};
    auto robot_translation = G->get_attrib_by_name<rt_translation_att>(robot_rt.value()); if (!robot_translation) {qInfo() << "Robot translation not found"; return{};};
    auto robot_rotation = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(robot_rt.value()); if (!robot_rotation) {qInfo() << "Robot rotastion not found"; return{};};
    auto robot_pose_timestamp = G->get_attrib_by_name<rt_timestamps_att>(robot_rt.value()); if (!robot_pose_timestamp) {qInfo() << "Robot timestamp not found"; return{};};
    return std::make_pair(robot_pose_timestamp.value().get()[0] / 1000.0, Pose3(Rot3::RzRyRx(robot_rotation.value().get()[0], robot_rotation.value().get()[1], robot_rotation.value().get()[2]),
                            Point3(robot_translation.value().get()[0] / 1000.0, robot_translation.value().get()[1] / 1000.0, robot_translation.value().get()[2] / 1000.0)));
}

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if(type == "room")
    {
//        init_graph = 0;
        actual_room_id = id;
    }
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{
//    if(id == actual_room_id and std::find(att_names.begin(), att_names.end(), "obj_checked") != att_names.end())
//    {
//        qInfo() << "Room node obj_checked attribute modified. Reinitializing graph.";
//        init_graph = 0;
//    }
}

void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    //Check if from node is room type?
    auto from_node = G->get_node(from); if(!from_node) return;

    if(type == "current" and from_node.value().type() == "room")
    {
        qInfo() << "Current edge inserted. Reinitializing graph.";
        init_graph = 0;
    }

//    auto to_node = G->get_node(to); if(!to_node) return;
//    if(type == "RT" and from_node.value().type() == "room" and to_node.value().name() == "Shadow")
//    {
//        auto rt_edge = G->get_edge(from, to, "RT");
//        auto now = std::chrono::system_clock::now();
//        auto time_difference = std::chrono::duration_cast<std::chrono::seconds>(now - rt_set_last_time);
//        //get rt translation from rt_edge
//        if (rt_edge->agent_id() == agent_id and time_difference.count() > rt_time_min)
//        {
//            room_initialized = false;
//            first_rt_set = true;
//            rt_set_last_time = std::chrono::system_clock::now();
//            //get translation from rt_edge
//            auto rt_translation = G->get_attrib_by_name<rt_translation_att>(rt_edge.value()); if(!rt_translation) return;
//            translation_to_set = rt_translation.value().get();
//            //get rotation from rt_edge
//            auto rt_rotation = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(rt_edge.value()); if(!rt_rotation) return;
//            rotation_to_set = rt_rotation.value().get();
//        }
//    }
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}

//SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
void SpecificWorker::FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)
{
    #ifdef HIBERNATION_ENABLED
        hibernation = true;
    #endif

    odom_buffer.put(std::make_tuple(
            double(pose.timestamp), Eigen::Vector3d(pose.vy, pose.vx, 0),
            Eigen::Vector3d(0, 0, pose.vrz)));
//    qInfo() << "odom value" << pose.vy << pose.vx << pose.vrz << double(pose.timestamp);
}



