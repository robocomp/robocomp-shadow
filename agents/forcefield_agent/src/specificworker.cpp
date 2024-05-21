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
#include <cppitertools/range.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/filter.hpp>
#include <cppitertools/chunked.hpp>
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
        this->params.DISPLAY = params.at("display").value == "true" or (params.at("delay").value == "True");
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

        rt = G->get_rt_api();
        inner_eigen = G->get_inner_eigen_api();

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

        widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));
        //widget_2d->setSceneRect(-5000, -5000,  10000, 10000);

        // A thread is created to read lidar data
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        // Save the current time using std::chrono
        starting_time = std::chrono::system_clock::now();

        // timers
        Period = params.PERIOD;    //  used in the lidar reader thread
        timer.start(Period);
        std::cout << "Worker initialized OK" << std::endl;
	}
}
void SpecificWorker::compute()
{
    /// Wait 5 seconds before starting using starting_time
//if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - starting_time).count() < 5)
//    {
//        std::cout << "Waiting 5 seconds before starting" << std::endl;
//        return;
//    }



    /// read LiDAR
    auto res_ = buffer_lidar_data.get_idemp();
//    if (not res_.has_value()) { qWarning() << "No lidar data available"; return; }
//    auto ldata = res_.value();

    /// Read robot node
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value()) { qWarning() << "No robot node available"; return; }
    auto robot_node = robot_node_.value();

    /// draw lidar
//    if(widget_2d != nullptr)
//        draw_lidar(ldata, &widget_2d->scene);

    /// extract lines from LiDAR data
    Lines lines = extract_2D_lines_from_lidar3D(res_.points, params.ranges_list);

    /// Check if any room node exists in the graph
    if(auto room_nodes = G->get_nodes_by_type("room"); room_nodes.empty())
    {
        /// Room detector
        auto current_room = room_detector.detect({lines[0]}, &widget_2d->scene, true);  // TODO: use upper lines in Helios

        /// If not current_room is found, return
        current_room.draw_on_2D_tab(current_room, "yellow", &widget_2d->scene);

        /// Check if TARGET edge exists
        if(auto target_edge_ = G->get_edge(robot_node.id(), robot_node.id(), "TARGET"); target_edge_.has_value())
        {
            Eigen::Vector2f room_center;
            if(not current_room.is_initialized)
            {
                qInfo() << "No room detected. Using estimated center.";
                room_center = current_room.estimated_center;
            }
            else
                room_center = current_room.get_center();
            qInfo() << "Room center: " << room_center.x() << " " << room_center.y();

            static std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> corner_data;
            static std::vector<std::vector<float>> odometry_data;
            static std::vector<Eigen::Vector2f> room_centers;
            static std::vector<Eigen::Vector2f> room_sizes;

            static std::map<std::vector<int>, int> room_size_histogram;
            static bool first_room_center_set = false;

            ///-------------------- ROBOT MOVEMENT ------------------------------///
            if(movement_completed(room_center, distance_to_target))
            {
                qInfo() << "Vector sizes: " << corner_data.size() << " " << odometry_data.size() << " " << room_centers.size() << " " << room_sizes.size();
                qInfo() << "Room center reached";
                // Generate room size histogram considering room_size_histogram vector and obtain the most common room size
                auto most_common_room_size = std::max_element(room_size_histogram.begin(), room_size_histogram.end(),
                                                              [](const auto &p1, const auto &p2){ return p1.second < p2.second; });
                /// Get the most common room size
                auto room_size = most_common_room_size->first;
                qInfo() << "Most common room size: " << room_size[0] << " " << room_size[1] << "#############################";
                /// Calculate corner_matching_threshold based on room size
                corner_matching_threshold = std::min(room_size[0]/3, room_size[1]/3);
                corner_matching_threshold_setted = true;
                /// Get the first room center obtained with a room size similar to the most common one
                int first_valid_center_id = 0;
                Eigen::Vector2f first_valid_room_center;
                /// Get first valid center
                for(const auto &[i, center] : iter::enumerate(room_centers))
                    if(std::abs(room_sizes[i][0] - room_size[0]) < 50 and std::abs(room_sizes[i][1] - room_size[1]) < 50)
                    {
                        first_valid_center_id = i;
                        first_valid_room_center = center;
                        break;

                    }
                /// Remove values in vectors behind first valid center
                corner_data.erase(corner_data.begin(), corner_data.begin() + first_valid_center_id);
                odometry_data.erase(odometry_data.begin(), odometry_data.begin() + first_valid_center_id);
                room_sizes.erase(room_sizes.begin(), room_sizes.begin() + first_valid_center_id);
//                qInfo() << "After cleaning valid corners";
//                for(const auto &corner : corner_data[0])
//                    qInfo() << corner.x() << " " << corner.y();
                /// Get robot initial pose in room and nominal corners
                auto robot_initial_pose = get_robot_initial_pose( first_valid_room_center, corner_data[0], room_size[0], room_size[1]);
                // Print nominal corners
                for(const auto &corner : robot_initial_pose.second)
                    qInfo() << "Nominal corner: " << corner.x() << " " << corner.y();
                /// Generate g2o graph considering first robot pose, nominal corners, corners and odometry measured along the trajectory to room center
                auto g2o_str = build_g2o_graph(corner_data, odometry_data, robot_initial_pose.first, robot_initial_pose.second, room_sizes, room_size);
                /// Stop robot
                set_robot_speeds(0.f, 0.f, 0.f);
                /// Call g2o optimizer
                auto optimization = this->g2ooptimizer_proxy->optimize(g2o_str);
                /// Get optimized nominal corners and actual robot pose
                auto g2o_data = extract_g2o_data(optimization);
                qInfo() << "Optimized corners: ";
                for(const auto &corner : std::get<0>(g2o_data))
                    qInfo() << corner.x() << " " << corner.y();
                qInfo() << "Optimized robot pose: " << std::get<1>(g2o_data).x() << " " << std::get<1>(g2o_data).y() << " " << std::get<1>(g2o_data).z();
                /// Associate start nominal corners with optimized
                auto associated_nominal_corners = calculate_rooms_correspondences_id(std::get<0>(g2o_data), robot_initial_pose.second, true);
//                for (const auto &[i, corner] : iter::enumerate(associated_nominal_corners))
//                    std::get<0>(g2o_data)[std::get<0>(associated_nominal_corners[i])] = std::get<2>(associated_nominal_corners[i]);
                /// Order corners clockwise
                std::sort(std::get<0>(g2o_data).begin(), std::get<0>(g2o_data).end(), [](const auto &p1, const auto &p2){ return std::atan2(p1.y(), p1.x()) < std::atan2(p2.y(), p2.x()); });
                /// Insert room data in graph
                insert_room_into_graph(g2o_data, current_room);
                /// Delete TARGET edge to make robot control agent stop taking into account speed commands
                G->delete_edge(robot_node.id(), robot_node.id(), "TARGET");
                current_room.draw_on_2D_tab(current_room, "yellow", &widget_2d->scene, true);

            }
            ///---------- If robot is not in room center, move robot to room center
            else
            {
//                qInfo() << "Trying to read room data";
                /// Check if room is initialized
                if(current_room.is_initialized)
                {
                    /// Get room center
                    room_centers.push_back(current_room.get_center());

                    /// Get room corner values
                    auto corners = current_room.get_corners();
                    corner_data.push_back(corners);

                    /// Get room dimension
                    auto room_width = static_cast<int>(current_room.get_width());
                    auto room_depth = static_cast<int>(current_room.get_depth());

                    /// Add room size to histogram considering first the biggest dimension to avoid duplicated dimensions
                    if(room_width > room_depth)
                    {
                        /// Add room size to vector
                        room_size_histogram[{room_width, room_depth}]++;
                        room_sizes.push_back({room_width, room_depth});
                    }
                    else
                    {
                        /// Add room size to vector
                        room_size_histogram[{room_depth, room_width}]++;
                        room_sizes.push_back({room_depth, room_width});
                    }
                    qInfo() << "Room size: " << room_width << " " << room_depth;

                    /// Get robot odometry
                    auto odometry = get_graph_odometry();
                    odometry_data.push_back(odometry);
                }
                else
                    qInfo() << "Room not initialized. Can't store data.";

                /// Generate speed commands towards room center and insert them in graph
                auto speeds = calculate_speed(room_center);
                set_robot_speeds(speeds[0],speeds[1],speeds[2]);
            }
        }
        else
            /// Generate target edge to make the robot control agent take into account speed commands
            generate_target_edge(robot_node);
    }
    /// If room node exists, update corner values
    else
    {
        /// check if corner_matching_threshold is set (in case forcefield had to be reset)
        if(not corner_matching_threshold_setted)
        {
            /// Get room size
            auto room_width = G->get_attrib_by_name<width_att>(room_nodes[0]);
            auto room_depth = G->get_attrib_by_name<depth_att>(room_nodes[0]);
            if(room_width.has_value() and room_depth.has_value())
            {
                /// Calculate corner_matching_threshold based on room size
                corner_matching_threshold = std::min(room_width.value()/3, room_depth.value()/3);
                corner_matching_threshold_setted = true;
            }
        }

        auto corners = room_detector.detect_corners({lines[0]}, &widget_2d->scene, true);
        update_room_data(corners, &widget_2d->scene);
    }
    graph_viewer->set_external_hz((int)(1.f/fps.get_period()*1000.f));
    fps.print("room_detector");
}

////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::generate_target_edge(DSR::Node node)
{
    DSR::Edge target_edge = DSR::Edge::create<TARGET_edge_type>(node.id(), node.id());
     if (G->insert_or_assign_edge(target_edge))
         std::cout << __FUNCTION__ << " Target edge successfully inserted: " << std::endl;
     else
         std::cout << __FUNCTION__ << " Fatal error inserting new edge: " << std::endl;
}
SpecificWorker::Lines SpecificWorker::extract_2D_lines_from_lidar3D(const RoboCompLidar3D::TPoints &points,
                                                                    const std::vector<std::pair<float, float>> &ranges)
{
    Lines lines(ranges.size());
    for(const auto &p: points)
        for(const auto &[i, r] : ranges | iter::enumerate)
            if(p.z > r.first and p.z < r.second)
                lines[i].emplace_back(p.x, p.y);
    return lines;
}

void SpecificWorker::read_lidar()
{
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarData(params.lidar_name, -90, 360, 3);
            buffer_lidar_data.put(std::move(data));
            // TODO: filter out zero and very large values
            //, [](auto &&I, auto &T)
            //    { for(auto &&i : I.points)
            //        if(i.z > 1000) T.points.push_back(i);  });
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D " << e << std::endl; }
        std::this_thread::sleep_for(std::chrono::milliseconds(params.LIDAR_SLEEP_PERIOD));} // TODO: dynamic adjustment
}
void SpecificWorker::set_robot_speeds(float adv, float side, float rot)
{
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    G->add_or_modify_attrib_local<robot_ref_adv_speed_att>(robot_node, adv);
    G->add_or_modify_attrib_local<robot_ref_side_speed_att>(robot_node, side);
    G->add_or_modify_attrib_local<robot_ref_rot_speed_att>(robot_node, rot);

    G->update_node(robot_node);
}
std::vector<float> SpecificWorker::calculate_speed(const Eigen::Matrix<float, 2, 1> &target)
{
    // Calculate the angle between the robot and the target
    Eigen::Matrix<float, 2, 1> robot_pos_vector = {0.0, 0.0};
    Eigen::Matrix<float, 2, 1> target_vector = target;
    Eigen::Matrix<float, 2, 1> target_vector_rotated = target_vector - robot_pos_vector;
    float angle = std::atan2(target_vector_rotated(0), target_vector_rotated(1));
    if (angle > M_PI)
        angle -= 2 * M_PI;
    if (angle < -M_PI)
        angle += 2 * M_PI;

    // If angle > 90 degrees, return only rotation
    if (std::abs(angle) > M_PI_2)
        return {0.0, 0.0, angle};

    // Calculate the advance speed // TODO: modificar antes de que Pablo lo vea
    float advance_speed = std::cos(angle) * max_robot_advance_speed;
    // Calculate the side speed
    float side_speed = std::sin(angle) * max_robot_side_speed;
    float rotation_speed = angle;

    // Return the speeds
    return {advance_speed, side_speed, rotation_speed};
}
bool SpecificWorker::movement_completed(const Eigen::Vector2f &target, float distance_to_target)
{
    return (target.norm() < distance_to_target);
}
std::tuple<std::vector<Eigen::Vector2d>, Eigen::Vector3d> SpecificWorker::extract_g2o_data(string optimization)
{
// Print function name
    std::cout << __FUNCTION__ << std::endl;

    std::istringstream iss(optimization);
    // Find VERTEX_XY with ids 1, 2, 3, 4
    std::vector<Eigen::Vector2d> corners;
    Eigen::Vector3d robot_pose;
    std::string line;
    while (std::getline(iss, line))
    {
        std::istringstream iss_line(line);
        std::string type;
        int id;
        float x, y, alpha;
        // Get type
        iss_line >> type;
        // Get id
        iss_line >> id;
        // Get x
        iss_line >> x;
        // Get y
        iss_line >> y;
        if(type == "VERTEX_XY")
            corners.push_back({x, y});
        else if(type == "VERTEX_SE2")
        {
            // Store vertex data with highest id
            // Get alpha
            iss_line >> alpha;
            robot_pose = {x, y, alpha};
        }

        else if(type == "FIX")
            // pass to next line
            qInfo() << "Fixed node";
        else
        {

            return std::make_tuple(corners, robot_pose);
        }
    }

}
void SpecificWorker::insert_room_into_graph(tuple<std::vector<Eigen::Vector2d>, Eigen::Vector3d> optimized_room_data, const rc::Room &current_room)
{
    auto root_node_ = G->get_node("root");
    if(not root_node_.has_value())
    { qWarning() << __FUNCTION__ << " No root node in graph"; return; }
    auto root_node = root_node_.value();

    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    auto robot_level_ = G->get_node_level(robot_node);
    if(not robot_level_.has_value())
    { qWarning() << __FUNCTION__ << " No robot level in graph"; return; }
    auto robot_level = robot_level_.value();

    // Calculate room dimensions with room data
    auto room_corners = std::get<0>(optimized_room_data);
    auto room_width = static_cast<int>((room_corners[1] - room_corners[0]).norm());
    auto room_depth = static_cast<int>((room_corners[0] - room_corners[3]).norm());
    // Create room node and insert it in graph
    DSR::Node room_node = DSR::Node::create<room_node_type>("room");
    G->add_or_modify_attrib_local<width_att>(room_node, room_width);
    G->add_or_modify_attrib_local<depth_att>(room_node, room_depth);
    G->add_or_modify_attrib_local<pos_x_att>(room_node, (float)(rand()%(170)));
    G->add_or_modify_attrib_local<pos_y_att>(room_node, (float)(rand()%170));
    G->add_or_modify_attrib_local<obj_checked_att>(room_node, false);
    G->add_or_modify_attrib_local<level_att>(room_node, robot_level);
    G->insert_node(room_node);

    rt->insert_or_assign_edge_RT(root_node, room_node.id(), { 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f });
    G->delete_edge(root_node.id(), robot_node.id(), "RT");

    // Increase robot node level
    G->add_or_modify_attrib_local<level_att>(robot_node, robot_level + 1);
    G->update_node(robot_node);
    // Insert robot pose in room
    auto robot_pose = std::get<1>(optimized_room_data);
    auto robot_pose_float = robot_pose.cast<float>();
    rt->insert_or_assign_edge_RT(room_node, robot_node.id(), {robot_pose_float.x(), robot_pose_float.y(), 0.f}, { 0.f, 0.f, robot_pose_float.z() });

    // insert walls and corners in graph
    // Iterate corners using iter enumerate
    std::vector<float> corner_pos, wall_pos;
    Eigen::Vector2d wall_center_point;
    float wall_angle = 0.0;
    for (const auto &[i, corner] : iter::enumerate(room_corners))
    {
        std::cout << "Corner " << i << ": " << corner.x() << " " << corner.y() << std::endl;
        // Check if i is the last corner
        if(i == room_corners.size() - 1)
        {
            wall_center_point = (room_corners[0] + corner)/2;
            std::cout << "Next corner " << 0 << ": " << room_corners[0].x() << " " << room_corners[0].y() << std::endl;
        }
        else
        {
            wall_center_point = (corner + room_corners[i+1])/2;
            std::cout << "Next corner " << i+1 << ": " << room_corners[i+1].x() << " " << room_corners[i+1].y() << std::endl;
        }
        // Obtain wall pose considering the corner and the next corner

        auto wall_center_point_float = wall_center_point.cast<float>();
        wall_pos = {wall_center_point_float.x(), wall_center_point_float.y(), 0.f};
        // Print wall center point
        std::cout << "Wall center point: " << wall_center_point_float.x() << " " << wall_center_point_float.y() << std::endl;
        wall_angle = std::atan2(wall_center_point.y(), wall_center_point.x()) - M_PI_2;
        std::cout << "Wall angle: " << wall_angle << std::endl;
        // Obtain corner pose with respect to the wall
        auto corner_float = corner.cast<float>();
        // Check if i is even
        if(i % 2 == 0)
            corner_pos = {-abs(corner_float.x()), 0.0, 0.0};
        else
            corner_pos = {-abs(corner_float.y()), 0.0, 0.0};
        std::cout << "Corner position: " << corner_pos[0] << " " << corner_pos[1] << std::endl;
        // insert nominal values
        create_wall(i, wall_pos, wall_angle, room_node);
        if(auto wall_node_ = G->get_node("wall_" + std::to_string(i)); wall_node_.has_value())
        {
            auto wall_node = wall_node_.value();
            create_corner(i, corner_pos, wall_node);
        }
    }
    // Transform optimized corners to robot frame
    auto transformed_corners = get_transformed_corners(&widget_2d->scene);
    // Compare optimized corners with measured corners
    // Get room corners
    auto target_points_ = current_room.get_corners();

    // Cast corners to Eigen::Vector2d through a lambda
    std::vector<Eigen::Vector2d> target_points;
    std::transform(target_points_.begin(), target_points_.end(), std::back_inserter(target_points),
                   [](const auto &p){ return Eigen::Vector2d(p.x(), p.y()); });
    // Print target_points_
    for(const auto &corner : target_points)
    {
        std::cout << "Corner measured" << ": "
                  << corner.x() << " " << corner.y()
                  << std::endl;
    }

    // Calculate correspondences between optimized and measured corners
    auto correspondences = calculate_rooms_correspondences_id(transformed_corners, target_points);
    // Insert measured corners in graph
    for (const auto &[id, p, p2, valid] : correspondences)
    {
        if(valid)
        {
            create_corner(id, {(float)p2.x(), (float)p2.y(), 0.0}, robot_node, false);
        }
    }

    auto new_room_node_ = G->get_node("room");
    if(not new_room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto new_room_node = new_room_node_.value();
    G->add_or_modify_attrib_local<valid_att>(new_room_node, true);
    G->update_node(new_room_node);

}
std::vector<Eigen::Vector2d> SpecificWorker::get_transformed_corners(QGraphicsScene *scene)
{
    std::vector<Eigen::Vector2d> rt_corner_values;
    //dest org
    static std::vector<QGraphicsItem *> items;
    for (const auto &i: items) {
        scene->removeItem(i);
        delete i;
    }
    items.clear();
    std::vector<Eigen::Vector3d> drawn_corners{4};
    for (int i = 0; i < 4; i++)
    {
        auto corner_aux_ = G->get_node("corner_" + std::to_string(i));
        if (not corner_aux_.has_value()) {
            qWarning() << __FUNCTION__ << " No nominal corner " << i << " in graph";
            return rt_corner_values;
        }
        auto corner_aux = corner_aux_.value();

        auto wall_aux_ = G->get_node("wall_" + std::to_string(i));
        if (not wall_aux_.has_value()) {
            qWarning() << __FUNCTION__ << " No wall " << i << " in graph";
            return rt_corner_values;
        }
        auto wall_aux = wall_aux_.value();

        auto room_node_ = G->get_node("room"); //TODO: Ampliar si existe más de una room en el grafo
        if(not room_node_.has_value())
        { qWarning() << __FUNCTION__ << " No room level in graph"; return rt_corner_values; }
        auto room_node = room_node_.value();

        auto robot_node_ = G->get_node("Shadow");
        if(not robot_node_.has_value())
        { qWarning() << __FUNCTION__ << " No robot node in graph"; return rt_corner_values; }
        auto robot_node = robot_node_.value();

        if (auto rt_corner_edge = rt->get_edge_RT(wall_aux, corner_aux.id()); rt_corner_edge.has_value())
            if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(
                        rt_corner_edge.value()); rt_translation.has_value())
            {
                auto rt_translation_value = rt_translation.value().get();
                // Get robot pose
                if (auto rt_robot_edge = rt->get_edge_RT(room_node, robot_node.id()); rt_robot_edge.has_value())
                    if (auto rt_translation_robot = G->get_attrib_by_name<rt_translation_att>(
                                rt_robot_edge.value()); rt_translation_robot.has_value())
                    {
                        auto rt_translation_robot_value = rt_translation_robot.value().get();
                        if (auto rt_rotation_robot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(
                                    rt_robot_edge.value()); rt_rotation_robot.has_value())
                        {
                            auto rt_rotation_robot_value = rt_rotation_robot.value().get();
//                            qInfo() << "Robot pose: " << rt_translation_robot_value[0] << " "
//                                    << rt_translation_robot_value[1] << " " << rt_rotation_robot_value[2];
                            // Transform nominal corner position to robot frame
                            Eigen::Vector3f corner_robot_pos_point(rt_translation_value[0],
                                                                   rt_translation_value[1], 0.f);
                            auto corner_robot_pos_point_double = corner_robot_pos_point.cast<double>();
                            if (auto corner_transformed = inner_eigen->transform(robot_node.name(),
                                                                                 corner_robot_pos_point_double,
                                                                                 wall_aux.name()); corner_transformed.has_value())
                            {
                                auto corner_transformed_value = corner_transformed.value();
                                rt_corner_values.push_back({corner_transformed_value.x(), corner_transformed_value.y()});
                            }
                            if (auto nominal_corner = inner_eigen->transform(room_node.name(),
                                                                                 corner_robot_pos_point_double,
                                                                                 wall_aux.name()); nominal_corner.has_value())
                            {
                                auto corner_transformed_value = nominal_corner.value();
                                drawn_corners[i] = corner_transformed_value;
                                // Print corner transformed value
                                qInfo() << "Corner " << i << " transformed: " << corner_transformed_value.x() << " " << corner_transformed_value.y();
                                // Draw corner
                                auto item = scene->addEllipse(-100, -100, 200, 200, QPen(QColor("green"), 100), QBrush(QColor("green")));
                                item->setPos(corner_transformed_value.x(), corner_transformed_value.y());
                                items.push_back(item);
                                // Draw line between ellipses
                                if(i > 0)
                                {
                                    if(i == 3)
                                    {
                                        auto line2 = scene->addLine(drawn_corners[0].x(), drawn_corners[0].y(), drawn_corners[i].x(), drawn_corners[i].y(), QPen(QColor("blue"), 100));
                                        items.push_back(line2);
                                    }
                                    auto line = scene->addLine(drawn_corners[i-1].x(), drawn_corners[i-1].y(), drawn_corners[i].x(), drawn_corners[i].y(), QPen(QColor("blue"), 100));
                                    items.push_back(line);

                                }
                            }
                        }
                    }
            }
    }
    return rt_corner_values;
}
std::pair<Eigen::Affine2d, std::vector<Eigen::Vector2d>> SpecificWorker::get_robot_initial_pose(Eigen::Vector2f &first_room_center, std::vector<Eigen::Matrix<float, 2, 1>> first_corners, int width, int depth)
{
    // Cast first corners to Eigen::Vector2d
    std::vector<Eigen::Vector2d> target_points;
    std::transform(first_corners.begin(), first_corners.end(), std::back_inserter(target_points),
                   [](const auto &p){ return Eigen::Vector2d(p.x(), p.y()); });

    // Create two possible nominal rooms
    std::vector<Eigen::Vector2d> imaginary_room_1 = {{ -width / 2, depth / 2}, { width / 2,  depth / 2}, { -width / 2,  -depth / 2}, { width / 2,  -depth / 2}};
    std::vector<Eigen::Vector2d> imaginary_room_2 = {{ -depth / 2, width / 2}, { depth / 2,  width / 2}, { -depth / 2,  -width / 2}, { depth / 2,  -width /2}};
    std::vector<Eigen::Vector2d> imaginary_room_1_robot_refsys = {{ -width / 2, depth / 2}, { width / 2,  depth / 2}, { -width / 2,  -depth / 2}, { width / 2,  -depth / 2}};
    std::vector<Eigen::Vector2d> imaginary_room_2_robot_refsys = {{ -depth / 2, width / 2}, { depth / 2,  width / 2}, { -depth / 2,  -width / 2}, { depth / 2,  -width /2}};
    //
    auto room_center_double = first_room_center.cast<double>();
    // Considering room center, traslate the imaginary rooms
    for(auto &p: imaginary_room_1)
        p += room_center_double;
    for(auto &p: imaginary_room_2)
        p += room_center_double;

    // Check which of the two imaginary rooms is the closest to the real room
    auto correspondence_1 = this->calculate_rooms_correspondences(target_points, imaginary_room_1);
    auto correspondence_2 = this->calculate_rooms_correspondences(target_points, imaginary_room_2);

    // Calcular la suma de las distancias cuadradas para cada par de puntos
    double sum_sq_dist1 = 0.0;
    double sum_sq_dist2 = 0.0;
    for (size_t i = 0; i < correspondence_1.size(); ++i)
    {
        sum_sq_dist1 += (correspondence_1[i].first - correspondence_1[i].second).squaredNorm();
        sum_sq_dist2 += (correspondence_2[i].first - correspondence_2[i].second).squaredNorm();
    }

    std::vector<Eigen::Vector2d> imaginary_room;
    std::vector<Eigen::Vector2d> imaginary_room_robot_refsys;
    if(sum_sq_dist1 < sum_sq_dist2)
    {
        imaginary_room = imaginary_room_1;
        imaginary_room_robot_refsys = imaginary_room_1_robot_refsys;
    }
    else
    {
        imaginary_room = imaginary_room_2;
        imaginary_room_robot_refsys = imaginary_room_2_robot_refsys;
    }

    // Get robot pose in room
    icp icp(imaginary_room, target_points);
    icp.align();

    auto rotation2d = Eigen::Rotation2Dd(icp.rotation());
    auto angle_rad = rotation2d.angle();

    //generate rotation matrix
    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << cos(angle_rad), sin(angle_rad),
            -sin(angle_rad), cos(angle_rad);


    Eigen::Vector2d translation = rotation_matrix * -room_center_double;

    // Generate affine matrix using Eigen library
    Eigen::Affine2d pose_matrix;
    // Set rotation and traslation data
    pose_matrix.setIdentity();
    pose_matrix.rotate(-angle_rad).pretranslate(translation);

    return std::make_pair(pose_matrix, imaginary_room_robot_refsys);
}
void SpecificWorker::update_room_data(const rc::Room_Detector::Corners &corners, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> items;
    for (const auto &i: items) {
        scene->removeItem(i);
        delete i;
    }
    items.clear();

    auto root_node_ = G->get_node("root");
    if(not root_node_.has_value())
    { qWarning() << __FUNCTION__ << " No root node in graph"; return; }
    auto root_node = root_node_.value();

    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    auto room_node_ = G->get_node("room"); //TODO: Ampliar si existe más de una room en el grafo
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room level in graph"; return; }
    auto room_node = room_node_.value();

    // match phase
    auto now = std::chrono::high_resolution_clock::now();

    // Get room corners
    auto target_points_ = corners;

    // Cast corners to Eigen::Vector2d through a lambda
    std::vector<Eigen::Vector2d> target_points;
    std::transform(target_points_.begin(), target_points_.end(), std::back_inserter(target_points),
                   [](const auto &p){ return Eigen::Vector2d(std::get<QPointF>(p).x(), std::get<QPointF>(p).y()); });

//    // Check if corners are the same as last time iterating
//    std::vector<int> not_updated_corners;
//    if (last_corners.size() == target_points.size())
//        for (int i = 0; i < 4; i++)
//            if (last_corners[i] == target_points[i])
//            {
//                std::string corner_name = "corner_" + std::to_string(i) + "_measured";
//                if (std::optional<DSR::Node> updated_corner = G->get_node(corner_name); updated_corner.has_value())
//                {
//                    G->add_or_modify_attrib_local<valid_att>(updated_corner.value(), false);
//                    G->update_node(updated_corner.value());
//                    not_updated_corners.push_back(i);
//                }
//            }

    // Get nominal corners data transforming to robot frame and measured corners
    if (auto edge_robot_ = rt->get_edge_RT(room_node, robot_node.id()); edge_robot_.has_value())
    {
        //get the rt values from edge in of corner measured nodes and insert the in a std::vector
        std::vector<Eigen::Vector2d> rt_corner_values;

        //OBTENER CORNERS NOMINALES TRANSFORMADOS
        rt_corner_values = get_transformed_corners(&widget_2d->scene);

        // Calculate correspondences between transformed nominal and measured corners
        auto rt_corners_correspondences = calculate_rooms_correspondences_id(rt_corner_values, target_points);
        std::vector<Eigen::Vector3d> drawn_corners{4};
        //Update measured corners
        for (const auto &[i, corner] : iter::enumerate(rt_corners_correspondences))
        {
//            if (std::find(not_updated_corners.begin(), not_updated_corners.end(), i) != not_updated_corners.end())
//                continue;
//            qInfo() << "############################################################################";
//
//            qInfo() << "Corner " << i << " measured: " << std::get<2>(rt_corners_correspondences[i]).x() << " " << std::get<2>(rt_corners_correspondences[i]).y();
//            qInfo() << "Corner " << i << " nominal: " << std::get<1>(rt_corners_correspondences[i]).x() << " " << std::get<1>(rt_corners_correspondences[i]).y();
            std::string corner_name = "corner_" + std::to_string(i) + "_measured";
            if (std::optional<DSR::Node> updated_corner = G->get_node(corner_name); updated_corner.has_value())
            {
                if (std::optional<DSR::Edge> edge = G->get_edge(robot_node.id(), updated_corner.value().id(),
                                                                "RT"); edge.has_value())
                {
                    if (auto corner_id = G->get_attrib_by_name<corner_id_att>(
                                updated_corner.value()); corner_id.has_value())
                    {
                        if (corner_id.value() == std::get<0>(rt_corners_correspondences[i]))
                        {
                            //insert the rt values in the edge
                            G->add_or_modify_attrib_local<valid_att>(updated_corner.value(), std::get<3>(
                                    rt_corners_correspondences[i]));
                            G->update_node(updated_corner.value());
                            G->add_or_modify_attrib_local<rt_translation_att>(edge.value(), std::vector<float>
                                    {
                                    (float) std::get<2>(rt_corners_correspondences[i]).x(),
                                    (float) std::get<2>(rt_corners_correspondences[i]).y(), 0.0f
                                    });
                            G->insert_or_assign_edge(edge.value());

                            // Draw transformed measured corners
                            Eigen::Vector3d rt_translation = {std::get<2>(rt_corners_correspondences[i]).x(),
                                                             std::get<2>(rt_corners_correspondences[i]).y(), 0.0};

                            if (auto corner_transformed = inner_eigen->transform(room_node.name(),
                                                                                 rt_translation,
                                                                                 robot_node.name()); corner_transformed.has_value())
                            {
                                auto corner_transformed_value = corner_transformed.value();
                                drawn_corners[i] = corner_transformed_value;
                                // Print corner transformed value
                                qInfo() << "Corner " << i << " transformed: " << corner_transformed_value.x() << " " << corner_transformed_value.y();
                                // Draw corner
                                if(std::get<3>(rt_corners_correspondences[i]))
                                {
                                    auto item = scene->addEllipse(-200, -200, 400, 400, QPen(QColor("red"), 100), QBrush(QColor("red")));
                                    item->setPos(corner_transformed_value.x(), corner_transformed_value.y());
                                    items.push_back(item);
                                }

                                // Draw line between ellipses
//                                if(i > 0)
//                                {
//                                    if(i == 3)
//                                    {
//                                        auto line2 = scene->addLine(drawn_corners[0].x(), drawn_corners[0].y(), drawn_corners[i].x(), drawn_corners[i].y(), QPen(QColor("blue"), 100));
//                                        items.push_back(line2);
//                                    }
//                                    auto line = scene->addLine(drawn_corners[i-1].x(), drawn_corners[i-1].y(), drawn_corners[i].x(), drawn_corners[i].y(), QPen(QColor("blue"), 100));
//                                    items.push_back(line);
//
//                                }
                            }
                        }
                    }
                }
            }
        }
    }
    last_corners = target_points;
}
//Ceate funcion to build .g2o string from corner_data, odometry_data, RT matrix
std::string SpecificWorker::build_g2o_graph(const std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> &corner_data, const std::vector<std::vector<float>> &odometry_data, const Eigen::Affine2d robot_pose , const std::vector<Eigen::Vector2d> nominal_corners, const std::vector<Eigen::Vector2f> &room_sizes, std::vector<int> room_size)
{
    std::string g2o_graph;
    int id = 0; // Id for g2o graph vertices
    auto updated_robot_pose = robot_pose;
    
    /// Add nominal corners as VERTEX_XY
    for (size_t i = 0; i < nominal_corners.size(); ++i)
    {
        g2o_graph += "VERTEX_XY " + std::to_string(id) + " " + std::to_string(nominal_corners[i].x()) + " " + std::to_string(nominal_corners[i].y()) + "\n";
        id++;
    }
    
    /// Add first robot pose as VERTEX_SE2
    Eigen::Rotation2Dd rotation2D(robot_pose.rotation());
    double angle = rotation2D.angle();
    g2o_graph += "VERTEX_SE2 " + std::to_string(id) + " " + std::to_string(robot_pose.translation().x()) + " " + std::to_string(robot_pose.translation().y()) + " " + std::to_string(angle) + "\n";
    // Set first robot pose in room as fixed
    g2o_graph += "FIX " + std::to_string(id) + "\n";
    id++;

    /// Store initial ID
    auto node_id = id;

    /// Add new VERTEX_SE2 for each odometry data and EDGE_SE2 from previous vertex
    for (size_t i = 0; i < odometry_data.size()-1; ++i)
    {
        double x_displacement = odometry_data[i][0] * odometry_time_factor;
        double y_displacement = odometry_data[i][1] * odometry_time_factor;
        double angle_displacement = odometry_data[i][2] * odometry_time_factor;
        
        /// Transform odometry data to room frame
        Eigen::Affine2d odometryTransform = Eigen::Translation2d(y_displacement, x_displacement) * Eigen::Rotation2Dd(angle_displacement);
        updated_robot_pose = updated_robot_pose * odometryTransform;
        double angle = Eigen::Rotation2Dd(updated_robot_pose.rotation()).angle();
        
        /// Add noise to odometry data
        g2o_graph += "VERTEX_SE2 " + std::to_string(id) + " " + std::to_string(updated_robot_pose.translation().x()) + " " + std::to_string(updated_robot_pose.translation().y()) + " " + std::to_string(angle) + "\n";
        g2o_graph += "EDGE_SE2 " + std::to_string(id-1) + " " + std::to_string(id) + " " + std::to_string(y_displacement) + " " + std::to_string(x_displacement) + " " + std::to_string(angle_displacement) + " 20 0 0 20 0 1 \n";
        id++;
    }

    std::vector<Eigen::Vector2d> first_corners = { corner_data[0][0].cast<double>(), corner_data[0][1].cast<double>(), corner_data[0][2].cast<double>(), corner_data[0][3].cast<double>() };
    auto first_correspondence = calculate_rooms_correspondences_id(nominal_corners, first_corners, true);

    for(int i = 0; i < 4; i++)
        g2o_graph += "EDGE_SE2_XY " + std::to_string(node_id-1) + " " + std::to_string(std::get<0>(first_correspondence[i])) + " " + std::to_string(std::get<2>(first_correspondence[i]).x()) + " " + std::to_string(std::get<2>(first_correspondence[i]).y()) + " 5 0 5 \n";

    this->aux_corners = std::vector<Eigen::Vector2d> {std::get<2>(first_correspondence[0]), std::get<2>(first_correspondence[1]), std::get<2>(first_correspondence[2]), std::get<2>(first_correspondence[3])};

    qInfo() << "Inserting landmarks in graph #################################3";
    // Add EDGE_SE2_XY landmarks for each position in corner_data (from pose vertex to nominal corner)
    for (size_t i = 1; i < corner_data.size(); ++i)
    {
        if(std::abs(room_sizes[i][0] - room_size[0]) < 50 and std::abs(room_sizes[i][1] - room_size[1]) < 50)
        {
            Eigen::Vector2d corner_data_point_0 = corner_data[i][0].cast<double>();
            Eigen::Vector2d corner_data_point_1 = corner_data[i][1].cast<double>();
            Eigen::Vector2d corner_data_point_2 = corner_data[i][2].cast<double>();
            Eigen::Vector2d corner_data_point_3 = corner_data[i][3].cast<double>();

            std::vector current_corners = {corner_data_point_0, corner_data_point_1, corner_data_point_2,
                                           corner_data_point_3};
            auto correspondences = calculate_rooms_correspondences_id(this->aux_corners, current_corners, true);

            for (size_t j = 0; j < corner_data[i].size(); ++j)
                g2o_graph += "EDGE_SE2_XY " + std::to_string(node_id) + " " +
                             std::to_string(std::get<0>(correspondences[j])) + " " +
                             std::to_string(std::get<2>(correspondences[j]).x()) + " " +
                             std::to_string(std::get<2>(correspondences[j]).y()) + " 5 0 5 \n";
            this->aux_corners = {std::get<2>(correspondences[0]), std::get<2>(correspondences[1]),
                                 std::get<2>(correspondences[2]), std::get<2>(correspondences[3])};
        }
        node_id++;
    }
    return g2o_graph;
}

std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> SpecificWorker::calculate_rooms_correspondences_id(const std::vector<Eigen::Vector2d> &source_points_, std::vector<Eigen::Vector2d> &target_points_, bool first_time)
{
    std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> correspondences;

    /// Generate a matrix to store the distances between source and target points using double vectors instead of Eigen
    std::vector<std::vector<double>> distances_matrix(source_points_.size(), std::vector<double>(target_points_.size()));
    /// Fill the matrix with the distances between source and target points
    for (size_t i = 0; i < source_points_.size(); ++i)
        for (size_t j = 0; j < target_points_.size(); ++j)
            distances_matrix[i][j] = (source_points_[i] - target_points_[j]).norm();
    /// Check if any row or column is empty
    if (distances_matrix.size() == 0 or distances_matrix[0].size() == 0)
    {
        qInfo() << "Empty source or target points";
//        std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> correspondences;
//        for (size_t i = 0; i < source_points_.size(); ++i)
//            correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, source_points_[i], Eigen::Vector2d(0, 0), false));
//        return correspondences;
    }
    /// Process metrics matrix with Hungarian algorithm
    vector<int> assignment;
    double cost = HungAlgo.Solve(distances_matrix, assignment);
    /// Check if every element in assignment is different from -1
    if (std::all_of(assignment.begin(), assignment.end(), [](int i){ return i != -1; }))
        for (unsigned int x = 0; x < assignment.size(); x++)
        {
            /// Check if assignment is valid and the distance is less than the threshold
            qInfo() << "Row " << x << " min distance: " << distances_matrix[x][assignment[x]] << " at column " << assignment[x];
            if (distances_matrix[x][assignment[x]] < corner_matching_threshold)
                correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(x, source_points_[x], target_points_[assignment[x]], true));
            else
                correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(x, source_points_[x], target_points_[assignment[x]], false));
        }

    /// If the assignment is not valid, return the source points
    else
        for (size_t i = 0; i < source_points_.size(); ++i)
            correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, source_points_[i], source_points_[i], false));
    return correspondences;
}

//std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> SpecificWorker::calculate_rooms_correspondences_id(const std::vector<Eigen::Vector2d> &source_points_, std::vector<Eigen::Vector2d> &target_points_, bool first_time)
//{
//
//    std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> correspondences;
//    // Generate dynamic Eigen matrix to hold the distances between source and target points
//    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> distances(source_points_.size(), target_points_.size());
//    // Fill the matrix with the distances between source and target points
//    for (size_t i = 0; i < source_points_.size(); ++i)
//        for (size_t j = 0; j < target_points_.size(); ++j)
//            distances(i, j) = (source_points_[i] - target_points_[j]).norm();
//    // Check if any row or column is empty
//    if (distances.rows() == 0 or distances.cols() == 0)
//    {
//        qInfo() << "Empty source or target points";
//        for (size_t i = 0; i < source_points_.size(); ++i)
//            correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, source_points_[i], Eigen::Vector2d(0, 0), false));
//        return correspondences;
//    }
//
//    // Get the minimun distance in each row
//    Eigen::Index minCol;
//    // Iterate over the rows of the matrix
//    for (size_t i = 0; i < source_points_.size(); ++i)
//    {
//        Eigen::VectorXd min_distances = distances.row(i);
//        double min = min_distances.minCoeff(&minCol);
//        qInfo() << "Row " << i << " min distance: " << min << " at column " << minCol;
//        // Store the correspondences if the distance is less than the threshold
//        if (min < corner_matching_threshold)
//            correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, source_points_[i], target_points_[minCol], true));
//        else
//            correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, source_points_[i], target_points_[minCol], false));
//    }
//    return correspondences;
//}

//std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> SpecificWorker::calculate_rooms_correspondences_id(const std::vector<Eigen::Vector2d> &source_points_, std::vector<Eigen::Vector2d> &target_points_, bool first_time)
//{
//    std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> correspondences;
//
//    // Asociar cada punto de origen con el punto más cercano en el conjunto de puntos objetivo
//    std::vector<Eigen::Vector2d> source_points_copy = source_points_;
//    std::vector<Eigen::Vector2d> target_points_copy = target_points_;
//    int i = 0;
//
//    for (auto source_iter = source_points_copy.begin(); source_iter != source_points_copy.end();)
//    {
//        double min_distance = std::numeric_limits<double>::max();
//        Eigen::Vector2d closest_point;
//        auto target_iter = target_points_copy.begin();
//        auto closest_target_iter = target_iter;
//
//        // Encontrar el punto más cercano en el conjunto de puntos objetivo
//        while (target_iter != target_points_copy.end())
//        {
//            double d = (*source_iter - *target_iter).norm();
//            if (d < min_distance)
//            {
//                min_distance = d;
//                closest_point = *target_iter;
//                closest_target_iter = target_iter;
//            }
//            ++target_iter;
//        }
//
//        // Almacenar la correspondencia encontrada
//        correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, *source_iter, closest_point, true));
//        i++;
//        // Eliminar los puntos correspondientes de sus vectores originales
//        source_iter = source_points_copy.erase(source_iter);
//        target_points_copy.erase(closest_target_iter);
//    }
//    if(not first_time)
//        for (auto &correspondence: correspondences)
//        {
//            if ((std::get<1>(correspondence) - std::get<2>(correspondence)).norm() > corner_matching_threshold)
//            {
//                std::get<3>(correspondence) = false;
//                qInfo() << "Corner " << std::get<0>(correspondence) << " is too far away from the target";
//                qInfo() << "Last corner" << std::get<1>(correspondence).x() << " " << std::get<1>(correspondence).y();
//                qInfo() << "Target corner" << std::get<2>(correspondence).x() << " " << std::get<2>(correspondence).y();
//            }
//        }
//    return correspondences;
//}
std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> SpecificWorker::calculate_rooms_correspondences(const std::vector<Eigen::Vector2d> &source_points_, const std::vector<Eigen::Vector2d> &target_points_)
{
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;
    // Asociar cada punto de origen con el punto más cercano en el conjunto de puntos objetivo
    std::vector<Eigen::Vector2d> source_points_copy = source_points_;
    std::vector<Eigen::Vector2d> target_points_copy = target_points_;

    for (auto source_iter = source_points_copy.begin(); source_iter != source_points_copy.end();)
    {
        double min_distance = std::numeric_limits<double>::max();
        Eigen::Vector2d closest_point;
        auto target_iter = target_points_copy.begin();
        auto closest_target_iter = target_iter;

        // Encontrar el punto más cercano en el conjunto de puntos objetivo
        while (target_iter != target_points_copy.end())
        {
            double d = (*source_iter - *target_iter).norm();
            if (d < min_distance)
            {
                min_distance = d;
                closest_point = *target_iter;
                closest_target_iter = target_iter;
            }
            ++target_iter;
        }

        // Almacenar la correspondencia encontrada
        correspondences.push_back({*source_iter, closest_point});

        // Eliminar los puntos correspondientes de sus vectores originales
        source_iter = source_points_copy.erase(source_iter);
        target_points_copy.erase(closest_target_iter);
    }
    return correspondences;
}
void SpecificWorker::create_wall(int id, const std::vector<float> &p, float angle, DSR::Node parent_node, bool nominal)
{
    auto parent_level_ = G->get_node_level(parent_node);
    if(not parent_level_.has_value())
    { qWarning() << __FUNCTION__ << " No parent level in graph"; return; }
    auto parent_level = parent_level_.value();

    std::string wall_name = "wall_" + std::to_string(id);
    if(not nominal)
    {
        wall_name = "wall_" + std::to_string(id) + "_measured";
    }

    auto new_wall = DSR::Node::create<wall_node_type>(wall_name);

    //Get room node pos_x and pos_y attributes
    auto room_node_ = G->get_node("room");

    if(not room_node_.has_value())
    {
        qWarning() << __FUNCTION__ << " No room node in graph"; return;
    }
    else
    {

        auto room_node = room_node_.value();
        auto pos_x = G->get_attrib_by_name<pos_x_att>(room_node);
        auto pos_y = G->get_attrib_by_name<pos_y_att>(room_node);
        //Set corner pos_x and pos_y attributes as corners of square room centered in pos_x and pos_y
        if(pos_x.has_value() and pos_y.has_value()){
            G->add_or_modify_attrib_local<pos_x_att>(new_wall, pos_x.value() + p[0]/ 25);
            G->add_or_modify_attrib_local<pos_y_att>(new_wall, pos_y.value() + p[1]/ 25);
        }
        else{
            qWarning() << __FUNCTION__ << " No pos_x or pos_y attributes in room node";
            return;
        }
    }

    G->add_or_modify_attrib_local<obj_id_att>(new_wall, id);
    G->add_or_modify_attrib_local<timestamp_creation_att>(new_wall, get_actual_time());
    G->add_or_modify_attrib_local<timestamp_alivetime_att>(new_wall, get_actual_time());
    G->add_or_modify_attrib_local<level_att>(new_wall, parent_level + 1);
    G->insert_node(new_wall);

    rt->insert_or_assign_edge_RT(parent_node, new_wall.id(), p, {0.0, 0.0, angle});
}
void SpecificWorker::create_corner(int id, const std::vector<float> &p, DSR::Node parent_node, bool nominal)
{
    auto parent_level_ = G->get_node_level(parent_node);
    if(not parent_level_.has_value())
    { qWarning() << __FUNCTION__ << " No parent level in graph"; return; }
    auto parent_level = parent_level_.value();
    std::string corner_name = "corner_" + std::to_string(id);
    if(not nominal)
    {
        corner_name = "corner_" + std::to_string(id) + "_measured";
    }
    auto new_corner = DSR::Node::create<corner_node_type>(corner_name);

    //Get room node pos_x and pos_y attributes
    auto room_node_ = G->get_node("room");

    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room node in graph"; return; }
    else
    {
        auto room_node = room_node_.value();

        auto pos_x = G->get_attrib_by_name<pos_x_att>(parent_node);
        auto pos_y = G->get_attrib_by_name<pos_y_att>(parent_node);
        //Set corner pos_x and pos_y attributes as corners of square room centered in pos_x and pos_y
        if(pos_x.has_value() and pos_y.has_value())
        {
            if (nominal) //Set nominal corner pose
            {
                G->add_or_modify_attrib_local<pos_x_att>(new_corner, pos_x.value() + 30);
                G->add_or_modify_attrib_local<pos_y_att>(new_corner, pos_y.value() + 30);
            }
            else //Set measured Corner
            {
                G->add_or_modify_attrib_local<pos_x_att>(new_corner, pos_x.value() - 160);
                G->add_or_modify_attrib_local<pos_y_att>(new_corner, pos_y.value() - 60 + id * 40);
            }
        }
        else{
            qWarning() << __FUNCTION__ << " No pos_x or pos_y attributes in room node";
            return;
        }
    }

    G->add_or_modify_attrib_local<corner_id_att>(new_corner, id);
    G->add_or_modify_attrib_local<timestamp_creation_att>(new_corner, get_actual_time());
    G->add_or_modify_attrib_local<timestamp_alivetime_att>(new_corner, get_actual_time());
    G->add_or_modify_attrib_local<level_att>(new_corner, parent_level + 1);
    G->insert_node(new_corner);

    rt->insert_or_assign_edge_RT(parent_node, new_corner.id(), p, {0.0, 0.0, 0.0});
}
void SpecificWorker::check_room_orientation()   // TODO: experimental
{
    // admissibility conditions
    auto room_nodes = G->get_nodes_by_type("room");
    if(room_nodes.empty()) return;
    auto room_node = room_nodes.front();

    // get list of objects
    auto object_nodes = G->get_nodes_by_type("chair");  //TODO: should be all
    if(object_nodes.empty()) { qWarning() << __FUNCTION__ << "No objects in graph" ; return;}

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    // if oriented return
    auto is_oriented_ = G->get_attrib_by_name<room_is_oriented_att>(room_node);
    if(is_oriented_.has_value() and is_oriented_.value()) return;

    // if not oriented get room width and depth and position wrt to robot
    auto room_width = G->get_attrib_by_name<width_att>(room_node);
    auto room_depth = G->get_attrib_by_name<depth_att>(room_node);
    if(not room_width.has_value() or not room_depth.has_value()) return;
    auto edge_robot_to_room = rt->get_edge_RT(robot_node.value(), room_node.id());
    if( not edge_robot_to_room.has_value()) { qWarning() << __FUNCTION__ << "No edge between robot and room"; return;}
    auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot_to_room.value());
    auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot_to_room.value());
    if (not tr.has_value() or not rot.has_value()) { qWarning() << __FUNCTION__ << "No translation or rotation between robot and room"; return;}
    auto x = tr.value().get()[0], y = tr.value().get()[1], ang = rot.value().get()[2];

    // now build and Eigen::Transform to encode the room position and orientation wrt to the robot
    Eigen::Transform<float, 2, Eigen::Affine> room_transform;
    room_transform.setIdentity();
    room_transform.translate(Eigen::Vector2f(x, y));
    room_transform.rotate(Eigen::Rotation2Df(ang));

    // transform the objects' coordinates to the room frame
    for(const auto &object_node : object_nodes)
    {
        if (auto edge_robot = rt->get_edge_RT(robot_node.value(), object_node.id()); edge_robot.has_value())
        {
            auto tr_obj = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
            auto rot_obj = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
            if (tr_obj.has_value() and rot_obj.has_value() /*and width_.has_value() and depth_.has_value()*/)
            {
                // now transform the object to the room frame
                auto obj_room = room_transform * Eigen::Vector2f(tr.value().get()[0], tr.value().get()[1]);
                // check if obj_room is very close to the room border
                if( is_on_a_wall(obj_room.x(), obj_room.y(), room_width.value(), room_depth.value()))
                {
                    // set the room as oriented towards the object. To do this we need two things:
                    // 1. add an edge in G between the room and the object
                    // 2. reset the odometry but setting the offset of the current transform between the robot and the room,
                    // so the next current pose sent by the lidar_odometry is the current pose of the robot in the room frame

                }
            }
        }
    }
}
bool SpecificWorker::is_on_a_wall(float x, float y, float width, float depth)
{
    auto is_approx = [](float a, float b){ return fabs(a - b) < 0.1; };

    if(((is_approx(fabs(x), 0.f) and is_approx(fabs(y), depth/2.f))) or (is_approx(fabs(y), 0.f) and is_approx(fabs(x), width/2.f)))
        return true;
    else return false;
}
uint64_t SpecificWorker::get_actual_time()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
// function to read robot odometry from graph and update the robot pose
std::vector<float> SpecificWorker::get_graph_odometry()
{
    static bool initialize = false;

    //TODO: Fix last_time initialization
    if (not initialize)
    {
        last_time = std::chrono::high_resolution_clock::now();
        initialize = true;
    }

    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    {
        qWarning() << __FUNCTION__ << " No robot node in graph. Can't get odometry data.";
        return{};
    }
    auto robot_node = robot_node_.value();
    if(auto adv_odom_att = G->get_attrib_by_name<robot_current_advance_speed_att>(robot_node); adv_odom_att.has_value())
        if(auto side_odom_att = G->get_attrib_by_name<robot_current_side_speed_att>(robot_node); side_odom_att.has_value())
            if(auto rot_odom_att = G->get_attrib_by_name<robot_current_angular_speed_att>(robot_node); rot_odom_att.has_value())
            {
                auto adv_odom = adv_odom_att.value();
                auto side_odom = side_odom_att.value();
                auto rot_odom = rot_odom_att.value();
                // Get difference time between last and current time
                auto now = std::chrono::high_resolution_clock::now();
                auto diff_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
                last_time = now;
                // Get robot new pose considering speeds and time
                float adv_disp = adv_odom * diff_time;
                float side_disp = side_odom * diff_time;
                float rot_disp = rot_odom * diff_time / 1000;

                // return the new robot odometry
                return {adv_disp, side_disp, rot_disp};
            }
    return {0, 0, 0};
}

/////////////////// Draw  /////////////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const RoboCompLidar3D::TData &data, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &p: data.points)
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20), QBrush(QColor("green")));
        item->setPos(p.x, p.y);
        items.push_back(item);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}




/**************************************/
// From the RoboCompG2Ooptimizer you can call this methods:
// this->g2ooptimizer_proxy->optimize(...)

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






//void SpecificWorker::update_room_data(const rc::Room &room)
//{
//    auto root_node_ = G->get_node("root");
//    if(not root_node_.has_value())
//    { qWarning() << __FUNCTION__ << " No root node in graph"; return; }
//    auto root_node = root_node_.value();
//
//    auto robot_node_ = G->get_node("Shadow");
//    if(not robot_node_.has_value())
//    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
//    auto robot_node = robot_node_.value();
//
//    auto robot_level_ = G->get_node_level(robot_node);
//    if(not robot_level_.has_value())
//    { qWarning() << __FUNCTION__ << " No robot level in graph"; return; }
//    auto robot_level = robot_level_.value();
//
//    // match phase
//    auto now = std::chrono::high_resolution_clock::now();
//
//    // Get room corners
//    auto target_points_ = room.get_corners();
//    // Cast corners to Eigen::Vector2d through a lambda
//    std::vector<Eigen::Vector2d> target_points;
//    std::transform(target_points_.begin(), target_points_.end(), std::back_inserter(target_points),
//                   [](const auto &p){ return Eigen::Vector2d(p.x(), p.y()); });
//
//    // Match phase. Check if the room is already in graph G
//    if(auto room_nodes = G->get_nodes_by_type("room"); not room_nodes.empty())
//    {
//
//        std::cout << "///////////////////////////////////////////////////THERE IS ROOM////////////////////////////////////////////////////////" << std::endl;
//
//        auto room_node = room_nodes.front();
//        auto node_insertion_time = G->get_attrib_by_name<timestamp_creation_att>(room_node);
//        auto room_checked = G->get_attrib_by_name<obj_checked_att>(room_node);
//        // Check if room was checked before. If not, check it.
//        if ( room_checked.has_value() and
//             not room_checked.value() and
//             node_insertion_time.has_value() and
//             ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() -
//               node_insertion_time.value()) > params.SECS_TO_GET_IN))
//        {
//            G->add_or_modify_attrib_local<obj_checked_att>(room_node, true);
//        }
//        // update room parameters for the stabilized room and the waiting room
//        G->add_or_modify_attrib_local<timestamp_alivetime_att>(room_node, get_actual_time());
//        G->update_node(room_node);
//
//        // Get nominal corners data transforming to robot frame and measured corners
//        if (auto edge_robot_ = rt->get_edge_RT(room_node, robot_node.id()); edge_robot_.has_value()) {
//            //get the rt values from edge in of corner measured nodes and insert the in a std::vector
//            std::vector<Eigen::Vector2d> rt_corner_values;
//            std::optional<DSR::Node> corner_aux_, wall_aux_, corner_measured_aux_;
//            //dest org
//            for (int i = 1; i < 5; i++) {
//                corner_aux_ = G->get_node("corner_" + std::to_string(i));
//                if (not corner_aux_.has_value()) {
//                    qWarning() << __FUNCTION__ << " No nominal corner " << i << " in graph";
//                    return;
//                }
//                auto corner_aux = corner_aux_.value();
//
//                wall_aux_ = G->get_node("wall_" + std::to_string(i));
//                if (not wall_aux_.has_value()) {
//                    qWarning() << __FUNCTION__ << " No wall " << i << " in graph";
//                    return;
//                }
//                auto wall_aux = wall_aux_.value();
//
//                corner_measured_aux_ = G->get_node("corner_" + std::to_string(i) + "_measured");
//                if (not corner_measured_aux_.has_value()) {
//                    qWarning() << __FUNCTION__ << " No measured corner " << i << " in graph";
//                    return;
//                }
//                auto corner_measured_aux = corner_measured_aux_.value();
//
//                if (auto rt_corner_edge = rt->get_edge_RT(wall_aux, corner_aux.id()); rt_corner_edge.has_value())
//                    if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(
//                                rt_corner_edge.value()); rt_translation.has_value()) {
//                        auto rt_translation_value = rt_translation.value().get();
//                        // Get robot pose
//                        if (auto rt_robot_edge = rt->get_edge_RT(room_node, robot_node.id()); rt_robot_edge.has_value())
//                            if (auto rt_translation_robot = G->get_attrib_by_name<rt_translation_att>(
//                                        rt_robot_edge.value()); rt_translation_robot.has_value()) {
//                                auto rt_translation_robot_value = rt_translation_robot.value().get();
//                                if (auto rt_rotation_robot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(
//                                            rt_robot_edge.value()); rt_rotation_robot.has_value()) {
//                                    auto rt_rotation_robot_value = rt_rotation_robot.value().get();
//                                    qInfo() << "Robot pose: " << rt_translation_robot_value[0] << " "
//                                            << rt_translation_robot_value[1] << " " << rt_rotation_robot_value[2];
//                                    // Transform nominal corner position to robot frame
//                                    Eigen::Vector3f corner_robot_pos_point(rt_translation_value[0],
//                                                                           rt_translation_value[1], 0.f);
//                                    auto corner_robot_pos_point_double = corner_robot_pos_point.cast<double>();
//                                    if (auto corner_transformed = inner_eigen->transform(robot_node.name(),
//                                                                                         corner_robot_pos_point_double,
//                                                                                         wall_aux.name()); corner_transformed.has_value()) {
//                                        auto corner_transformed_value = corner_transformed.value();
//                                        // Print corner
//                                        std::cout << "Corner transformed with transform" << i << ": "
//                                                  << corner_transformed_value.x() << " " << corner_transformed_value.y()
//                                                  << std::endl;
//                                        if (auto rt_corner_edge_measured = rt->get_edge_RT(robot_node,
//                                                                                           corner_measured_aux.id()); rt_corner_edge_measured.has_value())
//                                            if (auto rt_translation_measured = G->get_attrib_by_name<rt_translation_att>(
//                                                        rt_corner_edge_measured.value()); rt_translation_measured.has_value()) {
//                                                auto rt_corner_meauserd_value = rt_translation_measured.value().get();
//                                                // Print measured corner
//                                                std::cout << "Measured corner " << i << ": "
//                                                          << rt_corner_meauserd_value[0] << " "
//                                                          << rt_corner_meauserd_value[1] << std::endl;
//                                                // Calculate difference between transformed and measured corner
//                                                auto diff = Eigen::Vector3f(corner_transformed_value.x(),
//                                                                            corner_transformed_value.y(), 0.f) -
//                                                            Eigen::Vector3f(rt_corner_meauserd_value[0],
//                                                                            rt_corner_meauserd_value[1], 0.f);
//                                                // Print difference
//                                                std::cout << "Difference " << i << ": " << diff.x() << " " << diff.y()
//                                                          << std::endl;
//                                                rt_corner_values.push_back(
//                                                        {corner_transformed_value.x(), corner_transformed_value.y()});
//                                            }
//                                    }
//                                }
//                            }
//                    }
//            }
//            // Calculate correspondences between transformed nominal and measured corners
//            auto rt_corners_correspondences = calculate_rooms_correspondences_id(rt_corner_values, target_points);
//
//            for (int i = 1; i < 5; i++)
//            {
//                std::string corner_name = "corner_" + std::to_string(i) + "_measured";
//                if (std::optional<DSR::Node> updated_corner = G->get_node(corner_name); updated_corner.has_value())
//                    if (std::optional<DSR::Edge> edge = G->get_edge(robot_node.id(), updated_corner.value().id(),
//                                                                    "RT"); edge.has_value()) {
//                        if (auto corner_id = G->get_attrib_by_name<corner_id_att>(
//                                    updated_corner.value()); corner_id.has_value()) {
//                            //                                std::cout << "Corner_id: " << corner_id.value() << std::endl;
//                            //                                std::cout << "Corner id correspondence: " << std::get<0>(rt_corners_correspondences[i-1]) << std::endl;
//                            if (corner_id.value() == std::get<0>(rt_corners_correspondences[i - 1]))
//                            {
//                                //insert the rt values in the edge
//                                G->add_or_modify_attrib_local<valid_att>(updated_corner.value(), std::get<3>(
//                                        rt_corners_correspondences[i - 1]));
//                                G->update_node(updated_corner.value());
//                                G->add_or_modify_attrib_local<rt_translation_att>(edge.value(), std::vector<float>{
//                                        (float) std::get<2>(rt_corners_correspondences[i - 1]).x(),
//                                        (float) std::get<2>(rt_corners_correspondences[i - 1]).y(), 0.0f});
//                                G->insert_or_assign_edge(edge.value());
//                            }
//                        }
//                    }
//            }
//        }
//    }
//    else  // Insert phase. No room in graph. Insert room the first time
//    {
//        try
//        {
//            auto width = static_cast<int>(room.get_width());
//            auto height = static_cast<int>(room.get_height());
//            auto depth = static_cast<int>(room.get_depth());
//
//            DSR::Node room_node = DSR::Node::create<room_node_type>("room");
////            G->add_or_modify_attrib_local<person_id_att>(room_node, 0);  // TODO: current_room
//            G->add_or_modify_attrib_local<width_att>(room_node, width);
//            G->add_or_modify_attrib_local<height_att>(room_node, height);
//            G->add_or_modify_attrib_local<depth_att>(room_node, depth);
//
//            //give me the 4 corners of the room in the 0,0 coordinate system given witdh and depth
//            std::vector<Eigen::Vector2d> imaginary_room_1 = {{ -width / 2, depth / 2}, { width / 2,  depth / 2}, { -width / 2,  -depth / 2}, { width / 2,  -depth / 2}};
//            std::vector<Eigen::Vector2d> imaginary_room_2 = {{ -depth / 2, width / 2}, { depth / 2,  width / 2}, { -depth / 2,  -width / 2}, { depth / 2,  -width /2}};
//            //draw imaginary room
////                static std::vector<QGraphicsItem *> draw_points;
////                for (const auto &p: imaginary_room_1)
////                {
////                    auto o = widget_2d->scene.addRect(-20, 20, 200, 200, QPen(Qt::black), QBrush(Qt::black));
////                    o->setPos(p.x(), p.y());
////                    draw_points.push_back(o);
////                }
////
////                for (const auto &p: imaginary_room_2)
////                {
////                    auto o = widget_2d->scene.addRect(-20, 20, 200, 200, QPen(Qt::yellow), QBrush(Qt::yellow));
////                    o->setPos(p.x(), p.y());
////                    draw_points.push_back(o);
////                }
//
////            std::vector<Eigen::Vector2d> first_room;
////            for(int i = 1; i < 5; i++)
////            {
////                ss.clear();
////                ss.str(room_->attributes.at("corner" + std::to_string(i)));
////                float float1, float2; char coma;
////                ss >> float1 >> coma >> float2;
////                first_room.push_back(Eigen::Vector2d(float1, float2));
////            }
//
//            // Get room corners
//            auto first_room_ = room.get_corners();
//            // Cast corners to Eigen::Vector2d
//            std::vector<Eigen::Vector2d> first_room;
//            for (const auto &p: first_room_)
//            {
//                first_room.push_back(Eigen::Vector2d(p.x(), p.y()));
//            }
//            // TODO: Check if they are removed and if the corners are set with the values from the forcefield or the
//            // calculated by the dimensions of the room in the rooms reference system
//            // G->add_or_modify_attrib_local<corner1_att>(room_node, std::vector<float>{static_cast<float>(first_room[0].x()), static_cast<float>(first_room[0].y())});
//            // G->add_or_modify_attrib_local<corner2_att>(room_node, std::vector<float>{static_cast<float>(first_room[1].x()), static_cast<float>(first_room[1].y())});
//            // G->add_or_modify_attrib_local<corner3_att>(room_node, std::vector<float>{static_cast<float>(first_room[2].x()), static_cast<float>(first_room[2].y())});
//            // G->add_or_modify_attrib_local<corner4_att>(room_node, std::vector<float>{static_cast<float>(first_room[3].x()), static_cast<float>(first_room[3].y())});
//
//            auto correspondence_1 = this->calculate_rooms_correspondences(first_room, imaginary_room_1);
//            auto correspondence_2 = this->calculate_rooms_correspondences(first_room, imaginary_room_2);
//
//            // Calcular la suma de las distancias cuadradas para cada par de puntos
//            double sum_sq_dist1 = 0.0;
//            double sum_sq_dist2 = 0.0;
//            for (size_t i = 0; i < correspondence_1.size(); ++i)
//            {
//                sum_sq_dist1 += (correspondence_1[i].first - correspondence_1[i].second).squaredNorm();
//                sum_sq_dist2 += (correspondence_2[i].first - correspondence_2[i].second).squaredNorm();
//            }
//
//            std::vector<Eigen::Vector2d> imaginary_room;
//            if(sum_sq_dist1 < sum_sq_dist2)
//                imaginary_room = imaginary_room_1;
//            else
//                imaginary_room = imaginary_room_2;
//
//            icp icp(imaginary_room, first_room);
//            icp.align();
//
//            Eigen::Matrix2d transformation = icp.rotation();
//            Eigen::Vector2d translation = icp.translation();
//
//            float angle_rad = std::atan2(transformation(1,0), transformation(0,0));
//
//            // Convert angle from radians to degrees
//            G->add_or_modify_attrib_local<pos_x_att>(room_node, (float)(rand()%(170)));
//            G->add_or_modify_attrib_local<pos_y_att>(room_node, (float)(rand()%170));
//            G->add_or_modify_attrib_local<obj_checked_att>(room_node, false);
//            G->add_or_modify_attrib_local<level_att>(room_node, robot_level);
//            G->insert_node(room_node);
//
//            std::cout << "///////////////////////////////////////////////////Room node inserted////////////////////////////////////////////////////////" << std::endl;
//
//            std::vector<float> orientation_vector = { 0.0, 0.0, 0.0 };
//            std::vector<float> room_pos = { 0.f, 0.f, 0.f };
//
//            //Insert angle rad in Z? Between root and room?
//            orientation_vector = { 0.0f, 0.0f, 0.0f };
//            DSR::Node root = G->get_node(100).value();
//            rt->insert_or_assign_edge_RT(root, room_node.id(), room_pos, orientation_vector);
//            G->delete_edge(100, robot_node.id(), "RT");
//
//            // Increase robot node level
//            G->add_or_modify_attrib_local<level_att>(robot_node, robot_level + 1);
//            G->update_node(robot_node);
//
//            Eigen::Matrix3f rt_rotation_matrix;
//            rt_rotation_matrix << cos(angle_rad), -sin(angle_rad), translation.x(),
//                    sin(angle_rad), cos(angle_rad), translation.y(),
//                    0.0, 0.0, 1.0;
//
//            Eigen::Matrix3f rt_rotation_matrix_inv = rt_rotation_matrix.inverse();
//            std::vector<float> robot_pos = { rt_rotation_matrix_inv(0,2), rt_rotation_matrix_inv(1,2), 0.f };
//            rt->insert_or_assign_edge_RT(room_node, robot_node.id(), robot_pos, { 0.f, 0.f, -angle_rad });
//
////            std::vector<Eigen::Vector2d> imaginary_room_trans;
//            std::swap(imaginary_room[2], imaginary_room[3]);
//
//            std::vector<float> corner_pos, corner_measured_pos, wall_pos;
//            float wall_angle = 0.0;
//            DSR::Node new_corner, new_corner_measured;
//
//            auto first_correspondence = calculate_rooms_correspondences_id(imaginary_room, target_points, true);
//
//            for (int i = 1; i < 5; i++)
//            {
//                // Generate switch case for asigning angles and points depending on the wall/corner id
//                switch(i) {
//                    case 1:
//                        corner_pos = {-abs((float)std::get<1>(first_correspondence[i-1]).x()), 0.0, 0.0};
//                        wall_pos = {0.0, (float)std::get<1>(first_correspondence[i-1]).y(), 0.0};
//                        break;
//                    case 2:
//                        corner_pos = {-abs((float)std::get<1>(first_correspondence[i-1]).y()), 0.0, 0.0};
//                        wall_pos = {(float)std::get<1>(first_correspondence[i-1]).x(), 0.0, 0.0};
//                        wall_angle = -M_PI_2;
//                        break;
//                    case 3:
//                        corner_pos = {-abs((float) std::get<1>(first_correspondence[i - 1]).x()), 0.0, 0.0};
//                        wall_pos = {0.0, (float)std::get<1>(first_correspondence[i-1]).y(), 0.0};
//                        wall_angle = M_PI;
//                        break;
//                    case 4:
//                        corner_pos = {-abs((float)std::get<1>(first_correspondence[i-1]).y()), 0.0, 0.0};
//                        wall_pos = {(float)std::get<1>(first_correspondence[i-1]).x(), 0.0, 0.0};
//                        wall_angle = M_PI_2;
//                        break;
//                    default:
//                        cout << "No valid option." << endl;
//                        break;
//                }
//                // insert nominal values
//                create_wall(i, wall_pos, wall_angle, room_node);
//                if(auto wall_node_ = G->get_node("wall_" + std::to_string(i)); wall_node_.has_value())
//                {
//                    auto wall_node = wall_node_.value();
//                    create_corner(i, corner_pos, wall_node);
//                }
//                create_corner(i, {(float)std::get<2>(first_correspondence[i-1]).x(), (float)std::get<2>(first_correspondence[i-1]).y(), 0.0}, robot_node, false);
//            }
//        }
//        catch(const std::exception &e)
//        { std::cout << e.what() << " Error inserting node" << std::endl;}
////        qInfo() << __FUNCTION__ << " Object with id: " << room_->id << " inserted in graph";
//    }
//    // no remove of rooms for now
//}
