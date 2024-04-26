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

        // timers
        Period = params.PERIOD;    //  used in the lidar reader thread
        timer.start(Period);
        std::cout << "Worker initialized OK" << std::endl;
	}
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value()) { qWarning() << "No lidar data available"; return; }
    auto ldata = res_.value();

    // draw lidar
    if(widget_2d != nullptr)
        draw_lidar(ldata, &widget_2d->scene);

    /// extract lines from LiDAR data
    Lines lines = extract_2D_lines_from_lidar3D(ldata.points, params.ranges_list);

    /// Room detector
    auto current_room = room_detector.detect({lines[0]}, &widget_2d->scene, true);  // TODO: use upper lines in Helios
    current_room.draw_on_2D_tab(current_room, "yellow", &widget_2d->scene);

    process_room(current_room);

    // Check if there is a room and it is oriented
    check_room_orientation();

    graph_viewer->set_external_hz((int)(1.f/fps.get_period()*1000.f));
    fps.print("room_detector");

}

////////////////////////////////////////////////////////////////////////////////
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
void SpecificWorker::process_room(const rc::Room &room)
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

    // match phase
    auto now = std::chrono::high_resolution_clock::now();
    
    // Get room corners
    auto target_points_ = room.get_corners();
    // Cast corners to Eigen::Vector2d through a lambda
    std::vector<Eigen::Vector2d> target_points;
    std::transform(target_points_.begin(), target_points_.end(), std::back_inserter(target_points),
                   [](const auto &p){ return Eigen::Vector2d(p.x(), p.y()); });

    // Match phase. Check if the room is already in graph G
    if(auto room_nodes = G->get_nodes_by_type("room"); not room_nodes.empty())
    {
        auto room_node = room_nodes.front();
        auto node_insertion_time = G->get_attrib_by_name<timestamp_creation_att>(room_node);
        auto room_checked = G->get_attrib_by_name<obj_checked_att>(room_node);
        // Check if room was checked before. If not, check it.
        if ( room_checked.has_value() and
             not room_checked.value() and
             node_insertion_time.has_value() and
             ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() -
               node_insertion_time.value()) > params.SECS_TO_GET_IN))
        {
            G->add_or_modify_attrib_local<obj_checked_att>(room_node, true);
        }
        // update room parameters for the stabilized room and the waiting room
        G->add_or_modify_attrib_local<timestamp_alivetime_att>(room_node, get_actual_time());
        G->update_node(room_node);
        
        // Get nominal corners data transforming to robot frame and measured corners
        if (auto edge_robot_ = rt->get_edge_RT(room_node, robot_node.id()); edge_robot_.has_value()) {
            //get the rt values from edge in of corner measured nodes and insert the in a std::vector
            std::vector<Eigen::Vector2d> rt_corner_values;
            std::optional<DSR::Node> corner_aux_, wall_aux_, corner_measured_aux_;
            //dest org
            for (int i = 1; i < 5; i++) {
                corner_aux_ = G->get_node("corner_" + std::to_string(i));
                if (not corner_aux_.has_value()) {
                    qWarning() << __FUNCTION__ << " No nominal corner " << i << " in graph";
                    return;
                }
                auto corner_aux = corner_aux_.value();

                wall_aux_ = G->get_node("wall_" + std::to_string(i));
                if (not wall_aux_.has_value()) {
                    qWarning() << __FUNCTION__ << " No wall " << i << " in graph";
                    return;
                }
                auto wall_aux = wall_aux_.value();

                corner_measured_aux_ = G->get_node("corner_" + std::to_string(i) + "_measured");
                if (not corner_measured_aux_.has_value()) {
                    qWarning() << __FUNCTION__ << " No measured corner " << i << " in graph";
                    return;
                }
                auto corner_measured_aux = corner_measured_aux_.value();

                if (auto rt_corner_edge = rt->get_edge_RT(wall_aux, corner_aux.id()); rt_corner_edge.has_value())
                    if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(
                                rt_corner_edge.value()); rt_translation.has_value()) {
                        auto rt_translation_value = rt_translation.value().get();
                        // Get robot pose
                        if (auto rt_robot_edge = rt->get_edge_RT(room_node, robot_node.id()); rt_robot_edge.has_value())
                            if (auto rt_translation_robot = G->get_attrib_by_name<rt_translation_att>(
                                        rt_robot_edge.value()); rt_translation_robot.has_value()) {
                                auto rt_translation_robot_value = rt_translation_robot.value().get();
                                if (auto rt_rotation_robot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(
                                            rt_robot_edge.value()); rt_rotation_robot.has_value()) {
                                    auto rt_rotation_robot_value = rt_rotation_robot.value().get();
                                    qInfo() << "Robot pose: " << rt_translation_robot_value[0] << " "
                                            << rt_translation_robot_value[1] << " " << rt_rotation_robot_value[2];
                                    // Transform nominal corner position to robot frame
                                    Eigen::Vector3f corner_robot_pos_point(rt_translation_value[0],
                                                                           rt_translation_value[1], 0.f);
                                    auto corner_robot_pos_point_double = corner_robot_pos_point.cast<double>();
                                    if (auto corner_transformed = inner_eigen->transform(robot_node.name(),
                                                                                         corner_robot_pos_point_double,
                                                                                         wall_aux.name()); corner_transformed.has_value()) {
                                        auto corner_transformed_value = corner_transformed.value();
                                        // Print corner
                                        std::cout << "Corner transformed with transform" << i << ": "
                                                  << corner_transformed_value.x() << " " << corner_transformed_value.y()
                                                  << std::endl;
                                        if (auto rt_corner_edge_measured = rt->get_edge_RT(robot_node,
                                                                                           corner_measured_aux.id()); rt_corner_edge_measured.has_value())
                                            if (auto rt_translation_measured = G->get_attrib_by_name<rt_translation_att>(
                                                        rt_corner_edge_measured.value()); rt_translation_measured.has_value()) {
                                                auto rt_corner_meauserd_value = rt_translation_measured.value().get();
                                                // Print measured corner
                                                std::cout << "Measured corner " << i << ": "
                                                          << rt_corner_meauserd_value[0] << " "
                                                          << rt_corner_meauserd_value[1] << std::endl;
                                                // Calculate difference between transformed and measured corner
                                                auto diff = Eigen::Vector3f(corner_transformed_value.x(),
                                                                            corner_transformed_value.y(), 0.f) -
                                                            Eigen::Vector3f(rt_corner_meauserd_value[0],
                                                                            rt_corner_meauserd_value[1], 0.f);
                                                // Print difference
                                                std::cout << "Difference " << i << ": " << diff.x() << " " << diff.y()
                                                          << std::endl;
                                                rt_corner_values.push_back(
                                                        {corner_transformed_value.x(), corner_transformed_value.y()});
                                            }
                                    }
                                }
                            }
                    }
            }
            // Calculate correspondences between transformed nominal and measured corners
            auto rt_corners_correspondences = calculate_rooms_correspondences_id(rt_corner_values, target_points);

            for (int i = 1; i < 5; i++) {
                std::string corner_name = "corner_" + std::to_string(i) + "_measured";
                if (std::optional<DSR::Node> updated_corner = G->get_node(corner_name); updated_corner.has_value())
                    if (std::optional<DSR::Edge> edge = G->get_edge(robot_node.id(), updated_corner.value().id(),
                                                                    "RT"); edge.has_value()) {
                        if (auto corner_id = G->get_attrib_by_name<corner_id_att>(
                                    updated_corner.value()); corner_id.has_value()) {
                            //                                std::cout << "Corner_id: " << corner_id.value() << std::endl;
                            //                                std::cout << "Corner id correspondence: " << std::get<0>(rt_corners_correspondences[i-1]) << std::endl;
                            if (corner_id.value() == std::get<0>(rt_corners_correspondences[i - 1])) {
                                //insert the rt values in the edge
                                G->add_or_modify_attrib_local<valid_att>(updated_corner.value(), std::get<3>(
                                        rt_corners_correspondences[i - 1]));
                                G->update_node(updated_corner.value());
                                G->add_or_modify_attrib_local<rt_translation_att>(edge.value(), std::vector<float>{
                                        (float) std::get<2>(rt_corners_correspondences[i - 1]).x(),
                                        (float) std::get<2>(rt_corners_correspondences[i - 1]).y(), 0.0f});
                                G->insert_or_assign_edge(edge.value());
                            }
                        }
                    }
            }
        }
    }
    else  // Insert phase. No room in graph. Insert room the first time
    {
        try
        {
            auto width = static_cast<int>(room.get_width());
            auto height = static_cast<int>(room.get_height());
            auto depth = static_cast<int>(room.get_depth());

            DSR::Node room_node = DSR::Node::create<room_node_type>("room");
//            G->add_or_modify_attrib_local<person_id_att>(room_node, 0);  // TODO: current_room
            G->add_or_modify_attrib_local<width_att>(room_node, width);
            G->add_or_modify_attrib_local<height_att>(room_node, height);
            G->add_or_modify_attrib_local<depth_att>(room_node, depth);

            //give me the 4 corners of the room in the 0,0 coordinate system given witdh and depth
            std::vector<Eigen::Vector2d> imaginary_room_1 = {{ -width / 2, depth / 2}, { width / 2,  depth / 2}, { -width / 2,  -depth / 2}, { width / 2,  -depth / 2}};
            std::vector<Eigen::Vector2d> imaginary_room_2 = {{ -depth / 2, width / 2}, { depth / 2,  width / 2}, { -depth / 2,  -width / 2}, { depth / 2,  -width /2}};
            //draw imaginary room
//                static std::vector<QGraphicsItem *> draw_points;
//                for (const auto &p: imaginary_room_1)
//                {
//                    auto o = widget_2d->scene.addRect(-20, 20, 200, 200, QPen(Qt::black), QBrush(Qt::black));
//                    o->setPos(p.x(), p.y());
//                    draw_points.push_back(o);
//                }
//
//                for (const auto &p: imaginary_room_2)
//                {
//                    auto o = widget_2d->scene.addRect(-20, 20, 200, 200, QPen(Qt::yellow), QBrush(Qt::yellow));
//                    o->setPos(p.x(), p.y());
//                    draw_points.push_back(o);
//                }

//            std::vector<Eigen::Vector2d> first_room;
//            for(int i = 1; i < 5; i++)
//            {
//                ss.clear();
//                ss.str(room_->attributes.at("corner" + std::to_string(i)));
//                float float1, float2; char coma;
//                ss >> float1 >> coma >> float2;
//                first_room.push_back(Eigen::Vector2d(float1, float2));
//            }

            // Get room corners
            auto first_room_ = room.get_corners();
            // Cast corners to Eigen::Vector2d
            std::vector<Eigen::Vector2d> first_room;
            for (const auto &p: first_room_)
            {
                first_room.push_back(Eigen::Vector2d(p.x(), p.y()));
            }
            // TODO: Check if they are removed and if the corners are set with the values from the forcefield or the
            // calculated by the dimensions of the room in the rooms reference system
            // G->add_or_modify_attrib_local<corner1_att>(room_node, std::vector<float>{static_cast<float>(first_room[0].x()), static_cast<float>(first_room[0].y())});
            // G->add_or_modify_attrib_local<corner2_att>(room_node, std::vector<float>{static_cast<float>(first_room[1].x()), static_cast<float>(first_room[1].y())});
            // G->add_or_modify_attrib_local<corner3_att>(room_node, std::vector<float>{static_cast<float>(first_room[2].x()), static_cast<float>(first_room[2].y())});
            // G->add_or_modify_attrib_local<corner4_att>(room_node, std::vector<float>{static_cast<float>(first_room[3].x()), static_cast<float>(first_room[3].y())});

            auto correspondence_1 = this->calculate_rooms_correspondences(first_room, imaginary_room_1);
            auto correspondence_2 = this->calculate_rooms_correspondences(first_room, imaginary_room_2);

            // Calcular la suma de las distancias cuadradas para cada par de puntos
            double sum_sq_dist1 = 0.0;
            double sum_sq_dist2 = 0.0;
            for (size_t i = 0; i < correspondence_1.size(); ++i)
            {
                sum_sq_dist1 += (correspondence_1[i].first - correspondence_1[i].second).squaredNorm();
                sum_sq_dist2 += (correspondence_2[i].first - correspondence_2[i].second).squaredNorm();
            }

            std::vector<Eigen::Vector2d> imaginary_room;
            if(sum_sq_dist1 < sum_sq_dist2)
                imaginary_room = imaginary_room_1;
            else
                imaginary_room = imaginary_room_2;

            icp icp(imaginary_room, first_room);
            icp.align();

            Eigen::Matrix2d transformation = icp.rotation();
            Eigen::Vector2d translation = icp.translation();

            float angle_rad = std::atan2(transformation(1,0), transformation(0,0));

            // Convert angle from radians to degrees
            G->add_or_modify_attrib_local<pos_x_att>(room_node, (float)(rand()%(170)));
            G->add_or_modify_attrib_local<pos_y_att>(room_node, (float)(rand()%170));
            G->add_or_modify_attrib_local<obj_checked_att>(room_node, false);
            G->add_or_modify_attrib_local<level_att>(room_node, robot_level);
            G->insert_node(room_node);

            std::vector<float> orientation_vector = { 0.0, 0.0, 0.0 };
            std::vector<float> room_pos = { 0.f, 0.f, 0.f };

            //Insert angle rad in Z? Between root and room?
            orientation_vector = { 0.0f, 0.0f, 0.0f };
            DSR::Node root = G->get_node(100).value();
            rt->insert_or_assign_edge_RT(root, room_node.id(), room_pos, orientation_vector);
            G->delete_edge(100, robot_node.id(), "RT");

            // Increase robot node level
            G->add_or_modify_attrib_local<level_att>(robot_node, robot_level + 1);
            G->update_node(robot_node);

            Eigen::Matrix3f rt_rotation_matrix;
            rt_rotation_matrix << cos(angle_rad), -sin(angle_rad), translation.x(),
                    sin(angle_rad), cos(angle_rad), translation.y(),
                    0.0, 0.0, 1.0;

            Eigen::Matrix3f rt_rotation_matrix_inv = rt_rotation_matrix.inverse();
            std::vector<float> robot_pos = { rt_rotation_matrix_inv(0,2), rt_rotation_matrix_inv(1,2), 0.f };
            rt->insert_or_assign_edge_RT(room_node, robot_node.id(), robot_pos, { 0.f, 0.f, -angle_rad });

//            std::vector<Eigen::Vector2d> imaginary_room_trans;
            std::swap(imaginary_room[2], imaginary_room[3]);

            std::vector<float> corner_pos, corner_measured_pos, wall_pos;
            float wall_angle = 0.0;
            DSR::Node new_corner, new_corner_measured;

            auto first_correspondence = calculate_rooms_correspondences_id(imaginary_room, target_points, true);

            for (int i = 1; i < 5; i++)
            {
                // Generate switch case for asigning angles and points depending on the wall/corner id
                switch(i) {
                    case 1:
                        corner_pos = {-abs((float)std::get<1>(first_correspondence[i-1]).x()), 0.0, 0.0};
                        wall_pos = {0.0, (float)std::get<1>(first_correspondence[i-1]).y(), 0.0};
                        break;
                    case 2:
                        corner_pos = {-abs((float)std::get<1>(first_correspondence[i-1]).y()), 0.0, 0.0};
                        wall_pos = {(float)std::get<1>(first_correspondence[i-1]).x(), 0.0, 0.0};
                        wall_angle = -M_PI_2;
                        break;
                    case 3:
                        corner_pos = {-abs((float) std::get<1>(first_correspondence[i - 1]).x()), 0.0, 0.0};
                        wall_pos = {0.0, (float)std::get<1>(first_correspondence[i-1]).y(), 0.0};
                        wall_angle = M_PI;
                        break;
                    case 4:
                        corner_pos = {-abs((float)std::get<1>(first_correspondence[i-1]).y()), 0.0, 0.0};
                        wall_pos = {(float)std::get<1>(first_correspondence[i-1]).x(), 0.0, 0.0};
                        wall_angle = M_PI_2;
                        break;
                    default:
                        cout << "No valid option." << endl;
                        break;
                }
                // insert nominal values
                create_wall(i, wall_pos, wall_angle, room_node);
                if(auto wall_node_ = G->get_node("wall_" + std::to_string(i)); wall_node_.has_value())
                {
                    auto wall_node = wall_node_.value();
                    create_corner(i, corner_pos, wall_node);
                }
                create_corner(i, {(float)std::get<2>(first_correspondence[i-1]).x(), (float)std::get<2>(first_correspondence[i-1]).y(), 0.0}, robot_node, false);
            }
        }
        catch(const std::exception &e)
        { std::cout << e.what() << " Error inserting node" << std::endl;}
//        qInfo() << __FUNCTION__ << " Object with id: " << room_->id << " inserted in graph";
    }
    // no remove of rooms for now
}
std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> SpecificWorker::calculate_rooms_correspondences_id(const std::vector<Eigen::Vector2d> &source_points_, const std::vector<Eigen::Vector2d> &target_points_, bool first_time)
{
    std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> correspondences;

    // Asociar cada punto de origen con el punto más cercano en el conjunto de puntos objetivo
    std::vector<Eigen::Vector2d> source_points_copy = source_points_;
    std::vector<Eigen::Vector2d> target_points_copy = target_points_;
    int i = 1;

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
        correspondences.push_back(std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>(i, *source_iter, closest_point, true));
        i++;
        // Eliminar los puntos correspondientes de sus vectores originales
        source_iter = source_points_copy.erase(source_iter);
        target_points_copy.erase(closest_target_iter);
    }
    if(not first_time)
        for (auto &correspondence: correspondences)
        {
            if ((std::get<1>(correspondence) - std::get<2>(correspondence)).norm() > params.max_distance)
            {
                std::get<3>(correspondence) = false;
                qInfo() << "Corner " << std::get<0>(correspondence) << " is too far away from the target";
            }
        }
    return correspondences;
}
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

//    std::vector<Eigen::Vector2d> target_points;

// Get room corners from visual elements, draw them and insert them in target_points vector
//    std::stringstream ss(room_->attributes.at("corner1"));
//    float float1, float2; char coma;
//    ss >> float1 >> coma >> float2;
//    static QGraphicsEllipseItem* p0;
//    if (p0 != nullptr)
//        widget_2d->scene.removeItem(p0);
//    p0 = widget_2d->scene.addEllipse(float1, float2, 200, 200, QPen(Qt::black), QBrush(Qt::black));
//    target_points.push_back(Eigen::Vector2d(float1, float2));
//
//    ss.clear(); ss.str(room_->attributes.at("corner2"));
//    ss >> float1 >> coma >> float2;
//    static QGraphicsEllipseItem* p1;
//    if (p1 != nullptr)
//        widget_2d->scene.removeItem(p1);
//    p1 = widget_2d->scene.addEllipse(float1, float2, 200, 200, QPen(Qt::black), QBrush(Qt::black));
//    target_points.push_back(Eigen::Vector2d(float1, float2));
//
//    ss.clear(); ss.str(room_->attributes.at("corner3"));
//    ss >> float1 >> coma >> float2;
//    static QGraphicsEllipseItem* p2;
//    if (p2 != nullptr)
//        widget_2d->scene.removeItem(p2);
//    p2 = widget_2d->scene.addEllipse(float1, float2, 200, 200, QPen(Qt::black), QBrush(Qt::black));
//    target_points.push_back(Eigen::Vector2d(float1, float2));
//
//    ss.clear(); ss.str(room_->attributes.at("corner4"));
//    ss >> float1 >> coma >> float2;
//    static QGraphicsEllipseItem* p3;
//    if (p3 != nullptr)
//        widget_2d->scene.removeItem(p3);
//    p3 = widget_2d->scene.addEllipse(float1, float2, 200, 200, QPen(Qt::black), QBrush(Qt::black));
//    target_points.push_back(Eigen::Vector2d(float1, float2));






//            //get doors from data.objects
//            std::vector<RoboCompVisualElementsPub::TObject> doors;
//            for (const auto &object: data.objects | iter::filter([](auto &obj){return obj.attributes.at("name") == "door";}))
//            {
//                doors.push_back(object);
//                //print every attribute of the door
//                std::cout <<  object.attributes.at("name") << std::endl;
//                std::cout <<  object.attributes.at("width") << std::endl;
//                std::cout << object.attributes.at("height") << std::endl;
//                std::cout <<  object.attributes.at("position") << std::endl;
//            }

//create a node in the graph for each door
//            for (const auto &door: doors)
//            {
//                std::string node_name = "door_" + std::to_string(door.id);
//                DSR::Node new_node = DSR::Node::create<door_node_type>(node_name);
//                G->add_or_modify_attrib_local<obj_id_att>(new_node, door.id);
//                G->add_or_modify_attrib_local<pos_x_att>(new_node, std::stof(door.attributes.at("center_x")));
//                G->add_or_modify_attrib_local<pos_y_att>(new_node, std::stof(door.attributes.at("center_y")));
//                G->add_or_modify_attrib_local<timestamp_creation_att>(new_node, get_actual_time());
//                G->add_or_modify_attrib_local<timestamp_alivetime_att>(new_node, get_actual_time());
//                G->add_or_modify_attrib_local<level_att>(new_node, robot_level + 1);
//                G->add_or_modify_attrib_local<obj_checked_att>(new_node, false);
//                G->insert_node(new_node);
//                std::vector<float> vector_room_pos = {std::stof(room_->attributes.at("center_x")),
//                                                      std::stof(room_->attributes.at("center_y")), 0.0},
//                                                      orientation_vector = {0.0, 0.0, std::stof(room_->attributes.at("rotation"))};
//                vector_room_pos = { std::stof(door.attributes.at("center_x")), std::stof(door.attributes.at("center_y")), 0.f};
//                rt->insert_or_assign_edge_RT(room_node, new_node.id(), vector_room_pos, orientation_vector);
//            }