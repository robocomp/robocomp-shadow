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
        this->consts.DISPLAY = params.at("display").value == "true" or (params.at("delay").value == "True");
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

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/
		//graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);

        // A thread is created
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        // timers
        Period = 50;
        timer.start(Period);
        //        if( not consts.DISPLAY) hide();
	}
}

/// Proto CODE
// general goal: waits for a room to have a "current" self edge. Check if there are nominal doors in it, detect doors from sensors and try to match both sets.
//               The actions to take are:
//                   1. if there is NO nominal door matching a measured one in the room, start stabilization procedure
//                   2. if there is a nominal door matching a measured one, write in G the measured door as valid
//                   3. if there is a nominal door NOT matching a measured one in the room, IGNORE for now.

// requirements:

// compute
// real lidar data
// check for robot and current room
// get nominal doors from room in robot's frame  -> std::vector<Nominal_Doors> nominal_doors
// get measured doors from lidar in robot's frame -> std::vector<Doors> measured_doors
// match both sets
// write each nominal door matching a measured door in G as a valid measured door
// if there is a nominal door not matching a measured one, and is not being stabilized, start stabilization procedure for it
//      one option is to start a thread for each door to stabilize. This thread would have to:
//          1. initialize with current measured door and reset accumulators for trajectory
//          2. create a "target" edge, meaning that the agent has an intention to reach a target. "target" edge has the agent id as attribute
//          3. wait for the "target" edge to be upgraded to "goto" edge
//          4. once the "goto" edge is active, start recording poses and landmarks
//          5. once the target is reached, delete the "goto" edge and create a new "measured" door in G
//          6. finish thread
//     the thread has to respond when compute asks if a given measured door is being stabilized by it.

void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value()) { return; }
    auto ldata = res_.value();

    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    /// Check if room node exists
    auto room_node_ = G->get_node("room");
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room node in graph"; return; }
    auto room_node = room_node_.value();

    auto result = get_corners_and_wall_centers();
    if( not result.has_value())
    {
        qWarning() << __FUNCTION__ << " No corners and wall centers obtained";
        return;
    }
    /// Get corners and wall centers
    auto [corners, wall_centers] = result.value();

    ///----------------------------TODO:-------------------------------------------

    ///Comprobar si existen puertas_estabilizadas en el grafo.
    ///Si existen:
        ///Tracking puertas observadas.
    ///Si no existen:
        ///inyectar measured doors
        ///Elegir una no estabilizada else continuar con misma door
                ///ID: wall+nºpuerta
        ///Generar target estabilización.
            //punto normal d=1m
        ///Generar histograma (coordenada X respecto al wall, ancho puerta)
        ///Asociar puertas observadas
        ///Almacenar valores
        //Si target alcanzado:
            ///Generar .g2o (¿corner room estabilizados y fixed?, odometría, nominal puerta(vértices), medidas corners, puertas)
            ///llamar optimizador
            ///obtener puerta optimizada
            ///insertar puerta en el grafo
            ///TODO: (¿modificar g2o python para incorporar puertas estabilizadas para optimización pose?)

    /// Check if doors exist in the graph
    auto door_nodes = G->get_nodes_by_type("door");
    /// Get door nodes witch not contain "measured" in name
    door_nodes.erase(std::remove_if(door_nodes.begin(), door_nodes.end(), [](auto &n){ return n.name().find("measured") != std::string::npos; }), door_nodes.end());
    /// Generate a vector with door nominal poses structured in Eigen::Vector2f // TODO: to function

    vector<Eigen::Vector2f> door_poses = get_nominal_door_from_dsr(robot_node, door_nodes);

    /// TEMPORARY ADD A FOLLOWER TO TARGET WHEN EDGE GOTO IS ACTIVE
    ///SET TARGET TO CROSS THE DOOR

    /// Iterate over doors
    for (size_t i = 0; i < door_nodes.size(); ++i)
    {
        /// Check if any GOTO edge from robot to any door node exists
        if(auto goto_edge = G->get_edge(robot_node.id(), door_nodes[i].id(), "goto_action"); goto_edge.has_value())
        {
            /// Get door parent node
            if(auto parent_node = G->get_parent_node(door_nodes[i]); parent_node.has_value())
            {
                auto parent_node_value = parent_node.value();
                /// Get door position respect to robot
                if(auto rt_door_edge = rt->get_edge_RT(parent_node_value, door_nodes[i].id()); rt_door_edge.has_value())
                {
                    auto door_edge = rt_door_edge.value();
                    if(auto rt_translation = G->get_attrib_by_name<rt_translation_att>(door_edge); rt_translation.has_value())
                    {
                        auto rt_door_value = rt_translation.value().get();
                        /// Get door width to calculate door vertices
                        if(auto door_width_ = G->get_attrib_by_name<width_att>(door_nodes[i]); door_width_.has_value()) {
                            auto door_width = door_width_.value();
                            /// Generate Eigen::Vector3d for each door vertex
                            Eigen::Vector3d door_robot_pos_point_0{rt_door_value[0] - (door_width / 2),
                                                                   rt_door_value[1], 0.f};
                            Eigen::Vector3d door_robot_pos_point_1{rt_door_value[0] + (door_width / 2),
                                                                   rt_door_value[1], 0.f};
                            /// Transform door vertices to robot frame
                            if (auto door_transformed_0 = inner_eigen->transform(robot_node.name(),
                                                                                 door_robot_pos_point_0,
                                                                                 parent_node_value.name()); door_transformed_0.has_value())
                            {
                                auto door_transformed_0_value = door_transformed_0.value();
                                /// Transform door vertices to robot frame
                                if (auto door_transformed_1 = inner_eigen->transform(robot_node.name(),
                                                                                     door_robot_pos_point_1,
                                                                                     parent_node_value.name()); door_transformed_0.has_value())
                                {
                                    auto door_transformed_1_value = door_transformed_1.value();
                                    /// Generate a DoorDetector::Door with door vertices
                                    DoorDetector::Door act_door{Eigen::Vector2f{door_transformed_0_value.x(), door_transformed_0_value.y()},
                                                                Eigen::Vector2f{door_transformed_1_value.x(), door_transformed_1_value.y()}, 0, 0};
                                    /// Generate a target found at 1000 mm in the perpendicular direction to the door
                                    auto [p1, p2] = act_door.point_perpendicular_to_door_at();
                                    /// Transform both points to room frame
                                    if (auto p1_transformed = inner_eigen->transform(room_node.name(),
                                                                                     Eigen::Vector3d{p1.x(), p1.y(), 0.f},
                                                                                     robot_node.name()); p1_transformed.has_value())
                                    {
                                        auto p1_transformed_value = p1_transformed.value();
                                        if (auto p2_transformed = inner_eigen->transform(room_node.name(),
                                                                                         Eigen::Vector3d{p2.x(), p2.y(), 0.f},
                                                                                         robot_node.name()); p2_transformed.has_value())
                                        {
                                            auto p2_transformed_value = p2_transformed.value();
                                            /// Considering room dimensions, choose as target the point outside the room
                                            if(auto room_depth = G->get_attrib_by_name<depth_att>(room_node); room_depth.has_value())
                                            {
                                                auto room_depth_value = room_depth.value();
                                                if(auto room_width = G->get_attrib_by_name<width_att>(room_node); room_width.has_value())
                                                {
                                                    auto room_width_value = room_width.value();
                                                    // Check which point is inside the room
                                                    Eigen::Vector2f target;
                                                    if(p1_transformed_value.x() > -room_width_value / 2 and p1_transformed_value.x() < room_width_value / 2 and p1_transformed_value.y() > -room_depth_value / 2 and p1_transformed_value.y() < room_depth_value / 2)
                                                    {
                                                        qInfo() << "Point 1 inside room";
                                                        target = p2;
                                                    }
                                                    else
                                                    {
                                                        qInfo() << "Point 2 inside room";
                                                        target = p1;
                                                    }
                                                    /// Convert target to Eigen::Vector3d
                                                    Eigen::Vector3d target_point{target.x(), target.y(), 0.f};
                                                    /// Print target point
                                                    qInfo() << "Target point: " << target_point.x() << " " << target_point.y();
                                                    /// Generate speed commands towards room center and insert them in graph
                                                    auto speeds = calculate_speed(target_point);
                                                    set_robot_speeds(speeds[0],speeds[1],speeds[2]);
                                                    if(movement_completed(target_point, distance_to_target))
                                                    {
                                                        set_robot_speeds(0.f, 0.f, 0.f);
                                                        qInfo() << "Door point reached";
                                                        if(G->delete_edge(robot_node.id(), door_nodes[i].id(), "goto_action"))
                                                            std::cout << __FUNCTION__ << " Edge successfully deleted: " << std::endl;
                                                        else
                                                            std::cout << __FUNCTION__ << " Fatal error deleting edge: " << std::endl;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get doors
    auto doors = get_doors(ldata, corners, wall_centers, &widget_2d->scene);
    if(doors.empty())
    {
        // TODO: if no doors detected, set every measured door as no valid
        qWarning() << __FUNCTION__ << " No doors detected";
        return;
    }

    // If door exists, update
    if(door_nodes.size() > 0)
    {
        /// Do data association with Hungarian algorithm
        /// Generate a matrix to store the distances between measured doors poses and calculated door poseds using double vectors instead of Eigen
        std::vector<std::vector<double>> distances_matrix(doors.size(), std::vector<double>(door_nodes.size()));
        for(size_t i = 0; i < doors.size(); ++i)
        {
            Eigen::Vector2f target_point = (std::get<1>(doors[i]) + std::get<2>(doors[i])) / 2;
            qInfo() << "Door measured: " << target_point.x() << " " << target_point.y();
            for(size_t j = 0; j < door_nodes.size(); ++j)
            {
                qInfo() << "Door nominal transformed: " << door_poses[j].x() << " " << door_poses[j].y();
                distances_matrix[i][j] = (door_poses[j] - target_point).norm();
            }
        }
        /// Process metrics matrix with Hungarian algorithm
        vector<int> assignment;
        double cost = HungAlgo.Solve(distances_matrix, assignment);
        /// Get the index in assignment vector with a value different from -1
        auto door_index = std::find_if(assignment.begin(), assignment.end(), [](auto &a){ return a != -1; });

        /// Check if distance is greater than threshold
        if(distances_matrix[door_index - assignment.begin()][0] < door_center_matching_threshold)
        {
            /// Get door node associated
            auto associated_door = doors[door_index - assignment.begin()];
            update_door_in_graph(std::get<1>(associated_door), door_nodes[0].name()+"_measured");
            qInfo() << "Door updated";
        }
    }

    static std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> measured_corner_data;
    static std::vector<std::vector<float>> odometry_data;
    static std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> measured_door_points;
    /// Generate static map for histogram with coordinate in wall and door width
    static std::map<int, int> width_histogram;
    static std::map<int, int> pose_histogram;
    static Eigen::Affine2d first_robot_pose;

    /// Check if TARGET edge exists
    if(auto target_edge_ = G->get_edge(robot_node.id(), robot_node.id(), "TARGET"); target_edge_.has_value())
    {
        Eigen::Vector3d door_stabilization_target_transformed_value;
        /// Transform door_stabilization_target to robot frame
        if(auto door_stabilization_target_transformed = inner_eigen->transform(robot_node.name(), door_stabilization_target, room_node.name()); door_stabilization_target_transformed.has_value())
            door_stabilization_target_transformed_value = door_stabilization_target_transformed.value();
        else
        { qWarning() << __FUNCTION__ << " No door stabilization target transformed"; return; }

        if (door_to_stabilize_.has_value())
        {
            /// Get door to stabilize data
            auto door_to_stabilize = door_to_stabilize_.value();

            /// Generate a matrix to store the distances between measured doors poses and calculated door poseds using double vectors instead of Eigen
            std::vector<std::vector<double>> distances_matrix(1, std::vector<double>(doors.size()));
            for (size_t j = 0; j < doors.size(); ++j)
            {
                Eigen::Vector2f target_point = (std::get<1>(doors[j]) + std::get<2>(doors[j])) / 2;
                distances_matrix[0][j] = (door_to_stabilize.middle - target_point).norm();
            }
            /// Process metrics matrix with Hungarian algorithm
            vector<int> assignment;
            double cost = HungAlgo.Solve(distances_matrix, assignment);
            /// Get the index in assignment vector with a value different from -1
            auto door_index = std::find_if(assignment.begin(), assignment.end(), [](auto &a) { return a != -1; });


            /// Check if distance is greater than threshold
            if (distances_matrix[0][door_index - assignment.begin()] < door_center_matching_threshold)
            {
                auto associated_door = doors[door_index - assignment.begin()];
                /// Print bot door centers
                qInfo() << "Door to stabilize: " << door_to_stabilize.middle.x() << " " << door_to_stabilize.middle.y();

                DoorDetector::Door act_door{std::get<1>(associated_door), std::get<2>(associated_door), 0, 0};
                qInfo() << "Associated door: " << act_door.middle.x() << " " << act_door.middle.y();
                act_door.id = door_to_stabilize.id;
                act_door.wall_id = door_to_stabilize.wall_id;
                door_to_stabilize_ = act_door;

                auto wall_node_ = G->get_node("wall_" + std::to_string(std::get<0>(associated_door)));
                if (not wall_node_.has_value()) {
                    qWarning() << __FUNCTION__ << " No wall node in graph";
                    return;
                }
                auto wall_node = wall_node_.value();

                //// TARGET REACHED -> GENERATE G2O, OPTIMIZE, GET DOOR POSE & WIDTH, INSERT TO DSR, LIBERATE TARGET AND CLEAR
                if(movement_completed(door_stabilization_target_transformed_value, distance_to_target))
                {

                    set_robot_speeds(0.f, 0.f, 0.f);

                    qInfo() << "Door point reached";


                    // Generate room size histogram considering room_size_histogram vector and obtain the most common room size
                    auto most_common_door_width = std::max_element(width_histogram.begin(), width_histogram.end(),
                                                                  [](const auto &p1, const auto &p2){ return p1.second < p2.second; });
                    auto most_common_door_pose = std::max_element(pose_histogram.begin(), pose_histogram.end(),
                                                                   [](const auto &p1, const auto &p2){ return p1.second < p2.second; });
                    /// Get the most common room size
                    auto door_pose = most_common_door_pose->first;
                    auto door_width = most_common_door_width->first;
                    qInfo() << "Most common door data: " << door_pose << " " << door_width << "#############################";

                    /// Considering door pose respect to wall and door width, generate the door vertices
                    auto door_point_0 = Eigen::Vector3d{door_pose - (door_width * 0.5), 0.f, 0.f};
                    auto door_point_1 = Eigen::Vector3d{door_pose + (door_width * 0.5), 0.f, 0.f};

                    std::pair<Eigen::Vector2f, Eigen::Vector2f> nominal_door_points;


                    /// Transform nominal door points respect to room frame
                    if (auto nominal_door_point_0 = inner_eigen->transform(room_node.name(),
                                                                               door_point_0,
                                                                               wall_node.name()); nominal_door_point_0.has_value())
                        if (auto nominal_door_point_1 = inner_eigen->transform(room_node.name(),
                                                                               door_point_1,
                                                                               wall_node.name()); nominal_door_point_1.has_value())
                        {
                            auto nominal_door_point_0_float = nominal_door_point_0.value().cast<float>();
                            auto nominal_door_point_1_float = nominal_door_point_1.value().cast<float>();
                            /// Make pair with nominal door points
                            nominal_door_points = std::make_pair(Eigen::Vector2f{nominal_door_point_0_float.x(), nominal_door_point_0_float.y()}, Eigen::Vector2f{nominal_door_point_1_float.x(), nominal_door_point_1_float.y()});
                            qInfo() << "Nominal door points: " << nominal_door_point_0_float.x() << " " << nominal_door_point_0_float.y() << " " << nominal_door_point_1_float.x() << " " << nominal_door_point_1_float.y();
                        }
                        else
                        { qWarning() << __FUNCTION__ << " No nominal door point 1"; return; }
                    else
                    { qWarning() << __FUNCTION__ << " No nominal door point 0"; return; }


                    /// Get corner nodes without "measured" in name
                    auto corner_nodes = G->get_nodes_by_type("corner");
                    auto wall_nodes = G->get_nodes_by_type("wall");
                    if (corner_nodes.empty()) {
                        qWarning() << __FUNCTION__ << " No corner nodes in graph";
                        return;
                    }
                    ///Sort wall nodes by name
                    std::sort(wall_nodes.begin(), wall_nodes.end(),
                              [](auto &n1, auto &n2) { return n1.name() < n2.name(); });
                    if (corner_nodes.empty()) {
                        qWarning() << __FUNCTION__ << " No corner nodes in graph";
                        return;
                    }
                    /// Get nodes which name not contains "measured"
                    corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [](auto &n) {
                        return n.name().find("measured") != std::string::npos;
                    }), corner_nodes.end());
                    /// Sort corner nodes by name
                    std::sort(corner_nodes.begin(), corner_nodes.end(),
                              [](auto &n1, auto &n2) { return n1.name() < n2.name(); });

                    /// Create nominal corner data vector
                    std::vector<Eigen::Matrix<float, 2, 1>> nominal_corner_data;
                    for (const auto &[i, n]: corner_nodes | iter::enumerate)
                    {
                        /// Print corner name and wall name
                        std::cout << "Corner name: " << n.name() << std::endl;
                        std::cout << "Wall name: " << wall_nodes[i].name() << std::endl;
                        if (auto rt_corner_edge = rt->get_edge_RT(wall_nodes[i], n.id()); rt_corner_edge.has_value())
                        {
                            // Get corner pose
                            auto corner_edge = rt_corner_edge.value();
                            ///Get translation
                            if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(corner_edge); rt_translation.has_value())
                            {
                                auto rt_corner_value = rt_translation.value().get();

                                /// Generate Eigen::Vector3d
                                Eigen::Vector3d corner_robot_pos_point{rt_corner_value[0], rt_corner_value[1], 0.f};
                                /// Transform corner pose to room frame
                                if (auto corner_transformed = inner_eigen->transform(room_node.name(),
                                                                                     corner_robot_pos_point,
                                                                                     wall_nodes[i].name()); corner_transformed.has_value())
                                {
                                    auto corner_transformed_value = corner_transformed.value();
                                    qInfo() << "Corner position: " << corner_transformed_value.x() << " "
                                            << corner_transformed_value.y();
                                    auto corner_transformed_value_float = corner_transformed_value.cast<float>();
                                    qInfo() << "Corner position: " << corner_transformed_value_float.x() << " "
                                            << corner_transformed_value_float.y();
                                    nominal_corner_data.push_back(Eigen::Vector2f{corner_transformed_value_float.x(),
                                                                                  corner_transformed_value_float.y()});
                                }
                            }
                        }
                    }


                    /// BUILD G2O GRAPH
                    std::string g2o_graph_data = build_g2o_graph(measured_corner_data, nominal_corner_data, odometry_data, first_robot_pose, measured_door_points, nominal_door_points);
                    //
                    qInfo() << "----------------------------------- graph data--------------------------------------";
//                    std::cout << g2o_graph_data << std::endl;
                    /// Save string to file
                    std::ofstream file("graph_data.g2o");
                    file << g2o_graph_data;
                    file.close();

                    /// CALL OPTIMIZER
                    try
                    {
                        /// EXTRACT OPTIMIZED VALUES, call g2o_cpp proxy
                        auto optimization = this->g2ooptimizer_proxy->optimize(g2o_graph_data);


                        std::ofstream file_opt("graph_data_opt.g2o");
                        file_opt << optimization;
                        file_opt.close();

                        auto optimized_door_data = extract_g2o_data(optimization);
                        DoorDetector::Door nominal_door = {nominal_door_points.first, nominal_door_points.second, 0, 0};
                        DoorDetector::Door act_door = {measured_door_points[0].first, measured_door_points[0].second, 0, 0};
                        /// Assign id to doors
                        nominal_door.id = door_to_stabilize.id;
                        act_door.id = door_to_stabilize.id;
                        DSR::Node inserted_door;
                        inserted_door = insert_nominal_door_into_graph(nominal_door, door_to_stabilize.wall_id);
                        insert_measured_door_into_graph(act_door, door_to_stabilize.wall_id);

                        /// INSERT TO DSR

                        /// Clear static vectors
                        measured_corner_data.clear();
                        odometry_data.clear();
                        measured_door_points.clear();
                        width_histogram.clear();
                        pose_histogram.clear();
                        first_robot_pose = Eigen::Affine2d::Identity();

                        /// Clear door to stabilize
                        door_to_stabilize_ = std::nullopt;

                        ///TODO:
                        ///WAIT (DEBUG)
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        /// ADD EDGE GOTO FROM ROBOT TO DOOR
                        generate_edge_goto_door(robot_node,inserted_door);

                    }
                    catch (const std::exception &e)
                    {
                        qWarning() << __FUNCTION__ << " Error optimizing door data";
                        return;
                    }
                    /// Delete TARGET edge to make robot control agent stop taking into account speed commands
                    G->delete_edge(robot_node.id(), robot_node.id(), "TARGET");





                }
                else
                {
                    // Print static vector sizes
//                    qInfo() << "Measured corner data size: " << measured_corner_data.size();
//                    qInfo() << "Odometry data size: " << odometry_data.size();
//                    qInfo() << "Measured door points size: " << measured_door_points.size();
//                    qInfo() << "Pose width histogram size: " << pose_width_histogram.size();
                    measured_door_points.push_back(std::make_pair(std::get<1>(associated_door), std::get<2>(associated_door)));

                    // Generate Eigen::Vector3f for door position
                    Eigen::Vector3f door_pos_point{act_door.middle.x(), act_door.middle.y(), 0.f};
                    auto door_robot_pos_point_double = door_pos_point.cast<double>();

                    auto corner_node_ = G->get_node("corner_" + std::to_string(std::get<0>(associated_door)));
                    if (not corner_node_.has_value()) {
                        qWarning() << __FUNCTION__ << " No wall node in graph";
                        return;
                    }
                    auto corner_node = corner_node_.value();

                    int distance_to_wall_center;
                    if (auto door_transformed = inner_eigen->transform(wall_node.name(), door_robot_pos_point_double,
                                                                       robot_node.name()); door_transformed.has_value()) {
                        auto door_transformed_value = door_transformed.value();
                        auto door_transformed_value_float = door_transformed_value.cast<float>();
                        qInfo() << "Door position: " << door_transformed_value_float.x() << " "
                                << door_transformed_value_float.y();
                        distance_to_wall_center = static_cast<int>(door_transformed_value_float.x());
                    }
                    /// Insert door width and distance to wall center in histogram
                    int door_width = static_cast<int>((std::get<1>(associated_door) - std::get<2>(associated_door)).norm());
                    pose_histogram[distance_to_wall_center]++;
                    width_histogram[door_width]++;

                    /// Get measured corners from graph
                    auto corner_nodes = G->get_nodes_by_type("corner");
                    if (corner_nodes.empty()) {
                        qWarning() << __FUNCTION__ << " No corner nodes in graph";
                        return;
                    }
                    /// Get nodes which name contains "measured"
                    corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [](auto &n) {
                        return n.name().find("measured") == std::string::npos;
                    }), corner_nodes.end());
                    /// Order corner nodes by name
                    std::sort(corner_nodes.begin(), corner_nodes.end(),
                              [](auto &n1, auto &n2) { return n1.name() < n2.name(); });
                    /// Iterate over corners
                    std::vector<Eigen::Matrix<float, 2, 1>> actual_measured_corner_data;
                    for (const auto &[i, n]: corner_nodes | iter::enumerate) {
                        if (auto rt_robot_corner = rt->get_edge_RT(robot_node, n.id()); rt_robot_corner.has_value()) {
                            auto robot_corner_edge = rt_robot_corner.value();
                            if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(
                                        robot_corner_edge); rt_translation.has_value()) {
                                auto rt_corner_value = rt_translation.value().get();
                                Eigen::Vector2f corner_robot_pos_point(rt_corner_value[0], rt_corner_value[1]);
                                actual_measured_corner_data.push_back({corner_robot_pos_point});
                            }
                        }
                    }
                    measured_corner_data.push_back(actual_measured_corner_data);
                    /// Get robot odometry
                    auto odometry = get_graph_odometry();
                    odometry_data.push_back(odometry);

                    /// Generate speed commands towards room center and insert them in graph
                    auto speeds = calculate_speed(door_stabilization_target_transformed_value);
                    qInfo() << "Speeds: " << speeds[0] << " " << speeds[1] << " " << speeds[2];
                    set_robot_speeds(speeds[0], speeds[1], speeds[2]);
                }
            }
        }
    }
    /// If no doors exist, choose one to stabilize
    else if(door_nodes.empty() and not door_to_stabilize_.has_value())
    {
        qInfo() << "No doors in the graph";
        auto door = doors[0];
        DoorDetector::Door act_door{std::get<1>(door), std::get<2>(door), 0, 0};
        act_door.id = 0;
        act_door.wall_id = std::get<0>(door);
        /// Set the first door to stabilize
//        door_to_stabilize_ = act_door;

        // Get room -> robot RT edge
        auto rt_room_robot = rt->get_edge_RT(room_node, robot_node.id());
        if(not rt_room_robot.has_value())
        { qWarning() << __FUNCTION__ << " No room -> robot RT edge"; return; }
        auto rt_room_robot_edge = rt_room_robot.value();
        if(auto rt_translation = G->get_attrib_by_name<rt_translation_att>(rt_room_robot_edge); rt_translation.has_value())
        {
            if(auto rt_rotation = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(rt_room_robot_edge); rt_rotation.has_value())
            {
                auto rt_translation_value = rt_translation.value().get();
                auto rt_rotation_value = rt_rotation.value().get();

                // Set rotation and traslation data
                first_robot_pose.setIdentity();

                first_robot_pose.rotate(-rt_rotation_value[2]).pretranslate(Eigen::Vector2d {rt_translation_value[0], rt_translation_value[1]});
            }
        }
        else
        { qWarning() << __FUNCTION__ << " No room -> robot RT translation"; return; }

        measured_door_points.push_back(std::make_pair(std::get<1>(door), std::get<2>(door)));

        // Generate Eigen::Vector3f for door position
        Eigen::Vector3f door_pos_point {act_door.middle.x(), act_door.middle.y(), 0.f};
        auto door_robot_pos_point_double = door_pos_point.cast<double>();

        auto wall_node_ = G->get_node("wall_"+std::to_string(std::get<0>(door)));
        if(not wall_node_.has_value())
        { qWarning() << __FUNCTION__ << " No wall node in graph"; return; }
        auto wall_node = wall_node_.value();

        auto corner_node_ = G->get_node("corner_"+std::to_string(std::get<0>(door)));
        if(not corner_node_.has_value())
        { qWarning() << __FUNCTION__ << " No wall node in graph"; return; }
        auto corner_node = corner_node_.value();

        /// Distance from door center to center of wall
        int distance_to_wall_center;
        if(auto door_transformed = inner_eigen->transform(wall_node.name(), door_robot_pos_point_double, robot_node.name()); door_transformed.has_value())
        {
            auto door_transformed_value = door_transformed.value();
            auto door_transformed_value_float = door_transformed_value.cast<float>();
            qInfo() << "Door position: " << door_transformed_value_float.x() << " " << door_transformed_value_float.y();
            distance_to_wall_center = static_cast<int>(door_transformed_value_float.x());
            pose_histogram[distance_to_wall_center]++;
        }

        /// Insert door width and distance to wall center in histogram
        int door_width = static_cast<int>((std::get<1>(door) - std::get<2>(door)).norm());
        width_histogram[door_width]++;

        /// Get measured corners from graph
        auto corner_nodes = G->get_nodes_by_type("corner");
        if(corner_nodes.empty())
        { qWarning() << __FUNCTION__ << " No corner nodes in graph"; return; }
        /// Get nodes which name contains "measured"
        corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [](auto &n){ return n.name().find("measured") == std::string::npos; }), corner_nodes.end());
        /// Order corner nodes by name
        std::sort(corner_nodes.begin(), corner_nodes.end(), [](auto &n1, auto &n2){ return n1.name() < n2.name(); });
        /// Iterate over corners
        std::vector<Eigen::Matrix<float, 2, 1>> actual_measured_corner_data;
        for(const auto &[i, n] : corner_nodes | iter::enumerate)
        {
            if(auto rt_robot_corner = rt->get_edge_RT(robot_node, n.id()); rt_robot_corner.has_value())
            {
                auto robot_corner_edge = rt_robot_corner.value();
                if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(robot_corner_edge); rt_translation.has_value())
                {
                    auto rt_corner_value = rt_translation.value().get();
                    Eigen::Vector2f corner_robot_pos_point(rt_corner_value[0], rt_corner_value[1]);
                    actual_measured_corner_data.push_back({corner_robot_pos_point});
                }
            }
        }
        measured_corner_data.push_back(actual_measured_corner_data);
        /// Get robot odometry
        auto odometry = get_graph_odometry();
        odometry_data.push_back(odometry);
        door_to_stabilize_ = act_door;

        /// Generate a target found at 1000 mm in the perpendicular direction to the door
        auto [p1, p2] = act_door.point_perpendicular_to_door_at();
        /// Choose as target the closer point to the robot
        auto target = (p1.norm() < p2.norm()) ? p1 : p2;
        /// Convert target to Eigen::Vector3d
        Eigen::Vector3d target_point{target.x(), target.y(), 0.f};
        /// Transform target point to room frame

        if(auto target_transformed = inner_eigen->transform(room_node.name(), target_point, robot_node.name()); target_transformed.has_value())
        {
            door_stabilization_target = target_transformed.value();
            qInfo() << "Door stabilization target: " << door_stabilization_target.x() << " " << door_stabilization_target.y();
        }
        else
        { qWarning() << __FUNCTION__ << " No target transformed"; return; }

        /// Generate target edge to move the robot
        generate_target_edge(robot_node);
        /// Generate speed commands towards room center and insert them in graph
        auto speeds = calculate_speed(target_point);
        set_robot_speeds(speeds[0],speeds[1],speeds[2]);
    }

    draw_door(doors, &widget_2d->scene, QColor("red"));
    if(widget_2d != nullptr)
    {
        draw_lidar(ldata, &widget_2d->scene);
    }
    fps.print("door_detector");
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
std::optional<std::tuple<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>> SpecificWorker::get_corners_and_wall_centers()
{
    /// Check if robot node exists
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return{}; }
    auto robot_node = robot_node_.value();

    /// Get corner nodes from graph
    auto corner_nodes = G->get_nodes_by_type("corner");
    if(corner_nodes.empty())
    { qWarning() << __FUNCTION__ << " No corner nodes in graph"; return{}; }
    /// Get nodes which name not contains "measured"
    corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [](auto &n){ return n.name().find("measured") != std::string::npos; }), corner_nodes.end());
    /// print corner names
//    for(auto &n: corner_nodes) std::cout << __FUNCTION__ << " Corner node: " << n.name() << std::endl;
    /// Sort corner nodes by name
    std::sort(corner_nodes.begin(), corner_nodes.end(), [](auto &n1, auto &n2){ return n1.name() < n2.name(); });
    std::vector<Eigen::Vector2f> corners, wall_centers;
    // Iterate over corners
    for(const auto &[i, n] : corner_nodes | iter::enumerate)
    {
        if(auto parent_node_ = G->get_parent_node(n); parent_node_.has_value())
        {
            auto parent_node = parent_node_.value();
            /// If "wall" string is in parent node name
            if(parent_node.name().find("wall") != std::string::npos)
            {
                if(auto rt_corner_edge_measured = rt->get_edge_RT(parent_node, n.id()); rt_corner_edge_measured.has_value())
                {
                    auto corner_edge_measured = rt_corner_edge_measured.value();
                    if (auto rt_translation_measured = G->get_attrib_by_name<rt_translation_att>(rt_corner_edge_measured.value()); rt_translation_measured.has_value())
                    {
                        auto rt_corner_measured_value = rt_translation_measured.value().get();
                        Eigen::Vector3f corner_robot_pos_point(rt_corner_measured_value[0],
                                                               rt_corner_measured_value[1], 0.f);
                        auto corner_robot_pos_point_double = corner_robot_pos_point.cast<double>();
                        if (auto corner_transformed = inner_eigen->transform(robot_node.name(),
                                                                             corner_robot_pos_point_double,
                                                                             parent_node.name()); corner_transformed.has_value())
                        {
                            auto corner_transformed_value = corner_transformed.value();
                            auto corner_transformed_value_float = corner_transformed_value.cast<float>();
                            corners.push_back({corner_transformed_value_float.x(), corner_transformed_value_float.y()});
//                          // If i > 0, calculate the center of the wall
                            if(i > 0)
                            {
                                Eigen::Vector2f center;
                                center = (corners[i] + corners[i-1]) / 2;
                                wall_centers.push_back(center);
                                if(i == corner_nodes.size() - 1)
                                {
                                    center = (corners[i] + corners[0]) / 2;
                                    /// Insert in front of the vector
                                    wall_centers.insert(wall_centers.begin(), center);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return std::make_tuple(corners, wall_centers);
}

std::vector<tuple<int, Eigen::Vector2f, Eigen::Vector2f>> SpecificWorker::get_doors(const RoboCompLidar3D::TData &ldata, const std::vector<Eigen::Vector2f> &corners, const std::vector<Eigen::Vector2f> &wall_centers, QGraphicsScene *scene)
{
    /// // Create empty QPolygonF
    QPolygonF poly_room_in, poly_room_out, nominal_polygon;
    float d = 250;
    for(auto &c: corners)
    {
        /// Insert nominal polygon
        nominal_polygon << QPointF(c.x(), c.y());
        auto center = std::accumulate(corners.begin(), corners.end(), Eigen::Vector2f(0,0), [](const Eigen::Vector2f& acc, const Eigen::Vector2f& c){ return acc + c; }) / corners.size();
        auto dir = (center - c).normalized();
        Eigen::Vector2f new_corner_in = c + dir * d;
        poly_room_in << QPointF(new_corner_in.x(), new_corner_in.y());
        Eigen::Vector2f new_corner_out = c - dir * d;
        poly_room_out << QPointF(new_corner_out.x(), new_corner_out.y());
    }

    // Filter lidar points inside room polygon
    std::vector<bool> inside_poly_out (ldata.points.size());
    std::vector<bool> outside_poly_in (ldata.points.size());
    std::vector<bool> in_wall (ldata.points.size());
    std::vector<int> in_wall_indexes;

    for(const auto &[i, p] : ldata.points | iter::enumerate)
    {
        // if point z is between 1000 and 2500
        if(p.z < 100 and p.z > 2500)
            continue;
        if(poly_room_out.containsPoint(QPointF(p.x, p.y), Qt::OddEvenFill))
            inside_poly_out[i] = true;
        else
            inside_poly_out[i] = false;
        if(not poly_room_in.containsPoint(QPointF(p.x, p.y), Qt::OddEvenFill))
            outside_poly_in[i] = true;
        else
            outside_poly_in[i] = false;
        if(inside_poly_out[i] and outside_poly_in[i])
        {
            in_wall_indexes.push_back(i);
            in_wall[i] = true;
        }
        else
            in_wall[i] = false;
    }
    std::vector<tuple<int, Eigen::Vector2f, Eigen::Vector2f>> doors;

    qInfo() << "------------------------------------------------------------------------------";
    /// Iterate in_vall_indexes using sliding_window
    for(const auto &window : in_wall_indexes | iter::sliding_window(2))
        if(window.size() == 2)
        {
            auto p0 = ldata.points[window[0]];
            auto p1 = ldata.points[window[1]];
            auto line = Eigen::Vector2f(p1.x - p0.x, p1.y - p0.y);
            auto line_norm = line.norm();
            //print(line_norm) and p0 p1

            /// Door width condition
            if(line_norm > 700 and line_norm < 1500)
            {
//                qInfo()<< "Line norm: " << line_norm << " " << p0.x << p0.y << " " << p1.x << p1.y;
                bool is_door = true;
                /// Check all indexes between window[0] and window[1] in lidar points
                /// If there is a point inside the wall, then it is not a door
                for(int i = window[0]; i < window[1]; i++)
                    if(not outside_poly_in[i])
                    {
                        is_door = false;
                        break;
                    }
                if(is_door)
                {
                    /// Project p0 and p1 on the polygon
                    auto p0_projected = projectPointOnPolygon(QPointF(p0.x, p0.y), nominal_polygon);
                    auto p1_projected = projectPointOnPolygon(QPointF(p1.x, p1.y), nominal_polygon);
                    /// Check between which nominal corners the projected points are
                    auto p0_projected_eigen = Eigen::Vector2f(p0_projected.x(), p0_projected.y());
                    auto p1_projected_eigen = Eigen::Vector2f(p1_projected.x(), p1_projected.y());
                    /// Obtanin the central point of the door and, considering the nominal corners, check between which corners the door is
                    auto middle = (p0_projected_eigen + p1_projected_eigen) / 2;
//                    qInfo() << "Middle: " << middle.x() << middle.y();
                    /// Get the index of the wall point closer to middle point using a lambda
                    auto wall_center = *std::min_element(wall_centers.begin(), wall_centers.end(), [&middle](const Eigen::Vector2f &a, const Eigen::Vector2f &b){ return (a - middle).norm() < (b - middle).norm(); });
//                    qInfo() << "Wall center: " << wall_center.x() << wall_center.y();
                    /// Get the index of the wall center
                    auto wall_center_index = std::distance(wall_centers.begin(), std::find(wall_centers.begin(), wall_centers.end(), wall_center));
//                    qInfo() << "Wall center index: " << wall_center_index;
                    doors.push_back({wall_center_index, Eigen::Vector2f(p0_projected.x(), p0_projected.y()), Eigen::Vector2f(p1_projected.x(), p1_projected.y())});

                }
            }
        }
    /// Check if door exists between the last and the first point
    auto p0 = ldata.points[in_wall_indexes.back()];
    auto p1 = ldata.points[in_wall_indexes.front()];
    auto line = Eigen::Vector2f(p1.x - p0.x, p1.y - p0.y);
    auto line_norm = line.norm();
    if(line_norm > 700 and line_norm < 1500)
    {
        bool is_door = true;
        for(int i = in_wall_indexes.back(); i < in_wall_indexes.front(); i++)
        {
            if(not outside_poly_in[i])
            {
                is_door = false;
                break;
            }
        }
        if(is_door)
        {
            auto p0_projected = projectPointOnPolygon(QPointF(p0.x, p0.y), nominal_polygon);
            auto p1_projected = projectPointOnPolygon(QPointF(p1.x, p1.y), nominal_polygon);
            auto p0_projected_eigen = Eigen::Vector2f(p0_projected.x(), p0_projected.y());
            auto p1_projected_eigen = Eigen::Vector2f(p1_projected.x(), p1_projected.y());
            auto middle = (p0_projected_eigen + p1_projected_eigen) / 2;
            auto wall_center = *std::min_element(wall_centers.begin(), wall_centers.end(), [&middle](const Eigen::Vector2f &a, const Eigen::Vector2f &b){ return (a - middle).norm() < (b - middle).norm(); });
            auto wall_center_index = std::distance(wall_centers.begin(), std::find(wall_centers.begin(), wall_centers.end(), wall_center));
            doors.push_back({wall_center_index, Eigen::Vector2f(p0_projected.x(), p0_projected.y()), Eigen::Vector2f(p1_projected.x(), p1_projected.y())});
        }
    }
    draw_polygon(poly_room_in, poly_room_out, &widget_2d->scene, QColor("blue"));
    return doors;
}
//Ceate funcion to build .g2o string from corner_data, odometry_data, RT matrix
std::string SpecificWorker::build_g2o_graph(const std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> &measured_corner_data,
                                            const std::vector<Eigen::Matrix<float, 2, 1>> &nominal_corner_data,
                                            const std::vector<std::vector<float>> &odometry_data,
                                            const Eigen::Affine2d &robot_pose ,
                                            const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> &measured_door_vertices,
                                            const std::pair<Eigen::Vector2f, Eigen::Vector2f> &nominal_door_vertices)
{

    std::string g2o_graph;
    int id = 0; // Id for g2o graph vertices
    auto updated_robot_pose = robot_pose;

    /// Add nominal corners as VERTEX_XY (ROOM), FIXED using nominal_corners_data in
    for (size_t i = 0; i < nominal_corner_data.size(); ++i)
    {
        g2o_graph += "VERTEX_XY " + std::to_string(id) + " " + std::to_string(nominal_corner_data[i].x()) + " " + std::to_string(nominal_corner_data[i].y()) + "\n";
        g2o_graph += "FIX " + std::to_string(id) + "\n";
        id++;
    }

    /// ADD NOMINAL DOOR VERTEX CALCULATED FROM WIDTH AND DISTANCE (ROOM COORDINATES), NOT FIXED using nominal_door_vertices
    g2o_graph += "VERTEX_XY " + std::to_string(id) + " " + std::to_string(nominal_door_vertices.first.x()) + " " + std::to_string(nominal_door_vertices.first.y()) + "\n";
    auto v1_id = id;
    id++;
    g2o_graph += "VERTEX_XY " + std::to_string(id) + " " + std::to_string(nominal_door_vertices.second.x()) + " " + std::to_string(nominal_door_vertices.second.y()) + "\n";
    auto v2_id = id;
    id++;

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

    //Return to initial ID
    id = node_id;

    /// INSERT LANDMARKS TO ROOM CORNERS, using measured_corner_data
    for (size_t i = 0; i < measured_corner_data.size()-1; ++i)
    {
        for (size_t j = 0; j < measured_corner_data[i].size(); ++j)
        {
            g2o_graph += "EDGE_SE2_XY " + std::to_string(id) + " " + std::to_string(j) + " " + std::to_string(measured_corner_data[i][j].x()) + " " + std::to_string(measured_corner_data[i][j].y()) + " 5 0 5 \n";
        }
        id++;
    }

    //Return to initial ID
    id = node_id;
    // INSERT LANDMARKS TO DOOR VERTICES
    for (size_t i = 0; i < measured_door_vertices.size()-1; ++i)
    {
        g2o_graph += "EDGE_SE2_XY " + std::to_string(id) + " " + std::to_string(v1_id)  + " " + std::to_string(measured_door_vertices[i].first.x()) + " " + std::to_string(measured_door_vertices[i].first.y()) + " 5 0 5 \n";
        g2o_graph += "EDGE_SE2_XY " + std::to_string(id) + " " + std::to_string(v2_id)  + " " + std::to_string(measured_door_vertices[i].second.x()) + " " + std::to_string(measured_door_vertices[i].second.y()) + " 5 0 5 \n";
        id++;
    }

    return g2o_graph;
}
SpecificWorker::Lines SpecificWorker::extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges)
{
    Lines lines(ranges.size());
    for(const auto &p: points)
        for(const auto &[i, r] : ranges | iter::enumerate)
            if(p.z > r.first and p.z < r.second)
                lines[i].emplace_back(p.x, p.y);
    return lines;
}
DSR::Node SpecificWorker::insert_nominal_door_into_graph(const DoorDetector::Door &door, int wall_id)
{   DSR::Node nominal_door_node;

    /// Check if room node exists
    auto room_node_ = G->get_node("room");
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room node in graph"; return nominal_door_node; }
    auto room_node = room_node_.value();

    auto wall_node_ = G->get_node("wall_"+std::to_string(wall_id));
    if(not wall_node_.has_value())
    { qWarning() << __FUNCTION__ << " No wall node in graph"; return nominal_door_node; }
    auto wall_node = wall_node_.value();

    auto wall_node_level_ = G->get_node_level(wall_node);
    if(not wall_node_level_.has_value())
    { qWarning() << __FUNCTION__ << " No wall level in graph"; return nominal_door_node; }
    auto wall_node_level = wall_node_level_.value();


    std::vector<float> door_pos, door_orientation;

    nominal_door_node = DSR::Node::create<door_node_type>("door_"+std::to_string(wall_id)+"_"+std::to_string(door.id));
    // Add door attributes
    G->add_or_modify_attrib_local<pos_x_att>(nominal_door_node, (float)(rand()%(170)));
    G->add_or_modify_attrib_local<pos_y_att>(nominal_door_node, (float)(rand()%170));
    G->add_or_modify_attrib_local<width_att>(nominal_door_node, (int)door.width());
    G->add_or_modify_attrib_local<height_att>(nominal_door_node, (int)door.height());
    G->add_or_modify_attrib_local<level_att>(nominal_door_node, wall_node_level + 1);
    G->insert_node(nominal_door_node);

    // Generate Eigen::Vector3f for door position
    Eigen::Vector3f door_pos_point {door.middle.x(), door.middle.y(), 0.f};
    auto door_robot_pos_point_double = door_pos_point.cast<double>();
    if(auto door_transformed = inner_eigen->transform(wall_node.name(), door_robot_pos_point_double, room_node.name()); door_transformed.has_value())
    {
        auto door_transformed_value = door_transformed.value();
        auto door_transformed_value_float = door_transformed_value.cast<float>();
        door_pos = {door_transformed_value_float.x(), door_transformed_value_float.y(), 0.f};
    }

    // Add edge between door and robot
    door_orientation = {0.f, 0.f, 0.f};
    rt->insert_or_assign_edge_RT(wall_node, nominal_door_node.id(), door_pos, door_orientation);

    return nominal_door_node;
}

void SpecificWorker::insert_measured_door_into_graph(const DoorDetector::Door &door, int wall_id)
{
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    auto robot_node_level_ = G->get_node_level(robot_node);
    if(not robot_node_level_.has_value())
    { qWarning() << __FUNCTION__ << " No robot level in graph"; return; }
    auto robot_node_level = robot_node_level_.value();

    auto wall_node_ = G->get_node("wall_"+std::to_string(wall_id));
    if(not wall_node_.has_value())
    { qWarning() << __FUNCTION__ << " No wall node in graph"; return; }
    auto wall_node = wall_node_.value();

    auto wall_node_level_ = G->get_node_level(wall_node);
    if(not wall_node_level_.has_value())
    { qWarning() << __FUNCTION__ << " No wall level in graph"; return; }
    auto wall_node_level = wall_node_level_.value();

    std::vector<float> door_pos, door_orientation;

    auto door_node = DSR::Node::create<door_node_type>("door_"+std::to_string(wall_id)+"_"+std::to_string(door.id)+"_measured");
    // Add door attributes
    G->add_or_modify_attrib_local<pos_x_att>(door_node, (float)(rand()%(170)));
    G->add_or_modify_attrib_local<pos_y_att>(door_node, (float)(rand()%170));
    G->add_or_modify_attrib_local<width_att>(door_node, (int)door.width());
    G->add_or_modify_attrib_local<height_att>(door_node, (int)door.height());
    G->add_or_modify_attrib_local<level_att>(door_node, robot_node_level + 1);
    G->insert_node(door_node);

    // Add edge between door and robot
    door_pos = {door.middle.x(), door.middle.y(), 0.f};
    door_orientation = {0.f, 0.f, door.angle_to_robot()};
    rt->insert_or_assign_edge_RT(robot_node, door_node.id(), door_pos, door_orientation);
}
void SpecificWorker::update_door_in_graph(const Eigen::Vector2f &pose, std::string door_name)
{
    auto door_node_ = G->get_node(door_name);
    if(not door_node_.has_value())
    { qWarning() << __FUNCTION__ << " No door node in graph"; return; }
    auto door_node = door_node_.value();

    /// Get robot node
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    std::vector<float> door_pos, door_orientation;
    // Add edge between door and robot
    door_pos = {pose.x(), pose.y(), 0.f};
    door_orientation = {0.f, 0.f, 0.f};
    rt->insert_or_assign_edge_RT(robot_node, door_node.id(), door_pos, door_orientation);
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
std::vector<float> SpecificWorker::calculate_speed(const Eigen::Matrix<double, 3, 1> &target)
{
    // Calculate the angle between the robot and the target
    Eigen::Matrix<float, 2, 1> robot_pos_vector = {0.0, 0.0};
    Eigen::Matrix<float, 2, 1> target_vector = Eigen::Matrix<float, 2, 1>(target(0), target(1));
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
    return {static_cast<float>(advance_speed * 0.6), static_cast<float>(side_speed  * 0.6), static_cast<float>(rotation_speed  * 0.4)};
}
bool SpecificWorker::movement_completed(const Eigen::Vector3d &target, float distance_to_target)
{
    return (target.norm() < distance_to_target);
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

//////////////////// LIDAR /////////////////////////////////////////////////
void SpecificWorker::read_lidar()
{
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarData(consts.lidar_name, -90, 360, 3);
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D " << e << std::endl; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));}
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
// Función para calcular la proyección de un punto sobre un segmento de línea
QPointF SpecificWorker::projectPointOnLineSegment(const QPointF& p, const QPointF& v, const QPointF& w) {
    // Vector v -> w
    QPointF vw = w - v;
    // Vector v -> p
    QPointF vp = p - v;
    // Proyección escalar de vp sobre vw
    double lengthSquared = vw.x() * vw.x() + vw.y() * vw.y();
    double t = (vp.x() * vw.x() + vp.y() * vw.y()) / lengthSquared;

    // Clamping t to the range [0, 1]
    t = std::max(0.0, std::min(1.0, t));

    // Punto proyectado
    QPointF projection = v + t * vw;
    return projection;
}

// Función para calcular la proyección más cercana de un punto sobre un polígono
Eigen::Vector2f SpecificWorker::projectPointOnPolygon(const QPointF& p, const QPolygonF& polygon) {
    QPointF closestProjection;
    double minDistanceSquared = std::numeric_limits<double>::max();

    for (int i = 0; i < polygon.size(); ++i) {
        QPointF v = polygon[i];
        QPointF w = polygon[(i + 1) % polygon.size()]; // Ciclar al primer punto

        QPointF projection = projectPointOnLineSegment(p, v, w);
        double distanceSquared = std::pow(projection.x() - p.x(), 2) + std::pow(projection.y() - p.y(), 2);

        if (distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
            closestProjection = projection;
        }
    }
    // Convertir a Eigen::Vector2f
    Eigen::Vector2f closestProjectionEigen(closestProjection.x(), closestProjection.y());
    return closestProjectionEigen;
}
void SpecificWorker::draw_door(const std::vector<std::tuple<int, Eigen::Vector2f, Eigen::Vector2f>> doors, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    //Draw ellipse in the doors corners
    for(const auto &[id, p0, p1]: doors)
    {
        auto item = scene->addEllipse(-40, -40, 80, 80, QPen(QColor("red"), 40), QBrush(QColor("red")));
        item->setPos(p0.x(), p0.y());
        items.push_back(item);
        auto item2 = scene->addEllipse(-40, -40, 80, 80, QPen(QColor("red"), 40), QBrush(QColor("red")));
        item2->setPos(p1.x(), p1.y());
        items.push_back(item2);
    }
    // draw points inside doors
    for(const auto &[id, p0, p1]: doors)
    {
        auto item = scene->addLine(p0.x(), p0.y(), p1.x(), p1.y(), QPen(QColor("red"), 40));
        items.push_back(item);
    }
}
void SpecificWorker::draw_graph_doors(const std::vector<std::tuple<int, Eigen::Vector2f, Eigen::Vector2f>> doors, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    //Draw ellipse in the doors corners
    for(const auto &[id, p0, p1]: doors)
    {
        auto item = scene->addEllipse(-40, -40, 80, 80, QPen(QColor("yellow"), 40), QBrush(QColor("yellow")));
        item->setPos(p0.x(), p0.y());
        items.push_back(item);
        auto item2 = scene->addEllipse(-40, -40, 80, 80, QPen(QColor("yellow"), 40), QBrush(QColor("yellow")));
        item2->setPos(p1.x(), p1.y());
        items.push_back(item2);
    }
    // draw points inside doors
    for(const auto &[id, p0, p1]: doors)
    {
        auto item = scene->addLine(p0.x(), p0.y(), p1.x(), p1.y(), QPen(QColor("red"), 40));
        items.push_back(item);
    }
}
// Create funcion to draw QPolygonF on 2dwidget
void SpecificWorker::draw_polygon(const QPolygonF &poly_in, const QPolygonF &poly_out,QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &p: poly_in)
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(color, 20), QBrush(color));
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
    // Draw lines between corners
    for(int i = 0; i < poly_in.size(); i++)
    {
        auto line = scene->addLine(poly_in[i].x(), poly_in[i].y(), poly_in[(i+1)%poly_in.size()].x(), poly_in[(i+1)%poly_in.size()].y(), QPen(color, 20));
        items.push_back(line);
    }

    // draw points
    for(const auto &p: poly_out)
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(color, 20), QBrush(color));
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
    // Draw lines between corners
    for(int i = 0; i < poly_out.size(); i++)
    {
        auto line = scene->addLine(poly_out[i].x(), poly_out[i].y(), poly_out[(i+1)%poly_out.size()].x(), poly_out[(i+1)%poly_out.size()].y(), QPen(color, 20));
        items.push_back(line);
    }
}
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

std::vector<Eigen::Vector2f> SpecificWorker::extract_g2o_data(string optimization)
{



// Print function name
    std::cout << __FUNCTION__ << std::endl;

    std::istringstream iss(optimization);
    // Find VERTEX_XY with ids 1, 2, 3, 4
    std::vector<Eigen::Vector2f> door_vertices;
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
        /// Print type, id, x, y
        std::cout << type << " " << id << " " << x << " " << y << std::endl;
        if(type == "VERTEX_XY")
        {
            qInfo() << "Vertex XY";
            if(id == 4 or id == 5)
            {
                door_vertices.push_back(Eigen::Vector2f{x, y});
            }
        }


        else if(type == "FIX")
            // pass to next line
            qInfo() << "Fixed node";
        else
        {
            /// calculate distance between door vertices
            auto distance = (door_vertices[0] - door_vertices[1]).norm();
            qInfo() << "Door distance: " << distance;
            /// Print door vertices
            qInfo() << "Door vertices: " << door_vertices[0].x() << " " << door_vertices[0].y() << " " << door_vertices[1].x() << " " << door_vertices[1].y();
            return door_vertices;
        }
    }

}


void SpecificWorker::asociate_and_update_doors(const vector<tuple<int, Eigen::Vector2f, Eigen::Vector2f>> &doors,
                                               const vector<DSR::Node> &door_nodes,
                                               const vector<Eigen::Vector2f> &door_poses)
{
    vector<vector<double>> distances_matrix(doors.size(), vector<double>(door_nodes.size()));

    for(size_t i = 0; i < doors.size(); ++i)
    {
        Eigen::Vector2f target_point = (std::get<1>(doors[i]) + std::get<2>(doors[i])) / 2;
        qInfo() << "Door measured: " << target_point.x() << " " << target_point.y();
        for(size_t j = 0; j < door_nodes.size(); ++j)
        {
            qInfo() << "Door nominal transformed: " << door_poses[j].x() << " " << door_poses[j].y();
            distances_matrix[i][j] = (door_poses[j] - target_point).norm();
        }
    }
    /// Process metrics matrix with Hungarian algorithm
    vector<int> assignment;
    double cost = HungAlgo.Solve(distances_matrix, assignment);
    /// Get the index in assignment vector with a value different from -1
    auto door_index = find_if(assignment.begin(), assignment.end(), [](auto &a){ return a != -1; });
    /// Check if distance is greater than threshold
    if(distances_matrix[door_index - assignment.begin()][0] < door_center_matching_threshold)
    {
        auto associated_door = doors[door_index - assignment.begin()];
        /// Update door data in graph
        update_door_in_graph(std::get<1>(associated_door), door_nodes[door_index - assignment.begin()].name()+"_measured");
    }
}


vector<Eigen::Vector2f> SpecificWorker::get_nominal_door_from_dsr(DSR::Node &robot_node, vector<DSR::Node> &door_nodes)
{
    vector<Eigen::Vector2f> door_poses;
    for(auto door_node : door_nodes)
    {
        if(auto parent_node = G->get_parent_node(door_node); parent_node.has_value())
        {
            auto wall_node = parent_node.value();
            if(auto rt_door = rt->get_edge_RT(wall_node, door_node.id()); rt_door.has_value())
            {
                auto door_edge = rt_door.value();
                if(auto rt_translation = G->get_attrib_by_name<rt_translation_att>(door_edge); rt_translation.has_value())
                {
                    auto rt_door_value = rt_translation.value().get();
                    Eigen::Vector3d door_robot_pos_point(rt_door_value[0], rt_door_value[1], 0);
                    if(auto door_transformed = inner_eigen->transform(robot_node.name(), door_robot_pos_point, wall_node.name()); door_transformed.has_value())
                    {
                        auto door_transformed_value = door_transformed.value().cast<float>();
                        door_poses.push_back(Eigen::Vector2f{door_transformed_value.x(), door_transformed_value.y()});
                    }
                }
            }
        }
    }
    return door_poses;
}
void SpecificWorker::generate_edge_goto_door(DSR::Node &robot_node, DSR::Node &door_node)
{
    DSR::Edge goto_door_edge = DSR::Edge::create<goto_action_edge_type>(robot_node.id(), door_node.id());
    if(G->insert_or_assign_edge(goto_door_edge))
        std::cout << __FUNCTION__ << " Goto door edge successfully inserted: " << std::endl;
    else
        std::cout << __FUNCTION__ << " Fatal error inserting new edge: " << std::endl;
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

