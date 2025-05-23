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

}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
    // Create a locale with dot as the decimal separator
    std::locale dotLocale("C");
    // Store the current global locale
    std::locale originalLocale = std::locale::global(dotLocale);
	try
	{
		agent_name = params.at("agent_name").value;
		agent_id = stoi(params.at("agent_id").value);
		tree_view = params.at("tree_view").value == "true";
		graph_view = params.at("graph_view").value == "true";
		qscene_2d_view = params.at("2d_view").value == "true";
		osg_3d_view = params.at("3d_view").value == "true";
        cone_radius = std::stof(params["cone_radius"].value);
        cone_angle = std::stof(params["cone_angle"].value);
        qInfo() << "Params read from config: ";
        qInfo() << "    cone radius: " << cone_radius;
        qInfo() << "    cone angle: " << cone_angle;
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
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);
        rt = G->get_rt_api();
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
        widget_2d->setSceneRect(-5000, -5000,  10000, 10000);
        widget_2d->set_draw_axis(true);

        // mouse
        connect(widget_2d, &DSR::QScene2dViewer::mouse_left_click, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New click left arrived:" << p;

            // check person with the same item
//        if(auto person = std::ranges::find_if(people, [selectedItem](const Person &p)
//            { return p.get_item() == selectedItem; }); person != people.end())
//        {
//            qInfo() << "[MOUSE] Target selected";
//            person->set_target_element(true);
//        }
        });

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/
		//graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);

		timer.start(Period);
	}

}

void SpecificWorker::compute()
{
    // Check if robot node exists
    if(auto robot_node = G->get_node("Shadow"); robot_node.has_value())
    {
        auto robot_node_ = robot_node.value();
        // Get person nodes
        auto person_nodes = G->get_nodes_by_type("person");
        // Get object nodes
        auto chair_nodes = G->get_nodes_by_type("object");
        for(const auto &person : person_nodes)
        {
            if(auto person_rt_data = get_rt_data(robot_node_, person.id()); person_rt_data.has_value())
            {
                auto [x, y, z, ang] = person_rt_data.value();
                // Check if person cone exists
                auto person_cone_it = std::find_if(person_cones.begin(), person_cones.end(), [&person](const PersonCone &person_cone)
                {return person_cone.get_dsr_id() == person.id();});
                // Create person cone if not exists
                if(person_cone_it == person_cones.end())
                {
                    // Create person cone
                    PersonCone person_cone(gridder_proxy);
                    person_cone.set_dsr_id(person.id());
                    person_cone.init_item(&widget_2d->scene, x, y, ang, cone_radius, cone_angle, person.name());
                    person_cones.emplace_back(person_cone);
                    // Assign person cone iterator
                    person_cone_it = std::prev(person_cones.end());
                }
                // Update cone attributes
                else
                {
                    person_cone_it->update_attributes(x, y, ang);
                    person_cone_it->remove_intentions(&widget_2d->scene, chair_nodes);
                }
//                qInfo() << "Intentions number" << person_cone_it->get_act_intentions_number();
                // Map Pilar cone to parent rotation and translation
                auto pilar_cone_ = person_cone_it->get_pilar_cone();
                auto pilar_cone_conv = person_cone_it->get_item()->mapToParent(pilar_cone_);
                // Check if there are chairs and people in the cone
                for(const auto &chair : chair_nodes)
                {
                    // Get chair pose
                    if(auto chair_rt_data = get_rt_data(robot_node_, chair.id()); chair_rt_data.has_value())
                    {
                        auto [x_ch, y_ch, z_ch, ang_ch] = chair_rt_data.value();
                        // Check if object is inside the cone
                        auto is_in_cone = person_cone_it->is_inside_pilar_cone(pilar_cone_conv, x_ch, y_ch);
                        if(is_in_cone)
                        {
                            // If is in cone, check if element is an obstacle
                            if(auto is_an_obstacle = G->get_attrib_by_name<is_an_obstacle_att>(chair); is_an_obstacle.has_value())
                            {
                                if(!is_an_obstacle.value())
                                    insert_edge(person.id(), chair.id(), "has_intention");
                                if(auto intention = person_cone_it->get_intention(chair.name()); intention.has_value())
                                {
                                    auto intention_ = intention.value();
                                    if(!is_an_obstacle.value())
                                    {
                                        auto paths = person_cone_it->get_paths(person_rt_data.value(), chair_rt_data.value(), chair.name());
                                        if(paths.has_value()){intention_->update_path(paths.value());}
                                        else{qInfo() << "Path not found";}
                                    }
                                    intention_->update_attributes(x_ch, y_ch);
                                }
                                else
                                {
                                    person_cone_it->set_intention(person_rt_data.value(), chair_rt_data.value(), chair.name());
                                    if(auto intention = person_cone_it->get_intention(chair.name()); intention.has_value())
                                    {
                                        auto intention_ = intention.value();
                                        if(!is_an_obstacle.value())
                                        {
                                            auto paths = person_cone_it->get_paths(person_rt_data.value(), chair_rt_data.value(), chair.name());
                                            if(paths.has_value())
                                                intention_->update_path(paths.value());
                                            else{qInfo() << "Path not found";}
                                            intention_->init_item(&widget_2d->scene, x, y, 500, 500);
                                        }

                                        else
                                            intention_->init_item(&widget_2d->scene, x, y, 500, 500, true);
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(auto intention = person_cone_it->get_intention(chair.name()); intention.has_value())
                            {
                                qInfo() << "Removing intention";
                                auto intention_ = intention.value();
                                intention_->remove_item(&widget_2d->scene);
                                person_cone_it->remove_intention(chair.name());
                                delete_edge(person.id(), chair.id(), "has_intention");

                                //TODO: BORRAR delete edges y nodes, solo para pruebas
                                delete_edge(person.id(), chair.id(), "collision");
                                if(auto avoid_collision = G->get_nodes_by_type("intention"); avoid_collision.size() > 0)
                                {
                                    for(const auto &collision : avoid_collision)
                                    {
                                        //If edge has intention, delete it
                                        if (auto has_intention = G->get_edge(robot_node_.id(), collision.id(), "has_intention"); has_intention.has_value())
                                        {
                                            G->delete_edge(robot_node_.id(), collision.id(), "has_intention");
                                        }
                                        G->delete_node(collision.id());
                                    }
                                }


                                person_cone_it->draw_paths(&widget_2d->scene, true, RoboCompGridder::TPath{});
                            }
                        }
                        if(allow_prediction)
                        {
                            if (auto intention = person_cone_it->get_intention(chair.name()); intention.has_value())
                            {
                            // Create vectors for inserting obstacles inside and outside the cone
                                RoboCompGridder::TPointVector obstacles_inside;
                                RoboCompGridder::TPointVector obstacles_outside;
                                RoboCompGridder::TPointVector obstacles;

                                if(auto is_an_obstacle = G->get_attrib_by_name<is_an_obstacle_att>(chair); is_an_obstacle.has_value() and !is_an_obstacle.value())
                                {
                                    // Generate a pyramid cone to each object
                                    for (const auto &object: chair_nodes)
                                    {
                                        if (auto is_an_obstacle = G->get_attrib_by_name<is_an_obstacle_att>(object);
                                                is_an_obstacle.has_value() and is_an_obstacle.value())
                                                {
    //                                qInfo() << "Object" << QString::fromStdString(object.name()) << "is an obstacle";
                                            // Check if object isn't in the same position
                                            if (object.id() != chair.id()) {
                                                // Get object pose
                                                if (auto object_rt_data = get_rt_data(robot_node_,
                                                                                    object.id()); object_rt_data.has_value()) {
                                                    // Print object name
                                                    auto [x_obj, y_obj, z_obj, ang_obj] = object_rt_data.value();
                                                    RoboCompGridder::TPoint obstacle{.x=x_obj, .y=y_obj, .radius=250};
                                                    float distance_to_obstacle = (Eigen::Vector3f{x, y, z} -
                                                                                Eigen::Vector3f{x_ch, y_ch, z_ch}).norm();
                                                    // Check if object is inside the cone
    //                                        qInfo() << "Checking if object is inside the cone";
                                                    auto in_cone = element_inside_cone(Eigen::Vector3f{x_obj, y_obj, z_obj},
                                                                                    Eigen::Vector3f{x_ch, y_ch, z_ch},
                                                                                    Eigen::Vector3f{x, y, z *1.9},
                                                                                    distance_to_obstacle / 5);
                                                    if (!in_cone) {
    /*                                            qInfo() << "Object" << QString::fromStdString(object.name())
                                                        << "not seen in cone in path to"
                                                        << QString::fromStdString(chair.name());*/
                                                        if (auto intention = person_cone_it->get_intention(
                                                                    object.name()); intention.has_value()) {
                                                            auto intention_ = intention.value();
                                                            intention_->update_color(true);
                                                        }
                                                        obstacles_outside.push_back(obstacle);
                                                    } else {
    //                                            qInfo() << "Object" << QString::fromStdString(object.name()) << "seen in cone in path to" << QString::fromStdString(chair.name());
                                                        if (auto intention = person_cone_it->get_intention(object.name()); intention.has_value())
                                                        {
                                                            auto intention_ = intention.value();
                                                            intention_->update_color(false);
                                                        }
                                                        obstacles_inside.push_back(obstacle);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Concat obstacles inside and outside the cone usig std::move
                                    obstacles.insert(obstacles.end(), std::make_move_iterator(obstacles_inside.begin()),
                                                    std::make_move_iterator(obstacles_inside.end()));
                                    obstacles.insert(obstacles.end(), std::make_move_iterator(obstacles_outside.begin()),
                                                    std::make_move_iterator(obstacles_outside.end()));

//                                    // Print inside obstacles
//                                    for (const auto &o: obstacles_inside) {
//                                        qInfo() << "inside: " << o.x << " " << o.y << o.radius;
//                                    }
//                                    // Print outside obstacles
//                                    for (const auto &o: obstacles_outside) {
//                                        qInfo() << "outside: " << o.x << " " << o.y << o.radius;
//                                    }

                                    // Once the obstacles are detected, simulate the path to the target considering seeing obstacles
                                    try
                                    {
                                        auto path_to_target = gridder_proxy->setLocationAndGetPath(
                                                RoboCompGridder::TPoint{.x=x, .y=y, .radius=500},
                                                RoboCompGridder::TPoint{.x=x_ch, .y=y_ch, .radius=500},
                                                obstacles_outside, obstacles_inside);
                                        // Print path to target
                                        qInfo() << QString::fromStdString(path_to_target.errorMsg) << path_to_target.valid << path_to_target.paths.size();
                                        if (path_to_target.paths.size() > 0)
                                        {
                                            for (const auto &p: path_to_target.paths)
                                            {
                                                try
                                                {
                                                    auto sim_results = this->bulletsim_proxy->simulatePath(p, person_speed, obstacles);
                                                    if (sim_results.collision)
                                                    {
                                                        qInfo() << "Collision detected at" << sim_results.collisionTime << "seconds";
                                                        DSR::Edge edge = DSR::Edge::create<collision_edge_type>(
                                                                person.id(), chair.id());
                                                        G->add_or_modify_attrib_local<arrival_time_att>(edge, sim_results.collisionTime);
                                                        if (G->insert_or_assign_edge(edge))
                                                        {
                                                            std::cout << __FUNCTION__
                                                                    << " Edge successfully inserted: "
                                                                    << person.id() << "->" << chair.id()
                                                                    << " type: collision" << std::endl;
                                                            allow_prediction = false;
                                                        }
                                                        else
                                                        {
                                                            std::cout << __FUNCTION__
                                                                    << ": Fatal error inserting new edge: "
                                                                    << person.id() << "->" << chair.id()
                                                                    << " type: collision" << std::endl;
                                                        }
                                                    }
                                                }
                                                catch (const Ice::Exception &e) {
                                                    qInfo() << "Error simulating path";
                                                }
                                            }
                                            person_cone_it->draw_paths(&widget_2d->scene, false, path_to_target.paths[0]);
                                        }
                                    }
                                    catch (const Ice::Exception &e) {
                                        qInfo() << "Error simulating path";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Remove person cones if not exists
        for(auto it = person_cones.begin(); it != person_cones.end();)
        {
            if(auto person_node = G->get_node(it->get_dsr_id()); !person_node.has_value())
            {
                qInfo() << "Person dissappeared. Removing cone.";
                // Remove intention items
                // Iterate over intentions
                it->remove_intentions(&widget_2d->scene, chair_nodes, true);
                it->draw_paths(&widget_2d->scene, true, RoboCompGridder::TPath{});
                it->remove_item(&widget_2d->scene);
                person_cones.erase(it);
            }
            else{++it;}
        }
    }
}

bool SpecificWorker::element_inside_cone(const Eigen::Vector3f& point,
                                         const Eigen::Vector3f& basePoint,
                                         const Eigen::Vector3f& apexPoint,
                                         double radius) {
//    std::cout << "Cone parameters" << point.transpose() << "|" << basePoint.transpose()<< "|" << apexPoint.transpose() << "|" << radius << std::endl;
    double height = (apexPoint - basePoint).norm();
    Eigen::Vector3f direction = (apexPoint - basePoint).normalized();

    Eigen::Vector3f vecToPoint = point - basePoint;
    double distanceToVertex = vecToPoint.dot(direction);

    double radiusAtHeight = radius * (1 - distanceToVertex / height);

    Eigen::Vector3f pointOnAxis = basePoint + direction * distanceToVertex;
    double horizontalDistance = (point - pointOnAxis).norm();

    return (horizontalDistance <= radiusAtHeight);
}
std::optional<std::tuple<float, float, float, float>> SpecificWorker::get_rt_data(const DSR::Node &n, uint64_t to)
{
    // Get person pose
    if(auto robot_element_rt = rt->get_edge_RT(n, to); robot_element_rt.has_value())
    {
        auto robot_element_rt_ = robot_element_rt.value();
        if(auto element_pose = G->get_attrib_by_name<rt_translation_att >(robot_element_rt_); element_pose.has_value())
        {
            auto element_pose_ = element_pose.value();
            if(auto element_rotation = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(robot_element_rt_); element_rotation.has_value())
            {
                auto element_rotation_ = element_rotation.value();
                return std::make_tuple(element_pose_.get()[0], element_pose_.get()[1], element_pose_.get()[2], element_rotation_.get()[2]);
            }
            else{return std::make_tuple(element_pose_.get()[0], element_pose_.get()[1], element_pose_.get()[2], 0.f);}
        }
        else{return {};}
    }
    else{return {};}
}
void SpecificWorker::delete_edge(uint64_t from, uint64_t to, const std::string &edge_tag)
{
    if(auto has_edge = G->get_edge(from, to, edge_tag); has_edge.has_value())
    {
        if (G->delete_edge(from, to, edge_tag))
        {
            std::cout << __FUNCTION__ << " Edge successfully deleted: " << from << "->" << to
                      << " type: " << edge_tag << std::endl;
        }
        else
        {
            std::cout << __FUNCTION__ << ": Fatal error deleting edge: " << from << "->" << to
                      << " type: " << edge_tag << std::endl;
//            std::terminate();
        }
    }
}
void SpecificWorker::insert_edge(uint64_t from, uint64_t to, const std::string &edge_tag)
{
    if(auto has_edge = G->get_edge(from, to, edge_tag); !has_edge.has_value())
    {
        DSR::Edge edge = DSR::Edge::create<has_intention_edge_type>(from, to);
        if (G->insert_or_assign_edge(edge))
        {
            std::cout << __FUNCTION__ << " Edge successfully inserted: " << from << "->" << to
                      << " type: has" << std::endl;
        }
        else
        {
            std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << from << "->" << to
                      << " type: has" << std::endl;
//            std::terminate();

        }
    }
}
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if(type == "intention")
        if(auto intention_node = G->get_node(id); intention_node.has_value())
        {
            auto intention_node_ = intention_node.value();
            if(intention_node_.name() == "avoid_collision")
            {
                avoid_collision_node_id = id;
                allow_prediction = false;
            }
        }
}
void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{
    if(edge_tag == "collision")
        allow_prediction = true;

}
void SpecificWorker::del_node_slot(std::uint64_t from)
{
    if(from == avoid_collision_node_id)
        allow_prediction = true;
}
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}




/**************************************/
// From the RoboCompBulletSim you can call this methods:
// this->bulletsim_proxy->simulatePath(...)

/**************************************/
// From the RoboCompBulletSim you can use this types:
// RoboCompBulletSim::TPoint
// RoboCompBulletSim::Result

/**************************************/
// From the RoboCompGridder you can call this methods:
// this->gridder_proxy->IsPathBlocked(...)
// this->gridder_proxy->LineOfSightToTarget(...)
// this->gridder_proxy->getClosestFreePoint(...)
// this->gridder_proxy->getDimensions(...)
// this->gridder_proxy->getPaths(...)
// this->gridder_proxy->setGridDimensions(...)
// this->gridder_proxy->setLocationAndGetPath(...)

/**************************************/
// From the RoboCompGridder you can use this types:
// RoboCompGridder::TPoint
// RoboCompGridder::TDimensions
// RoboCompGridder::Result


