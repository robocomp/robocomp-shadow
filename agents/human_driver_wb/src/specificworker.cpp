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
//	QLoggingCategory::setFilterRules("*.debug=false\n");
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
	{
		G->delete_node(grid);
	}
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

        //Print initialize
        std::cout << "Initialize worker" << std::endl;

		timer.start(Period);
	}

}

void SpecificWorker::compute()
{    // Check if robot node exists
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
                //Print person ID
                std::cout << "Person ID: " << person.id() << std::endl;

                auto [x, y, z, ang] = person_rt_data.value();
                // Check if person cone exists
                auto person_cone_it = std::find_if(person_cones.begin(), person_cones.end(), [&person](const PersonCone &person_cone)
                {return person_cone.get_dsr_id() == person.id();});
                // Create person cone if not exists
                if(person_cone_it == person_cones.end())
                {
                    std::cout << "1 " << person.id() << std::endl;
                    // Create person cone
                    PersonCone person_cone(gridder_proxy);
                    std::cout << "1.1 " << person.id() << std::endl;
                    person_cone.set_dsr_id(person.id());
                    std::cout << "1.2 " << person.id() << std::endl;
                    person_cone.init_item(&widget_2d->scene, x, y, ang, cone_radius, cone_angle, person.name());
                    std::cout << "1.3 " << person.id() << std::endl;
                    person_cones.emplace_back(person_cone);
                    // Assign person cone iterator
                    std::cout << "1.4 " << person.id() << std::endl;
                    person_cone_it = std::prev(person_cones.end());
                }
                    // Update cone attributes
                else
                {
                    std::cout << "2 " << person.id() << std::endl;
                    person_cone_it->update_attributes(x, y, ang);
                    std::cout << "2.1" << person.id() << std::endl;
                    person_cone_it->remove_intentions(&widget_2d->scene, chair_nodes);
                }

                // Map Pilar cone to parent rotation and translation
                auto pilar_cone_ = person_cone_it->get_pilar_cone();
                std::cout << "3" << person.id() << std::endl;
                auto pilar_cone_conv = person_cone_it->get_item()->mapToParent(pilar_cone_);
                std::cout << "Cones resolved" << std::endl;
                //TODO: Move variable

                // Create empty gridder Tpoint target
                RoboCompGridder::TPoint target;

                // Check if there are intention edge between person and any object
                for(const auto &chair : chair_nodes)
                {
                    if(auto intention = G->get_edge(person.id(), chair.id(), "has_intention"); intention.has_value())
                    {
                        // Get intention attributes
                        auto intention_ = intention.value();
                        if(auto intention_rt_data = get_rt_data(robot_node_, intention_.to()); intention_rt_data.has_value())
                        {
                            auto [x_ch, y_ch, z_ch, ang_ch] = intention_rt_data.value();
                            //Print intention rt data
                            std::cout << "Intention rt data: " << x_ch << " " << y_ch << " " << z_ch << " " << ang_ch << std::endl;
                            // Check if object is inside the pilar cone

                            std::cout << "Person target found" << std::endl;
                            //Assign object as target
                            target.x = x_ch;
                            target.y = y_ch;
                            target.radius = 500;

                            std::cout << "Getting obstacles" << std::endl;
                            // Create vectors for inserting obstacles inside and outside the cone
                            RoboCompGridder::TPointVector obstacles_inside;
                            RoboCompGridder::TPointVector obstacles_outside;
                            RoboCompGridder::TPointVector obstacles;


                            // Generate a pyramid cone to each object
                            for (const auto &object: chair_nodes)
                            {
                                // Check if object is an obstacle
                                if (auto is_an_obstacle = G->get_attrib_by_name<is_an_obstacle_att>(object);
                                        is_an_obstacle.has_value() and is_an_obstacle.value())
                                {
                                    // Check if object isn't in the same position
                                    if (object.id() != intention_.to())
                                    {
                                        // Get object pose
                                        if (auto object_rt_data = get_rt_data(robot_node_,
                                                                              object.id()); object_rt_data.has_value())
                                        {

                                            auto [x_obj, y_obj, z_obj, ang_obj] = object_rt_data.value();
                                            RoboCompGridder::TPoint obstacle{.x=x_obj, .y=y_obj, .radius=250};
                                            float distance_to_obstacle = (Eigen::Vector3f{x, y, z} -
                                                                          Eigen::Vector3f{x_ch, y_ch, z_ch}).norm();
                                            // Check if object is inside the cone
                                            auto in_cone = element_inside_cone(Eigen::Vector3f{x_obj, y_obj, z_obj},
                                                                               Eigen::Vector3f{x_ch, y_ch, z_ch},
                                                                               Eigen::Vector3f{x, y, z * 1.9},
                                                                               distance_to_obstacle / 5);
                                            if (!in_cone)
                                            {
                                                obstacles_outside.push_back(obstacle);
                                                //Print "in cone"
                                                qInfo() << "Obstacle outside: " << x_obj << " " << y_obj << " " << z_obj;
                                            }
                                            else
                                            {
                                                obstacles_inside.push_back(obstacle);
                                                qInfo() << "Obstacle inside: " << x_obj << " " << y_obj << " " << z_obj;
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

                            // Print inside obstacles
                            for (const auto &o: obstacles_inside)
                            {
                                qInfo() << "inside: " << o.x << " " << o.y << o.radius;
                            }
                            // Print outside obstacles
                            for (const auto &o: obstacles_outside)
                            {
                                qInfo() << "outside: " << o.x << " " << o.y << o.radius;
                            }

                            // Once the obstacles are detected, simulate the path to the target considering seeing obstacles
                            //TODO: introduce bool, all obstacles visible or not

                            bool all_visible = false;

                            if (all_visible) // All objects visible
                            {
                                obstacles_outside = {};
                                obstacles_inside = obstacles;
                            } else // All objects are obstacles non visible
                            {
                                obstacles_outside = obstacles;
                                obstacles_inside = {};
                            }

                            try
                            {
                                auto path_to_target = gridder_proxy->setLocationAndGetPath(
                                        RoboCompGridder::TPoint{.x=x, .y=y, .radius=500},
                                        RoboCompGridder::TPoint{.x=x_ch, .y=y_ch, .radius=500},
                                        obstacles_outside, obstacles_inside);
                                if (path_to_target.valid and not path_to_target.paths.empty())
                                {
                                    //TODO: delete simulation, use weboots2robocomp proxy to set the path to the human

                                    try
                                    {
                                        webots2robocomp_proxy->setPathToHuman(0, path_to_target.paths[0]);
//                                        exit(0);
                                    }
                                    catch (const Ice::Exception &e)
                                    {
                                        qInfo() << "Error setting path";
                                    }
                                }
                                //                                    person_cone_it->draw_paths(&widget_2d->scene, false, path_to_target.paths[0]);
                            }
                            catch (const Ice::Exception &e)
                            {
                                qInfo() << "Error getting/setting path";
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}
//    // Remove person cones if not exists
//    for(auto it = person_cones.begin(); it != person_cones.end();)
//    {
//        if(auto person_node = G->get_node(it->get_dsr_id()); !person_node.has_value())
//        {
//            qInfo() << "Person dissappeared. Removing cone.";
//            // Remove intention items
//            // Iterate over intentions
//            it->remove_intentions(&widget_2d->scene, chair_nodes);
//            it->draw_paths(&widget_2d->scene, true, RoboCompGridder::TPath{});
//            it->remove_item(&widget_2d->scene);
//            person_cones.erase(it);
//        }
//        else{++it;}
//    }
//}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}
//-------------FUNCTIONS----------------

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
bool SpecificWorker::element_inside_cone(const Eigen::Vector3f& point,
                                         const Eigen::Vector3f& basePoint,
                                         const Eigen::Vector3f& apexPoint,
                                         double radius) {
    std::cout << "Cone parameters" << point.transpose() << "|" << basePoint.transpose()<< "|" << apexPoint.transpose() << "|" << radius << std::endl;
    // Calcular la altura del cono
    double height = (apexPoint - basePoint).norm();

    // Calcular el vector de dirección del cono
    Eigen::Vector3f direction = (apexPoint - basePoint).normalized();

    // Calcular la distancia del punto al vértice del cono
    Eigen::Vector3f vecToPoint = point - basePoint;
    double distanceToVertex = vecToPoint.dot(direction);

    // Calcular el radio del cono en la altura del punto
    double radiusAtHeight = radius * (1 - distanceToVertex / height);

    // Calcular la distancia horizontal del punto al eje del cono
    Eigen::Vector3f pointOnAxis = basePoint + direction * distanceToVertex;
    double horizontalDistance = (point - pointOnAxis).norm();

    // Verificar si el punto está dentro del cono
    return (horizontalDistance <= radiusAtHeight);
}
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

/**************************************/
// From the RoboCompWebots2Robocomp you can call this methods:
// this->webots2robocomp_proxy->setPathToHuman(...)

/**************************************/
// From the RoboCompWebots2Robocomp you can use this types:
// RoboCompWebots2Robocomp::Vector3
// RoboCompWebots2Robocomp::Quaternion

