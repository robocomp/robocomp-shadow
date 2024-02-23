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
#include "Gridder.h"

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
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
		connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

        // Init RT API
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

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/
		//graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);

        //Find collision edges and put them in the buffer
        if(auto collision_edges = G->get_edges_by_type("collision"); not collision_edges.empty())
        {
            buffer_collision_edge.put(std::move(collision_edges));
        }

		timer.start(Period);
	}

}

void SpecificWorker::compute()
{
    static RoboCompGridder::TPoint target_pos;
    static unsigned long target_id;
    std::tuple<float, float, float, float>person_pose, target_pose;
    auto collision_flag = false;

    //get timestamp
    auto timestamp = std::chrono::system_clock::now();

    // check if there is a new intention edge or a change in existing  intention edges
    if(auto collisions = buffer_collision_edge.try_get(); collisions.has_value())
    {
        //For collision in collisions
        auto collision = collisions.value().front(); //TODO: check if there are more than one collision? if yes, what to do?
        // if there is an intention edge marked as dangerous
        // check if there is an action that can be taken to avoid the danger
        if(auto robot_node_ = G->get_node("Shadow"); robot_node_.has_value()) //TODO: move to global variable?
        {
            auto robot_node = robot_node_.value();

            //Get person node (collision parent)
            auto person_node_id = collision.from();
            if(auto person_pose_ = get_rt_data(robot_node, person_node_id); person_pose_.has_value())
                person_pose = person_pose_.value();

            //Get person intention target
            if(auto intention_edges = G->get_edges_by_type("has_intention"); not intention_edges.empty())
            {
                for (auto &&e : iter::filter([person_node_id](auto &e)
                                                 { return e.from() == person_node_id; }, intention_edges))
                {

                   if(auto target_rt = get_rt_data(robot_node, e.to()); target_rt.has_value())
                   {
                       target_pose = target_rt.value();

                       target_pos.x = std::get<0>(target_pose);
                       target_pos.y = std::get<1>(target_pose);
                       target_pos.radius = 500;  //TODO: move to global variable?
                       target_id = e.to();
                       qInfo()<<"Intention edge from person node found"<< target_pos.x << target_pos.y;
                   }
                   else{return;}
                }
            }
            else{return;}

            // Get the list of possible robot actions from the list of visual elements, excluding the robot and  the person
            auto object_nodes = G->get_nodes_by_type("object");
            // for each action and target object associated

            RoboCompGridder::TPointVector visible_objects;
            RoboCompGridder::TPointVector non_visible_objects;

            //Get object pose and check if it is inside the visual cone of the person
            for(const auto &object : object_nodes)
            {
//                qInfo() << "Object =" << QString::fromStdString(object.name());
                // Get the RT between the robot and the object
                if (object.id() != target_id)
                {
                    if (auto object_pose_ = get_rt_data(robot_node, object.id()); object_pose_.has_value())
                    {
                        auto object_pose = object_pose_.value();

                        float distance_to_obstacle = (
                                Eigen::Vector3f{std::get<0>(target_pose), std::get<1>(target_pose),
                                                std::get<2>(target_pose)} -
                                Eigen::Vector3f{std::get<0>(person_pose), std::get<1>(person_pose),
                                                std::get<2>(person_pose)}).norm();

                        auto in_cone = element_inside_cone(
                                Eigen::Vector3f{std::get<0>(object_pose), std::get<1>(object_pose),
                                                std::get<2>(object_pose)},
                                Eigen::Vector3f{std::get<0>(target_pose), std::get<1>(target_pose),
                                                std::get<2>(target_pose)},
                                Eigen::Vector3f{std::get<0>(person_pose), std::get<1>(person_pose),
                                                std::get<2>(person_pose)},
                                distance_to_obstacle / 2.5);

                        if (in_cone)
                        {
                            visible_objects.push_back(
                                    RoboCompGridder::TPoint{std::get<0>(object_pose), std::get<1>(object_pose), 300});
//                            qInfo() << "Object inside the cone" << QString::fromStdString(object.name());
                        } else
                        {
                            non_visible_objects.push_back(
                                    RoboCompGridder::TPoint{std::get<0>(object_pose), std::get<1>(object_pose), 300});
//                            qInfo() << "Object OUTSIDE the cone" << QString::fromStdString(object.name());
                        }
                    }
                }
            }

            //Create object vector
            std::vector<RoboCompGridder::TPoint> objects;
            // Concat obstacles inside and outside the cone using std::move
            objects.insert(objects.end(), std::make_move_iterator(visible_objects.begin()),
                             std::make_move_iterator(visible_objects.end()));
            objects.insert(objects.end(), std::make_move_iterator(non_visible_objects.begin()),
                             std::make_move_iterator(non_visible_objects.end()));

//            qInfo() << "////////////////////////////////////////////////////////////////////////////////////////////////////////////";
//            qInfo() << "............................................................................................................";
//            qInfo() << "............................................................................................................";
//            //Print visible objects, non visible objects
//            for (auto &p : visible_objects)
//            {
//                qInfo() << "Visible object: " << p.x << " " << p.y;
//            }
//            for (auto &p : non_visible_objects)
//            {
//                qInfo() << "Non visible object: " << p.x << " " << p.y;
//            }
//            for (auto &p : objects)
//            {
//                qInfo() << "Object: " << p.x << " " << p.y;
//            }
//            qInfo() << "............................................................................................................";
//            qInfo() << "............................................................................................................";
//
            for(const auto &object : objects)
            {
                visible_objects.emplace_back(object);
                try
                {
                    auto path = gridder_proxy->setLocationAndGetPath(RoboCompGridder::TPoint{
                            std::get<0>(person_pose) //TODO: change method to std::vector<TPoint>
                            , std::get<1>(person_pose), 300}, target_pos, non_visible_objects, visible_objects );

                    //Print visible objects, non visible objects
                    for (auto &p : visible_objects)
                    {
                        qInfo() << "Visible object: " << p.x << " " << p.y;
                    }
                    for (auto &p : non_visible_objects)
                    {
                        qInfo() << "Non visible object: " << p.x << " " << p.y;
                    }

                    if (path.paths.size() > 0)
                    {
                        auto sim_result = bulletsim_proxy->simulatePath(path.paths[0], 1,
                                                                        objects);
                        //print sim result point
//                        qInfo()<< "COLLISION POINT" << sim_result.collision << sim_result.collisionPose.x << " " << sim_result.collisionPose.y;
                        if (!sim_result.collision)
                        {
                            qInfo() << "-----------------GOING TO TARGET-----------------------";
                            //Create node intention to avoid collision:
                            //Get robot level
                            auto robot_level_ = G->get_node_level(robot_node);
                            if(not robot_level_.has_value())
                            { qWarning() << __FUNCTION__ << " No robot level in graph"; return; }
                            auto robot_level = robot_level_.value();
                            //Create intention node and add attributes
                            DSR::Node intention_node = DSR::Node::create<intention_node_type>("avoid_collision");
                            auto pos_x = static_cast<float>(rand()%170);
                            auto pos_y = static_cast<float>(rand()%170);


                            G->add_or_modify_attrib_local<pos_x_att>(intention_node, pos_x);
                            G->add_or_modify_attrib_local<pos_y_att>(intention_node, pos_y);
                            G->add_or_modify_attrib_local<level_att>(intention_node, robot_level + 1);
                            //Set object position attributes
                            G->add_or_modify_attrib_local<robot_target_x_att>(intention_node, object.x);
                            G->add_or_modify_attrib_local<robot_target_y_att>(intention_node, object.y);

                            //Insert intention node and edge
                            try{
                                G->insert_node(intention_node);
                                qInfo() << "Intention node created";
                                insert_edge(robot_node.id(), intention_node.id(), "go_to_action");
                                qInfo() << "Edge go_to_action created";
                            }
                            catch(const std::exception &e)
                            { std::cout << e.what() << " Error inserting node" << std::endl;}
                        }
                    }
                }
                catch(const Ice::Exception &e){
                    std::cout << "Error reading from Gridder" << std::endl;
                }
                visible_objects.pop_back();
                //get and print diff time using timestamp
                auto diff = std::chrono::system_clock::now() - timestamp;
                std::cout << "Time difference = " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
            }
        }
        else
        {
            return;
        }
    }
}
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}
///////////////////FUNCTIONS////////////////////////

/////Returns the RT data of a node (x, y, z, theta)
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

bool SpecificWorker::element_inside_cone(const Eigen::Vector3f& point,
                                         const Eigen::Vector3f& basePoint,
                                         const Eigen::Vector3f& apexPoint,
                                         double radius) {
//    std::cout << "Cone parameters" << point.transpose() << "|" << basePoint.transpose()<< "|" << apexPoint.transpose() << "|" << radius << std::endl;
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

[[maybe_unused]] void SpecificWorker::insert_edge(uint64_t from, uint64_t to, const std::string &edge_tag)
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
////////////////////////// SLOTS //////////////////////////
void SpecificWorker::modify_node_slot(std::uint64_t, const std::string &type)
{

};
void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{

};
void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    if(type == "collision")
    {
        //Find collision edges and put them in the buffer
        if(auto collision_edges = G->get_edges_by_type("collision"); not collision_edges.empty())
        {
            buffer_collision_edge.put(std::move(collision_edges));
        }
    }
};
void SpecificWorker::modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names)
{

};
void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{

};
void SpecificWorker::del_node_slot(std::uint64_t from)
{

};



