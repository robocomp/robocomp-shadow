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
#include <cppitertools/enumerate.hpp>
#include <cppitertools/filter.hpp>

//include opencv libraries
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    }
    catch(const std::exception &e){ std::cout << e.what() << " Error reading params from config file" << std::endl;};

    return true;
}
void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = 100;
	if(this->startup_check_flag)
		this->startup_check();
	else
	{
	// create graph
	G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
	qInfo() << __FUNCTION__ << "Graph loaded. Agent:" << agent_name.c_str() << "ID:" << agent_id << " from "  << "shadow_scene.json";

	//dsr update signals
	connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	//connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
	//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
	//connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
	//connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);
    rt = G->get_rt_api();

	// Graph viewer
	using opts = DSR::DSRViewer::view;
	int current_opts = 0;
	opts main = opts::none;
	if(tree_view)
        current_opts = current_opts | opts::tree;
	if(graph_view)
	    current_opts = current_opts | opts::graph;
	if(qscene_2d_view)
	    current_opts = current_opts | opts::scene;
    main = opts::graph;
	graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
	setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

	/***
	Custom Widget
	In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
	The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
	either with a QtDesigner or directly from scratch in a class of its own.
	The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
	***/

//    graph_viewer->add_custom_widget_to_dock("Intention_prediction", &custom_widget);
    widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));
    widget_2d->setSceneRect(-5000, -5000,  10000, 10000);
    widget_2d->set_draw_axis(true);

    // mouse
    connect(widget_2d, &DSR::QScene2dViewer::mouse_left_click, [this](QPointF p)
    {
        qInfo() << "[MOUSE] New click left arrived:" << p;

    });

    //Right click
    connect(widget_2d, &DSR::QScene2dViewer::mouse_right_click, [this](int x, int y, uint64_t id)
    {
        qInfo() << "[MOUSE] New right click arrived:";
    });

        timer.start(params.PERIOD);
        qInfo() << "Timer started";
    }
}

void SpecificWorker::compute()
{

    // Read visual elements from buffer_visual_elements
    auto ve_ = buffer_visual_elements.try_get();
    if (ve_.has_value() and not ve_.value().objects.empty())
    {
        process_people(ve_.value());
//        process_room_objects(ve_.value());
    }
    else
    {
        qWarning() << __FUNCTION__ << " No objects in data. Checking objects in graph";
        // Remove phase. Check for people not updated in more than 2 seconds
        if (auto robot_node_ = G->get_node("Shadow"); robot_node_.has_value()) {
            auto robot_node = robot_node_.value();
            auto now = std::chrono::high_resolution_clock::now();
            auto people_nodes = G->get_nodes_by_type("person");
            auto object_nodes = G->get_nodes_by_type("object");
            // Concatenate people and object nodes
            people_nodes.insert(people_nodes.end(), object_nodes.begin(), object_nodes.end());
            for (auto &person: people_nodes)
                if (auto update_time = G->get_attrib_by_name<timestamp_alivetime_att>(person); update_time.has_value())
                    if ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() -
                         update_time.value()) > params.SECS_TO_GET_OUT) {
                        qInfo() << "Person with id: " << person.id() << " has been removed";
                        G->delete_edge(robot_node.id(), person.id(), "RT");
                        G->delete_edge(robot_node.id(), person.id(), "following_action");
                        G->delete_node(person.id());
                    }
        }
    }

    this->hz = fps.print("FPS:", 3000);
}

void SpecificWorker::reset_graph_elements()
{
    //Get and delete all nodes type object
    auto nodes = G->get_nodes_by_type("object");
    for (auto node : nodes)
    {
        G->delete_node(node);
        //Print deleted node
        qInfo() << __FUNCTION__ << "Deleted node: " << node.id();
    }
    //Get and delete all person nodes
    auto person_nodes = G->get_nodes_by_type("person");
    for (auto node : person_nodes)
    {
        G->delete_node(node);
        //Print deleted node
        qInfo() << __FUNCTION__ << "Deleted node: " << node.id();
    }
    auto intention_nodes = G->get_nodes_by_type("intention");
    for (auto node : intention_nodes)
    {
        G->delete_node(node);
        //Print deleted node
        qInfo() << __FUNCTION__ << "Deleted node: " << node.id();
    }
}

//////////////////////////////// SpecificWorker /////////////////////////////////////////////////
void SpecificWorker::process_people(const RoboCompVisualElementsPub::TData &data)
{
    // admissibility conditions
    if(data.objects.empty())
    { qWarning() << __FUNCTION__ << " No objects in data"; return; }

    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    auto robot_level_ = G->get_node_level(robot_node);
    if(not robot_level_.has_value())
    { qWarning() << __FUNCTION__ << " No robot level in graph"; return; }
    auto robot_level = robot_level_.value();

    // Match phase. Check if the new objects are already in the existing objects vector
    std::vector<RoboCompVisualElementsPub::TObject> remaining_objects;
    auto now = std::chrono::high_resolution_clock::now();
    auto people_nodes = G->get_nodes_by_type("person");
    for (const auto &object: data.objects | iter::filter([this](auto &obj){return obj.type == params.PERSON;}))  // people
    {
        // Print person with id
        qInfo() << "Person" << object.id;
        // Check if the person is already in the graph
        if(auto person_in_graph = std::ranges::find_if(people_nodes, [&object, this](const DSR::Node &p)
                    {
                        if(auto person_id = G->get_attrib_by_name<person_id_att>(p); person_id.has_value())
                            return person_id.value() == object.id;
                        else return false;
                    }); person_in_graph != people_nodes.end())
        {
            if(auto node_insertion_time = G->get_attrib_by_name<timestamp_creation_att>(*person_in_graph); node_insertion_time.has_value())
            {
                auto node_insertion_time_ = node_insertion_time.value();
                if(auto person_checked = G->get_attrib_by_name<obj_checked_att>(*person_in_graph);
                person_checked.has_value() and not person_checked.value() and
                ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() - node_insertion_time_) > params.SECS_TO_GET_IN))
                {
                    G->add_or_modify_attrib_local<obj_checked_att>(*person_in_graph, true);
                    G->update_node(*person_in_graph);
                }
                if (auto edge_robot = rt->get_edge_RT(robot_node, person_in_graph->id()); edge_robot.has_value())
                {
                    G->add_or_modify_attrib_local<timestamp_alivetime_att>(*person_in_graph, get_actual_time());
                    G->update_node(*person_in_graph);
                    std::vector<float> new_robot_orientation, new_robot_pos = {0.0, 0.0, 0.0};
                    if (object.attributes.contains("x_pos") and object.attributes.contains("y_pos") and object.attributes.contains("z_pos"))
                        G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(),
                                             std::vector<float>{ std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), std::stof(object.attributes.at("z_pos"))});
                    if(object.attributes.contains("orientation"))
                        G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(edge_robot.value(),
                                                     std::vector<float>{0.0, 0.0, std::stof(object.attributes.at("orientation"))});
                    G->insert_or_assign_edge(edge_robot.value());
                }
            }
        }
        else     // Emplace the object in the remaining objects vector
            remaining_objects.push_back(object);
    }

    // Add phase. Process remaining objects by adding new persons to the list
    for (const auto &object: remaining_objects)
    {
        std::string node_name = "person_" + std::to_string(object.id);
        DSR::Node new_node = DSR::Node::create<person_node_type>(node_name);
        G->add_or_modify_attrib_local<person_id_att>(new_node, object.id);
        auto pos_x = static_cast<float>(rand()%170);
        auto pos_y = static_cast<float>(rand()%170);
        G->add_or_modify_attrib_local<pos_x_att>(new_node, pos_x);
        G->add_or_modify_attrib_local<pos_y_att>(new_node, pos_y);
        G->add_or_modify_attrib_local<timestamp_creation_att>(new_node, get_actual_time());
        G->add_or_modify_attrib_local<timestamp_alivetime_att>(new_node, get_actual_time());
        G->add_or_modify_attrib_local<level_att>(new_node, robot_level + 1);
        G->add_or_modify_attrib_local<obj_checked_att>(new_node, false);
        try
        {
            G->insert_node(new_node);
            qInfo() << __FUNCTION__ << " Person with id: " << object.id << " inserted in graph";
            std::vector<float> vector_robot_pos = {0.0, 0.0, 0.0}, orientation_vector = {0.0, 0.0, 0.0};
            if (object.attributes.contains("x_pos") and object.attributes.contains("y_pos") and object.attributes.contains("z_pos"))
                vector_robot_pos = { std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), std::stof(object.attributes.at("z_pos"))};
            if(object.attributes.contains("orientation"))
                orientation_vector = {0.0, 0.0, std::stof(object.attributes.at("orientation"))};
            rt->insert_or_assign_edge_RT(robot_node, new_node.id(), vector_robot_pos, orientation_vector);
        }
        catch(const std::exception &e)
        { std::cout << e.what() << " Error inserting node" << std::endl;}
    }

    // Remove phase. Check for people not updated in more than 2 seconds
    for (auto &person: people_nodes)
        if(auto update_time = G->get_attrib_by_name<timestamp_alivetime_att>(person); update_time.has_value())
            if ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() - update_time.value()) > params.SECS_TO_GET_OUT)
            {
                //qInfo() << "Person with id: " << person.id() << " has been removed";
                G->delete_edge(robot_node.id(), person.id(), "RT");
                G->delete_edge(robot_node.id(), person.id(), "following_action");
                G->delete_node(person.id());
            }
}

// function for processing every element detected by YOLO
void SpecificWorker::process_room_objects(const RoboCompVisualElementsPub::TData &data)
{
    // admissibility conditions
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();
    auto object_nodes = G->get_nodes_by_type("object");
    auto now = std::chrono::high_resolution_clock::now();
    if(data.objects.empty())
    {
        qWarning() << __FUNCTION__ << " No objects in data. Checking objects in graph"; return;
    }
    auto robot_level_ = G->get_node_level(robot_node);
    if(not robot_level_.has_value())
    { qWarning() << __FUNCTION__ << " No robot level in graph"; return; }
    auto robot_level = robot_level_.value();

    // Match phase. Check if the new objects are already in the existing objects vector
    std::vector<RoboCompVisualElementsPub::TObject> remaining_objects;
    // Get the index of the element in YOLO_NAMES = "chair"
    size_t idx = std::distance(YOLO_NAMES.begin(), std::ranges::find(YOLO_NAMES, "person"));
    for (const auto &object: data.objects | iter::filter([idx](auto &obj){return obj.type != static_cast<int>(idx) and !obj.attributes.contains("name");}))
    {
        // Check if the object is already in Graph
        if(auto object_in_graph = std::ranges::find_if(object_nodes, [&object, this](const DSR::Node &p)
            {
                auto object_id = G->get_attrib_by_name<obj_id_att>(p);
                if( object_id.has_value() and (object_id.value() == object.id)) return true;
                else return false;
            }); object_in_graph != object_nodes.end())
        {
            auto node_insertion_time = G->get_attrib_by_name<timestamp_creation_att>(*object_in_graph);
            auto object_checked = G->get_attrib_by_name<obj_checked_att>(*object_in_graph);
            if( node_insertion_time.has_value() and
                object_checked.has_value() and
                not object_checked.value() and
                ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() - node_insertion_time.value()) > params.SECS_TO_GET_IN))
            {
                G->add_or_modify_attrib_local<obj_checked_att>(*object_in_graph, true);
            }
            if (auto edge_robot = rt->get_edge_RT(robot_node, object_in_graph->id()); edge_robot.has_value())
            {
                G->add_or_modify_attrib_local<timestamp_alivetime_att>(*object_in_graph, get_actual_time());
                G->update_node(*object_in_graph);
                std::vector<float> new_robot_orientation, new_robot_pos = {0.0, 0.0, 0.0};
                if (object.attributes.contains("x_pos") and object.attributes.contains("y_pos") and object.attributes.contains("z_pos"))
                    G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(),
                                                                      std::vector<float>{ std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), std::stof(object.attributes.at("z_pos"))});
                if(object.attributes.contains("orientation"))
                    G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(edge_robot.value(),
                                                                             std::vector<float>{0.0, 0.0, std::stof(object.attributes.at("orientation"))});
                G->insert_or_assign_edge(edge_robot.value());
            }
        }
        else     // Emplace the object in the remaining objects vector
            remaining_objects.push_back(object);
    }

    // Add phase. Process remaining objects by adding new persons to the list
    for (const auto &object: remaining_objects)
    {
        // Get string name for the object from YOLO_NAMES
        std::string node_name = YOLO_NAMES[object.type] + "_" + std::to_string(object.id);
        qInfo() << QString::fromStdString(node_name);
        DSR::Node new_node = DSR::Node::create<object_node_type>(node_name);
        G->add_or_modify_attrib_local<obj_id_att>(new_node, object.id);
        if(object.attributes.contains("width") and object.attributes.contains("height"))
        {
            G->add_or_modify_attrib_local<pos_x_att>(new_node, stof(object.attributes.at("width")));
            G->add_or_modify_attrib_local<pos_y_att>(new_node, stof(object.attributes.at("height")));
        }
        G->add_or_modify_attrib_local<pos_x_att>(new_node, static_cast<float>(rand()%170));
        G->add_or_modify_attrib_local<pos_y_att>(new_node, static_cast<float>(rand()%170));
        G->add_or_modify_attrib_local<timestamp_creation_att>(new_node, get_actual_time());
        G->add_or_modify_attrib_local<timestamp_alivetime_att>(new_node, get_actual_time());
        G->add_or_modify_attrib_local<level_att>(new_node, robot_level + 1);
        G->add_or_modify_attrib_local<obj_checked_att>(new_node, false);
        // Check if the object is an obstacle or a target with target_names and obstacle_names data
        if(auto target = std::ranges::find(target_names, YOLO_NAMES[object.type]); target != target_names.end())
            G->add_or_modify_attrib_local<is_an_obstacle_att>(new_node, false);
        else
            G->add_or_modify_attrib_local<is_an_obstacle_att>(new_node, true);

        try
        {
            G->insert_node(new_node);
            qInfo() << __FUNCTION__ << " Object with id: " << object.id << " inserted in graph";
            std::vector<float> vector_robot_pos = {0.0, 0.0, 0.0}, orientation_vector = {0.0, 0.0, 0.0};
            if (object.attributes.contains("center_x") and object.attributes.contains("center_y"))
                vector_robot_pos = { std::stof(object.attributes.at("center_x")), std::stof(object.attributes.at("center_y")), 0.f};
            if(object.attributes.contains("rotation"))
                orientation_vector = {0.0, 0.0, std::stof(object.attributes.at("rotation"))};
            rt->insert_or_assign_edge_RT(robot_node, new_node.id(), vector_robot_pos, orientation_vector);
        }
        catch(const std::exception &e)
        { std::cout << e.what() << " Error inserting node" << std::endl;}
    }

    // Remove phase. Check for people not updated in more than 2 seconds
    for (auto &object: object_nodes)
        if(auto update_time = G->get_attrib_by_name<timestamp_alivetime_att>(object); update_time.has_value())
            if ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() - update_time.value()) > params.SECS_TO_GET_OUT)
            {
                qInfo() << "Object with id: " << object.id() << " has been removed";
                G->delete_edge(robot_node.id(), object.id(), "RT");
                G->delete_edge(robot_node.id(), object.id(), "following_action");
                G->delete_node(object.id());
            }
}

void SpecificWorker::print_people()
{
    auto people = G->get_nodes_by_type("person");
    qInfo() << "People: ";
    qInfo() << "    Num people: " << people.size();
    for (const auto &person: people)
    {
        qInfo() << "    Person id: " << G->get_attrib_by_name<person_id_att>(person).value();
    }
    qInfo() << "-----------------------------";
}
uint64_t SpecificWorker::get_actual_time()
{
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

//select_target function
bool SpecificWorker::select_target_from_lclick(QPointF &p)
{
    //Creation of the on_focus edge in the graph
    auto rt_edges = G->get_edges_by_type("RT");

    //check if rt_edges is empty
    if(rt_edges.empty())
        return false;

    //get shadow node
    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value())
        return false;

    //Get the x,y values from all the rt_edges
    for (const auto &rt_edge: rt_edges)
    {
        if (auto tr = G->get_attrib_by_name<rt_translation_att>(rt_edge); tr.has_value())
        {
            //Check if p is in a radius r of the x,y values
            if (p.x() < tr.value().get()[0] + 100 && p.x() > tr.value().get()[0] - 100 &&
                p.y() < tr.value().get()[1] + 100 && p.y() > tr.value().get()[1] - 100)
            {
                if(auto node = G->get_node(rt_edge.to()); node.has_value())
                {
                    if (auto edge_on_focus = G->get_edges_by_type("following_action"); edge_on_focus.size() > 0)
                    {
                        G->delete_edge(edge_on_focus[0].from(), edge_on_focus[0].to(), "following_action");
                        edge_on_focus[0].to(node.value().id());
                        G->insert_or_assign_edge(edge_on_focus[0]);
                    }
                    else // If does not exists create the edge
                    {
                        auto edge = DSR::Edge::create<following_action_edge_type >(robot_node.value().id(), node.value().id());
                        G->insert_or_assign_edge(edge);
                    }
                    G->update_node(robot_node.value());
                    return true;
                }
            }
        }
    }
    return false;
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
//delete_target function
void SpecificWorker::delete_target_from_rclick()
{
    //Delete following_action edge
    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value())
        return;

    auto edge_on_focus = G->get_edges_by_type("following_action");

    if(edge_on_focus.size() > 0)
    {
        G->delete_edge(edge_on_focus[0].from(), edge_on_focus[0].to(), "following_action");
        G->update_node(robot_node.value());
    }
}

//////////////////////////////// Draw ///////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &points, int decimate, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points)
    { scene->removeItem(p); delete p; }
    draw_points.clear();

    QPen pen = QPen(params.LIDAR_COLOR);
    QBrush brush = QBrush(params.LIDAR_COLOR);
    for (const auto &[i, p]: points |iter::enumerate)
    {
        // skip 2 out of 3 points
        if(i % decimate == 0)
        {
            auto o = widget_2d->scene.addRect(-20, 20, 40, 40, pen, brush);
            o->setPos(p.x(), p.y());
            draw_points.push_back(o);
        }
    }
}
void SpecificWorker::draw_room(const RoboCompVisualElementsPub::TObject &obj)
{
    //check if obj.attributes.contains the key name if it does print the value
    if(obj.attributes.contains("name"))
    {
        if (obj.attributes.at("name") == "room")
        {
            //save the attributes of the room width, depth,height,center_x,center_y,rotation
            float width = std::stof(obj.attributes.at("width"));
            float depth = std::stof(obj.attributes.at("depth"));
            //float height = std::stof(obj.attributes.at("height"));
            float center_x = std::stof(obj.attributes.at("center_x"));
            float center_y = std::stof(obj.attributes.at("center_y"));
            float rotation = std::stof(obj.attributes.at("rotation"));

            static QGraphicsRectItem *item = nullptr;
            if (item != nullptr)
                widget_2d->scene.removeItem(item);

            item = widget_2d->scene.addRect(-width / 2, -depth / 2, width, depth, QPen(QColor("black"),50));
            item->setPos(QPointF(center_x, center_y));
            item->setRotation(rotation+90);
        }
        else
            qWarning() << __FUNCTION__ << " Error. The object by parameter is not a room";
    }
    else
        qWarning() << __FUNCTION__ << " Error. The object does not contain the key name";

}
void SpecificWorker::draw_room_graph(QGraphicsScene *scene)
{
    static std::pair<uint64_t , QGraphicsItem*> draw_room = {0, nullptr};  // holds draw element

    // admissibility conditions
    auto room_nodes = G->get_nodes_by_type("room");
    if(room_nodes.empty()) return;

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    auto room_node = room_nodes.front();
    if (auto edge_robot = rt->get_edge_RT(robot_node.value(), room_node.id()); edge_robot.has_value())
    {
        auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
        auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
        auto width_ = G->get_attrib_by_name<width_att>(room_node);
        auto depth_ = G->get_attrib_by_name<depth_att>(room_node);
        if (tr.has_value() and rot.has_value() and width_.has_value() and depth_.has_value())
        {
            auto x = tr.value().get()[0], y = tr.value().get()[1];
            auto ang = rot.value().get()[2];
            if (draw_room.first == room_node.id())  // update room
            {
                draw_room.second->setPos(x, y);
                draw_room.second->setRotation(ang);
            }
            else   // add it
            {
                auto item = scene->addRect(-width_.value()/2.f, -depth_.value()/2.f, width_.value(), depth_.value(),
                                                     QPen(QColor("black"), 50));
                item->setPos(x, y);
                item->setRotation(ang);
                draw_room = {room_node.id(), item};
            }
        }
    }
}
void SpecificWorker::draw_objects_graph(QGraphicsScene *scene)
{
    static std::map<uint64_t , QGraphicsItem*> draw_objects; // holds draw elements

    // admissibility conditions
    auto object_nodes = G->get_nodes_by_type("chair");
    if(object_nodes.empty()) return;

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    for(const auto &object_node : object_nodes)
    {
        if (auto edge_robot = rt->get_edge_RT(robot_node.value(), object_node.id()); edge_robot.has_value())
        {
            auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
            auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
            //auto width_ = G->get_attrib_by_name<width_att>(object_node);
            //auto depth_ = G->get_attrib_by_name<depth_att>(object_node);
            if (tr.has_value() and rot.has_value() /*and width_.has_value() and depth_.has_value()*/)
            {
                auto x = tr.value().get()[0], y = tr.value().get()[1];
                auto ang = rot.value().get()[2];
                if (draw_objects.contains(object_node.id()))  // update room
                {
                    draw_objects.at(object_node.id())->setPos(x, y);
                    draw_objects.at(object_node.id())->setRotation(ang + 90);
                } else   // add it
                {
                    auto width = 400, depth = 400;  // TODO: get width and depth from graph
                    auto item = scene->addRect(-width / 2.f, -depth / 2.f, width, depth,
                                                         QPen(QColor("magenta"), 50));
                    item->setPos(x, y);
                    item->setRotation(ang + 90);    // TODO: orient with wall
                    draw_objects[object_node.id()] =  item;
                }
            }
        }
    }
    // check if object is not in the graph, but it is in the draw_map. Remove it.
    for (auto it = draw_objects.begin(); it != draw_objects.end();)
    {
        if (std::ranges::find_if(object_nodes, [&it, this](const DSR::Node &p)
        { return p.id() == it->first;}) == object_nodes.end())
        {
            scene->removeItem(it->second);
            delete it->second;
            it = draw_objects.erase(it);
        }
        else ++it;
    }
}
void SpecificWorker::draw_people_graph(QGraphicsScene *scene)
{
    float s = 500;
    auto color = QColor("orange");

    // draw people
    static std::map<uint64_t , QGraphicsItem *> draw_map;
    auto people_nodes = G->get_nodes_by_type("person");

    auto robot_node = G->get_node("Shadow");
    if(not robot_node.has_value()) return;

    for(const auto &person_node : people_nodes)
    {
        auto id = person_node.id();
        if (auto edge_robot = rt->get_edge_RT(robot_node.value(), id); edge_robot.has_value())
        {
            auto tr = G->get_attrib_by_name<rt_translation_att>(edge_robot.value());
            auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge_robot.value());
            if (tr.has_value() and rot.has_value())
            {
                auto x = tr.value().get()[0], y = tr.value().get()[1];
                auto ang = qRadiansToDegrees(rot.value().get()[2]);
                if (draw_map.contains(id))
                {
                    draw_map.at(id)->setPos(x, y);
                    draw_map.at(id)->setRotation(ang);
                }
                else    // add person
                {
                    auto circle = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("black"), 20),
                                                    QBrush(color));
                    auto line = scene->addLine(0, 0, 0, 250,
                                               QPen(QColor("black"), 20, Qt::SolidLine, Qt::RoundCap));
                    line->setParentItem(circle);
                    circle->setPos(x, y);
                    circle->setRotation(ang);
                    circle->setZValue(100);  // set it higher than the grid so it can be selected with mouse
                    draw_map.emplace(id, circle);
                }
            }
        }
    }
    // check person is not in the graph, but it is in the draw_map. Remove it.
    for (auto it = draw_map.begin(); it != draw_map.end();)
    {
        if (std::ranges::find_if(people_nodes, [&it, this](const DSR::Node &p)
        { return p.id() == it->first;}) == people_nodes.end())
        {
            scene->removeItem(it->second);
            delete it->second;
            it = draw_map.erase(it);
        }
        else ++it;  // removes without dismantling the iterators
    }
}
void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only)
{
//    for(auto p : points)
//        scene->removeItem(p);
//    points.clear();
//
//    if(erase_only) return;
//
//    float s = 100;
//    auto color = QColor("green");
//    for(const auto &p: path)
//    {
//        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(color), QBrush(color));
//        ptr->setPos(QPointF(p.x(), p.y()));
//        points.push_back(ptr);
//    }
}
void SpecificWorker::draw_scenario(const std::vector<Eigen::Vector3f> &points, QGraphicsScene *scene)
{
    draw_lidar(points, 3, scene);
//    draw_room_graph(scene);
    draw_objects_graph(scene);
    draw_people_graph(scene);
}
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if(type == "intention")
        if(auto intention_node = G->get_node(id); intention_node.has_value())
            if(intention_node.value().name() == "STOP")
                reset = true;
}
//////////////////////////////// Interfaces /////////////////////////////////////////////////
void SpecificWorker::VisualElementsPub_setVisualObjects(RoboCompVisualElementsPub::TData data)
{
    if (data.objects.empty())
        return;
    if(data.publisher == "forcefield")
        buffer_room_elements.put(std::move(data));
    else
        buffer_visual_elements.put(std::move(data));
}

/////////////////////////////// Testing ////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

