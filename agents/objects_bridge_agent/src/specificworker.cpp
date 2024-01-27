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
	this->Period = 100;
	if(this->startup_check_flag)
		this->startup_check();
	else
	{
	// create graph
	G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, "shadow_scene.json"); // Init nodes
	qInfo() << __FUNCTION__ << "Graph loaded. Agent:" << agent_name.c_str() << "ID:" << agent_id << " from "  << "shadow_scene.json";

	//dsr update signals
	//connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
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

	//graph_viewer->add_custom_widget_to_dock("Intention Prediction", &custom_widget);
    widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));
    widget_2d->setSceneRect(-5000, -5000,  10000, 10000);
    widget_2d->set_draw_axis(true);

    // Lidar thread is created
    read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
    std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

    // mouse
    connect(widget_2d, &DSR::QScene2dViewer::mouse_left_click, [this](QPointF p)
    {
        qInfo() << "[MOUSE] New click left arrived:" << p;
        // Check item at position corresponding to person representation (person can't be selected if cone layer is superposed)
        // TODO:  selectedItem->setZValue() changes the order of the items in the scene. Use this to avoid the cone layer to be superposed
        QList<QGraphicsItem *> itemsAtPosition = widget_2d->scene.items(p, Qt::IntersectsItemShape, Qt::DescendingOrder, QTransform());
        QGraphicsItem *selectedItem = nullptr;

        if(auto res = std::ranges::find_if(itemsAtPosition, [p](auto it)
                { return (Eigen::Vector2f(it->pos().x(), it->pos().y()) - Eigen::Vector2f(p.x(), p.y())).norm() < 100;}); res != itemsAtPosition.end())
            selectedItem = *res;
        else return;

        // check person with the same item
//        if(auto person = std::ranges::find_if(people, [selectedItem](const Person &p)
//            { return p.get_item() == selectedItem; }); person != people.end())
//        {
//            qInfo() << "[MOUSE] Target selected";
//            person->set_target_element(true);
//        }
    });
    //Right click
    connect(widget_2d, &DSR::QScene2dViewer::mouse_right_click, [this](int x, int y, uint64_t id)
    {
        qInfo() << "[MOUSE] New right click arrived:";
//        for (auto &person: people)
//            person.set_target_element(false);
        try
        {
            segmentatortrackingpub_pubproxy->setTrack(RoboCompVisualElementsPub::TObject{.id = -1});
            draw_path({}, &widget_2d->scene, false);
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error setting target" << e << std::endl; }
    });

    timer.start(params.PERIOD);
    qInfo() << "Timer started";
}
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value())  { /*qWarning() << "No data Lidar";*/ return; }
    auto points = res_.value();

    // Read visual elements from buffer_visual_elements
    auto ve_ = buffer_visual_elements.try_get();
    if (ve_.has_value() and not ve_.value().objects.empty())
    {
        process_people(ve_.value());
        process_room_objects(ve_.value());
    }

    // if there is a target person, compute the path from robot
    //    postprocess_target_person(people);

    // Read room element from buffer_room_elements
        if(auto re_ = buffer_room_elements.try_get(); re_.has_value())
            process_room(re_.value());

    this->hz = fps.print("FPS:", 3000);
    draw_scenario(points, &widget_2d->scene);
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
                    if (object.attributes.contains("x_pos") and object.attributes.contains("y_pos"))
                        G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(),
                                             std::vector<float>{ std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), 0.f});
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
            if (object.attributes.contains("x_pos") and object.attributes.contains("y_pos"))
                vector_robot_pos = { std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), 0.f};
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
                G->delete_node(person.id());
            }
}
void SpecificWorker::process_room_objects(const RoboCompVisualElementsPub::TData &data)
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
    auto object_nodes = G->get_nodes_by_type("chair");
    // Get the index of the element in YOLO_NAMES = "chair"
    size_t idx = std::distance(YOLO_NAMES.begin(), std::ranges::find(YOLO_NAMES, "chair"));
    for (const auto &object: data.objects | iter::filter([idx](auto &obj){return obj.type == static_cast<int>(idx);}))
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
                if (object.attributes.contains("x_pos") and object.attributes.contains("y_pos"))
                    G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(),
                                                                      std::vector<float>{ std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), 0.f});
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
        std::string node_name = "chair_" + std::to_string(object.id);
        DSR::Node new_node = DSR::Node::create<chair_node_type>(node_name);
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
    for (auto &person: object_nodes)
        if(auto update_time = G->get_attrib_by_name<timestamp_alivetime_att>(person); update_time.has_value())
            if ((std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count() - update_time.value()) > params.SECS_TO_GET_OUT)
            {
                //qInfo() << "Person with id: " << person.id() << " has been removed";
                G->delete_edge(robot_node.id(), person.id(), "RT");
                G->delete_node(person.id());
            }
}
void SpecificWorker::process_room(const RoboCompVisualElementsPub::TData &data)
{
    // Admissibility conditions
    if (data.objects.empty())
    { qWarning() << __FUNCTION__ << " No objects in data"; return; }

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
    if(auto room_ = std::ranges::find_if(data.objects, [](auto &o)
                        { return o.attributes.at("name") == "room";}); room_ != data.objects.end())
    {
        // Match phase. Check if the room is already in graph G
        if(auto room_nodes = G->get_nodes_by_type("room"); not room_nodes.empty())
        {
            auto room_node = room_nodes.front();
            auto node_insertion_time = G->get_attrib_by_name<timestamp_creation_att>(room_node);
            auto room_checked = G->get_attrib_by_name<obj_checked_att>(room_node);
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
            G->add_or_modify_attrib_local<width_att>(room_node, std::stoi(room_->attributes.at("width")));
            G->add_or_modify_attrib_local<height_att>(room_node, std::stoi(room_->attributes.at("height")));
            G->add_or_modify_attrib_local<depth_att>(room_node, std::stoi(room_->attributes.at("depth")));
            G->update_node(room_node);
            if (auto edge_robot = rt->get_edge_RT(robot_node, room_node.id()); edge_robot.has_value())
            {
                std::vector<float> new_robot_orientation, new_robot_pos = {0.0, 0.0, 0.0};
                G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(), std::vector<float>{
                        std::stof(room_->attributes.at("center_x")),
                        std::stof(room_->attributes.at("center_y")), 0.f});
                G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(edge_robot.value(),
                                                                         std::vector<float>{0.0, 0.0, std::stof(
                                                                                 room_->attributes.at(
                                                                                         "rotation"))});
                G->insert_or_assign_edge(edge_robot.value());
            }
        } else  // Insert phase. No room in graph. Insert room
        {
            try
            {
                DSR::Node new_node = DSR::Node::create<room_node_type>("room");
                G->add_or_modify_attrib_local<person_id_att>(new_node, 0);  // TODO: current_room
                G->add_or_modify_attrib_local<width_att>(new_node, std::stoi(room_->attributes.at("width")));
                G->add_or_modify_attrib_local<height_att>(new_node, std::stoi(room_->attributes.at("height")));
                G->add_or_modify_attrib_local<depth_att>(new_node, std::stoi(room_->attributes.at("depth")));
                G->add_or_modify_attrib_local<pos_x_att>(new_node, (float)(rand()%(170)));
                G->add_or_modify_attrib_local<pos_y_att>(new_node, (float)(rand()%170));
                G->add_or_modify_attrib_local<obj_checked_att>(new_node, false);
                G->add_or_modify_attrib_local<level_att>(new_node, robot_level + 1);
                G->insert_node(new_node);
                std::vector<float> vector_robot_pos = {0.0, 0.0, 0.0}, orientation_vector = {0.0, 0.0, 0.0};
                vector_robot_pos = { std::stof(room_->attributes.at("center_x")), std::stof(room_->attributes.at("center_y")), 0.f};
                orientation_vector = {0.0, 0.0, std::stof(room_->attributes.at("rotation"))};
                rt->insert_or_assign_edge_RT(robot_node, new_node.id(), vector_robot_pos, orientation_vector);
            }
            catch(const std::exception &e)
            { std::cout << e.what() << " Error inserting node" << std::endl;}
            qInfo() << __FUNCTION__ << " Object with id: " << room_->id << " inserted in graph";
        }
    }
    // no remove of rooms for now
}
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
                draw_room.second->setRotation(ang + 90);
            }
            else   // add it
            {
                auto item = scene->addRect(-width_.value()/2.f, -depth_.value()/2.f, width_.value(), depth_.value(),
                                                     QPen(QColor("black"), 50));
                item->setPos(x, y);
                item->setRotation(ang + 90);
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
                    auto width = 400, depth = 400;
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
    draw_room_graph(scene);
    draw_objects_graph(scene);
    draw_people_graph(scene);
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


//void SpecificWorker::postprocess_target_person(const People &people_)
//{
// search for the target person in people list
//    if(auto target_person = std::ranges::find_if(people_, [](const Person &p){return p.is_target_element();}); target_person != people_.end())
//    {
//        auto x_ = target_person->get_attribute("x_pos");
//        auto y_ = target_person->get_attribute("y_pos");
//        if(x_.has_value()  and y_.has_value())
//        {
//            try
//            {
//                Eigen::Vector2f t{x_.value(), y_.value()};
//                if(t.norm() > 500.f)    // TODO: move to params
//                    t = t.normalized() * (t.norm() - 500.f);
//                auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{0.f, 0.f},
//                                                      RoboCompGridder::TPoint{t.x(), t.y()},
//                                                      1, true, true);
//                if (result.valid and not result.paths.empty())
//                {
//                    std::vector<Eigen::Vector2f> path;
//                    for (const auto &p: result.paths.front())
//                        path.emplace_back(p.x, p.y);
////                    draw_path(path, &widget_2d->scene, false); // draws path from robot to target person
//                    //target_person->draw_paths(&viewer->scene, false, true);  // draws paths from person to objects
//                }
//            }
//            catch (const Ice::Exception &e)
//            { std::cout << __FUNCTION__ << " Error reading plans from Gridder. " << e << std::endl; }
//            // publish target to driver
//            try
//            { segmentatortrackingpub_pubproxy->setTrack(target_person->get_target()); }
//            catch (const Ice::Exception &e)
//            { std::cout << "Error setting target: " << e << std::endl; }
//        }
//        else
//            qWarning() << __FUNCTION__ << " Error. The object does not contain the key x_pos or y_pos";
//    }
//}

//void SpecificWorker::insert_person_in_graph(const RoboCompVisualElementsPub::TObject &person)
//{
//    if(auto robot_node = G->get_node("Shadow"); robot_node.has_value())
//    {
//        if (auto robot_level = G->get_node_level(robot_node.value()); robot_level.has_value())
//        {
//            std::string node_name = "person_" + std::to_string(person.id);
//            DSR::Node new_node = DSR::Node::create<person_node_type>(node_name);
//            G->add_or_modify_attrib_local<person_id_att>(new_node, person.id);
//            auto now = std::chrono::high_resolution_clock::now();
//            float pos_x = rand()%(170);
//            float pos_y = rand()%(170);
//            G->add_or_modify_attrib_local<pos_x_att>(new_node, pos_x);
//            G->add_or_modify_attrib_local<pos_y_att>(new_node, pos_y);
//            G->add_or_modify_attrib_local<timestamp_creation_att>(new_node, get_actual_time());
//            G->add_or_modify_attrib_local<timestamp_alivetime_att>(new_node, get_actual_time());
//            G->add_or_modify_attrib_local<level_att>(new_node, robot_level.value() + 1);
//            G->add_or_modify_attrib_local<obj_checked_att>(new_node, false);
//            try
//            {
//                G->insert_node(new_node);
//                qInfo() << "Person with id: " << person.id << " inserted in graph";
//                // TODO: revise optionals in get_attributes
//                std::vector<float> vector_robot_pos = {0.0, 0.0, 0.0}, orientation_vector = {0.0, 0.0, 0.0};
//                if (person.attributes.contains("x_pos") and
//                    person.attributes.contains("y_pos"))
//                    vector_robot_pos = { std::stof(person.attributes.at("x_pos")), std::stof(person.attributes.at("y_pos")), 0.f};
//                if(person.attributes.contains("orientation"))
//                    orientation_vector = {0.0, 0.0, std::stof(person.attributes.at("orientation"))};
//                rt->insert_or_assign_edge_RT(robot_node.value(), new_node.id(), vector_robot_pos, orientation_vector);
//            }
//            catch(const std::exception &e)
//            { std::cout << e.what() << " Error inserting node" << std::endl;}
//        }
//    }
//}

//void SpecificWorker::update_person_in_graph(const RoboCompVisualElementsPub::TObject &person, DSR::Node person_node)
//{
//    if(auto robot_node = G->get_node("Shadow"); robot_node.has_value())
//    {
//        if (auto edge_robot = rt->get_edge_RT(robot_node.value(), person_node.id()); edge_robot.has_value())
//        {
//            G->add_or_modify_attrib_local<timestamp_alivetime_att>(person_node, get_actual_time());
//            G->update_node(person_node);
//            std::vector<float> new_robot_orientation, new_robot_pos = {0.0, 0.0, 0.0};
//            if (person.attributes.contains("x_pos") and
//                person.attributes.contains("y_pos"))
//                new_robot_pos = { std::stof(person.attributes.at("x_pos")), std::stof(person.attributes.at("y_pos")), 0.f};
//            if(person.attributes.contains("orientation"))
//                new_robot_orientation = {0.0, 0.0, std::stof(person.attributes.at("orientation"))};
//            G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(edge_robot.value(), new_robot_orientation);
//            G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(), new_robot_pos);
//            if (G->insert_or_assign_edge(edge_robot.value()))
//            //    qInfo() << __FUNCTION__ << "UPDATED" << new_robot_pos[0] << new_robot_pos[1];
//            {}
//            //qInfo() << "    New position: " << new_robot_pos[0] << ", " << new_robot_pos[1] << ", " << new_robot_pos[2];
//            //qInfo() << "    New orientation: " << new_robot_orientation[0] << ", " << new_robot_orientation[1] << ", " << new_robot_orientation[2];
//            //qInfo() << "    New timestamp: " << get_actual_time();
//        }
//    }
//}

  /*  if(auto object_in_list = std::ranges::find_if(objects, [&object](const Object &o)
        { return o.get_id() == object.id; }); object_in_list != objects.end())
    {
        if (std::chrono::duration_cast<std::chrono::seconds>(now - object_in_list->get_insertion_time()).count() >
            params.SECS_TO_GET_IN)
        {

            if (object_in_list->get_item() == nullptr) // First time after admission. Initialize item
            {

            }
//                    object_in_list->init_item(&widget_2d->scene, std::stof(object.attributes.at("x_pos")),
//                                              std::stof(object.attributes.at("y_pos")),
//                                              500, 500);
            else    // A veteran. Update attributes
            {
                // Print accessing
                qInfo() << "Object with id: " << object_in_list->get_id() << " is being accessed";
                object_in_list->update_attributes(object);
            }


            object_in_list->update_last_update_time();  // update time for all
        }
    }
    else     // Emplace the object in the remaining objects vector
        remaining_objects.push_back(object);
}
// for each object, print the attributes.

//    for (const auto &object: objects)
//    {
//        std::cout << "Object id: " << object.get_id() << std::endl;
//        for (const auto &attribute: object.obj.attributes)
//            std::cout << "    " << attribute.first << ": " << attribute.second << std::endl;
//    }
// Add phase. Process remaining objects by adding new objects to the list
for (const auto &rem_object: remaining_objects)
{
// Create a new object
//        Object new_object;
//        new_object.set_object_data(rem_object);
//        new_object.set_insertion_time();
//        new_object.update_last_update_time();
//        objects.emplace_back(new_object);
}*/

// Remove phase. Check for people not updated in more than 2 seconds
//    for (auto &object: objects)
//        if (std::chrono::duration_cast<std::chrono::seconds>(now - object.get_last_update_time()).count() > params.SECS_TO_GET_OUT)
//        {
//            //qInfo() << "Person with id: " << person.get_id() << " has been removed";
////            object.remove_item(&widget_2d->scene);
//            objects.erase(std::ranges::find_if(objects, [&object](const Object &p) { return p.get_id() == object.get_id(); }));
//        }*/