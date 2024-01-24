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
//	QLoggingCategory::setFilterRules("*.debug=false\n");
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
	std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;  

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
//	graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);
//    widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));

        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        wanted_person = Person(gridder_proxy);
//        wanted_person.set_target_element(true);
//        wanted_person.init_item(&viewer->scene, 0.f, 1000.f, 0.f);

        // mouse
//        connect(&custom_widget.viewer->scene, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
//        {
//            qInfo() << "[MOUSE] New click left arrived:" << p;
//            // Check item at position corresponding to person representation (person can't be selected if cone layer is superposed)
//            // TODO:  selectedItem->setZValue() changes the order of the items in the scene. Use this to avoid the cone layer to be superposed
//            QList<QGraphicsItem *> itemsAtPosition = widget_2d->scene.items(p, Qt::IntersectsItemShape, Qt::DescendingOrder, QTransform());
//            QGraphicsItem *selectedItem = nullptr;
//
//            if(auto res = std::ranges::find_if(itemsAtPosition, [p](auto it)
//                    { return (Eigen::Vector2f(it->pos().x(), it->pos().y()) - Eigen::Vector2f(p.x(), p.y())).norm() < 100;}); res != itemsAtPosition.end())
//                selectedItem = *res;
//            else return;
//
//            // check person with the same item
//            if(auto person = std::ranges::find_if(people, [selectedItem](const Person &p)
//                { return p.get_item() == selectedItem; }); person != people.end())
//            {
//                qInfo() << "[MOUSE] Target selected";
//                person->set_target_element(true);
//            }
//        });
//        //Right click
//        connect(custom_widget.viewer, &AbstractGraphicViewer::right_click, [this](QPointF p)
//        {
//            qInfo() << "[MOUSE] New right click arrived:" << p;
//            for (auto &person: people)
//                person.set_target_element(false);
//            try
//            {
//                segmentatortrackingpub_pubproxy->setTrack(RoboCompVisualElementsPub::TObject{.id = -1});
//                draw_path({}, &widget_2d->scene, false);
//            }
//            catch (const Ice::Exception &e)
//            { std::cout << "Error setting target" << e << std::endl; }
//        });

        timer.start(params.PERIOD);
        qInfo() << "Timer started";
	}
}
void SpecificWorker::compute()
{
    /// read LiDAR
//    auto res_ = buffer_lidar_data.try_get();
//    if (not res_.has_value())  { /*qWarning() << "No data Lidar";*/ return; }
//    auto points = res_.value();
//    draw_lidar(points, params.LIDAR_LOW_DECIMATION_FACTOR);

    // Read visual elements from buffer_visual_elements
    auto ve_ = buffer_visual_elements.try_get();
    if (ve_.has_value() and not ve_.value().objects.empty())
    {
        process_people(ve_.value());
        process_room_objects(ve_.value());
    }

    // if there is a target person, compute the path from robot
//    postprocess_target_person(people);

    // Read room elements from buffer_room_elements
    auto re_ = buffer_room_elements.try_get();
    if (re_.has_value())
        process_room(re_.value());
    // print_people();
//    this->custom_widget.lcdNumber_people->display((int)people.size());
    this->hz = fps.print("FPS:", 3000);
//    this->custom_widget.lcdNumber_hz->display(this->hz);
}

//////////////////////////////// SpecificWorker /////////////////////////////////////////////////
void SpecificWorker::process_people(const RoboCompVisualElementsPub::TData &data)
{
    /// Go through three phases: match, add and remove.
    // Match phase. Check if the new objects are already in the existing objects vector
    std::vector<RoboCompVisualElementsPub::TObject> remaining_objects;
    auto now = std::chrono::high_resolution_clock::now();
    for (const auto &object: data.objects | iter::filter([this](auto &obj){return obj.type == params.PERSON;}))  // people
    {
        // Check if the person is already in the people vector
        if(auto person_in_list = std::ranges::find_if(people, [&object](const Person &p)
                    { return p.get_id() == object.id; }); person_in_list != people.end())
        {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - person_in_list->get_insertion_time()).count() >
                params.SECS_TO_GET_IN)
            {
                if (person_in_list->get_dsr_id() == -1) // First time after admission. Initialize item
                {
                    insert_person_in_graph(*person_in_list);
//                    qInfo() << "Person with id: " << person_in_list->get_id() << " inserted in graph with id: " << person_in_list->get_dsr_id();
                }

                else    // A veteran. Update attributes
                {
                    person_in_list->update_attributes(data.objects);    //TODO: change to object
                    qInfo() << "Person with id: " << person_in_list->get_id() << " is being updated";
                    update_person_in_graph(*person_in_list);
//                    person_in_list->is_inside_pilar_cone(data.objects);
                }
                person_in_list->update_last_update_time();  // update time for all
            }
        }
        else     // Emplace the object in the remaining objects vector
            remaining_objects.push_back(object);
    }

    // Add phase. Process remaining objects by adding new persons to the list
    for (const auto &object: remaining_objects)
    {
        // Create a new person
        Person new_person(gridder_proxy);
        new_person.set_person_data(object);
        new_person.set_insertion_time();
        new_person.update_last_update_time();
        people.emplace_back(new_person);
    }

    // Remove phase. Check for people not updated in more than 2 seconds
    for (auto &person: people)
        if (std::chrono::duration_cast<std::chrono::seconds>(now - person.get_last_update_time()).count() > params.SECS_TO_GET_OUT)
        {
            //qInfo() << "Person with id: " << person.get_id() << " has been removed";
//            person.draw_paths(&widget_2d->scene, true, false);
//            person.remove_item(&widget_2d->scene);
            people.erase(std::ranges::find_if(people, [&person](const Person &p) { return p.get_id() == person.get_id(); }));
        }
}
void SpecificWorker::postprocess_target_person(const People &people_)
{
    // search for the target person in people list
    if(auto target_person = std::ranges::find_if(people_, [](const Person &p){return p.is_target_element();}); target_person != people_.end())
    {
        auto x_ = target_person->get_attribute("x_pos");
        auto y_ = target_person->get_attribute("y_pos");
        if(x_.has_value()  and y_.has_value())
        {
            try
            {
                Eigen::Vector2f t{x_.value(), y_.value()};
                if(t.norm() > 500.f)    // TODO: move to params
                    t = t.normalized() * (t.norm() - 500.f);
                auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{0.f, 0.f},
                                                      RoboCompGridder::TPoint{t.x(), t.y()},
                                                      1, true, true);
                if (result.valid and not result.paths.empty())
                {
                    std::vector<Eigen::Vector2f> path;
                    for (const auto &p: result.paths.front())
                        path.emplace_back(p.x, p.y);
//                    draw_path(path, &widget_2d->scene, false); // draws path from robot to target person
                    //target_person->draw_paths(&viewer->scene, false, true);  // draws paths from person to objects
                }
            }
            catch (const Ice::Exception &e)
            { std::cout << __FUNCTION__ << " Error reading plans from Gridder. " << e << std::endl; }
            // publish target to driver
            try
            { segmentatortrackingpub_pubproxy->setTrack(target_person->get_target()); }
            catch (const Ice::Exception &e)
            { std::cout << "Error setting target: " << e << std::endl; }
        }
        else
            qWarning() << __FUNCTION__ << " Error. The object does not contain the key x_pos or y_pos";
    }
}
void SpecificWorker::process_room_objects(const RoboCompVisualElementsPub::TData &data)
{
    /// Go through three phases: match, add and remove.
    // Match phase. Check if the new objects are already in the existing objects vector
    std::vector<RoboCompVisualElementsPub::TObject> remaining_objects;
    auto now = std::chrono::high_resolution_clock::now();
    // Get the index of the element in YOLO_NAMES = "chair"
    size_t idx = std::distance(YOLO_NAMES.begin(), std::ranges::find(YOLO_NAMES, "chair"));
    for (const auto &object: data.objects | iter::filter([idx](auto &obj){return obj.type == static_cast<int>(idx);}))
    {
        // Check if the object is already in the objects vector
        if(auto object_in_list = std::ranges::find_if(objects, [&object](const Object &o)
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
        Object new_object;
        new_object.set_object_data(rem_object);
        new_object.set_insertion_time();
        new_object.update_last_update_time();
        objects.emplace_back(new_object);
    }

    // Remove phase. Check for people not updated in more than 2 seconds
    for (auto &object: objects)
        if (std::chrono::duration_cast<std::chrono::seconds>(now - object.get_last_update_time()).count() > params.SECS_TO_GET_OUT)
        {
            //qInfo() << "Person with id: " << person.get_id() << " has been removed";
//            object.remove_item(&widget_2d->scene);
            objects.erase(std::ranges::find_if(objects, [&object](const Object &p) { return p.get_id() == object.get_id(); }));
        }
}
void SpecificWorker::process_room(const RoboCompVisualElementsPub::TData &data)
{
    // Check if there is data in data.objects
    if (data.objects.empty())
    { qWarning() << __FUNCTION__ << " No objects in data"; return; }

    //iterate over the objects and draw the rooms
//    for (const auto &o : data.objects | iter::filter([](auto &a){return a.attributes.at("name") == "room";}))
//        draw_room(o);
}
void SpecificWorker::insert_person_in_graph(Person &person)
{
    if(auto robot_node = G->get_node("Shadow"); robot_node.has_value())
    {
        if (auto robot_level = G->get_node_level(robot_node.value()); robot_level.has_value())
        {
                std::string node_name = "person_" + std::to_string(person.get_id());
                DSR::Node new_node = DSR::Node::create<person_node_type>(node_name);
                G->add_or_modify_attrib_local<person_id_att>(new_node, person.get_id());
                G->add_or_modify_attrib_local<level_att>(new_node, robot_level.value() + 1);
                try
                {
                    G->insert_node(new_node);
                    person.set_dsr_id(new_node.id());
                    qInfo() << "Person with id: " << person.get_id() << " inserted in graph with id: " << person.get_dsr_id();
                    // TODO: revise optionals in get_attributes
                    std::vector<float> vector_robot_pos =
                            { person.get_attribute("x_pos").value(), person.get_attribute("y_pos").value(), 0.f};
                    std::vector<float> orientation_vector = {0.0, 0.0, person.get_attribute("orientation").value()};
                    rt->insert_or_assign_edge_RT(robot_node.value(), new_node.id(), vector_robot_pos, orientation_vector);
                }
                catch(const std::exception &e)
                { std::cout << e.what() << " Error inserting node" << std::endl;}
        }
    }
}
void SpecificWorker::update_person_in_graph(const Person &person)
{
    if(auto robot_node = G->get_node("Shadow"); robot_node.has_value())
    {
        qInfo() << "Person with id: " << person.get_id() << " is being updated";
        if(auto person_node = G->get_node(person.get_dsr_id()); person_node.has_value())
        {
            qInfo() << "Person with id: " << person.get_id() << " is being updated";
            if (auto edge_robot = rt->get_edge_RT(robot_node.value(), person_node.value().id()); edge_robot.has_value())
            {
                std::vector<float> new_robot_pos ={ person.get_attribute("x_pos").value(), person.get_attribute("y_pos").value(), 0.f};
                std::vector<float> new_robot_orientation = {0.0, 0.0, person.get_attribute("orientation").value()};
                G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(edge_robot.value(), new_robot_orientation);
                G->add_or_modify_attrib_local<rt_translation_att>(edge_robot.value(), new_robot_pos);
                if (G->insert_or_assign_edge(edge_robot.value()))
                    qInfo() << __FUNCTION__ << "UPDATED" << new_robot_pos[0] << new_robot_pos[1];
                qInfo() << "Person with id: " << person.get_id() << " updated in graph with id: " << person.get_dsr_id();
                qInfo() << "    New position: " << new_robot_pos[0] << ", " << new_robot_pos[1] << ", " << new_robot_pos[2];
                qInfo() << "    New orientation: " << new_robot_orientation[0] << ", " << new_robot_orientation[1] << ", " << new_robot_orientation[2];
            }
        }
    }
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
void SpecificWorker::print_people(const People &ppol)
{
    qInfo() << " Num people: " << ppol.size();
    for (const auto &person: ppol)
        person.print();
    qInfo() << "-----------------------------";}

//////////////////////////////// Draw ///////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &points, int decimate)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points)
    {
        widget_2d->scene.removeItem(p);
        delete p;
    }
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
void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only)
{
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    if(erase_only) return;

    float s = 100;
    auto color = QColor("green");
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(color), QBrush(color));
        ptr->setPos(QPointF(p.x(), p.y()));
        points.push_back(ptr);
    }
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
