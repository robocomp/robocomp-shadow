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
//	THE FOLLOWING IS JUST AN EXAMPLE
//	To use innerModelPath parameter you should uncomment specificmonitor.cpp readConfig method content
//	try
//	{
//		RoboCompCommonBehavior::Parameter par = params.at("InnerModelPath");
//		std::string innermodel_path = par.value;
//		innerModel = std::make_shared(innermodel_path);
//	}
//	catch(const std::exception &e) { qFatal("Error reading config params"); }





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

void SpecificWorker::compute()
{
    // Check if room node exists
    auto room_node_ = G->get_node("room");
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room node in graph"; return; }
    // Get robot transformed nominal corners
    auto corner_nodes = G->get_nodes_by_type("corner");
    auto wall_nodes = G->get_nodes_by_type("wall");
    if(corner_nodes.empty()) { qWarning() << __FUNCTION__ << " No corner nodes in graph"; return; }
    // remove corner nodes with "measured" string in its name and order them by last name string value
    corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [](auto &n){ return n.name().find("measured") != std::string::npos; }), corner_nodes.end());
    std::sort(corner_nodes.begin(), corner_nodes.end(), [](auto &n1, auto &n2){ return n1.name() < n2.name(); });

    // Generate vector of tuples with corner id and corner position
    std::vector<std::pair<std::uint64_t, Eigen::Vector2f>> corners;

    // Get robot node fpr getting rt
    if(auto robot_node_ = G->get_node("Shadow"); robot_node_.has_value())
    {
        auto robot_node = robot_node_.value();
        // Iterate over corners

        std::vector<Eigen::Vector2f> corners;
        for(const auto &n: corner_nodes)
        {
//            if(auto rt_corner_edge_measured = rt->get_edge_RT(robot_node, n.id()); rt_corner_edge_measured.has_value())
//            {
//                auto corner_edge_measured = rt_corner_edge_measured.value();
//                if (auto rt_translation_measured = G->get_attrib_by_name<rt_translation_att>(rt_corner_edge_measured.value()); rt_translation_measured.has_value())
//                {
//                    auto rt_corner_measured_value = rt_translation_measured.value().get();
//                    corners.push_back({rt_corner_measured_value[0], rt_corner_measured_value[1]});
//                }
//            }
        }

        //given corners, create new_corners at distance d from the original corners in the direction of the center of the room
        std::vector<Eigen::Vector2f> new_corners;

        // Create empty QPolygonF
        QPolygonF poly_room;

//        for(auto &c: corners)
//        {
//            Eigen::Vector2f center = Eigen::Vector2f::Zero();
//            for(auto &c_: corners)
//                center += c_;
//            center /= corners.size();
//            Eigen::Vector2f dir = c - center;
//            dir.normalize();
//            Eigen::Vector2f new_corner = c+ dir * 1000;
//            new_corners.push_back(new_corner);
//            poly_room << QPointF(new_corner.x(), new_corner.y());
//        }

        float d = 100;
        for(auto &c: corners)
        {
            auto center = std::accumulate(corners.begin(), corners.end(), Eigen::Vector2f(0,0), [](const Eigen::Vector2f& acc, const Eigen::Vector2f& c){ return acc + c; }) / corners.size();
            auto dir = (center - c).normalized();
            Eigen::Vector2f new_corner = c + dir * d;
            poly_room << QPointF(new_corner.x(), new_corner.y());
            new_corners.push_back(new_corner);
        }

        /// read LiDAR
        auto res_ = buffer_lidar_data.try_get();
        if (not res_.has_value()) { return; }
        auto ldata = res_.value();

        // Filter lidar points inside room polygon
        RoboCompLidar3D::TData ldata_filtered;
        for(const auto &p: ldata.points)
            if(not poly_room.containsPoint(QPointF(p.x, p.y), Qt::OddEvenFill))
                ldata_filtered.points.push_back(p);

//         draw lidar
        if(widget_2d != nullptr)
        {
            draw_lidar(ldata_filtered, &widget_2d->scene);
            draw_polygon(poly_room, &widget_2d->scene, QColor("blue"));
        }

        Lines lines = extract_lines(ldata_filtered.points, consts.ranges_list);


//    /// Door detector
    auto doors = door_detector.detect(lines, &widget_2d->scene, corners);
    }
    fps.print("door_detector");
}

////////////////////////////////////////////////////////////////////////////////
SpecificWorker::Lines SpecificWorker::extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges)
{
    Lines lines(ranges.size());
    for(const auto &p: points)
        for(const auto &[i, r] : ranges | iter::enumerate)
            if(p.z > r.first and p.z < r.second)
                lines[i].emplace_back(p.x, p.y);
    return lines;
}
void SpecificWorker::insert_door_into_graph(const DoorDetector::Door &door)
{
    auto wall_node_ = G->get_node("wall_1");
    if(not wall_node_.has_value())
    { qWarning() << __FUNCTION__ << " No wall node in graph"; return; }
    auto wall_node = wall_node_.value();

    auto wall_node_level_ = G->get_node_level(wall_node);
    if(not wall_node_level_.has_value())
    { qWarning() << __FUNCTION__ << " No wall level in graph"; return; }
    auto wall_node_level = wall_node_level_.value();

    // Create door node
    auto door_node = DSR::Node::create<door_node_type>("door__measured");

    //    auto door_node = DSR::Node::create<door_node_type>("door_"+std::to_string(i)+"_measured");
    // Add door attributes
    G->add_or_modify_attrib_local<pos_x_att>(door_node, (float)(rand()%(170)));
    G->add_or_modify_attrib_local<pos_y_att>(door_node, (float)(rand()%170));
    G->add_or_modify_attrib_local<width_att>(door_node, (int)door.width());
    G->add_or_modify_attrib_local<height_att>(door_node, (int)door.height());
//    G->add_or_modify_attrib_local<angle_att>(door_node, door.angle_to_robot());
    G->add_or_modify_attrib_local<level_att>(door_node, wall_node_level + 1);

    // Add edge between door and robot
    std::vector<float> door_pos = {door.middle.x(), door.middle.y(), 0.f};
    std::vector<float> door_orientation = {0.f, 0.f, door.angle_to_robot()};
    rt->insert_or_assign_edge_RT(wall_node, door_node.id(), door_pos, door_orientation);

//    // Add edge between door and wall
//    Eigen::Vector3f door_robot_pos_point {door.middle.x(), door.middle.y(), 0.f};
//    auto door_robot_pos_double = door_robot_pos_point.cast <double>();


}
void SpecificWorker::update_door_in_graph(const DoorDetector::Door &door)
{
    auto door_node_ = G->get_node("door_1_measured");
    if(not door_node_.has_value())
    { qWarning() << __FUNCTION__ << " No door node in graph"; return; }
    auto door_node = door_node_.value();

    // Add door attributes
//    G->add_or_modify_attrib_local<pos_x_att>(door_node, door.middle.x());
//    G->add_or_modify_attrib_local<pos_y_att>(door_node, door.middle.y());
    G->add_or_modify_attrib_local<width_att>(door_node, (int)door.width());
    G->add_or_modify_attrib_local<height_att>(door_node, (int)door.height());
//    G->add_or_modify_attrib_local<angle_att>(door_node, door.angle_to_robot());
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
// Create funcion to draw QPolygonF on 2dwidget
void SpecificWorker::draw_polygon(const QPolygonF &poly, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &p: poly)
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(color, 20), QBrush(color));
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
    // Draw lines between corners
    for(int i = 0; i < poly.size(); i++)
    {
        auto line = scene->addLine(poly[i].x(), poly[i].y(), poly[(i+1)%poly.size()].x(), poly[(i+1)%poly.size()].y(), QPen(color, 20));
        items.push_back(line);
    }
}
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

