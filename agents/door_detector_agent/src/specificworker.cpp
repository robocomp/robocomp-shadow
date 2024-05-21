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
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value()) { return; }
    auto ldata = res_.value();

    /// Check if robot node exists
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
    auto robot_node = robot_node_.value();

    /// Check if room node exists
    auto room_node_ = G->get_node("room");
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room node in graph"; return; }

    /// Get corner nodes from graph
    auto corner_nodes = G->get_nodes_by_type("corner");
    /// Get nodes which name not contains "measured"
    corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [](auto &n){ return n.name().find("measured") != std::string::npos; }), corner_nodes.end());
    /// print corner names
//    for(auto &n: corner_nodes) std::cout << __FUNCTION__ << " Corner node: " << n.name() << std::endl;
    /// Sort corner nodes by name
    std::sort(corner_nodes.begin(), corner_nodes.end(), [](auto &n1, auto &n2){ return n1.name() < n2.name(); });
    // Iterate over corners
    std::vector<Eigen::Vector2f> corners, wall_centers;


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
                                    wall_centers.push_back(center);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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

//        // Filter lidar points inside room polygon
//        RoboCompLidar3D::TData ldata_filtered;
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
        {
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
                    qInfo()<< "Line norm: " << line_norm << " " << p0.x << p0.y << " " << p1.x << p1.y;
                    bool is_door = true;
                    /// Check all indexes between window[0] and window[1] in lidar points
                    /// If there is a point inside the wall, then it is not a door
                    for(int i = window[0]; i < window[1]; i++)
                    {
                        if(not outside_poly_in[i])
                        {
                            is_door = false;
                            break;
                        }
                    }
                    if(is_door)
                    {
                        //project p0 and p1 on the polygon
                        auto p0_projected = projectPointOnPolygon(QPointF(p0.x, p0.y), nominal_polygon);
                        auto p1_projected = projectPointOnPolygon(QPointF(p1.x, p1.y), nominal_polygon);
                        /// Check between which nominal corners the projected points are
                        auto p0_projected_eigen = Eigen::Vector2f(p0_projected.x(), p0_projected.y());
                        auto p1_projected_eigen = Eigen::Vector2f(p1_projected.x(), p1_projected.y());
                        /// Obtanin the central point of the door and, considering the nominal corners, check between which corners the door is
                        auto middle = (p0_projected_eigen + p1_projected_eigen) / 2;
                        qInfo() << "Middle: " << middle.x() << middle.y();
                        /// Get the index of the wall point closer to middle point using a lambda
                        auto wall_center = *std::min_element(wall_centers.begin(), wall_centers.end(), [&middle](const Eigen::Vector2f &a, const Eigen::Vector2f &b){ return (a - middle).norm() < (b - middle).norm(); });
                        qInfo() << "Wall center: " << wall_center.x() << wall_center.y();
                        /// Get the index of the wall center
                        auto wall_center_index = std::distance(wall_centers.begin(), std::find(wall_centers.begin(), wall_centers.end(), wall_center));
                        qInfo() << "Wall center index: " << wall_center_index;
                        doors.push_back({wall_center_index, Eigen::Vector2f(p0_projected.x(), p0_projected.y()), Eigen::Vector2f(p1_projected.x(), p1_projected.y())});

                    }
                }
            }
        }
//        /// Iterate over doors
//        /// Check if doors exists in the graph
//        auto door_nodes = G->get_nodes_by_type("door");
//        if(door_nodes.empty())
//        {
//            /// Iterate over doors using enumerate
//            for(const auto &[i, door]: doors | iter::enumerate)
//            {
//                DoorDetector::Door act_door{std::get<1>(door), std::get<2>(door), 0, 0};
//                act_door.id = i;
//                insert_door_into_graph(act_door, std::get<0>(door));
//            }
//        }
//        else
//        {
//            /// If doors exists in graph, match found doors with graph doors using Hungarian algorithm
//            std::vector<std::vector<double>> distances_matrix(doors.size(), std::vector<double>(door_nodes.size()));
//            for(size_t i = 0; i < doors.size(); i++)
//            {
//                for(size_t j = 0; j < door_nodes.size(); j++)
//                {
//                    auto door_node = door_nodes[j];
//                    if(auto door_pos = G->get_attrib_by_name<pos_x_att>(door_node); door_pos.has_value())
//                    {
//                        auto door_pos_value = door_pos.value().get();
//                        Eigen::Vector2f door_pos_eigen(door_pos_value, G->get_attrib_by_name<pos_y_att>(door_node).value().get());
//                        distances_matrix[i][j] = (door_pos_eigen - (std::get<1>(doors[i]) + std::get<2>(doors[i]) / 2)).norm();
//                    }
//                }
//            }
//        }

        draw_door(doors, &widget_2d->scene, QColor("red"));
        if(widget_2d != nullptr)
        {
            draw_lidar(ldata, &widget_2d->scene);
            draw_polygon(poly_room_in, poly_room_out, &widget_2d->scene, QColor("blue"));

        }

        /// get room perimeter, lines in ranges of z
//        Lines lines = extract_lines(ldata_filtered.points, consts.ranges_list);


    /// Door detector
//    auto doors = door_detector.detect(lines, &widget_2d->scene, corners);
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

void SpecificWorker::insert_door_into_graph(const DoorDetector::Door &door, int wall_id)
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

    // Create door node
//    auto door_node = DSR::Node::create<door_node_type>("door__measured");

    std::vector<float> door_pos, door_orientation;

    auto door_node = DSR::Node::create<door_node_type>("door_"+std::to_string(door.id)+"_measured");
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

    auto nominal_door_node = DSR::Node::create<door_node_type>("door_"+std::to_string(door.id));
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
    if(auto door_transformed = inner_eigen->transform(wall_node.name(), door_robot_pos_point_double, robot_node.name()); door_transformed.has_value())
    {
        auto door_transformed_value = door_transformed.value();
        auto door_transformed_value_float = door_transformed_value.cast<float>();
        door_pos = {door_transformed_value_float.x(), door_transformed_value_float.y(), 0.f};
    }

    // Add edge between door and robot
    door_orientation = {0.f, 0.f, 0.f};
    rt->insert_or_assign_edge_RT(wall_node, nominal_door_node.id(), door_pos, door_orientation);


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

