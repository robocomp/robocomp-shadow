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

        // A thread for lidar reading is started
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        Period = 50;
        timer.start(Period);
    }

}

void SpecificWorker::compute()
{
// general goal: waits for a room to have a "current" self edge. Check if there are nominal doors in it, detect doors from sensors and try to match both sets.
//               The actions to take are:
//                   1. if there is NO nominal door matching a measured one in the room, start stabilization procedure
//                   2. if there is a nominal door matching a measured one, write in G the measured door as valid
//                   3. if there is a nominal door NOT matching a measured one in the room, IGNORE for now.

// requirements:
// 1. read lidar data
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value()) { return; }
    auto ldata = res_.value();
// 2. check for current room
    auto room_node_ = G->get_node("room");
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room node in graph. Waiting for room..."; return; }
    auto room_node = room_node_.value();
    // Check if robot node exists in graph
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; std::terminate(); }
    auto robot_node = robot_node_.value();
//3. get nominal and measured doors from graph from room in robot's frame
    auto door_nodes = get_measured_and_nominal_doors(room_node);
//4. get measured doors from lidar in robot's frame
    auto doors = get_doors(ldata, &widget_2d->scene, robot_node, room_node);
//5. matching function between DSR doors_nominals data and measured doors in last lidar. return matching and "other" doors.
// update measured_door corresponding to these match with new measure
    auto nominal_matches = door_matching(doors, std::get<1>(door_nodes));
//   6. remove from doors detected the doors matched with DSR data and make new association with the measure_door_nodes,
//   return matched doors and update measured_door nodes.
    update_and_remove_doors(nominal_matches, doors, std::get<1>(door_nodes), true, room_node);
    //  7 matching function between DSR doors_measured data and measured doors in last lidar. return matching and "other" doors.
    auto measure_matches = door_matching(doors, std::get<0>(door_nodes));
    /// Print measure matches
//  8 remove from observed doors detected the last matched doors. If door_detected>0 it's a door in process of stabilization or a new_door
//  and update the measured_nodes
    auto to_prefilter_doors = update_and_remove_doors(measure_matches, doors, std::get<0>(door_nodes), false, room_node);
//  9 Get the rest of doors observed and start the stabilization process. if door.size>0 (observed door not matched) it's a new door.
//    10 LAST MATCH: PREVIOUS INSERTION BUFFER NEEDED TO AVOID SPURIOUS, generate a vector of doors candidates to insertion.
//    If door has been seen for N times. insertion
    door_prefilter(to_prefilter_doors);
    qInfo() << "------------------------set_doors_to_stabilize--------------------------------";
    set_doors_to_stabilize(to_prefilter_doors, room_node);

//  10 create new_measured door to stabilize.
//        insert measured_id_door
//    start a thread for each door to stabilize. This thread would have to:
//          1. reset accumulators for trajectory
//          2. create "intention" edge, meaning that the agent has an intention to reach a target.
//              [offset::agent_id]
//              [offset::is_active] boolean signalling that the intention has been activated by the scheduler (false)
//              [offset::state] type enumerated mission status indicator (waiting, in progress, aborted, failed, completed) (waiting)
//              [offset::intention]  3D coordinates of a point wrt the object’s frame (3D pose from door, 1m front, final destination)
//                 [offset::orientation] 3 orientation angles wrt to the object’s frame. For a planar, 2D situation (angle between robot and door (180º)
//                 [0, 0, rz] is enough, and the robot’s Y axis must end aligned with a vector rotated rz in the object’s frame (Y+ -> 0º)
//              [tolerance::intention] 6-vector specifying the maximum deviation from the target pose (200,200,0,0,0,0.15)
//          3. wait for the "intention" is_active = true,  once the is active, start recording poses and landmarks
//
//          4. once the target is reached, (state, completed, is_active=false) delete the "intention" edge, generate opt petition, stabilize the object and create a new nominal_door in G

//          6. finish thread
//     the thread has to respond when compute asks if a given measured door is being stabilized by it.

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::pair<std::vector<DoorDetector::Door>, std::vector<DoorDetector::Door>> SpecificWorker::get_measured_and_nominal_doors(DSR::Node room_node)
{
    std::vector<DoorDetector::Door> nominal_doors, measured_doors;

    auto door_nodes = G->get_nodes_by_type("door");
    /// Iterate over door nodes and check if they contain "measured" in their name
    for(auto &n: door_nodes)
    {
        /// Get wall id knowing that is the 5 string element in the name
        auto wall_id = std::stoi(n.name().substr(5, 1));
        /// Get door id knowing that is the 7 string element in the name
        auto door_id = std::stoi(n.name().substr(7, 1));
        /// Get wall node
        auto wall_node_ = G->get_parent_node(n);
        if(not wall_node_.has_value())
        { qWarning() << __FUNCTION__ << " No wall node for door"; continue; }
        auto wall_node = wall_node_.value();
//        qInfo() << "Wall node: " << QString::fromStdString(wall_node.name());
        /// Get measured door RT edge
        if(auto rt_edge = rt->get_edge_RT(wall_node, n.id()); rt_edge.has_value())
        {
            auto edge = rt_edge.value();
            if(auto rt_translation = G->get_attrib_by_name<rt_translation_att>(edge); rt_translation.has_value())
            {
                auto rt_translation_value = rt_translation.value().get();
                /// Given door center, calculate door peaks considering width
                auto door_center = Eigen::Vector3d {rt_translation_value[0], rt_translation_value[1], 0};
                if(auto door_width = G->get_attrib_by_name<width_att>(n); door_width.has_value())
                {
                    auto p_0_ = Eigen::Vector3d {door_center.x() - (double)door_width.value() / 2, 0, 0};
                    auto p_1_ = Eigen::Vector3d {door_center.x() + (double)door_width.value() / 2, 0, 0};

//                    qInfo() << "NP_0: " << p_0_.x() << p_0_.y() << " P_1: " << p_1_.x() << p_1_.y();

                    if (auto p_0_transformed = inner_eigen->transform(room_node.name(),
                                                                      p_0_,
                                                                         wall_node.name()); p_0_transformed.has_value())
                    {
                        auto p_0 = p_0_transformed.value();
                        if (auto p_1_transformed = inner_eigen->transform(room_node.name(),
                                                                          p_1_,
                                                                             wall_node.name()); p_1_transformed.has_value())
                        {
//                            if (auto matrix = inner_eigen->get_transformation_matrix(room_node.name(),
//                                                                              wall_node.name()); matrix.has_value())
//                            {
//                                /// Considering matrix is a Mat::RTMat 4x4, print transflation and rotation
//                                auto mat = matrix.value();
//                                qInfo() << "Matrix: " << mat(0, 3) << mat(1, 3) << mat(2, 3);
//                                qInfo() << "Rotation: " << mat(0, 0) << mat(0, 1) << mat(0, 2) << mat(1, 0) << mat(1, 1) << mat(1, 2) << mat(2, 0) << mat(2, 1) << mat(2, 2);
//                                /// Transform p_0 and p_1 to room reference frame using matrix
//                                auto p_0_transformed = mat * p_0_;
//                                auto p_1_transformed = mat * p_1_;
//                                qInfo() << "transformed P_0: " << p_0_transformed.x() << p_0_transformed.y() << " P_1: " << p_1_transformed.x() << p_1_transformed.y();
//                            }

                            auto p_1 = p_1_transformed.value();
                            DoorDetector::Door door (Eigen::Vector2f{p_0.x(), p_0.y()}, Eigen::Vector2f{p_1.x(), p_1.y()}, 0, 0);
                            door.id = door_id;
                            door.wall_id = wall_id;
//                            qInfo() << "Wall id: " << wall_id << " Door id: " << door_id;
//                            qInfo() << "Door name: " << QString::fromStdString(n.name()) << " Width: " << door.width() << " Center: " << door.middle[0] << door.middle[1];
                            if(n.name().find("_pre") != std::string::npos)
                                measured_doors.push_back(door);
                            else
                                nominal_doors.push_back(door);
                        }
                    }

                }
            }
        }

    }
    return std::pair<std::vector<DoorDetector::Door>, std::vector<DoorDetector::Door>>{measured_doors, nominal_doors}; //Reference frame changed to wall
}

std::vector<DoorDetector::Door> SpecificWorker::get_doors(const RoboCompLidar3D::TData &ldata, QGraphicsScene *scene, DSR::Node robot_node, DSR::Node room_node)
{
    /// Get wall centers and corners from graph
    /// Get corners and wall centers
    auto corners_and_wall_centers = get_corners_and_wall_centers();
    if(not corners_and_wall_centers.has_value())
    { qWarning() << __FUNCTION__ << " No corners and wall centers detected"; return{}; }
    auto [corners, wall_centers] = corners_and_wall_centers.value();


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
    std::vector<DoorDetector::Door> doors;

    vector<tuple<int, int>> door_number_for_wall;
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
                    auto p0_projected = project_point_on_polygon(QPointF(p0.x, p0.y), nominal_polygon);
                    auto p1_projected = project_point_on_polygon(QPointF(p1.x, p1.y), nominal_polygon);
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
                    qInfo() << "Wall center index: " << wall_center_index;
                    //TODO: transform to room reference frame.
                    Eigen::Vector3d p_0_ {p0_projected_eigen.x(), p0_projected_eigen.y(), 0};
                    Eigen::Vector3d p_1_ {p1_projected_eigen.x(), p1_projected_eigen.y(), 0};
                    /// Transform p_0 and p_1 to room reference frame
                    if (auto p_0_transformed = inner_eigen->transform(room_node.name(),
                                                                      p_0_,
                                                                      robot_node.name()); p_0_transformed.has_value())
                    {
                        auto p_0 = p_0_transformed.value();
                        if (auto p_1_transformed = inner_eigen->transform(room_node.name(),
                                                                          p_1_,
                                                                          robot_node.name()); p_1_transformed.has_value())
                        {
                            auto p_1 = p_1_transformed.value();
                            DoorDetector::Door door (Eigen::Vector2f{p_0.x(), p_0.y()}, Eigen::Vector2f{p_1.x(), p_1.y()}, 0, 0);
                            /// Check if element in wall_center_index is not in door_number_for_wall
                            if(std::find_if(door_number_for_wall.begin(), door_number_for_wall.end(), [wall_center_index](const auto &t){ return std::get<0>(t) == wall_center_index; }) == door_number_for_wall.end())
                            {
                                door_number_for_wall.push_back(std::make_tuple(wall_center_index, 0));
                                door.id = 0;
                            }
                            else
                            {
                                auto it = std::find_if(door_number_for_wall.begin(), door_number_for_wall.end(), [wall_center_index](const auto &t){ return std::get<0>(t) == wall_center_index; });
                                door.id = std::get<1>(*it) + 1;
                                *it = std::make_tuple(wall_center_index, door.id);
                            }

                            door.wall_id = wall_center_index;

                            qInfo() << __FUNCTION__ << "Door: " << door.id << " Wall: " << door.wall_id;
                            doors.push_back(door);
                        }
                    }
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
            auto p0_projected = project_point_on_polygon(QPointF(p0.x, p0.y), nominal_polygon);
            auto p1_projected = project_point_on_polygon(QPointF(p1.x, p1.y), nominal_polygon);
            auto p0_projected_eigen = Eigen::Vector2f(p0_projected.x(), p0_projected.y());
            auto p1_projected_eigen = Eigen::Vector2f(p1_projected.x(), p1_projected.y());
            auto middle = (p0_projected_eigen + p1_projected_eigen) / 2;
            auto wall_center = *std::min_element(wall_centers.begin(), wall_centers.end(), [&middle](const Eigen::Vector2f &a, const Eigen::Vector2f &b){ return (a - middle).norm() < (b - middle).norm(); });
            auto wall_center_index = std::distance(wall_centers.begin(), std::find(wall_centers.begin(), wall_centers.end(), wall_center));
            qInfo() << "Wall center index: " << wall_center_index;
            //TODO: transform to room reference frame.
            Eigen::Vector3d p_0_ {p0_projected_eigen.x(), p0_projected_eigen.y(), 0};
            Eigen::Vector3d p_1_ {p1_projected_eigen.x(), p1_projected_eigen.y(), 0};
            /// Transform p_0 and p_1 to room reference frame
            if (auto p_0_transformed = inner_eigen->transform(room_node.name(),
                                                              p_0_,
                                                              robot_node.name()); p_0_transformed.has_value())
            {
                auto p_0 = p_0_transformed.value();
                if (auto p_1_transformed = inner_eigen->transform(room_node.name(),
                                                                  p_1_,
                                                                  robot_node.name()); p_1_transformed.has_value())
                {
                    auto p_1 = p_1_transformed.value();
                    //TODO: transform to room reference frame.
                    DoorDetector::Door door (Eigen::Vector2f{p_0.x(), p_0.y()}, Eigen::Vector2f{p_1.x(), p_1.y()}, 0, 0);
                    door.wall_id = wall_center_index;
                    /// Check if element in wall_center_index is not in door_number_for_wall
                    if(std::find_if(door_number_for_wall.begin(), door_number_for_wall.end(), [wall_center_index](const auto &t){ return std::get<0>(t) == wall_center_index; }) == door_number_for_wall.end())
                    {
                        door_number_for_wall.push_back(std::make_tuple(wall_center_index, 0));
                        door.id = 0;
                    }
                    else
                    {
                        auto it = std::find_if(door_number_for_wall.begin(), door_number_for_wall.end(), [wall_center_index](const auto &t){ return std::get<0>(t) == wall_center_index; });
                        door.id = std::get<1>(*it) + 1;
                        *it = std::make_tuple(wall_center_index, door.id);
                    }
                    qInfo() << "Door: " << door.id << " Wall: " << door.wall_id;
                    doors.push_back(door);
                }
            }
        }
    }

//TODO: FIX
//    // Create new poly_room_in and poly_room_out in the room reference frame
//    QPolygonF poly_room_in_room, poly_room_out_room;
//    for(auto &c: corners)
//    {
//        // Transform corners to room reference frame
//        if (auto c_transformed = inner_eigen->transform(robot_node.name(),
//                                                        Eigen::Vector3d{c.x(), c.y(), 0},
//                                                        room_node.name()); c_transformed.has_value())
//        {
//            auto c = c_transformed.value();
//            auto center = std::accumulate(corners.begin(), corners.end(), Eigen::Vector2f(0,0), [](const Eigen::Vector2f& acc, const Eigen::Vector2f& c){ return acc + c; }) / corners.size();
//            auto dir = (center - c).normalized();
//            Eigen::Vector2f new_corner_in = c + dir * d;
//            poly_room_in_room << QPointF(new_corner_in.x(), new_corner_in.y());
//            Eigen::Vector2f new_corner_out = c - dir * d;
//            poly_room_out_room << QPointF(new_corner_out.x(), new_corner_out.y());
//        }
//    }

    draw_polygon(poly_room_in, poly_room_out, &widget_2d->scene, QColor("blue"));
//
//    draw_polygon(poly_room_in, poly_room_out, &widget_2d->scene, QColor("blue"));
    draw_door(doors, &widget_2d->scene, QColor("red"));
    return doors;
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

// Función para calcular la proyección más cercana de un punto sobre un polígono
Eigen::Vector2f SpecificWorker::project_point_on_polygon(const QPointF& p, const QPolygonF& polygon)
{
    QPointF closestProjection;
    double minDistanceSquared = std::numeric_limits<double>::max();

    for (int i = 0; i < polygon.size(); ++i) {
        QPointF v = polygon[i];
        QPointF w = polygon[(i + 1) % polygon.size()]; // Ciclar al primer punto

        QPointF projection = project_point_on_line_segment(p, v, w);
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

// Función para calcular la proyección de un punto sobre un segmento de línea
QPointF SpecificWorker::project_point_on_line_segment(const QPointF& p, const QPointF& v, const QPointF& w) {
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
/**
 * @brief This function filters the detected doors based on their occurrence in consecutive frames.
 *
 * @param detected_door A vector of detected doors. This vector is updated in-place to contain only the doors that have been detected consistently over N frames.
 */
void SpecificWorker::door_prefilter(vector<DoorDetector::Door> &detected_door)
{
    // A static vector of tuples to store the last detected doors and a counter to store the number of times a door has been detected
    static vector<tuple<DoorDetector::Door, int>> last_detected_doors;
    int N = 40; // The number of consecutive frames a door must be detected in to be considered valid

    for(const auto &[i, d] : last_detected_doors | iter::enumerate)
    {
        if(get<1>(d) == N)
            last_detected_doors.erase(last_detected_doors.begin() + i);
    }

//    qInfo() << "DOOR PREFILTER";
    // If detected_door is empty, clear the last_detected_doors vector and return
    if(detected_door.empty())
    {
//        qInfo() << "--------------NO DOORS--------------------";
        last_detected_doors.clear();
        return;
    }
    if(last_detected_doors.empty())
    {
//        qInfo() << "--------------FIRST DOORS DETECTED--------------------";
        for(auto &d: detected_door)
        {
            qInfo() << "First doors detected: " << d.wall_id << d.id;
            last_detected_doors.push_back({d, 1});
        }

        detected_door.clear();
        return;
    }

    // Create a matrix of distances between the last detected doors and the current detected doors
    vector<vector<double>> distances_matrix(last_detected_doors.size(), vector<double>(detected_door.size()));
    for(size_t i = 0; i < last_detected_doors.size(); i++)
        for(size_t j = 0; j < detected_door.size(); j++)
        {
            distances_matrix[i][j] = (get<0>(last_detected_doors[i]).middle - detected_door[j].middle).norm();
        }


    /// Draw distances matrix in matrix form
    qInfo() << "Distance Matrix prefilter";
    for(size_t i = 0; i < distances_matrix.size(); i++)
        for(size_t j = 0; j < distances_matrix[i].size(); j++)
            qInfo() << distances_matrix[i][j];
    qInfo() << "---------------";

    // Use the Hungarian algorithm to solve the assignment problem
    vector<int> assignment;
    double cost = HungAlgo.Solve(distances_matrix, assignment);
    // Create a vector to store the filtered doors
    vector<DoorDetector::Door> filtered_doors;

    // For each matched door, increase the counter. If counter == N, push the door to the filtered_doors vector and
    // remove the door from the last_detected_doors vector. If not matched, remove the door from the
    // last_detected_doors vector
    for(size_t i = 0; i < assignment.size(); i++)
    {
        qInfo() << "Assignment: " << i << " --- " << assignment[i];
        if(assignment[i] != -1)
        {
            qInfo() << "Match condition: " << distances_matrix[i][assignment[i]] << " < " << get<0>(last_detected_doors[assignment[i]]).width() * 0.75;

            if(distances_matrix[i][assignment[i]] < detected_door[assignment[i]].width() * 0.75)
            {
                qInfo() << "Matching: " << detected_door[i].wall_id << detected_door[i].id << " --- " << get<0>(last_detected_doors[assignment[i]]).wall_id << get<0>(last_detected_doors[assignment[i]]).id;
                qInfo() << "Mid points" << detected_door[i].middle[0] << detected_door[i].middle[1] << get<0>(last_detected_doors[assignment[i]]).middle[0] << get<0>(last_detected_doors[assignment[i]]).middle[1];
                get<1>(last_detected_doors[i])++;
                if(get<1>(last_detected_doors[i]) == N)
                {
                    filtered_doors.push_back(get<0>(last_detected_doors[i]));
                }
            }
            else
                last_detected_doors.erase(last_detected_doors.begin() + i);
        }
        else
            last_detected_doors.erase(last_detected_doors.begin() + i);
    }
    // Assign to detected_door the filtered_doors vector

    detected_door = filtered_doors;
}
std::vector<std::pair<int, int>> SpecificWorker::door_matching(const std::vector<DoorDetector::Door> &measured_doors, const std::vector<DoorDetector::Door> &nominal_doors)
{
    vector<pair<int, int>> matching;
    if(measured_doors.empty())
    {
        /// Iterate over nominal doors and insert -1 in tuple
        for(int i = 0; i < nominal_doors.size(); i++)
            matching.push_back({-1, i});
        return matching;
    }
    if(nominal_doors.empty())
    {
        for(int i = 0; i < measured_doors.size(); i++)
            matching.push_back({i, -1});
        return matching;
    }


    vector<vector<double>> distances_matrix(measured_doors.size(), vector<double>(nominal_doors.size()));
    for(size_t i = 0; i < measured_doors.size(); i++)
        for(size_t j = 0; j < nominal_doors.size(); j++)
        {
            distances_matrix[i][j] = (measured_doors[i].middle - nominal_doors[j].middle).norm(); //TODO: incorporate rotation or door width to distance_matrix
//            qInfo() << "Distance: " << measured_doors[i].wall_id << measured_doors[i].id << " --- " << nominal_doors[j].wall_id << nominal_doors[j].id << distances_matrix[i][j];
        }
//    qInfo() << "Distance Matrix";
    for(size_t i = 0; i < distances_matrix.size(); i++)
        for(size_t j = 0; j < distances_matrix[i].size(); j++)
            qInfo() << distances_matrix[i][j];
//    qInfo() << "---------------";
    vector<int> assignment;
    double cost = HungAlgo.Solve(distances_matrix, assignment);
    for(size_t i = 0; i < assignment.size(); i++)
    {
//        qInfo() << "Assignment: " << i << " --- " << assignment[i];
        if(assignment[i] != -1)
        {
//            qInfo() << "Match condition: " << distances_matrix[i][assignment[i]] << " < " << nominal_doors[assignment[i]].width() * 0.75;
            if(distances_matrix[i][assignment[i]] < nominal_doors[assignment[i]].width() * 0.75)
            {
//                qInfo() << "Matching: " << measured_doors[i].wall_id << measured_doors[i].id << " --- " << nominal_doors[assignment[i]].wall_id << nominal_doors[assignment[i]].id;
                matching.push_back({i, assignment[i]});
            }
            else
             matching.push_back({i, -1});
        }
    }

    //TODO: PRINT COMPLETE MATCHING



    qInfo() << "########################################################################";
    return matching;
}
std::vector<DoorDetector::Door> SpecificWorker::update_and_remove_doors(std::vector<std::pair<int, int>> matches, const std::vector<DoorDetector::Door> &measured_doors, const std::vector<DoorDetector::Door> &graph_doors, bool nominal, DSR::Node room_node)
{
    /// vector for indexes to remove at the end of the function
    std::vector<int> indexes_to_remove;
    std::vector<DoorDetector::Door> not_associated_doors;
    /// Iterate over matches using enumerate
    for(const auto &[i, match] : matches | iter::enumerate)
        if(match.second != -1 and match.first != -1)
        {
//            if(nominal)
//            {
//                qInfo() << "Deleting because is matched with a nominal";
//                measured_doors.erase(measured_doors.begin() + i);
//                continue;
//            }
//            qInfo() << "Updating door data in graph";
            /// Get graph door name
            auto door_name = "door_" + std::to_string(graph_doors[match.second].wall_id) + "_" + std::to_string(graph_doors[match.second].id) + "_pre";
            qInfo() << "Door name first: " << QString::fromStdString(door_name) << "Mid point: " << measured_doors[match.first].middle.x() << measured_doors[match.first].middle.y() ;
            qInfo() << "Door name second: " << QString::fromStdString(door_name) << "Mid point: " << measured_doors[match.second].middle.x() << measured_doors[match.second].middle.y() ;
            /// Update measured door with the matched door
            update_door_in_graph(measured_doors[match.first], door_name, room_node);
            /// Insert in indexes_to_remove vector the index of the door to remove
            indexes_to_remove.push_back(match.first);
        }
    /// Insert in not_associated_doors vector the doors that are not associated using indexes_to_remove
    for(const auto &[i, door] : measured_doors | iter::enumerate)
        if(std::find(indexes_to_remove.begin(), indexes_to_remove.end(), i) == indexes_to_remove.end())
            not_associated_doors.push_back(door);
    return not_associated_doors;
}
void SpecificWorker::update_door_in_graph(const DoorDetector::Door &door, std::string door_name, DSR::Node room_node)
{
    auto door_node_ = G->get_node(door_name);
    if(not door_node_.has_value())
    { qWarning() << __FUNCTION__ << " No door node in graph"; return; }
    auto door_node = door_node_.value();

    /// Get parent node
    auto parent_node_ = G->get_parent_node(door_node);
    if(not parent_node_.has_value())
    { qWarning() << __FUNCTION__ << " No parent node in graph"; return; }
    auto parent_node = parent_node_.value();

    auto pose = door.middle;
    qInfo() << "Updating node" << door_node.id() << "Name:" << QString::fromStdString(door_node.name());
    qInfo() << "Pose to update" << pose[0] << pose[1];
    /// Transform pose to parent node reference frame
    if(auto pose_transformed_ = inner_eigen->transform(parent_node.name(), Eigen::Vector3d{pose[0], pose[1], 0.f}, room_node.name()); pose_transformed_.has_value())
    {
        auto pose_transformed = pose_transformed_.value().cast<float>();
        qInfo() << "Pose to update transformed" << pose_transformed.x() << pose_transformed.y();
        // Add edge between door and robot
        rt->insert_or_assign_edge_RT(parent_node, door_node.id(), {pose_transformed.x(), 0.f, 0.f}, {0.f, 0.f, 0.f});
    }
}
//Function to insert a new measured door in the graph, generate a thread in charge of stabilizing the door and set a has_intention edge
void SpecificWorker::set_doors_to_stabilize(const std::vector<DoorDetector::Door> &doors, DSR::Node room_node)
{
    qInfo() << "Door sizes set_doors_to_stabilize: " << doors.size();
    /// Iterate over doors using enumerate
    for(const auto &[i, door] : doors | iter::enumerate)
    {

        //Locate the wall where the door is
        auto wall_id = door.wall_id;
        qInfo() << "WALL ID PREOCUPANTE" << wall_id;
        //find the highest door id in the wall
        auto door_nodes = G->get_nodes_by_type("door");
        auto door_ids = std::vector<int>();
        for(auto &n: door_nodes)
            if(n.name().find("door") != std::string::npos)
            {
                auto wall_id_graph = std::stoi(n.name().substr(5, 1));
                auto door_id_graph = std::stoi(n.name().substr(7, 1));
                if(wall_id == wall_id_graph)
                    door_ids.push_back(door_id_graph);
            }
        int new_door_id;
        if(door_ids.empty())
            new_door_id = 0;
        else
            new_door_id = *std::max_element(door_ids.begin(), door_ids.end()) + 1;
//        std::string door_name = "door_" + std::to_string(wall_id) + "_" + std::to_string(new_door_id) + "_measured";
        std::string door_name = "door_" + std::to_string(wall_id) + "_" + std::to_string(new_door_id) + "_pre";
        qInfo() << "Door name: " << QString::fromStdString(door_name) << "Mid point: " << door.middle.x() << door.middle.y() << "Width: " << door.width();
        //  10 create new_measured door to stabilize.
        insert_measured_door_in_graph(door, room_node, door_name);

        //TODO: Meter robot node como parametro de la función
        //Create and has_intention edge from robot to door
        auto robot_node_ = G->get_node("Shadow");
        auto door_node = G->get_node(door_name);

        if(not door_node.has_value())
        { qWarning() << __FUNCTION__ << " No door node in graph"; return; }

        if(not robot_node_.has_value())
        { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
        //Create and has_intention edge from robot to door
        auto robot_node = robot_node_.value();
//        G->insert_edge(robot_node, door_node.value().id(), "has_intention", {{"is_active", false}, {"state", "waiting"}});
        DSR::Edge intention_edge = DSR::Edge::create<has_intention_edge_type>(robot_node.id(), door_node.value().id());
        G->add_or_modify_attrib_local<is_active_att>(intention_edge, false);
        G->insert_or_assign_edge(intention_edge);

//        insert measured_id_door
//    start a thread for each door to stabilize. This thread would have to:
        /// Generate a thread for each door to stabilize

        std::thread stabilize_thread(&SpecificWorker::stabilize_door, this, door, door_name);
        stabilize_thread.detach();
        sleep(3);
        G->add_or_modify_attrib_local<is_active_att>(intention_edge, true);
        G->insert_or_assign_edge(intention_edge);
        sleep(4);
        G->add_or_modify_attrib_local<is_active_att>(intention_edge, false);
        G->insert_or_assign_edge(intention_edge);

//          1. reset accumulators for trajectory
//          2. create "intention" edge, meaning that the agent has an intention to reach a target.
//              [offset::agent_id]
//              [offset::is_active] boolean signalling that the intention has been activated by the scheduler (false)
//              [offset::state] type enumerated mission status indicator (waiting, in progress, aborted, failed, completed) (waiting)
//              [offset::intention]  3D coordinates of a point wrt the object’s frame (3D pose from door, 1m front, final destination)
//                 [offset::orientation] 3 orientation angles wrt to the object’s frame. For a planar, 2D situation (angle between robot and door (180º)
//                 [0, 0, rz] is enough, and the robot’s Y axis must end aligned with a vector rotated rz in the object’s frame (Y+ -> 0º)
//              [tolerance::intention] 6-vector specifying the maximum deviation from the target pose (200,200,0,0,0,0.15)
//          3. wait for the "intention" is_active = true,  once the is active, start recording poses and landmarks
//
//          4. once the target is reached, (state, completed, is_active=false) delete the "intention" edge, generate opt petition, stabilize the object and create a new nominal_door in G

//          6. finish thread
//     the thread has to respond when compute asks if a given measured door is being stabilized by it.
    }

}
void SpecificWorker::insert_measured_door_in_graph(DoorDetector::Door door, DSR::Node room, std::string door_name)
{
    /// Get wall node from door id
    auto wall_node_ = G->get_node("wall_" + std::to_string(door.wall_id));
    if(not wall_node_.has_value())
    { qWarning() << __FUNCTION__ << " No wall node in graph"; return; }

    auto wall_node = wall_node_.value();
    auto door_node = DSR::Node::create<door_node_type>(door_name);

    // Transform door middle to wall reference frame
    auto door_middle = Eigen::Vector3d {door.middle.x(), door.middle.y(), 0.0};

    if( auto door_middle_transformed = inner_eigen->transform(wall_node.name(), door_middle, room.name()); door_middle_transformed.has_value())
    {
        auto door_middle_transformed_value = door_middle_transformed.value().cast<float>();
        // Get wall node level
        auto wall_node_level = G->get_attrib_by_name<level_att>(wall_node).value();

        // get wall node pos_x and pos_y attributes
        auto wall_node_pos_x = G->get_attrib_by_name<pos_x_att>(wall_node).value();
        auto wall_node_pos_y = G->get_attrib_by_name<pos_y_att>(wall_node).value();

        // Add door attributes
        // set pos_x and pos_y attributes to door node
        G->add_or_modify_attrib_local<pos_x_att>(door_node, wall_node_pos_x + (int)(door_middle_transformed_value.x() / 20.0) );
        G->add_or_modify_attrib_local<pos_y_att>(door_node, wall_node_pos_y + (int)(door_middle_transformed_value.y() / 20.0));
        G->add_or_modify_attrib_local<width_att>(door_node, (int)door.width());
        G->add_or_modify_attrib_local<level_att>(door_node, wall_node_level + 1);
        G->insert_node(door_node);

        auto pose = door.middle;
        qInfo() << " name: " << QString::fromStdString(door_name) << " Width: " << door.width() << " Center: " << door_middle_transformed_value.x() << door_middle_transformed_value.y();
        // Add edge between door and robot
        rt->insert_or_assign_edge_RT(wall_node, door_node.id(), {door_middle_transformed_value.x(),door_middle_transformed_value.y(), 0.0},
                                     {0.0, 0.0, 0.0});
    }
}
void SpecificWorker::stabilize_door(DoorDetector::Door door, std::string door_name)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto door_data = door;
    auto door_name_ = door_name;
    bool is_stabilized = false;

    std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> measured_corner_data;
    std::vector<std::vector<float>> odometry_data;
    std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> measured_door_points;
    /// Generate static map for histogram with coordinate in wall and door width
    std::map<int, int> width_histogram;
    std::map<int, int> pose_histogram;
    Eigen::Affine2d first_robot_pose;
    bool first_time = true;

    while(not is_stabilized)
    {

        auto start = std::chrono::high_resolution_clock::now();

        /// get has intention edge between robot and doo
        auto robot_node_ = G->get_node("Shadow");
        if(not robot_node_.has_value())
        { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
        auto robot_node = robot_node_.value();
        auto door_node_ = G->get_node(door_name_);
        if(not door_node_.has_value())
        { qWarning() << __FUNCTION__ << " No door node in graph"; return; }
        auto door_node = door_node_.value();

        auto intention_edge_ = G->get_edge(robot_node, door_node.id(), "has_intention");
        //get is active attribute from intention edge
        auto is_active = G->get_attrib_by_name<is_active_att>(intention_edge_.value());

        if(not is_active.has_value())
        { qWarning() << __FUNCTION__ << " No is_active attribute in graph"; return; }

        std::cout << "Door name: " << door_name_ << " is_active: " << is_active.value() << std::endl;
        if(is_active.value())
        {
            //print is active attribute
            qInfo() << "Door is active";
            qInfo() << "Stabilizing door: " << QString::fromStdString(door_name_);
            //TODO:: Check if robot is in has_intention target position

            if(first_time)
            {
                first_time = false;
                /// Get room node
                auto room_node_ = G->get_node("room");
                if(not room_node_.has_value())
                { qWarning() << __FUNCTION__ << " No room node in graph"; return; }
                auto room_node = room_node_.value();

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

                // Get door corners transformed to robot reference frame using inner_eigen transform
                //TODO: Pablo, aquí renunciamos.
                //TODO: ALMACENAR DOOR MIDDLE EN EL SISTEMA DE REFERENCIA DEL ROBOT Y ANCHO DE PUERTA
                //TODO:: ALMACENAR HISTOGRAMAS (CENTRO, ANCHO).
                //TODO:: GET NOMINAL CORNER & MEASUREMENT Y ODOMETRÍA PARA MONTAR G2O

            }
            else
            {
                //TODO:: ALMACENAR HISTOGRAMAS (CENTRO, ANCHO).
                //TODO: ESPERAR A IS_ACTIVE == FALSE, STATE=="completed" PARA MONTAR ARCHIVO G20, PROXY->G2O, RECOGER PUERTA OPTIMIZADA
                //TODO: INSERTAR PUERTA OPTIMIZADA, ELIMINAR INTENTION EDGE
            }
        }
        else
        {
            qInfo() << "Door is not active";
        }

        auto elapsed_total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        std::this_thread::sleep_for(std::chrono::milliseconds(50-elapsed_total.count()));
    }
    /// Kill thread
    qInfo() << "Thread finished";
}
////////////////////// LIDAR /////////////////////////////////////////////////
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
///////////////////// Draw  /////////////////////////////////////////////////////////////
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
void SpecificWorker::draw_door(const std::vector<DoorDetector::Door> &doors, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    Eigen::Vector2i offset(40, 40);

    //Draw ellipse in the doors corners
    for(const auto &door: doors)
    {
        // draw points inside doors
        for(const auto &door: doors)
        {
            auto item = scene->addLine(door.p0.x(), door.p0.y(), door.p1.x(), door.p1.y(), QPen(QColor("red"), 40));
            items.push_back(item);
        }

        auto item = scene->addEllipse(-40, -40, 80, 80, QPen(QColor("red"), 40), QBrush(QColor("red")));
        item->setPos(door.p0.x(), door.p0.y());
        items.push_back(item);

        auto item2 = scene->addEllipse(-40, -40, 80, 80, QPen(QColor("red"), 40), QBrush(QColor("red")));
        item2->setPos(door.p1.x(), door.p1.y());
        items.push_back(item2);
        // Draw Coordinate x,y of the corner
        auto item_text = scene->addText(QString::number(door.p0.x(), 'f', 2) + QString::fromStdString(",") + QString::number(door.p0.y(), 'f', 2), QFont("Arial", 100));
        item_text->setPos(door.p0.x() + offset.x(), door.p0.y() + offset.y());
        item_text->setTransform(QTransform().scale(1, -1));
        items.push_back(item_text);



        // Draw Coordinate x,y of the corner
        auto item_text2 = scene->addText(QString::number(door.p1.x(), 'f', 2) + QString::fromStdString(",") + QString::number(door.p1.y(), 'f', 2), QFont("Arial", 100));
        item_text2->setPos(door.p1.x() + offset.x(), door.p1.y() + offset.y());
        item_text2->setTransform(QTransform().scale(1, -1));
        items.push_back(item_text2);

    }

    //Draw door id in the middle of the door
    for(const auto &door: doors)
    {

        auto item = scene->addText(QString::number(door.id) + QString::fromStdString("_") + QString::number(door.wall_id), QFont("Arial", 300));
        item->setPos(door.middle.x() + offset.x(), door.middle.y() + offset.y());
        item->setTransform(QTransform().scale(1, -1));
        items.push_back(item);
    }

}
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}




/**************************************/
// From the RoboCompG2Ooptimizer you can call this methods:
// this->g2ooptimizer_proxy->optimize(...)

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
