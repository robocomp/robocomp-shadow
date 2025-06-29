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
SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
        #endif


        //dsr update signals
        //connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
        connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
        //connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
//        connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
        //connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
        //connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

        QLoggingCategory::setFilterRules("*.debug=false\n");
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}
/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
    std::cout << "Destroying SpecificWorker" << std::endl;
}

void SpecificWorker::initialize()
{
    std::cout << "Initialize worker" << std::endl;

    // Graph viewer
    using opts = DSR::DSRViewer::view;
    int current_opts = 0;
    opts main = opts::none;
    if(this->configLoader.get<bool>("ViewAgent.tree"))
    {
        current_opts = current_opts | opts::tree;
    }
    if(this->configLoader.get<bool>("ViewAgent.graph"))
    {
        current_opts = current_opts | opts::graph;
        main = opts::graph;
    }
    if(this->configLoader.get<bool>("ViewAgent.2d"))
    {
        current_opts = current_opts | opts::scene;
    }
    if(this->configLoader.get<bool>("ViewAgent.3d"))
    {
        current_opts = current_opts | opts::osg;
    }

    graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
    std::cout << "Initialize worker" << std::endl;
    setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));
    std::cout << "Initialize worker" << std::endl;
    widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(DSR::DSRViewer::view::scene));
    std::cout << "Initialize worker" << std::endl;
    // A thread for lidar reading is started
//        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
//        std::cout << "Started lidar reader" << std::endl;

    room_widget = new CustomWidget();
    graph_viewer->add_custom_widget_to_dock("room view", room_widget);
    std::cout << "Initialize worker" << std::endl;
    rt = G->get_rt_api();
    inner_eigen = G->get_inner_eigen_api();

    /// Clear doors when room_widget->ui->pushButton_stop is clicked
    connect(room_widget->ui->pushButton_stop, &QPushButton::clicked, this, [this](){ this->clear_doors(); });
}
void SpecificWorker::compute()
{
//    qInfo() << "Wait to current edge" << wait_to_current;
// general goal: waits for a room to have a "current" self edge. Check if there are nominal doors in it, detect doors from sensors and try to match both sets.
//               The actions to take are:
//                   1. if there is NO nominal door matching a measured one in the room, start stabilization procedure
//                   2. if there is a nominal door matching a measured one, write in G the measured door as valid
//                   3. if there is a nominal door NOT matching a measured one in the room, IGNORE for now.

// requirements:
// 1. read lidar data
    auto res_ = buffer_lidar_data.try_get();
    if(!res_.has_value()) return;
    auto [lidar_timestamp, lidar_data] = res_.value();

    std::sort(lidar_data.begin(), lidar_data.end(),
              [](const Eigen::Vector3f& a, const Eigen::Vector3f& b)
              {
                  float angle_a = std::atan2(a.y(), a.x());
                  float angle_b = std::atan2(b.y(), b.x());
                  return angle_a < angle_b;
              });

    // Check if robot node exists in graph
    auto robot_node_ = G->get_node("Shadow");
    if(not robot_node_.has_value())
    { qWarning() << __FUNCTION__ << " No robot node in graph"; std::terminate(); }
    auto robot_node = robot_node_.value();

    // Check

    // Check if nodes_to_remove is not empty
    if(not nodes_to_remove.empty())
    {
        for(auto node_id : nodes_to_remove)
            delete_pre_node(node_id);
        nodes_to_remove.clear();
    }

// 2. check for current room
    auto current_edges = G->get_edges_by_type("current");
    if(current_edges.empty())
    {
        qWarning() << __FUNCTION__ << " No current edges in graph";
        return;
    //     // Check if "exit" door exists in graph
    //     auto exit_edges = G->get_edges_by_type("exit");
    //     if(exit_edges.empty())
    //     { qWarning() << __FUNCTION__ << " No exit edges in graph"; return;}

    //     auto exit_node_ = G->get_node(exit_edges[0].to());
    //     if(not exit_node_.has_value())
    //     { qWarning() << __FUNCTION__ << " No exit node in graph"; return; }
    //     auto exit_node = exit_node_.value();

    //     auto exit_node_room_id = G->get_attrib_by_name<room_id_att>(exit_node.id());
    //     if(not exit_node_room_id.has_value())
    //     { qWarning() << __FUNCTION__ << " No room id att in graph"; return; }
    //     auto exit_node_room_id_ = exit_node_room_id.value();

    //     //get robot parent node
    //     auto robot_parent = G->get_attrib_by_name<parent_att>(robot_node.id());
    //     if(not robot_parent.has_value())
    //     { qWarning() << __FUNCTION__ << " No robot parent node in graph"; return; }
    //     auto robot_parent_node_ = G->get_node(robot_parent.value());
    //     if(not robot_parent_node_.has_value())
    //     { qWarning() << __FUNCTION__ << " No robot parent node in graph"; return; }
    //     auto robot_parent_node = robot_parent_node_.value();
    //     // Get parent node name room_id
    //     auto room_id_ = G->get_attrib_by_name<room_id_att>(robot_parent_node.id());
    //     if(not room_id_.has_value())
    //     { qWarning() << __FUNCTION__ << " No room id att in graph"; return; }
    //     auto room_id = room_id_.value();

    //     if(exit_node_room_id_ == room_id)
    //     {
    //         if (auto exit_door_robot_pose_ = inner_eigen->transform(robot_node.name(),
    //                                                                 exit_node.name()); exit_door_robot_pose_.has_value())
    //         {
    //             auto exit_door_robot_pose = exit_door_robot_pose_.value();
    //             std::cout << "Exit door robot pose: " << exit_door_robot_pose.x() << " " << exit_door_robot_pose.y() << std::endl;
    //             exit_door_center = exit_door_robot_pose;
    //             exit_door_exists = true;
    //         }
    //     }
    //     return;
    }

    if(wait_to_current)
    {
//        qInfo() << __FUNCTION__ << " Waiting for current edge to be removed";
        return;
    }

    // -------- WAIT TO CROSS
    // Check if affordance cross is active
    affordance();
    auto affordance_nodes = G->get_nodes_by_type("affordance");
    auto affordance_active = std::ranges::find_if(affordance_nodes, [this](DSR::Node n) { return G->get_attrib_by_name<active_att>(n.id()).value() and n.name().find("cross") != std::string::npos; });
    // Check if affordance_active is not end of affordance_nodes
    if(affordance_active != affordance_nodes.end())
    {
        std::cout << "Affordance node in progress.Waiting to cross" << std::endl;
        wait_to_current = true;
        return; // Wait for affordance to finish
    }



    auto room_node_ = G->get_node(current_edges[0].to());
    if(not room_node_.has_value())
    { qWarning() << __FUNCTION__ << " No room level in graph"; return; }
    auto room_node = room_node_.value();
    std::cout << "Room node name: " << room_node.name() << std::endl;

    // Get room id att from room node
    auto room_id_att_ = G->get_attrib_by_name<room_id_att>(room_node.id());
    if(not room_id_att_.has_value())
    { qWarning() << __FUNCTION__ << " No room id att in graph"; return; }
    actual_room_id = room_id_att_.value();

//3. get nominal and measured doors from graph from room in robot's frame
    auto door_nodes = get_measured_and_nominal_doors(room_node, robot_node, lidar_timestamp);
    //Draw nominal doors in green and measured doors in red
    if(widget_2d != nullptr)
    {
        draw_lidar(lidar_data, &widget_2d->scene);
        draw_door_robot_frame(std::get<1>(door_nodes), std::get<0>(door_nodes), QColor("blue"), QColor("red"), &widget_2d->scene);

    }
//4. get measured doors from lidar in robot's frame
    auto doors = get_doors(lidar_data, &widget_2d->scene, robot_node, room_node, lidar_timestamp);
//    draw_door_robot_frame(doors, std::vector<DoorDetector::Door>{}, QColor("blue"), QColor("red"), &widget_2d->scene);
//5. matching function between DSR doors_nominals data and measured doors in last lidar. return matching and "other" doors.
// update measured_door corresponding to these match with new measure
    auto nominal_matches = door_matching(doors, std::get<1>(door_nodes));
//   6. remove from doors detected the doors matched with DSR data and make new association with the measure_door_nodes,
//   return matched doors and update measured_door nodes.
    auto to_measured_doors = update_and_remove_doors(nominal_matches, doors, std::get<1>(door_nodes), true, room_node, lidar_timestamp);
//    // Get affordance nodes
//    auto affordance_nodes = G->get_nodes_by_type("affordance");
//    // Check if any affordance node is active using a lambda function
//    auto affordance_active = std::ranges::find_if(affordance_nodes, [this](DSR::Node n) { return G->get_attrib_by_name<active_att>(n.id()).value(); });
//    if(affordance_active != affordance_nodes.end())
//    {
//        // Print affordance node active
//        std::cout << "Affordance node active" << std::endl;
//        affordance();
//        return;
//    }
    //Print to_measured_doors size
//    std::cout << "TO MEASURED DOORS SIZE" << to_measured_doors.size() << std::endl;

    //  7 matching function between DSR doors_measured data and measured doors in last lidar. return matching and "other" doors.
    auto measure_matches = door_matching(to_measured_doors, std::get<0>(door_nodes));
//    std::cout << "Measure match size: " << measure_matches.size() << std::endl;
    /// Print measure matches
//  8 remove from observed doors detected the last matched doors. If door_detected>0 it's a door in process of stabilization or a new_door
//  and update the measured_nodes
    auto to_prefilter_doors = update_and_remove_doors(measure_matches, to_measured_doors, std::get<0>(door_nodes), false, room_node, lidar_timestamp);

//    std::cout << "to_prefilter_doors match size: " << to_prefilter_doors.size() << std::endl;

    //  9 Get the rest of doors observed and start the stabilization process. if door.size>0 (observed door not matched) it's a new door.
//    10 LAST MATCH: PREVIOUS INSERTION BUFFER NEEDED TO AVOID SPURIOUS, generate a vector of doors candidates to insertion.
//    If door has been seen for N times. insertion

    door_prefilter(to_prefilter_doors);

//    qInfo() << "To prefilter doors size: " << to_prefilter_doors.size();

//    // Check if any affordance node is active using a lambda function
//    auto affordance_nodes = G->get_nodes_by_type("affordance");
//    auto affordance_active = std::ranges::find_if(affordance_nodes, [this](DSR::Node n) { return G->get_attrib_by_name<active_att>(n.id()).value(); });
//    if(affordance_active == affordance_nodes.end())
//        set_doors_to_stabilize(to_prefilter_doors, room_node);
//    qInfo() << "ROBOT INSIDE POLYGON" << inside_polygon;
    if(inside_polygon)
        set_doors_to_stabilize(to_prefilter_doors, room_node, lidar_timestamp);


    auto exit_edges = G->get_edges_by_type("exit");if(!exit_edges.empty() and exit_door_exists)
    {    match_exit_door(); }
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
void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}



//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}
void SpecificWorker::affordance()
{

    static std::map<std::uint64_t , std::thread> threads;

    auto nominal_doors = G->get_nodes_by_type("door");

    if (nominal_doors.empty())
    { qWarning() << __FUNCTION__ << " No nominal doors in graph"; return; }

    //print nominal doors size
    auto edges = G->get_edges_by_type("has");

    for (const auto &door : nominal_doors)
    {
        //check if door node name does not contain "pre"
        if (door.name().find("_pre") == std::string::npos)
        {
            if (auto r = std::ranges::find_if(edges, [&door](DSR::Edge e) { return e.from() == door.id(); }); r == edges.end())
            {
                std::cout << "No affordance node found, inserting" << std::endl;

                string aux = "";
                if(door.name().find("_") != std::string::npos)
                    aux = door.name().substr(door.name().find("_"));

                //get pos_x_att and pos_y_att from door node
                auto pos_x = G->get_attrib_by_name<pos_x_att>(door.id()).value();
                auto pos_y = G->get_attrib_by_name<pos_y_att>(door.id()).value();

                //create DSR::Node of type affordance
                DSR::Node affordance = DSR::Node::create<affordance_node_type>("aff_cross" + aux);
                G->add_or_modify_attrib_local<parent_att>(affordance, door.id());
                G->add_or_modify_attrib_local<active_att>(affordance, false);
                G->add_or_modify_attrib_local<bt_state_att>(affordance, std::string("waiting"));
                G->add_or_modify_attrib_local<pos_x_att>(affordance, (float)(pos_x) );
                G->add_or_modify_attrib_local<pos_y_att>(affordance, pos_y + 25);
                G->add_or_modify_attrib_local<room_id_att>(affordance, actual_room_id);
                if (auto door_level = G->get_attrib_by_name<level_att>(door.id()) ; door_level.has_value())
                    G->add_or_modify_attrib_local<level_att>(affordance,  door_level.value() + 1);

                //TODO: CHANGE
                if (auto aff_id = G->insert_node(affordance); aff_id.has_value())
                {
                    std::cout << "Affordance node inserted aff_id:" << aff_id.value() << std::endl;
                    //create has edge from door to affordance
                    DSR::Edge has = DSR::Edge::create<has_edge_type>(door.id(), aff_id.value());
                    qInfo() << "Edge insertion result:" << G->insert_or_assign_edge(has);
                }
            }
        }
    }

    //check all exisisitng affordance nodes for an active attribute that is true
    auto affordance_nodes = G->get_nodes_by_type("affordance");
    // Filter by affordance nodes which parent node are "door" type
    affordance_nodes.erase(std::remove_if(affordance_nodes.begin(), affordance_nodes.end(),
                                          [this](DSR::Node n) {
                                              auto parent_id = G->get_attrib_by_name<parent_att>(n.id());
                                              if (!parent_id.has_value())
                                                  return true; // remove node if no parent
                                              if (auto parent_node = G->get_node(parent_id.value()); parent_node.has_value())
                                                  return parent_node.value().name().find("door") == std::string::npos; // check parent node's name
                                              else
                                                  return true; // remove node if parent node is invalid
                                          }), affordance_nodes.end());
    for (const auto &aff : affordance_nodes)
    {
        if (auto active = G->get_attrib_by_name<active_att>(aff.id()); active.has_value())
        {
            if (active.value())
            {
                //check if there is a thread for this affordance node
                if (threads.contains(aff.id()))
                {
                    //read affordance node BT status attribute is_completed
                    if (auto bt_state = G->get_attrib_by_name<bt_state_att>(aff.id()); bt_state.has_value())
                    {
                        if (bt_state.value() == "completed")
                        {
                            std::cout << "Affordance node completed" << std::endl;
                            //join thread
                            threads[aff.id()].join();
                            threads.erase(aff.id());
                        }
                    }
                }
                else
                {
                    if (auto bt_state = G->get_attrib_by_name<bt_state_att>(aff.id()); bt_state.has_value())
                    {
                        if (bt_state.value() == "waiting")
                        {
                            //create a thread for this affordance node
                            threads[aff.id()] = std::thread(&SpecificWorker::affordance_thread, this, aff.id());
                        }
                    }
                }
            }
        }
    }
}

void SpecificWorker::match_exit_door()
{
    //get room with current edge
    auto current_edges = G->get_edges_by_type("current");
    if (current_edges.empty())
    {qWarning() << __FUNCTION__ << " No current edges in graph"; return;}

    auto room_node_ = G->get_node(current_edges[0].to());
    if (!room_node_.has_value())
    {qWarning() << __FUNCTION__ << " No room level in graph";return;}

    // Get room_id from room node
    auto room_id_att_ = G->get_attrib_by_name<room_id_att>(room_node_.value().id());
    if (!room_id_att_.has_value())
    {qWarning() << __FUNCTION__ << " No room id att in graph";return;}
    auto room_number_id = room_id_att_.value();

    //compare exit_door_pose with all nominal doors of the current room
    auto nominal_doors = G->get_nodes_by_type("door");

    //return if the number of doors with "_pre" in the name is not '0
    if (std::ranges::count_if(nominal_doors, [](DSR::Node n) { return n.name().find("_pre") != std::string::npos; }) != 0)
    {qWarning() << __FUNCTION__ << " There are doors with _pre in the name"; return;}

    std::pair<std::string, float> door_map = {"", std::numeric_limits<float>::max()};

    for (const auto &nominal_door : nominal_doors)
    {
        // Get room_id from nominal door node
        auto nominal_door_room_id = G->get_attrib_by_name<room_id_att>(nominal_door.id());
        if (!nominal_door_room_id.has_value())
        {qWarning() << __FUNCTION__ << " No room id att in graph";continue;}
        auto nominal_door_room_id_ = nominal_door_room_id.value();

        if (nominal_door.name().find("_pre") == std::string::npos and nominal_door_room_id_ == room_number_id)
        {
            //print nominal door name
            std::cout << "Nominal door name: " << nominal_door.name() << std::endl;

            //get rt translation of norminal door
            if (auto nominal_door_pose_ = inner_eigen->transform(room_node_.value().name(),
                                                                 nominal_door.name()); nominal_door_pose_.has_value())
            {
                auto nominal_door_robot_pose = nominal_door_pose_.value();
                std::cout << "Nominal door robot pose: " << nominal_door_robot_pose.x() << " " << nominal_door_robot_pose.y() << std::endl;
                //print exit_door_center
                std::cout << "Exit door center: " << exit_door_room_pose.x() << " " << exit_door_room_pose.y() << std::endl;

                if ((nominal_door_robot_pose - exit_door_room_pose).norm() < door_map.second)
                {
                    door_map = {nominal_door.name(), (nominal_door_robot_pose - exit_door_room_pose).norm()};
                    qInfo() << "Door name: " << QString::fromStdString(nominal_door.name()) << " distance: " << door_map.second;
                }
            }
        }
    }

    //Check if norminal_door_robot_pose is close to exit_door_pose
    if (door_map.second < 600)
    {
        //get exit edges in graph
        auto exit_edges = G->get_edges_by_type("exit");

        //get to from exit_edges[0]
        auto exited_door = G->get_node(exit_edges[0].to());
        //get exited_door node has value
        if (exited_door.has_value())
        {
            //get nominal_door node
            auto nominal_door = G->get_node(door_map.first);
            if (!nominal_door.has_value())
            {qWarning() << __FUNCTION__ << " No nominal door in graph";return;}

            //create a edge called match from nominal_door to exit_door
            DSR::Edge match = DSR::Edge::create<match_edge_type>(nominal_door.value().id(), exited_door.value().id());
            G->insert_or_assign_edge(match);
            exit_door_exists = false;

        }
    }
    else
    {
        std::cout << "Nominal door not matching exit door" << std::endl;
    }

}
void SpecificWorker::affordance_thread(uint64_t aff_id)
{
    BT::BehaviorTreeFactory factory;
    BT::Tree tree;

    //tick a behavior tree that goes through door
    factory.registerNodeType<Nodes::ExistsParent>("ExistsParent", this->G, aff_id);
    factory.registerNodeType<Nodes::CreateHasIntention>("CreateHasIntention", this->G, aff_id);
    factory.registerNodeType<Nodes::IsIntentionCompleted>("IsIntentionCompleted", this->G, aff_id);

    auto door_parent = G->get_attrib_by_name<parent_att>(aff_id);

    //Get parent attribute from node aff_id
    if (!door_parent.has_value())
    { qWarning () << __FUNCTION__ << " No parent attribute found"; return; }

    // Create BehaviorTree
    try
    {
        //Executing in /bin
        tree = factory.createTreeFromFile("./src/bt_affordance.xml"); // , blackboard
    } catch (const std::exception& e) { std::cerr << __FUNCTION__ << " Error creating BehaviorTree: " << e.what() << std::endl; }

    // Execute the behavior tree
    BT::NodeStatus status = BT::NodeStatus::RUNNING;
    while (status == BT::NodeStatus::RUNNING)
    {
//        std::cout << "Status: " << status << std::endl;
        status = tree.tickOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

//    std::cout << "Status: " << status << std::endl;

    //check is status is success and change bt_state from affordance to completed
    if (status == BT::NodeStatus::SUCCESS)
    {
        //Check if node with door_parent id exists in G
        if (auto door_parent_node = G->get_node(door_parent.value()); door_parent_node.has_value())
        {
            auto parent_node = door_parent_node.value();
//            if(auto other_side_door = G->get_attrib_by_name<other_side_door_name_att>(parent_node); not other_side_door.has_value())
//            {
                //create a edge called exist from robot node to door node
                DSR::Edge exit = DSR::Edge::create<exit_edge_type>(params.ROBOT_ID, door_parent_node.value().id());
                G->insert_or_assign_edge(exit);
                qInfo() << "EXIT EDGE INSERTED: GETTING EXIT DOOR POSE RESPECT TO ROBOT";
                auto exit_node_room_id = G->get_attrib_by_name<room_id_att>(door_parent_node.value().id());
                if(not exit_node_room_id.has_value())
                { qWarning() << __FUNCTION__ << " No room id att in graph"; return; }
                auto exit_node_room_id_ = exit_node_room_id.value();
                auto robot_node_ = G->get_node("Shadow");
                if(not robot_node_.has_value())
                { qWarning() << __FUNCTION__ << " No robot node in graph"; std::terminate(); }
                auto robot_node = robot_node_.value();
                //get robot parent node
                auto robot_parent = G->get_attrib_by_name<parent_att>(robot_node.id());
                if(not robot_parent.has_value())
                { qWarning() << __FUNCTION__ << " No robot parent node in graph"; return; }
                auto robot_parent_node_ = G->get_node(robot_parent.value());
                if(not robot_parent_node_.has_value())
                { qWarning() << __FUNCTION__ << " No robot parent node in graph"; return; }
                auto robot_parent_node = robot_parent_node_.value();
                // Get parent node name room_id
                auto room_id_ = G->get_attrib_by_name<room_id_att>(robot_parent_node.id());
                if(not room_id_.has_value())
                { qWarning() << __FUNCTION__ << " No room id att in graph"; return; }
                auto room_id = room_id_.value();

                if(exit_node_room_id_ == room_id)
                {
                    if (auto exit_door_robot_pose_ = inner_eigen->transform(robot_node.name(),
                                                                            door_parent_node.value().name()); exit_door_robot_pose_.has_value())
                    {
                        auto exit_door_robot_pose = exit_door_robot_pose_.value();
//                        std::cout << "Exit door robot pose: " <<  parent_node.name() <<exit_door_robot_pose.x() << " " << exit_door_robot_pose.y() << std::endl;
                        exit_door_center = exit_door_robot_pose;
                        exit_door_exists = true;
                    }
                }
//            }
        }

        if (auto aff = G->get_node(aff_id); aff.has_value())
        {
            G->add_or_modify_attrib_local<bt_state_att>(aff.value(), std::string("completed"));
            G->update_node(aff.value());
        }
    }
    else if (status == BT::NodeStatus::FAILURE)
    {
        if (auto aff = G->get_node(aff_id); aff.has_value())
        {
            G->add_or_modify_attrib_local<bt_state_att>(aff.value(), std::string("failed"));
            G->update_node(aff.value());
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::pair<std::vector<DoorDetector::Door>, std::vector<DoorDetector::Door>> SpecificWorker::get_measured_and_nominal_doors(DSR::Node room_node, DSR::Node robot_node, std::uint64_t lidar_timestamp)
{
    std::vector<DoorDetector::Door> nominal_doors, measured_doors;
    auto door_nodes = G->get_nodes_by_type("door");
    if(door_nodes.empty())
    { qWarning() << __FUNCTION__ << " No door nodes in graph"; return{}; }

    // remove doors which room_id attribute is different from actual_room_id
    door_nodes.erase(std::remove_if(door_nodes.begin(), door_nodes.end(), [this](DSR::Node n)
    {
        if(auto room_id = G->get_attrib_by_name<room_id_att>(n); room_id.has_value())
        {
            return room_id.value() != actual_room_id;
        }
        return true;
    }), door_nodes.end());

    /// Iterate over door nodes and check if they contain "measured" in their name
    for(auto &n: door_nodes)
    {
        auto room_id_att_ = G->get_attrib_by_name<room_id_att>(n.id());
        if(not room_id_att_.has_value())
        { qWarning() << __FUNCTION__ << " No room id att in graph"; return{}; }
        auto room_id = room_id_att_.value();

//        qInfo() << "Door name: " << QString::fromStdString(n.name());
        /// Get wall id knowing that is the 5 string element in the name
        auto wall_id = std::stoi(n.name().substr(7, 1));
        /// Get door id knowing that is the 7 string element in the name
        auto door_id = std::stoi(n.name().substr(9, 1));
        // Check if door still exists in graph

        if(not G->get_node(n.id()).has_value())
        { qWarning() << __FUNCTION__ << " Door node does not exist in graph"; continue; }
        
        if(auto door_width = G->get_attrib_by_name<width_att>(n); door_width.has_value())
        {
            auto p_0_ = Eigen::Vector3d {-(double)door_width.value() / 2, 0, 0};
            auto p_1_ = Eigen::Vector3d {(double)door_width.value() / 2, 0, 0};

            if (auto p_0_transformed = inner_eigen->transform(room_node.name(),
                                                              p_0_,
                                                              n.name(), lidar_timestamp); p_0_transformed.has_value())
            {
                auto p_0 = p_0_transformed.value();
                if (auto p_1_transformed = inner_eigen->transform(room_node.name(),
                                                                  p_1_,
                                                                  n.name(), lidar_timestamp); p_1_transformed.has_value())
                {
                    auto p_1 = p_1_transformed.value();
                    if (auto p_0_transformed_robot = inner_eigen->transform(robot_node.name(),
                                                                            p_0_,
                                                                            n.name(), lidar_timestamp); p_0_transformed_robot.has_value())
                    {
                        auto p_0_robot = p_0_transformed_robot.value();
                        if (auto p_1_transformed_robot = inner_eigen->transform(robot_node.name(),
                                                                                p_1_,
                                                                                n.name(), lidar_timestamp); p_1_transformed_robot.has_value())
                        {
                            auto p_1_robot = p_1_transformed_robot.value();
                            DoorDetector::Door door(Eigen::Vector2f{p_0.x(), p_0.y()},
                                                    Eigen::Vector2f{p_1.x(), p_1.y()},
                                                    Eigen::Vector2f{p_0_robot.x(), p_0_robot.y()},
                                                    Eigen::Vector2f{p_1_robot.x(), p_1_robot.y()});
                            door.id = door_id;
                            door.wall_id = wall_id;
//                            qInfo() << "Wall id: " << wall_id << " Door id: " << door_id;
//                            qInfo() << "Door name: " << QString::fromStdString(n.name()) << " Width: " << door.width() << " Center: " << door.middle[0] << door.middle[1];
                            if (n.name().find("_pre") != std::string::npos)
                                measured_doors.push_back(door);

                            else if(room_id == actual_room_id)
                                nominal_doors.push_back(door);
                        }
                    }
                }
            }
        }
    }
    return std::pair<std::vector<DoorDetector::Door>, std::vector<DoorDetector::Door>>{measured_doors, nominal_doors}; //Reference frame changed to wall
}

std::vector<DoorDetector::Door> SpecificWorker::get_doors(const std::vector<Eigen::Vector3f> &ldata, QGraphicsScene *scene, DSR::Node robot_node, DSR::Node room_node, std::uint64_t lidar_timestamp)
{
    /// Get wall centers and corners from graph
    /// Get corners and wall centers
//    qInfo() << "GETTING CORNERS AND WALL CENTERS";
    auto corners_and_wall_centers = get_corners_and_wall_centers(lidar_timestamp);
    if(not corners_and_wall_centers.has_value())
    { qWarning() << __FUNCTION__ << " No corners and wall centers detected"; return{}; }
    auto [corners, wall_centers] = corners_and_wall_centers.value();

    // Convert corners to Eigen vector double
    std::vector<Eigen::Vector2d> corners_eigen;
    corners_eigen.reserve(corners.size());
    for(const auto &c : corners)
        corners_eigen.emplace_back(c.x(), c.y());
    auto [outer_poly, inner_poly] = generate_inner_outer_polygons(corners_eigen, 150);

    // Transform outer and inner polygons to QPolygonF
    QPolygonF poly_room_out, poly_room_in, nominal_polygon;
    poly_room_out.reserve(outer_poly.size());
    poly_room_in.reserve(inner_poly.size());
    nominal_polygon.reserve(corners.size());
    for(const auto &p : outer_poly)
        poly_room_out << QPointF(p.x, p.y);
    for(const auto &p : inner_poly)
        poly_room_in << QPointF(p.x, p.y);
    for(const auto &c : corners)
    {
        qInfo() << c.x() << c.y();
        nominal_polygon << QPointF(c.x(), c.y());
    }




//    std::vector<Eigen::Vector2d> wall_points = filter_points_between_polygons(ldata, outer_poly, inner_poly);
//
//    draw_room_filtered_lidar(wall_points, &widget_2d->scene);

    draw_polygon(poly_room_in, poly_room_out, &widget_2d->scene, QColor("blue"));

    // Check if robot pose (0,0) is inside the room
    if(not is_point_in_polygon(Point64(0, 0), inner_poly))
        inside_polygon = false;
    else
        inside_polygon = true;

    // Filter lidar points inside room polygon
    std::vector<bool> inside_poly_out (ldata.size());
    std::vector<bool> outside_poly_in (ldata.size());
    std::vector<bool> in_wall (ldata.size());
    std::vector<int> in_wall_indexes;

//    qInfo() << "Lidar data size" << ldata.size();

    for(const auto &[i, p] : ldata | iter::enumerate)
    {
        // if point z is between 1000 and 2500
        if(p.z() < consts.ranges_list.first and p.z() > consts.ranges_list.second)
            continue;
        Point64 act_point(static_cast<int64_t>(p.x() * SCALE), static_cast<int64_t>(p.y() * SCALE));
        if(is_point_in_polygon(act_point, outer_poly))
            inside_poly_out[i] = true;
        else
            inside_poly_out[i] = false;
        if(not is_point_in_polygon(act_point, inner_poly))
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
    /// Push back the first element to the end of the vector
    if(not in_wall_indexes.empty())
        in_wall_indexes.push_back(in_wall_indexes.front());
    else
    {
        qWarning() << __FUNCTION__ << " No points inside the wall";
        return {};
    }
//    qInfo() << "in wall points size" << in_wall_indexes.size();
    std::vector<DoorDetector::Door> doors;

    vector<tuple<int, int>> door_number_for_wall;

    /// Iterate in_vall_indexes using sliding_window
    for(const auto &window : in_wall_indexes | iter::sliding_window(2))
        if(window.size() == 2)
        {
            auto p0 = ldata[window[0]];
            auto p1 = ldata[window[1]];

            auto line = Eigen::Vector2f(p1.x() - p0.x(), p1.y() - p0.y());
            auto line_norm = line.norm();

            /// Door width condition
            if(line_norm > 700 and line_norm < 1500)
            {
//                qInfo() << "Two points:" << p0.x() << p0.y() << p1.x() << p1.y() << "angle" << atan2(p0.y(), p0.x()) << atan2(p1.y(), p1.x()) << abs(atan2(p0.y(), p0.x()) - atan2(p1.y(), p1.x())) << "distance:" << line_norm;
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
                    auto p0_projected = project_point_on_polygon(QPointF(p0.x(), p0.y()), nominal_polygon);
                    auto p1_projected = project_point_on_polygon(QPointF(p1.x(), p1.y()), nominal_polygon);

                    Point64 p0_projected_64(static_cast<int64_t>(p0_projected.x() * SCALE), static_cast<int64_t>(p0_projected.y() * SCALE));
                    Point64 p1_projected_64(static_cast<int64_t>(p1_projected.x() * SCALE), static_cast<int64_t>(p1_projected.y() * SCALE));

                    //Check if p_0_ and p_1_ is outside poly in and insdide poly_out
                    if(is_point_in_polygon(p0_projected_64, inner_poly) and not is_point_in_polygon(p0_projected_64, outer_poly))
                        if(is_point_in_polygon(p1_projected_64, inner_poly) and not is_point_in_polygon(p1_projected_64, outer_poly))
                            continue;

                    /// Check between which nominal corners the projected points are
                    auto p0_projected_eigen = Eigen::Vector2f(p0_projected.x(), p0_projected.y());
                    auto p1_projected_eigen = Eigen::Vector2f(p1_projected.x(), p1_projected.y());
                    /// Obtanin the central point of the door and, considering the nominal corners, check between which corners the door is
                    auto middle = (p0_projected_eigen + p1_projected_eigen) / 2;

                    if(middle.norm() > min_door_distance)
                        continue;

                    // TESTING wall id a partir de proyección ortogonal
                    // Encuentra el segmento de pared más cercano a 'middle' según la proyección ortogonal
                    float min_dist = std::numeric_limits<float>::max();
                    int closest_wall_index = -1;
                    Eigen::Vector2f closest_projection;

                    for (size_t i = 0; i < corners.size(); ++i)
                    {
                        Eigen::Vector2f a = corners[i];
                        Eigen::Vector2f b = corners[(i + 1) % corners.size()];

                        Eigen::Vector2f ab = b - a;
                        float ab_len2 = ab.squaredNorm();
                        if (ab_len2 == 0) continue; // segmento degenerado

                        // Proyección de 'middle' sobre el segmento ab
                        float t = ((middle - a).dot(ab)) / ab_len2;
                        t = std::clamp(t, 0.0f, 1.0f);
                        Eigen::Vector2f projection = a + t * ab;

                        float dist = (middle - projection).norm();
                        if (dist < min_dist)
                        {
                            min_dist = dist;
                            closest_wall_index = i;
                            closest_projection = projection;
                        }
                    }


                    auto wall_center_index = closest_wall_index +1;


//                  auto middle_closest_point = middle_point_of_closest_segment(corners, middle);
//                    /// Get the index of the wall point closer to middle point using a lambda
//                    auto wall_center = *std::min_element(wall_centers.begin(), wall_centers.end(), [&middle_closest_point](const Eigen::Vector2f &a, const Eigen::Vector2f &b){ return (a - middle_closest_point).norm() < (b - middle_closest_point).norm(); });
                    /// Get the index of the wall center
//                    auto wall_center_index = std::distance(wall_centers.begin(), std::find(wall_centers.begin(), wall_centers.end(), wall_center));
//                    qInfo() << "Wall center index: " << wall_center_index << " Wall center: " << wall_center.x() << wall_center.y();

                    //TODO: transform to room reference frame.
                    Eigen::Vector3d p_0_ {p0_projected_eigen.x(), p0_projected_eigen.y(), 0};
                    Eigen::Vector3d p_1_ {p1_projected_eigen.x(), p1_projected_eigen.y(), 0};


                    /// Transform p_0 and p_1 to room reference frame
                    if (auto p_0_transformed = inner_eigen->transform(room_node.name(),
                                                                      p_0_,
                                                                      robot_node.name(), lidar_timestamp); p_0_transformed.has_value())
                    {
                        auto p_0 = p_0_transformed.value();
                        if (auto p_1_transformed = inner_eigen->transform(room_node.name(),
                                                                          p_1_,
                                                                          robot_node.name(), lidar_timestamp); p_1_transformed.has_value())
                        {
                            auto p_1 = p_1_transformed.value();
                            DoorDetector::Door door (Eigen::Vector2f{p_0.x(), p_0.y()}, Eigen::Vector2f{p_1.x(), p_1.y()}, Eigen::Vector2f{p_0_.x(), p_0_.y()}, Eigen::Vector2f{p_1_.x(), p_1_.y()});
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
                            qInfo() << "Measured door center: " << door.middle_measured[0] << door.middle_measured[1];

                            doors.push_back(door);
                        }
                    }
                }
            }
        }
    qInfo() << "Doors size" << doors.size();
    return doors;

}


// Convierte Eigen::Vector2d a Clipper2 Path64
Path64 SpecificWorker::toClipperPath64(const std::vector<Eigen::Vector2d>& poly)
{
    Path64 path;
    for (const auto& p : poly)
        path.emplace_back(static_cast<int64_t>(p.x() * SCALE), static_cast<int64_t>(p.y() * SCALE));
    return path;
}

// Comprueba si un punto está dentro de un Path64
bool SpecificWorker::is_point_in_polygon(const Point64& pt, const Path64& poly)
{
    return PointInPolygon(pt, poly) != PointInPolygonResult::IsOutside;
}

// Genera los polígonos exterior e interior a partir del polígono original y márgenes
std::pair<Path64, Path64> SpecificWorker::generate_inner_outer_polygons(const std::vector<Eigen::Vector2d>& polygon, double margin)
{
    Path64 original = toClipperPath64(polygon);
    ClipperOffset offsetter;

    offsetter.AddPath(original, JoinType::Miter, EndType::Polygon);

    Paths64 outer_solution, inner_solution;
    offsetter.Execute(margin * SCALE, outer_solution);      // Hacia afuera
    offsetter.Execute(-margin * SCALE, inner_solution);     // Hacia adentro

    if (outer_solution.empty() || inner_solution.empty())
        return {{}, {}};

    return {outer_solution[0], inner_solution[0]};
}

//std::vector<Eigen::Vector2d> SpecificWorker::filter_points_between_polygons(const std::vector<RoboCompLidar3D::TPoint>& lidar_points,
//                                                            const Path64& outer, const Path64& inner)
//{
//    std::vector<Eigen::Vector2d> points_between;
//    for (const auto& p : lidar_points)
//    {
//        Point64 pt(static_cast<int64_t>(p.x * SCALE), static_cast<int64_t>(p.y * SCALE));
//        if (is_point_in_polygon(pt, outer) && !is_point_in_polygon(pt, inner))
//            points_between.push_back(Eigen::Vector2d(p.x, p.y));
//    }
//    return points_between;
//}

Eigen::Vector2f SpecificWorker::middle_point_of_closest_segment(const std::vector<Eigen::Vector2f> &polygon, const Eigen::Vector2f &point)
{
    if (polygon.size() < 2) throw std::invalid_argument("Polygon must have at least 2 points");

    auto sq = [](float x) { return x * x; };
    auto dist2 = [&](const Eigen::Vector2f &a, const Eigen::Vector2f &b) { return sq(a.x() - b.x()) + sq(a.y() - b.y()); };

    float min_dist = std::numeric_limits<float>::max();
    int min_index = -1;

    for (size_t i = 0; i < polygon.size(); ++i) {
        Eigen::Vector2f a = polygon[i];
        Eigen::Vector2f b = polygon[(i + 1) % polygon.size()];
        float l2 = dist2(a, b);
        float dist;
        if (l2 == 0.0f) {
            dist = std::sqrt(dist2(point, a));
        } else {
            float t = std::max(0.0f, std::min(1.0f, ((point - a).dot(b - a)) / l2));
            Eigen::Vector2f projection = a + t * (b - a);
            dist = std::sqrt(dist2(point, projection));
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }

    Eigen::Vector2f segment_a = polygon[min_index];
    Eigen::Vector2f segment_b = polygon[(min_index + 1) % polygon.size()];
    return (segment_a + segment_b) / 2.0f;
}
std::optional<std::tuple<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>> SpecificWorker::get_corners_and_wall_centers(std::uint64_t lidar_timestamp)
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
    /// remove doors which name last number is different from actual room id
    corner_nodes.erase(std::remove_if(corner_nodes.begin(), corner_nodes.end(), [this](auto &n)
    {
        if(auto room_id = G->get_attrib_by_name<room_id_att>(n); room_id.has_value())
            return room_id.value() != actual_room_id;
        return true;
    }), corner_nodes.end());
    /// Sort corner nodes by name
    std::sort(corner_nodes.begin(), corner_nodes.end(), [](auto &n1, auto &n2){ return n1.name() < n2.name(); });
    std::vector<Eigen::Vector2f> corners, wall_centers;
    // Iterate over corners
    for(const auto &[i, n] : corner_nodes | iter::enumerate)
    {
//        qInfo() << "Corner name: " << QString::fromStdString(n.name());

//        if (auto corner_transformed = inner_eigen->transform(robot_node.name(),
//                                                             n.name(), lidar_timestamp); corner_transformed.has_value())
        if (auto corner_transformed = transform_point(n.name(),
        robot_node.name(),
        lidar_timestamp); corner_transformed.has_value())
        {
            auto [corner_timestamp, corner_transformed_value] = corner_transformed.value();
            auto corner_transformed_value_float = corner_transformed_value.cast<float>();
//            qInfo() << "Corner transformed value: " << corner_transformed_value_float.x() << corner_transformed_value_float.y();
            corners.push_back({corner_transformed_value_float.x(), corner_transformed_value_float.y()});
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
    int N = 5; // The number of consecutive frames a door must be detected in to be considered valid

    for(const auto &[i, d] : last_detected_doors | iter::enumerate)
    {
        if(get<1>(d) == N)
        {
            std::cout << "Door detected for N times: " << get<0>(d).wall_id << " " << get<0>(d).id << std::endl;
            last_detected_doors.erase(last_detected_doors.begin() + i);
        }
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
//            qInfo() << "First doors detected: " << d.wall_id << d.id;
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
            // Print the last detected door and the current detected door
            // qInfo() << "Last detected door: " << get<0>(last_detected_doors[i]).wall_id << get<0>(last_detected_doors[i]).id << "Current detected door: " << detected_door[j].wall_id << detected_door[j].id;
            // Print poses
            // qInfo() << "Last detected door pose: " << get<0>(last_detected_doors[i]).middle[0] << get<0>(last_detected_doors[i]).middle[1] << "Current detected door pose: " << detected_door[j].middle[0] << detected_door[j].middle[1];
            distances_matrix[i][j] = (get<0>(last_detected_doors[i]).middle - detected_door[j].middle).norm();
        }

    // Use the Hungarian algorithm to solve the assignment problem
    vector<int> assignment;
    // Print distances matrix and assignment
    // std::cout << "Distances matrix:" << std::endl;
    // for(const auto &row : distances_matrix)
    // {
    //     for(const auto &d : row)
    //         std::cout << d << " ";
    //     std::cout << std::endl;
    // }

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

            if(distances_matrix[i][assignment[i]] < detected_door[assignment[i]].width() * 0.25)
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

//    std::cout << "Measured doors: " << measured_doors.size() << std::endl;
//    std::cout << "Nominal doors: " << nominal_doors.size() << std::endl;

    vector<vector<double>> distances_matrix(measured_doors.size(), vector<double>(nominal_doors.size()));
    for(size_t i = 0; i < measured_doors.size(); i++)
        for(size_t j = 0; j < nominal_doors.size(); j++)
        {
            double angle_difference_sin = sin(measured_doors[i].angle - nominal_doors[j].angle);
            distances_matrix[i][j] = (measured_doors[i].middle - nominal_doors[j].middle).norm(); //TODO: incorporate rotation or door width to distance_matrix
//            qInfo() << "Distance: " << measured_doors[i].wall_id << measured_doors[i].id << " --- " << nominal_doors[j].wall_id << nominal_doors[j].id << distances_matrix[i][j];
        }

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
    return matching;
}
std::vector<DoorDetector::Door> SpecificWorker::update_and_remove_doors(std::vector<std::pair<int, int>> matches, const std::vector<DoorDetector::Door> &measured_doors, const std::vector<DoorDetector::Door> &graph_doors, bool nominal, DSR::Node room_node, std::uint64_t lidar_timestamp)
{
    /// vector for indexes to remove at the end of the function
    std::vector<int> indexes_to_remove;
    std::vector<int> indexes_to_set_no_valid;
    std::vector<DoorDetector::Door> not_associated_doors;
    /// Iterate over matches using enumerate
    for(const auto &[i, match] : matches | iter::enumerate)
    {
//        qInfo() << "Match: " << match.first << match.second;
        std::string door_name;
        if(match.second != -1)
        {
            if(nominal)
                door_name = "door_" + std::to_string(actual_room_id)  + "_" + std::to_string(graph_doors[match.second].wall_id) + "_" + std::to_string(graph_doors[match.second].id) ;
            else
                door_name = "door_" + std::to_string(actual_room_id)  + "_" + std::to_string(graph_doors[match.second].wall_id) + "_" + std::to_string(graph_doors[match.second].id) + "_pre";
            if(match.first != -1)
            {
            if(nominal)
            {
//                std::cout << "Deleting because is matched with a nominal" << door_name << std::endl;
//                measured_doors.erase(measured_doors.begin() + i);
                indexes_to_remove.push_back(match.first);
                continue;
            }
//            qInfo() << "Updating door data in graph";
                /// Get graph door name

//            qInfo() << "Door name first: " << QString::fromStdString(door_name) << "Mid point: " << measured_doors[match.first].middle.x() << measured_doors[match.first].middle.y() ;
//            qInfo() << "Door name second: " << QString::fromStdString(door_name) << "Mid point: " << measured_doors[match.second].middle.x() << measured_doors[match.second].middle.y() ;
                /// Update measured door with the matched door
                update_door_in_graph(measured_doors[match.first], door_name, room_node, lidar_timestamp);
                /// Insert in indexes_to_remove vector the index of the door to remove
                indexes_to_remove.push_back(match.first);
            }
        }
    }

    // Set not matched graph doors as not valid
    for(const auto &[i, door] : graph_doors | iter::enumerate)
    {
        if(std::find_if(matches.begin(), matches.end(), [i](const auto &m){ return m.second == i; }) == matches.end())
        {
            std::string door_name;
            if(nominal)
                door_name = "door_" + std::to_string(actual_room_id) + "_" + std::to_string(door.wall_id) + "_" + std::to_string(door.id)  ;
            else
                door_name = "door_" + std::to_string(actual_room_id)+ "_" + std::to_string(door.wall_id) + "_" + std::to_string(door.id) + "_pre";
            auto door_node_ = G->get_node(door_name);
            if(not door_node_.has_value())
            { qWarning() << __FUNCTION__ << " No door node in graph"; continue; }
            auto door_node = door_node_.value();
//            qInfo() << "Setting door" << QString::fromStdString(door_name) << "as not valid";
            G->add_or_modify_attrib_local<valid_att>(door_node, false);
            G->update_node(door_node);
        }
    }

    /// Insert in not_associated_doors vector the doors that are not associated using indexes_to_remove
    for(const auto &[i, door] : measured_doors | iter::enumerate)
        if(std::find(indexes_to_remove.begin(), indexes_to_remove.end(), i) == indexes_to_remove.end())
            not_associated_doors.push_back(door);
    return not_associated_doors;
}
void SpecificWorker::update_door_in_graph(const DoorDetector::Door &door, std::string door_name, DSR::Node room_node, std::uint64_t lidar_timestamp)
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
    auto pose_measured = door.middle_measured;
//    qInfo() << "Updating node" << door_node.id() << "Name:" << QString::fromStdString(door_node.name());
//    qInfo() << "Pose to update" << pose[0] << pose[1];

    /// Insert robot reference pose in node attribute
    std::vector<float> measured_pose = {pose_measured[0], pose_measured[1], 0.f};
    std::vector<float> measured_orientation = {0.f, 0.f, 0.f};
    G->add_or_modify_attrib_local<rt_translation_att>(door_node, measured_pose);
    G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att >(door_node, measured_orientation);
    G->add_or_modify_attrib_local<timestamp_alivetime_att>(door_node, lidar_timestamp);
    G->add_or_modify_attrib_local<valid_att>(door_node, true);

    if(auto door_node_ = G->get_node(door_name); door_node_.has_value())
        G->update_node(door_node);

    /// Transform pose to parent node reference frame
    if(auto pose_transformed_ = inner_eigen->transform(parent_node.name(), Eigen::Vector3d{pose[0], pose[1], 0.f}, room_node.name(), lidar_timestamp); pose_transformed_.has_value())
    {
        auto pose_transformed = pose_transformed_.value().cast<float>();
        qInfo() << "Pose to update transformed" << pose_transformed.x() << pose_transformed.y();
        // Add edge between door and robot
        //print door_node.name()
//        for(int i = 0; i<35; i++)
//        {
//            rt->insert_or_assign_edge_RT(parent_node, door_node.id(), {pose_transformed.x(), 0.f, 0.f}, {0.f, 0.f, 0.f}, lidar_timestamp);
//        }
        auto rt_edge = DSR::Edge::create<RT_edge_type>(parent_node.id(), door_node.id());
        auto rt_translation_ori = std::vector<float>{pose_transformed.x(), 0.f, 0.f};
        auto rt_rotation_ori = std::vector<float>{0.f, 0.f, 0.f};
        auto rt_timestamps_ori = std::vector<uint64_t >{lidar_timestamp};
        std::vector<float> rt_translation, rt_rotation;
        std::vector<uint64_t> rt_timestamps;
        rt_translation.reserve(3 * 20);
        rt_rotation.reserve(3 * 20);
        rt_timestamps.reserve(20);
        for(int i = 0; i < 20; i++)
        {
            rt_translation.insert(rt_translation.end(), rt_translation_ori.begin(), rt_translation_ori.end());
            rt_rotation.insert(rt_rotation.end(), rt_rotation_ori.begin(), rt_rotation_ori.end());
            rt_timestamps.insert(rt_timestamps.end(), rt_timestamps_ori.begin(), rt_timestamps_ori.end());
        }
        G->add_or_modify_attrib_local<rt_translation_att>(rt_edge, rt_translation);
        G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(rt_edge, rt_rotation);
        G->add_or_modify_attrib_local<rt_timestamps_att>(rt_edge, rt_timestamps);
        G->insert_or_assign_edge(rt_edge);

    }
}
//Function to insert a new measured door in the graph, generate a thread in charge of stabilizing the door and set a has_intention edge
void SpecificWorker::set_doors_to_stabilize(std::vector<DoorDetector::Door> doors, DSR::Node room_node, uint64_t lidar_timestamp)
{
//    qInfo() << "Door sizes set_doors_to_stabilize: " << doors.size();
    /// Iterate over doors using enumerate
    for(const auto &[i, door] : doors | iter::enumerate)
    {
        //Locate the wall where the door is
        auto wall_id = door.wall_id;
        //find the highest door id in the wall
        auto door_nodes = G->get_nodes_by_type("door");
        auto door_ids = std::vector<int>();
        for(auto &n: door_nodes)
            if(n.name().find("door") != std::string::npos)
            {
                auto wall_id_graph = std::stoi(n.name().substr(7, 1));
                auto door_id_graph = std::stoi(n.name().substr(9, 1));
                if(wall_id == wall_id_graph)
                    door_ids.push_back(door_id_graph);
            }
        int new_door_id;
        if(door_ids.empty())
            new_door_id = 0;
        else
            new_door_id = *std::max_element(door_ids.begin(), door_ids.end()) + 1;
        door.id = new_door_id;
        std::string door_name = "door_" + std::to_string(actual_room_id) + "_" + std::to_string(wall_id) + "_" + std::to_string(new_door_id)  +  "_pre";
        qInfo() << "Door name: " << QString::fromStdString(door_name) << "Mid point: " << door.middle.x() << door.middle.y() << "Width: " << door.width();
        //  10 create new_measured door to stabilize.
        insert_door_in_graph(door, room_node, door_name, lidar_timestamp);

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
        G->add_or_modify_attrib_local<active_att>(intention_edge, false);

        /// Set robot target to a position close to the room center
        std::vector<float> offset_target = {0, -1000, 0};
        G->add_or_modify_attrib_local<offset_xyz_att>(intention_edge, offset_target);

        /// Set tolerance to reach the target
        std::vector<float> tolerance = {200, 200, 0.f, 0.f, 0.f, 0.5};
        G->add_or_modify_attrib_local<tolerance_att>(intention_edge, tolerance);

        /// Set intention status to "waiting"
        std::string intention_status = "waiting";
        G->add_or_modify_attrib_local<state_att>(intention_edge, intention_status);

        G->insert_or_assign_edge(intention_edge);

//        insert measured_id_door
//    start a thread for each door to stabilize. This thread would have to:
        /// Generate a thread for each door to stabilize

        std::thread stabilize_thread(&SpecificWorker::stabilize_door, this, door, door_name);
        stabilize_thread.detach();

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
DSR::Node SpecificWorker::insert_door_in_graph(DoorDetector::Door door, DSR::Node room, const std::string& door_name, uint64_t lidar_timestamp)
{

    /// Get wall node from door id
    auto wall_node_ = G->get_node("wall_" + std::to_string(actual_room_id) + "_" + std::to_string(door.wall_id));
    if(not wall_node_.has_value())
    { qWarning() << __FUNCTION__ << " No wall node in graph"; return DSR::Node{}; }

    auto wall_node = wall_node_.value();
    auto door_node = DSR::Node::create<door_node_type>(door_name);

    // Transform door middle to wall reference frame
    auto door_middle = Eigen::Vector3d {door.middle.x(), door.middle.y(), 0.0};

    if( auto door_middle_transformed = inner_eigen->transform(wall_node.name(), door_middle, room.name(), lidar_timestamp); door_middle_transformed.has_value())
    {
        auto door_middle_transformed_value = door_middle_transformed.value().cast<float>();
        qInfo() << "########## Door world middle: " << door_middle.x() << door_middle.y();
        qInfo() << "########## Door world to wall transformed: " << door_middle_transformed_value.x() << door_middle_transformed_value.y();
        // Get wall node level
        auto wall_node_level = G->get_attrib_by_name<level_att>(wall_node).value();

        // get wall node pos_x and pos_y attributes
        auto wall_node_pos_x = G->get_attrib_by_name<pos_x_att>(wall_node).value();
        auto wall_node_pos_y = G->get_attrib_by_name<pos_y_att>(wall_node).value();

        // Add door attributes
        // set pos_x and pos_y attributes to door node
        G->add_or_modify_attrib_local<pos_x_att>(door_node, (float)(door.middle.x() * 0.1));
        G->add_or_modify_attrib_local<pos_y_att>(door_node, (float)(-door.middle.y() * 0.1));
        G->add_or_modify_attrib_local<width_att>(door_node, (int)door.width());
        G->add_or_modify_attrib_local<level_att>(door_node, wall_node_level + 1);
        G->add_or_modify_attrib_local<room_id_att>(door_node, actual_room_id);
        G->add_or_modify_attrib_local<timestamp_alivetime_att >(door_node, lidar_timestamp);

        // PROVISIONAL

        G->add_or_modify_attrib_local<parent_att>(door_node, wall_node.id());
        G->insert_node(door_node);

        // Add edge between door and robot
        std::cout << "INSERT OR ASSIGN EDGE" << door_node.name() << std::endl;
//        for(int i = 0; i<35; i++)
//        {
//            rt->insert_or_assign_edge_RT(wall_node, door_node.id(), {door_middle_transformed_value.x(),0.0, 0.0},
//                                         {0.0, 0.0, 0.0}, lidar_timestamp);
//        }
        auto rt_edge = DSR::Edge::create<RT_edge_type>(wall_node.id(), door_node.id());
        auto rt_translation_ori = std::vector<float>{door_middle_transformed_value.x(), 0.f, 0.f};
        auto rt_rotation_ori = std::vector<float>{0.f, 0.f, 0.f};
        auto rt_timestamps_ori = std::vector<uint64_t >{lidar_timestamp};
        std::vector<float> rt_translation, rt_rotation;
        std::vector<uint64_t> rt_timestamps;
        rt_translation.reserve(3 * 20);
        rt_rotation.reserve(3 * 20);
        rt_timestamps.reserve(20);
        for(int i = 0; i < 20; i++)
        {
            rt_translation.insert(rt_translation.end(), rt_translation_ori.begin(), rt_translation_ori.end());
            rt_rotation.insert(rt_rotation.end(), rt_rotation_ori.begin(), rt_rotation_ori.end());
            rt_timestamps.insert(rt_timestamps.end(), rt_timestamps_ori.begin(), rt_timestamps_ori.end());
        }
        G->add_or_modify_attrib_local<rt_translation_att>(rt_edge, rt_translation);
        G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(rt_edge, rt_rotation);
        G->add_or_modify_attrib_local<rt_timestamps_att>(rt_edge, rt_timestamps);
        G->insert_or_assign_edge(rt_edge);
    }

    //Find the node with "door_name" name in the graph
    auto door_node_ = G->get_node(door_name);
    if(not door_node_.has_value())
    { qWarning() << __FUNCTION__ << " No door node in graph"; return DSR::Node{}; }

    return door_node_.value();

}


void SpecificWorker::stabilize_door(DoorDetector::Door door, std::string door_name)
{
    bool is_stabilized = false;
    bool first_time = true;
    Eigen::Affine2d first_robot_pose;

    /// Data vectors for store corners, door and robot odometry data
    std::vector<Eigen::Vector2f> measured_door_points;

    /// Generate static map for histogram with coordinate in wall and door width
    std::map<int, int> width_histogram;
    std::map<int, int> pose_histogram;

    bool start_counter = false;
    int time_not_valid = 0;
    int time_collecting_data = 1500;

    uint64_t last_data_capture_timestamp = 0;
    auto start = std::chrono::high_resolution_clock::now();


    while(not is_stabilized)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto loop_start = std::chrono::high_resolution_clock::now();
//        qInfo() << "Stabilizing door: " << QString::fromStdString(door_name)<< "Time collecting data: " << time_collecting_data;
        /// get has intention edge between robot and doo
        auto robot_node_ = G->get_node("Shadow");
        if(not robot_node_.has_value())
        { qWarning() << __FUNCTION__ << " No robot node in graph"; return; }
        auto robot_node = robot_node_.value();

        auto door_node_ = G->get_node(door_name);
        if(not door_node_.has_value())
        { qWarning() << __FUNCTION__ << " No door node in graph"; return; }
        auto door_node = door_node_.value();

        auto door_node_timestamp = G->get_attrib_by_name<timestamp_alivetime_att>(door_node);
        if(not door_node_timestamp.has_value())
        { qWarning() << __FUNCTION__ << " No timestamp attribute in door node"; return; }
        auto door_node_timestamp_value = door_node_timestamp.value();

        if(door_node_timestamp_value == last_data_capture_timestamp)
        {
            if (auto door_data_valid = G->get_attrib_by_name<valid_att>(door_node); door_data_valid.has_value())
            {
//                qInfo() << "Door data valid: " << door_data_valid.value() << "Door name: "
//                        << QString::fromStdString(door_name);
                if (not door_data_valid.value()) {
                    measured_door_points.push_back(Eigen::Vector2f{0.f, 0.f});
                    time_not_valid += 1;
                    qInfo() << "Door data not valid. Waiting to stabilize door" << QString::fromStdString(door_name)
                            << "Time not valid: " << time_not_valid;
                    if (time_not_valid > 25) {
                        qInfo() << "Door data not valid for more than 25 iterations. Removing door"
                                << QString::fromStdString(door_name);
                        remove_to_stabilize_door_in_graph(door_node.id());
                        initialize_odom = false;
                        return;
                    }
                }
                sleep(0.1); // Sleep for 10ms to avoid busy waiting
                continue;
            }
        }
        if(time_not_valid > 0)
            time_not_valid -= 1;
//        qInfo() << "Door node timestamp: " << door_node_timestamp_value;
        last_data_capture_timestamp = door_node_timestamp_value;
        //TODO:: Check if robot is in has_intention target position
        auto intention_edge_ = G->get_edge(robot_node, door_node.id(), "has_intention");
        if(not intention_edge_.has_value())
        { qWarning() << __FUNCTION__ << " No has_intention edge in graph"; return; }
        auto intention_edge = intention_edge_.value();

        //get is active attribute from intention edge
        auto is_active = G->get_attrib_by_name<active_att>(intention_edge);
        if(not is_active.has_value())
        { qWarning() << __FUNCTION__ << " No is_active attribute in graph"; return; }
        if(is_active.value()) {start_counter = true;}

        auto intention_state = G->get_attrib_by_name<state_att>(intention_edge);
        if(not intention_state.has_value())
        { qWarning() << __FUNCTION__ << " No state attribute in graph"; return; }
        auto intention_state_value = intention_state.value();

        auto current_edges = G->get_edges_by_type("current");
        if(current_edges.empty())
        {qWarning() << __FUNCTION__ << " No current edges in graph";
//            remove_to_stabilize_door_in_graph(door_node.id());
            return;}

        auto room_node_ = G->get_node(current_edges[0].to());
        if(not room_node_.has_value())
        { qWarning() << __FUNCTION__ << " No room level in graph"; return; }
        auto room_node = room_node_.value();

        /// Get door wall node
        auto wall_node_ = G->get_parent_node(door_node);
        if(not wall_node_.has_value())
        { qWarning() << __FUNCTION__ << " No wall node in graph"; return; }
        auto wall_node = wall_node_.value();

        auto act_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(act_time - start).count();
//        qInfo() << "Elapsed time: " << elapsed << "ms";
        /// If "is active" attribute is true, the schedules assigns priority to stabilizing the door
        if(is_active.value() or elapsed < 1500)
        {
            /// check if robot is outside the polygon for killing the thread
            if(not inside_polygon)
            {
                qInfo() << "Robot outside polygon. Removing node and taking door into account for future exploration.";
                nodes_to_remove.push_back(door_node.id());
                initialize_odom = false;
                return;
            }
            /// If intention state is "in_progress", start getting data from the door and the room to stabilize the door
            if(intention_state_value == "in_progress" or elapsed < 1500)
            {
                /// Get door width and insert in histogram
                if (auto door_width = G->get_attrib_by_name<width_att>(door_node); door_width.has_value())
                    width_histogram[door_width.value()]++;

                /// Get door pose transformed to robot reference frame using inner_eigen transform
                if (auto wall_door_rt = rt->get_edge_RT(wall_node, door_node.id()); wall_door_rt.has_value())
                    if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(
                                wall_door_rt.value()); rt_translation.has_value())
                    {
                        auto rt_translation_value = rt_translation.value().get();
                        /// Insert distance to wall center in histogram considering distance is x value
                        pose_histogram[static_cast<int>(rt_translation_value[0])]++;
                    }
                /// Get door center coordinates in robot reference frame
                if (auto door_translation_ = G->get_attrib_by_name<rt_translation_att>(
                            door_node); door_translation_.has_value())
                {
                    auto door_translation = door_translation_.value().get();
                    measured_door_points.push_back(Eigen::Vector2f{door_translation[0], door_translation[1]});
                }

            }
            else if(intention_state_value == "waiting")
            {
//                    qInfo() << "Waiting to base controller...";
                continue;
            }
            else if(intention_state_value == "aborted" or intention_state_value == "failed")
            {
                qInfo() << "Problem found stabilizing door. Door is aborted or failed. Waiting scheduler to remove data";
                continue;
            }
            else if(intention_state_value == "completed")
            {

//                qInfo() << "Action Completed. Waiting scheduler to optimize door data.";
                continue;
            }
        }
        else // IS_ACTIVE == FALSE
        {
            if(intention_state_value == "aborted" or intention_state_value == "failed") //WAITING TO START is_active== false and state = waiting
            {
                qInfo() << "Problem found stabilizing door. Door is aborted or failed.";
                /// in case a problem is found stabilizing the door:
                /// remove intention edge ????
                /// Remove measured door in graph ????

                nodes_to_remove.push_back(door_node.id());

                initialize_odom = false;
                return;

            }
            else if(intention_state_value == "completed") //TODO: is_active == false $ state == completed, else if (state== fail, aborted)
            {
                /// Robot arrived to designed position
                /// Getting most common door width and pose
                qInfo() << "Door point reached";

                if(measured_door_points.empty())
                {
                    qInfo() << "No measured points found. removing door" << QString::fromStdString(door_name);

                    nodes_to_remove.push_back(door_node.id());

                    is_stabilized = true;
                    initialize_odom = false;
                    return;
                }
                //print width_histogram size
//                qInfo() << "Width histogram size: " << width_histogram.size();
                // Generate room size histogram considering room_size_histogram vector and obtain the most common room size
                auto most_common_door_width = std::max_element(width_histogram.begin(), width_histogram.end(),
                                                               [](const auto &p1, const auto &p2){ return p1.second < p2.second; });
                auto most_common_door_pose = std::max_element(pose_histogram.begin(), pose_histogram.end(),
                                                              [](const auto &p1, const auto &p2){ return p1.second < p2.second; });
                /// Get the most common room size
                auto door_pose_x = most_common_door_pose->first;
                auto door_width = most_common_door_width->first;
                qInfo() << "Most common door data: " << door_pose_x << " " << door_width << "#############################";
                Eigen::Vector3d door_nominal_respect_to_wall{(double)door_pose_x, 0.f, 0.f};
                /// Transform to room frame
                Eigen::Vector2f door_nominal_respect_to_room_value;
                if(auto door_nominal_respect_to_room = inner_eigen->transform(room_node.name(), door_nominal_respect_to_wall, wall_node.name()); door_nominal_respect_to_room.has_value())
                {
                    door_nominal_respect_to_room_value = {door_nominal_respect_to_room.value().x(), door_nominal_respect_to_room.value().y()};
                    qInfo() << "Most common door data transformed: " << door_nominal_respect_to_room_value.x() << " " << door_nominal_respect_to_room_value.y() << "#############################";
                    try
                    {
                        // Get RT edge from room to wall
                        auto wall_room_rt_edge_ = rt->get_edge_RT(room_node, wall_node.id());
                        // Check if exists
                        if(not wall_room_rt_edge_.has_value()){qInfo() << "No wall node, returning"; return;}
                        auto wall_room_rt_edge = wall_room_rt_edge_.value();
                        // Get rt orientation
                        auto wall_matrix = rt->get_edge_RT_as_rtmat(wall_room_rt_edge, std::numeric_limits<uint64_t >::max());
                        if(wall_matrix.has_value())
                        {
                            auto wall_matrix_value = wall_matrix.value();
                            // Get rotation angle from wall matrix
                            auto rotation_angle = std::atan2(wall_matrix_value(1, 0), wall_matrix_value(0, 0));
                            //Build new door with optimized data
                            auto optimized_door = DoorDetector::Door{Eigen::Vector2f{door_nominal_respect_to_room_value.x(), door_nominal_respect_to_room_value.y()},  door_width, door.wall_id, door.id, rotation_angle};

                            std::string door_name = "door_" + std::to_string(actual_room_id)+ "_" + std::to_string(optimized_door.wall_id) + "_" + std::to_string(optimized_door.id)  ;
                            //Create new door node
                            auto nominal_node = insert_door_in_graph(optimized_door, room_node, door_name, last_data_capture_timestamp);
                            std::vector<float> door_pose_vector = {door_nominal_respect_to_room_value.x(), door_nominal_respect_to_room_value.y(), 0.0};
                            std::vector<float> door_rot_vector = {0.0, 0.0, 0.0};
                            G->add_or_modify_attrib_local<rt_translation_att>(nominal_node, door_pose_vector);
                            G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(nominal_node, door_rot_vector);
                        }
                        else
                        {
                            qWarning() << __FUNCTION__ << " No wall matrix found";
                        }


                    }
                    catch (const std::exception &e)
                    {
                        qWarning() << __FUNCTION__ << " Error optimizing door data";

                    }
                }


                qInfo() << "Action Completed, delete intention_edge and close thread. ";
                nodes_to_remove.push_back(door_node.id());

                is_stabilized = true;
                initialize_odom = false;
                return;
            }
            else if(intention_state_value == "waiting")
            {
//                qInfo() << "Waiting activation.";
                continue;
            }
            else if(intention_state_value == "in_progress")
            {
                qInfo() << "This state shouldn't appear.";
                continue;
            }
        }

//        auto elapsed_total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
//        std::this_thread::sleep_for(std::chrono::milliseconds(50-elapsed_total.count()));
    }
    /// Kill thread
    qInfo() << "Thread finished";

}


void SpecificWorker::delete_pre_node(uint64_t node_id)
{
    /// Get node from id
    auto node_ = G->get_node(node_id);
    if(not node_.has_value())
    { qWarning() << __FUNCTION__ << " No node in graph"; return; }
    auto node = node_.value();
    //Delete edge between wall and door in all cases
    if (G->delete_edge(200, node.id(), "has_intention"))
        std::cout << __FUNCTION__ << " has_intention edge successfully deleted: " << std::endl;
    else
        std::cout << __FUNCTION__ << " Fatal error deleting has_intention robot-door: " << std::endl;

    auto parent_node_ = G->get_parent_node(node);
    if(not parent_node_.has_value())
    { qWarning() << __FUNCTION__ << " No parent node in graph"; return; }
    auto parent_node = parent_node_.value();

    if (G->delete_edge(parent_node.id(), node.id(), "RT"))
        std::cout << __FUNCTION__ << " RT from wall to door measured edge successfully deleted: " << std::endl;
    else
        std::cout << __FUNCTION__ << " Fatal error deleting rt edge wall-door: " << std::endl;

    //delete door node
    if(auto door_node__ = G->get_node(node.id()); door_node__.has_value())
        G->delete_node(node.id());
}


void SpecificWorker::remove_to_stabilize_door_in_graph(uint64_t door_id)
{
    //Delete edge between wall and door in all cases
    if (G->delete_edge(200, door_id, "has_intention"))
        std::cout << __FUNCTION__ << " has_intention edge successfully deleted: " << std::endl;
    else
        std::cout << __FUNCTION__ << " Fatal error deleting node: " << std::endl;

    // Get door node parent
    auto door_node_ = G->get_node(door_id);
    if(not door_node_.has_value())
    { qWarning() << __FUNCTION__ << " No door node in graph"; return; }
    auto door_node = door_node_.value();
    auto parent_node_ = G->get_parent_node(door_node);
    if(not parent_node_.has_value())
    { qWarning() << __FUNCTION__ << " No parent node in graph"; return; }
    auto parent_node = parent_node_.value();

    if (G->delete_edge(parent_node.id(), door_id, "RT"))
        std::cout << __FUNCTION__ << " RT from wall to door measured edge successfully deleted: " << std::endl;
    else
        std::cout << __FUNCTION__ << " Fatal error deleting node: " << std::endl;
    std::cout << "DELETE NODE " << door_node.name() << std::endl;
    //delete door node
    G->delete_node(door_id);
}
std::vector<float> SpecificWorker::get_graph_odometry()
{
    //TODO: Fix last_time initialization
    if (not initialize_odom)
    {
        last_time = std::chrono::high_resolution_clock::now();
        initialize_odom = true;
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
                //print odometry data
//                qInfo() << "Odometry data: " << adv_disp << side_disp << rot_disp;
                return {adv_disp, side_disp, rot_disp};
            }
    return {0, 0, 0};
}

////////////////////// LIDAR /////////////////////////////////////////////////
void SpecificWorker::read_lidar()
{
    while(true)
    {
        try
        {
//            auto data = lidar3d_proxy->getLidarData(consts.lidar_name, -90, 360, 3);
//            // Modify dada.points for removing points with z < consts.ranges_list through a lambda function
//            data.points.erase(std::remove_if(data.points.begin(), data.points.end(), [this](const RoboCompLidar3D::TPoint &p){ return p.z < consts.ranges_list.first or p.z > consts.ranges_list.second; }), data.points.end());
//
//            buffer_lidar_data.put<0>(std::move(data), static_cast<size_t>(data.timestamp));
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D " << e << std::endl; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

///////////////////// Draw  /////////////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &data, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &p: data)
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20), QBrush(QColor("green")));
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
}

void SpecificWorker::draw_room_filtered_lidar(const std::vector<Eigen::Vector2d> &data, QGraphicsScene *scene, QColor color)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();

    // draw points
    for(const auto &p: data)
    {
        auto item = scene->addEllipse(-20, -20, 40, 40, QPen(QColor("red"), 20), QBrush(QColor("red")));
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
}

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
        auto line = scene->addLine(poly_in[i].x(), poly_in[i].y(), poly_in[(i+1)%poly_in.size()].x(), poly_in[(i+1)%poly_in.size()].y(), QPen(QColor("pink"), 20));
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
void SpecificWorker::draw_door(const std::vector<DoorDetector::Door> &nominal_doors, const std::vector<DoorDetector::Door> &measured_doors, QColor nominal_color, QColor measured_color, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();
    //Print function name
    std::cout << __FUNCTION__ << std::endl;

    //draw
    draw_vector_of_doors(scene, items, nominal_doors, nominal_color);
    draw_vector_of_doors(scene, items, measured_doors, measured_color);

}
void SpecificWorker::draw_vector_of_doors(QGraphicsScene *scene, vector<QGraphicsItem *> &items,
                                          std::vector<DoorDetector::Door> doors, QColor color) const
{
    Eigen::Vector2i offset(40, 40);
    //Draw ellipse in the doors corners
    for(const auto &door: doors)
    {
        //Calculate p0, p1 using door.middle, door.width and door.angle
        Eigen::Vector2f p0 = door.p0_measured;
        Eigen::Vector2f p1 = door.p1_measured;

        // draw points inside doors
        for(const auto &door: doors)
        {
            auto item = scene->addLine(p0.x(), p0.y(), p1.x(), p1.y(), QPen(color, 40));
            items.push_back(item);
        }

        auto item = scene->addEllipse(-40, -40, 80, 80, QPen(color, 40), QBrush(color));
        item->setPos(p0.x(), p0.y());
        items.push_back(item);

        auto item2 = scene->addEllipse(-40, -40, 80, 80, QPen(color, 40), QBrush(color));
        item2->setPos(p1.x(), p1.y());
        items.push_back(item2);

        // Draw Coordinate x,y of the corner
        auto item_text = scene->addText(QString::number(p0.x(), 'f', 2) + QString::fromStdString(",") + QString::number(p0.y(), 'f', 2), QFont("Arial", 100));
        item_text->setPos(p0.x() + offset.x(), p0.y() + offset.y());
        item_text->setTransform(QTransform().scale(1, -1));
        items.push_back(item_text);

        // Draw Coordinate x,y of the corner
        auto item_text2 = scene->addText(QString::number(p1.x(), 'f', 2) + QString::fromStdString(",") + QString::number(p1.y(), 'f', 2), QFont("Arial", 100));
        item_text2->setPos(p1.x() + offset.x(), p1.y() + offset.y());
        item_text2->setTransform(QTransform().scale(1, -1));
        items.push_back(item_text2);
    }
    //Draw door id in the middle of the door
    for(const auto &door: doors)
    {
        auto item = scene->addText(QString::number(door.wall_id)+ QString::fromStdString("_") + QString::number(door.id), QFont("Arial", 300));
        item->setPos(door.middle_measured.x() + offset.x(), door.middle_measured.y() + offset.y());
        item->setTransform(QTransform().scale(1, -1));
        items.push_back(item);
    }
}
void SpecificWorker::draw_door_robot_frame(const std::vector<DoorDetector::Door> &nominal_doors, const std::vector<DoorDetector::Door> &measured_doors, QColor nominal_color, QColor measured_color, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> items;
    for(const auto &i: items){ scene->removeItem(i); delete i;}
    items.clear();
    //Print function name
//    std::cout << __FUNCTION__ << std::endl;


    //draw
//    qInfo() << "Sizes nominal_doors: " << nominal_doors.size() << " measured_doors: " << measured_doors.size();
    draw_vector_of_doors(scene, items, nominal_doors, nominal_color);
    draw_vector_of_doors(scene, items, measured_doors, measured_color);
}

//Ceate funcion to build .g2o string from corner_data, odometry_data, RT matrix
std::string SpecificWorker::build_g2o_graph(const std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> &measured_corner_points,
const std::vector<Eigen::Matrix<float, 2, 1>> &nominal_corner_data,
const std::vector<std::vector<float>> &odometry_data,
const Eigen::Affine2d &robot_pose,
const std::vector<Eigen::Vector2f> &measured_door_center,
const Eigen::Vector2f &nominal_door_center)
{
    std::string g2o_graph;
    int id = 0; // Id for g2o graph vertices
    auto updated_robot_pose = robot_pose;
    
    //set std::to_string decimal separator dot
    std::setlocale(LC_NUMERIC, "C");

    /// Add nominal corners as VERTEX_XY (ROOM), FIXED using nominal_corners_data in
    for (size_t i = 0; i < nominal_corner_data.size(); ++i)
    {
        g2o_graph += "VERTEX_XY " + std::to_string(id) + " " + std::to_string(nominal_corner_data[i].x()) + " " + std::to_string(nominal_corner_data[i].y()) + "\n";
        g2o_graph += "FIX " + std::to_string(id) + "\n";
        id++;
    }
    /// ADD NOMINAL DOOR VERTEX CALCULATED FROM WIDTH AND DISTANCE (ROOM COORDINATES), NOT FIXED using nominal_door_vertices
    g2o_graph += "VERTEX_XY " + std::to_string(id) + " " + std::to_string(nominal_door_center.x()) + " " + std::to_string(nominal_door_center.y()) + "\n";
    auto door_vertex_id = id;
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
    /// INSERT LANDMARKS TO ROOM CORNERS, using measured_corner_points
    for (size_t i = 0; i < measured_corner_points.size()-1; ++i)
    {
        for (size_t j = 0; j < measured_corner_points[i].size(); ++j)
        {
            /// Check if measured_corner_points[i][j] is valid
            if(measured_corner_points[i][j].x() != 0.f and measured_corner_points[i][j].y() != 0.f)
                g2o_graph += "EDGE_SE2_XY " + std::to_string(id) + " " + std::to_string(j) + " " + std::to_string(measured_corner_points[i][j].x()) + " " + std::to_string(measured_corner_points[i][j].y()) + " 10 0 10 \n";
        }
        id++;
    }
    //Return to initial ID
    id = node_id;
    // INSERT LANDMARKS TO DOOR CENTER
    for (size_t i = 0; i < measured_door_center.size()-1; ++i)
    {
        /// Check if measured_door_center[i] is valid
        if(measured_door_center[i].x() != 0.f and measured_door_center[i].y() != 0.f)
            g2o_graph += "EDGE_SE2_XY " + std::to_string(id) + " " + std::to_string(door_vertex_id)  + " " + std::to_string(measured_door_center[i].x()) + " " + std::to_string(measured_door_center[i].y()) + " 1 0 1 \n";
        id++;
    }
    return g2o_graph;
}
Eigen::Vector2f SpecificWorker::extract_g2o_data(string optimization)
{
    std::istringstream iss(optimization);
    // Find VERTEX_XY with ids 1, 2, 3, 4
    Eigen::Vector2f door_center;
    std::string line;
    while (std::getline(iss, line))
    {
        std::istringstream iss_line(line);
        std::string type;
        int id;
        float x, y;
        // Get type
        iss_line >> type;
        // Get id
        iss_line >> id;
        // Get x
        iss_line >> x;
        // Get y
        iss_line >> y;
        if(type == "VERTEX_XY")
        {
            if(id == g2o_nominal_door_id)
            {
                qInfo() << "VERTEX_XY DOOR ID FOUND: " << id;
                door_center = Eigen::Vector2f{x, y};
                qInfo() << "NOMINAL DOOR CENTER: x =" << door_center.x() << " y= " << door_center.y();
                return door_center;
            }
        }
    }
    return{};
}
void SpecificWorker::clear_doors()
{
    /// Delete affordance nodes if parent is a door
    auto affordance_nodes = G->get_nodes_by_type("affordance");
    for(const auto &affordance: affordance_nodes)
    {
        auto parent_node = G->get_parent_node(affordance);
        if(not parent_node.has_value())
        { qWarning() << __FUNCTION__ << " No parent node in graph"; continue; }
        if(parent_node.value().type() == "door")
        {
            if(not G->delete_node(affordance.id()))
            { qWarning() << __FUNCTION__ << " Error deleting affordance node"; continue; }
            if(not G->delete_edge(parent_node.value().id(), affordance.id(), "has"))
            { qWarning() << __FUNCTION__ << " Error deleting edge"; continue; }
        }

    }
    //Delete all doors
    auto door_nodes = G->get_nodes_by_type("door");
    for(const auto &door: door_nodes)
    {
        /// Get parent node
        auto edges_to_door = G->get_edges_to_id(door.id());
        if(edges_to_door.empty())
        { qWarning() << __FUNCTION__ << " No edges to door node in graph"; return; }
        for(const auto &edge: edges_to_door)
            if(not G->delete_edge(edge.from(), edge.to(), edge.type()))
            { qWarning() << __FUNCTION__ << " Error deleting edge"; return; }
        if(not G->delete_node(door.id()))
        { qWarning() << __FUNCTION__ << " Error deleting door node"; return; }
    }
}

std::optional<std::pair<std::uint64_t, Eigen::Vector3d>> SpecificWorker::transform_point(const std::string& from_node_name,
                                                                                         const std::string& to_node_name,
                                                                                         std::uint64_t timestamp)
{
    if (auto corner_transformed = inner_eigen->transform(to_node_name,
                                                         from_node_name, timestamp); corner_transformed.has_value())
    {
        auto corner_transformed_ = corner_transformed.value();
        return std::make_pair(static_cast<std::uint64_t>(timestamp), Eigen::Vector3d(corner_transformed_.x(), corner_transformed_.y(), 0.0));
    }
    else
    {
        std::cout << "There is no corner transformed" << std::endl;
        return {};
    }
//    auto from_node = G->get_node(from_node_name);
//    if (!from_node) {qInfo() << "From node not found"; return{};};
//    auto from_node_value = from_node.value();
//    if(auto parent_node = G->get_parent_node(from_node_value); parent_node.has_value())
//    {
//        auto parent_node_value = parent_node.value();
//        if(auto rt_parent_node = rt->get_edge_RT(parent_node_value, from_node_value.id()); rt_parent_node.has_value())
//        {
//            auto rt_parent_node_ = rt_parent_node.value();
//            auto parent_node_pose = rt->get_edge_RT_as_stamped_rtmat(rt_parent_node_, timestamp);
//            if (!parent_node_pose.has_value()) {std::cout << "There is no parent node pose" << std::endl; return{};}
//            auto parent_node_pose_ = parent_node_pose.value();
//            auto parent_node_pose_timestamp = parent_node_pose_.first;
//            auto parent_node_pose_matrix = parent_node_pose_.second.matrix();
//            Eigen::Vector3f pose(parent_node_pose_matrix(0, 3), parent_node_pose_matrix(1, 3), 0.f);
//            auto pose_double = pose.cast<double>();
//
//            if (auto transformed_pose = inner_eigen->transform(to_node_name,
//                                                             pose_double,
//                                                             parent_node_value.name()); transformed_pose.has_value())
//            {
//                return std::make_pair(static_cast<std::uint64_t>(parent_node_pose_timestamp), transformed_pose.value());
//            }
//        }
//        else
//        {
//            std::cout << "No RT edge found " << std::endl;
//            return{};
//        }
//    }
//    else
//    {
//        std::cout << "No parent node found " << std::endl;
//        return{};
//    }
}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

//create update_edge_slot
void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    //check if type == current
    if(type == "current")
    {
        //get from node
        auto from_node = G->get_node(from);
        if(not from_node.has_value())
        { qWarning() << __FUNCTION__ << " No from node in graph"; return; }
        //get to node

        auto room_node = G->get_node(to);
        if(not room_node.has_value())
        { qWarning() << __FUNCTION__ << " No room node in graph"; return; }
        std::cout << "Transforming robot to room: " << room_node.value().name() << std::endl;
        if(exit_door_exists)
        {
            if (auto exit_door_room_pose_ = inner_eigen->transform(room_node.value().name(),
                                                                   exit_door_center, params.robot_name); exit_door_room_pose_.has_value())
            {
                exit_door_room_pose = exit_door_room_pose_.value();
                std::cout << "SLOT! Exit door room pose: " << exit_door_room_pose.x() << " " << exit_door_room_pose.y() << std::endl;
            }
        }
        wait_to_current = false;
    }

    // If exit door exists, transform it to room frame

}

//SUBSCRIPTION to pushLidarData method from Lidar3DPub interface
void SpecificWorker::Lidar3DPub_pushLidarData(RoboCompLidar3D::TDataCategory lidarData)
{
    // Reservar espacio para no hacer reallocs
    std::vector<Eigen::Vector3f> filtered_lidar;
    filtered_lidar.reserve(lidarData.XArray.size());

    // Crear set para búsqueda O(1)
    const std::unordered_set<int> category_set(categories_filter.begin(), categories_filter.end());

    // Lambda para decidir si un punto se incluye
    auto is_point_valid = [&](size_t i) -> bool {
        return category_set.find(lidarData.CategoriesArray[i]) != category_set.end() &&
               lidarData.ZArray[i] > consts.ranges_list.first &&
               lidarData.ZArray[i] < consts.ranges_list.second;
    };

    // Filtrar
    for (size_t i = 0; i < lidarData.CategoriesArray.size(); ++i)
    {
        if (is_point_valid(i))
        {
            filtered_lidar.emplace_back(lidarData.XArray[i], lidarData.YArray[i], lidarData.ZArray[i]);
        }
    }

    buffer_lidar_data.put(std::make_pair(lidarData.timestamp, filtered_lidar));
}

