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
    this->Period = 500;

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
        connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
        //connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

        // Graph viewer
        using opts = DSR::DSRViewer::view;
        qInfo() << "asdg";
        int current_opts = 0;
        qInfo() << "asdg";
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
        qInfo() << "asdg";
        graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
        qInfo() << "asdg";
        setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

        /***
        Custom Widget
        In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
        The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
        either with a QtDesigner or directly from scratch in a class of its own.
        The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
        ***/
        //graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);
        hide();
        timer.start(Period);
    }

}

void SpecificWorker::compute()
{

    if(wait)
    {
        if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - wait_start_time).count() > waiting_time)
        {
            wait = false;
        }
        return;
    }
    // Print affordance_map
    for(auto [key, value] : affordance_map)
    {
        std::cout << "Affordance: " << key << " Value: " << value << std::endl;
    }
    // Check if any affordance to increase exists
    if(not affordances_to_increase.empty())
    {
        std::cout << "Affordances to increase not empty" << std::endl;
        for(auto aff : affordances_to_increase)
        {
            std::cout << aff << std::endl;
        }
        // Get door nodes
        auto door_nodes = G->get_nodes_by_type("door");
        // Check if the door node with the other side door name attribute is found
        for(const auto &door : door_nodes)
        {
            if(auto other_side_door = G->get_attrib_by_name<other_side_door_name_att>(door); other_side_door.has_value())
            {
                if(std::find(affordances_to_increase.begin(), affordances_to_increase.end(), other_side_door.value().get()) != affordances_to_increase.end())
                {
                    // Check if the affordance map has the other side door name
                    if(affordance_map.find(other_side_door.value().get()) == affordance_map.end())
                    {
                        affordance_map[door.name()] = 1;
                    }
                    else
                    {
                        affordance_map[door.name()]++;
                    }
                    affordances_to_increase.erase(std::remove(affordances_to_increase.begin(), affordances_to_increase.end(), other_side_door.value().get()), affordances_to_increase.end());
                }

            }
        }
    }

    //FIND IF EXIST ONE AFFORDANCE ACTIVATED, TO UPDATE MISSION STATUS
    //Get all nodes of type affordance
    auto affordance_nodes = G->get_nodes_by_type("affordance");
    // If not empty, find the affordance node with the active attribute set to true
    for(auto affordance : affordance_nodes)
    {
        // Get parent node
        auto parent_node_ = G->get_parent_node(affordance);
        if(not parent_node_.has_value())
        { qWarning() << __FUNCTION__ << " No parent node in graph"; return; }
        auto parent_node = parent_node_.value();
        if(affordance_map.find(parent_node.name()) == affordance_map.end())
            affordance_map[parent_node.name()] = 0;
        if(auto affordance_active = G->get_attrib_by_name<active_att>(affordance); affordance_active.has_value())
        {
            if(affordance_active.value())
            {
                //Check if affordance node is completed
                if(auto affordance_state = G->get_attrib_by_name<bt_state_att>(affordance); affordance_state.has_value())
                {
                    std::cout << CYAN << "Affordance name: " << affordance.name() << RESET << std::endl;
                    std::cout << CYAN << "Affordance state: " << affordance_state.value() << RESET << std::endl;
                    if(affordance_state.value() == "completed")
                    {
                        G->add_or_modify_attrib_local<active_att>(affordance, false);
                        G->update_node(affordance);
                        //Print AFFORDANCE NODE COMPLETED
                        std::cout << CYAN << "AFFORDANCE NODE COMPLETED, setting active to false" << RESET << std::endl;
                        wait_start_time = std::chrono::system_clock::now();
                        wait = true;
                        // Check if affordance parent node is a door
                        /// Get parent node
                        auto parent_node_ = G->get_parent_node(affordance);
                        if(not parent_node_.has_value())
                        { qWarning() << __FUNCTION__ << " No parent node in graph"; return; }
                        auto parent_node = parent_node_.value();

                        // Check if parent node is a door
                        if(auto parent_type = parent_node.type(); parent_type == "door")
                        {
                            if(auto other_side_door = G->get_attrib_by_name<other_side_door_name_att>(parent_node); other_side_door.has_value())
                            {
                                auto other_side_door_name = other_side_door.value();
                                if(other_side_door_name.get() == parent_node.name())
                                {
                                    if(affordance_map.find(other_side_door_name.get()) == affordance_map.end())
                                    {
                                        std::cout << "Affordance not found in map" << std::endl;
                                        affordance_map[other_side_door_name.get()] = 1;

                                    }
                                    else
                                    {
                                        affordance_map[other_side_door_name.get()]++;
                                    }
                                    break;
                                }
                            }
                            else
                            {
                                qWarning() << __FUNCTION__ << " No other side door name attribute found. Waiting for the adjacent door";
                                affordances_to_increase.push_back(parent_node.name());
                            }
                        }

                        //Print the affordance name, state and value
                        return;
                    }
                    else if(affordance_state.value () == "in_progress" or affordance_state.value () == "waiting")
                    {
                        std::cout << BLUE << "AFFORDANCE NODE IN PROGRESS: " << affordance.name() << RESET << std::endl;
                        return;
                    }
                }
            }
        }
    }

    //FIND IF EXIST ONE HAS_INTENTION ACTIVATED, TO UPDATE MISSION STATUS
    //Get all edges of type has_intention
    auto has_intention_edges = G->get_edges_by_type("has_intention");
    // If not empty, find the edge with the active attribute set to true
    for(auto edge : has_intention_edges)
        // Get intention edge active
        if(auto edge_active = G->get_attrib_by_name<active_att>(edge); edge_active.has_value())
            // Get intention edge state
            if(auto edge_state = G->get_attrib_by_name<state_att>(edge); edge_state.has_value())
            {
                if(not edge_active.value() and edge_state.value() == "completed")
                {
                    qInfo() << "Waiting intention edge removal";
                    return;
                }
                else if(edge_active.value())
                {
                    //Print the edge from and to names
                    qInfo() << "Edge from: " << QString::fromStdString(G->get_node(edge.from()).value().name()) << "Edge to: " << QString::fromStdString(G->get_node(edge.to()).value().name());
                    //Print the edge state and value
                    qInfo() << "Edge state: " << QString::fromStdString(edge_state.value());

                    if(edge_state.value() == "completed")
                    {
                        G->add_or_modify_attrib_local<active_att>(edge, false);
                        G->insert_or_assign_edge(edge);
                        std::cout << RED << "HAS INTENTION EDGE COMPLETED, setting active to false" << RESET << std::endl;
                        return;
                    }
                    else if (edge_state.value() == "in_progress" or edge_state.value() == "waiting")
                    {
                        std::cout << GREEN << "HAS INTENTION EDGE IN PROGRESS" << RESET << std::endl;
                        return;
                    }
                }
            }

    //NO CURRENT MISSION ACTIVATED, TIME TO SELECT ONE

    std::cout << YELLOW << "************************************** SEPE ***************************************" << RESET << std::endl;

    //IN ORDER TO ACCOMPLISH ROOM TRANSITION, FIRST ACTIVATE HAS INTENTION EDGES TO STABILIZE ROOM ELEMENTS
    //Get first has_intention edge with state waiting
    for(auto edge : has_intention_edges)
    {
        if(auto edge_state = G->get_attrib_by_name<state_att>(edge); edge_state.has_value())
        {
            if(edge_state.value() == "waiting")
            {
                //cout << "NEW INTENTION EDGE FOUND"  in color GREEN using printf
                //Print NEW INTENTION ACTIVE in GREEN color
                std::cout << GREEN << "----------------- NEW INTENTION ACTIVE ----------------------------" << RESET << std::endl;
                //Print the edge from and to names
                qInfo() << "Edge from: " << QString::fromStdString(G->get_node(edge.from()).value().name()) << "Edge to: " << QString::fromStdString(G->get_node(edge.to()).value().name());
                //Print the edge state and value
                qInfo() << "Edge state: " << QString::fromStdString(edge_state.value());
                //Set the intention att active of the edge to true
                G->add_or_modify_attrib_local<active_att>(edge, true);
                G->insert_or_assign_edge(edge);

                return;
            }
        }
    }

    auto exit_edges = G->get_edges_by_type("exit");

    //This code is for the case when affordances must not be activated because someone is doing important things
    if (not exit_edges.empty())
    {
        std::cout << "EXIT EDGE FOUND" << std::endl;
        return;
    }

    std::string best_aff = "";
    //get current edge with types
    auto current_edges = G->get_edges_by_type("current");

    if(current_edges.empty())
    { std::cout << "No current edges found" << std::endl; return; }

    //get to node from current edges [0]
    auto room_node = G->get_node(current_edges[0].to());
    //check to_node value
    if(!room_node.has_value())
    { std::cout << "No to node found" << std::endl; return; }

    //get room id from room_node.value() with get attrib by name room_id_att
    auto room_id = G->get_attrib_by_name<room_id_att>(room_node.value());
    //check room_id value
    if(!room_id.has_value())
    { std::cout << "No room id found" << std::endl; return; }

    //Get first affordance node with state waiting
    for(auto affordance : affordance_nodes)
    {
        std::cout << "Affordance name: " << affordance.name() << std::endl;
        //get affordance room_id attribute
        if(auto affordance_room_id = G->get_attrib_by_name<room_id_att>(affordance); affordance_room_id.has_value())
        {
            std::cout << "Affordance affordance_room_id: " << affordance_room_id.value() << std::endl;
            if(auto affordance_state = G->get_attrib_by_name<bt_state_att>(affordance); affordance_state.has_value() and affordance_room_id.value() == room_id.value())
            {
                // Get parent node
                auto parent_node_ = G->get_parent_node(affordance);
                if(not parent_node_.has_value())
                { qWarning() << __FUNCTION__ << " No parent node in graph"; return; }
                auto parent_node = parent_node_.value();
                std::cout << "Affordance state: " << affordance_state.value() << std::endl;
                //check if affordance state is waiting
//            if(affordance_state.value() == "waiting" and executed_affordances.end() == std::find(executed_affordances.begin(), executed_affordances.end(), affordance.name()))
                if(affordance_state.value() == "waiting")
                {
                    //check if affordance name is in the map
                    if(affordance_map.find(parent_node.name()) == affordance_map.end())
                    {
                        std::cout << "Affordance not found in map" << std::endl;
                        affordance_map[parent_node.name()] = 0;
                        best_aff = parent_node.name();
                        break;
                    }
                    else
                    {
                        //check if best affordance int in map is less than the current affordance
                        if(best_aff.empty() or affordance_map[parent_node.name()] < affordance_map[best_aff])
                        {
                            best_aff = parent_node.name();
                        }
                    }
                }
            }
        }
    }
    std::cout << "Best affordance: " << best_aff << std::endl;

    //get node from best_aff
    if (auto door_affordance_ = G->get_node(best_aff); door_affordance_.has_value())
    {
        auto door_affordance = door_affordance_.value();
        // Iterate over affordance nodes and get the affordance node which parent node is the best_aff
        for(auto affordance : affordance_nodes)
        {
            if (auto parent = G->get_parent_node(affordance); parent.has_value())
            {
                if (parent.value().name() == best_aff)
                {
                    //Print NEW AFFORDANCE ACTIVE in GREEN color using printf
                    //Print NEW AFFORDANCE ACTIVE in GREEN color
                    std::cout << BLUE << "----------------- NEW AFFORDANCE ACTIVE ----------------------------" << RESET << std::endl;
                    //Print the affordance name and state
                    qInfo() << "Affordance name: " << QString::fromStdString(affordance.name());
                    // add to affordancce

                    //Set the affordance active to true
                    G->add_or_modify_attrib_local<active_att>(affordance, true);
                    //Set the affordance state to in_progress
                    G->update_node(affordance);
                    affordance_map[best_aff]++;
                    break;
                }
            }
        }

    }
    else
    {
        std::cout << "No affordance found with besft aff string" << std::endl;
    }

//    auto has_intention_edges = G->get_edges_by_type("has_intention");
//
//    //print this->intention_active
//    qInfo() << "Intention active: " << this->intention_active;
//
//    //get_edges of type exit
//    auto exit_edges = G->get_edges_by_type("exit");
//
//    //This code is for the case when affordances must not be activated because someone is doing important things
//    if (not exit_edges.empty() and not affordance_activated)
//    {
//        affordance_activated = true;
//    }
//
//    if(!this->intention_active)
//    {
//        if (!has_intention_edges.empty())
//        {
//            qInfo()<< "Has intention edges found";
//            for (auto edge : has_intention_edges)
//            {
//                if (auto edge_state = G->get_attrib_by_name<state_att>(edge); edge_state.has_value())
//                {
//                    if (edge_state.value() == "waiting")
//                    {
//                        qInfo() << "Intention edge found and waiting";
//                        this->active_node_id = set_intention_active(edge, true);
//                        this->intention_active = true;
//
//                        qInfo() << "Intention_active = true";
//                        break;
//                    }
//                }
//            }
//        }
//        else
//        {
//            if(!affordance_activated)
//            {
//                //get nodes of type affordance
//                auto affordance_nodes = G->get_nodes_by_type("affordance");
//                // change the active attribute of an affordance node to true
//                for(auto affordance : affordance_nodes)
//                {
//                    //get affordance bt_state attribute
//                    if(auto affordance_state = G->get_attrib_by_name<bt_state_att>(affordance); affordance_state.has_value())
//                    {
//                        if (affordance_state.value() == "waiting")
//                        {
//                            qInfo() << "Affordance node found and waiting";
//                            G->add_or_modify_attrib_local<active_att>(affordance, true);
//                            //set affordance bt_state attribute to in_progress
//                            G->add_or_modify_attrib_local<bt_state_att>(affordance, std::string("in_progress"));
//                            affordance_activated_id = affordance.id();
//                            G->update_node(affordance);
//                            affordance_activated = true;
//                            return;
//                        }
//                    }
//                }
//            }
//            else
//            {
//                std::cout << "Affordance activated" << std::endl;
//
//                //Check if affordance node is completed
//                if(auto affordance_node = G->get_node(affordance_activated_id); affordance_node.has_value())
//                {
//                    if(auto affordance_state = G->get_attrib_by_name<bt_state_att>(affordance_node.value()); affordance_state.has_value())
//                    {
//                        std::cout << "Affordance bt_state:" << affordance_state.value() << std::endl;
//
//                        if(affordance_state.value() == "completed")
//                        {
//                            std::cout << "Affordance node completed" << std::endl;
//                            G->add_or_modify_attrib_local<active_att>(affordance_node.value(), false);
//                            G->update_node(affordance_node.value());
//                            affordance_activated = false;
//                            affordance_activated_id = -1;
//                            this->intention_active = false;
////                            std::terminate();
//                        }
//                    }
//                }
//            }
//        }
//    }
//    else //Intention ACTIVE
//    {
//        qInfo() << "Intention active";
//        if (auto active_node_ = G->get_node(this->active_node_id); active_node_.has_value())
//        {
//            std::cout << "Node name" << active_node_.value().name() << std::endl;
//
//            if(auto edge = G->get_edge(params.ROBOT_ID, this->active_node_id, "has_intention"); edge.has_value())
//            {
//                if(auto edge_state = G->get_attrib_by_name<state_att>(edge.value()); edge_state.has_value())
//                {
//                    std::cout << "Edge state: " << edge_state.value() << std::endl;
//
//                    if(edge_state.value() == "completed")
//                    {
//                        qInfo() << "HAS INTENTION EDGE COMPLETED";
//                        this->active_node_id = set_intention_active(edge.value(), false);
//                    }
//                }
//            }
//        }
//    }
}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

u_int64_t SpecificWorker::set_intention_active(DSR::Edge &edge, bool active)
{
    qInfo() << "Setting intention active: " << active;
    G->add_or_modify_attrib_local<active_att>(edge, active);
    G->insert_or_assign_edge(edge);
    this->intention_active = active;

    if (!active)
        return -1;
    else
        return edge.to();
}

void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{
    if(edge_tag == "exit")
    {
        qInfo() << "Exit edge deleted";
        affordance_activated = false;
        affordance_activated_id = -1;
    }
}


