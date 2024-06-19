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
		timer.start(Period);
	}

}

void SpecificWorker::compute()
{

    qInfo() << "************************************Compute*************************************************";
    auto has_intention_edges = G->get_edges_by_type("has_intention");

    //print this->intention_active
    qInfo() << "Intention active: " << this->intention_active;

    //get_edges of type exit
    auto exit_edges = G->get_edges_by_type("exit");

    //This code is for the case when affordances must not be activated because someone is doing important things
    if (not exit_edges.empty() and not affordance_activated)
    {
        affordance_activated = true;
    }

    if(!this->intention_active)
    {
        if (!has_intention_edges.empty())
        {
            qInfo()<< "Has intention edges found";
            for (auto edge : has_intention_edges)
            {
                if (auto edge_state = G->get_attrib_by_name<state_att>(edge); edge_state.has_value())
                {
                    if (edge_state.value() == "waiting")
                    {
                        qInfo() << "Intention edge found and waiting";
                        this->active_node_id = set_intention_active(edge, true);
                        this->intention_active = true;

                        qInfo() << "Intention_active = true";
                        break;
                    }
                }
            }
        }
        else
        {
            if(!affordance_activated)
            {
                //get nodes of type affordance
                auto affordance_nodes = G->get_nodes_by_type("affordance");
                // change the active attribute of an affordance node to true
                for(auto affordance : affordance_nodes)
                {
                    //get affordance bt_state attribute
                    if(auto affordance_state = G->get_attrib_by_name<bt_state_att>(affordance); affordance_state.has_value())
                    {
                        if (affordance_state.value() == "waiting")
                        {
                            qInfo() << "Affordance node found and waiting";
                            G->add_or_modify_attrib_local<active_att>(affordance, true);
                            //set affordance bt_state attribute to in_progress
                            G->add_or_modify_attrib_local<bt_state_att>(affordance, std::string("in_progress"));
                            affordance_activated_id = affordance.id();
                            G->update_node(affordance);
                            affordance_activated = true;
                            return;
                        }
                    }
                }
            }
            else
            {
                std::cout << "Affordance activated" << std::endl;

                //Check if affordance node is completed
                if(auto affordance_node = G->get_node(affordance_activated_id); affordance_node.has_value())
                {
                    if(auto affordance_state = G->get_attrib_by_name<bt_state_att>(affordance_node.value()); affordance_state.has_value())
                    {
                        std::cout << "Affordance bt_state:" << affordance_state.value() << std::endl;

                        if(affordance_state.value() == "completed")
                        {
                            std::cout << "Affordance node completed" << std::endl;
                            G->add_or_modify_attrib_local<active_att>(affordance_node.value(), false);
                            G->update_node(affordance_node.value());
                            affordance_activated = false;
                            affordance_activated_id = -1;
                            this->intention_active = false;
//                            std::terminate();
                        }
                    }
                }
            }
        }
    }
    else //Intention ACTIVE
    {
        qInfo() << "Intention active";
        if (auto active_node_ = G->get_node(this->active_node_id); active_node_.has_value())
        {
            std::cout << "Node name" << active_node_.value().name() << std::endl;

            if(auto edge = G->get_edge(params.ROBOT_ID, this->active_node_id, "has_intention"); edge.has_value())
            {
                if(auto edge_state = G->get_attrib_by_name<state_att>(edge.value()); edge_state.has_value())
                {
                    std::cout << "Edge state: " << edge_state.value() << std::endl;

                    if(edge_state.value() == "completed")
                    {
                        qInfo() << "HAS INTENTION EDGE COMPLETED";
                        this->active_node_id = set_intention_active(edge.value(), false);
                    }
                }
            }
        }
    }
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


