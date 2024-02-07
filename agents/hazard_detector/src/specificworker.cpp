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
//	QLoggingCategory::setFilterRules("*.debug=false\n");
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

    // check if there is a new intention edge or a change in existing  intention edges
    // buffer_for_changes_in_intention_edges.try_get();

    // if there is an intention edge marked as dangerous
    // check if there is an action that can be taken to avoid the danger

    // Get the list of possible robot actions from the list of visual elements, excluding the robot and  the person
    // for each action and target object associated
    //      get the terminal state of the action: the target position where the robot will be after the action is executed
    //      For each object in VE excluding the robot and the person,
    //          guess a human intention linking her to the i_object.
    //          use the cones-algorithm to obtain the objects inside the cone going from the person to the object
    //          use the  gridder to compute paths from the person to the i_object, with the constraints:
    //              if there are objects in VE outside the cone,
    //                  compute one path removing all of them  from the grid
    //                  compute one additional path leaving on object at a time in the grid
    //              if the object in the cone is the virtual robot, add it as an occupied region to the grid
    //         with the new paths, check if they are safe, free of collisions.
    //         if all the paths are safe for the i_object intentions, and the ones in the  graph were not,
    //              select this action as a candidate  action to remove the danger
    //
    

	
	
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}




