/*
 *    Copyright (C) 2022 by YOUR NAME HERE
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
	G->write_to_json_file("./"+agent_name+".json");
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





	agent_name = params["agent_name"].value;
	agent_id = stoi(params["agent_id"].value);

	tree_view = params["tree_view"].value == "true";
	graph_view = params["graph_view"].value == "true";
	qscene_2d_view = params["2d_view"].value == "true";
	osg_3d_view = params["3d_view"].value == "true";

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
		timer.start(Period);
		// create graph
		G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
		std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;  

		//dsr update signals
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		// connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		// connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
		// connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

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
		hide();
		this->Period = period;
		timer.start(Period);
	}

}

void SpecificWorker::compute()
{
	//computeCODE
	//QMutexLocker locker(mutex);
	//try
	//{
	//  camera_proxy->getYImage(0,img, cState, bState);
	//  memcpy(image_gray.data, &img[0], m_width*m_height*sizeof(uchar));
	//  searchTags(image_gray);
	//}
	//catch(const Ice::Exception &e)
	//{
	//  std::cout << "Error reading from Camera" << e << std::endl;
	//}
	
	
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


void SpecificWorker::SoundRotation_gotKeyWord(int positionAngle)
{
    DSR::Node new_virtual_node = DSR::Node::create<virtual_person_node_type>("listened_person");
    G->add_or_modify_attrib_local<position_deg_att>(new_virtual_node, positionAngle);
    G->insert_node(new_virtual_node);
    G->update_node(new_virtual_node);
    DSR::Edge lost_edge = DSR::Edge::create<looking_for_edge_type>(G->get_node("robot").value().id(), new_virtual_node.id());
    if (G->insert_or_assign_edge(lost_edge))
    {
        std::cout << __FUNCTION__ << " Edge successfully inserted: " << G->get_node("robot").value().id() << "->" << new_virtual_node.id()
                  << " type: lost" << std::endl;
    }
    else
    {
        std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << G->get_node("robot").value().id() << "->" << new_virtual_node.id()
                  << " type: lost" << std::endl;
    }
    DSR::Edge has_edge = DSR::Edge::create<has_edge_type>(G->get_node(robot_mind_name).value().id(), new_virtual_node.id());
    if (G->insert_or_assign_edge(has_edge))
    {
        std::cout << __FUNCTION__ << " Edge successfully inserted: " << G->get_node(robot_mind_name).value().id() << "->" << new_virtual_node.id()
                  << " type: has" << std::endl;
    }
    else
    {
        std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << G->get_node(robot_mind_name).value().id() << "->" << new_virtual_node.id()
                  << " type: has" << std::endl;
    }

}
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
	if(type == current_path_name)
		if(auto looking_for_edges = G->get_edges_by_type(looking_for_type_name); looking_for_edges.size() > 0)
			path_id = id;
}

void SpecificWorker::del_node_slot(std::uint64_t from)
{
	if(path_id = from)
	{
		if(auto listened_person = G->get_node("listened_person"); listened_person.has_value())
        {
			G->delete_node(listened_person.value().id());
			G->delete_edge(G->get_node(robot_name).value().id(), listened_person.value().id(), looking_for_type_name);
			G->delete_edge(G->get_node(robot_mind_name).value().id(), listened_person.value().id(), "has");
		}
	}
}


