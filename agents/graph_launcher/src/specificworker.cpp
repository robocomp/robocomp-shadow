/*
 *    Copyright (C) 2025 by YOUR NAME HERE
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
#include <QStringListModel> // <-- Add this
#include <QDebug>           // <-- Add this

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

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	//G->write_to_json_file("./"+agent_name+".json");
}


void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
	//dsr update signals
	//connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	//connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
	//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
	//connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
	//connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

	/***
	Custom Widget
	In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
	The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
	either with a QtDesigner or directly from scratch in a class of its own.
	The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
	***/
    graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
	graph_viewer_LTSM = std::make_unique<DSR::DSRViewer>(this, G_LTSM, current_opts_LTSM, main_LTSM);


	//graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);
    save_widget = new CustomWidget();
    graph_viewer->add_custom_widget_to_dock("graph save", save_widget);

    connect(save_widget->ui->pushButton, &QPushButton::clicked, this, [this](){ this->save_json(); });

	// Create a model for the QListView
    QStringListModel *graphListModel = new QStringListModel(this);
    QStringList list;
    list << "G (Short-Term Memory)" << "G_LTSM (Long-Term Memory)";
    graphListModel->setStringList(list);
    
    // Set the model to the graph_list UI element
    save_widget->ui->graph_list->setModel(graphListModel);
    
    // Set a default selection (e.g., the first item)
    save_widget->ui->graph_list->setCurrentIndex(graphListModel->index(0));

    //initializeCODE
	clean_dsr();
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
	//hide();

}



void SpecificWorker::compute()
{
	//computeCODE
	//try
	//{
	//  camera_proxy->getYImage(0,img, cState, bState);
    //    if (img.empty())
    //        emit goToEmergency()
	//  memcpy(image_gray.data, &img[0], m_width*m_height*sizeof(uchar));
	//  searchTags(image_gray);
	//}
	//catch(const Ice::Exception &e)
	//{
	//  std::cout << "Error reading from Camera" << e << std::endl;
	//}
}

void SpecificWorker::clean_dsr()
{
	auto nodes = G_LTSM->get_nodes();
	std::vector<std::tuple<uint64_t, uint64_t, std::string>> edges_to_delete;
	for (const auto &node : nodes)
	{
		auto node_edges_opt = G_LTSM->get_edges(node.id());
		// Check if the optional contains a value
        if (node_edges_opt.has_value())
        {
            // Iterate through the map of edges for this node
            // The map item is a pair: <Key, Value>
            // Key: std::pair<uint64_t, std::string> (dest_id, type)
            // Value: Edge (the edge object itself)
            for (const auto& edge_pair : node_edges_opt.value())
            {
                const auto& edge = edge_pair.second; // Get the Edge object
                uint64_t dest_id = edge.to();
                
                // Check if the destination node exists in the graph
                std::optional<DSR::Node> dest_node = G_LTSM->get_node(dest_id);
                
                if (!dest_node.has_value())
                {
                    // Destination node does not exist. Mark this edge for deletion.
                    uint64_t src_id = edge.from(); // This should be node.id()
                    std::string edge_type = edge.type();

                    qWarning() << "Dangling edge found: " 
                               << src_id << "->" << dest_id 
                               << " (Type: " << QString::fromStdString(edge_type) 
                               << "). Destination node missing. Marking for deletion.";
                    G_LTSM->delete_edge(src_id, dest_id, edge_type);
					if (auto removed_edge = G_LTSM->get_edge(src_id, dest_id, edge_type); removed_edge.has_value());
					{
						std::cout << "Not removed edge. Destroying" << std::endl;
					}
                }
            }
        }
	}
}

void SpecificWorker::save_json()
{
	clean_dsr();

    // 1. Get the filename from the text edit
    QString base_name = save_widget->ui->json_name->toPlainText().trimmed();
    if (base_name.isEmpty()) 
    {
        qWarning() << "Save JSON: No filename provided.";
        return;
    }
    std::string filename = base_name.toStdString() + ".json";

    // 2. Get the currently selected item from the QListView
    QModelIndex currentIndex = save_widget->ui->graph_list->currentIndex();
    if (!currentIndex.isValid())
    {
        qWarning() << "Save JSON: No graph selected from the list.";
        return;
    }
    
    QString selectedGraph = currentIndex.data(Qt::DisplayRole).toString();

    // 3. Save the correct graph based on the selection
    if (selectedGraph.startsWith("G (Short-Term Memory)"))
    {
        qInfo() << "Saving G to" << QString::fromStdString(filename);
        G->write_to_json_file(filename);
    }
    else if (selectedGraph.startsWith("G_LTSM (Long-Term Memory)"))
    {
        qInfo() << "Saving G_LTSM to" << QString::fromStdString(filename);
        G_LTSM->write_to_json_file(filename);
    }
    else
    {
        qWarning() << "Save JSON: Unknown selection.";
        return;
    }

    // 4. Clear the text field
    save_widget->ui->json_name->clear();
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


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}



