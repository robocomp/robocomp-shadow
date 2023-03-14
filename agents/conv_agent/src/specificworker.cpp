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
		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
//		connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
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

		this->Period = period;
		timer.start(Period);
        hide();

//        auto person_nodes = G->get_nodes_by_type(person_type_name);
//        for(auto person : person_nodes)
//        {
//            if(auto person_name = G->get_attrib_by_name<person_name_att>(person); person_name.has_value())
//            {
//                interest_person_name = person_name.value();
//                interest_person_node_id = person.id();
//                this->conversation_proxy->lost(person_name.value(), "person");
//            }
//        }
	}
}

void SpecificWorker::compute()
{
    qInfo() << first_follow;
    // check for existing missions
    if (auto plan_o = plan_buffer.try_get(); plan_o.has_value())
    {
        current_plan = plan_o.value();
        current_plan.set_running();
    }

}

void SpecificWorker:: start_mission(int intention)
{
    if(not temporary_plan.is_valid())
    {
        stop_mission();
        switch (intention)
        {
            case 0:
                create_ask_for_follow_plan();
                break;
            case 1:
                create_ask_for_stop_following_plan();
                break;
            case 2:
                create_talking_plan();
                break;
            case 3:
                create_ask_for_stop_talking_plan();
                break;
        }
    }

    if(temporary_plan.is_complete())
    {
        qInfo() << "############################";
        qInfo() << "STARTING ASK FOR FOLLOW PLAN";
        qInfo() << "############################";
        insert_intention_node(temporary_plan);
        auto temp_plan = temporary_plan;
        plan_buffer.put(std::move(temp_plan));
    }
    else
        qWarning() << __FUNCTION__ << "Plan is not complete. Mission cannot be created";
}

void SpecificWorker::stop_mission()
{
    if(auto intention = G->get_node(person_current_intention_name); intention.has_value())
    {
        if(auto intention_parent_id = G->get_attrib_by_name<parent_att>(intention.value()); intention_parent_id.has_value())
        {
//            if(auto person_mind = G->get_node(intention_parent_id.value()); person_mind.has_value())
//            {
//                if (auto has_mind_intention = G->get_edge(person_mind.value().id(),intention.value().id(),"has");has_mind_intention.has_value())
//                    G->delete_edge(person_mind.value().id(),intention.value().id(), "has");

                G->delete_node(intention.value().id());
//            }
        }
    }
    else
        qWarning() << __FUNCTION__ << "No intention node found";

    temporary_plan.reset();
    current_plan.reset();
}

void SpecificWorker::create_ask_for_follow_plan()
{
    temporary_plan.new_plan(Plan::Actions::ASK_FOR_FOLLOWING);
    if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
        temporary_plan.new_plan(Plan::Actions::ASK_FOR_FOLLOWING);
    std::ostringstream oss;
    oss << interest_person_node_id;
    std::string intAsString(oss.str());

    temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
//    temporary_plan.insert_attribute("person_name",  QString::fromStdString(intAsString));
    std::cout << "PLAN: " << temporary_plan.pprint() << std::endl;
    return;
}

void SpecificWorker::create_talking_plan()
{
    this->conversation_proxy->talking(interest_person_name, "person", "hablar");
    temporary_plan.new_plan(Plan::Actions::TALKING_WITH_ROBOT);
    if (not temporary_plan.is_valid())
        temporary_plan.new_plan(Plan::Actions::TALKING_WITH_ROBOT);
    std::ostringstream oss;
    oss << interest_person_node_id;
    std::string intAsString(oss.str());

    temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
//    temporary_plan.insert_attribute("person_name",  QString::fromStdString(intAsString));
    std::cout << "PLAN: " << temporary_plan.pprint() << std::endl;
    return;
}

void SpecificWorker::create_ask_for_stop_following_plan()
{
    temporary_plan.new_plan(Plan::Actions::ASK_FOR_STOP_FOLLOWING);
    if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
        temporary_plan.new_plan(Plan::Actions::ASK_FOR_STOP_FOLLOWING);
    std::ostringstream oss;
    oss << interest_person_node_id;
    std::string intAsString(oss.str());

    temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
    std::cout << "PLAN: " << temporary_plan.pprint() << std::endl;
    // To say "te sigo" in the next following interaction
    return;
}

void SpecificWorker::create_ask_for_stop_talking_plan()
{
    temporary_plan.new_plan(Plan::Actions::ASK_FOR_STOP_TALKING);
    if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
        temporary_plan.new_plan(Plan::Actions::ASK_FOR_STOP_TALKING);
    std::ostringstream oss;
    oss << interest_person_node_id;
    std::string intAsString(oss.str());

    temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
    std::cout << "PLAN: " << temporary_plan.pprint() << std::endl;
    // To say "te sigo" in the next following interaction
    return;
}

void SpecificWorker::insert_intention_node(const Plan &plan)
{
    if(auto interested_person_node = G->get_node(interest_person_node_id); interested_person_node.has_value())
    {
    qInfo() << "PERSON NODE:" << interested_person_node.value().id();
        auto mind_nodes = G->get_nodes_by_type("transform");
        for(auto mind_node : mind_nodes)
        {
            if(auto act_mind_node_id = G->get_attrib_by_name<parent_att>(mind_node); act_mind_node_id.has_value())
            {
                if(act_mind_node_id.value() == interest_person_node_id)
                {
                    qInfo() << "MIND NODE FOUND";
                    auto intention_nodes = G->get_nodes_by_type("intention");
                    // TODO: REVISAR SI SE PUEDE MEJORAR ESTO
                    if(intention_nodes.size() > 0)
                    {
                        for(auto intention_node : intention_nodes)
                        {
                            if(auto intention_parent_id = G->get_attrib_by_name<parent_att>(intention_node); intention_parent_id.has_value())
                            {
                                if(intention_parent_id.value() == mind_node.id())
                                {
                                    std::cout << __FUNCTION__ << ": Updating existing intention node with Id: " << intention_node.id() << std::endl;
                                    G->add_or_modify_attrib_local<current_intention_att>(intention_node, plan.to_json());
                                    G->update_node(intention_node);
                                    std::cout << "INSERT: " << plan.to_json() << std::endl;
                                    return;
                                }
                            }
                        }
                    }
                    DSR::Node intention_node = DSR::Node::create<intention_node_type>(person_current_intention_name);
                    G->add_or_modify_attrib_local<parent_att>(intention_node, mind_node.id());
                    if (auto mind_level = G->get_node_level(mind_node); mind_level.has_value())
                    {
                        G->add_or_modify_attrib_local<level_att>(intention_node, mind_level.value() + 1);
                    }
                    G->add_or_modify_attrib_local<pos_x_att>(intention_node, (float) -290);
                    G->add_or_modify_attrib_local<pos_y_att>(intention_node, (float) -500);
                    G->add_or_modify_attrib_local<current_intention_att>(intention_node, plan.to_json());
                    std::cout << "DATOS DEL PALN: " << plan.to_json() << std::endl;
                    if (std::optional<int> intention_node_id = G->insert_node(intention_node); intention_node_id.has_value())
                    {
                        DSR::Edge edge = DSR::Edge::create<has_edge_type>(mind_node.id(), intention_node.id());
                        if (G->insert_or_assign_edge(edge))
                        {
                            std::cout << __FUNCTION__ << " Edge successfully inserted: " << mind_node.id() << "->" << intention_node.id()
                                      << " type: has" << std::endl;
                            G->add_or_modify_attrib_local<current_intention_att>(intention_node, plan.to_json());
                            G->update_node(intention_node);
                        }
                        else
                        {
                            std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << mind_node.id() << "->" << intention_node_id.value()
                                      << " type: has" << std::endl;
                            std::terminate();

                        }
                    } else
                    {
                        std::cout << __FUNCTION__ << ": Fatal error inserting_new 'intention' node" << std::endl;
                        std::terminate();
                    }
                    break;
                }
            }
        }
    }
}
void SpecificWorker::create_waiting_person_node()
{
    DSR::Node new_virtual_node = DSR::Node::create<virtual_person_node_type>("person_to_wait");
    G->add_or_modify_attrib_local<person_name_att>(new_virtual_node, interest_person_name);
    G->insert_node(new_virtual_node);
    G->update_node(new_virtual_node);
    DSR::Edge waiting_edge = DSR::Edge::create<waiting_edge_type>(G->get_node("robot").value().id(), new_virtual_node.id());
    if (G->insert_or_assign_edge(waiting_edge))
    {
        std::cout << __FUNCTION__ << " Edge successfully inserted: " << G->get_node("robot").value().id() << "->" << new_virtual_node.id()
                    << " type: waiting_" << std::endl;
    }
    else
    {
        std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << G->get_node("robot").value().id() << "->" << new_virtual_node.id()
                    << " type: waiting_" << std::endl;
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
void SpecificWorker::remove_waiting_person_node()
{
    if(auto person_to_wait_in_mind = G->get_node("person_to_wait"); person_to_wait_in_mind.has_value())
    {
        if (auto has_edge = G->get_edge(G->get_node(robot_mind_name).value().id(),person_to_wait_in_mind.value().id(),"has");has_edge.has_value())
            G->delete_edge(G->get_node(robot_mind_name).value().id(),person_to_wait_in_mind.value().id(),"has");
        if (auto waiting_edge = G->get_edge(G->get_node(robot_name).value().id(),person_to_wait_in_mind.value().id(),"waiting");waiting_edge.has_value())
            G->delete_edge(G->get_node(robot_name).value().id(),person_to_wait_in_mind.value().id(),"waiting");
        G->delete_node(person_to_wait_in_mind.value().id());    
    }
}
uint64_t SpecificWorker::node_string2id(Plan currentPlan)
{
    auto person_id = currentPlan.get_attribute("person_node_id");
    uint64_t value;
    std::istringstream iss(person_id.toString().toUtf8().constData());
    iss >> value;
    return value;
}
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if(type == intention_type_name)
    {
        if(auto intention_node = G->get_node(id); intention_node.has_value())
        {
            if(intention_node.value().name() == robot_current_intention_name)
            {
                std::optional <std::string> plan = G->get_attrib_by_name<current_intention_att>(intention_node.value());
                if (plan.has_value())
                {
                    Plan received_plan =plan.value();
                    auto action_name = received_plan.get_action();
                    auto follow_action_name = QString::fromStdString("FOLLOW_PEOPLE");
                    if (action_name == follow_action_name)
                    {
                        interest_robot_intention_node_id = id;
                        current_plan = received_plan;
//                        interest_person_name = current_plan.get_attribute("person_name").toString().toStdString();
//                        interest_person_node_id = node_string2id(current_plan);
                        if(first_follow)
                        {
                            this->conversation_proxy->following(interest_person_name, "person");
                            first_follow = false;
                        }
                            

                    }
//                    auto talking_with_people_action_name = QString::fromStdString("TALKING_WITH_PEOPLE");
//                    if (action_name == talking_with_people_action_name or action_name == follow_action_name)
//                    {
//                        qInfo() << "5";
//                        interest_robot_intention_node_id = id;
//                        current_plan = received_plan;
//                        interest_person_name = current_plan.get_attribute("person_name").toString().toStdString();
//                        interest_person_node_id = node_string2id(current_plan);
//                        if (action_name == talking_with_people_action_name)
//                        {
//                            qInfo() << "6";
//                            std::cout << interest_person_name << std::endl;
//                            qInfo() << interest_person_node_id;
//                            this->conversation_proxy->sayHi(interest_person_name, "person");
//                            this->conversation_proxy->listenToHuman();
//                        }
//                        else if(action_name == follow_action_name)
//                        {
//                            qInfo() << "7";
//                            this->conversation_proxy->following(interest_person_name, "person");
//                        }
//                    }
                }
            }
        }
    }
}
void SpecificWorker::del_node_slot(std::uint64_t from)
{
    if(from == interest_robot_intention_node_id or from == interest_person_node_id)
    {
        stop_mission();
    }
}

void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{
    if(edge_tag == interacting_type_name)
    {
        qInfo() << __FUNCTION__ << "LLEGA";
    }
}

void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    if(type == interacting_type_name)
    {
        qInfo() << __FUNCTION__ << "LLEGA";
        if(auto edge = G->get_edge(from, to, type); edge.has_value())
        {
            if(auto person_interacting_node = G->get_node(to); person_interacting_node.has_value())
            {
                interest_person_node_id = to;
                if(auto person_name = G->get_attrib_by_name<person_name_att>(person_interacting_node.value()); person_name.has_value())
                {
                    interest_person_name = person_name.value();
                    std::cout << "NEW PERSON NAME:" << interest_person_name<< std::endl;
                    std::string person_name_o = person_name.value();
                    if(auto lost_followed_person = G->get_node("lost_person"); lost_followed_person.has_value())
                    {
                        if(auto lost_followed_person_name = G->get_attrib_by_name<person_name_att>(lost_followed_person.value()); lost_followed_person_name.has_value())
                        {
                            std::string lost_followed_person_name_o = lost_followed_person_name.value();
                            if(person_name_o == lost_followed_person_name_o)
                            {
                                std::string talking_mission_name = "TALKING_WITH_PEOPLE"; 
                                std::string follow_mission_name = "FOLLOW_PEOPLE"; 
                                if(auto lost_person_mission = G->get_attrib_by_name<person_mission_att>(lost_followed_person.value()); lost_person_mission.has_value())
                                {
                                    std::string lost_person_mission_name = lost_person_mission.value();
                                    if(lost_person_mission_name == talking_mission_name)
                                    {
                                        this->conversation_proxy->saySomething(interest_person_name, "Quería seguir charlando contigo");
                                        // Start Mission again
                                        start_mission(2);
                                    }
                                    else if(lost_person_mission_name == follow_mission_name)
                                    {
                                        this->conversation_proxy->saySomething(interest_person_name, "Que susto. Vuelvo a seguirte");
                                        // Start Mission again
                                        start_mission(0);                                    
                                    }
                                }
                            }
                        }
                    }
                    if(auto person_to_wait = G->get_node("person_to_wait"); person_to_wait.has_value())
                    {
                        if(auto person_to_wait_name = G->get_attrib_by_name<person_name_att>(person_to_wait.value()); person_to_wait_name.has_value())
                        {
                            std::string person_to_wait_name_o = person_to_wait_name.value();
                            if(person_name_o == person_to_wait_name_o)
                            {
                                this->conversation_proxy->saySomething(interest_person_name, "Sigamos con el paseo");
                                remove_waiting_person_node();
                                start_mission(0);
                            }
                        }
                    }
                    else
                    {
                        if(first_follow)
                            this->conversation_proxy->sayHi(interest_person_name, "person");
                        this->conversation_proxy->listenToHuman();
                        first_follow = false;
                    }

                }
            }
        }
    }
    if(type == lost_type_name)
    {
        this->conversation_proxy->lost(interest_person_name, "person");
    }
}

//void SpecificWorker::modify_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
//{
//    if(auto person_node = G->get_node(id); person_node.has_value())
//    {
//        // Check if the attribute that changed is the name of the person. That means that Face ID identified the person
////        if(person_node.value().type() == person_type_name)
//        if(person_node.value().type() == person_type_name)
//        {
//            if(std::find(att_names. begin(), att_names. end(), "person_name") != att_names. end())
//            {
//                if(auto person_name = G->get_attrib_by_name<person_name_att>(person_node.value()); person_name.has_value())
//                {
//                    interest_person_name = person_name.value();
//                    interest_person_node_id = person_node.value().id();
//                    this->conversation_proxy->lost(person_name.value(), "person");
//                }
//            }
//            if(std::find(att_names. begin(), att_names. end(), "is_ready") != att_names. end())
//            {
//                if(auto person_name = G->get_attrib_by_name<person_name_att>(person_node.value()); person_name.has_value())
//                {
//                    this->conversation_proxy->listenToHuman();
//                }
//
//            }
//        }
//    }
//}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


void SpecificWorker::AgenteConversacional_componentState(int state)
{
//    switch(state)
//    {
//        case 0:
//            isTalking = false;
//            remove_intention_edge("talking");
//    }
}

int SpecificWorker::AgenteConversacional_situationChecking() 
{
    if(auto lost_edges = G->get_edges_by_type(lost_type_name); lost_edges.size() > 0)
        return 1;
    else
        return 0;
}

void SpecificWorker::AgenteConversacional_asynchronousIntentionReceiver(int intention) {
    if (auto robot_node = G->get_node("robot"); robot_node.has_value()) {
        auto robot_node_value = robot_node.value();
        switch (intention) {
            case 0:
                cout << "Seguir" << endl;
                this->conversation_proxy->following(interest_person_name, "person");
                // stop_mission();
                first_follow = false;
                create_ask_for_follow_plan();
                start_mission(0);
                break;
           case 1:
                cout << "Hablar" << endl;
                
                // stop_mission();
                create_talking_plan();
                start_mission(2);
               break;
            case 2:
                cout << "Dejar de seguir" << endl;
                this->conversation_proxy->stopFollowing(interest_person_name, "person");
                // stop_mission();
                create_ask_for_stop_following_plan();
                start_mission(1);
                first_follow = true;
                break;
           case 3:
                cout << "Esperar" << endl;
                // Check if person is being followed
                if(auto intention_node = G->get_node(robot_current_intention_name); intention_node.has_value())
                {
                    std::optional <std::string> plan = G->get_attrib_by_name<current_intention_att>(intention_node.value());
                    if (plan.has_value())
                    {
                        Plan received_plan =plan.value();
                        auto action_name = received_plan.get_action();
                        auto follow_action_name = QString::fromStdString("FOLLOW_PEOPLE");
                        if (action_name == follow_action_name)
                        {
                            this->conversation_proxy->waiting(interest_person_name, "person");
                            // Generate virtual_person node named "waiting_person"
                            create_waiting_person_node();
                            // Stop current mission
                            stop_mission();
                            create_ask_for_stop_following_plan();
                            first_follow = false;
                            start_mission(1);
                        }   
                        else
                        {
                            // Mensaje indicando que no se está siguiendo a la persona
                        }
                    }
                    
                }
                else
                {
                    // Mensaje indicando que no se está siguiendo a la persona
                }
                break;
            case 4:
                qInfo() << "DEJAR DE HABLAR";

                create_ask_for_stop_talking_plan();
                start_mission(3);
                first_follow = true;
                break;

            case -1:
                cout << "No identifica keyword" << endl;
//                isListening = false;
                this->conversation_proxy->talking(interest_person_name, "person", "preguntar");
                break;
            case -99:
                if(auto lost_edge = G->get_edge(robot_node_value.id(), interest_person_node_id, lost_type_name))
                {
                    this->conversation_proxy->lost(interest_person_name, "person");
                    break;
                }
                if(auto interacting_edge = G->get_edge(robot_node_value.id(), interest_person_node_id, interacting_type_name))
                {
                    this->conversation_proxy->listenToHuman();
                }

                break;
        }
    }
}


/**************************************/
// From the RoboCompConversation you can call this methods:
// this->conversation_proxy->following(...)
// this->conversation_proxy->isBlocked(...)
// this->conversation_proxy->isFollowing(...)
// this->conversation_proxy->isTalking(...)
// this->conversation_proxy->listenToHuman(...)
// this->conversation_proxy->lost(...)
// this->conversation_proxy->sayHi(...)
// this->conversation_proxy->talking(...)

