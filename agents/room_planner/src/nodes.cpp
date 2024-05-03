#include "nodes.h"

namespace Nodes
{
////////////////////////////////////////// CONDITION NODES ////////////////////////////////////////////
    BT::NodeStatus IsDoor(std::shared_ptr<DSR::DSRGraph> G)
    {
        if (std::optional<DSR::Node> person_node_ = G->get_node("person_1"); person_node_.has_value())
        {
            DSR::Node person_node = person_node_.value();

            std::cout << "Person node found" << person_node.id() << std::endl;
            //Pasar nodo persona por los puertos? para no checkear en el siguiente nodo si se encuentra o no?
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            std::cout << "Person node not found" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

    BT::NodeStatus IsPerson::tick()
    {
        std::cout << this->name() << std::endl;

        BT::Expected<std::string> msg = getInput<std::string>("person");
        // Check if expected is valid. If not, throw its error
        if (!msg)
            throw BT::RuntimeError("ISPERSON: missing required input [message]: ", msg.error() );

        if (std::optional<DSR::Node> person_node_ = G->get_node("person_" + msg.value()); person_node_.has_value())
        {
            DSR::Node person_node = person_node_.value();

            std::cout << "Person node found" << person_node.id() << std::endl;
            //Pasar nodo persona por los puertos? para no checkear en el siguiente nodo si se encuentra o no?
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            std::cout << "Person node not found" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

    BT::NodeStatus Reached::tick()
    {
        std::cout << this->name() << std::endl;

        BT::Expected<std::string> msg = getInput<std::string>("person");
        // Check if expected is valid. If not, throw its error
        if (!msg)
            throw BT::RuntimeError("REACHED: missing required input [message]: ", msg.error() );

        if (std::optional<DSR::Node> person_node_ = G->get_node("person_" + msg.value()); person_node_.has_value())
        {
            DSR::Node person_node = person_node_.value();
            DSR::Node robot_node;
            if (std::optional<DSR::Node> robot_node_ = G->get_node("Shadow"); robot_node_.has_value())
                robot_node = robot_node_.value();
            else
                return BT::NodeStatus::FAILURE;

            std::unique_ptr<DSR::RT_API> rt = G->get_rt_api();

            if (auto edge_robot_ = rt->get_edge_RT(robot_node, person_node.id()); edge_robot_.has_value())
            {
                if (auto rt_translation = G->get_attrib_by_name<rt_translation_att>(edge_robot_.value()); rt_translation.has_value())
                {
                    auto rt_translation_value = rt_translation.value().get();
                    auto norm = std::sqrt(rt_translation_value[0] * rt_translation_value[0] + rt_translation_value[1] * rt_translation_value[1]);
                    if (norm < 200)
                    {
                        std::cout << "Reached person successful" << std::endl;
                        G->delete_edge(robot_node.id(), person_node.id(), "goto_action");
                        return BT::NodeStatus::SUCCESS;
                    }
                    else
                    {
                        std::cout << "Not reached person" << std::endl;
                        return BT::NodeStatus::FAILURE;
                    }
                }
            }else
            {
                std::cout << "Edge not found" << std::endl;
                return BT::NodeStatus::FAILURE;
            }
        }

        std::cout << "Reached person failed" << std::endl;
        return BT::NodeStatus::FAILURE;
    }

//////////////////////////////////////////// ACTION NODES ////////////////////////////////////////////

    BT::NodeStatus Rotate::onStart()
    {
        std::cout << "onStart" << std::endl;
        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus Rotate::onRunning()
    {
        std::cout << "Rotating" << std::endl;
        return BT::NodeStatus::RUNNING;
    }

    void Rotate::onHalted()
    {
        std::cout << "Stop rotating" << std::endl;
    }

    BT::NodeStatus MoveToPerson::tick()
    {
        std::cout << this->name() << std::endl;

        BT::Expected<std::string> msg = getInput<std::string>("person");
        // Check if expected is valid. If not, throw its error
        if (!msg)
            throw BT::RuntimeError("MOVETOPERSON: missing required input [message]: ", msg.error());

        if (std::optional<DSR::Node> person_node_ = G->get_node("person_" + msg.value()); person_node_.has_value())
        {
            DSR::Node person_node = person_node_.value();
            DSR::Node robot_node;
            if (std::optional<DSR::Node> robot_node_ = G->get_node("Shadow"); robot_node_.has_value())
                robot_node = robot_node_.value();
            else
                return BT::NodeStatus::FAILURE;

            //Create a new edge from robot to person_node called target
            DSR::Edge goto_edge = DSR::Edge::create<goto_action_edge_type>(robot_node.id(), person_node.id());
            G->insert_or_assign_edge(goto_edge);
            std::cout << "Move to person successful" << std::endl;

            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            std::cout << "Move to person failed" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }


//    BT::NodeStatus MoveToDoor::tick()
//    {
//        if (std::optional<DSR::Node> person_node_ = G->get_node("person_1 "); person_node_.has_value())
//        {
//            DSR::Node person_node = person_node_.value();
//            DSR::Node robot_node;
//            if (std::optional<DSR::Node> robot_node_ = G->get_node("Shadow"); robot_node_.has_value())
//                robot_node = robot_node_.value();
//            else
//                return BT::NodeStatus::FAILURE;
//
//            //Create a new edge from robot to person_node called target
//            DSR::Edge goto_edge = DSR::Edge::create<goto_action_edge_type>(robot_node.id(), person_node.id());
//            G->insert_or_assign_edge(goto_edge);
//            std::cout << "Move to door successful" << std::endl;
//
//            return BT::NodeStatus::SUCCESS;
//        }
//        else
//        {
//            std::cout << "Move to door failed" << std::endl;
//            return BT::NodeStatus::FAILURE;
//        }
//    }
//
//    BT::NodeStatus GoThroughDoor::tick()
//    {
///*        if (std::optional<DSR::Node> shadow_node_ = G->get_node("Shadow"); shadow_node_.has_value())
//        {
//            DSR::Node shadow_node = shadow_node_.value();
//
//            std::cout << "Shadow node found" << shadow_node.id() << std::endl;
//        }
//        return BT::NodeStatus::FAILURE;*/
//
//    }
}

