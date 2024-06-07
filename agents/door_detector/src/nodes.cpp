#include "nodes.h"

namespace Nodes
{

#pragma region CONDITION_NODES

    BT::NodeStatus ExistsParent::tick()
    {
        if (auto parent = G->get_attrib_by_name<parent_att>(aff_id); parent.has_value())
        {
            std::cout << "Parent found:" << parent.value() << std::endl;
            setOutput("door_id", parent.value());
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            std::cout << "Parent not found" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

    BT::NodeStatus IsIntentionCompleted::tick()
    {
        std::cout << "IntentionCompleted" << std::endl;

        auto parent = G->get_attrib_by_name<parent_att>(aff_id);
        //get parent attribute from aff_id
        if (!parent.has_value())
            return BT::NodeStatus::FAILURE;

        if (auto has_intention_edge = G->get_edge(params.ROBOT_ID, parent.value(),
                                                  "has_intention"); has_intention_edge.has_value())
        {
            if (auto active = G->get_attrib_by_name<active_att>(has_intention_edge.value()); active.has_value())
            {
                if (auto state = G->get_attrib_by_name<state_att>(has_intention_edge.value()); state.has_value())
                {
                    //get afford node and check if has value
                    std::optional<DSR::Node> aff_node_ = G->get_node(aff_id);
                    if (!aff_node_.has_value())
                    { qWarning() << __FUNCTION__ << " No affordance node found"; return BT::NodeStatus::FAILURE; }
                    auto aff_node = aff_node_.value();

                    //get aff_node state attribute value nad check if has value
                    if (auto aff_state = G->get_attrib_by_name<bt_state_att>(aff_node); state.has_value())
                    {
                        if (state.value() == "completed" && active.value())
                        {
                            //change affordance node attribute called bt_state to completed
//                            G->add_or_modify_attrib_local<bt_state_att>(aff_node, std::string("completed"));
//                            G->update_node(aff_node);

                            //delete intention edge
                            G->delete_edge(params.ROBOT_ID, parent.value(), "has_intention");
                            std::cout << "Intention edge found and completed" << std::endl;
                            return BT::NodeStatus::SUCCESS;
                        }
                        else if (state.value() == "aborted" || state.value() == "failed" || aff_state == "aborted" || aff_state == "failed")
                             {
                                std::cout << "Intention edge found and failed" << std::endl;
                                return BT::NodeStatus::FAILURE;
                             }
                             else if (state.value() == "in_progress" && aff_state == "in_progress")
                             {
                                std::cout << "Intention edge found but not completed" << std::endl;
                                return BT::NodeStatus::RUNNING;
                             }
                    }
                }
            } else { std::cout << "Active attribute not found" << std::endl; }
            return BT::NodeStatus::FAILURE;
        }
        return BT::NodeStatus::FAILURE;
    }

#pragma endregion CONDITION_NODES

#pragma region ACTION_NODES

    BT::NodeStatus CreateHasIntention::tick()
    {
        std::cout << this->name() << std::endl;

        BT::Expected<uint64_t> door_id_ = getInput<uint64_t>("door_id");
        BT::Expected<int> target_vector_ = getInput<int>("target_vector");
        // Check if expected is valid. If not, throw its error
        if (!door_id_ )
        { throw BT::RuntimeError("Missing required input [door_id]: ", door_id_.error() ); }
        if (!target_vector_ )
        { throw BT::RuntimeError("Missing required input [target_vector]: ", target_vector_.error() ); }

        uint64_t door_id = door_id_.value();
        int target_vector = target_vector_.value();

        //get afford node and check if has value
        std::optional<DSR::Node> aff_node_ = G->get_node(aff_id);
        if (!aff_node_.has_value())
        { qWarning() << __FUNCTION__ << " No affordance node found"; return BT::NodeStatus::FAILURE; }

        //get affordance node
        DSR::Node aff_node = aff_node_.value();
        //change affordance node attribute called bt_state to waiting
        G->add_or_modify_attrib_local<bt_state_att>(aff_node, std::string("in_progress"));
        G->update_node(aff_node);

        DSR::Edge intention_edge = DSR::Edge::create<has_intention_edge_type>(params.ROBOT_ID, door_id);
        G->add_or_modify_attrib_local<active_att>(intention_edge, true);
        G->add_or_modify_attrib_local<state_att>(intention_edge, std::string("in_progress"));

        std::vector<float> offset_target = {0.f, 0.f + (float) target_vector, 0.f};
        G->add_or_modify_attrib_local<offset_xyz_att>(intention_edge, offset_target);

        std::vector<float> tolerance = {100.f, 100.f, 0.f, 0.f, 0.f, 0.5f};
        G->add_or_modify_attrib_local<tolerance_att>(intention_edge, tolerance);

        if (G->insert_or_assign_edge(intention_edge))
        {
            std::cout << __FUNCTION__ << " Intention edge successfully inserted, going in front of door: " << std::endl;
            return BT::NodeStatus::SUCCESS;
        }
        else
            std::cout << __FUNCTION__ << " Fatal error inserting Intention edge: " << std::endl;

        return BT::NodeStatus::FAILURE;
    }

#pragma endregion ACTION_NODES

} // namespace Nodes
