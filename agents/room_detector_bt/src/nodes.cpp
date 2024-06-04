#include "nodes.h"

namespace Nodes
{

#pragma region CONDITION_NODES

    BT::NodeStatus ExistsRoom(std::shared_ptr<DSR::DSRGraph> G)
    {
        if (std::optional<DSR::Node> room_node_ = G->get_node("room"); room_node_.has_value())
        {
            DSR::Node room_node = room_node_.value();

            std::cout << __FUNCTION__  << " Current room node found" << room_node.id() << std::endl;
            //Pasar nodo persona por los puertos? para no checkear en el siguiente nodo si se encuentra o no?
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            std::cout << "Room node not found" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

    BT::NodeStatus InRoomCenter::tick()
    {
        //get start time
        int distance = distance_to_center();

        //Calculate distance to room center
        if (distance < 150.)    // TODO: Move to params
        {
            static auto start = std::chrono::high_resolution_clock::now();

            if(time_in_center < 0)
            {
                //std::cout << __FUNCTION__ << " In_room_center" << std::endl;
                return BT::NodeStatus::SUCCESS;
            }
            auto end = std::chrono::high_resolution_clock::now();
            //get end time and calculate duration
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            time_in_center -= elapsed;
            start = end;
            //std::cout << __FUNCTION__ << " Time in center:" << time_in_center << std::endl;
            return BT::NodeStatus::FAILURE;
        }
        else
        {
            //std::cout << __FUNCTION__ << " Not in room center" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

#pragma endregion CONDITION_NODES

#pragma region ACTION_NODES

    BT::NodeStatus CreateTargetEdge::tick()
    {
        std::cout << this->name() << std::endl;

        DSR::Node shadow_node;
        if (std::optional<DSR::Node> shadow_node_ = G->get_node("Shadow"); not shadow_node_.has_value())
        { qWarning() << __FUNCTION__ << " No Shadow node found"; return BT::NodeStatus::FAILURE;}
        else
            shadow_node = shadow_node_.value();

        DSR::Node room_measured = DSR::Node::create<room_node_type>("room_measured");
        G->add_or_modify_attrib_local<pos_x_att>(room_measured, (float)(rand()%(170)));
        G->add_or_modify_attrib_local<pos_y_att>(room_measured, (float)(rand()%170));
        G->add_or_modify_attrib_local<obj_checked_att>(room_measured, false);
        G->add_or_modify_attrib_local<level_att>(room_measured, G->get_node_level(shadow_node).value());
        G->insert_node(room_measured);
        G->update_node(room_measured);
        room_measured = G->get_node(room_measured.id()).value();
        G->get_rt_api()->insert_or_assign_edge_RT(shadow_node, room_measured.id(), { 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f });
        DSR::Edge intention_edge = DSR::Edge::create<has_intention_edge_type>(shadow_node.id(), room_measured.id());

        // Valid attribute of intention edge. Should be set by the SCHEDULER
        G->add_or_modify_attrib_local<active_att>(intention_edge, false);

        /// Set robot target to a position close to the room center
        std::vector<float> offset_target = {0, 0, 0};
        G->add_or_modify_attrib_local<offset_xyz_att>(intention_edge, offset_target);

        /// Set tolerance to reach the target
        std::vector<float> tolerance = {100, 100, 0.f, 0.f, 0.f, 0.5};
        G->add_or_modify_attrib_local<tolerance_att>(intention_edge, tolerance);

        /// Set intention status to "waiting"
        std::string intention_status = "waiting";
        G->add_or_modify_attrib_local<state_att>(intention_edge, intention_status);
        
        if (G->insert_or_assign_edge(intention_edge))
        {
            std::cout << __FUNCTION__ << " Intention edge successfully inserted: " << std::endl;
            return BT::NodeStatus::SUCCESS;
        }
        else
            std::cout << __FUNCTION__ << " Fatal error inserting Intention edge: " << std::endl;

        return BT::NodeStatus::FAILURE;
    }

    BT::NodeStatus DeleteTargetEdge::tick()
    {
        std::cout << this->name() << std::endl;
        if (std::optional<DSR::Node> shadow_node_ = G->get_node("Shadow"); shadow_node_.has_value())
        {
            DSR::Node shadow_node = shadow_node_.value();
            if (std::optional<DSR::Edge> target_edge_ = G->get_edge(shadow_node.id(), shadow_node.id(),"TARGET"); target_edge_.has_value())
            {
                DSR::Edge target_edge = target_edge_.value();
                if ( G->delete_edge(shadow_node.id(), shadow_node.id(), "TARGET") )
                {
                    std::cout << __FUNCTION__ << " Target edge successfully deleted: " << std::endl;
                    return BT::NodeStatus::SUCCESS;
                } else
                    std::cout << __FUNCTION__ << " Fatal error deleting edge: " << std::endl;
            }
        }
        return BT::NodeStatus::FAILURE;
    }

    BT::NodeStatus RoomStabilitation::onStart()
    {
        std::cout << this->name() << "onStart" << std::endl;
        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus RoomStabilitation::onRunning()
    {
        std::cout << this->name() << "onRunning" << std::endl;
        room_stabilitation();
        return BT::NodeStatus::RUNNING;
    }

    void RoomStabilitation::onHalted()
    {
        std::cout << "Room stabilitation halted" << std::endl;
        if (std::optional<DSR::Node> shadow_node_ = G->get_node("Shadow"); shadow_node_.has_value())
        {
            if(auto room_measured = G->get_node("room_measured"); room_measured.has_value())
            {
                if (G->delete_node(room_measured.value().id()))
                    std::cout << __FUNCTION__ << " Room measured node successfully deleted: " << std::endl;
                else
                    std::cout << __FUNCTION__ << " Fatal error deleting node: " << std::endl;
            }

//            DSR::Node shadow_node = shadow_node_.value();
//            if (std::optional<DSR::Edge> target_edge_ = G->get_edge(shadow_node.id(), shadow_node.id(),"TARGET"); target_edge_.has_value())
//            {
//                DSR::Edge target_edge = target_edge_.value();
//                if ( G->delete_edge(shadow_node.id(), shadow_node.id(), "TARGET") )
//                {
//                    std::cout << __FUNCTION__ << " Target edge successfully deleted: " << std::endl;
//
//                } else
//                    std::cout << __FUNCTION__ << " Fatal error deleting edge: " << std::endl;
//            }
        }
    }

    BT::NodeStatus CreateRoom::tick()
    {
        std::cout << this->name() << std::endl;
        create_room();
        return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus UpdateRoom::onStart()
    {
        std::cout << this->name() << "onStart" << std::endl;
        check_corner_matching();
        return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus UpdateRoom::onRunning()
    {
//        std::cout << this->name() << "onRunning" << std::endl;
//        check_corner_matching();
//        update_room();
        return BT::NodeStatus::RUNNING;
    }

    void UpdateRoom::onHalted()
    {
        std::cout << this->name() << "onHalted" << std::endl;
    }

#pragma endregion ACTION_NODES

} // namespace Nodes
