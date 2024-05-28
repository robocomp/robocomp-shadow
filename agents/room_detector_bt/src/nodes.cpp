#include "nodes.h"

namespace Nodes
{

#pragma region CONDITION_NODES

    BT::NodeStatus ExistsRoom(std::shared_ptr<DSR::DSRGraph> G)
    {
        if (std::optional<DSR::Node> room_node_ = G->get_node("room"); room_node_.has_value())
        {
            DSR::Node room_node = room_node_.value();

            std::cout << "Room node found" << room_node.id() << std::endl;
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
        if (distance < 150.)
        {
            static auto start = std::chrono::high_resolution_clock::now();

            if(time_in_center < 0)
            {
                std::cout << "In_room_center" << std::endl;
                return BT::NodeStatus::SUCCESS;
            }
            auto end = std::chrono::high_resolution_clock::now();
            //get end time and calculate duration
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            time_in_center -= elapsed;
            start = end;
            std::cout << "Time in center:" << time_in_center << std::endl;
            return BT::NodeStatus::FAILURE;
        }
        else
        {
            std::cout << "Not in room center" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

#pragma endregion CONDITION_NODES

#pragma region ACTION_NODES

    BT::NodeStatus CreateTargetEdge::tick()
    {
        std::cout << this->name() << std::endl;

        if (std::optional<DSR::Node> shadow_node_ = G->get_node("Shadow"); shadow_node_.has_value())
        {
            DSR::Node shadow_node = shadow_node_.value();
            DSR::Edge target_edge = DSR::Edge::create<TARGET_edge_type>(shadow_node.id(), shadow_node.id());
            if (G->insert_or_assign_edge(target_edge))
            {
                std::cout << __FUNCTION__ << " Target edge successfully inserted: " << std::endl;
                return BT::NodeStatus::SUCCESS;
            }
            else
                std::cout << __FUNCTION__ << " Fatal error inserting new edge: " << std::endl;
        }

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
            DSR::Node shadow_node = shadow_node_.value();
            if (std::optional<DSR::Edge> target_edge_ = G->get_edge(shadow_node.id(), shadow_node.id(),"TARGET"); target_edge_.has_value())
            {
                DSR::Edge target_edge = target_edge_.value();
                if ( G->delete_edge(shadow_node.id(), shadow_node.id(), "TARGET") )
                {
                    std::cout << __FUNCTION__ << " Target edge successfully deleted: " << std::endl;
                } else
                    std::cout << __FUNCTION__ << " Fatal error deleting edge: " << std::endl;
            }
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
        return BT::NodeStatus::RUNNING;
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
