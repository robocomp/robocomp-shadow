#pragma once

//#include <behaviortree_cpp/action_node.h>
#include "behaviortree_cpp/behavior_tree.h"
#include <behaviortree_cpp/blackboard.h>
#include <behaviortree_cpp/bt_factory.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <cppitertools/range.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/filter.hpp>
#include <cppitertools/chunked.hpp>
#include "../../door_detector/src/params.h"

namespace Nodes
{

    #pragma region CONDITION_NODES

    BT::NodeStatus ExistsRoom(std::shared_ptr<DSR::DSRGraph> G);

    class InRoomCenter : public BT::ConditionNode
    {
        public:
            InRoomCenter(const std::string& name) : BT::ConditionNode(name, {}) {}
            InRoomCenter(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_) :
                BT::ConditionNode(name, {}), G(G_) {}

        protected:
            virtual BT::NodeStatus tick() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
            rc::Params params;
            int time_in_center = 2500;
            bool initialize_counter = true;
            std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    };

    class ExistsCurrent : public BT::ConditionNode
    {
    public:
        ExistsCurrent(const std::string& name) : BT::ConditionNode(name, {}) {}
        ExistsCurrent(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_, const std::function<void()>& _insert_measured_corners_loaded_room) :
                BT::ConditionNode(name, {}), G(G_), insert_measured_corners_loaded_room(_insert_measured_corners_loaded_room) {}

    protected:
        virtual BT::NodeStatus tick() override;

    private:
        std::shared_ptr<DSR::DSRGraph> G;
        rc::Params params;
        std::function<void()> insert_measured_corners_loaded_room;
    };

#pragma endregion CONDITION_NODES

#pragma region ACTION_NODES

    class CreateTargetEdge : public BT::SyncActionNode
    {
        public:
            CreateTargetEdge(const std::string& name) : BT::SyncActionNode(name, {}) {}
            CreateTargetEdge(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_) :
                    BT::SyncActionNode(name, {}), G(G_) {}

        protected:
            virtual BT::NodeStatus tick() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
    };

    class DeleteTargetEdge : public BT::SyncActionNode
    {
        public:
            DeleteTargetEdge(const std::string& name) : BT::SyncActionNode(name, {}) {}
            DeleteTargetEdge(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_) :
                    BT::SyncActionNode(name, {}), G(G_) {}

        protected:
            virtual BT::NodeStatus tick() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
    };

    class CreateRoom : public BT::SyncActionNode
    {
        public:
            CreateRoom(const std::string& name) : BT::SyncActionNode(name, {}) {}
            CreateRoom(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_, const std::function<void()>& _create_room) :
                BT::SyncActionNode(name, {}), G(G_), create_room(_create_room) {}

        protected:
            virtual BT::NodeStatus tick() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
            std::function<void()> create_room;
    };

    class UpdateRoom : public BT::StatefulActionNode
    {
        public:
            UpdateRoom(const std::string& name) : BT::StatefulActionNode(name, {}) {}
            UpdateRoom(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_, const std::function<void()>& _check_corner_matching
                       , const std::function<void()>& _update_room) :
                    BT::StatefulActionNode(name, {}), G(G_), check_corner_matching(_check_corner_matching), update_room(_update_room) {}

        protected:
            // this function is invoked once at the beginning.
            BT::NodeStatus onStart() override;

            // You must override the virtual function onRunning()
            BT::NodeStatus onRunning() override;

            // callback to execute if the action was aborted by another node
            void onHalted() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
            std::function<void()> check_corner_matching;
            std::function<void()> update_room;
    };

    class RoomStabilitation : public BT::StatefulActionNode
    {
        public:
            RoomStabilitation(const std::string& name) : BT::StatefulActionNode(name, {}) {}
            RoomStabilitation(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_, const std::function<void()>& _room_stabilitation) :
                    BT::StatefulActionNode(name, {}), G(G_), room_stabilitation(_room_stabilitation) {}

            // this function is invoked once at the beginning.
            BT::NodeStatus onStart() override;

            // You must override the virtual function onRunning()
            BT::NodeStatus onRunning() override;

            // callback to execute if the action was aborted by another node
            void onHalted() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
            std::function<void()> room_stabilitation;
            rc::Params params;

    };

#pragma endregion ACTION_NODES

} // namespace Nodes