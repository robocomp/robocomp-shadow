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
#include "params.h"

namespace Nodes
{

    #pragma region CONDITION_NODES

    class ExistsParent : public BT::ConditionNode
    {
        public:
        ExistsParent(const std::string& name, const BT::NodeConfig& config) : BT::ConditionNode(name, config) {}
        ExistsParent(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<DSR::DSRGraph> G_, uint64_t aff_id_) :
                BT::ConditionNode(name, config), G(G_), aff_id(aff_id_) {}

        static BT::PortsList providedPorts()
        {
            return { BT::OutputPort<uint64_t>("door_id") };
        }

        protected:
            virtual BT::NodeStatus tick() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
            uint64_t aff_id;
    };

    class IsIntentionCompleted : public BT::ConditionNode
    {
    public:
        IsIntentionCompleted(const std::string& name, const BT::NodeConfig& config) : BT::ConditionNode(name, config) {}
        IsIntentionCompleted(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<DSR::DSRGraph> G_, uint64_t aff_id_) :
                BT::ConditionNode(name, config), G(G_), aff_id(aff_id_) {}

        // It is mandatory to define this STATIC method.
        static BT::PortsList providedPorts()
        {
            return { BT::InputPort<uint64_t>("door_id") };
        }

    protected:
        virtual BT::NodeStatus tick() override;

    private:
        std::shared_ptr<DSR::DSRGraph> G;
        uint64_t aff_id;
        rc::Params params;
    };

#pragma endregion CONDITION_NODES

#pragma region ACTION_NODES

    class CreateHasIntention : public BT::SyncActionNode
    {
        public:
            CreateHasIntention(const std::string& name, const BT::NodeConfig& config) : BT::SyncActionNode(name, config) {}
            CreateHasIntention(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<DSR::DSRGraph> G_, uint64_t aff_id_) :
                    BT::SyncActionNode(name, config), G(G_), aff_id(aff_id_) {}

        // It is mandatory to define this STATIC method.
    static BT::PortsList providedPorts()
    {
        return { BT::InputPort<uint64_t>("door_id"), BT::InputPort<int>("target_vector") };
    }

    protected:
            virtual BT::NodeStatus tick() override;

        private:
            std::shared_ptr<DSR::DSRGraph> G;
            uint64_t aff_id;
            rc::Params params;
    };

#pragma endregion ACTION_NODES

} // namespace Nodes