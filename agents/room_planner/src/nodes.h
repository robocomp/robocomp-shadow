#include <behaviortree_cpp/bt_factory.h>
//#include <behaviortree_cpp/action_node.h>
#include "behaviortree_cpp/behavior_tree.h"
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"

namespace Nodes
{

////////////////////////////////////////// CONDITION NODES ////////////////////////////////////////////
BT::NodeStatus IsDoor(std::shared_ptr<DSR::DSRGraph> G_);

class IsPerson : public BT::ConditionNode {
    public:
        IsPerson(const std::string& name, const BT::NodeConfig& config) : BT::ConditionNode(name, config) {}
        IsPerson(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<DSR::DSRGraph> G_) :
        BT::ConditionNode(name, config), G(G_) {}

    // It is mandatory to define this STATIC method.
    static BT::PortsList providedPorts()
    {
        // This action has a single input port called "message"
        return { BT::InputPort<std::string>("person") };
    }

    protected:
        virtual BT::NodeStatus tick() override;

    private:
        std::shared_ptr<DSR::DSRGraph> G;
};

class Reached : public BT::ConditionNode {
public:
    Reached(const std::string& name, const BT::NodeConfig& config) : BT::ConditionNode(name, config) {}
    Reached(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<DSR::DSRGraph> G_) :
            BT::ConditionNode(name, config), G(G_) {}

    // It is mandatory to define this STATIC method.
    static BT::PortsList providedPorts()
    {
        // This action has a single input port called "message"
        return { BT::InputPort<std::string>("person") };
    }

protected:
    virtual BT::NodeStatus tick() override;

private:
    std::shared_ptr<DSR::DSRGraph> G;
};

//////////////////////////////////////////// ACTION NODES ////////////////////////////////////////////
class MoveToDoor : public BT::SyncActionNode {
    public:
        MoveToDoor(const std::string& name) : BT::SyncActionNode(name, {}) {}
        MoveToDoor(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_) :
        BT::SyncActionNode(name, {}), G(G_) {}

    static BT::PortsList providedPorts()
    {
        // This action has a single input port called "message"
        return { BT::InputPort<std::string>("person") };
    }

    protected:
        virtual BT::NodeStatus tick() override;

    private:
        std::shared_ptr<DSR::DSRGraph> G;
};

class MoveToPerson : public BT::SyncActionNode {
public:
    MoveToPerson(const std::string& name, const BT::NodeConfig& config) : BT::SyncActionNode(name, config) {}
    MoveToPerson(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<DSR::DSRGraph> G_) :
            BT::SyncActionNode(name, config), G(G_) {}

    static BT::PortsList providedPorts()
    {
        // This action has a single input port called "message"
        return { BT::InputPort<std::string>("person") };
    }

protected:
    virtual BT::NodeStatus tick() override;

private:
    std::shared_ptr<DSR::DSRGraph> G;
};

// Example of custom SyncActionNode (synchronous action)
// without ports.
class Rotate : public BT::StatefulActionNode
{
public:
    Rotate(const std::string& name) :
            BT::StatefulActionNode(name, {})
    {}
    Rotate(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_) :
            BT::StatefulActionNode(name, {}), G(G_) {}

    // this function is invoked once at the beginning.
    BT::NodeStatus onStart() override;

    // You must override the virtual function onRunning()
    BT::NodeStatus onRunning() override;

    // callback to execute if the action was aborted by another node
    void onHalted() override;

private:
    std::shared_ptr<DSR::DSRGraph> G;
};

/*class GoThroughDoor : public BT::SyncActionNode {
    public:
        GoThroughDoor(const std::string& name) : BT::SyncActionNode(name, {}) {}
        GoThroughDoor(const std::string& name, std::shared_ptr<DSR::DSRGraph> G_) :
        BT::SyncActionNode(name, {}), G(G_) {}

    protected:
        virtual BT::NodeStatus tick() override;

    private:
        std::shared_ptr<DSR::DSRGraph> G;
};*/

} // namespace Nodes