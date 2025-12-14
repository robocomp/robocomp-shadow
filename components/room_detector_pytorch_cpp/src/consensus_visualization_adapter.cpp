/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "consensus_visualization_adapter.h"
#include "consensus_graph.h"
#include <QDebug>

ConsensusVisualizationAdapter::ConsensusVisualizationAdapter(ConsensusManager* manager,
                                                             ConsensusGraphWidget* widget,
                                                             QObject* parent)
    : QObject(parent)
    , manager_(manager)
    , widget_(widget)
{
    if (!manager_ || !widget_)
    {
        qWarning() << "ConsensusVisualizationAdapter: null manager or widget";
        return;
    }

    // Connect ConsensusManager signals to our slots
    connect(manager_, &ConsensusManager::initialized,
            this, &ConsensusVisualizationAdapter::onManagerInitialized);

    connect(manager_, &ConsensusManager::consensusReady,
            this, &ConsensusVisualizationAdapter::onConsensusReady);

    connect(manager_, &ConsensusManager::graphChanged,
        this, &ConsensusVisualizationAdapter::onGraphChanged);

    qDebug() << "ConsensusVisualizationAdapter: Connected manager to widget";

    // If manager is already initialized, trigger initial update
    if (manager_->isInitialized())
    {
        onManagerInitialized();
    }
}

void ConsensusVisualizationAdapter::onGraphChanged()
{
    const auto& graph = manager_->getGraph();
    const auto& gtsam_graph = graph.getGraph();
    const auto& values     = graph.getValues();

    GraphUpdateEvent event;
    event.type   = GraphEventType::CONSTRAINT_ADDED;   // routes to updateFromGTSAM in your widget
    event.graph  = &gtsam_graph;
    event.values = &values;

    QMetaObject::invokeMethod(widget_, [w = widget_, event]()
    {
        w->onGraphUpdated(event);
    }, Qt::QueuedConnection);
}

void ConsensusVisualizationAdapter::onManagerInitialized()
{
    qDebug() << "ConsensusVisualizationAdapter: Manager initialized, updating visualization";

    // Get current graph state
    const auto& graph = manager_->getGraph();
    const auto& gtsam_graph = graph.getGraph();
    const auto& values = graph.getValues();

    // Create event for room initialization
    GraphUpdateEvent event;
    event.type = GraphEventType::ROOM_INITIALIZED;
    event.graph = &gtsam_graph;
    event.values = &values;

    QMetaObject::invokeMethod(widget_, [w = widget_, event]()
    {
        w->onGraphUpdated(event);
    }, Qt::QueuedConnection);

}

void ConsensusVisualizationAdapter::onConsensusReady(const ConsensusResult& result)
{
    // qDebug() << "ConsensusVisualizationAdapter: Consensus ready, updating visualization";
    // qDebug() << "  Error:" << result.initial_error << "→" << result.final_error
    //          << "Iterations:" << result.iterations;

    // Get current graph state
    const auto& graph = manager_->getGraph();
    const auto& gtsam_graph = graph.getGraph();
    const auto& values = graph.getValues();

    // Extract covariances from result. We store them keyed by the full
    // GTSAM variable key so that the visualization widget can attach
    // the correct covariance matrix to each GraphNode.
    std::map<size_t, Eigen::Matrix3d> covariances;

    // Room covariance
    {
        gtsam::Symbol room_sym = ConsensusGraph::RoomSymbol();
        covariances[static_cast<size_t>(room_sym.key())] = result.room_covariance;
    }

    // Wall covariances
    for (const auto& [wall_id, cov] : result.wall_covariances)
    {
        gtsam::Symbol wall_sym = ConsensusGraph::WallSymbol(wall_id);
        covariances[static_cast<size_t>(wall_sym.key())] = cov;
    }

    // Robot covariances
    for (size_t i = 0; i < result.robot_covariances.size(); ++i)
    {
        gtsam::Symbol robot_sym = ConsensusGraph::RobotSymbol(i);
        covariances[static_cast<size_t>(robot_sym.key())] = result.robot_covariances[i];
    }

    // Object covariances (doors, etc.)
    for (const auto& [idx, cov] : result.object_covariances)
    {
        gtsam::Symbol obj_sym = ConsensusGraph::ObjectSymbol(idx);
        covariances[static_cast<size_t>(obj_sym.key())] = cov;
    }

    auto cov_copy = covariances;  // copy for queued delivery

    QMetaObject::invokeMethod(widget_,
        [w = widget_,
         init = result.initial_error,
         fin  = result.final_error,
         iters = result.iterations,
         g = &gtsam_graph,
         v = &values,
         cov = std::move(cov_copy)]() mutable
    {
        w->onOptimizationCompleted(init, fin, iters, g, v, cov);
    }, Qt::QueuedConnection);

}
