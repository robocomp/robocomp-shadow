/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "consensus_visualization_adapter.h"
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

    qDebug() << "ConsensusVisualizationAdapter: Connected manager to widget";

    // If manager is already initialized, trigger initial update
    if (manager_->isInitialized())
    {
        onManagerInitialized();
    }
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

    // Send to widget
    widget_->onGraphUpdated(event);
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

    // Extract covariances from result
    std::map<size_t, Eigen::Matrix3d> covariances;

    // Add robot covariances
    for (size_t i = 0; i < result.robot_covariances.size(); ++i)
    {
        covariances[i] = result.robot_covariances[i];
    }

    // Add object covariances
    for (const auto& [idx, cov] : result.object_covariances)
    {
        covariances[idx] = cov;
    }

    // Send optimization completed event
    widget_->onOptimizationCompleted(
        result.initial_error,
        result.final_error,
        result.iterations,
        &gtsam_graph,
        &values,
        covariances
    );
}
