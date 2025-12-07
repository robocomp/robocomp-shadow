/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    Adapter to connect your existing ConsensusManager to ConsensusGraphWidget
 */

#ifndef CONSENSUS_VISUALIZATION_ADAPTER_H
#define CONSENSUS_VISUALIZATION_ADAPTER_H

#include <QObject>
#include "consensus_manager.h"
#include "consensus_graph_widget.h"

/**
 * @brief Connects your existing ConsensusManager to ConsensusGraphWidget
 * 
 * This adapter listens to ConsensusManager signals and translates them
 * into the GraphUpdateEvent format that the widget expects.
 * 
 * Usage:
 *   ConsensusManager* manager = new ConsensusManager(this);
 *   ConsensusGraphWidget* widget = new ConsensusGraphWidget(this);
 *   
 *   // Create adapter and it handles all connections
 *   new ConsensusVisualizationAdapter(manager, widget, this);
 *   
 *   // Now manager->runOptimization() automatically updates the widget!
 */
class ConsensusVisualizationAdapter : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Create adapter and automatically connect signals
     * 
     * @param manager Your existing ConsensusManager instance
     * @param widget The visualization widget
     * @param parent QObject parent
     */
    explicit ConsensusVisualizationAdapter(ConsensusManager* manager,
                                          ConsensusGraphWidget* widget,
                                          QObject* parent = nullptr);

private Q_SLOTS:
    /**
     * @brief Handle initialization signal from ConsensusManager
     */
    void onManagerInitialized();

    /**
     * @brief Handle consensus ready signal from ConsensusManager
     */
    void onConsensusReady(const ConsensusResult& result);

private:
    ConsensusManager* manager_;
    ConsensusGraphWidget* widget_;
};

#endif // CONSENSUS_VISUALIZATION_ADAPTER_H
