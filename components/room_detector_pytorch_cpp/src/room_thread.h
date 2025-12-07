/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    RoomThread: Thread wrapper for RoomConcept with Qt signal/slot communication
 */

#ifndef ROOM_THREAD_H
#define ROOM_THREAD_H

#ifdef slots
  #undef slots
#endif
#include <torch/torch.h>
#ifdef Q_SLOTS
  #define slots Q_SLOTS
#endif
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QTimer>
#include <memory>
#include <optional>
#include <atomic>

#include "room_concept.h"
#include "room_model.h"
#include <Lidar3D.h>

/**
 * @brief Thread wrapper for RoomConcept
 *
 * Runs RoomConcept in a separate thread, communicating with the main
 * thread via Qt signals and slots. This allows the room optimization
 * to run independently without blocking the main loop.
 *
 * Usage:
 *   1. Create RoomThread
 *   2. Connect signals/slots
 *   3. Call start()
 *   4. Send data via slots (onNewLidarData, onNewOdometry)
 *   5. Receive results via signals (roomUpdated, roomInitialized)
 */
class RoomThread : public QThread
{
    Q_OBJECT

public:
    explicit RoomThread(QObject* parent = nullptr);
    ~RoomThread() override;

    /**
     * @brief Check if room model is initialized
     */
    bool isInitialized() const { return initialized_.load(); }

    /**
     * @brief Get current room model (thread-safe copy)
     */
    std::shared_ptr<RoomModel> getModel();

    /**
     * @brief Get last result (thread-safe copy)
     */
    std::optional<rc::RoomConcept::Result> getLastResult();

    /**
     * @brief Request thread to stop
     */
    void requestStop();

Q_SIGNALS:
    /**
     * @brief Emitted when room model is first initialized
     * @param room_params [half_width, half_depth] extracted in thread (thread-safe)
     */
    void roomInitialized(std::shared_ptr<RoomModel> model, std::vector<float> room_params);

    /**
     * @brief Emitted after each successful optimization update
     * @param room_params [half_width, half_depth] extracted in thread (thread-safe)
     */
    void roomUpdated(std::shared_ptr<RoomModel> model, rc::RoomConcept::Result result, std::vector<float> room_params);

    /**
     * @brief Emitted when room state changes (MAPPING <-> LOCALIZED)
     */
    void stateChanged(RoomState new_state);

    /**
     * @brief Emitted on errors
     */
    void errorOccurred(const QString& error);

public Q_SLOTS:
    /**
     * @brief Receive new LiDAR data from main thread
     */
    void onNewLidarData(const TimePoints& points,
                        const VelocityHistory &velocity_history_);

    /**
     * @brief Force room remapping
     */
    void onRemapRequested();

protected:
    void run() override;

private:
    // Room concept instance (owned by this thread)
    std::unique_ptr<rc::RoomConcept> room_concept_;
    std::shared_ptr<RoomModel> room_model_;

    // Thread-safe data exchange
    mutable QMutex data_mutex_;
    QWaitCondition data_condition_;

    // Input data (protected by data_mutex_)
    TimePoints pending_lidar_points_;
    rc::RoomConcept::OdometryPrior pending_odometry_;
    VelocityHistory velocity_history_;
    bool has_new_lidar_data_ = false;
    bool has_new_odometry_ = false;
    bool remap_requested_ = false;

    // State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> stop_requested_{false};

    // Last result (protected by data_mutex_)
    std::optional<rc::RoomConcept::Result> last_result_;
    RoomState last_state_ = RoomState::MAPPING;

};

#endif // ROOM_THREAD_H