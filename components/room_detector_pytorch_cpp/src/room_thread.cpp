/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "room_thread.h"
#include <QDebug>

RoomThread::RoomThread(QObject* parent)
    : QThread(parent)
{
    // Create room model (shared between concept and external users)
    room_model_ = std::make_shared<RoomModel>();
}

RoomThread::~RoomThread()
{
    requestStop();
    if (isRunning())
    {
        wait(5000);  // Wait up to 5 seconds
        if (isRunning())
        {
            qWarning() << "RoomThread: Force terminating thread";
            terminate();
            wait();
        }
    }
}

void RoomThread::requestStop()
{
    stop_requested_.store(true);
    data_condition_.wakeAll();
}

std::shared_ptr<RoomModel> RoomThread::getModel()
{
    QMutexLocker lock(&data_mutex_);
    return room_model_;
}

std::optional<rc::RoomConcept::Result> RoomThread::getLastResult()
{
    QMutexLocker lock(&data_mutex_);
    return last_result_;
}

void RoomThread::onNewLidarData(const TimePoints& points,
                                const VelocityHistory &velocity_history)
{
    QMutexLocker lock(&data_mutex_);
    pending_lidar_points_ = points;
    velocity_history_ = velocity_history;
    has_new_lidar_data_ = true;
    data_condition_.wakeOne();
}

void RoomThread::onRemapRequested()
{
    QMutexLocker lock(&data_mutex_);
    remap_requested_ = true;
    data_condition_.wakeOne();
}

void RoomThread::run()
{
    qInfo() << "RoomThread: Starting";

    // Create room concept in this thread (important for thread affinity)
    room_concept_ = std::make_unique<rc::RoomConcept>();

    while (!stop_requested_.load())
    {
        // Wait for new data
        TimePoints lidar_points;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
        rc::RoomConcept::OdometryPrior odometry;
        bool has_lidar = false;
        bool do_remap = false;

        {
            QMutexLocker lock(&data_mutex_);

            // Wait until we have data or stop is requested
            while (!has_new_lidar_data_ && !stop_requested_.load())
            {
                data_condition_.wait(&data_mutex_, 100);  // 100ms timeout for responsiveness
            }

            if (stop_requested_.load())
                break;

            // Copy data
            if (has_new_lidar_data_)
            {
                lidar_points = std::move(pending_lidar_points_);
                has_lidar = true;
                has_new_lidar_data_ = false;
            }

            if (remap_requested_)
            {
                do_remap = true;
                remap_requested_ = false;
            }
        }

        // Handle remap request
        if (do_remap)
        {
            qInfo() << "RoomThread: Remap requested";
            //room_concept_->request_remap();
        }

        // Process data
        if (has_lidar)
        {
            try
            {
                // Run update cycle
                auto result = room_concept_->update(lidar_points, velocity_history_);

                // Extract room parameters while in thread (thread-safe access)
                std::vector<float> room_params{0.0f, 0.0f};
                if (room_concept_->is_initialized())
                {
                    room_params = room_concept_->get_room_model()->get_room_parameters();
                }

                // Check initialization
                const bool was_initialized = initialized_.load();
                const bool now_initialized = room_concept_->is_initialized();

                if (not was_initialized and now_initialized)
                {
                    initialized_.store(true);
                    Q_EMIT roomInitialized(room_model_, room_params);
                    qInfo() << "RoomThread: Room initialized";
                }

                // Store result and Q_EMIT update
                {
                    QMutexLocker lock(&data_mutex_);
                    last_result_ = result;
                }

                // Check state change
                if (auto current_state = room_concept_->get_state(); current_state != last_state_)
                {
                    last_state_ = current_state;
                    Q_EMIT stateChanged(current_state);
                }

                // Clone tensors for thread-safe transfer
                // Tensors passed across Qt signal/slot boundaries must be cloned
                rc::RoomConcept::Result result_copy = result;
                if (result.covariance.defined())
                    result_copy.covariance = result.covariance.clone().contiguous();
                if (result.propagated_cov.defined())
                    result_copy.propagated_cov = result.propagated_cov.clone().contiguous();
                if (result.prior.covariance.defined())
                    result_copy.prior.covariance = result.prior.covariance.clone().contiguous();
                if (result.prediction_state.propagated_cov.defined())
                    result_copy.prediction_state.propagated_cov = result.prediction_state.propagated_cov.clone().contiguous();

                // Q_EMIT update signal with cloned tensors
                Q_EMIT roomUpdated(room_model_, result_copy, room_params);
            }
            catch (const std::exception& e)
            {
                qWarning() << "RoomThread: Exception during update:" << e.what();
                Q_EMIT errorOccurred(QString("Update error: %1").arg(e.what()));
            }
        }
    }

    qInfo() << "RoomThread: Stopping";
}