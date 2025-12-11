/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "table_thread.h"
#include <QDebug>

TableThread::TableThread(const RoboCompCamera360RGBD::Camera360RGBDPrxPtr& camera_proxy,
                       QObject* parent)
    : QThread(parent)
    , camera_proxy_(camera_proxy)
{
}

TableThread::~TableThread()
{
    requestStop();
    if (isRunning())
    {
        wait(5000);  // Wait up to 5 seconds
        if (isRunning())
        {
            qWarning() << "TableThread: Force terminating thread";
            terminate();
            wait();
        }
    }
}

void TableThread::requestStop()
{
    stop_requested_.store(true);
    data_condition_.wakeAll();
}

std::shared_ptr<TableModel> TableThread::getModel()
{
    QMutexLocker lock(&data_mutex_);
    return current_model_;
}

std::optional<rc::TableConcept::Result> TableThread::getLastResult()
{
    QMutexLocker lock(&data_mutex_);
    return last_result_;
}

rc::TableConcept::OptimizationConfig& TableThread::getConfig()
{
    // Note: This should only be called before thread starts or with proper synchronization
    return table_concept_->get_config();
}

void TableThread::onNewRGBDData(const RoboCompCamera360RGBD::TRGBD& rgbd)
{
    QMutexLocker lock(&data_mutex_);
    pending_rgbd_ = rgbd;
    has_new_rgbd_ = true;
    data_condition_.wakeOne();
}

void TableThread::onNewLidarData(const RoboCompLidar3D::TPoints& points)
{
    QMutexLocker lock(&data_mutex_);
    pending_lidar_points_ = points;
    has_new_lidar_ = true;
    data_condition_.wakeOne();
}

void TableThread::onNewOdometry(const Eigen::Vector3f& motion)
{
    QMutexLocker lock(&data_mutex_);
    pending_motion_ += motion;
    has_new_motion_ = true;
    data_condition_.wakeOne();
}

void TableThread::onConsensusPrior(const Eigen::Vector3f& pose, const Eigen::Matrix3f& covariance)
{
    QMutexLocker lock(&data_mutex_);
    pending_prior_pose_ = pose;
    pending_prior_covariance_ = covariance;
    has_new_prior_ = true;
    data_condition_.wakeOne();
}

void TableThread::onClearConsensusPrior()
{
    QMutexLocker lock(&data_mutex_);
    clear_prior_requested_ = true;
    data_condition_.wakeOne();
}

void TableThread::onForceRedetection()
{
    QMutexLocker lock(&data_mutex_);
    force_redetection_ = true;
    data_condition_.wakeOne();
}

void TableThread::onReset()
{
    QMutexLocker lock(&data_mutex_);
    reset_requested_ = true;
    data_condition_.wakeOne();
}

void TableThread::run()
{
    qInfo() << "TableThread: Starting";

    // Create table concept in this thread (important for thread affinity)
    table_concept_ = std::make_unique<rc::TableConcept>(camera_proxy_);

    while (!stop_requested_.load())
    {
        // Local copies of data
        RoboCompCamera360RGBD::TRGBD rgbd;
        RoboCompLidar3D::TPoints lidar_points;
        Eigen::Vector3f motion = Eigen::Vector3f::Zero();
        bool has_rgbd = false;
        bool has_lidar = false;

        // Consensus prior
        Eigen::Vector3f prior_pose;
        Eigen::Matrix3f prior_covariance;
        bool apply_prior = false;
        bool clear_prior = false;

        // Control flags
        bool do_redetection = false;
        bool do_reset = false;

        {
            QMutexLocker lock(&data_mutex_);

            // Wait until we have RGBD data or stop is requested
            while (!has_new_rgbd_ && !stop_requested_.load())
            {
                data_condition_.wait(&data_mutex_, 100);  // 100ms timeout
            }

            if (stop_requested_.load())
                break;

            // Copy data
            if (has_new_rgbd_)
            {
                rgbd = std::move(pending_rgbd_);
                has_rgbd = true;
                has_new_rgbd_ = false;
            }

            if (has_new_lidar_)
            {
                lidar_points = std::move(pending_lidar_points_);
                has_lidar = true;
                has_new_lidar_ = false;
            }

            if (has_new_motion_)
            {
                motion = pending_motion_;
                pending_motion_.setZero();
                has_new_motion_ = false;
            }

            // Consensus prior
            if (has_new_prior_)
            {
                prior_pose = pending_prior_pose_;
                prior_covariance = pending_prior_covariance_;
                apply_prior = true;
                has_new_prior_ = false;
            }

            if (clear_prior_requested_)
            {
                clear_prior = true;
                clear_prior_requested_ = false;
            }

            // Control flags
            if (force_redetection_)
            {
                do_redetection = true;
                force_redetection_ = false;
            }

            if (reset_requested_)
            {
                do_reset = true;
                reset_requested_ = false;
            }
        }

        // Handle reset
        if (do_reset)
        {
            qInfo() << "TableThread: Reset requested";
            table_concept_->reset();
            tracking_.store(false);
            has_detection_.store(false);
            {
                QMutexLocker lock(&data_mutex_);
                current_model_ = nullptr;
                last_result_.reset();
            }
            Q_EMIT trackingLost();
            continue;
        }

        // Handle force redetection
        if (do_redetection)
        {
            qInfo() << "TableThread: Force redetection requested";
            table_concept_->force_redetection();
        }

        // Apply or clear consensus prior
        if (clear_prior)
        {
            table_concept_->clearConsensusPrior();
        }
        if (apply_prior)
        {
            table_concept_->setConsensusPrior(prior_pose, prior_covariance);
        }

        // Process RGBD data
        if (has_rgbd)
        {
            try
            {
                // Run update cycle
                auto result_opt = table_concept_->update(rgbd, motion);

                bool was_tracking = tracking_.load();
                bool now_tracking = table_concept_->is_tracking();

                if (result_opt.has_value())
                {
                    auto& result = result_opt.value();

                    // Store result
                    {
                        QMutexLocker lock(&data_mutex_);
                        last_result_ = result;
                        current_model_ = result.table;
                    }

                    // Check for new detection
                    if (!was_tracking && now_tracking && result.table)
                    {
                        has_detection_.store(true);
                        tracking_.store(true);
                        Q_EMIT doorDetected(result.table);
                        qInfo() << "TableThread: Table detected";
                    }

                    //Q_EMIT update
                    if (result.success && result.table)
                    {
                        tracking_.store(true);

                        // Clone tensors for thread-safe transfer
                        rc::TableConcept::Result result_copy = result;
                        if (result.covariance.defined())
                            result_copy.covariance = result.covariance.clone().contiguous();

                        Q_EMIT doorUpdated(result.table, result_copy);
                    }
                }
                else
                {
                    // No result - check if tracking lost
                    if (was_tracking && !now_tracking)
                    {
                        tracking_.store(false);
                       Q_EMIT trackingLost();
                        qInfo() << "TableThread: Tracking lost";
                    }
                }
            }
            catch (const std::exception& e)
            {
                qWarning() << "TableThread: Exception during update:" << e.what();
               Q_EMIT errorOccurred(QString("Update error: %1").arg(e.what()));
            }
        }
    }

    qInfo() << "TableThread: Stopping";
}