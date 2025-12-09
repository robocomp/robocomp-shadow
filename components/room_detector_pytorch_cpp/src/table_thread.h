/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    TableThread: Thread wrapper for TableConcept with Qt signal/slot communication
 */

#ifndef TABLE_THREAD_H
#define TABLE_THREAD_H

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
#include <memory>
#include <optional>
#include <atomic>

#include "table_concept.h"
#include "table_model.h"
#include <Lidar3D.h>
#include <Camera360RGBD.h>

/**
 * @brief Thread wrapper for TableConcept
 *
 * Runs TableConcept in a separate thread, communicating with the main
 * thread via Qt signals and slots. This allows table detection and tracking
 * to run independently without blocking the main loop.
 *
 * Usage:
 *   1. Create TableThread
 *   2. Connect signals/slots
 *   3. Call start()
 *   4. Send data via slots (onNewRGBDData, onNewLidarData, onNewOdometry)
 *   5. Receive results via signals (tableUpdated, tableDetected, trackingLost)
 */
class TableThread : public QThread
{
    Q_OBJECT

public:
    /**
     * @brief Construct TableThread
     * @param camera_proxy Proxy for 360 RGBD camera (can be nullptr if using external data)
     */
    explicit TableThread(const RoboCompCamera360RGBD::Camera360RGBDPrxPtr& camera_proxy = nullptr,
                        QObject* parent = nullptr);
    ~TableThread() override;

    /**
     * @brief Check if table is being tracked
     */
    bool isTracking() const { return tracking_.load(); }

    /**
     * @brief Check if table has been detected
     */
    bool hasDetection() const { return has_detection_.load(); }

    /**
     * @brief Get current table model (thread-safe copy)
     */
    std::shared_ptr<TableModel> getModel();

    /**
     * @brief Get last result (thread-safe copy)
     */
    std::optional<rc::TableConcept::Result> getLastResult();

    /**
     * @brief Get table concept configuration (for tuning)
     */
    rc::TableConcept::OptimizationConfig& getConfig();

    /**
     * @brief Request thread to stop
     */
    void requestStop();

Q_SIGNALS:
    /**
     * @brief Emitted when a new table is detected
     */
    void tableDetected(std::shared_ptr<TableModel> model);

    /**
     * @brief Emitted after each successful tracking update
     */
    void tableUpdated(std::shared_ptr<TableModel> model, rc::TableConcept::Result result);

    /**
     * @brief Emitted when tracking is lost
     */
    void trackingLost();

    /**
     * @brief Emitted on errors
     */
    void errorOccurred(const QString& error);

public Q_SLOTS:

    /**
     * @brief Receive new RGBD data from main thread
     */
    void onNewRGBDData(const RoboCompCamera360RGBD::TRGBD& rgbd);

    /**
     * @brief Receive new LiDAR data from main thread (for full 3D filtering)
     */
    void onNewLidarData(const RoboCompLidar3D::TPoints& points);

    /**
     * @brief Receive odometry update from main thread
     */
    void onNewOdometry(const Eigen::Vector3f& motion);

    /**
     * @brief Set consensus prior from factor graph
     */
    void onConsensusPrior(const Eigen::Vector4f& pose, const Eigen::Matrix4f& covariance);

    /**
     * @brief Clear consensus prior
     */
    void onClearConsensusPrior();

    /**
     * @brief Force redetection
     */
    void onForceRedetection();

    /**
     * @brief Reset table tracking
     */
    void onReset();

protected:
    void run() override;

private:
    // Table concept instance (owned by this thread)
    std::unique_ptr<rc::TableConcept> table_concept_;
    RoboCompCamera360RGBD::Camera360RGBDPrxPtr camera_proxy_;

    // Thread-safe data exchange
    mutable QMutex data_mutex_;
    QWaitCondition data_condition_;

    // Input data (protected by data_mutex_)
    RoboCompCamera360RGBD::TRGBD pending_rgbd_;
    RoboCompLidar3D::TPoints pending_lidar_points_;
    Eigen::Vector3f pending_motion_ = Eigen::Vector3f::Zero();
    bool has_new_rgbd_ = false;
    bool has_new_lidar_ = false;
    bool has_new_motion_ = false;

    // Consensus prior (protected by data_mutex_)
    Eigen::Vector4f pending_prior_pose_;
    Eigen::Matrix4f pending_prior_covariance_;
    bool has_new_prior_ = false;
    bool clear_prior_requested_ = false;

    // Control flags (protected by data_mutex_)
    bool force_redetection_ = false;
    bool reset_requested_ = false;

    // State
    std::atomic<bool> tracking_{false};
    std::atomic<bool> has_detection_{false};
    std::atomic<bool> stop_requested_{false};

    // Last result (protected by data_mutex_)
    std::optional<rc::TableConcept::Result> last_result_;
    std::shared_ptr<TableModel> current_model_;
};

#endif // TABLE_THREAD_H
