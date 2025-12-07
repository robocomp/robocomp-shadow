/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    DoorThread: Thread wrapper for DoorConcept with Qt signal/slot communication
 */

#ifndef DOOR_THREAD_H
#define DOOR_THREAD_H

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

#include "door_concept.h"
#include "door_model.h"
#include <Lidar3D.h>
#include <Camera360RGBD.h>

/**
 * @brief Thread wrapper for DoorConcept
 *
 * Runs DoorConcept in a separate thread, communicating with the main
 * thread via Qt signals and slots. This allows door detection and tracking
 * to run independently without blocking the main loop.
 *
 * Usage:
 *   1. Create DoorThread
 *   2. Connect signals/slots
 *   3. Call start()
 *   4. Send data via slots (onNewRGBDData, onNewLidarData, onNewOdometry)
 *   5. Receive results via signals (doorUpdated, doorDetected, trackingLost)
 */
class DoorThread : public QThread
{
    Q_OBJECT

    public:
        /**
         * @brief Construct DoorThread
         * @param camera_proxy Proxy for 360 RGBD camera (can be nullptr if using external data)
         */
        explicit DoorThread(const RoboCompCamera360RGBD::Camera360RGBDPrxPtr& camera_proxy = nullptr,
                            QObject* parent = nullptr);
        ~DoorThread() override;

        /**
         * @brief Check if door is being tracked
         */
        bool isTracking() const { return tracking_.load(); }

        /**
         * @brief Check if door has been detected
         */
        bool hasDetection() const { return has_detection_.load(); }

        /**
         * @brief Get current door model (thread-safe copy)
         */
        std::shared_ptr<DoorModel> getModel();

        /**
         * @brief Get last result (thread-safe copy)
         */
        std::optional<rc::DoorConcept::Result> getLastResult();

        /**
         * @brief Get door concept configuration (for tuning)
         */
        rc::DoorConcept::OptimizationConfig& getConfig();

        /**
         * @brief Request thread to stop
         */
        void requestStop();

        Q_SIGNALS:
            /**
             * @brief Emitted when a new door is detected
             */
            void doorDetected(std::shared_ptr<DoorModel> model);

            /**
             * @brief Emitted after each successful tracking update
             */
            void doorUpdated(std::shared_ptr<DoorModel> model, rc::DoorConcept::Result result);

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
        void onConsensusPrior(const Eigen::Vector3f& pose, const Eigen::Matrix3f& covariance);

        /**
         * @brief Clear consensus prior
         */
        void onClearConsensusPrior();

        /**
         * @brief Force redetection
         */
        void onForceRedetection();

        /**
         * @brief Reset door tracking
         */
        void onReset();

    protected:
        void run() override;

    private:
        // Door concept instance (owned by this thread)
        std::unique_ptr<rc::DoorConcept> door_concept_;
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
        Eigen::Vector3f pending_prior_pose_;
        Eigen::Matrix3f pending_prior_covariance_;
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
        std::optional<rc::DoorConcept::Result> last_result_;
        std::shared_ptr<DoorModel> current_model_;
};

#endif // DOOR_THREAD_H