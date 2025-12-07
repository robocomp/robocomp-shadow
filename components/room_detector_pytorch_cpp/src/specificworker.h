/*
 *    Copyright (C) 2025 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#ifdef slots
  #undef slots
#endif
#include <torch/torch.h>
#ifdef Q_SLOTS
  #define slots Q_SLOTS
#endif

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include "time_series_plotter.h"
#include <genericworker.h>
#include <expected>
#include <vector>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <Eigen/Dense>
#include "door_detector.h"
#include "room_freezing_manager.h"
#include "qt3d_visualizer.h"
#include "door_concept.h"
#include "yolo_detector_onnx.h"
#include "room_concept.h"
#include "consensus_manager.h"
#include "room_thread.h"
#include "door_thread.h"
#include "consensus_graph_widget.h"
#include "consensus_visualization_adapter.h"

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker final : public GenericWorker
{
	Q_OBJECT
	public:
	    /**
	     * \brief Constructor for SpecificWorker.
	     * \param configLoader Configuration loader for the component.
	     * \param tprx Tuple of proxies required for the component.
	     * \param startup_check Indicates whether to perform startup checks.
	     */
		SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
		void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);

		/**
	     * \brief Destructor for SpecificWorker.
	     */
		~SpecificWorker();

	public Q_SLOTS:

		void initialize();
		void compute();
		void emergency();
		void restore();
		int startup_check();

	private:
		bool startup_check_flag;

		struct Params
		{
			float ROBOT_WIDTH = 460;  // mm
			float ROBOT_LENGTH = 480;  // mm
			float MAX_ADV_SPEED = 1000; // mm/s
			float MAX_ROT_SPEED = 1; // rad/s
			float MAX_SIDE_SPEED = 50; // mm/s
			float MAX_TRANSLATION = 500; // mm/s
			float MAX_ROTATION = 0.2;
			float STOP_THRESHOLD = 700; // mm
			std::string LIDAR_NAME_LOW = "bpearl";
			std::string LIDAR_NAME_HIGH = "helios";
			QRectF GRID_MAX_DIM{-5000, 2500, 10000, -5000};
		};
		Params params;

		// velocity commands
		boost::circular_buffer<VelocityCommand> velocity_history_{10}; // Keep last 10 commands
		rc::RoomConcept::OdometryPrior compute_odometry_prior(
					std::chrono::time_point<std::chrono::high_resolution_clock> t_start,
					std::chrono::time_point<std::chrono::high_resolution_clock> t_end) const;

		Eigen::Vector3f integrate_velocity_over_window(
					std::chrono::time_point<std::chrono::high_resolution_clock> t_start,
					std::chrono::time_point<std::chrono::high_resolution_clock> t_end) const;

		// viewer
		AbstractGraphicViewer *viewer, *viewer_room;
		QGraphicsPolygonItem *robot_draw, *robot_room_draw;
		QGraphicsItem *room_draw = nullptr;

		// aux
		TimePoints read_data();
		RoboCompLidar3D::TPoints filter_isolated_points_torch(const RoboCompLidar3D::TPoints &points, float d);
		RoboCompCamera360RGBD::TRGBD read_image();
		void draw_lidar(const RoboCompLidar3D::TPoints &filtered_points, QGraphicsScene *scene);
		void update_robot_view(const Eigen::Affine2f &robot_pose, const rc::RoomConcept::Result &result, QGraphicsScene *scene);
		// Helper to update GUI widgets (extracted from update_robot_view)
		void update_gui(const Eigen::Affine2f &robot_pose, const rc::RoomConcept::Result &result);
		void update_viewers(const TimePoints &points_,
		                    const RoboCompCamera360RGBD::TRGBD &rgbd,
		                    const rc::RoomConcept::Result &room_result, const std::optional<rc::DoorConcept::Result> &door_result, QGraphicsScene *
		                    robot_scene, double elapsed);

		QGraphicsEllipseItem* draw_uncertainty_ellipse(
				QGraphicsScene *scene,
				const torch::Tensor &covariance,
				const std::vector<float> &robot_pose,
				const QColor &color,
				float scale_factor);  // 2-sigma = 95% confidence

		inline QPointF to_qpointf(const Eigen::Vector2f &v) const
        { return QPointF(v.x(), v.y()); }
		void print_status(
			const rc::RoomConcept::Result &result);

		// random number generator
		std::random_device rd;

		// timing
		std::chrono::time_point<std::chrono::high_resolution_clock> last_time = std::chrono::high_resolution_clock::now();

		// plotter
		std::shared_ptr<TimeSeriesPlotter> room_loss_plotter, stddev_plotter, epoch_plotter, door_loss_plotter;
		std::vector<int> graphs;

		// qt3d
		std::unique_ptr<RoomVisualizer3D> viewer3d;

		// Yolo detector
		std::unique_ptr<YOLODetectorONNX> yolo_detector;

		// room concept
		std::unique_ptr<rc::RoomConcept> room_concept;

		// door concept
		DoorDetector door_detector;
		std::unique_ptr<rc::DoorConcept> door_concept;

		void update_visualization(const std::chrono::high_resolution_clock::time_point &last_time);

		// Threaded detectors
		std::unique_ptr<RoomThread> room_thread_;
		std::unique_ptr<DoorThread> door_thread_;

		// Consensus manager (QObject, runs in main thread)
		ConsensusManager consensus_manager_;
		std::unique_ptr<ConsensusGraphWidget> consensus_graph_widget;
		std::unique_ptr<ConsensusVisualizationAdapter> viz_adapter;

		// Thread-safe caches of latest results (for visualization)
		mutable QMutex results_mutex_;
		std::shared_ptr<RoomModel> latest_room_model_;
		std::optional<rc::RoomConcept::Result> latest_room_result_;
		std::shared_ptr<DoorModel> latest_door_model_;
		std::optional<rc::DoorConcept::Result> latest_door_result_;

		// Cached room parameters (thread-safe, extracted in RoomThread)
		std::vector<float> cached_room_params_{0.0f, 0.0f};  // [half_width, half_depth]

		// Cached RGBD for door visualization
		RoboCompCamera360RGBD::TRGBD cached_rgbd_;

		// Cached consensus door pose (in room coordinates)
		struct ConsensusDoorPose {
			bool valid = false;
			float x = 0, y = 0, z = 0, theta = 0;
			float width = 0, height = 0, opening_angle = 0;
		};
		ConsensusDoorPose cached_consensus_door_;

	Q_SIGNALS:
		// Signals to send data to threads
		void newLidarData(const TimePoints& points, const VelocityHistory &velocity_history);
		void newRGBDData(const RoboCompCamera360RGBD::TRGBD& rgbd);
		void newRoomOdometry(const rc::RoomConcept::OdometryPrior& odometry);
		void newDoorOdometry(const Eigen::Vector3f& motion);

		// Signals to consensus manager
		void roomDataReady(const std::shared_ptr<RoomModel>& model,
		                   const Eigen::Vector3f& robot_pose,
		                   const Eigen::Matrix3f& robot_covariance,
		                   const std::vector<float>& room_params);
		void doorDetectionReady(const std::shared_ptr<DoorModel>& model,
		                        const Eigen::Matrix3f& detection_covariance);
		void doorUpdateReady(const Eigen::Vector3f& door_pose,
		                     const Eigen::Matrix3f& door_covariance);

	private Q_SLOTS:
		// Slots to receive results from threads
		void onRoomInitialized(std::shared_ptr<RoomModel> model, std::vector<float> room_params);
		void onRoomUpdated(std::shared_ptr<RoomModel> model, rc::RoomConcept::Result result, std::vector<float> room_params);
		void onRoomStateChanged(RoomState new_state);

		void onDoorDetected(std::shared_ptr<DoorModel> model);
		void onDoorUpdated(std::shared_ptr<DoorModel> model, rc::DoorConcept::Result result);
		void onDoorTrackingLost();

		// Slot for consensus door pose (in room coordinates)
		void onConsensusDoorPose(size_t door_index, float x, float y, float z, float theta,
		                         float width, float height, float opening_angle);
};

#endif