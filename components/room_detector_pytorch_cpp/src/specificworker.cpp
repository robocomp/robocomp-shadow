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

// Undefine Qt macros BEFORE including PyTorch
#ifdef slots
#undef slots
#endif
#ifdef signals
#undef signals
#endif
#ifdef emit
#undef emit
#endif
#include <execution>
#include <torch/torch.h>
#include "room_model.h"
#include "specificworker.h"
#include <iostream>
#include <iomanip>
#include "common_types.h"
#include <cppitertools/enumerate.hpp>
#include <cppitertools/groupby.hpp>
#include "door_projection.h"

SpecificWorker::SpecificWorker(const ConfigLoader &configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(
	configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if (this->startup_check_flag)
	{
		this->startup_check();
	} else
	{
#ifdef HIBERNATION_ENABLED
		hibernationChecker.start(500);
#endif

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0)
		{
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	qInfo() << "Stopping detector threads...";

	if (room_thread_)
	{
		room_thread_->requestStop();
		room_thread_->wait(3000);
	}

	if (door_thread_)
	{
		door_thread_->requestStop();
		door_thread_->wait(3000);
	}

	qInfo() << "Detector threads stopped";
}


void SpecificWorker::initialize()
{
	std::cout << "initialize worker" << std::endl;

	std::cout << "PyTorch C++ Library Test\n";
	std::cout << "========================\n\n";

	// Check if CUDA is available
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Training on GPU.\n";
	} else
	{
		std::cout << "Training on CPU.\n";
	}

	// Viewer
	viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
	auto [r, e] = viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
	robot_draw = r;

	// viewer_room = new AbstractGraphicViewer(this->frame_room, params.GRID_MAX_DIM);
	// auto [rr, re] = viewer_room->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
	// robot_room_draw = rr;
	this->resize(1500, 800);
	show();

	// Factor graph widget
	consensus_graph_widget = std::make_unique<ConsensusGraphWidget>(this->frame_graph);
	QVBoxLayout* layout_graph = new QVBoxLayout(this->frame_graph);
	layout_graph->setContentsMargins(0, 0, 0, 0);  // No margins
	layout_graph->addWidget(consensus_graph_widget.get());
	this->frame_graph->setLayout(layout_graph);
	viz_adapter = std::make_unique<ConsensusVisualizationAdapter>(&consensus_manager_, consensus_graph_widget.get(), this->frame_graph);
	consensus_graph_widget->show();

	// Initialize RoomModel
	const auto &[points, lidar_time] = read_data();
	if (points.empty())
	{
		std::cout << __FUNCTION__ << " No LiDAR points available\n";
		return;
	}
	draw_lidar(points, &viewer->scene);

	// Room concept
	room_concept = std::make_unique<rc::RoomConcept>();

	// Door concept
	door_concept = std::make_unique<rc::DoorConcept>(camera360rgbd_proxy);

	// loss plotter for match error
	TimeSeriesPlotter::Config plotConfig;

	plotConfig.title = "Loss (likelihood + prior)";
	plotConfig.yAxisLabel = "Error";
	plotConfig.xAxisLabel = "";
	plotConfig.timeWindowSeconds = 7.0; // Show a 15-second window
	plotConfig.autoScaleY = true; // We will set a fixed range
	plotConfig.showLegend = true; // Show graph legend

	room_loss_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error_1, plotConfig);
	graphs.push_back(room_loss_plotter->addGraph("post", Qt::blue));
	graphs.push_back(room_loss_plotter->addGraph("like", Qt::green));
	graphs.push_back(room_loss_plotter->addGraph("prior", Qt::red));

	plotConfig.title = "robot pose std";
	plotConfig.yAxisLabel = "std";
	stddev_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error_2, plotConfig);
	stddev_plotter->addGraph("upd_x", Qt::blue);
	stddev_plotter->addGraph("upd_y", Qt::cyan);
	stddev_plotter->addGraph("upd_theta", Qt::red);

	plotConfig.title = "door loss";
	plotConfig.yAxisLabel = "loss";
	door_loss_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error_3, plotConfig);
	door_loss_plotter->addGraph("post", Qt::blue);

	plotConfig.title = "epoch time";
	plotConfig.yAxisLabel = "time ms";
	epoch_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error_4, plotConfig);
	epoch_plotter->addGraph("elapsed", Qt::blue);

	// Create 3D viewer
	viewer3d = std::make_unique<RoomVisualizer3D>("src/meshes/shadow.obj");
	QWidget *viewer3d_widget = viewer3d->getWidget();
	QVBoxLayout *layout = new QVBoxLayout(frame_3d); // Your QFrame name here
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(viewer3d_widget);
	viewer3d->show();

	/////////////////////////////////////////////////////////
	qInfo() << "Initializing detector threads...";

    // Create room thread
    room_thread_ = std::make_unique<RoomThread>();

    // Create door thread (with camera proxy if needed)
    door_thread_ = std::make_unique<DoorThread>(camera360rgbd_proxy);

    // === Connect signals FROM main thread TO room thread ===
    connect(this, &SpecificWorker::newLidarData,
            room_thread_.get(), &RoomThread::onNewLidarData,
            Qt::QueuedConnection);

    // === Connect signals FROM room thread TO main thread ===
    connect(room_thread_.get(), &RoomThread::roomInitialized,
            this, &SpecificWorker::onRoomInitialized,
            Qt::QueuedConnection);

    connect(room_thread_.get(), &RoomThread::roomUpdated,
            this, &SpecificWorker::onRoomUpdated,
            Qt::QueuedConnection);

    connect(room_thread_.get(), &RoomThread::stateChanged,
            this, &SpecificWorker::onRoomStateChanged,
            Qt::QueuedConnection);

    // === Connect signals FROM main thread TO door thread ===
    connect(this, &SpecificWorker::newRGBDData,
            door_thread_.get(), &DoorThread::onNewRGBDData,
            Qt::QueuedConnection);

    // === Connect signals FROM door thread TO main thread ===
    connect(door_thread_.get(), &DoorThread::doorDetected,
            this, &SpecificWorker::onDoorDetected,
            Qt::QueuedConnection);

    connect(door_thread_.get(), &DoorThread::doorUpdated,
            this, &SpecificWorker::onDoorUpdated,
            Qt::QueuedConnection);

    connect(door_thread_.get(), &DoorThread::trackingLost,
            this, &SpecificWorker::onDoorTrackingLost,
            Qt::QueuedConnection);

    // === Connect signals TO consensus manager ===
    connect(this, &SpecificWorker::roomDataReady,
            &consensus_manager_, &ConsensusManager::onRoomUpdated,
            Qt::DirectConnection);  // Same thread, direct is fine

    connect(this, &SpecificWorker::doorDetectionReady,
            &consensus_manager_, &ConsensusManager::onDoorDetected,
            Qt::DirectConnection);

    connect(this, &SpecificWorker::doorUpdateReady,
            &consensus_manager_, &ConsensusManager::onDoorUpdated,
            Qt::DirectConnection);

    // === Connect consensus manager output TO door thread ===
    // Transform consensus prior from room frame to robot frame before sending
    connect(&consensus_manager_, &ConsensusManager::doorPriorReady,
            this, [this](size_t door_index, const Eigen::Vector3f& pose_room, const Eigen::Matrix3f& cov) {
                // Forward to door thread (index 0 for now, single door)
                if (door_index == 0 && door_thread_)
                {
                    // Get current robot pose in room frame
                    Eigen::Vector3f robot_pose_room;
                    {
                        QMutexLocker lock(&results_mutex_);
                        if (!latest_room_result_.has_value() || latest_room_result_->optimized_pose.size() < 3)
                        {
                            qWarning() << "Cannot transform consensus prior: no robot pose available";
                            return;
                        }
                        const auto& pose = latest_room_result_->optimized_pose;
                        robot_pose_room = Eigen::Vector3f(pose[0], pose[1], pose[2]);
                    }

                    // Transform door pose from room frame to robot frame
                    // door_robot = robot_room⁻¹ * door_room
                    float cos_r = std::cos(robot_pose_room.z());
                    float sin_r = std::sin(robot_pose_room.z());

                    // Translation: p_robot = R(-theta) * (p_room - t_robot)
                    float dx = pose_room.x() - robot_pose_room.x();
                    float dy = pose_room.y() - robot_pose_room.y();

                    Eigen::Vector3f pose_robot;
                    pose_robot.x() = cos_r * dx + sin_r * dy;
                    pose_robot.y() = -sin_r * dx + cos_r * dy;
                    pose_robot.z() = pose_room.z() - robot_pose_room.z();

                    // Normalize theta to [-π, π]
                    while (pose_robot.z() > M_PI) pose_robot.z() -= 2 * M_PI;
                    while (pose_robot.z() < -M_PI) pose_robot.z() += 2 * M_PI;

                    // Transform covariance: Cov_robot = R * Cov_room * R^T
                    // For simplicity, we rotate the position covariance but keep theta covariance
                    Eigen::Matrix2f R;
                    R << cos_r, sin_r,
                        -sin_r, cos_r;

                    Eigen::Matrix3f cov_robot = Eigen::Matrix3f::Zero();
                    cov_robot.block<2,2>(0,0) = R * cov.block<2,2>(0,0) * R.transpose();
                    cov_robot(2,2) = cov(2,2);  // theta variance unchanged

                    door_thread_->onConsensusPrior(pose_robot, cov_robot);
                }
            }, Qt::DirectConnection);

    // === Connect consensus door pose for visualization ===
    connect(&consensus_manager_, &ConsensusManager::doorPoseInRoom,
            this, &SpecificWorker::onConsensusDoorPose,
            Qt::DirectConnection);

    // Start threads
    room_thread_->start();
    door_thread_->start();

    qInfo() << "Detector threads started";

}

void SpecificWorker::compute()
{
	// Read LiDAR data (in robot frame)
	const auto time_points = read_data();
	if (std::get<0>(time_points).empty()) {	std::cout << "No LiDAR points available\n";	return;	}

	// === Draw LiDAR points in 2D view (every frame) ===
	const auto& [points, lidar_timestamp] = time_points;
	draw_lidar(points, &viewer->scene);

	// === Emit LiDAR data to room thread (non-blocking) ===
	Q_EMIT newLidarData(time_points, velocity_history_);

	// === Read and cache RGBD for door detection and visualization ===
	const auto rgbd = read_image();
	if (!rgbd.rgb.empty())
	{
		{
			QMutexLocker lock(&results_mutex_);
			cached_rgbd_ = rgbd;
		}
		// Emit to door thread
		Q_EMIT newRGBDData(rgbd);
	}

	// === Update visualization from cached results ===
	update_visualization(last_time);
	// Update visualization
	//consensus_graph_widget->updateFromGraph(consensus_manager_.getGraph());
	// consensus_graph_widget->setOptimizationInfo(result.initial_error,
	// 							   			   result.final_error,
	// 							               result.iterations);

	last_time = std::chrono::high_resolution_clock::now();
}

///////////////////////////////////////////////////////////////////////////////
// OLD non-threaded version - commented out
// void SpecificWorker::run_consensus( const rc::RoomConcept::Result &room_result,
// 									const std::optional<rc::DoorConcept::Result> &door_result)
// {
// 	// Consensus manager
// 	if (not consensus_manager.isInitialized() and room_result.uncertainty_valid)
// 	{
// 		Eigen::Vector3f pose = {room_result.optimized_pose[0], room_result.optimized_pose[1], room_result.optimized_pose[2]};
// 		auto cov_tensor = room_result.covariance.to(torch::kCPU).contiguous();
// 		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> cov_row = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(cov_tensor.data_ptr<float>());
// 		Eigen::Matrix3f cov = cov_row;
// 		consensus_manager.initializeFromRoom(room_concept->get_room_model(), pose, cov, cached_room_params_);
// 	}
//
// 	if (not consensus_manager.has_doors() and door_result.has_value() and door_result->success)
// 	{
// 		auto params = door_result->optimized_params;
// 		auto cov_tensor = door_result->covariance.to(torch::kCPU).contiguous();
// 		// extract the 3x3 covariance matrix from the 7x7 tensor
// 		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> cov_pose = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(cov_tensor.data_ptr<float>());
// 		consensus_manager.addDoor(door_concept->get_model(), cov_pose);
// 	}
// 	// Run optimization
// 	ConsensusResult result = consensus_manager.optimize();
// }

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::update_visualization(const std::chrono::high_resolution_clock::time_point &last_time)
{
    QMutexLocker lock(&results_mutex_);

    // Check if we have valid room data
    if (!latest_room_result_.has_value() || latest_room_result_->optimized_pose.empty())
        return;

    if (cached_room_params_.size() < 2 || (cached_room_params_[0] == 0.0f && cached_room_params_[1] == 0.0f))
        return;

    const auto& room_result = latest_room_result_.value();
    const auto& pose = room_result.optimized_pose;
    const auto& room_params = cached_room_params_;

    // Build robot pose transform
    Eigen::Affine2f robot_pose_display;
    robot_pose_display.translation() = Eigen::Vector2f(pose[0], pose[1]);
    robot_pose_display.linear() = Eigen::Rotation2Df(pose[2]).toRotationMatrix();

    //////////////////////////////////////////////////////////////////////////////
    /// 2D Viewer - Robot frame
    //////////////////////////////////////////////////////////////////////////////
    update_robot_view(robot_pose_display, room_result, &viewer->scene);

    //////////////////////////////////////////////////////////////////////////////
    /// 3D Viewer - Room
    //////////////////////////////////////////////////////////////////////////////
    if (viewer3d)
    {
        viewer3d->updateRoom(room_params[0], room_params[1]);
        viewer3d->updateRobotPose(Eigen::Vector3f{pose[0], pose[1], pose[2]});
    }

    //////////////////////////////////////////////////////////////////////////////
    /// 3D Viewer - Door (use consensus pose in room coordinates)
    //////////////////////////////////////////////////////////////////////////////
    if (viewer3d && cached_consensus_door_.valid)
    {
        // Draw door using consensus-optimized pose (in room coordinates)
        viewer3d->draw_door(
            cached_consensus_door_.x,
            cached_consensus_door_.y,
            cached_consensus_door_.z,
            cached_consensus_door_.theta,
            cached_consensus_door_.width,
            cached_consensus_door_.height,
            cached_consensus_door_.opening_angle
        );

        // Transform point cloud from robot frame to room frame
        if (latest_door_model_ && !latest_door_model_->roi_points.empty())
        {
            // Get robot pose in room frame
            float rx = pose[0], ry = pose[1], rtheta = pose[2];
            float cos_t = std::cos(rtheta);
            float sin_t = std::sin(rtheta);

            // Transform points: p_room = R * p_robot + t
            std::vector<Eigen::Vector3f> points_room;
            points_room.reserve(latest_door_model_->roi_points.size());

            for (const auto& p : latest_door_model_->roi_points)
            {
                Eigen::Vector3f p_room;
                p_room.x() = cos_t * p.x() - sin_t * p.y() + rx;
                p_room.y() = sin_t * p.x() + cos_t * p.y() + ry;
                p_room.z() = p.z();  // z unchanged
                points_room.push_back(p_room);
            }

            viewer3d->updatePointCloud(points_room);
        }
    }
    // Fallback to detector pose if no consensus yet (points in robot frame)
    else if (viewer3d && latest_door_result_.has_value() && latest_door_result_->success && latest_door_model_)
    {
        const auto& door_params = latest_door_result_->optimized_params;
        if (door_params.size() >= 7)
        {
            viewer3d->draw_door(door_params[0], door_params[1], door_params[2],
                               door_params[3], door_params[4], door_params[5], door_params[6]);
            viewer3d->updatePointCloud(latest_door_model_->roi_points);
        }
    }

    //////////////////////////////////////////////////////////////////////////////
    /// RGB panoramic with door overlay
    //////////////////////////////////////////////////////////////////////////////
    if (cached_rgbd_.rgb.size() > 0 && cached_rgbd_.width > 0 && cached_rgbd_.height > 0)
    {
        cv::Mat img(cached_rgbd_.height, cached_rgbd_.width, CV_8UC3,
                    const_cast<unsigned char*>(cached_rgbd_.rgb.data()));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Draw door ROI and projection if available
        if (latest_door_result_.has_value() && latest_door_result_->success && latest_door_model_)
        {
            cv::rectangle(img, latest_door_model_->roi, cv::Scalar(0, 255, 0), 2);
            DoorProjection::projectDoorOnImage(latest_door_model_, img, Eigen::Vector3f{0.0f, 0.0f, 1.2f});
        }

        const QImage qimg(img.data, img.cols, img.rows, static_cast<int>(img.step), QImage::Format_RGB888);
        label_img->clear();
        label_img->setPixmap(QPixmap::fromImage(qimg).scaled(label_img->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
    }

    //////////////////////////////////////////////////////////////////////////////
    /// Time series plots - Room
    //////////////////////////////////////////////////////////////////////////////
    room_loss_plotter->addDataPoint(0, room_result.final_loss);
    room_loss_plotter->addDataPoint(1, room_result.measurement_loss);
    room_loss_plotter->addDataPoint(2, room_result.prior_loss);

    // Extract std devs from covariance (use data_ptr for thread safety)
    if (room_result.covariance.defined() && room_result.covariance.numel() >= 9)
    {
        auto cov_cpu = room_result.covariance.to(torch::kCPU).contiguous();
        auto cov_ptr = cov_cpu.data_ptr<float>();
        const float upd_std_x = std::sqrt(std::max(1e-10f, cov_ptr[0]));      // [0][0]
        const float upd_std_y = std::sqrt(std::max(1e-10f, cov_ptr[4]));      // [1][1]
        const float upd_std_theta = std::sqrt(std::max(1e-10f, cov_ptr[8]));  // [2][2]
        stddev_plotter->addDataPoint(0, upd_std_x);
        stddev_plotter->addDataPoint(1, upd_std_y);
        stddev_plotter->addDataPoint(2, upd_std_theta);
    }

    //////////////////////////////////////////////////////////////////////////////
    /// Time series plots - Door
    //////////////////////////////////////////////////////////////////////////////
    if (latest_door_result_.has_value() && latest_door_result_->success)
    {
        float loss_rounded = std::round(latest_door_result_->measurement_loss * 1000.0f) / 1000.0f;
        if (std::fabs(loss_rounded) < 1e-6f) loss_rounded = 0.0f;
        door_loss_plotter->addDataPoint(0, loss_rounded);
    }

	//////////////////////////////////////////////////////////////////////////////
	/// Time series plots - Elapsed time
	//////////////////////////////////////////////////////////////////////////////
	const auto now = std::chrono::high_resolution_clock::now();
	epoch_plotter->addDataPoint(0, std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count());

	// Update all plotters
	room_loss_plotter->update();
	stddev_plotter->update();
	door_loss_plotter->update();
	epoch_plotter->update();
}
void SpecificWorker::update_viewers(const TimePoints &points_,
									const RoboCompCamera360RGBD::TRGBD &rgbd,
                                    const rc::RoomConcept::Result &room_result,
                                    const std::optional<rc::DoorConcept::Result> &door_result,
                                    QGraphicsScene *robot_scene,
                                    double elapsed)
{
	const auto &[points, lidar_timestamp] = points_;
	const auto robot_pose = room_concept->get_room_model()->get_robot_pose();

	//////////////////////////////////////////////////////////////////////////////
	/// Robot frame
	//////////////////////////////////////////////////////////////////////////////
	Eigen::Affine2f robot_pose_display;
	robot_pose_display.translation() = Eigen::Vector2f(robot_pose[0], robot_pose[1]);
	robot_pose_display.linear() = Eigen::Rotation2Df(robot_pose[2]).toRotationMatrix();
	draw_lidar(points, robot_scene);
	update_robot_view(robot_pose_display, room_result, robot_scene);

	// Detect and draw doors
	door_detector.draw_doors(false, robot_scene, nullptr, robot_pose_display);

	//////////////////////////////////////////////////////////////////////////////
	/// Viewer 3D ROOM
	//////////////////////////////////////////////////////////////////////////////
	//viewer3d->updatePointCloud(points);
	const auto room_params = room_concept->get_room_model()->get_room_parameters();
	viewer3d->updateRoom(room_params[0], room_params[1]); // half-width, half-height
	viewer3d->updateRobotPose(Eigen::Vector3f(robot_pose[0], robot_pose[1], robot_pose[2]));

	//////////////////////////////////////////////////////////////////////////////
	/// Viewer 3D DOORS
	//////////////////////////////////////////////////////////////////////////////
	if (door_result.has_value() and door_result->door)
	{
		const auto params = door_result->optimized_params;
		viewer3d->draw_door(params[0], params[1], params[2],params[3], params[4], params[5], params[6]);
		viewer3d->updatePointCloud(door_result->door->roi_points);
	}

	//////////////////////////////////////////////////////////////////////////////
	/// RGB panoramic
	//////////////////////////////////////////////////////////////////////////////
	cv::Mat img(rgbd.height, rgbd.width, CV_8UC3, const_cast<unsigned char*>(rgbd.rgb.data()));
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	if (door_result.has_value() and door_result->door)
	{
		cv::rectangle(img, door_result.value().door->roi, cv::Scalar(0, 255, 0), 2);
		DoorProjection::projectDoorOnImage(door_result->door, img, Eigen::Vector3f{0.0f, 0.0f, 1.2f});
	}
	const QImage qimg(img.data, img.cols, img.rows, static_cast<int>(img.step), QImage::Format_RGB888);
	label_img->clear();
	label_img->setPixmap(QPixmap::fromImage(qimg).scaled(label_img->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

	//////////////////////////////////////////////////////////////////////////////
	/// Time series plot
	//////////////////////////////////////////////////////////////////////////////
	room_loss_plotter->addDataPoint(0, room_result.final_loss);
	room_loss_plotter->addDataPoint(2, room_result.prior_loss);
	room_loss_plotter->addDataPoint(1, room_result.measurement_loss);

	// Plot updated uncertainty (after measurement update)
	const float upd_std_x = std::sqrt(room_result.covariance[0][0].item<float>()) ;
	const float upd_std_y = std::sqrt(room_result.covariance[1][1].item<float>()) ;
	const float upd_std_theta = std::sqrt(room_result.covariance[2][2].item<float>()) ;
	stddev_plotter->addDataPoint(0, upd_std_x);    // Green line
	stddev_plotter->addDataPoint(1, upd_std_y);    // Green line
	stddev_plotter->addDataPoint(2, upd_std_theta); // Green line

	if (door_result.has_value())
	{
		float loss_rounded = std::round(door_result->measurement_loss * 1000.0f) / 1000.0f; // 3 decimales
		if (std::fabs(loss_rounded) < 1e-6f) loss_rounded = 0.0f;   // opcional: evitar -0.000
		door_loss_plotter->addDataPoint(0, loss_rounded);
	}

	epoch_plotter->addDataPoint(0, elapsed);

	room_loss_plotter->update();
	stddev_plotter->update();
	epoch_plotter->update();
	door_loss_plotter->update();
	last_time = std::chrono::high_resolution_clock::now();
}

RoboCompCamera360RGBD::TRGBD SpecificWorker::read_image()
{
	RoboCompCamera360RGBD::TRGBD rgbd;
	try { rgbd = camera360rgbd_proxy->getROI(-1, -1, -1, -1, -1, -1); }
	catch (const Ice::Exception &e)
	{
		std::cout << e.what() << " Error reading 360 RGBD camera " << std::endl;
		return {};
	}
	return rgbd;
}

void SpecificWorker::print_status(const rc::RoomConcept::Result &result)
{
    static int frame_counter = 0;
    const auto &robot_pose = room_concept->get_room_model()->get_robot_pose();
    const auto &room_params = room_concept->get_room_model()->get_room_parameters();
    const bool is_localized = room_concept->room_freezing_manager.should_freeze_room();

    // --- Header ---
    qInfo().noquote() << "\n" << QString(60, '=');
    qInfo().noquote() << QString("FRAME %1").arg(frame_counter++, 5).rightJustified(38);
    qInfo().noquote() << QString(60, '=');

    // --- System State ---
    qInfo().noquote() << "System State:" << room_concept->room_freezing_manager.get_state_string().data();
    qInfo().noquote() << "";

    // --- Robot Status ---
    qInfo().noquote() << "-- ROBOT STATUS --";
    qInfo().noquote() << QString("  Position (X, Y):      (%1 m, %2 m)")
                           .arg(robot_pose[0], 7, 'f', 3)
                           .arg(robot_pose[1], 7, 'f', 3);
    qInfo().noquote() << QString("  Orientation (θ):      %1 rad (%2°)")
                           .arg(robot_pose[2], 7, 'f', 3)
                           .arg(qRadiansToDegrees(robot_pose[2]), 6, 'f', 1);

	// ESTALLA !!!!!!!!!!!!!1
    // --- Odometry Prediction (only in localization mode) ---
    // if (odom_prior.valid and is_localized and result.prediction_state.have_propagated)
    // {
    //      qInfo().noquote() << "\n-- ODOMETRY PREDICTION --";
    //      const auto  pred_pose = result.prediction_state.predicted_pose;
    //      qInfo().noquote() << QString("  Predicted Pose (X,Y,θ): (%1, %2, %3)")
    //                             .arg(pred_pose[0], 7, 'f', 3)
    //                             .arg(pred_pose[1], 7, 'f', 3)
    //                             .arg(pred_pose[2], 7, 'f', 3);
    //      qInfo().noquote() << QString("  Odometry Delta (X,Y,θ): (%1, %2, %3)")
    //                             .arg(odom_prior.delta_pose[0], 7, 'f', 3)
    //                             .arg(odom_prior.delta_pose[1], 7, 'f', 3)
    //                             .arg(odom_prior.delta_pose[2], 7, 'f', 3);
    //
    //      const float error_pos = std::hypot(odom_prior.delta_pose[0], odom_prior.delta_pose[1]);
    //      float error_theta = odom_prior.delta_pose[2];
    //      // Normalize angle error
    //      while (error_theta > M_PI) error_theta -= 2 * M_PI;
    //      while (error_theta < -M_PI) error_theta += 2 * M_PI;
    //
    //      qInfo().noquote() << QString("  Correction (Pos, θ):    %1 mm, %2°")
    //                             .arg(error_pos * 1000, 5, 'f', 1)
    //                             .arg(qRadiansToDegrees(error_theta), 5, 'f', 2);
    //  }

    // --- Room Parameters ---
    qInfo().noquote() << "\n-- ROOM PARAMETERS --";
    qInfo().noquote() << QString("  Status:               %1").arg(is_localized ? "FROZEN" : "OPTIMIZING");
    qInfo().noquote() << QString("  Dimensions (W x H):   %1 m x %2 m")
                           .arg(room_params[0] * 2, 0, 'f', 2)
                           .arg(room_params[1] * 2, 0, 'f', 2);
    const float dist_to_wall_x = room_params[0] - std::abs(robot_pose[0]);
    const float dist_to_wall_y = room_params[1] - std::abs(robot_pose[1]);
    qInfo().noquote() << QString("  Dist. to walls (X,Y): %1 m, %2 m")
                           .arg(dist_to_wall_x, 0, 'f', 2)
                           .arg(dist_to_wall_y, 0, 'f', 2);

    // --- Optimization Metrics ---
    qInfo().noquote() << "\n-- OPTIMIZATION METRICS --";
    qInfo().noquote() << QString("  Total Loss:           %1").arg(result.final_loss, 0, 'f', 6);
    qInfo().noquote() << QString("    - Measurement Loss: %1").arg(result.measurement_loss, 0, 'f', 6);
    qInfo().noquote() << QString("    - Prior Loss:       %1").arg(result.prior_loss, 0, 'f', 6);
    qInfo().noquote() << QString("  Adaptive Prior Weight:  %1").arg(result.prior.prior_weight, 0, 'f', 2);

    // --- Estimated Uncertainty (1-sigma) ---
    if (!result.std_devs.empty())
    {
        qInfo().noquote() << "\n-- ESTIMATED UNCERTAINTY (1σ) --";
        if (is_localized) // LOCALIZED Mode: [x, y, theta]
        {
            qInfo().noquote() << QString("  Position (X, Y):        ±%1 mm, ±%2 mm")
                                   .arg(result.std_devs[0] * 1000, 5, 'f', 1)
                                   .arg(result.std_devs[1] * 1000, 5, 'f', 1);
            qInfo().noquote() << QString("  Orientation (θ):        ±%1°")
                                   .arg(qRadiansToDegrees(result.std_devs[2]), 5, 'f', 2);
        }
        else // MAPPING Mode: [room_w, room_h, x, y, theta]
        {
            qInfo().noquote() << QString("  Room (W, H):            ±%1 mm, ±%2 mm")
                                   .arg(result.std_devs[0] * 1000, 5, 'f', 1)
                                   .arg(result.std_devs[1] * 1000, 5, 'f', 1);
            qInfo().noquote() << QString("  Position (X, Y):        ±%1 mm, ±%2 mm")
                                   .arg(result.std_devs[2] * 1000, 5, 'f', 1)
                                   .arg(result.std_devs[3] * 1000, 5, 'f', 1);
            qInfo().noquote() << QString("  Orientation (θ):        ±%1°")
                                   .arg(qRadiansToDegrees(result.std_devs[4]), 5, 'f', 2);
        }
    }
    qInfo().noquote() << QString(60, '=') << "\n";
}



// RoboCompCamera360RGBD::TRGBD SpecificWorker::read_image()
// {
// 	RoboCompCamera360RGB::TImage img;
// 	try { img = camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1); } catch (const Ice::Exception &e)
// 	{
// 		std::cout << e.what() << " Error reading 360 camera " << std::endl;
// 		return {};
// 	}
//
// 	// convert to cv::Mat
// 	cv::Mat cv_img(img.height, img.width, CV_8UC3, img.image.data());
//
// 	// extract a ROI leaving out borders (same as original logic)
// 	const int left_offset = cv_img.cols / 8;
// 	const int vert_offset = cv_img.rows / 4;
// 	const cv::Rect roi(left_offset, vert_offset, cv_img.cols - 2 * left_offset, cv_img.rows - 2 * vert_offset);
// 	if (roi.width <= 0 || roi.height <= 0) return {};
// 	cv_img = cv_img(roi);
//
// 	// Convert BGR -> RGB for display
// 	cv::Mat display_img;
// 	cv::cvtColor(cv_img, display_img, cv::COLOR_BGR2RGB);
// 	// resize to 640x480
// 	cv::resize(display_img, display_img, cv::Size(640, 480));
//
// 	return display_img.clone();
// }


/// Read LiDAR data and apply filters. Points are converted to meters
TimePoints SpecificWorker::read_data()
{
	RoboCompLidar3D::TData ldata;
	try
	{
		ldata = lidar3d_proxy->getLidarData("helios", 0, 2 * M_PI, 1);
	} catch (const Ice::Exception &e)
	{ std::cout << e << " " << "No lidar data from sensor" << std::endl; return {}; }
	if (ldata.points.empty()) return {};

	//ldata.points = filter_same_phi(ldata.points);
	//ldata.points = filter_isolated_points(ldata.points, 150);
	ldata.points = filter_isolated_points_torch(ldata.points, 150);

	// Use door detector to filter points (removes points near detected doors)
	ldata.points = door_detector.filter_points(ldata.points);

	return {ldata.points,std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::milliseconds(ldata.timestamp))};
}

// Filter isolated points using GPU acceleration (LibTorch)
RoboCompLidar3D::TPoints SpecificWorker::filter_isolated_points_torch(const RoboCompLidar3D::TPoints &points, float d)
{
    if (points.empty()) return {};

    int n = points.size();

    // --- 1. PREPARE DATA (CPU) ---
    // We flatten them into a vector for easy conversion to Tensor.
    std::vector<float> flat_points;
    flat_points.reserve(n * 3);  //3D
    for(const auto& p : points)
    {
        flat_points.push_back(p.x);
        flat_points.push_back(p.y);
    	flat_points.push_back(p.z);
    }

    // --- 2. MOVE TO GPU (LibTorch) ---
    // Create tensor of shape (N, 2)
    torch::Tensor t_points = torch::from_blob(flat_points.data(), {n, 3}, torch::kFloat);

    // Transfer to GPU (ensure you have a CUDA device available)
    if (torch::cuda::is_available()) {
        t_points = t_points.to(torch::kCUDA);
    }

    // --- 3. COMPUTE DISTANCES ---
    // torch::cdist computes the Euclidean distance between every pair of points.
    // Result is an (N, N) matrix.
    const auto dists = torch::cdist(t_points, t_points);

    // --- 4. APPLY LOGIC ---
    // We need dist <= d.
    // However, the distance from a point to itself is 0, which is always <= d.
    // We must ignore the diagonal (self-loops).

    // Fill diagonal with infinity so it doesn't trigger the threshold
    dists.fill_diagonal_(std::numeric_limits<float>::infinity());

    // Check threshold: creates a boolean matrix (N, N)
    // Then .any(1) reduces it to (N), true if *any* neighbor exists for that row.
    const auto has_neighbor_tensor = (dists <= d).any(1);

    // --- 5. RETRIEVE RESULTS ---
    // Move the boolean mask back to CPU
    const auto has_neighbor_cpu = has_neighbor_tensor.to(torch::kCPU);

    // Get efficient access to the data
    auto accessor = has_neighbor_cpu.accessor<bool, 1>();

    RoboCompLidar3D::TPoints result;
    result.reserve(n);

    // Filter the original C++ vector using the Torch mask
    for(int i = 0; i < n; ++i) {
        if (accessor[i]) {
            result.push_back(points[i]);
        }
    }

    return result;
}

void SpecificWorker::draw_lidar(const RoboCompLidar3D::TPoints &filtered_points, QGraphicsScene *scene)
{
	static std::vector<QGraphicsItem *> items; // store items so they can be shown between iterations

	// remove all items drawn in the previous iteration
	for (const auto i: items)
	{
		scene->removeItem(i);
		delete i;
	}
	items.clear();

	const auto color = QColor(Qt::darkGreen);
	const auto brush = QBrush(QColor(Qt::darkGreen));
	for (const auto &p: filtered_points)
	{
		const auto item = scene->addRect(-25, -25, 50, 50, color, brush);
		item->setPos(p.x, p.y);
		items.push_back(item);
	}
}

void SpecificWorker::update_robot_view(const Eigen::Affine2f &robot_pose,
                                       const rc::RoomConcept::Result &result,
                                       QGraphicsScene *scene)
{
	// draw room in robot viewer
	static QGraphicsItem *room_draw_robot = nullptr;
	if (room_draw_robot != nullptr)
	{
		viewer->scene.removeItem(room_draw_robot);
		delete room_draw_robot;
	}
	// compute room in robot frame - use cached_room_params_ (thread-safe)
	const auto& rp = cached_room_params_;
	if (rp.size() < 2 || (rp[0] == 0.0f && rp[1] == 0.0f))
		return;  // Room not initialized yet

	Eigen::Vector2f top_left(-rp[0], rp[1]);
	Eigen::Vector2f top_right(rp[0], rp[1]);
	Eigen::Vector2f bottom_left(-rp[0], -rp[1]);
	Eigen::Vector2f bottom_right(rp[0], -rp[1]);
	const Eigen::Matrix2f R = robot_pose.rotation().transpose();
	const Eigen::Vector2f t = -R * robot_pose.translation();
	top_left = R * top_left + t;
	top_right = R * top_right + t;
	bottom_left = R * bottom_left + t;
	bottom_right = R * bottom_right + t;
	QPolygonF polygon;
	polygon << QPointF(top_left.x() * 1000, top_left.y() * 1000)
			<< QPointF(top_right.x() * 1000, top_right.y() * 1000)
			<< QPointF(bottom_right.x() * 1000, bottom_right.y() * 1000)
			<< QPointF(bottom_left.x() * 1000, bottom_left.y() * 1000);
	room_draw_robot = viewer->scene.addPolygon(polygon, QPen(QColor("cyan"), 40));

	// ===== DRAW DUAL UNCERTAINTY VISUALIZATION =====
	static QGraphicsItem *cov_item=nullptr, *propagated_cov_item=nullptr;
	if (cov_item != nullptr)
	{ scene->removeItem(cov_item); delete cov_item; cov_item=nullptr; }
	if (propagated_cov_item != nullptr)
	{ scene->removeItem(propagated_cov_item); delete propagated_cov_item; propagated_cov_item = nullptr; }


	const std::vector<float> robot_pose_vec{robot_pose.translation().x(),
										    robot_pose.translation().y(),
										    Eigen::Rotation2Df(robot_pose.rotation()).angle()}; // ellipse is drawn at origin in robot frame

	// 2. Draw UPDATED uncertainty (after measurements)
	//    This shows how measurements reduced uncertainty
	if (result.covariance.defined() && result.covariance.numel() > 0)
	{
		cov_item = draw_uncertainty_ellipse(
			scene,
			result.covariance,
			robot_pose_vec,
			QColor(0, 255, 0, 100), // Green, semi-transparent
			2.0f // 2-sigma
		);
	}

	update_gui(robot_pose, result);
}

void SpecificWorker::update_gui(const Eigen::Affine2f &robot_pose, const rc::RoomConcept::Result &result)
{
    // get last adv and rot commands from buffer
    float adv = 0.0f;
    float rot = 0.0f;
    if (not velocity_history_.empty())
    {
        const auto v = velocity_history_.back();
        adv = v.adv_z;
        rot = v.rot;
    }
    lcdNumber_adv->display(adv);
    lcdNumber_rot->display(rot);
    lcdNumber_x->display(robot_pose.translation().x());
    lcdNumber_y->display(robot_pose.translation().y());
    const float angle = Eigen::Rotation2Df(robot_pose.rotation()).angle();
    lcdNumber_angle->display(angle);
    // Use cached state - don't access room_concept from main thread
    // label_state is updated in onRoomStateChanged
}

QGraphicsEllipseItem* SpecificWorker::draw_uncertainty_ellipse(
	    QGraphicsScene *scene,
	    const torch::Tensor &covariance,
	    const std::vector<float> &robot_pose,
	    const QColor &color,
	    float scale_factor)  // 2-sigma = 95% confidence
{
    if (!covariance.defined() || covariance.numel() < 9) {
        return nullptr;
    }

    // Extract position covariance (2x2 submatrix) using data_ptr (thread-safe with cloned tensor)
    auto cov_cpu = covariance.to(torch::kCPU).contiguous();
    auto cov_ptr = cov_cpu.data_ptr<float>();

    // Assuming row-major 3x3: [0][0]=0, [0][1]=1, [1][0]=3, [1][1]=4
    float var_x = cov_ptr[0];      // [0][0]
    float var_y = cov_ptr[4];      // [1][1]
    float cov_xy = cov_ptr[1];     // [0][1]

    // Compute eigenvalues and eigenvectors for ellipse orientation
    // Covariance matrix: [var_x  cov_xy]
    //                    [cov_xy var_y ]
    float trace = var_x + var_y;
    float det = var_x * var_y - cov_xy * cov_xy;
    float discriminant = std::sqrt(trace * trace / 4.0f - det);

    float lambda1 = trace / 2.0f + discriminant;  // Larger eigenvalue
    float lambda2 = trace / 2.0f - discriminant;  // Smaller eigenvalue

    // Standard deviations (semi-axes of ellipse)
    float std_major = std::sqrt(std::max(lambda1, 1e-8f));  // Avoid sqrt of negative
    float std_minor = std::sqrt(std::max(lambda2, 1e-8f));

    // Ellipse orientation (angle of major axis)
    float angle_rad = 0.0f;
    if (std::abs(cov_xy) > 1e-6f) {
        angle_rad = 0.5f * std::atan2(2.0f * cov_xy, var_x - var_y);
    }
    float angle_deg = angle_rad * 180.0f / M_PI;

    // Convert to millimeters for visualization
    float width_mm = 2.0f * scale_factor * std_major * 1000.0f;   // 2-sigma = 95%
    float height_mm = 2.0f * scale_factor * std_minor * 1000.0f;

    // Create ellipse (centered at robot)
    const auto ellipse = scene->addEllipse(
      - width_mm/2.0f,
      - height_mm/2.0f,
        width_mm,
        height_mm,
        QPen(color, 30),  // Thicker pen for visibility
        QBrush(color)  // No fill, just outline
    );

    // Rotate ellipse to match covariance orientation
    ellipse->setTransformOriginPoint(0, 0);
    ellipse->setRotation(angle_deg);  // Qt uses clockwise, we use counterclockwise

    // Set Z-order (draw on top)
    ellipse->setZValue(100);
	return ellipse;
}

///////////////////////////////////////////////////////////////////////////////
/////SUBSCRIPTION to sendData method from JoystickAdapter interface
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
	// Parse velocity command (assuming axes[0]=adv_x, axes[1]=adv_z, axes[2]=rot)
	VelocityCommand cmd;
	for (const auto &axis: data.axes)
	{
		if (axis.name == "rotate")
			cmd.rot = axis.value;
		else if (axis.name == "advance") // forward is positive Z. Right-hand rule
			cmd.adv_z = axis.value;
		else if (axis.name == "side")
			cmd.adv_x = 0.0f; // not lateral motion allowed
	}
	cmd.timestamp = std::chrono::high_resolution_clock::now();
	velocity_history_.push_back(cmd);

	//qDebug() << __FUNCTION__ << "Velocity stored:" << cmd.adv_x << "mm/s," << cmd.adv_z << "mm/s," << cmd.rot << "rad/s";
	// Predict next pose for visualization
	//const auto dt = std::chrono::duration<float>(cmd.timestamp - last_time).count();
	//auto delta = integrate_odometry(cmd, dt);
	// Could show predicted pose as ghost overlay
}

///////////////////////////////////////////////////////////////////////////////
// =============================================================================
// SLOTS - Receive results from threads
// =============================================================================

void SpecificWorker::onRoomInitialized(std::shared_ptr<RoomModel> model, std::vector<float> room_params)
{
    qInfo() << "Room initialized callback received";

    // Store room params and model reference (both thread-safe now)
    {
        QMutexLocker lock(&results_mutex_);
        latest_room_model_ = model;
        cached_room_params_ = room_params;
    }

    qInfo() << "Room model reference stored, params:" << room_params[0] << "x" << room_params[1];
}

void SpecificWorker::onRoomUpdated(std::shared_ptr<RoomModel> model, rc::RoomConcept::Result result, std::vector<float> room_params)
{
    // Cache results for visualization (called from compute)
    {
        QMutexLocker lock(&results_mutex_);
        latest_room_model_ = model;
        latest_room_result_ = result;
        cached_room_params_ = room_params;
    }

    // Emit to consensus manager (via signal)
    if (result.uncertainty_valid && !result.optimized_pose.empty())
    {
        Eigen::Vector3f robot_pose(result.optimized_pose[0], result.optimized_pose[1], result.optimized_pose[2]);

        // Extract covariance from cloned tensor
        Eigen::Matrix3f cov = Eigen::Matrix3f::Identity() * 0.1f;
        if (result.covariance.defined() && result.covariance.numel() >= 9)
        {
            auto cov_cpu = result.covariance.to(torch::kCPU).contiguous();
            auto cov_ptr = cov_cpu.data_ptr<float>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cov(i, j) = cov_ptr[i * 3 + j];
        }

        Q_EMIT roomDataReady(model, robot_pose, cov, room_params);
    }
}

void SpecificWorker::onRoomStateChanged(RoomState new_state)
{
	const auto state_str = (new_state == RoomState::MAPPING) ? "MAPPING" : "LOCALIZED";
    qInfo() << "Room state changed to:" << state_str;

    // Update UI
    label_state->setText(state_str);
}

void SpecificWorker::onDoorDetected(std::shared_ptr<DoorModel> model)
{
    qInfo() << "Door detected - ID:" << model->id;

    {
        QMutexLocker lock(&results_mutex_);
        latest_door_model_ = model;
    }

    // Emit to consensus manager (it will check if initialized)
    Eigen::Matrix3f cov = Eigen::Matrix3f::Identity() * 0.15f;
    {
        QMutexLocker lock(&results_mutex_);
        if (latest_door_result_.has_value() && latest_door_result_->covariance.defined())
        {
            auto cov_tensor = latest_door_result_->covariance.to(torch::kCPU).contiguous();
            if (cov_tensor.numel() >= 9)
            {
                auto cov_ptr = cov_tensor.data_ptr<float>();
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        cov(i, j) = cov_ptr[i * 3 + j];
            }
        }
    }
    Q_EMIT doorDetectionReady(model, cov);
}

void SpecificWorker::onDoorUpdated(std::shared_ptr<DoorModel> model, rc::DoorConcept::Result result)
{
    // Cache results for visualization
    {
        QMutexLocker lock(&results_mutex_);
        latest_door_model_ = model;
        latest_door_result_ = result;
    }

    // Emit to consensus manager for optimization
    if (!result.optimized_params.empty())
    {
        Eigen::Vector3f door_pose(result.optimized_params[0], result.optimized_params[1], result.optimized_params[3]);

        Eigen::Matrix3f cov = Eigen::Matrix3f::Identity() * 0.1f;
        if (result.covariance.defined() && result.covariance.numel() >= 9)
        {
            auto cov_cpu = result.covariance.to(torch::kCPU).contiguous();
            auto cov_ptr = cov_cpu.data_ptr<float>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cov(i, j) = cov_ptr[i * 3 + j];
        }

        Q_EMIT doorUpdateReady(door_pose, cov);
    }
}

void SpecificWorker::onDoorTrackingLost()
{
    qInfo() << "Door tracking lost";

    {
        QMutexLocker lock(&results_mutex_);
        latest_door_model_ = nullptr;
        latest_door_result_.reset();
        cached_consensus_door_.valid = false;
    }
}

void SpecificWorker::onConsensusDoorPose(size_t door_index, float x, float y, float z, float theta,
                                          float width, float height, float opening_angle)
{
    QMutexLocker lock(&results_mutex_);
    cached_consensus_door_.valid = true;
    cached_consensus_door_.x = x;
    cached_consensus_door_.y = y;
    cached_consensus_door_.z = z;
    cached_consensus_door_.theta = theta;
    cached_consensus_door_.width = width;
    cached_consensus_door_.height = height;
    cached_consensus_door_.opening_angle = opening_angle;
}

///////////////////////////////////////////////////////////////////////////////

void SpecificWorker::emergency()
{
	std::cout << "Emergency worker" << std::endl;
	//emergencyCODE
	//
	//if (SUCCESSFUL) //The componet is safe for continue
	//  emmit goToRestore()
}

//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
	std::cout << "Restore worker" << std::endl;
	//restoreCODE
	//Restore emergency component
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}