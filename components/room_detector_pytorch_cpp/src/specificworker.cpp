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

		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period,
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

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
	this->resize(1000, 600);
	show();

	// Initialize RoomModel
	const auto &[points, lidar_time] = read_data();
	if (points.empty())
	{
		std::cout << __FUNCTION__ << " No LiDAR points available\n";
		return;
	}
	draw_lidar(points, &viewer->scene);
	room.init(points);
	room.init_odometry_calibration(1.0f, 1.0f); // Start with no correction

	// time series plotter for match error
	TimeSeriesPlotter::Config plotConfig;
	// all fields have to be initialized, otherwise garbage values get to the constructor
	plotConfig.title = "Loss (likelihood + prior)";
	plotConfig.yAxisLabel = "Error";
	plotConfig.xAxisLabel = "";
	plotConfig.timeWindowSeconds = 7.0; // Show a 15-second window
	plotConfig.autoScaleY = true; // We will set a fixed range
	plotConfig.showLegend = true; // Show graph legend
	plotConfig.yMin = 0;
	plotConfig.yMax = 0.4;
	time_series_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error, plotConfig);
	graphs.push_back(time_series_plotter->addGraph("post", Qt::blue));
	graphs.push_back(time_series_plotter->addGraph("like", Qt::green));
	graphs.push_back(time_series_plotter->addGraph("prior", Qt::red));

	// Create 3D viewer
	viewer3d = std::make_unique<RoomVisualizer3D>("src/meshes/shadow.obj");
	QWidget *viewer3d_widget = viewer3d->getWidget();
	QVBoxLayout *layout = new QVBoxLayout(frame_3d); // Your QFrame name here
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(viewer3d_widget);
	viewer3d->show();

	// Yolo door detector
	std::string model_path = "best.torchscript";
	yolo_detector = std::make_unique<YOLODetector>(model_path, std::vector<std::string>{}, 0.25f, 0.45f, 640, true);
}

void SpecificWorker::compute()
{
	//auto init_time = std::chrono::high_resolution_clock::now();
	// Read LiDAR data (in robot frame)
	const auto time_points = read_data();
	//frame_counter++;
	if (std::get<0>(time_points).empty())
	{
		std::cout << "No LiDAR points available\n";
		return;
	}

	//now = std::chrono::high_resolution_clock::now();
	//qInfo() << "dt2" << std::chrono::duration_cast<std::chrono::milliseconds>(now - init_time).count();

	// Optimize SDF likelihood + odometry prior
	//const auto result = optimizer.optimize(points, room, time_series_plotter, 150,
	//									   0.01f,0.01f, odom_prior, frame_counter);
	const auto result = optimizer.optimize(time_points,
	                                       room, velocity_history_,
	                                       time_series_plotter,
	                                       150,
	                                       0.01f,
	                                       0.01f);

	//now = std::chrono::high_resolution_clock::now();
	//qInfo() << "dt3" << std::chrono::duration_cast<std::chrono::milliseconds>(now - init_time).count();

	update_viewers(time_points, result, &viewer->scene);
	print_status(result);

	//now = std::chrono::high_resolution_clock::now();
	//qInfo() << "dt4" << std::chrono::duration_cast<std::chrono::milliseconds>(now - init_time).count();

	last_time = std::chrono::high_resolution_clock::now();;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::update_viewers(const TimePoints &points_,
                                    const RoomOptimizer::Result &result,
                                    QGraphicsScene *scene)
{
	const auto &[points, lidar_timestamp] = points_;
	const auto robot_pose = room.get_robot_pose();

	Eigen::Affine2f robot_pose_display;
	robot_pose_display.translation() = Eigen::Vector2f(robot_pose[0], robot_pose[1]);
	robot_pose_display.linear() = Eigen::Rotation2Df(robot_pose[2]).toRotationMatrix();

	draw_lidar(points, scene);
	door_detector.draw_doors(false, scene, nullptr, robot_pose_display);

	update_robot_view(robot_pose_display, result, scene);
	//viewer3d->updatePointCloud(points);
	const auto room_params = room.get_room_parameters();
	viewer3d->updateRoom(room_params[0], room_params[1]); // half-width, half-height
	viewer3d->updateRobotPose(room.get_robot_pose()[0], room.get_robot_pose()[1], room.get_robot_pose()[2]);

	// Show uncertainty when localized
	// if (room.are_room_parameters_frozen())
	// {
	// 	viewer3d->updateUncertainty(
	// 		result.std_devs[0],  // X stddev
	// 		result.std_devs[1],  // Y stddev
	// 		result.std_devs[2]   // Theta stddev
	// 	);
	// 	viewer3d->showUncertainty(true);
	// } else
	// 	viewer3d->showUncertainty(false);  // Hide during mapping


	// Visual effect: You should see TWO ellipses
	// - Orange (larger): uncertainty after motion
	// - Green (smaller): uncertainty after measurement update
	// The difference shows the information gain from measurements!

	// Time series plot
	// if (time_series_plotter && result.covariance.defined()) {
	// 	// Plot propagated uncertainty (before measurement update)
	// 	if (result.propagated_cov.defined() && result.propagated_cov.numel() > 0) {
	// 		float prop_std_x = std::sqrt(result.propagated_cov[0][0].item<float>()) * 1000;
	// 		float prop_std_y = std::sqrt(result.propagated_cov[1][1].item<float>()) * 1000;
	//
	// 		time_series_plotter->addDataPoint(3, prop_std_x);  // Orange line
	// 		time_series_plotter->addDataPoint(4, prop_std_y);  // Orange line
	// 	}
	//
	// 	// Plot updated uncertainty (after measurement update)
	// 	float upd_std_x = std::sqrt(result.covariance[0][0].item<float>()) * 1000;
	// 	float upd_std_y = std::sqrt(result.covariance[1][1].item<float>()) * 1000;
	// 	float upd_std_theta = std::sqrt(result.covariance[2][2].item<float>()) * 57.3;
	//
	// 	time_series_plotter->addDataPoint(5, upd_std_x);    // Green line
	// 	time_series_plotter->addDataPoint(6, upd_std_y);    // Green line
	// 	time_series_plotter->addDataPoint(7, upd_std_theta); // Green line
	// }
}

void SpecificWorker::print_status(const RoomOptimizer::Result &result)
{
	static int frame_counter = 0;
	// Store predicted pose for later comparison
	Eigen::Vector3f predicted_pose = Eigen::Vector3f::Zero();
	bool have_prediction = false;
	auto odom_prior = result.prior;
	if (odom_prior.valid && optimizer.room_freezing_manager.should_freeze_room())
	{
		const auto current_pose = room.get_robot_pose();
		predicted_pose[0] = current_pose[0] + odom_prior.delta_pose[0];
		predicted_pose[1] = current_pose[1] + odom_prior.delta_pose[1];
		predicted_pose[2] = current_pose[2] + odom_prior.delta_pose[2];
		have_prediction = true;
	}

	// ===== COMPREHENSIVE DEBUG OUTPUT =====
	auto robot_pose = room.get_robot_pose();
	auto room_params = room.get_room_parameters();
	qInfo() << "=====================================================";
	qInfo() << "\n========== FRAME" << frame_counter++ << "==========";

	// Current state
	qInfo() << "STATE:" << optimizer.room_freezing_manager.get_state_string().data();

	// Robot pose
	qInfo() << "ROBOT POSE:";
	qInfo() << "  Position: (" << QString::number(robot_pose[0], 'f', 3)
			<< "," << QString::number(robot_pose[1], 'f', 3) << ") m";
	qInfo() << "  Orientation:" << QString::number(robot_pose[2], 'f', 3) << "rad"
			<< "(" << QString::number(qRadiansToDegrees(robot_pose[2]), 'f', 1) << "°)";

	// Predicted pose (if odometry was used)
	qInfo() << odom_prior.valid << optimizer.room_freezing_manager.should_freeze_room();
	if (odom_prior.valid && optimizer.room_freezing_manager.should_freeze_room())
	{
		auto prev_pose = room.get_robot_pose(); // This is AFTER optimization
		// Calculate what the predicted pose was
		float pred_x = prev_pose[0] - odom_prior.delta_pose[0];
		float pred_y = prev_pose[1] - odom_prior.delta_pose[1];
		float pred_theta = prev_pose[2] - odom_prior.delta_pose[2];

		qInfo() << "ODOMETRY PREDICTION:";
		qInfo() << "  Predicted: (" << QString::number(pred_x + odom_prior.delta_pose[0], 'f', 3)
				<< "," << QString::number(pred_y + odom_prior.delta_pose[1], 'f', 3)
				<< "," << QString::number(pred_theta + odom_prior.delta_pose[2], 'f', 3) << ")";
		qInfo() << "  Delta: (" << QString::number(odom_prior.delta_pose[0], 'f', 3)
				<< "," << QString::number(odom_prior.delta_pose[1], 'f', 3)
				<< "," << QString::number(odom_prior.delta_pose[2], 'f', 3) << ")";

		if (have_prediction)
		{
			float pred_error_x = robot_pose[0] - predicted_pose[0];
			float pred_error_y = robot_pose[1] - predicted_pose[1];
			float pred_error_theta = robot_pose[2] - predicted_pose[2];

			// Normalize angle error to [-π, π]
			while (pred_error_theta > M_PI) pred_error_theta -= 2 * M_PI;
			while (pred_error_theta < -M_PI) pred_error_theta += 2 * M_PI;

			float pred_error_pos = std::sqrt(pred_error_x * pred_error_x + pred_error_y * pred_error_y);

			qInfo() << "PREDICTION ERROR:";
			qInfo() << "  Position:" << QString::number(pred_error_pos * 1000, 'f', 1) << "mm";
			qInfo() << "  Orientation:" << QString::number(qRadiansToDegrees(pred_error_theta), 'f', 2) << "°";
		}
	}

	// Room parameters
	qInfo() << "ROOM:";
	qInfo() << "  Size:" << QString::number(room_params[0] * 2, 'f', 2) << "x"
			<< QString::number(room_params[1] * 2, 'f', 2) << "m";
	qInfo() << "  Status:" << (optimizer.room_freezing_manager.should_freeze_room() ? "FROZEN" : "OPTIMIZING");
	room.print_info();

	// Loss and uncertainty
	qInfo() << "METRICS:";
	qInfo() << "  Final loss:" << QString::number(result.final_loss, 'f', 6);

	if (!result.std_devs.empty())
	{
		if (optimizer.room_freezing_manager.should_freeze_room())
		{
			// LOCALIZED mode: [x, y, theta]
			qInfo() << "  Uncertainty (1σ):";
			qInfo() << "    Position: ±" << QString::number(result.std_devs[0] * 1000, 'f', 1) << "mm (X),"
					<< "±" << QString::number(result.std_devs[1] * 1000, 'f', 1) << "mm (Y)";
			qInfo() << "    Orientation: ±" << QString::number(qRadiansToDegrees(result.std_devs[2]), 'f', 2) << "°";
		} else
		{
			// MAPPING mode: [room_w, room_h, x, y, theta]
			qInfo() << "  Uncertainty (1σ):";
			qInfo() << "    Room: ±" << QString::number(result.std_devs[0] * 1000, 'f', 1) << "mm (W),"
					<< "±" << QString::number(result.std_devs[1] * 1000, 'f', 1) << "mm (H)";
			qInfo() << "    Position: ±" << QString::number(result.std_devs[2] * 1000, 'f', 1) << "mm (X),"
					<< "±" << QString::number(result.std_devs[3] * 1000, 'f', 1) << "mm (Y)";
			qInfo() << "    Orientation: ±" << QString::number(qRadiansToDegrees(result.std_devs[4]), 'f', 2) << "°";
		}
	}

	// Distance to walls
	const float dist_to_wall_x = room_params[0] - std::abs(robot_pose[0]);
	const float dist_to_wall_y = room_params[1] - std::abs(robot_pose[1]);
	qInfo() << "  Distance to walls: X=" << QString::number(dist_to_wall_x, 'f', 2) << "m,"
			<< "Y=" << QString::number(dist_to_wall_y, 'f', 2) << "m";

	qInfo() << "================================\n";
}

cv::Mat SpecificWorker::read_image()
{
	RoboCompCamera360RGB::TImage img;
	try { img = camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1); } catch (const Ice::Exception &e)
	{
		std::cout << e.what() << " Error reading 360 camera " << std::endl;
		return cv::Mat{};
	}

	// convert to cv::Mat
	cv::Mat cv_img(img.height, img.width, CV_8UC3, img.image.data());

	// extract a ROI leaving out borders (same as original logic)
	const int left_offset = cv_img.cols / 8;
	const int vert_offset = cv_img.rows / 4;
	const cv::Rect roi(left_offset, vert_offset, cv_img.cols - 2 * left_offset, cv_img.rows - 2 * vert_offset);
	if (roi.width <= 0 || roi.height <= 0) return cv::Mat{};
	cv_img = cv_img(roi);

	// Convert BGR -> RGB for display
	cv::Mat display_img;
	cv::cvtColor(cv_img, display_img, cv::COLOR_BGR2RGB);

	return display_img.clone();
}

TimePoints SpecificWorker::read_data()
{
	RoboCompLidar3D::TData ldata;
	try
	{
		ldata = lidar3d_proxy->getLidarData("helios", 0, 2 * M_PI, 2);
	} catch (const Ice::Exception &e)
	{
		std::cout << e << " " << "No lidar data from sensor" << std::endl;
		return {};
	}
	if (ldata.points.empty()) return {};

	// // Compute mean and variance (Welford) only over valid z (>1000 <2000)
	double mean = 0.0;
	double M2 = 0.0;
	std::size_t count = 0;
	for (const auto &p: ldata.points)
	{
		if (p.z <= 1000 or p.z > 2000) continue; // keep same validity criterion
		++count;
		const double delta = p.r - mean;
		mean += delta / static_cast<double>(count);
		M2 += delta * (p.r - mean);
	}
	if (count == 0) { return {}; }

	// population variance to match original behavior (divide by N)
	const double var = M2 / static_cast<double>(count);
	const double stddev = std::sqrt(var);
	if (stddev == 0.0)
	{
		qInfo() << __FUNCTION__ << "Zero variance in range data, all points have same r: " << mean;
		return {};
	}

	// filter out points with r beyond 2 stddevs (apply only if stddev > 0)
	RoboCompLidar3D::TPoints no_outliers;
	no_outliers.reserve(ldata.points.size());
	const double threshold = 2.0 * stddev;
	for (const auto &p: ldata.points)
		if (std::fabs(p.r - mean) <= threshold)
			no_outliers.push_back(p);
	ldata.points = std::move(no_outliers);

	//ldata.points = filter_same_phi(ldata.points);
	ldata.points = filter_isolated_points(ldata.points, 200);

	// Use door detector to filter points (removes points near detected doors)
	ldata.points = door_detector.filter_points(ldata.points);

	return {
		ldata.points,
		std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::milliseconds(ldata.timestamp))
	};
}

// Filter isolated points: keep only points with at least one neighbor within distance d
RoboCompLidar3D::TPoints SpecificWorker::filter_isolated_points(const RoboCompLidar3D::TPoints &points, float d)
{
	if (points.empty()) return {};

	const float d_squared = d * d; // Avoid sqrt by comparing squared distances
	std::vector<bool> hasNeighbor(points.size(), false);

	// Create index std::vector for parallel iteration
	std::vector<size_t> indices(points.size());
	std::iota(indices.begin(), indices.end(), size_t{0});

	// Parallelize outer loop - each thread checks one point
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i)
	{
		const auto &p1 = points[i];
		// Sequential inner loop (avoid nested parallelism)
		for (auto &&[j,p2]: iter::enumerate(points))
		//for (size_t j = 0; j < points.size(); ++j)
		{
			if (i == j) continue;
			//const auto& p2 = points[j]
			const float dx = p1.x - p2.x;
			const float dy = p1.y - p2.y;
			if (dx * dx + dy * dy <= d_squared)
			{
				hasNeighbor[i] = true;
				break;
			}
		}
	});

	// Collect results
	std::vector<RoboCompLidar3D::TPoint> result;
	result.reserve(points.size());
	for (auto &&[i, p]: iter::enumerate(points))
		if (hasNeighbor[i])
			result.push_back(points[i]);
	return result;
}

std::expected<int, std::string> SpecificWorker::closest_lidar_index_to_given_angle(const auto &points, float angle)
{
	// search for the point in points whose phi value is closest to angle
	auto res = std::ranges::find_if(points, [angle](auto &a) { return a.phi > angle; });
	if (res != std::end(points))
		return std::distance(std::begin(points), res);
	else
		return std::unexpected("No closest value found in method <closest_lidar_index_to_given_angle>");
}

RoboCompLidar3D::TPoints SpecificWorker::filter_same_phi(const RoboCompLidar3D::TPoints &points)
{
	if (points.empty())
		return {};

	RoboCompLidar3D::TPoints result;
	result.reserve(points.size());

	for (auto &&[angle, group]: iter::groupby(points, [](const auto &p)
	{
		float multiplier = std::pow(10.0f, 2);
		return std::floor(p.phi * multiplier) / multiplier;
	}))
	{
		auto min = std::min_element(std::begin(group), std::end(group),
		                            [](const auto &a, const auto &b) { return a.r < b.r; });
		result.emplace_back(*min);
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

void SpecificWorker::update_robot_view(const Eigen::Affine2f &robot_pose, const RoomOptimizer::Result &result, QGraphicsScene *scene)
{
	// draw room in robot viewer
	static QGraphicsItem *room_draw_robot = nullptr;
	if (room_draw_robot != nullptr)
	{
		viewer->scene.removeItem(room_draw_robot);
		delete room_draw_robot;
	}
	// compute room in robot frame
	Eigen::Vector2f top_left(-room.get_room_parameters()[0], room.get_room_parameters()[1]);
	Eigen::Vector2f top_right(room.get_room_parameters()[0], room.get_room_parameters()[1]);
	Eigen::Vector2f bottom_left(-room.get_room_parameters()[0], -room.get_room_parameters()[1]);
	Eigen::Vector2f bottom_right(room.get_room_parameters()[0], -room.get_room_parameters()[1]);
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
	// // 1. Draw PROPAGATED uncertainty (after motion, before measurements)
	// //    This shows how much uncertainty grew due to motion
	// if (result.propagated_cov.defined() && result.propagated_cov.numel() > 0)
	// {
	// 	propagated_cov_item = draw_propagated_uncertainty_ellipse(
	// 		scene,
	// 		result.propagated_cov,
	// 		robot_pose_vec,
	// 		QColor(255, 0, 0, 100), // Orange, semi-transparent
	// 		2.0f // 2-sigma
	// 	);
	// }

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

	// update GUI
	time_series_plotter->update();
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
	//lcdNumber_room->display(current_room);
	//const double angle = qRadiansToDegrees(std::atan2(robot_pose.rotation()(1, 0), robot_pose.rotation()(0, 0)));
	const float angle = Eigen::Rotation2Df(robot_pose.rotation()).angle();
	lcdNumber_angle->display(angle);
	label_state->setText(
		optimizer.room_freezing_manager.state_to_string(optimizer.room_freezing_manager.get_state()).data());
}

QGraphicsEllipseItem* SpecificWorker::draw_uncertainty_ellipse(
	    QGraphicsScene *scene,
	    const torch::Tensor &covariance,
	    const std::vector<float> &robot_pose,
	    const QColor &color,
	    float scale_factor)  // 2-sigma = 95% confidence
{
    if (!covariance.defined() || covariance.numel() == 0) {
        return nullptr;
    }

    // Extract position covariance (2x2 submatrix)
    float var_x = covariance[0][0].item<float>();
    float var_y = covariance[1][1].item<float>();
    float cov_xy = covariance[0][1].item<float>();

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

    // Robot position in mm
    float robot_x_mm = robot_pose[0] * 1000.0f;
    float robot_y_mm = robot_pose[1] * 1000.0f;

    // Create ellipse (centered at robot)
    auto ellipse = scene->addEllipse(
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
QGraphicsEllipseItem* SpecificWorker::draw_propagated_uncertainty_ellipse(
	    QGraphicsScene *scene,
	    const torch::Tensor &covariance,
	    const std::vector<float> &robot_pose,
	    const QColor &color,
	    float scale_factor)  // 2-sigma = 95% confidence
{
    if (!covariance.defined() || covariance.numel() == 0) {
        return nullptr;
    }

    // Extract position covariance (2x2 submatrix)
    float var_x = covariance[0][0].item<float>();
    float var_y = covariance[1][1].item<float>();
    float cov_xy = covariance[0][1].item<float>();

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

    // Robot position in mm
    float robot_x_mm = robot_pose[0] * 1000.0f;
    float robot_y_mm = robot_pose[1] * 1000.0f;

    // Create ellipse (centered at robot)
    auto ellipse = scene->addEllipse(
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

/**************************************/
// From the RoboCompLidar3D you can call this methods:
// RoboCompLidar3D::TColorCloudData this->lidar3d_proxy->getColorCloudData()
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarData(string name, float start, float len, int decimationDegreeFactor)
// RoboCompLidar3D::TDataImage this->lidar3d_proxy->getLidarDataArrayProyectedInImage(string name)
// RoboCompLidar3D::TDataCategory this->lidar3d_proxy->getLidarDataByCategory(TCategories categories, long timestamp)
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarDataProyectedInImage(string name)
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarDataWithThreshold2d(string name, float distance, int decimationDegreeFactor)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData
// RoboCompLidar3D::TDataCategory
// RoboCompLidar3D::TColorCloudData

/**************************************/
// From the RoboCompOmniRobot you can call this methods:
// RoboCompOmniRobot::void this->omnirobot_proxy->correctOdometer(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->getBasePose(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->getBaseState(RoboCompGenericBase::TBaseState state)
// RoboCompOmniRobot::void this->omnirobot_proxy->resetOdometer()
// RoboCompOmniRobot::void this->omnirobot_proxy->setOdometer(RoboCompGenericBase::TBaseState state)
// RoboCompOmniRobot::void this->omnirobot_proxy->setOdometerPose(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->setSpeedBase(float advx, float advz, float rot)
// RoboCompOmniRobot::void this->omnirobot_proxy->stopBase()

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams
