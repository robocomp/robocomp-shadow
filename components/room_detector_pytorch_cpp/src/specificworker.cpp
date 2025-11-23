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
	this->resize(1000, 800);
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

	// loss plotter for match error
	TimeSeriesPlotter::Config plotConfig;
	// all fields have to be initialized, otherwise garbage values get to the constructor
	plotConfig.title = "Loss (likelihood + prior)";
	plotConfig.yAxisLabel = "Error";
	plotConfig.xAxisLabel = "";
	plotConfig.timeWindowSeconds = 7.0; // Show a 15-second window
	plotConfig.autoScaleY = true; // We will set a fixed range
	plotConfig.showLegend = true; // Show graph legend
	plotConfig.yMin = 0;
	plotConfig.yMax = 0.;
	loss_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error_1, plotConfig);
	graphs.push_back(loss_plotter->addGraph("post", Qt::blue));
	graphs.push_back(loss_plotter->addGraph("like", Qt::green));
	graphs.push_back(loss_plotter->addGraph("prior", Qt::red));

	plotConfig.title = "standard deviation";
	plotConfig.yAxisLabel = "std";
	stddev_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error_2, plotConfig);
	// stddev_plotter->addGraph("prop_x", Qt::red); // Orange
	// stddev_plotter->addGraph("prop_y", Qt::magenta); // Orange
	// stddev_plotter->addGraph("prop_theta", Qt::green); // Orange
	stddev_plotter->addGraph("upd_x", Qt::blue);
	stddev_plotter->addGraph("upd_y", Qt::cyan);
	stddev_plotter->addGraph("upd_theta", Qt::red);

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
	                                       100,
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
    update_robot_view(robot_pose_display, result, scene);

	// Detect and draw doors
    door_detector.draw_doors(false, scene, nullptr, robot_pose_display);

	//viewer3d->updatePointCloud(points);
	const auto room_params = room.get_room_parameters();
	viewer3d->updateRoom(room_params[0], room_params[1]); // half-width, half-height
	viewer3d->updateRobotPose(room.get_robot_pose()[0], room.get_robot_pose()[1], room.get_robot_pose()[2]);

	// Time series plot
	loss_plotter->addDataPoint(0, result.final_loss);
	loss_plotter->addDataPoint(2, result.prior_loss);
	loss_plotter->addDataPoint(1, result.measurement_loss);

	// Plot propagated uncertainty (before measurement update)
	// if (result.propagated_cov.defined() && result.propagated_cov.numel() > 0)
	// {
	// 	const float prop_std_x = std::sqrt(result.propagated_cov[0][0].item<float>()) * 1000;
	// 	const float prop_std_y = std::sqrt(result.propagated_cov[1][1].item<float>()) * 1000;
	// 	const float prop_std_theta = std::sqrt(result.propagated_cov[1][1].item<float>()) * 1000;
	// 	stddev_plotter->addDataPoint(3, prop_std_x);  // Orange line
	// 	stddev_plotter->addDataPoint(4, prop_std_y);  // Orange line
	// 	stddev_plotter->addDataPoint(5, prop_std_theta); // Orange line
	// }

	// Plot updated uncertainty (after measurement update)
	const float upd_std_x = std::sqrt(result.covariance[0][0].item<float>()) ;
	const float upd_std_y = std::sqrt(result.covariance[1][1].item<float>()) ;
	const float upd_std_theta = std::sqrt(result.covariance[2][2].item<float>()) ;
	stddev_plotter->addDataPoint(0, upd_std_x);    // Green line
	stddev_plotter->addDataPoint(1, upd_std_y);    // Green line
	stddev_plotter->addDataPoint(2, upd_std_theta); // Green line

	loss_plotter->update();
	stddev_plotter->update();
}

void SpecificWorker::print_status(const RoomOptimizer::Result &result)
{
    static int frame_counter = 0;
    const auto &robot_pose = room.get_robot_pose();
    const auto &room_params = room.get_room_parameters();
    const bool is_localized = optimizer.room_freezing_manager.should_freeze_room();
    const auto &odom_prior = result.prior;

    // --- Header ---
    qInfo().noquote() << "\n" << QString(60, '=');
    qInfo().noquote() << QString("FRAME %1").arg(frame_counter++, 5).rightJustified(38);
    qInfo().noquote() << QString(60, '=');

    // --- System State ---
    qInfo().noquote() << "System State:" << optimizer.room_freezing_manager.get_state_string().data();
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

/*
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
	ldata.points = std::move(no_outliers); */

	//ldata.points = filter_same_phi(ldata.points);
	ldata.points = filter_isolated_points(ldata.points, 100);

	// Use door detector to filter points (removes points near detected doors)
	ldata.points = door_detector.filter_points(ldata.points);

	return {ldata.points,std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::milliseconds(ldata.timestamp))};
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

void SpecificWorker::update_robot_view(const Eigen::Affine2f &robot_pose,
                                       const RoomOptimizer::Result &result,
                                       QGraphicsScene *scene)
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
