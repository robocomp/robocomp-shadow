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

#include <cppitertools/enumerate.hpp>
#include <cppitertools/groupby.hpp>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
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
		if (error.length() > 0){
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
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU.\n";
	} else {
		std::cout << "Training on CPU.\n";
	}

	// Viewer
	viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
	auto [r, e] = viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
	robot_draw = r;

	viewer_room = new AbstractGraphicViewer(this->frame_room, params.GRID_MAX_DIM);
	auto [rr, re] = viewer_room->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
	robot_room_draw = rr;
	this->resize(900, 600);
	show();

	// initialise robot pose
	robot_pose_final.setIdentity();
	robot_pose_final.translate(Eigen::Vector2f(0.0,0.0));

	// Initialize RoomModel
	// Compute initial guess for room SIZE from point cloud bounds
	auto points = read_data();
	if (points.empty()) { std::cout << "No LiDAR points available\n"; return;}
	draw_lidar(points, &viewer->scene);

	// Convert LiDAR points to PyTorch tensor [N, 2]
	// Keep points in ROBOT FRAME - the model will transform them
	std::vector<float> points_data;
	points_data.reserve(points.size() * 2);
	for (const auto& p : points) {
		points_data.push_back(p.x / 1000.0f);  // Convert mm to meters
		points_data.push_back(p.y / 1000.0f);
	}
	torch::Tensor points_tensor = torch::from_blob
	(
		points_data.data(),
		{static_cast<long>(points.size()), 2},
		torch::kFloat32
	).clone();
	const auto x_coords = points_tensor.index({torch::indexing::Slice(), 0});
	const auto y_coords = points_tensor.index({torch::indexing::Slice(), 1});

	const float x_min = x_coords.min().item<float>();
	const float x_max = x_coords.max().item<float>();
	const float y_min = y_coords.min().item<float>();
	const float y_max = y_coords.max().item<float>();

	// Initial room size (room will be at origin by definition)
	const float half_width = (x_max - x_min) / 2.0f;
	const float half_height = (y_max - y_min) / 2.0f;

	// Initial robot pose (offset from room center to point cloud center)
	// If point cloud is centered at (cx, cy), robot is at (-cx, -cy) in room frame
	const float point_cloud_center_x = (x_min + x_max) / 2.0f;
	const float point_cloud_center_y = (y_min + y_max) / 2.0f;
	const float robot_x = -point_cloud_center_x;  // Negative because room is at origin
	const float robot_y = -point_cloud_center_y;
	const float robot_theta = 1.0f;

	std::cout << "\nInitial guess:\n";
	std::cout << "  Point cloud bounds: X[" << x_min << ", " << x_max << "], Y[" << y_min << ", " << y_max << "]\n";
	std::cout << "  Room size (at origin): " << 2*half_width << " x " << 2*half_height << " m\n";
	std::cout << "  Robot pose (relative to room): (" << robot_x << ", " << robot_y << ", " << robot_theta << ")\n";
	std::cout << "----------------------------------------\n";

	// Create room model with FIXED room at origin (5 parameters)
	room.init(half_width, half_height, robot_x, robot_y, robot_theta);

	// Room freezing manager
	RoomFreezingManager::Params freezing_params;

	// time series plotter for match error
	TimeSeriesPlotter::Config plotConfig;  // all fields have to be initialized, otherwise garbage values get to the constructor
	plotConfig.title = "Maximum Match Error Over Time";
	plotConfig.yAxisLabel = "Error (mm)";
	plotConfig.xAxisLabel = "";
	plotConfig.timeWindowSeconds = 7.0; // Show a 15-second window
	plotConfig.autoScaleY = true;       // We will set a fixed range
	plotConfig.showLegend = false;            // Show graph legend
	plotConfig.yMin = 0;
	plotConfig.yMax = 0.4;
	time_series_plotter = std::make_unique<TimeSeriesPlotter>(frame_plot_error, plotConfig);
	time_series_plotter->addGraph("", Qt::blue);

}

void SpecificWorker::compute()
{
	// Read LiDAR data (in robot frame)
	const auto points = read_data();
	if (points.empty()) { std::cout << "No LiDAR points available\n"; return;}
	draw_lidar(points, &viewer->scene);
	door_detector.draw_doors(false, &viewer->scene, &viewer_room->scene, robot_pose_final);

	optimize_room_and_robot(points);

	// update robot pose and draw robot in viewer
	const auto robot_pose = room.get_robot_pose();
	robot_pose_final.translation() = Eigen::Vector2f(robot_pose[0], robot_pose[1]);
	robot_pose_final.linear() = Eigen::Rotation2Df(robot_pose[2]).toRotationMatrix();

	update_viewers();

	last_time = std::chrono::high_resolution_clock::now();;
}


///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::optimize_room_and_robot(const RoboCompLidar3D::TPoints &points)
{
	if (points.empty()) return;

	// Convert LiDAR points to PyTorch tensor [N, 2]
	// Keep points in ROBOT FRAME - the model will transform them
	std::vector<float> points_data;
	points_data.reserve(points.size() * 2);
	for (const auto& p : points) {
		points_data.push_back(p.x / 1000.0f);  // Convert mm to meters
		points_data.push_back(p.y / 1000.0f);
	}

	torch::Tensor points_tensor = torch::from_blob
	(
		points_data.data(),
		{static_cast<long>(points.size()), 2},
		torch::kFloat32
	).clone();

	// ADAPTIVE PARAMETER SELECTION
	std::vector<torch::Tensor> params_to_optimize;
	bool is_localized = room_freezing_manager.should_freeze_room();

	if (room_freezing_manager.should_freeze_room())
	{
		// Only optimize robot pose
		params_to_optimize = room.get_robot_parameters();
		room.freeze_room_parameters();
		qInfo() << "🔒 LOCALIZED: Optimizing robot pose only";
	} else
	{
		// Optimize everything
		params_to_optimize = room.parameters();
		room.unfreeze_room_parameters();
		qInfo() << "🗺️  MAPPING: Optimizing room + robot pose";
	}

	// Setup optimizer
	torch::optim::Adam optimizer(params_to_optimize, torch::optim::AdamOptions(0.01));

	// Optimization loop
	const int num_iterations = 150;
	float final_loss = 0.0f;
	const int print_every = 30;
	// Set a threshold for early stopping (e.g., for Huber loss)
	const float min_loss_threshold = 0.01f;

	for (int iter = 0; iter < num_iterations; ++iter)
	{
		optimizer.zero_grad();

		// Compute loss (points are in robot frame)
		torch::Tensor loss = RoomLoss::compute_loss(points_tensor, room, 0.1f);

		// Backward pass
		loss.backward();

		// Update parameters
		optimizer.step();

		if (iter == num_iterations - 1)
			final_loss = loss.item<float>();
		const bool is_last_iter = (iter == num_iterations - 1);
		time_series_plotter->addDataPoint(0, loss.item<double>());

		// early stopping
		if (loss.item<float>() < min_loss_threshold)
		{
			std::cout << "  Stopping early: Loss (" << std::fixed << std::setprecision(6)
					  << loss.item<float>() << ") is below threshold (" << min_loss_threshold << ")" << " iter: " << iter << "\n";

			// Print the final state if it wasn't just printed
			if (iter % print_every != 0 && !is_last_iter)
			{
				const auto robot_pose = room.get_robot_pose();
				std::cout << "  Final State " << std::setw(3) << iter
						  << " | Loss: " << std::fixed << std::setprecision(6)
						  << loss.item<float>()
						  << " | Robot: (" << std::setprecision(2)
						  << robot_pose[0] << ", " << robot_pose[1] << ", "
						  << robot_pose[2] << ")\n";
			}
			break; // Exit the loop
		}
		if (iter % print_every == 0 || iter == num_iterations - 1)
		{
			auto robot_pose = room.get_robot_pose();
			// std::cout << "  Iteration " << std::setw(3) << iter
			// 		  << " | Loss: " << std::fixed << std::setprecision(6)
			// 		  << loss.item<float>()
			// 		  << " | Robot: (" << std::setprecision(2)
			// 		  << robot_pose[0] << ", " << robot_pose[1] << ", "
			// 		  << robot_pose[2] << ")\n";
		}
	}

	// Print final result
	// std::cout << "\n========================================\n";
	// std::cout << "Optimization completed!\n";
	// std::cout << "========================================\n";
	// room.print_info();

	// Compute uncertainty (now 5×5 covariance matrix)
	// std::cout << "\nComputing uncertainty (Laplace approximation)...\n";
	//torch::Tensor covariance = UncertaintyEstimator::compute_covariance(points_tensor, room, 0.1f);
	//UncertaintyEstimator::print_uncertainty(covariance, room);

	///  UPDATE FREEZING MANAGER
	// Compute uncertainties
	const torch::Tensor covariance = UncertaintyEstimator::compute_covariance (points_tensor, room, 0.1f);
	const auto std_devs = UncertaintyEstimator::get_std_devs(covariance);

	std::vector<float> room_std_devs;
	std::vector<float> robot_std_devs;

	if (is_localized)
	{
		// LOCALIZED: covariance is 3x3 (robot only)
		// std_devs = [robot_x_std, robot_y_std, robot_theta_std]
		robot_std_devs = {std_devs[0], std_devs[1], std_devs[2]};

		// Room uncertainty is zero (frozen)
		room_std_devs = {0.0f, 0.0f};

		qInfo() << "Robot uncertainty: X=" << robot_std_devs[0]
				<< " Y=" << robot_std_devs[1]
				<< " θ=" << robot_std_devs[2];
	}
	else
	{
		// MAPPING: covariance is 5x5 (room + robot)
		// std_devs = [half_width_std, half_height_std, robot_x_std, robot_y_std, robot_theta_std]
		room_std_devs = {std_devs[0], std_devs[1]};
		robot_std_devs = {std_devs[2], std_devs[3], std_devs[4]};

		qInfo() << "Room uncertainty: W=" << room_std_devs[0]
				<< " H=" << room_std_devs[1];
		qInfo() << "Robot uncertainty: X=" << robot_std_devs[0]
				<< " Y=" << robot_std_devs[1]
				<< " θ=" << robot_std_devs[2];
	}

	// =========================================================================
	// UPDATE FREEZING MANAGER
	// =========================================================================

	const auto room_params = room.get_room_parameters();
	const bool state_changed = room_freezing_manager.update(
		room_params,
		room_std_devs,
		robot_std_devs,
		room.get_robot_pose(),
		final_loss,
		num_iterations
	);

	if (state_changed) {
		room_freezing_manager.print_status();
		UncertaintyEstimator::print_uncertainty(covariance, room);
	}

	// viewer update
	update_viewers();

}
RoboCompLidar3D::TPoints SpecificWorker::read_data()
{
	RoboCompLidar3D::TData ldata;
	try
	{ ldata = lidar3d_proxy->getLidarData("helios", 0, 2 * M_PI, 2);}
	catch (const Ice::Exception &e) { std::cout << e << " " << "No lidar data from sensor" << std::endl; return {};}
	if (ldata.points.empty()) return {};

	//filter out invalid points (z <= 1000 or z > 2000)
	RoboCompLidar3D::TPoints valid_points;
	valid_points.reserve(ldata.points.size());
	for (const auto &p : ldata.points)
		if (p.z > 1000 and p.z <= 2000)
			valid_points.push_back(p);
	ldata.points = std::move(valid_points);

	// // compute the mean and variance of points and reject outliers beyond 3 stddevs
	// // Compute mean and variance (Welford) only over valid z (>1000)
	double mean = 0.0;
	double M2 = 0.0;
	std::size_t count = 0;
	for (const auto &p : ldata.points)
	{
		if (p.z <= 1000 or p.z > 2000) continue; // keep same validity criterion
		++count;
		const double delta = p.r - mean;
		mean += delta / static_cast<double>(count);
		M2 += delta * (p.r - mean);
	}

	if (count == 0) { return {};}

	// population variance to match original behavior (divide by N)
	const double var = M2 / static_cast<double>(count);
	const double stddev = std::sqrt(var);
	if (stddev == 0.0)
	{ qInfo() << __FUNCTION__ << "Zero variance in range data, all points have same r: " << mean; return {};}

	// filter out points with r beyond 3 stddevs (apply only if stddev > 0)
	RoboCompLidar3D::TPoints no_outliers;
	no_outliers.reserve(ldata.points.size());
	const double threshold = 2.0 * stddev;
	for (const auto &p : ldata.points)
		if (std::fabs(p.r - mean) <= threshold)
			no_outliers.push_back(p);

	// qInfo() << __FUNCTION__ << "Filtered out "
	// 		<< (ldata.points.size() - no_outliers.size())
	// 		<< " outliers based on range statistics (mean: "
	// 		<< mean << ", stddev: " << stddev << ")";
	ldata.points = std::move(no_outliers);

	//ldata.points = filter_same_phi(ldata.points);
	//return filter_isolated_points(ldata.points, 200);

	// Use door detector to filter points (removes points near detected doors)
	ldata.points = door_detector.filter_points(ldata.points);

	return ldata.points;
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
	auto res = std::ranges::find_if(points, [angle](auto &a){ return a.phi > angle;});
	if(res != std::end(points))
		return std::distance(std::begin(points), res);
	else
		return std::unexpected("No closest value found in method <closest_lidar_index_to_given_angle>");
}
RoboCompLidar3D::TPoints SpecificWorker::filter_same_phi(const RoboCompLidar3D::TPoints& points)
{
	if (points.empty())
		return {};

	RoboCompLidar3D::TPoints result; result.reserve(points.size());

	for (auto&& [angle, group]: iter::groupby(points, [](const auto& p)
	{
		float multiplier=std::pow(10.0f, 2); return std::floor(p.phi*multiplier)/multiplier;
	})) {
		auto min=std::min_element(std::begin(group), std::end(group), [](const auto& a, const auto& b){return a.r<b.r;});
		result.emplace_back(*min);
	}
	return result;
}
void SpecificWorker::draw_lidar(const RoboCompLidar3D::TPoints &filtered_points, QGraphicsScene *scene)
{
	static std::vector<QGraphicsItem*> items;   // store items so they can be shown between iterations

	// remove all items drawn in the previous iteration
	for(const auto i: items)
	{
		scene->removeItem(i);
		delete i;
	}
	items.clear();

	const auto color = QColor(Qt::darkGreen);
	const auto brush = QBrush(QColor(Qt::darkGreen));
	for(const auto &p : filtered_points)
	{
		const auto item = scene->addRect(-25, -25, 50, 50, color, brush);
		item->setPos(p.x, p.y);
		items.push_back(item);
	}
}
void SpecificWorker::update_viewers()
{
	robot_room_draw->setPos(robot_pose_final.translation().x()*1000, robot_pose_final.translation().y()*1000);
	const double angle = qRadiansToDegrees(std::atan2(robot_pose_final.rotation()(1, 0), robot_pose_final.rotation()(0, 0)));
	robot_room_draw->setRotation(angle);
	// draw room in viewer_room given the current robot pose
	if (room_draw != nullptr) { viewer_room->scene.removeItem(room_draw); delete room_draw;}
	room_draw = viewer_room->scene.addRect(-room.get_room_parameters()[0]*1000,
	                                       -room.get_room_parameters()[1]*1000,
	                                        room.get_room_parameters()[0]*2*1000,
	                                        room.get_room_parameters()[1]*2*1000,
	                                        QPen(QColor(200,200,200), 30));

	// draw room in robot viewer
	if (room_draw_robot != nullptr) { viewer->scene.removeItem(room_draw_robot); delete room_draw_robot;}
	// compute room corners in robot frame
	Eigen::Vector2f top_left(-room.get_room_parameters()[0], room.get_room_parameters()[1]);
	Eigen::Vector2f top_right(room.get_room_parameters()[0], room.get_room_parameters()[1]);
	Eigen::Vector2f bottom_left(-room.get_room_parameters()[0], -room.get_room_parameters()[1]);
	Eigen::Vector2f bottom_right(room.get_room_parameters()[0], -room.get_room_parameters()[1]);
	Eigen::Matrix2f R = robot_pose_final.rotation().transpose();
	Eigen::Vector2f t = -R * robot_pose_final.translation();
	top_left = R * top_left + t;
	top_right = R * top_right + t;
	bottom_left = R * bottom_left + t;
	bottom_right = R * bottom_right + t;
	QPolygonF polygon;
	polygon << QPointF(top_left.x()*1000, top_left.y()*1000)
	        << QPointF(top_right.x()*1000, top_right.y()*1000)
	        << QPointF(bottom_right.x()*1000, bottom_right.y()*1000)
	        << QPointF(bottom_left.x()*1000, bottom_left.y()*1000);
	room_draw_robot = viewer->scene.addPolygon(polygon, QPen(QColor("cyan"), 30));

	// update GUI
	time_series_plotter->update();
	//lcdNumber_adv->display(adv);
	//lcdNumber_rot->display(rot);
	lcdNumber_x->display(robot_pose_final.translation().x());
	lcdNumber_y->display(robot_pose_final.translation().y());
	//lcdNumber_room->display(current_room);
	lcdNumber_angle->display(qRadiansToDegrees(angle));
	label_state->setText(room_freezing_manager.state_to_string(room_freezing_manager.get_state()).data());
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