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

	// viewer_room = new AbstractGraphicViewer(this->frame_room, params.GRID_MAX_DIM);
	// auto [rr, re] = viewer_room->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
	// robot_room_draw = rr;
	this->resize(1000, 600);
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

	//optimizer
	//optimizer.enable_motion_propagation(true);

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
	time_series_plotter = std::make_shared<TimeSeriesPlotter>(frame_plot_error, plotConfig);
	time_series_plotter->addGraph("", Qt::blue);

	// Create 3D viewer
	viewer3d = std::make_unique<RoomVisualizer3D>("src/meshes/shadow.obj");
	QWidget* viewer3d_widget = viewer3d->getWidget();
	QVBoxLayout* layout = new QVBoxLayout(frame_3d);  // Your QFrame name here
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(viewer3d_widget);
	viewer3d->show();
}

void SpecificWorker::compute()
{
	// Read LiDAR data (in robot frame)
	const auto points = read_data();
	auto current_time = std::chrono::high_resolution_clock::now();
	if (points.empty()) { std::cout << "No LiDAR points available\n"; return;}
	draw_lidar(points, &viewer->scene);
	door_detector.draw_doors(false, &viewer->scene, nullptr, robot_pose_final);

	// Compute odometry prior
	const auto odom_prior = compute_odometry_prior();

	// Optimize with prior
	const auto result = optimizer.optimize(points,
							   			 room,
											time_series_plotter,
											150,
											0.01f,
											0.01f,
											 odom_prior);

	// Propagate covariance using velocity (only if prior is valid)
	if (odom_prior.valid)
	{
		const auto dt = std::chrono::duration<float>(current_time - last_time).count();
		torch::Tensor propagated_cov = optimizer.uncertainty_manager.propagate_with_velocity(
			velocity_history_.back(), dt, result.covariance, room.are_room_parameters_frozen()
		);
		// Store propagated_cov for next iteration
		optimizer.uncertainty_manager.set_previous_cov(propagated_cov);
	}

	// update robot pose
	const auto robot_pose = room.get_robot_pose();
	robot_pose_final.translation() = Eigen::Vector2f(robot_pose[0], robot_pose[1]);
	robot_pose_final.linear() = Eigen::Rotation2Df(robot_pose[2]).toRotationMatrix();

	// update viewers
	update_viewers();
	//viewer3d->updatePointCloud(points);
	const auto room_params = room.get_room_parameters();
	viewer3d->updateRoom(room_params[0], room_params[1]); // half-width, half-height
	viewer3d->updateRobotPose(room.get_robot_pose()[0], room.get_robot_pose()[1], room.get_robot_pose()[2]);

	// Show uncertainty when localized
	if (room.are_room_parameters_frozen())
	{
		viewer3d->updateUncertainty(
			result.std_devs[0],  // X stddev
			result.std_devs[1],  // Y stddev
			result.std_devs[2]   // Theta stddev
		);
		viewer3d->showUncertainty(true);
	} else
		viewer3d->showUncertainty(false);  // Hide during mapping

	qDebug() << "Pose:" << robot_pose[0] << robot_pose[1] << robot_pose[2];
	qDebug() << "Uncertainty:" << result.std_devs[0] << result.std_devs[1] << result.std_devs[2];
	last_time = std::chrono::high_resolution_clock::now();;
}

///////////////////////////////////////////////////////////////////////////////
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
	//robot_room_draw->setPos(robot_pose_final.translation().x()*1000, robot_pose_final.translation().y()*1000);
	const double angle = qRadiansToDegrees(std::atan2(robot_pose_final.rotation()(1, 0), robot_pose_final.rotation()(0, 0)));
	//robot_room_draw->setRotation(angle);
	// draw room in viewer_room given the current robot pose
	// if (room_draw != nullptr) { viewer_room->scene.removeItem(room_draw); delete room_draw;}
	// room_draw = viewer_room->scene.addRect(-room.get_room_parameters()[0]*1000,
	//                                        -room.get_room_parameters()[1]*1000,
	//                                         room.get_room_parameters()[0]*2*1000,
	//                                         room.get_room_parameters()[1]*2*1000,
	//                                         QPen(QColor(200,200,200), 30));

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
	label_state->setText(optimizer.room_freezing_manager.state_to_string(optimizer.room_freezing_manager.get_state()).data());
}

Eigen::Vector3f SpecificWorker::integrate_velocity(const VelocityCommand& cmd, float dt) const
{
    // Integrate in robot's local frame
    float dx_local = (cmd.adv_x * dt) / 1000.0f;  // Convert mm → m
    float dy_local = (cmd.adv_z * dt) / 1000.0f;
    float dtheta   = cmd.rot * dt;

    // Transform to global frame using current robot heading
    float theta = std::atan2(robot_pose_final.rotation()(1, 0),
                             robot_pose_final.rotation()(0, 0));

    Eigen::Vector3f delta_global;
    delta_global[0] = dx_local * std::cos(theta) - dy_local * std::sin(theta);
    delta_global[1] = dx_local * std::sin(theta) + dy_local * std::cos(theta);
    delta_global[2] = dtheta;

    return delta_global;
}

OdometryPrior SpecificWorker::compute_odometry_prior() const
{
    OdometryPrior prior;
    prior.valid = false;

    if (velocity_history_.empty())
    {
        qWarning() << "No velocity commands in buffer";
        return prior;
    }

    // Use the most recent command (arrived just before LiDAR scan)
    const auto& latest_cmd = velocity_history_.back();

    // Time since last compute() call (typical RoboComp period: 10-100ms)
    static auto last_compute_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(current_time - last_compute_time).count();
    last_compute_time = current_time;

    // Reject if dt is unreasonable
    if (dt <= 0 || dt > 0.5f) {
        qWarning() << "Invalid dt for odometry:" << dt << "s";
        return prior;
    }

    // Integrate velocity to get pose delta
    prior.delta_pose = integrate_velocity(latest_cmd, dt);

    // Compute process noise covariance (proportional to command magnitude)
    float trans_noise_std = params.NOISE_TRANS *
        (std::abs(latest_cmd.adv_x) + std::abs(latest_cmd.adv_z)) / 1000.0f * dt;
    float rot_noise_std   = params.NOISE_ROT * std::abs(latest_cmd.rot) * dt;

    prior.covariance = torch::eye(3, torch::kFloat32);
    prior.covariance[0][0] = trans_noise_std * trans_noise_std;
    prior.covariance[1][1] = trans_noise_std * trans_noise_std;
    prior.covariance[2][2] = rot_noise_std * rot_noise_std;

    prior.valid = true;
    return prior;
}
///////////////////////////////////////////////////////////////////////////////
/////SUBSCRIPTION to sendData method from JoystickAdapter interface
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
	// Parse velocity command (assuming axes[0]=adv_x, axes[1]=adv_z, axes[2]=rot)
	VelocityCommand cmd;
	cmd.adv_x = data.axes[0].value;  //mm/sg
	cmd.adv_z = data.axes[1].value;
	cmd.rot = data.axes[2].value;
	cmd.timestamp = std::chrono::high_resolution_clock::now();
	velocity_history_.push_back(cmd);

	qDebug() << "Velocity stored:" << cmd.adv_x << "mm/s," << cmd.rot << "rad/s";
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