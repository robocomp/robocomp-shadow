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


// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include "time_series_plotter.h"
#include <genericworker.h>
#include <expected>
#include <vector>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <Eigen/Dense>
#include "room_model.h"
#include "door_detector.h"
#include "room_freezing_manager.h"
#include "qt3d_visualizer.h"
#include "room_optimizer.h"

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
			float ADVANCE_THRESHOLD = ROBOT_WIDTH * 3; // mm
			float LIDAR_FRONT_SECTION = 0.2; // rads, aprox 12 degrees
			// wall
			float LIDAR_RIGHT_SIDE_SECTION = M_PI/3; // rads, 90 degrees
			float LIDAR_LEFT_SIDE_SECTION = -M_PI/3; // rads, 90 degrees
			float WALL_MIN_DISTANCE = ROBOT_WIDTH*1.2;
			// match error correction
			float MATCH_ERROR_SIGMA = 150.f; // mm
			float DOOR_REACHED_DIST = 300.f;
			std::string LIDAR_NAME_LOW = "bpearl";
			std::string LIDAR_NAME_HIGH = "helios";
			QRectF GRID_MAX_DIM{-5000, 2500, 10000, -5000};
			float NOISE_TRANS = 0.02f;  // 2cm stddev per meter
			float NOISE_ROT = 0.1f;     // 0.1 rad
		};
		Params params;

		// velocity commands
		boost::circular_buffer<VelocityCommand> velocity_history_{10}; // Keep last 10 commands
		OdometryPrior compute_odometry_prior() const;
		Eigen::Vector3f integrate_velocity(const VelocityCommand& cmd, float dt) const;

		// viewer
		AbstractGraphicViewer *viewer, *viewer_room;
		QGraphicsPolygonItem *robot_draw, *robot_room_draw;
		QGraphicsItem *room_draw = nullptr, *room_draw_robot = nullptr;

		// robot
		Eigen::Affine2f robot_pose_final;

		// room
		RoomModel room;
		//RoomFreezingManager room_freezing_manager;

		// aux
		RoboCompLidar3D::TPoints read_data();
		void draw_lidar(const RoboCompLidar3D::TPoints &filtered_points, QGraphicsScene *scene);

		void update_viewers();

		std::expected<int, std::string> closest_lidar_index_to_given_angle(const auto &points, float angle);
		RoboCompLidar3D::TPoints filter_same_phi(const RoboCompLidar3D::TPoints &points);
		RoboCompLidar3D::TPoints filter_isolated_points(const RoboCompLidar3D::TPoints &points, float d);
		inline QPointF to_qpointf(const Eigen::Vector2f &v) const
        { return QPointF(v.x(), v.y()); }
		// random number generator
		std::random_device rd;

		// timing
		std::chrono::time_point<std::chrono::high_resolution_clock> last_time = std::chrono::high_resolution_clock::now();

		// plotter
		std::shared_ptr<TimeSeriesPlotter> time_series_plotter;

		// door
		DoorDetector door_detector;

		// qt3d
		std::unique_ptr<RoomVisualizer3D> viewer3d;

		// optimizer
		RoomOptimizer optimizer;

	Q_SIGNALS:
		//void customSignal();
};

#endif
