/*
 *    Copyright (C) 2023 by YOUR NAME HERE
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
#define DEBUG 0

#include <genericworker.h>
#include <Eigen/Dense>

#define QT6
#ifdef QT5
    #include <abstract_graphic_viewer_qt5/abstract_graphic_viewer.h>
#else
    #include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#endif

#include <fps/fps.h>
#include <math.h>
#include <doublebuffer/DoubleBuffer.h>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <timer/timer.h>

class SpecificWorker : public GenericWorker
{
	Q_OBJECT
	public:
		SpecificWorker(TuplePrx tprx, bool startup_check);
		~SpecificWorker();
		bool setParams(RoboCompCommonBehavior::ParameterList params);
		RoboCompGridPlanner::TPlan GridPlanner_modifyPlan(RoboCompGridPlanner::TPlan plan);   // not used
        void GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan);
        void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);

public slots:
		void compute();
		int startup_check();
		void initialize(int period);
        void new_mouse_coordinates(QPointF);

	private:
		bool startup_check_flag;

        // Parameters
        struct Params
        {
            bool DISPLAY = false;               //
            bool REACTION = true;               // true -> reaction to the bumper
            float OUTER_RIG_DISTANCE = 1500.f;  // external maximum reach to search (mm) when subsampling the robot contourn
            float MIN_BAND_WIDTH = 25.f;		// minimum distance to the obstacle that repels the object
            float MAX_BAND_WIDTH = 250.f;		// maximum distance to the obstacle that repels the object
            double BELT_ANGULAR_STEP = 0.1f;    // angular step to create the belt
            float BELT_LINEAR_STEP = 30.f;      // linear step to create the belt
            float MAX_DIST_TO_LOOK_AHEAD = MAX_BAND_WIDTH;  // mm in search of a valid displacement to free the bumper
            float ROBOT_WIDTH = 460;  // mm
            float ROBOT_LENGTH = 480;  // mm
            float ROBOT_SEMI_WIDTH = ROBOT_WIDTH / 2.f;     // mm
            float ROBOT_SEMI_LENGTH = ROBOT_LENGTH / 2.f;    // mm
            float MAX_ADV_SPEED = 1500;    // mm/s
            float MAX_BACKWARDS_ADV_SPEED = 200;    // mm/s
            float MAX_SIDE_SPEED = 0;   // mm/s
            float MAX_ROT_SPEED = 2.0;  // rad/s
            std::string LIDAR_NAME = "bpearl";
            float MAX_LIDAR_RANGE = 10000;  // mm
            float LIDAR_DECIMATION_FACTOR = 1;
            QRectF viewer_dim{-3000, -3000, 6000, 6000};
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            float REPULSION_GAIN = 10.f;
            int PERIOD = 50;    // ms (20 Hz) for compute timer
            float LAMBDA_GAIN = 0.2f;   // gain to split contributions between the bumper and the target. 1 -> target
            bool ENABLE_JOYSTICk = true;
        };
        Params params;

        struct Target
        {
            float x = 0.f, y = 0.f, ang = 0.f;  // target
            float side{}, adv{}, rot{}; // assigned speed
            bool active = false;
            void set(float x_, float y_, float rot_)
            {
                x=x_; y=y_; ang = atan2(x, y); active = true;
                side = x; adv = y; rot = rot_;
            };

            void print(QString txt="") const
            {
                qInfo() << "Target " << txt;
                qInfo() << "    side:" << side << "adv:" << adv << "rot:" << rot;
                qInfo() << "    x:" << x << "y:" << y << "ang:" << ang;
            }
            float distance_to_go(float px, float py)
            {
                return std::hypot(x-px, y-py);
            }
            Eigen::Vector2f eigen() const
            {
                return Eigen::Vector2f{x, y};
            }
        };

   		// Outer rig
        std::vector<Eigen::Vector2f> create_edge_points(const QPolygonF &robot_safe_band);  // creates the outer_rig in polar coordinates
		std::vector<Eigen::Vector2f> edge_points;   // outer_rig in polar coordinates

        // Targets
        Target target; // holds the current target to be reached
        Target reaction;  // local target generated by the bumper

        // Viewer
		AbstractGraphicViewer *viewer;
        QPolygonF robot_contour, robot_safe_band;

        Eigen::Vector3f robot_current_speed = {0.f, 0.f, 0.f}; // side, adv, rot
        bool robot_stopped = false;

        // Draw method
        void draw_edge_points(const vector<QPointF> &points, const Eigen::Vector2f &result, QGraphicsScene *scene);
        void draw_edge(const std::vector<Eigen::Vector2f> &edge_points, QGraphicsScene *scene);
        void draw_target(const Target &t, bool erase=false);
        void draw_target_original(const Target &t, bool erase = false, float scale=1);
        void draw_target(double x, double y, bool erase=false);
        void draw_target_breach(const Target &t, bool erase = false);
        void draw_points_in_belt(const std::vector<Eigen::Vector2f> &points_in_belt);
        void draw_displacements(std::vector<Eigen::Matrix<float, 2, 1>> displacement_points, QGraphicsScene *scene);
        void draw_lidar(const RoboCompLidar3D::TPoints &points);
        void draw_robot_contour(const QPolygonF &robot_contour, const QPolygonF &robot_safe_band, QGraphicsScene *pScene);

        // Lidar Thread
        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // Timing
        void self_adjust_period(int new_period);
        FPSCounter fps;
        rc::Timer<> clock;

        // Double Buffer 
        //DoubleBuffer<std::tuple<float, float, float, bool>,std::tuple<float, float, float, bool>> buffer_target;
        DoubleBuffer<RoboCompGridPlanner::TPlan, RoboCompGridPlanner::TPlan> buffer_target;

        // Processing steps
        //void stop_robot(const std::string_view txt);
        std::vector<Eigen::Vector2f> check_safety(const RoboCompLidar3D::TPoints &points);
        QPolygonF adjust_band_size(const Eigen::Vector3f &velocity);
        void move_robot(const Target &target, const Target &reaction, bool stop = false);
        void target_active_and_security_breach(const vector<Eigen::Vector2f> &displacements);
        void not_target_active_and_not_security_breach(const vector<Eigen::Vector2f> &displacements);
        void not_target_active_and_security_breach(const vector<Eigen::Vector2f> &displacements);
};

#endif
