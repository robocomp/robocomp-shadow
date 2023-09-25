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
#include "fastgicp.h"
#include <pcl/common/io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

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

class SpecificWorker : public GenericWorker
{
	Q_OBJECT
	public:
		SpecificWorker(TuplePrx tprx, bool startup_check);
		~SpecificWorker();
		bool setParams(RoboCompCommonBehavior::ParameterList params);

		void OmniRobot_correctOdometer(int x, int z, float alpha);
		void OmniRobot_getBasePose(int &x, int &z, float &alpha);
		void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state);
		void OmniRobot_resetOdometer();
		void OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state);
		void OmniRobot_setOdometerPose(int x, int z, float alpha);
		void OmniRobot_setSpeedBase(float advx, float advz, float rot);
		void OmniRobot_stopBase();

		void SegmentatorTrackingPub_setTrack(RoboCompVisualElements::TObject target);

	public slots:
		void compute();
		int startup_check();
		void initialize(int period);
        void new_mouse_coordinates(QPointF);

	private:
		bool startup_check_flag;
		int DEGREES_NUMBER = 360;		// division of the circle
		int OUTER_RIG_DISTANCE = 1000;  // external maximum reach to search (mm) when subsampling the robot contourn
        int BAND_WIDTH = 300;			// distance to the obstacle that repels the object

        // Structs
        struct Band
        {
            float frontal_distance = 300.0f;
            float back_distance = 300.0f;
            float left_distance = 300.0f;
            float right_distance = 300.0f;
        };
        struct Robot_speed
        {
            float adv = 0.0f;
            float rot = 0.0f;
            float side = 0.0f;
        };
        struct Block
        {
            float A_ang = 0.0f;
            float A_dist = 0.0f;
            float B_ang = 0.0f;
            float B_dist = 0.0f;
            bool concave = true;
            float dist() const { return (A_dist+B_dist)/2.f; };
        };
        struct LPoint
        {
            float ang = 0.0f;
            float dist = 0.0f;
            int block;
            bool concave;
            bool kinematics;
            float coste = 0.f;
        };
        struct Target
        {
            float x = 0.f, y = 0.f, ang = 0,f, dist = 0.f;  // target
            float side, adv, rot; // assigned speed
            bool active = false;
            void set(float x_, float y_, float rot_)
                { x=x_; y=y_; ang = atan2(x, y); dist = sqrt(x*x+y*y); active = true;
                  side = x; adv = y; rot = rot_;};
            void print() const {
                qInfo() << "Target:";
                qInfo() << "    side:" << side << "adv:" << adv << "rot:" << rot;
                qInfo() << "    x:" << x << "y:" << y << "ang:" << ang;
            }
            float distance_to_go(float px, float py)
            {
                return sqrt((x-px)*(x-px) + (y-py)*(y-py));
            }
        };

   		//int z_lidar_height = 750;
        std::vector<float> create_map_of_points();
		std::vector<float> map_of_points;

        //Struct declaration
        Band band;
        Robot_speed robot_speed;
        Target target;

        // Viewer
		AbstractGraphicViewer *viewer;
        QPolygonF robot_contour, robot_safe_band;

        // BW pose for drawing
        QGraphicsRectItem* rectItem;

		// Thread variable
		std::thread read_lidar_th;

        // DoubleBuffer variable
		DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

        // Robot speed gains
        float x_gain = 20;
        float y_gain = 8;
        bool robot_stop = false;
		const float max_adv = 500;
		const float max_side = 500;
        const float R = 250;

        FastGICP fastgicp;

		// Methods
        // Draw method
        void draw_ring_points(const vector<QPointF> &points, const Eigen::Vector2f &result, QGraphicsScene *scene);
        void draw_ring(const vector<float> &dists, QGraphicsScene *scene);
        void draw_discr_points(const std::vector<std::tuple<float, float>> &discr_points, QGraphicsScene *scene);
        void draw_enlarged_points(const std::vector<std::tuple<float, float>> &enlarged_points, QGraphicsScene *scene);
        void draw_blocks(const std::vector<Block> &blocks, QGraphicsScene *scene);
        void draw_band_width(QGraphicsScene *scene);

        // Thread method
        void read_lidar();
        Band adjustSafetyZone(Eigen::Vector3f velocity);
        std::vector<Eigen::Vector3f> filterPointsInRectangle(const std::vector<Eigen::Vector3f>& points);

        // Timing
        void self_adjust_period(int new_period);
        FPSCounter fps;

        // Double Buffer 
        DoubleBuffer<std::tuple<float, float, float>,std::tuple<float, float, float>> buffer_dwa;

        // Processing steps
        std::vector<tuple<float, float>> discretize_lidar(const RoboCompLidar3D::TPoints &ldata);
        std::vector<tuple<float, float>> configuration_space(const std::vector<std::tuple<float, float>> &points);
        std::pair<std::vector<Block>, std::vector<LPoint>>
            get_blocks(const std::vector<std::tuple<float, float>> &enlarged_points);
        std::vector<Block> set_blocks_symbol(const std::vector<SpecificWorker::Block> &blocks);
        LPoint cost_function(const std::vector<LPoint> &points, const std::vector<Block> &blocks, const Target &target);

        void repulsion_force(const RoboCompLidar3D::TData &ldata);
        void draw_result(const LPoint &res);
        void stop_robot();
        void draw_target(const Target &t, bool erase=false);
        bool inside_contour(const Target &target, const std::vector<LPoint> &contour);

        void draw_target_original(const Target &t);
        void draw_target(double x, double y, bool erase=false);
        Eigen::Vector2f robot_current_speed = {0.f, 0.f};

        LPoint check_safety(const vector<LPoint> &points);
};

#endif
