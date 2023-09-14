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

	private:
		bool startup_check_flag;
		int DEGREES_NUMBER = 360;		// division of the circle
		int OUTER_RIG_DISTANCE = 1000;  // external maximum reach to search (mm) when subsampling the robot contourn
        int BAND_WIDTH = 400;			// distance to the obstacle that repels the object

        // Struct that correspond to the band witdh of the robot
        struct Band
        {
            float frontal_distance = 400.0f;
            float back_distance = 400.0f;
            float left_distance = 400.0f;
            float right_distance = 400.0f;
        };


        struct
        {
            float adv_speed = 0.0f;
            float rot_speed = 0.0f;
            float side_speed = 0.0f;

        }robot_speed;

   		//int z_lidar_height = 750;
        std::vector<float> create_map_of_points();
		std::vector<float> map_of_points;
        Band band;

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

		// Methods
        // Draw method
        void draw_ring_points(const vector<QPointF> &points, const Eigen::Vector2f &result, QGraphicsScene *scene);
        void draw_ring(const vector<float> &dists, QGraphicsScene *scene);
        void draw_all_points(const RoboCompLidar3D::TPoints &points, const Eigen::Vector2f &result, QGraphicsScene *scene);
        void draw_band_width(QGraphicsScene *scene);
        void draw_histogram(const RoboCompLidar3D::TPoints &ldata);

        // Thread method
        void read_lidar();
        Band adjustSafetyZone(Eigen::Vector3f velocity);
        std::vector<Eigen::Vector3f> filterPointsInRectangle(const std::vector<Eigen::Vector3f>& points);

        // Timing
        void self_adjust_period(int new_period);
        FPSCounter fps;
        // Double Buffer 
        DoubleBuffer<std::tuple<float, float, float>,std::tuple<float, float, float>> buffer_dwa;
};

#endif
