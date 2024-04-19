/*
 *    Copyright (C) 2024 by YOUR NAME HERE
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

#include <random>
#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <doublebuffer/DoubleBuffer.h>
#include "../../../agents/forcefield_agent/src/door_detector.h"
#include "room_detector.h"
#include "room.h"
#include <fps/fps.h>

#include <icp.h>
#include <timer/timer.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <timer/timer.h>



class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);
    using Line = std::vector<Eigen::Vector2f>;
    using Lines = std::vector<Line>;



public slots:
	void compute();
	int startup_check();
	void initialize(int period);
	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};
private:
	// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;
    std::unique_ptr<DSR::RT_API> rt;
    std::shared_ptr<DSR::InnerEigenAPI> inner_eigen;

	//DSR params
	std::string agent_name;
	int agent_id;

	bool tree_view;
	bool graph_view;
	bool qscene_2d_view;
	bool osg_3d_view;

	// DSR graph viewer
	std::unique_ptr<DSR::DSRViewer> graph_viewer;
	QHBoxLayout mainLayout;
	bool startup_check_flag;

    //local widget
    DSR::QScene2dViewer* widget_2d;

    struct Constants
    {
        std::string lidar_name = "helios";
        std::vector<std::pair<float, float>> ranges_list = {{1000, 2500}};
        bool DISPLAY = false;
    };
    Constants consts;
    struct Params
    {
        float ROBOT_WIDTH = 460;  // mm
        float ROBOT_LENGTH = 480;  // mm
        float ROBOT_SEMI_WIDTH = ROBOT_WIDTH / 2.f;     // mm
        float ROBOT_SEMI_LENGTH = ROBOT_LENGTH / 2.f;    // mm
        float TILE_SIZE = 100;   // mm
        float MIN_DISTANCE_TO_TARGET = ROBOT_WIDTH / 2.f; // mm
        std::string LIDAR_NAME_LOW = "bpearl";
        std::string LIDAR_NAME_HIGH = "helios";
        float MAX_LIDAR_LOW_RANGE = 10000;  // mm
        float MAX_LIDAR_HIGH_RANGE = 10000;  // mm
        float MAX_LIDAR_RANGE = 10000;  // mm used in the grid
        int LIDAR_LOW_DECIMATION_FACTOR = 2;
        int LIDAR_HIGH_DECIMATION_FACTOR = 1;
        QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
        float CARROT_DISTANCE = 400;   // mm
        float CARROT_ANGLE = M_PI_4 / 6.f;   // rad
        long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
        int PERIOD = 100;    // ms (10 Hz) for compute timer
        float MIN_ANGLE_TO_TARGET = 1.f;   // rad
        int MPC_HORIZON = 8;
        bool USE_MPC = true;
        unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
        int NUM_PATHS_TO_SEARCH = 3;
        float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm
        unsigned int SECS_TO_GET_IN = 1; // secs
        unsigned int SECS_TO_GET_OUT = 2; // sec//
        int max_distance = 2500; // mm

        // YOLO
        int STOP_SIGN = 11;
        int PERSON = 0;
        int BENCH = 13;
        int CHAIR = 56;

        // colors
        QColor TARGET_COLOR= {"orange"};
        QColor LIDAR_COLOR = {"LightBlue"};
        QColor PATH_COLOR = {"orange"};
        QColor SMOOTHED_PATH_COLOR = {"magenta"};
    };
    Params params;
    // draw
    void draw_line(const Line &line, QGraphicsScene *scene, QColor color="magenta");
    // Room detector
    rc::Room_Detector room_detector;

    // DoubleBuffer variable
    DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

    // Lidar
    void read_lidar();
    std::thread read_lidar_th;

    // Lines extractor
    Lines extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges);

    void process_room(rc::Room room);
    std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> calculate_rooms_correspondences_id(const std::vector<Eigen::Vector2d> &source_points_, const std::vector<Eigen::Vector2d> &target_points_, bool first_time = false);
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> calculate_rooms_correspondences(const std::vector<Eigen::Vector2d> &source_points_, const std::vector<Eigen::Vector2d> &target_points_);

    void check_room_orientation();
    bool is_on_a_wall(float x, float y, float width, float depth);
    static uint64_t get_actual_time();
    // fps
    FPSCounter fps;

    //Last valid corners
    std::vector<Eigen::Vector2d> last_valid_corners;
    std::vector<Eigen::Vector2d> corners_nominal_values;
};

#endif
