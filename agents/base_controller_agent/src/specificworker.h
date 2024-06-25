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

#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include "GridPlanner.h"
#include <doublebuffer/DoubleBuffer.h>
#include <fps/fps.h>
#include <timer/timer.h>
#include <Eigen/Eigen>
#include <custom_widget.h>
#include <ui_localUI.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

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
        DSR::QScene2dViewer* widget_2d;

        //Local widget
        CustomWidget *room_widget;

        struct Params
        {
            std::string robot_name = "Shadow";
            float ROBOT_WIDTH = 250;  // mm
            float ROBOT_LENGTH = 480;  // mm
            float ROBOT_SEMI_WIDTH = ROBOT_WIDTH / 2.f;     // mm
            float ROBOT_SEMI_LENGTH = ROBOT_LENGTH / 2.f;    // mm
            float MAX_ADVANCE_VELOCITY = 500;  // mm/s
            float MAX_SIDE_VELOCITY = 150;  // mm/s
            float MAX_ROTATION_VELOCITY = 0.57;  // rad/s
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
            //RoboCompGridder::TDimensions gdim;
            float CARROT_DISTANCE = 400;   // mm
            float CARROT_ANGLE = M_PI_4 / 6.f;   // rad
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            int PERIOD = 100;    // ms (20 Hz) for compute timer
            float MIN_ANGLE_TO_TARGET = 1.f;   // rad
            int MPC_HORIZON = 8;
            bool USE_MPC = true;
            unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
            int NUM_PATHS_TO_SEARCH = 3;
            float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm
            bool DISPLAY = true;
            size_t MAX_PATH_STEPS = 1000;
            float TRACKING_DISTANCE_TO_TARGET = 1000; // mm

            // colors
            QColor TARGET_COLOR= {"orange"};
            QColor LIDAR_COLOR = {"green"};
            QColor PATH_COLOR = {"orange"};
            QColor SMOOTHED_PATH_COLOR = {"magenta"};
        };
        Params params;

        // FPS
        FPSCounter fps;
        int hz = 0;

        // Timer
        rc::Timer<> clock;

        // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        //  draw
        void draw_lidar_in_robot_frame(const std::vector<Eigen::Vector3f> &data, QGraphicsScene *scene, QColor color="green", int step=1);

        // RT APi
        std::unique_ptr<DSR::RT_API> rt_api;
        std::unique_ptr<DSR::InnerEigenAPI> inner_api;

        // path
        std::list<Eigen::Vector2f> current_path;  // list to remove elements from the front

        std::optional<DSR::Edge> there_is_intention_edge_marked_as_active();
        bool target_node_is_measurement(const DSR::Edge &edge);
        optional<Eigen::Vector3d> get_translation_vector_from_target_node(const DSR::Edge &edge);
        bool robot_at_target(const Eigen::Vector3d &matrix, const DSR::Edge &edge);
        void stop_robot();
        bool line_of_sight(const Eigen::Vector3d &matrix, const std::vector<Eigen::Vector3f> &ldata, QGraphicsScene *pScene);

        RoboCompGridPlanner::Points compute_line_of_sight_target(const Eigen::Vector2d &target);
        void draw_vector_to_target(const Eigen::Vector3d &matrix, QGraphicsScene *pScene);
        std::tuple<float, float, float> compute_line_of_sight_target_velocities(const Eigen::Vector3d &matrix);
        void move_robot(float adv, float side, float rot);

    void set_intention_edge_state(DSR::Edge &edge, const std::string &string);

    void draw_robot_in_room(QGraphicsScene *pScene, const vector<Eigen::Vector3f> &lidar_data);
};

#endif
