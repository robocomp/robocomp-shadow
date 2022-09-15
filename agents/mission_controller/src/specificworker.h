/*
 *    Copyright (C) 2021 by YOUR NAME HERE
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
#include <doublebuffer/DoubleBuffer.h>
#include <custom_widget.h>
#include  "/home/robocomp/robocomp/components/robocomp-shadow/etc/graph_names.h"
#include <opencv2/opencv.hpp>
#include "/home/robocomp/robocomp/components/robocomp-shadow/etc/plan.h"
#include <QListWidget>
#include <QSpinBox>
#include "ui_mission_pointUI.h"
#include "ui_mission_pathfollowUI.h"
#include "/home/robocomp/robocomp/classes/abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <Eigen/Geometry>



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
    void slot_start_mission();
    void slot_stop_mission();
    void slot_cancel_mission();
    void slot_change_mission_selector(int);
    void trace_button_slot(bool);

private:
	// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;
    std::shared_ptr<DSR::CameraAPI> cam_api;
    std::shared_ptr<DSR::InnerEigenAPI> inner_eigen;
    std::shared_ptr<DSR::RT_API> rt_api;

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
	void add_or_assign_node_slot(std::uint64_t id, const std::string &type);
    void modify_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
	void add_or_assign_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type);
    void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag);
	void del_node_slot(std::uint64_t from);
	bool startup_check_flag;
	Eigen::IOFormat OctaveFormat, CommaInitFmt;

    // local widgets
    DSR::QScene2dViewer* widget_2d;
    Custom_widget custom_widget;
    Ui_Goto_UI point_dialog;
    Ui_PathFollow_UI pathfollow_dialog;

    // Laser
    using LaserData = std::tuple<std::vector<float>, std::vector<float>>;  //<angles, dists>
    DoubleBuffer<LaserData, std::tuple<std::vector<float>, std::vector<float>, QPolygonF, std::vector<QPointF>>> laser_buffer;

    // Robot and shape
    QPolygonF robot_polygon;
    void send_command_to_robot(const std::tuple<float, float, float> &speeds);   //adv, rot, side

    // Camera
    DoubleBuffer<std::vector<std::uint8_t>, std::vector<std::uint8_t>> virtual_camera_buffer;
    void read_camera();

    // Missions
    DoubleBuffer<Plan, Plan> plan_buffer;
    Plan temporary_plan;
    Plan current_plan;
    void insert_intention_node(const Plan &plan);
    void create_goto_mission();
    void create_follow_people_mission(uint64_t person_id = 0);
    void create_recognize_people_mission(uint64_t person_id);
    void create_talking_people_mission(uint64_t person_id);
    void create_searching_person_mission();
    void create_bouncer_mission();
    void create_path_mission();
    AbstractGraphicViewer *pathfollow_draw_widget;

    // People
    void people_checker();

    // ID
    uint64_t interacting_person_id;
    uint64_t followed_person_id;
    uint64_t talkative_person_id;

    //Path
    std::vector<Eigen::Vector2f> path;  // check if can be made local
    QPointF last_point;
    Mat::Vector3d last_person_pos;
    std::vector<QGraphicsLineItem *> lines;
    DoubleBuffer<std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>> path_buffer;
    void draw_path(std::vector<Eigen::Vector2f> &path, QGraphicsScene* viewer_2d, bool remove = false);
    void follow_path_copy_path_to_graph(const std::vector<float> &x_values, const std::vector<float> &y_values);

    uint64_t from_variant_to_uint64(QVariant value);

    uint64_t node_string2id(Plan currentPlan);
};

#endif
