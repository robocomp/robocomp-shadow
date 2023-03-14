/*
 *    Copyright (C) 2022 by YOUR NAME HERE
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
#include "/home/robocomp/robocomp/components/robocomp-shadow/etc/graph_names.h"
#include "/home/robocomp/robocomp/components/robocomp-shadow/etc/plan.h"
#include <cppitertools/zip.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/chunked.hpp>
#include <cppitertools/range.hpp>
#include <algorithm>
#include <QPointF>
#include <ranges>

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);
	void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);


public slots:
	void compute();
	int startup_check();
	void initialize(int period);
private:
	// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;
    std::shared_ptr<DSR::InnerEigenAPI> inner_eigen;
    std::unique_ptr<DSR::DSRViewer> graph_viewer;
    std::unique_ptr<DSR::RT_API> rt;
	//DSR params
	std::string agent_name;
	int agent_id;

	bool tree_view;
	bool graph_view;
	bool qscene_2d_view;
	bool osg_3d_view;
    //drawing
    DSR::QScene2dViewer* widget_2d;
	QHBoxLayout mainLayout;
	void modify_node_slot(std::uint64_t id, const std::string &type);
	void modify_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type);

	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag);
	void del_node_slot(std::uint64_t from){};     
	bool startup_check_flag;

	// Constants
	struct CONSTANTS
	{
		float robot_length = 700;
		float robot_width = 600;
		float robot_radius = robot_length / 2.0;
		float max_adv_speed = 400;
		float max_rot_speed = 2;
		float max_side_speed = 0;
		float max_lag = 100;  // ms
		float lateral_correction_gain = 1.2;
		float lateral_correction_for_side_velocity = 0;
		float rotation_gain = 0.9;
		float times_final_distance_to_target_before_zero_rotation = 3;
		float advance_gaussian_cut_x = 0.7;
		float advance_gaussian_cut_y = 0.3;
		float final_distance_to_target = 500; // mm
	};
	CONSTANTS consts;

    // PLan
    DoubleBuffer<Plan, Plan> plan_buffer;
    Plan current_plan;

    // Pose
    struct Pose2D  // pose X,Y + ang
    {
    public:
        void set_active(bool v) { active.store(v); };
        bool is_active() const { return active.load();};
        QGraphicsEllipseItem *draw = nullptr;
        void set_pos(const Eigen::Vector2f &p)
        {
            std::lock_guard<std::mutex> lg(mut);
            pos_ant = pos;
            pos = p;
        };
        void set_grid_pos(const Eigen::Vector2f &p)
        {
            std::lock_guard<std::mutex> lg(mut);
            grid_pos = p;
        };
        void set_angle(float a)
        {
            std::lock_guard<std::mutex> lg(mut);
            ang_ant = ang;
            ang = a;
        };
        Eigen::Vector2f get_pos() const
        {
            std::lock_guard<std::mutex> lg(mut);
            return pos;
        };
        Eigen::Vector2f get_grid_pos() const
        {
            std::lock_guard<std::mutex> lg(mut);
            return grid_pos;
        };
        Eigen::Vector2f get_last_pos() const
        {
            std::lock_guard<std::mutex> lg(mut);
            return pos_ant;
        };
        float get_ang() const
        {
            std::lock_guard<std::mutex> lg(mut);
            return ang;
        };
        Eigen::Vector3f to_eigen_3() const
        {
            std::lock_guard<std::mutex> lg(mut);
            return Eigen::Vector3f(pos.x() / 1000.f, pos.y() / 1000.f, 1.f);
        }
        QPointF to_qpoint() const
        {
            std::lock_guard<std::mutex> lg(mut);
            return QPointF(pos.x(), pos.y());
        };
    private:
        Eigen::Vector2f pos, pos_ant, grid_pos{0.f, 0.f};
        float ang, ang_ant = 0.f;
        std::atomic_bool active = ATOMIC_VAR_INIT(false);
        mutable std::mutex mut;
    };
    void set_robot_poses();
    void set_target_pose(DSR::Node node_path);
    Pose2D target;
    Pose2D robot;
    Pose2D robot_nose;


    //Laser
    Eigen::Vector2f get_predicted_force_vector(Eigen::Vector2f laser_point);
    Eigen::Vector2f compute_laser_forces(const RoboCompLaser::TLaserData &laser_data);
    std::tuple<RoboCompLaser::TLaserData, RoboCompLaser::TLaserData> read_laser();
    using LaserData = std::tuple<std::vector<float>, std::vector<float>>;  //<angles, dists>
    void change_force_parameters(string button_id);
    int force_distance = 700;
    float force_gain = 22;
    Eigen::Vector2f upper_left_robot{-consts.robot_width/2, consts.robot_length/2};
    Eigen::Vector2f upper_right_robot{consts.robot_width/2, consts.robot_length/2};
    Eigen::Vector2f bottom_left_robot{-consts.robot_width/2, -consts.robot_length/2};
    Eigen::Vector2f bottom_right_robot{consts.robot_width/2, -consts.robot_length/2};
    std::vector<std::tuple<Eigen::ParametrizedLine< float, 2 >, Eigen::Vector2f, Eigen::Vector2f>> robot_limits;
    Eigen::Vector2f get_intersection_point(Eigen::Vector2f laser_point);
//    std::vector<Eigen::ParametrizedLine< float, 2 >> robot_limit {Eigen::ParametrizedLine< float, 2 >::Through( *upper_left_robot, *upper_right_robot )};

    // Speed
    bool only_rotation = false;
    std::tuple<float, float, float> joystick_speeds = std::make_tuple(0.0, 0.0, 0.0);
    std::tuple<float, float, float> update();
    std::tuple<float, float, float> send_command_to_robot(const std::tuple<float, float, float> &speeds);
    std::tuple<float, float, float> combine_forces_and_speed(const std::tuple<float, float, float> &speeds, Eigen::Vector2f force_vector, float gain);
    // Path
    std::vector<Eigen::Vector2f> path;
    DoubleBuffer<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>> path_buffer;
    float dist_along_path(const std::vector<Eigen::Vector2f> &path);
    // Other
    void print_current_state(float adv, float side, float rot);
    float exponentialFunction(float value, float xValue, float yValue, float min);
    std::clock_t c_start;


    QGraphicsLineItem *target_draw_speed = nullptr;
    QGraphicsLineItem *target_draw_line = nullptr;
    QGraphicsLineItem *target_draw_comb = nullptr;
    QGraphicsLineItem *laser_draw_min = nullptr;

};

#endif
