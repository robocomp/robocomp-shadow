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

#include "../../etc/graph_names.h"
#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <doublebuffer/DoubleBuffer.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fps/fps.h>

class SpecificWorker : public GenericWorker
{
    using MyClock = std::chrono::system_clock;
    using mSec = std::chrono::duration<double, std::milli>;

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
    void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
    void add_or_assign_node_slot(std::uint64_t, const std::string &type);
    void modify_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
    void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
    void del_node_slot(std::uint64_t from){};

private:
    bool startup_check_flag;

	/// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;
    std::shared_ptr<DSR::InnerEigenAPI> inner_eigen;
    std::unique_ptr<DSR::RT_API> rt;
    std::unique_ptr<DSR::AgentInfoAPI> agent_info_api;

	///DSR params
	std::string agent_name;
	int agent_id;
    std::string dsr_input_file;
	bool tree_view;
	bool graph_view;
	bool qscene_2d_view;
	bool osg_3d_view;

	/// DSR graph viewer
	std::unique_ptr<DSR::DSRViewer> graph_viewer;
	QHBoxLayout mainLayout;

    float av_anterior = 9999;
    float rot_anterior = 9999;
    float servo_speed_anterior = 0;
    float servo_pos_anterior = 0;

	void update_robot_localization();
    void read_battery();
    //void read_RSSI();
    void update_cameras();
    cv::Mat compute_camera_simple_frame();
    void update_camera_simple(std::string camera, const cv::Mat &virtual_frame);
    void update_servo_position();

    // aux
    bool are_different(const vector<float> &a, const vector<float> &b, const vector<float> &epsilon);
    Eigen::Vector2f from_world_to_robot(const Eigen::Vector2f &p,
                                        const RoboCompFullPoseEstimation::FullPoseEuler &r_state);

    // Robot
    Eigen::Matrix3f convertion_matrix;
    Eigen::Matrix3f get_new_grid_matrix(Eigen::Vector3d robot_position, Eigen::Vector3d robot_rotation);

    // FPS
    FPSCounter fps;


};

#endif
