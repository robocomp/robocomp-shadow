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
#include <doublebuffer/DoubleBuffer.h>
#include "door_detector.h"
#include "Hungarian.h"
#include <cppitertools/range.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/enumerate.hpp>
#include <fps/fps.h>

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    using Line = std::vector<Eigen::Vector2f>;
    using Lines = std::vector<Line>;

	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);

    struct Constants
    {
        std::string lidar_name = "helios";
        std::vector<std::pair<float, float>> ranges_list = {{1000, 2500}};
        bool DISPLAY = false;
    };
    Constants consts;


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

    //local widget
    DSR::QScene2dViewer* widget_2d;

    // Door detector
    DoorDetector door_detector;

    //Robot const
    float max_robot_advance_speed = 800.0;
    float max_robot_side_speed = 200;

    // DoubleBuffer variable
    DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

    // Lidar
    void read_lidar();
    void draw_lidar(const RoboCompLidar3D::TData &data, QGraphicsScene *scene, QColor color="green");
    void draw_polygon(const QPolygonF &poly_in, const QPolygonF &poly_out,QGraphicsScene *scene, QColor color);


    std::thread read_lidar_th;

    // Lines extractor
    Lines extract_lines(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges);
    void insert_measured_door_into_graph(const DoorDetector::Door &door, int wall_id);
    DSR::Node insert_nominal_door_into_graph(const DoorDetector::Door &door, int wall_id);
	// DSR graph viewer
	std::unique_ptr<DSR::DSRViewer> graph_viewer;
	QHBoxLayout mainLayout;
	bool startup_check_flag;

    // fps
    FPSCounter fps;

    void
    draw_door(const vector<std::tuple<int, Eigen::Vector2f, Eigen::Vector2f>> doors, QGraphicsScene *scene, QColor color);

    Eigen::Vector2f projectPointOnPolygon(const QPointF &p, const QPolygonF &polygon);

    QPointF projectPointOnLineSegment(const QPointF &p, const QPointF &v, const QPointF &w);

    HungarianAlgorithm HungAlgo;

    std::optional<std::tuple<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>>
    get_corners_and_wall_centers();

    vector<tuple<int, Eigen::Vector2f, Eigen::Vector2f>>
    get_doors(const RoboCompLidar3D::TData &ldata, const vector<Eigen::Vector2f> &corners,
              const vector<Eigen::Vector2f> &wall_centers, QGraphicsScene *scene);
    // Door to stabilize variable initialized as nullptr
    std::optional<DoorDetector::Door> door_to_stabilize_ = std::nullopt;
    float distance_to_target = 150;
    void set_robot_speeds(float adv, float side, float rot);

    vector<float> calculate_speed(const Eigen::Matrix<double, 3, 1> &target);

    bool movement_completed(const Eigen::Vector3d &target, float distance_to_target);

    std::chrono::time_point<std::chrono::system_clock> last_time;
    vector<float> get_graph_odometry();

    float door_center_matching_threshold = 1500;
    float odometry_time_factor = 1;

    Eigen::Vector3d door_stabilization_target = {0, 0, 0};

    void generate_target_edge(DSR::Node node);

    string build_g2o_graph(const vector<std::vector<Eigen::Matrix<float, 2, 1>>> &measured_corner_data,
                           const vector<Eigen::Matrix<float, 2, 1>> &nominal_corner_data,
                           const vector<std::vector<float>> &odometry_data,
                           const Eigen::Affine2d &robot_pose,
                           const vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> &measured_door_vertices,
                           const pair<Eigen::Vector2f, Eigen::Vector2f> &nominal_door_vertices);
    std::vector<Eigen::Vector2f> extract_g2o_data(string optimization);
    void draw_graph_doors(const std::vector<std::tuple<int, Eigen::Vector2f, Eigen::Vector2f>> doors, QGraphicsScene *scene, QColor color);
    vector<Eigen::Vector2f> get_nominal_door_from_dsr(DSR::Node &robot_node, vector<DSR::Node> &door_nodes);

    void asociate_and_update_doors(const vector<tuple<int, Eigen::Vector2f, Eigen::Vector2f>> &doors,
                                   const vector<DSR::Node> &door_nodes, const vector<Eigen::Vector2f> &door_poses);
    void generate_edge_goto_door(DSR::Node &robot_node, DSR::Node &door_node);
    void update_door_in_graph(const Eigen::Vector2f &pose, string door_name);

};

#endif
