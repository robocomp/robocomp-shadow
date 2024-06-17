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
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/enumerate.hpp>
#include "door_detector.h"
#include "Hungarian.h"
#include "nodes.h"
#include "params.h"
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

	// DSR graph viewer
	std::unique_ptr<DSR::DSRViewer> graph_viewer;
	QHBoxLayout mainLayout;
	bool startup_check_flag;

    //Local widget
    CustomWidget *room_widget;

    struct Constants
    {
        std::string lidar_name = "helios";
        std::vector<std::pair<float, float>> ranges_list = {{1000, 2500}};
        bool DISPLAY = false;
    };
    Constants consts;

    rc::Params params;

    // Lidar
    void read_lidar();
    void draw_lidar(const RoboCompLidar3D::TData &data, QGraphicsScene *scene, QColor color="green");
    void draw_polygon(const QPolygonF &poly_in, const QPolygonF &poly_out,QGraphicsScene *scene, QColor color);
    void draw_door(const std::vector<DoorDetector::Door> &nominal_doors, const std::vector<DoorDetector::Door> &measured_doors, QColor nominal_color, QColor measured_color, QGraphicsScene *scene);

    // Thread for reading lidar
    std::thread read_lidar_th;

//    // DSR nodes
//    DSR::Node robot_node, room_node;

    // Hungarian algorithm
    HungarianAlgorithm HungAlgo;

    // DoubleBuffer variable
    DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

    //DOOR METHODS
    std::pair<std::vector<DoorDetector::Door>, std::vector<DoorDetector::Door>> get_measured_and_nominal_doors(DSR::Node room_node, DSR::Node robot_node);

    std::vector<DoorDetector::Door> get_doors(const RoboCompLidar3D::TData &ldata, QGraphicsScene *scene, DSR::Node robot_node, DSR::Node room_node);
    optional<tuple<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>> get_corners_and_wall_centers();
    Eigen::Vector2f project_point_on_polygon(const QPointF& p, const QPolygonF& polygon);
    QPointF project_point_on_line_segment(const QPointF& p, const QPointF& v, const QPointF& w);

    vector<pair<int, int>>
    door_matching(const vector<DoorDetector::Door> &measured_doors, const vector<DoorDetector::Door> &nominal_doors);
    std::vector<DoorDetector::Door> update_and_remove_doors(vector<std::pair<int, int>> matches, const vector<DoorDetector::Door> &measured_doors,
                                 const vector<DoorDetector::Door> &graph_doors, bool nominal, DSR::Node world_node);
    void update_door_in_graph(const DoorDetector::Door &door, std::string door_name, DSR::Node world_node);
    void set_doors_to_stabilize(std::vector<DoorDetector::Door> doors, DSR::Node room_node);

    DSR::Node insert_door_in_graph(DoorDetector::Door door, DSR::Node room, const std::string& door_name);

    void stabilize_door(DoorDetector::Door door, string door_name);
    vector<float> get_graph_odometry();
    std::chrono::time_point<std::chrono::system_clock> last_time;
    float odometry_time_factor = 1;
    void door_prefilter(vector<DoorDetector::Door> &detected_door);


    int g2o_nominal_door_id = 4;

    std::string build_g2o_graph(const std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> &measured_corner_data,
        const std::vector<Eigen::Matrix<float, 2, 1>> &nominal_corner_data,
        const std::vector<std::vector<float>> &odometry_data,
        const Eigen::Affine2d &robot_pose,
        const std::vector<Eigen::Vector2f> &measured_door_center,
        const Eigen::Vector2f &nominal_door_center);

    Eigen::Vector2f extract_g2o_data(string optimization);

    void draw_vector_of_doors(QGraphicsScene *scene, vector<QGraphicsItem *> &items, vector<DoorDetector::Door> doors,
                              QColor color) const;

    void
    draw_door_robot_frame(const vector<DoorDetector::Door> &nominal_doors,
                          const vector<DoorDetector::Door> &measured_doors,
                          QColor nominal_color, QColor measured_color, QGraphicsScene *scene);

    void clear_doors();

    void affordance();
    void affordance_thread(uint64_t aff_id);

    int actual_room_id = -1;

};

#endif
