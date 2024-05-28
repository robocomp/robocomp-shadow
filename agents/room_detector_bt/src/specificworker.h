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

#include "params.h"
#include <random>
#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <doublebuffer/DoubleBuffer.h>
#include "../../../agents/forcefield_agent/src/door_detector.h"
#include "room_detector.h"
#include "room.h"
#include "Hungarian.h"
#include <fps/fps.h>
#include <icp.h>
#include <timer/timer.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <timer/timer.h>
#include "behaviortree_cpp/behavior_tree.h"
#include <behaviortree_cpp/blackboard.h>
#include <behaviortree_cpp/bt_factory.h>

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

    public:
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

        // Params
        rc::Params params;

        // Room detector
        rc::Room_Detector room_detector;

        //Robot const
        float max_robot_advance_speed = 800.0;
        float max_robot_side_speed = 200;


        // Lidar
        void read_lidar();
        std::thread read_lidar_th;
        void draw_lidar(const RoboCompLidar3D::TData &data, QGraphicsScene *scene, QColor color="green");
        void draw_transformed_corners(QGraphicsScene *scene, QColor color="red");

        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

        void generate_target_edge(DSR::Node node);

        // Lines extractor
        Lines extract_2D_lines_from_lidar3D(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges);

        void update_room_data(const rc::Room_Detector::Corners &corners, QGraphicsScene *scene);
        std::vector<std::tuple<int, Eigen::Vector2d, Eigen::Vector2d, bool>> calculate_rooms_correspondences_id(const std::vector<Eigen::Vector2d> &source_points_, std::vector<Eigen::Vector2d> &target_points_, bool first_time = false);
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> calculate_rooms_correspondences(const std::vector<Eigen::Vector2d> &source_points_, const std::vector<Eigen::Vector2d> &target_points_);

        void create_wall(int id, const std::vector<float> &p, float angle, DSR::Node parent_node, bool nominal=true);
        void create_corner(int id, const std::vector<float> &p, DSR::Node parent_node, bool nominal=true);

        void check_room_orientation();
        bool is_on_a_wall(float x, float y, float width, float depth);
        static uint64_t get_actual_time();
        // fps
        FPSCounter fps;

        //Last valid corners
        std::vector<Eigen::Vector2d> last_valid_corners;
        std::vector<Eigen::Vector2d> corners_nominal_values;

        // Room validation
        std::string g2o_graph_data;
        float odometry_time_factor = 1;
        std::pair<Eigen::Affine2d, std::vector<Eigen::Vector2d>> get_robot_initial_pose(Eigen::Vector2f &first_room_center, std::vector<Eigen::Matrix<float, 2, 1>> first_corners, int width, int depth);
        std::vector<Eigen::Vector2d> aux_corners;

        std::vector<Eigen::Vector2d> last_corner_values{4};
        std::vector<float> last_robot_pose{0.f, 0.f, 0.f};
        std::chrono::time_point<std::chrono::system_clock> last_time;
        bool movement_completed(const Eigen::Vector2f &room_center, float distance_to_target);

        //MISC
        void set_robot_speeds(float adv, float side, float rot);
        std::vector<float> calculate_speed(const Eigen::Matrix<float, 2, 1> &target);
        float distance_to_target = 50;
        std::vector<float> get_graph_odometry();
        double corner_matching_threshold = 1000;
        bool corner_matching_threshold_setted = false;
        std::vector<Eigen::Vector2d> last_corners;
        std::chrono::time_point<std::chrono::system_clock> starting_time;
        HungarianAlgorithm HungAlgo;

        std::tuple<std::vector<Eigen::Vector2d>, Eigen::Vector3d> extract_g2o_data(string optimization);

        void insert_room_into_graph(tuple<std::vector<Eigen::Vector2d>, Eigen::Vector3d> optimized_room_data, const rc::Room &current_room);
        std::vector<Eigen::Vector2d> get_transformed_corners(QGraphicsScene *scene);

        string build_g2o_graph( const vector<std::vector<Eigen::Matrix<float, 2, 1>>> &corner_data,
                               const vector<std::vector<float>> &odometry_data, const Eigen::Affine2d robot_pose,
                               const vector<Eigen::Vector2d> nominal_corners, const std::vector<Eigen::Vector2f> &room_sizes, std::vector<int> room_size);

        void test();
        void room_stabilitation();
        int distance_to_center();
        void create_room();
        void check_corner_matching();
        void update_room();

    //BehaviorTrees
    BT::BehaviorTreeFactory factory;
    BT::Tree tree;
    std::thread BT_th;

    struct Data{
        double corner_matching_threshold = 1000;
        bool corner_matching_threshold_setted = false;
        Eigen::Matrix<float,2,1> room_center = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        std::vector<std::vector<Eigen::Matrix<float, 2, 1>>> corner_data;
        std::vector<Eigen::Vector2f> room_centers;
        std::vector<Eigen::Vector2f> room_sizes;
        std::map<std::vector<int>, int> room_size_histogram;
        std::vector<std::vector<float>> odometry_data;
    };
    Data BTdata;

    void BTFunction();

};

#include "nodes.h"

#endif
