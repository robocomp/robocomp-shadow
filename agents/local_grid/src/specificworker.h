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
#include "/home/robocomp/robocomp/classes/local_grid/local_grid.h"
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <custom_widget.h>
#include <timer/timer.h>
#include <random/random.hpp>


#include "GRANSAC.hpp"
#include "LineModel.hpp"

using Random = effolkronium::random_static;
using Point3f = std::tuple<float, float, float>;

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
        void modify_edge_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
        void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
        void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
        void del_node_slot(std::uint64_t from){};

private:
        bool startup_check_flag;
        struct Constants
        {
            float tile_size = 100;
            const float max_laser_range = 4000;
            const float max_camera_depth_range = 5000;
            const float min_camera_depth_range = 300;
            const float omni_camera_height_meters = 0.4; //mm
            float robot_length = 500;
            float num_angular_bins = 360;
            float scaling_factor = 19.f;
        };
        Constants consts;

        // DSR graph
        std::shared_ptr<DSR::DSRGraph> G;
        std::shared_ptr<DSR::InnerEigenAPI> inner_eigen;
        std::unique_ptr<DSR::CameraAPI> cam_omni_api, cam_head_api;
        std::unique_ptr<DSR::AgentInfoAPI> agent_info_api;

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
        DSR::QScene2dViewer* widget_2d;

        //local widget
        Custom_widget custom_widget;
        QWidget grid_widget;
        AbstractGraphicViewer *grid_viewer;

        // Array of sets for representing sectors
        cv::Mat read_depth_omni();
        std::vector<Point3f> get_omni_3d_points(const cv::Mat &depth_frame, const cv::Mat &rgb_frame);
        struct compare
        { bool operator()(const std::tuple<Eigen::Vector3f, std::tuple<float, float, float>> &a, const std::tuple<Eigen::Vector3f, std::tuple<float, float, float>> &b) const
            { return std::get<Eigen::Vector3f>(a).norm() < std::get<Eigen::Vector3f>(b).norm(); }
        };
        using SetsType = std::vector<std::set<std::tuple<Eigen::Vector3f, std::tuple<float, float, float>>, compare>>;
        SetsType group_by_angular_sectors(const std::vector<Point3f> &points, bool draw=false);
        vector<Eigen::Vector2f> compute_floor_line(const SetsType &sets, bool draw=false);

        // grid
        //std::shared_ptr<std::vector<std::tuple<float, float, float>>> points, colors;
        Local_Grid local_grid;

        // YOLO objects
        RoboCompYoloObjects::TObjects get_yolo_objects();
        cv::Mat draw_yolo_objects(const RoboCompYoloObjects::TObjects &objects, cv::Mat img);
        RoboCompYoloObjects::TObjectNames yolo_object_names;
        RoboCompYoloObjects::TJointData yolo_joint_data;
        std::vector<int> excluded_yolo_types;

        // FPS
        FPSCounter fps;
        rc::Timer<> stimer;

        // dRAW
        void draw_on_2D_tab(const std::vector<Eigen::Vector2f> &points, QString color="green", bool clean = true);
        void draw_on_2D_tab(const RoboCompYoloObjects::TObjects &objects);
        void draw_on_2D_tab(const std::vector<cv::Vec3d> &lines);
        std::map<int, QPixmap> object_pixmaps;

        GRANSAC::RANSAC<Line2DModel, 2> Estimator;


};

#endif
