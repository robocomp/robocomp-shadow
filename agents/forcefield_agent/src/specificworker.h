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

        // Params
        rc::Params params;

        // Room detector
        rc::Room_Detector room_detector;

        // Lidar
        void read_lidar();
        std::thread read_lidar_th;
        void draw_lidar(const RoboCompLidar3D::TData &data, QGraphicsScene *scene, QColor color="green");
        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

        // Lines extractor
        Lines extract_2D_lines_from_lidar3D(const RoboCompLidar3D::TPoints &points, const std::vector<std::pair<float, float>> &ranges);

        void process_room(const rc::Room &room);
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
