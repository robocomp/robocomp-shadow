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
#include <innermodel/innermodel.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

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

    private:
        std::shared_ptr < InnerModel > innerModel;
        bool startup_check_flag;

        std::vector<std::vector<Eigen::Vector2f>> get_multi_level_3d_points(const cv::Mat &depth_frame);

    struct Constants
    {
        const float max_camera_depth_range = 5000;
        const float min_camera_depth_range = 300;
        const float omni_camera_height = 600; //mm
        float robot_length = 500;
        float num_angular_bins = 360;
        float scaling_factor = 19.f;
    };
    Constants consts;

    // graphics
    AbstractGraphicViewer *viewer;
    QGraphicsPolygonItem *robot_polygon;
    QGraphicsRectItem *laser_in_robot_polygon;
    QRectF viewer_dimensions;

    void draw_floor_line(const vector<vector<Eigen::Vector2f>> &lines);

    Eigen::Vector2f compute_repulsion_forces(vector<Eigen::Vector2f> &floor_line);

    void draw_forces(const Eigen::Vector2f &force);
};

#endif
