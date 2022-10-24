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
#include <timer/timer.h>
#include "signalviewer.h"

#include "kalman.hpp"

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
        const float omni_camera_height = 580; //mm
        float robot_length = 500;
        float num_angular_bins = 360;
        float coppelia_depth_scaling_factor = 19.f;
        float dreamvu_depth_scaling_factor = 10.f;
        const float max_hor_angle_error = 0.6; // rads
    };
    Constants consts;
    float current_servo_angle = 0.f;

    // graphics
    AbstractGraphicViewer *viewer;
    QGraphicsPolygonItem *robot_polygon;
    QGraphicsRectItem *laser_in_robot_polygon;
    QRectF viewer_dimensions;
    void draw_floor_line(const vector<vector<Eigen::Vector2f>> &lines, int i=1);
    Eigen::Vector2f compute_repulsion_forces(vector<Eigen::Vector2f> &floor_line);
    void draw_forces(const Eigen::Vector2f &force, const Eigen::Vector2f &target, const Eigen::Vector2f &res);
    RoboCompGenericBase::TBaseState read_robot_state();
    void draw_3d_points(const RoboCompCameraRGBDSimple::TPoints &scan);
    void draw_lines_on_image(cv::Mat &rgb, const cv::Mat &depth_frame);
    cv::Mat read_depth_coppelia();
    RoboCompCameraRGBDSimple::TPoints read_points_dreamvu(cv::Mat omni_depth_frame);
    cv::Mat read_rgb(const std::string &camera_name);
    cv::Mat read_depth_dreamvu(cv::Mat omni_rgb_frame);
    bool IS_COPPELIA = false;
    void draw_humans(RoboCompYoloObjects::TObjects objects, const RoboCompYoloObjects::TBox &leader);
    void draw_legs(RoboCompLegDetector2DLidar::Legs legs);
    void eye_track(bool active_person, const RoboCompYoloObjects::TBox &person_box);
    RoboCompLegDetector2DLidar::Legs leg_detector(vector<Eigen::Vector2f> &lidar_line);
    RoboCompYoloObjects::TObjects yolo_detect_people(cv::Mat rgb, float threshold = 0.8);
    std::tuple<bool, RoboCompYoloObjects::TBox, Eigen::Vector2f> update_leader(RoboCompYoloObjects::TObjects &people); // removes leader from people
    float iou(const RoboCompYoloObjects::TBox &a, const RoboCompYoloObjects::TBox &b);
    void remove_leader_from_detected_legs(RoboCompLegDetector2DLidar::Legs &legs, const RoboCompYoloObjects::TBox &leader);
    void remove_lidar_points_from_leader(vector<Eigen::Vector2f> line, const RoboCompYoloObjects::TBox &leader);

    //Clock
    rc::Timer<> wtimer;

    // Kalman
    KalmanFilter kalman;
    int kalman_n; // Number of states
    int kalman_m; // Number of measurements
    void initialize_kalman(int period);
};



#endif
