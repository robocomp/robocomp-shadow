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
#define QT6
#include <genericworker.h>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <timer/timer.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dynamic_window.h>
#include <timer/timer.h>
#include "robot.h"
#include "camera.h"
#include "sm_search_and_approach.h"
#include <random>
#include <qcustomplot/qcustomplot.h>
#include "door_detector.h"
#include "room_detector.h"
#include "room.h"
#include "preobject.h"
#include <doublebuffer/DoubleBuffer.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    using v2f = Eigen::Vector2f;
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

        void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);

    public slots:
        void compute();
        void compute_follow_human();
        void compute_explore_rooms();
        int startup_check();
        void initialize(int period);

    private:
        rc::Robot robot;
        rc::Camera top_camera;

        struct Constants
        {
            const float max_distance_for_repulsion = 1000; // mm. Distance beyond which repulsion vanishes. It follows an inverse law with current robot speed
            const float speed_for_max_repulsion = 1000;
            //float dynamic_threshold = max_distance_for_repulsion/(speed_for_max_repulsion*speed_for_max_repulsion);
            double xset_gaussian = 0.4;             // gaussian break x set value
            double yset_gaussian = 0.3;             // gaussian break y set value
        };
        Constants consts;
        float current_servo_angle = 0.f;
        bool startup_check_flag;

        // world representation
        AbstractGraphicViewer *viewer;

        void move_robot(Eigen::Vector2f force);
        RoboCompYoloObjects::TObjects yolo_detect_objects(cv::Mat rgb);

        // draw
        void draw_floor_line(const vector<vector<Eigen::Vector2f>> &lines, std::initializer_list<int> list);
        void draw_objects_on_2dview(const std::vector<rc::PreObject> &objects, const rc::PreObject &selected);
        void draw_lidar(const RoboCompLidar3D::TData &data);

        // objects
        RoboCompYoloObjects::TObjectNames yolo_object_names;
        Eigen::MatrixX3f COLORS;

        // joy
        void set_target_force(const Eigen::Vector3f &vec);
        Eigen::Vector3f target_coordinates{0.f, 0.f, 0.f};  //third component for pure  rotations

        // state machine
        SM_search_and_approach sm_search_and_approach;

        float iou(const RoboCompYoloObjects::TBox &a, const RoboCompYoloObjects::TBox &b);
        float closest_distance_ahead(const vector<Eigen::Vector2f> &line);
        float gaussian(float x);

        // bumper
        std::map<float, float> bumper_points;

        // Clock
        rc::Timer<> clock;

        // Second compute timer
        QTimer timer2;

        //distance_lines
        std::vector<Eigen::Vector2f> current_line;

        // QCustomPlot
        QCustomPlot custom_plot;
        QCPGraph *side_acc, *adv_acc, *track_err;
        void draw_timeseries(float side, float adv, float track);

        // Door detector
        DoorDetector door_detector;

        // Room detector
        rc::Room_Detector room_detector;

        // preobjects
        std::vector<rc::PreObject> preobjects;

        // DoubleBuffer variable
        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

        // Lidar
        void read_lidar();
        std::thread read_lidar_th;

};

#endif
