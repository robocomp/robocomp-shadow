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
#include <timer/timer.h>
//#include "sm_search_and_approach.h"
#include <random>
#include <qcustomplot/qcustomplot.h>
#include "door_detector.h"
#include "room_detector.h"
#include "room.h"
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
        int startup_check();
        void initialize(int period);

    private:
        struct Constants
        {

        };
        Constants consts;
        bool startup_check_flag;

        // world representation
        AbstractGraphicViewer *viewer;

        // draw
        void draw_lidar(const RoboCompLidar3D::TData &data);

        // joy
        void set_target_force(const Eigen::Vector3f &vec);
        Eigen::Vector3f target_coordinates{0.f, 0.f, 0.f};  //third component for pure  rotations


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

        // DoubleBuffer variable
        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;

        // Lidar
        void read_lidar();
        std::thread read_lidar_th;

};

#endif
